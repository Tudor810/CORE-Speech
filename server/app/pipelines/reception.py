"""Pipeline I: frames -> MediaPipe -> LSTM -> local LLM -> OpenVoice -> WS audio.

Designed as a single async orchestrator that owns:
  - a frame consumer task that produces landmark vectors
  - a "phrase" buffer that collects gesture tokens until a quiet period
  - a synthesis task that streams tokens to the LLM and bytes to the WS

Frames come from a pluggable async iterator: by default that's the server's
shared `CameraStream`, but the browser-facing endpoint passes a queue-backed
iterator fed by JPEG frames decoded from the WebSocket.

The whole thing runs on the event loop; CPU-heavy steps (MediaPipe, LSTM)
already offload to threads inside their respective modules.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

import numpy as np

from app.core.logging import get_logger
from app.core.state import AppState
from app.llm.local_client import translate_signs
from app.ml.classifier import GestureClassifier
from app.tts.openvoice_client import OpenVoiceTTS
from app.vision.capture import CameraStream
from app.vision.landmarks import HandLandmarker

log = get_logger(__name__)

# How long without a new gesture before we consider the phrase "complete".
_PHRASE_QUIET_SECONDS = 1.0
# Cap phrase length to avoid runaway buffering.
_MAX_PHRASE_TOKENS = 12
# Label name that means "the user is not signing"; matches the train-time
# idle class produced by ``extract_dataset.py --label idle --mode windows``.
_IDLE_LABEL = "idle"
# A label has to be the dominant prediction for at least this long before we
# commit it to the phrase. Mirrors live_predict.py's ``--stable-time``.
_STABLE_COMMIT_SECONDS = 0.25
# After committing a sign we suppress new commits for a moment so a single
# slow-changing pose doesn't get re-emitted as soon as ``is_new`` flips
# (e.g. brief idle then back to the same sign). Mirrors ``--cooldown``.
_COMMIT_COOLDOWN_SECONDS = 0.6


class ReceptionPipeline:
    def __init__(
        self,
        device_id: str,
        app_state: AppState,
        landmarker: HandLandmarker,
        classifier: GestureClassifier,
        tts: OpenVoiceTTS,
        send_bytes: Callable[[bytes], Awaitable[None]],
        send_json: Callable[[dict[str, Any]], Awaitable[None]],
        *,
        camera: Optional[CameraStream] = None,
        frames: Optional[AsyncIterator[np.ndarray]] = None,
        emit_landmarks: bool = False,
    ) -> None:
        if camera is None and frames is None:
            raise ValueError("ReceptionPipeline requires either `camera` or `frames`")
        self._device_id = device_id
        self._state = app_state
        self._camera = camera
        self._frames = frames
        self._landmarker = landmarker
        self._classifier = classifier
        self._tts = tts
        self._send_bytes = send_bytes
        self._send_json = send_json
        self._emit_landmarks = emit_landmarks

        self._phrase: list[str] = []
        self._last_token_at: float = 0.0
        self._lock = asyncio.Lock()
        self._stopped = False

        # Stable-commit / cooldown state (sliding LSTM debouncer).
        self._candidate_label: str | None = None
        self._candidate_first_seen: float = 0.0
        self._candidate_best_conf: float = 0.0
        self._candidate_committed: bool = False
        self._cooldown_until: float = 0.0

    async def run(self) -> None:
        await self._send_json({"type": "status", "msg": "ready"})
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._consume_frames(), name="reception-cam")
                tg.create_task(self._phrase_watcher(), name="reception-watch")
        except* Exception as eg:  # noqa: BLE001
            for exc in eg.exceptions:
                log.exception("Reception pipeline error: %s", exc)

    async def stop(self) -> None:
        self._stopped = True

    async def _frame_iterator(self) -> AsyncIterator[np.ndarray]:
        if self._frames is not None:
            async for frame in self._frames:
                yield frame
            return
        assert self._camera is not None  # validated in __init__
        async for frame in self._camera.subscribe():
            yield frame

    async def _consume_frames(self) -> None:
        async for frame in self._frame_iterator():
            if self._stopped:
                break
            try:
                hand = await self._landmarker.process(frame)
            except Exception as exc:  # noqa: BLE001
                # A single bad frame must not tear down the whole pipeline.
                log.warning("Landmark detection failed for one frame: %s", exc)
                continue
            if self._emit_landmarks:
                # Forward landmark coords so the browser can overlay the
                # skeleton on top of the live video.
                await self._send_json(
                    {
                        "type": "landmarks",
                        "frame": hand.to_json(),
                        "ts": time.time(),
                    }
                )
            if not hand.has_hand:
                # Reset the in-progress candidate so the next sign starts
                # cleanly when the hands come back.
                self._candidate_label = None
                self._candidate_committed = False
                continue

            self._classifier.push(hand.to_vector())
            pred = await self._classifier.predict()
            if not pred:
                continue

            now = time.monotonic()

            # Drop the dedicated "the user is not signing" class so it never
            # leaks into the phrase buffer.
            if pred.label == _IDLE_LABEL:
                self._candidate_label = None
                self._candidate_committed = False
                continue

            if now < self._cooldown_until:
                continue

            # Time-stable commit (mirror of live_predict's --stable-time):
            # require the same label to dominate for _STABLE_COMMIT_SECONDS
            # before we treat it as a real sign.
            if self._candidate_label != pred.label:
                self._candidate_label = pred.label
                self._candidate_first_seen = now
                self._candidate_best_conf = pred.confidence
                self._candidate_committed = False
                continue

            if pred.confidence > self._candidate_best_conf:
                self._candidate_best_conf = pred.confidence

            if self._candidate_committed:
                continue
            if (now - self._candidate_first_seen) < _STABLE_COMMIT_SECONDS:
                continue

            label = self._candidate_label
            confidence = self._candidate_best_conf
            self._candidate_committed = True
            self._cooldown_until = now + _COMMIT_COOLDOWN_SECONDS

            async with self._lock:
                # Skip back-to-back duplicates of the exact same token.
                if not self._phrase or self._phrase[-1] != label:
                    self._phrase.append(label)
                self._last_token_at = now
            await self._send_json(
                {
                    "type": "gesture",
                    "label": label,
                    "confidence": confidence,
                }
            )
            if len(self._phrase) >= _MAX_PHRASE_TOKENS:
                await self._flush_phrase()

    async def _phrase_watcher(self) -> None:
        while not self._stopped:
            await asyncio.sleep(0.2)
            async with self._lock:
                if not self._phrase:
                    continue
                idle = time.monotonic() - self._last_token_at
            if idle >= _PHRASE_QUIET_SECONDS:
                await self._flush_phrase()

    async def _flush_phrase(self) -> None:
        async with self._lock:
            if not self._phrase:
                return
            tokens = self._phrase[:]
            self._phrase.clear()

        log.info("[%s] phrase -> %s", self._device_id, tokens)
        await self._send_json({"type": "phrase", "tokens": tokens})

        emotion = self._state.data.emotion_preset
        intensity = self._state.data.emotion_intensity
        voice_id = self._state.data.active_voice_id
        speaker_key = self._state.data.tts_speaker_key

        # Tee the LLM stream so we can both send the text to the ES (for
        # captioning) and feed it into the local TTS.
        text_q_for_tts: asyncio.Queue[str | None] = asyncio.Queue()
        text_q_for_view: asyncio.Queue[str | None] = asyncio.Queue()

        async def llm_producer() -> None:
            try:
                async for piece in translate_signs(tokens, emotion, intensity):
                    await text_q_for_tts.put(piece)
                    await text_q_for_view.put(piece)
            finally:
                await text_q_for_tts.put(None)
                await text_q_for_view.put(None)

        async def queue_to_iter(q: asyncio.Queue):
            while True:
                item = await q.get()
                if item is None:
                    break
                yield item

        async def caption_forwarder() -> None:
            buf = ""
            async for piece in queue_to_iter(text_q_for_view):
                buf += piece
            if buf:
                await self._send_json({"type": "translation", "text": buf})

        async def tts_forwarder() -> None:
            chunk_count = 0
            async for audio in self._tts.synthesize_stream(
                queue_to_iter(text_q_for_tts),
                voice_id=voice_id,
                speaker_key=speaker_key,
                emotion=emotion,
                intensity=intensity,
            ):
                chunk_count += 1
                await self._send_bytes(audio)
            await self._send_json({"type": "audio_end", "chunks": chunk_count})

        with contextlib.suppress(Exception):
            await asyncio.gather(llm_producer(), caption_forwarder(), tts_forwarder())
