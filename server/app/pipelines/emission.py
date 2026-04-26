"""Pipeline II: ES mic stream -> Vosk STT -> text back to ES.

We accumulate raw PCM bytes into a rolling buffer and run Vosk whenever the
buffer reaches `stt_chunk_seconds * stt_sample_rate * 2` bytes (16-bit samples).
A worker task consumes a `asyncio.Queue` of audio buffers so the WS receive
loop never blocks on inference.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Awaitable, Callable

from app.config import settings
from app.core.logging import get_logger
from app.stt.vosk_client import VoskClient

log = get_logger(__name__)


class EmissionPipeline:
    def __init__(
        self,
        device_id: str,
        stt: VoskClient,
        send_json: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        self._device_id = device_id
        self._stt = stt
        self._send_json = send_json

        # Bytes per STT window (16-bit mono).
        self._target_bytes = int(
            settings.stt_sample_rate * settings.stt_chunk_seconds * 2
        )
        self._buffer = bytearray()
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=8)
        self._stopped = False
        self._worker: asyncio.Task | None = None

    async def run(self) -> None:
        self._worker = asyncio.create_task(
            self._consume(), name=f"emission-worker-{self._device_id}"
        )
        await self._send_json({"type": "status", "msg": "ready"})
        await self._worker

    async def stop(self) -> None:
        self._stopped = True
        if self._worker:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker

    async def feed_audio(self, pcm: bytes) -> None:
        if self._stopped or not pcm:
            return
        self._buffer.extend(pcm)
        while len(self._buffer) >= self._target_bytes:
            chunk = bytes(self._buffer[: self._target_bytes])
            del self._buffer[: self._target_bytes]
            if self._queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    self._queue.get_nowait()
            await self._queue.put(chunk)

    async def flush(self) -> None:
        """Force-process whatever's left in the rolling buffer.

        Used by the bridge on push-to-talk release: even if the user only spoke
        for, say, 0.4 s, we still want Vosk to take a swing at it instead of
        waiting for the next utterance to push the buffer past the chunk size.
        """
        if not self._buffer:
            return
        chunk = bytes(self._buffer)
        self._buffer.clear()
        await self._queue.put(chunk)

    async def reset(self) -> None:
        """Discard whatever audio has been buffered but not yet transcribed.

        Sent by the bridge on push-to-talk press so leftover noise from before
        the button was held doesn't get glued onto the new utterance.
        """
        self._buffer.clear()

    async def _consume(self) -> None:
        try:
            while not self._stopped:
                chunk = await self._queue.get()
                try:
                    result = await self._stt.transcribe_chunk(chunk)
                except Exception as exc:  # noqa: BLE001
                    log.exception("STT error: %s", exc)
                    await self._send_json({"type": "error", "msg": str(exc)})
                    continue
                if not result or not result.text:
                    continue
                await self._send_json(
                    {
                        "type": "transcript",
                        "text": result.text,
                        "language": result.language,
                        "duration": result.duration,
                        "final": True,
                    }
                )
        except asyncio.CancelledError:
            raise
