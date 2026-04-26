"""Live webcam test of the trained SignLSTM.

Open the default webcam, run MediaPipe Hands (Tasks API, VIDEO mode), apply
the same IDLE -> RECORDING -> IDLE state machine used by ``extract_dataset.py``,
and once a segment ends feed it through the trained checkpoint at
``settings.lstm_checkpoint_path``. The predicted label + confidence are shown
on the video.

Two prediction modes are supported:

    --mode sliding   (default)  Predict every ``--stride`` frames in real time
                                 from a rolling buffer. Predictions start as
                                 soon as a hand is visible (the buffer is
                                 zero-padded on the right, exactly like during
                                 training) so you see live updates while
                                 signing instead of having to drop your hands.

    --mode segment              Predict only when a full sign segment ends
                                 (hands missing for ``--patience`` frames).
                                 Matches the training distribution exactly,
                                 so it can be slightly higher confidence, at
                                 the cost of only outputting one label per
                                 sign after the hands leave the frame.

Sign segmentation (sliding mode)
--------------------------------
A label is committed to the running sentence as soon as the model has been
predicting it above ``--threshold`` for at least ``--stable-time`` seconds
(default 0.2 s) — i.e. while the user is still signing. The rolling buffer
is reset on either:

    * hands missing for ``--patience`` frames, OR
    * smoothed motion below ``--motion-threshold`` for ``--pause-frames``
      frames (a brief hold/rest between signs without lowering the hands).

By default consecutive duplicates are skipped; pass ``--allow-repeats`` to
allow them.

Sentence completion -> local LLM -> TTS
---------------------------------------
When no new sign has been committed for ``--sentence-timeout`` seconds
(default 1.0) the buffered tokens are sent to ``app.llm.local_client.
translate_signs`` (which talks to a local Ollama server) to be turned
into a natural sentence, then spoken via **in-process OpenVoice v2 +
MeloTTS** (no separate server needed). The ``active_voice_id`` and
``tts_speaker_key`` are read from ``app_state.json`` so a cloned voice
is automatically used when active. Audio plays through ``pygame``.

If OpenVoice checkpoints or dependencies are missing, the script falls
back to ``pyttsx3`` (OS-native TTS, no cloning). Pass ``--no-tts`` to
skip audio entirely. Requires ``pygame`` for playback.

If Ollama isn't reachable or the configured model isn't pulled,
``translate_signs`` falls back to a minimal local stub.

Hotkeys
-------
    q   quit
    r   reset the rolling buffer / last prediction
    c   clear the running sentence (and last NL output)
    h   toggle on-screen help

Usage
-----
    python scripts/live_predict.py
    python scripts/live_predict.py --mode sliding --stride 5
    python scripts/live_predict.py --camera 1 --threshold 0.6
"""

from __future__ import annotations

import argparse
import asyncio
import io
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import httpx
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# Make `app.*` importable when running from anywhere.
_HERE = Path(__file__).resolve()
_SERVER_ROOT = _HERE.parents[1]
if str(_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVER_ROOT))

from app.config import settings  # noqa: E402
from app.ml.lstm import load_checkpoint  # noqa: E402

NUM_LANDMARKS = 21
COORDS = 3
HAND_VECTOR_SIZE = NUM_LANDMARKS * COORDS         # 63
FRAME_VECTOR_SIZE = 2 * HAND_VECTOR_SIZE          # 126

LEFT_BLOCK = slice(0, HAND_VECTOR_SIZE)
RIGHT_BLOCK = slice(HAND_VECTOR_SIZE, FRAME_VECTOR_SIZE)

# Same 21-point hand connection graph as extract_dataset.py.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

_DEFAULT_MODEL_PATH = _SERVER_ROOT / "models" / "hand_landmarker.task"


# ---------------------------------------------------------------------------
# Per-frame -> 126-dim two-hand vector (mirrors extract_dataset.py exactly)
# ---------------------------------------------------------------------------
def landmarks_to_frame_vector(result) -> np.ndarray:
    vec = np.zeros(FRAME_VECTOR_SIZE, dtype=np.float32)
    if not result.hand_landmarks:
        return vec
    handedness = result.handedness or []
    for i, lm_list in enumerate(result.hand_landmarks):
        label = "Left"
        if i < len(handedness) and handedness[i]:
            label = handedness[i][0].category_name
        offset = 0 if label == "Left" else HAND_VECTOR_SIZE
        flat = np.array(
            [[lm.x, lm.y, lm.z] for lm in lm_list],
            dtype=np.float32,
        ).flatten()
        if flat.size == HAND_VECTOR_SIZE:
            vec[offset : offset + HAND_VECTOR_SIZE] = flat
    return vec


def to_dominant_hand_seq(seq_126: np.ndarray) -> np.ndarray:
    """(T, 126) -> (T, 63), Left if present else Right else zeros.

    Matches train_from_dataset.py so live inputs share the train distribution.
    """
    out = np.zeros((seq_126.shape[0], HAND_VECTOR_SIZE), dtype=np.float32)
    left = seq_126[:, LEFT_BLOCK]
    right = seq_126[:, RIGHT_BLOCK]
    has_left = np.any(left != 0.0, axis=1)
    has_right = np.any(right != 0.0, axis=1)
    out[has_left] = left[has_left]
    use_right = (~has_left) & has_right
    out[use_right] = right[use_right]
    return out


def standardize_sequence(frames: list[np.ndarray], seq_len: int) -> np.ndarray:
    """Zero-pad (right) or truncate to exactly ``seq_len`` rows."""
    if not frames:
        return np.zeros((seq_len, FRAME_VECTOR_SIZE), dtype=np.float32)
    arr = np.stack(frames, axis=0).astype(np.float32)
    n = arr.shape[0]
    if n >= seq_len:
        return arr[:seq_len]
    pad = np.zeros((seq_len - n, FRAME_VECTOR_SIZE), dtype=np.float32)
    return np.concatenate([arr, pad], axis=0)


def hand_motion_energy(curr: np.ndarray, prev: np.ndarray) -> float:
    """Mean per-landmark L2 displacement between two 126-d frames.

    Uses each hand block independently and averages over hands that are
    visible in BOTH frames. Returns 0 if no hand is visible in both, which
    naturally treats appearance / disappearance as a non-motion event (the
    hand-presence state machine already handles those).
    """
    motions: list[float] = []
    for blk in (LEFT_BLOCK, RIGHT_BLOCK):
        a = curr[blk]
        b = prev[blk]
        if np.any(a) and np.any(b):
            diff = a.reshape(NUM_LANDMARKS, COORDS) - b.reshape(
                NUM_LANDMARKS, COORDS
            )
            motions.append(float(np.linalg.norm(diff, axis=1).mean()))
    if not motions:
        return 0.0
    return float(np.mean(motions))


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_hand_landmarks(frame_bgr: np.ndarray, result) -> None:
    if not result.hand_landmarks:
        return
    h, w = frame_bgr.shape[:2]
    for lm_list in result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame_bgr, pts[a], pts[b], (0, 255, 0), 2, cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(frame_bgr, (x, y), 3, (0, 0, 255), -1, cv2.LINE_AA)


def draw_hud(
    frame_bgr: np.ndarray,
    state: str,
    buf_len: int,
    last_label: str,
    last_conf: float,
    threshold: float,
    fps: float,
    show_help: bool,
    cooldown_remaining: float = 0.0,
    motion: float = 0.0,
    motion_threshold: float = 0.0,
    sentence: Optional[list[str]] = None,
    natural_text: str = "",
    is_idle: bool = False,
    pending_dispatch: bool = False,
) -> None:
    h, w = frame_bgr.shape[:2]

    cv2.rectangle(frame_bgr, (0, 0), (w, 38), (0, 0, 0), -1)
    state_color = (0, 0, 255) if state == "RECORDING" else (0, 200, 0)
    cd_str = f"   cool: {cooldown_remaining:.1f}s" if cooldown_remaining > 0 else ""
    motion_str = (
        f"   mot:{motion:5.3f}/{motion_threshold:.3f}"
        if motion_threshold > 0
        else ""
    )
    cv2.putText(
        frame_bgr,
        f"State: {state}   Buf: {buf_len}   FPS: {fps:5.1f}{motion_str}{cd_str}",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        state_color,
        2,
        cv2.LINE_AA,
    )

    # Running sentence of committed signs (just below the status bar).
    sentence_bottom = 38
    if sentence:
        sent_h = 44
        cv2.rectangle(
            frame_bgr, (0, 38), (w, 38 + sent_h), (20, 20, 20), -1
        )
        prefix = "..." if pending_dispatch else "TOKENS:"
        sentence_text = f"{prefix} " + " ".join(s.upper() for s in sentence)
        color = (180, 180, 180) if pending_dispatch else (255, 255, 255)
        cv2.putText(
            frame_bgr,
            sentence_text,
            (12, 38 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            color,
            2,
            cv2.LINE_AA,
        )
        sentence_bottom = 38 + sent_h

    # Natural-language sentence (most recent local-LLM translation).
    if natural_text:
        nl_h = 44
        cv2.rectangle(
            frame_bgr,
            (0, sentence_bottom),
            (w, sentence_bottom + nl_h),
            (10, 60, 10),
            -1,
        )
        # Truncate to fit on one line (rough char-width estimate).
        max_chars = max(20, w // 14)
        nl = natural_text if len(natural_text) <= max_chars else natural_text[: max_chars - 1] + "\u2026"
        cv2.putText(
            frame_bgr,
            nl,
            (12, sentence_bottom + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (200, 255, 200),
            2,
            cv2.LINE_AA,
        )

    band_h = 90
    cv2.rectangle(frame_bgr, (0, h - band_h), (w, h), (0, 0, 0), -1)
    if last_label:
        commit = last_conf >= threshold
        color = (0, 255, 0) if commit else (0, 165, 255)
        prefix = "" if commit else "(low) "
        cv2.putText(
            frame_bgr,
            f"{prefix}{last_label.upper()}",
            (14, h - 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            color,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"confidence: {last_conf*100:5.1f}%   threshold: {threshold*100:.0f}%",
            (14, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        bar_x0, bar_x1 = w - 220, w - 20
        bar_y = h - 56
        cv2.rectangle(frame_bgr, (bar_x0, bar_y), (bar_x1, bar_y + 18), (60, 60, 60), -1)
        fill = int((bar_x1 - bar_x0) * float(np.clip(last_conf, 0.0, 1.0)))
        cv2.rectangle(
            frame_bgr, (bar_x0, bar_y), (bar_x0 + fill, bar_y + 18), color, -1
        )
    else:
        ready_text = "IDLE" if is_idle else "READY"
        ready_color = (140, 180, 255) if is_idle else (180, 180, 180)
        cv2.putText(
            frame_bgr,
            ready_text,
            (14, h - 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            ready_color,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            "make a sign",
            (14, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (140, 140, 140),
            1,
            cv2.LINE_AA,
        )

    if show_help:
        help_lines = [
            "q: quit   r: reset   c: clear sentence   h: toggle help",
        ]
        # Push help below whichever bars are currently rendered so they
        # don't overlap.
        bars_h = (44 if sentence else 0) + (44 if natural_text else 0)
        y = (38 + bars_h + 18) if bars_h else 60
        for line in help_lines:
            cv2.putText(
                frame_bgr, line, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
            )
            y += 18


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
class Predictor:
    def __init__(self, threshold: float) -> None:
        model, labels = load_checkpoint()
        if model is None or not labels:
            raise SystemExit(
                f"No trained checkpoint found at {settings.lstm_checkpoint_path}.\n"
                "Run scripts/train_from_dataset.py first."
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.labels = labels
        self.threshold = threshold

    @torch.no_grad()
    def predict(self, seq_126: np.ndarray) -> tuple[str, float]:
        """seq_126: (seq_len, 126). Returns (label, confidence)."""
        if self.model.input_size == HAND_VECTOR_SIZE:
            seq = to_dominant_hand_seq(seq_126)
        else:
            seq = seq_126
        x = torch.from_numpy(seq).float().unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])


# ---------------------------------------------------------------------------
# Local-LLM translation + TTS playback (runs in a background thread so the
# live preview keeps rendering at full FPS while the local-LLM call /
# synthesis is in flight).
# ---------------------------------------------------------------------------
class _TtsUtterance:
    __slots__ = ("text", "emotion", "intensity")

    def __init__(self, text: str, emotion: str, intensity: float) -> None:
        self.text = text
        self.emotion = emotion
        self.intensity = intensity


class _TTSWorker:
    """Background thread: direct in-process OpenVoice synthesis + pygame playback.

    Loads OpenVoiceTTS lazily on the worker thread so the main webcam loop
    stays at full FPS.  Reads ``active_voice_id`` / ``tts_speaker_key`` from
    ``app_state.json`` so a cloned voice is automatically used when active.

    Falls back to ``pyttsx3`` (offline, OS-native TTS) when OpenVoice
    checkpoints or dependencies are missing.
    """

    def __init__(self, disabled: bool, api_base: str = "") -> None:
        self._enabled = False
        self._q: "queue.Queue[Optional[_TtsUtterance]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._remote_server_http_base = ""
        self._remote_device_id = ""

        if disabled:
            return

        # Check that at least one audio backend is available.
        has_pygame = False
        has_pyttsx3 = False
        try:
            import pygame  # type: ignore  # noqa: F401
            has_pygame = True
        except ImportError:
            pass
        try:
            import pyttsx3  # type: ignore  # noqa: F401
            has_pyttsx3 = True
        except ImportError:
            pass
        if not has_pygame and not has_pyttsx3:
            print(
                "Note: TTS requires pygame or pyttsx3. "
                "Install with: pip install pygame pyttsx3"
            )
            return

        self._enabled = True
        self._thread = threading.Thread(
            target=self._run, name="tts-worker", daemon=True
        )
        self._thread.start()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable_remote(self, server_http_base: str, device_id: str) -> None:
        self._remote_server_http_base = (server_http_base or "").strip()
        self._remote_device_id = (device_id or "").strip()

    def say(
        self, text: str, *, emotion: str = "neutral", intensity: float = 0.5
    ) -> None:
        if self._enabled and text:
            self._q.put(_TtsUtterance(text, emotion, float(intensity)))

    def shutdown(self) -> None:
        if self._thread is not None:
            self._q.put(None)
            self._thread.join(timeout=5.0)

    # ---- worker thread ----------------------------------------------------

    def _run(self) -> None:
        if self._remote_server_http_base and self._remote_device_id:
            print(
                "  [TTS] remote mode: POST /api/voice/speak "
                f"(device_id={self._remote_device_id})"
            )
            self._run_remote_loop()
            return

        # Try to load OpenVoice in-process (heavy: loads models once).
        tts_engine = self._try_load_openvoice()

        if tts_engine is not None:
            # OpenVoice available — use pygame for audio output.
            try:
                import pygame  # type: ignore

                pygame.mixer.init(
                    frequency=int(settings.openvoice_output_sample_rate),
                    size=-16,
                    channels=1,
                    buffer=2048,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [TTS] pygame.mixer init failed ({exc}); trying pyttsx3 fallback.")
                tts_engine = None

        if tts_engine is not None:
            print("  [TTS] OpenVoice loaded in-process (cloned voice supported).")
            self._run_openvoice_loop(tts_engine)
        else:
            # pyttsx3 fallback — does NOT use pygame at all (owns its audio).
            print("  [TTS] OpenVoice unavailable; trying pyttsx3 fallback...")
            self._run_pyttsx3_loop()

    def _run_remote_loop(self) -> None:
        base = self._remote_server_http_base.rstrip("/")
        url = f"{base}/api/voice/speak"
        while True:
            item = self._q.get()
            if item is None:
                return
            if not item.text:
                continue
            payload = {
                "device_id": self._remote_device_id,
                "text": item.text,
                "emotion": item.emotion,
                "intensity": item.intensity,
            }
            try:
                with httpx.Client(timeout=20.0) as client:
                    r = client.post(url, json=payload)
                    r.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] remote TTS failed: {exc}")

    def _run_openvoice_loop(self, engine) -> None:
        import pygame  # type: ignore

        while True:
            item = self._q.get()
            if item is None:
                return
            if not item.text:
                continue
            self._speak_openvoice(item, engine, pygame)

    def _run_pyttsx3_loop(self) -> None:
        # On Windows, use SAPI COM directly — pyttsx3.runAndWait() has a
        # known threading bug where it hangs after the first invocation.
        if sys.platform == "win32":
            try:
                import comtypes  # noqa: F401 — force COM init on this thread
            except ImportError:
                pass
            try:
                import win32com.client  # type: ignore

                # CoInitialize for this thread (MTA/STA).
                import pythoncom  # type: ignore
                pythoncom.CoInitialize()

                voice = win32com.client.Dispatch("SAPI.SpVoice")
                print("  [TTS] Windows SAPI ready (OS-native voice, no cloning).")

                while True:
                    item = self._q.get()
                    if item is None:
                        pythoncom.CoUninitialize()
                        return
                    if not item.text:
                        continue
                    try:
                        voice.Speak(item.text)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  [warn] SAPI TTS: {exc}")
                return  # noqa: TRY300
            except ImportError:
                print("  [TTS] win32com not available, trying pyttsx3...")
            except Exception as exc:  # noqa: BLE001
                print(f"  [TTS] SAPI init failed ({exc}), trying pyttsx3...")

        # Non-Windows or win32com unavailable: try pyttsx3.
        try:
            import pyttsx3  # type: ignore

            tts = pyttsx3.init()
            tts.setProperty("rate", tts.getProperty("rate") - 30)
            print("  [TTS] pyttsx3 ready (OS-native voice, no cloning).")
        except Exception as exc:  # noqa: BLE001
            print(f"  [TTS] pyttsx3 init failed: {exc}. Audio disabled.")
            self._enabled = False
            return

        while True:
            item = self._q.get()
            if item is None:
                return
            if not item.text:
                continue
            try:
                tts.say(item.text)
                tts.runAndWait()
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] pyttsx3 TTS: {exc}")

    @staticmethod
    def _try_load_openvoice():
        """Attempt to import and warm up the OpenVoiceTTS engine.

        Returns the engine instance on success, ``None`` on any failure.
        """
        try:
            from app.tts.openvoice_client import OpenVoiceTTS

            engine = OpenVoiceTTS()
            # Warm up synchronously (we're in a worker thread).
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(engine.warmup())
            finally:
                loop.close()
            if not engine.is_ready:
                return None
            return engine
        except Exception as exc:  # noqa: BLE001
            print(f"  [TTS] OpenVoice init failed: {exc}")
            return None

    @staticmethod
    def _speak_openvoice(item: _TtsUtterance, engine, pygame) -> None:
        """Synthesise via in-process OpenVoiceTTS and play through pygame.

        Re-reads the active voice / speaker from ``app_state.json`` on every
        utterance, so switching a voice in the web UI takes effect on the
        very next sentence dispatched by ``live_predict.py`` -- no script
        restart needed.
        """
        from app.core.state import get_state

        state = get_state()
        voice_id = state.data.active_voice_id
        speaker_key = state.data.tts_speaker_key

        loop = asyncio.new_event_loop()
        try:
            pcm = loop.run_until_complete(
                engine.synthesize_to_pcm(
                    item.text,
                    voice_id=voice_id,
                    speaker_key=speaker_key,
                    emotion=item.emotion,
                    intensity=item.intensity,
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] OpenVoice synthesis: {exc}")
            return
        finally:
            loop.close()

        if not pcm:
            return

        # Wrap raw PCM in a WAV header so pygame can play it.
        import wave as _wave
        wav_buf = io.BytesIO()
        with _wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(settings.openvoice_output_sample_rate)
            wf.writeframes(pcm)
        wav_buf.seek(0)

        try:
            sound = pygame.mixer.Sound(wav_buf)
            channel = sound.play()
            while channel.get_busy():
                time.sleep(0.05)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] TTS playback: {exc}")


def _shutdown_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel all pending tasks and close the event loop cleanly."""
    try:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
    except Exception:  # noqa: BLE001
        pass
    finally:
        loop.close()


async def _collect_translation(
    tokens: list[str], emotion: str, intensity: float
) -> str:
    """Drive ``app.llm.local_client.translate_signs`` and concatenate the
    streamed tokens into a single string."""
    from app.llm.local_client import translate_signs

    pieces: list[str] = []
    async for chunk in translate_signs(tokens, emotion=emotion, intensity=intensity):
        pieces.append(chunk)
    return "".join(pieces).strip()


def _sentence_worker(
    tokens: list[str],
    emotion: str,
    intensity: float,
    tts_worker: _TTSWorker,
    result_q: "queue.Queue[tuple[str, object]]",
    do_llm: bool,
) -> None:
    """Run translate_signs (optional) + queue text for the TTS worker.

    Posts events to ``result_q`` for the main thread to consume:
      ("text", str)    natural-language sentence ready to display
      ("error", str)   non-fatal failure during LLM
      ("done", None)   pipeline finished (success or failure)

    TTS playback happens asynchronously inside ``_TTSWorker``; this
    worker just enqueues the sentence and returns immediately so the
    next sign phrase isn't blocked on audio playback.
    """
    natural_text = " ".join(tokens)
    try:
        if do_llm:
            try:
                loop = asyncio.new_event_loop()
                try:
                    natural_text = loop.run_until_complete(
                        _collect_translation(tokens, emotion, intensity)
                    ) or natural_text
                finally:
                    # Properly shut down any lingering httpx async generators.
                    _shutdown_loop(loop)
            except Exception as exc:  # noqa: BLE001
                result_q.put(("error", f"LLM: {exc}"))
        result_q.put(("text", natural_text))
        if tts_worker.enabled and natural_text:
            tts_worker.say(
                natural_text, emotion=emotion, intensity=intensity
            )
    finally:
        result_q.put(("done", None))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def list_cameras(max_index: int = 6) -> None:
    """Probe ``cv2.VideoCapture`` indices 0..max_index-1 and print which open.

    On Windows we use the DirectShow backend (CAP_DSHOW), same as ``main()``,
    so the listed indices match what you'll get with ``--camera N``.
    """
    print("Probing camera indices (DirectShow backend)...")
    found: list[int] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, frame = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            label = "OK " if ok and frame is not None else "open-but-no-frame"
            print(f"  [{i}] {label:<18} {w}x{h}")
            found.append(i)
        else:
            print(f"  [{i}] not available")
        cap.release()
    if not found:
        print("No cameras detected.")
    else:
        print(
            f"\nDetected cameras at: {found}. "
            f"Pass with --camera <index>. The laptop camera is usually 0."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live webcam test for the trained SignLSTM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--camera", type=int, default=1,
                   help="cv2.VideoCapture index. 0 = laptop built-in, "
                        "1 = first external/secondary camera (default).")
    p.add_argument("--list-cameras", action="store_true",
                   help="Probe available camera indices and exit.")
    p.add_argument("--width",  type=int, default=settings.camera_width)
    p.add_argument("--height", type=int, default=settings.camera_height)
    p.add_argument("--mode", choices=("segment", "sliding"), default="sliding",
                   help="sliding: predict every --stride frames in real time; "
                        "segment: predict only at end of sign (after hands leave).")
    p.add_argument("--seq-len", type=int, default=settings.lstm_window_size,
                   help="Frames per inference window (must match training).")
    p.add_argument("--patience", type=int, default=5,
                   help="Frames of missing hands to end a segment (segment mode).")
    p.add_argument("--min-frames", type=int, default=5,
                   help="Discard segments with fewer real frames than this.")
    p.add_argument("--stride", type=int, default=5,
                   help="Predict every N frames in sliding mode.")
    p.add_argument("--threshold", type=float, default=settings.lstm_confidence_threshold,
                   help="Confidence threshold for a 'committed' prediction.")
    p.add_argument("--idle-label", default="idle",
                   help="Label name treated as 'no sign'. When the model "
                        "predicts this label the on-screen sign text is hidden "
                        "and the HUD shows 'ready' instead.")
    p.add_argument("--cooldown", type=float, default=1.5,
                   help="After committing a non-idle prediction, suppress "
                        "further committed predictions for this many seconds.")
    # --- Sign segmentation (sliding mode) ---------------------------------
    p.add_argument("--motion-threshold", type=float, default=0.004,
                   help="Smoothed mean per-landmark motion (in normalized "
                        "image units) below which the hands are considered "
                        "'at rest'. Lower => more sensitive (more boundaries).")
    p.add_argument("--motion-smooth", type=int, default=5,
                   help="Number of recent frames to average for motion "
                        "energy. Larger smooths jitter but adds latency.")
    p.add_argument("--pause-frames", type=int, default=6,
                   help="Frames of below-threshold motion (with hands still "
                        "visible) that count as a sign-to-sign boundary.")
    p.add_argument("--stable-frames", type=int, default=2,
                   help="(Deprecated, ignored.) Replaced by --stable-time. "
                        "Kept for backwards compatibility with old commands.")
    p.add_argument("--max-sentence", type=int, default=12,
                   help="How many of the most recently committed signs to "
                        "keep on screen as a running sentence.")
    p.add_argument("--allow-repeats", action="store_true",
                   help="Allow the same sign to be committed twice in a row. "
                        "Off by default to avoid duplicate emissions when a "
                        "single sign spans multiple sub-segments.")
    p.add_argument("--stable-time", type=float, default=0.2,
                   help="A label must be predicted above --threshold for at "
                        "least this many seconds before it is committed to "
                        "the running sentence (live, mid-sign).")
    # --- Sentence completion -> local LLM -> TTS --------------------------
    p.add_argument("--sentence-timeout", type=float, default=1.0,
                   help="If no new sign has been committed for this many "
                        "seconds the buffer is treated as a completed "
                        "sentence and dispatched to the local-LLM translator "
                        "(Ollama) and (if available) the local TTS engine.")
    p.add_argument("--emotion", default="neutral",
                   help="Tone preset forwarded to the local-LLM translator "
                        "(neutral / friendly / excited / serious / calm / urgent).")
    p.add_argument("--intensity", type=float, default=0.5,
                   help="Tone intensity 0..1 forwarded to the translator.")
    p.add_argument("--no-llm", action="store_true",
                   help="Skip the local-LLM translation step; the raw sign "
                        "tokens are spoken directly by the TTS instead.")
    p.add_argument("--no-tts", action="store_true",
                   help="Disable TTS playback. The natural-language "
                        "sentence is still printed and shown on screen.")
    p.add_argument(
        "--tts-server",
        default="",
        help=(
            "If set, send TTS to the FastAPI server instead of playing on this PC. "
            "Example: http://localhost:8000"
        ),
    )
    p.add_argument(
        "--tts-device-id",
        default="glasses-01",
        help="Target device_id on the server's /ws/reception/{device_id}.",
    )
    p.add_argument(
        "--api-base",
        default="",
        help="(Deprecated, ignored.) TTS now runs in-process via OpenVoice. "
            "Kept for backwards compatibility.",
    )
    p.add_argument("--max-hands", type=int, default=2)
    p.add_argument("--detection-confidence", type=float, default=0.5)
    p.add_argument("--presence-confidence", type=float, default=0.5)
    p.add_argument("--tracking-confidence", type=float, default=0.5)
    p.add_argument("--model", default=str(_DEFAULT_MODEL_PATH),
                   help="Path to hand_landmarker.task.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    if args.list_cameras:
        list_cameras()
        return 0

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"Error: hand_landmarker.task not found at {model_path}", file=sys.stderr)
        return 1

    predictor = Predictor(args.threshold)
    print(f"Loaded LSTM with labels: {predictor.labels}")
    print(
        f"Mode: {args.mode}   threshold: {args.threshold}   "
        f"stable: {args.stable_time:.2f}s   sentence-timeout: "
        f"{args.sentence_timeout:.2f}s   "
        f"local-llm: {'off (fallback)' if args.no_llm else f'{settings.ollama_model} @ {settings.ollama_base_url}'}"
    )
    if not args.no_tts:
        print("TTS: OpenVoice in-process (falls back to pyttsx3 if unavailable)")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS,          settings.camera_fps)
    if not cap.isOpened():
        print(f"Error: could not open camera index {args.camera}", file=sys.stderr)
        return 1

    fps_hint = cap.get(cv2.CAP_PROP_FPS) or float(settings.camera_fps)
    if fps_hint <= 1e-3:
        fps_hint = 30.0
    ms_per_frame = 1000.0 / fps_hint

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=args.max_hands,
        min_hand_detection_confidence=args.detection_confidence,
        min_hand_presence_confidence=args.presence_confidence,
        min_tracking_confidence=args.tracking_confidence,
    )

    state = "IDLE"
    current_seq: list[np.ndarray] = []
    missing_streak = 0
    sliding_buf: deque[np.ndarray] = deque(maxlen=args.seq_len)
    last_label = ""
    last_conf = 0.0
    show_help = True
    frames_since_last_pred = 0
    cooldown_until = 0.0  # monotonic seconds; new commits suppressed before this

    # --- Sign segmentation state (shared by both modes) -------------------
    sentence: list[str] = []  # committed signs, displayed as a running line
    # Smoothed per-frame motion energy + previous-frame vector for diffing.
    motion_buf: deque[float] = deque(maxlen=max(1, args.motion_smooth))
    prev_frame_vec = np.zeros(FRAME_VECTOR_SIZE, dtype=np.float32)
    low_motion_streak = 0

    # Time-based stable-commit state (sliding mode). A label only enters
    # `sentence` after it has been the dominant prediction for at least
    # `--stable-time` seconds. After commit, the candidate is "locked" and
    # only re-arms on a label change, idle, motion-pause or hand-gone.
    candidate_label: Optional[str] = None
    candidate_first_seen = 0.0
    candidate_best_conf = 0.0
    candidate_committed = False

    last_was_idle = False  # last non-empty prediction was the idle class
    last_commit_time = 0.0
    last_natural_text = ""  # most recent local-LLM translation (for HUD)

    # Background pipeline: tokens -> local LLM -> TTS (local playback or remote server).
    tts_worker = _TTSWorker(disabled=args.no_tts)
    if not args.no_tts and args.tts_server:
        tts_worker.enable_remote(args.tts_server, args.tts_device_id)
    result_q: queue.Queue[tuple[str, object]] = queue.Queue()
    pending_dispatch = False

    def commit_sign(label: str, conf: float, source: str) -> None:
        """Append ``label`` to the running sentence (with dedup) and log it."""
        nonlocal last_commit_time
        if not args.allow_repeats and sentence and sentence[-1] == label:
            print(f"  {source} -> {label} ({conf*100:.1f}%) [duplicate, skipped]")
            return
        sentence.append(label)
        if len(sentence) > args.max_sentence:
            del sentence[: len(sentence) - args.max_sentence]
        last_commit_time = time.monotonic()
        print(
            f"  {source} -> {label} ({conf*100:.1f}%)   "
            f"sentence: {' '.join(sentence)}"
        )

    idle_known = args.idle_label in predictor.labels
    if not idle_known:
        print(
            f"Note: '{args.idle_label}' is not a trained label. The model has "
            f"no idle class, so it will always pick a sign. Train one by "
            f"running `extract_dataset.py --mode windows --label "
            f"{args.idle_label}` on an idle clip and re-training."
        )

    fps_t0 = time.monotonic()
    fps_count = 0
    fps_value = 0.0

    print(
        "Press 'q' to quit, 'r' to reset buffer, 'c' to clear sentence, "
        "'h' to toggle help."
    )

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    print("Camera read failed; exiting.", file=sys.stderr)
                    break

                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int(frame_idx * ms_per_frame)
                result = landmarker.detect_for_video(mp_image, ts_ms)

                has_hand = bool(result.hand_landmarks)
                frame_vec = landmarks_to_frame_vector(result)
                smoothed_motion = 0.0  # populated by sliding mode each frame

                now_s = time.monotonic()
                in_cooldown = now_s < cooldown_until

                if args.mode == "segment":
                    if state == "IDLE":
                        if has_hand:
                            state = "RECORDING"
                            current_seq = [frame_vec]
                            missing_streak = 0
                    else:  # RECORDING
                        current_seq.append(frame_vec)
                        if has_hand:
                            missing_streak = 0
                        else:
                            missing_streak += 1
                            if missing_streak >= args.patience:
                                trailing = missing_streak
                                captured = (
                                    current_seq[:-trailing]
                                    if trailing else list(current_seq)
                                )
                                if len(captured) >= args.min_frames:
                                    seq = standardize_sequence(captured, args.seq_len)
                                    label, conf = predictor.predict(seq)
                                    is_idle = label == args.idle_label
                                    if is_idle:
                                        # Don't overwrite the previously
                                        # displayed sign with "idle"; just log.
                                        print(
                                            f"  segment ({len(captured)} frames) -> "
                                            f"idle ({conf*100:.1f}%) [hidden]"
                                        )
                                    elif in_cooldown:
                                        print(
                                            f"  segment ({len(captured)} frames) -> "
                                            f"{label} ({conf*100:.1f}%) [cooldown]"
                                        )
                                    else:
                                        last_label, last_conf = label, conf
                                        if conf >= args.threshold:
                                            cooldown_until = now_s + args.cooldown
                                            commit_sign(
                                                label, conf,
                                                f"segment ({len(captured)}f)",
                                            )
                                        else:
                                            print(
                                                f"  segment ({len(captured)} "
                                                f"frames) -> {label} "
                                                f"({conf*100:.1f}%) [low conf]"
                                            )
                                state = "IDLE"
                                current_seq = []
                                missing_streak = 0
                else:  # sliding (real-time, time-stable commits)
                    # --- Motion energy (smoothed) ---------------------------
                    inst_motion = hand_motion_energy(frame_vec, prev_frame_vec)
                    motion_buf.append(inst_motion)
                    smoothed_motion = float(np.mean(motion_buf)) if motion_buf else 0.0
                    prev_frame_vec = frame_vec

                    # --- Hand-presence + motion bookkeeping -----------------
                    if has_hand:
                        sliding_buf.append(frame_vec)
                        missing_streak = 0
                        state = "RECORDING"
                        if smoothed_motion >= args.motion_threshold:
                            low_motion_streak = 0
                        else:
                            low_motion_streak += 1
                    else:
                        missing_streak += 1
                        low_motion_streak = 0
                        state = "IDLE"

                    # --- Live prediction every --stride frames --------------
                    frames_since_last_pred += 1
                    if (
                        has_hand
                        and len(sliding_buf) >= max(args.min_frames, 1)
                        and frames_since_last_pred >= args.stride
                    ):
                        frames_since_last_pred = 0
                        seq = standardize_sequence(list(sliding_buf), args.seq_len)
                        label, conf = predictor.predict(seq)
                        is_idle = label == args.idle_label
                        if is_idle:
                            # Idle "blanks out" the live label and resets
                            # any in-progress candidate so a new sign can
                            # start cleanly the next time we see real motion.
                            last_label, last_conf = "", 0.0
                            last_was_idle = True
                            candidate_label = None
                            candidate_committed = False
                        else:
                            last_label, last_conf = label, conf
                            last_was_idle = False
                            if conf >= args.threshold:
                                # Time-based stability: a new label restarts
                                # the timer; the same label keeps it.
                                if candidate_label != label:
                                    candidate_label = label
                                    candidate_first_seen = now_s
                                    candidate_best_conf = conf
                                    candidate_committed = False
                                else:
                                    if conf > candidate_best_conf:
                                        candidate_best_conf = conf
                                if (
                                    not candidate_committed
                                    and (now_s - candidate_first_seen)
                                    >= args.stable_time
                                ):
                                    commit_sign(
                                        candidate_label,
                                        candidate_best_conf,
                                        f"stable ({args.stable_time:.2f}s)",
                                    )
                                    candidate_committed = True

                    # --- Buffer-reset on hand-gone or motion-pause ---------
                    # We don't commit here (commits already happened mid-sign
                    # via the time-stable path); we just clear stale frames
                    # and re-arm the candidate so the next sign starts fresh.
                    end_by_missing = (
                        missing_streak >= args.patience and sliding_buf
                    )
                    end_by_pause = (
                        has_hand
                        and low_motion_streak >= args.pause_frames
                        and sliding_buf
                    )
                    if end_by_missing or end_by_pause:
                        sliding_buf.clear()
                        candidate_label = None
                        candidate_committed = False
                        low_motion_streak = 0
                        frames_since_last_pred = 0
                        last_label, last_conf = "", 0.0
                        if end_by_missing:
                            # Treat hand-gone as the user "putting their
                            # hands down": the IDLE indicator should clear,
                            # not stick around as the latest state.
                            last_was_idle = False

                    # --- Sentence-completion dispatch ----------------------
                    # If the user has paused adding new words for
                    # `--sentence-timeout` seconds, ship the buffered tokens
                    # off to the local LLM for translation (and then to the TTS).
                    # The sentence buffer is cleared immediately so the next
                    # utterance can start building while the previous one
                    # is still being translated / spoken in the background.
                    if (
                        sentence
                        and not pending_dispatch
                        and last_commit_time > 0.0
                        and (now_s - last_commit_time) >= args.sentence_timeout
                    ):
                        pending_dispatch = True
                        tokens_to_send = list(sentence)
                        sentence.clear()
                        last_commit_time = 0.0
                        print(
                            f"  [dispatch] tokens={tokens_to_send} -> "
                            f"LLM{'+TTS' if tts_worker.enabled else ''}"
                        )
                        threading.Thread(
                            target=_sentence_worker,
                            args=(
                                tokens_to_send,
                                args.emotion,
                                args.intensity,
                                tts_worker,
                                result_q,
                                not args.no_llm,
                            ),
                            daemon=True,
                        ).start()

                # --- Drain completed dispatch results -----------------------
                # Runs every frame so the natural-language sentence appears
                # on screen as soon as the LLM responds. The sentence buffer
                # was already cleared at dispatch time so the user can start
                # signing the next utterance immediately while TTS plays.
                while True:
                    try:
                        kind, payload = result_q.get_nowait()
                    except queue.Empty:
                        break
                    if kind == "text":
                        last_natural_text = str(payload)
                        print(f"  [LLM] -> {last_natural_text}")
                    elif kind == "error":
                        print(f"  [warn] {payload}")
                    elif kind == "done":
                        pending_dispatch = False

                draw_hand_landmarks(frame_bgr, result)
                buf_len = (
                    len(current_seq) if args.mode == "segment"
                    else len(sliding_buf)
                )

                fps_count += 1
                now = time.monotonic()
                if now - fps_t0 >= 0.5:
                    fps_value = fps_count / (now - fps_t0)
                    fps_t0 = now
                    fps_count = 0

                draw_hud(
                    frame_bgr,
                    state=state,
                    buf_len=buf_len,
                    last_label=last_label,
                    last_conf=last_conf,
                    threshold=args.threshold,
                    fps=fps_value,
                    show_help=show_help,
                    cooldown_remaining=max(0.0, cooldown_until - time.monotonic()),
                    motion=smoothed_motion if args.mode == "sliding" else 0.0,
                    motion_threshold=(
                        args.motion_threshold if args.mode == "sliding" else 0.0
                    ),
                    sentence=sentence,
                    natural_text=last_natural_text,
                    is_idle=last_was_idle and has_hand,
                    pending_dispatch=pending_dispatch,
                )

                cv2.imshow("live_predict", frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    current_seq = []
                    sliding_buf.clear()
                    motion_buf.clear()
                    prev_frame_vec = np.zeros(FRAME_VECTOR_SIZE, dtype=np.float32)
                    missing_streak = 0
                    low_motion_streak = 0
                    frames_since_last_pred = 0
                    candidate_label = None
                    candidate_committed = False
                    last_was_idle = False
                    state = "IDLE"
                    last_label, last_conf = "", 0.0
                    print("(reset)")
                if key == ord("c"):
                    sentence.clear()
                    last_natural_text = ""
                    last_commit_time = 0.0
                    print("(sentence cleared)")
                if key == ord("h"):
                    show_help = not show_help

                frame_idx += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()
            tts_worker.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
