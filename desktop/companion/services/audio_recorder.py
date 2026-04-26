"""Microphone capture for voice cloning.

Uses :mod:`sounddevice` (cross-platform PortAudio wrapper) to grab mono
PCM at 16 kHz — the same format MeloTTS / OpenVoice want for the
reference voice clip. The recording is written to a temp WAV that the
voice service can hand straight to ``OpenVoiceTTS.clone_voice``.
"""

from __future__ import annotations

import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Callable, Optional

TARGET_SAMPLE_RATE = 16_000
MAX_SECONDS = 6.0
CHANNELS = 1


class AudioCaptureError(RuntimeError):
    pass


class MicRecorder:
    """Record up to ``MAX_SECONDS`` of mono 16 kHz PCM into a temp WAV."""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._frames: list[bytes] = []
        self._stream = None
        self._lock = threading.Lock()
        self._started_at: float = 0.0
        self._tick_cb: Optional[Callable[[float], None]] = None
        self._done_cb: Optional[Callable[[Optional[Path], Optional[str]], None]] = None
        self._duration: float = MAX_SECONDS

    @property
    def is_recording(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        *,
        duration: float = MAX_SECONDS,
        on_tick: Optional[Callable[[float], None]] = None,
        on_done: Optional[Callable[[Optional[Path], Optional[str]], None]] = None,
    ) -> None:
        """Begin recording. ``on_done(path, error)`` fires when finished."""
        if self.is_recording:
            return

        try:
            import sounddevice  # noqa: F401
        except ImportError as exc:
            raise AudioCaptureError(
                "sounddevice is required for voice cloning. "
                "Install it with `pip install sounddevice`."
            ) from exc

        self._duration = max(0.5, min(duration, MAX_SECONDS))
        self._tick_cb = on_tick
        self._done_cb = on_done
        self._stop.clear()
        with self._lock:
            self._frames = []

        self._thread = threading.Thread(
            target=self._run, name="mic-recorder", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def cancel(self) -> None:
        self._done_cb = None
        self._tick_cb = None
        self.stop()

    # ------------------------- worker thread -------------------------------
    def _run(self) -> None:
        import numpy as np
        import sounddevice as sd

        path: Optional[Path] = None
        err: Optional[str] = None
        stream = None

        try:
            self._started_at = time.monotonic()

            def callback(indata, frames, time_info, status):  # noqa: ARG001
                if status:
                    # Don't bail on dropouts — just log.
                    print(f"  [mic] {status}")
                if self._stop.is_set():
                    return
                arr = np.array(indata, copy=True)
                if arr.ndim > 1:
                    arr = arr.mean(axis=1, keepdims=False)
                pcm = np.clip(arr, -1.0, 1.0)
                pcm = (pcm * 32767.0).astype(np.int16)
                with self._lock:
                    self._frames.append(pcm.tobytes())

            stream = sd.InputStream(
                samplerate=TARGET_SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                callback=callback,
            )
            stream.start()

            while not self._stop.is_set():
                elapsed = time.monotonic() - self._started_at
                if self._tick_cb is not None:
                    try:
                        self._tick_cb(min(elapsed, self._duration))
                    except Exception:
                        pass
                if elapsed >= self._duration:
                    break
                time.sleep(0.05)

            with self._lock:
                pcm_bytes = b"".join(self._frames)

            if not pcm_bytes:
                err = "no audio captured"
            else:
                path = self._write_wav(pcm_bytes)

        except Exception as exc:
            err = str(exc)
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

            if self._done_cb is not None:
                try:
                    self._done_cb(path, err)
                except Exception:
                    pass

    @staticmethod
    def _write_wav(pcm: bytes) -> Path:
        tmp_dir = Path(tempfile.gettempdir()) / "ar_companion_voice"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        path = tmp_dir / f"voice_{int(time.time() * 1000)}.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_SAMPLE_RATE)
            wf.writeframes(pcm)
        return path
