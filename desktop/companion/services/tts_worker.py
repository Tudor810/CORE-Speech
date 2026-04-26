"""In-process OpenVoice TTS worker.

Lifted almost verbatim from ``server/scripts/live_predict.py``'s
``_TTSWorker`` so the desktop app reuses the cloned-voice playback path
that already works in the live_predict CLI. The differences are:

* The worker exposes an ``on_done`` callback so the live engine can
  signal "audio finished" back to the GUI without polling.

* OpenVoice loads lazily on the worker thread; ``warmup_async`` lets
  the GUI kick off the load early on startup so the first sentence
  isn't penalised.

If OpenVoice / pygame aren't usable the worker falls back to:
  1. Windows SAPI via ``win32com`` (offline, no cloning), or
  2. ``pyttsx3`` on other platforms.

Pass ``disabled=True`` to mute everything (the engine still emits
``audio_end`` immediately so the UI animates correctly).
"""

from __future__ import annotations

import asyncio
import io
import queue
import sys
import threading
import time
import wave as _wave
from dataclasses import dataclass
from typing import Callable, Optional

# Pull in the desktop runtime first so ``server/`` is on ``sys.path``
# before we touch ``app.*``.
from .. import runtime as _runtime  # noqa: F401  (side effect: sys.path insert)
from app.config import settings  # noqa: E402


@dataclass
class _Utterance:
    text: str
    emotion: str
    intensity: float
    on_done: Optional[Callable[[], None]] = None


class TtsWorker:
    """Lazy OpenVoice + pygame playback in a background thread."""

    def __init__(self, *, disabled: bool = False) -> None:
        self._enabled = False
        self._q: "queue.Queue[Optional[_Utterance]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._engine_ready = threading.Event()
        self._using_openvoice = False

        if disabled:
            return

        # Sanity-check that *some* audio backend is importable.
        has_pygame = self._try_import("pygame")
        has_pyttsx3 = self._try_import("pyttsx3")
        if not has_pygame and not has_pyttsx3:
            print(
                "  [TTS] disabled — install pygame or pyttsx3 for audio output."
            )
            return

        self._enabled = True
        self._thread = threading.Thread(
            target=self._run, name="tts-worker", daemon=True
        )
        self._thread.start()

    # --------------------------- public API ---------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def using_openvoice(self) -> bool:
        return self._using_openvoice

    def warmup_async(self) -> None:
        """No-op: the worker thread already loads the engine eagerly."""
        # Kept for symmetry with the previous wiring; the worker thread
        # starts loading OpenVoice as soon as it spins up.

    def say(
        self,
        text: str,
        *,
        emotion: str = "neutral",
        intensity: float = 0.5,
        on_done: Optional[Callable[[], None]] = None,
    ) -> None:
        if self._enabled and text:
            self._q.put(_Utterance(text, emotion, float(intensity), on_done))
        elif on_done is not None:
            on_done()

    def shutdown(self) -> None:
        if self._thread is not None:
            self._q.put(None)
            self._thread.join(timeout=5.0)
            self._thread = None

    # ----------------------- worker-thread internals ------------------------
    @staticmethod
    def _try_import(name: str) -> bool:
        try:
            __import__(name)
            return True
        except ImportError:
            return False

    def _run(self) -> None:
        engine = self._try_load_openvoice()
        pygame_ref = None
        if engine is not None:
            try:
                import pygame

                pygame.mixer.init(
                    frequency=int(settings.openvoice_output_sample_rate),
                    size=-16,
                    channels=1,
                    buffer=2048,
                )
                pygame_ref = pygame
                self._using_openvoice = True
                print("  [TTS] OpenVoice loaded in-process (cloned voice supported).")
            except Exception as exc:
                print(f"  [TTS] pygame init failed ({exc}); using OS fallback.")
                engine = None

        self._engine_ready.set()

        if engine is not None:
            self._run_openvoice_loop(engine, pygame_ref)
        else:
            self._run_fallback_loop()

    @staticmethod
    def _try_load_openvoice():
        try:
            from app.tts.openvoice_client import OpenVoiceTTS

            engine = OpenVoiceTTS()
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(engine.warmup())
            finally:
                loop.close()
            if not engine.is_ready:
                return None
            return engine
        except Exception as exc:
            print(f"  [TTS] OpenVoice init failed: {exc}")
            return None

    def _run_openvoice_loop(self, engine, pygame_ref) -> None:
        from app.core.state import get_state

        while True:
            item = self._q.get()
            if item is None:
                return
            if not item.text:
                self._fire_done(item.on_done)
                continue
            try:
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
                finally:
                    loop.close()

                if not pcm:
                    self._fire_done(item.on_done)
                    continue

                wav_buf = io.BytesIO()
                with _wave.open(wav_buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(settings.openvoice_output_sample_rate)
                    wf.writeframes(pcm)
                wav_buf.seek(0)

                sound = pygame_ref.mixer.Sound(wav_buf)
                channel = sound.play()
                while channel and channel.get_busy():
                    time.sleep(0.05)
            except Exception as exc:
                print(f"  [warn] TTS playback failed: {exc}")
            finally:
                self._fire_done(item.on_done)

    def _run_fallback_loop(self) -> None:
        # SAPI on Windows, pyttsx3 elsewhere.
        if sys.platform == "win32":
            speak = self._init_sapi()
            if speak is not None:
                self._run_simple_loop(speak)
                return

        try:
            import pyttsx3
        except ImportError:
            print("  [TTS] no fallback available; audio disabled.")
            self._enabled = False
            return

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", engine.getProperty("rate") - 30)
            print("  [TTS] pyttsx3 ready (OS-native voice, no cloning).")
        except Exception as exc:
            print(f"  [TTS] pyttsx3 init failed: {exc}")
            self._enabled = False
            return

        def speak(text: str) -> None:
            engine.say(text)
            engine.runAndWait()

        self._run_simple_loop(speak)

    def _init_sapi(self) -> Optional[Callable[[str], None]]:
        try:
            import pythoncom
            import win32com.client
        except ImportError:
            return None

        try:
            pythoncom.CoInitialize()
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            print("  [TTS] Windows SAPI ready (OS-native voice, no cloning).")
        except Exception as exc:
            print(f"  [TTS] SAPI init failed: {exc}")
            return None

        def speak(text: str) -> None:
            try:
                voice.Speak(text)
            except Exception as exc:
                print(f"  [warn] SAPI: {exc}")

        return speak

    def _run_simple_loop(self, speak: Callable[[str], None]) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            if item.text:
                try:
                    speak(item.text)
                except Exception as exc:
                    print(f"  [warn] TTS speak failed: {exc}")
            self._fire_done(item.on_done)

    @staticmethod
    def _fire_done(cb: Optional[Callable[[], None]]) -> None:
        if cb is not None:
            try:
                cb()
            except Exception:
                pass
