"""Vosk wrapper for chunked, low-latency, fully-local transcription.

Vosk (https://alphacephei.com/vosk/) is an Apache-2.0 streaming ASR engine
built on top of Kaldi. We use it as a drop-in replacement for the previous
``FasterWhisperClient``:

* The acoustic model is loaded lazily on first call so server boot stays fast.
* ``transcribe_chunk`` accepts raw PCM bytes (16-bit mono LE @ ``stt_sample_rate``)
  and returns the recognised text for that chunk.
* Inference runs inside ``asyncio.to_thread`` so the event loop is never blocked.

A single :class:`vosk.Model` is shared process-wide (cheap on RAM and thread-safe).
For each chunk we instantiate a fresh ``KaldiRecognizer`` -- its allocation is
trivial compared to the model load, and it neatly avoids cross-device
recogniser state from bleeding into one another.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class TranscriptionResult:
    text: str
    language: str | None
    duration: float


class VoskClient:
    """Local STT backed by Vosk / Kaldi.

    Two ways to point at a model (checked in order):

    1. ``VOSK_MODEL_PATH`` -- absolute or repo-relative directory containing an
       unpacked Vosk model (``am/``, ``conf/``, ``graph/`` ...).
    2. ``VOSK_MODEL_NAME`` -- model identifier from
       https://alphacephei.com/vosk/models which Vosk will auto-download into
       ``~/.cache/vosk`` on first use.

    Defaults to ``vosk-model-small-en-us-0.15`` (~40 MB, English) -- fast on CPU.
    Swap to ``vosk-model-en-us-0.22`` (~1.8 GB) for higher accuracy.
    """

    def __init__(self) -> None:
        self._model = None
        self._lock = asyncio.Lock()

    # ----------------------------- lifecycle -----------------------------

    async def _ensure_model(self) -> None:
        if self._model is not None:
            return
        async with self._lock:
            if self._model is not None:
                return
            self._model = await asyncio.to_thread(self._load_model)

    def _load_model(self):
        # Vosk is rather chatty on stderr; clamp it to warnings unless the
        # operator explicitly raised our log level to DEBUG.
        try:
            from vosk import Model, SetLogLevel
        except ImportError as exc:  # pragma: no cover -- surfaced to caller
            raise RuntimeError(
                "Vosk is not installed. Run `pip install vosk` (or "
                "`pip install -r requirements.txt`)."
            ) from exc

        SetLogLevel(0 if settings.log_level.upper() == "DEBUG" else -1)

        model_path = (settings.vosk_model_path or "").strip()
        model_name = (settings.vosk_model_name or "").strip()

        if model_path:
            resolved = Path(model_path)
            if not resolved.is_absolute():
                # Resolve relative paths against the server root so users can
                # write ``models/vosk-en`` in .env.
                resolved = (Path(__file__).resolve().parents[2] / resolved).resolve()
            if not resolved.exists():
                raise FileNotFoundError(
                    f"VOSK_MODEL_PATH points at '{resolved}', which does not exist. "
                    "Download a model from https://alphacephei.com/vosk/models, unzip "
                    "it into that folder, or set VOSK_MODEL_NAME to auto-download."
                )
            log.info("Loading Vosk model from %s", resolved)
            return Model(model_path=str(resolved))

        if not model_name:
            raise RuntimeError(
                "No Vosk model configured. Set VOSK_MODEL_PATH (local folder) "
                "or VOSK_MODEL_NAME (auto-download) in .env."
            )

        log.info(
            "Loading Vosk model '%s' (auto-download to ~/.cache/vosk on first run)",
            model_name,
        )
        return Model(model_name=model_name)

    # ----------------------------- helpers -------------------------------

    @staticmethod
    def _pcm_to_int16_bytes(pcm_bytes: bytes) -> bytes:
        """Vosk consumes 16-bit little-endian mono PCM directly.

        We accept the same wire format as the previous Whisper client (raw
        ``int16`` LE bytes) so callers in :mod:`app.pipelines.emission` and
        :mod:`scripts.serial_bridge` don't have to change. The only thing we
        do here is make sure the buffer is contiguous and has an even length.
        """
        if not pcm_bytes:
            return b""
        if len(pcm_bytes) % 2 != 0:
            # Drop a trailing odd byte rather than raising; the serial bridge
            # occasionally hands us a partial sample at end-of-window.
            pcm_bytes = pcm_bytes[:-1]
        return bytes(pcm_bytes)

    # ----------------------------- public API ----------------------------

    async def transcribe_chunk(self, pcm_bytes: bytes) -> Optional[TranscriptionResult]:
        await self._ensure_model()
        pcm = self._pcm_to_int16_bytes(pcm_bytes)
        if not pcm:
            return None
        return await asyncio.to_thread(self._transcribe_blocking, pcm)

    def _transcribe_blocking(self, pcm: bytes) -> TranscriptionResult:
        from vosk import KaldiRecognizer

        assert self._model is not None
        sample_rate = settings.stt_sample_rate

        # Fresh recogniser per chunk: KaldiRecognizer is *not* documented as
        # thread-safe, and we may have several /ws/emission devices running
        # concurrently. The allocation is essentially free vs. the model load.
        rec = KaldiRecognizer(self._model, float(sample_rate))
        rec.SetWords(False)

        rec.AcceptWaveform(pcm)
        try:
            payload: dict[str, Any] = json.loads(rec.FinalResult() or "{}")
        except json.JSONDecodeError:
            payload = {}

        text = (payload.get("text") or "").strip()
        # Best-effort duration: we know exactly how many samples we fed in.
        duration = (len(pcm) // 2) / float(sample_rate) if sample_rate else 0.0

        return TranscriptionResult(
            text=text,
            language=settings.vosk_language or None,
            duration=duration,
        )


# Re-export the result dataclass under a backend-neutral alias so future swaps
# (Parakeet, Moonshine, ...) stay transparent to the pipeline.
__all__ = ["VoskClient", "TranscriptionResult"]


# Silence Vosk's noisy ALSA/PortAudio probing on Linux servers without sound
# cards. Harmless on Windows.
os.environ.setdefault("VOSK_LOG_LEVEL", "-1")
