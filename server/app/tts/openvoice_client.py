"""OpenVoice v2 + MeloTTS local TTS adapter.

Public surface (kept identical to the previous ElevenLabs adapter so callers
do not change):

    OpenVoiceTTS.synthesize_stream(text_iter, voice_id, emotion, intensity)
        -> AsyncIterator[bytes]   (16-bit little-endian PCM @ output_sample_rate)

    OpenVoiceTTS.clone_voice(name, audio_bytes, filename) -> voice_id

Two-stage pipeline per text chunk:
  1. **MeloTTS** synthesises the text in the configured language/speaker.
  2. **OpenVoice ToneColorConverter** swaps the timbre to match the target
     speaker embedding extracted from the user's reference audio.

If the requested voice is `default` (or no checkpoints are present), step 2
is skipped and we just stream the MeloTTS output (resampled to 16 kHz PCM).

OpenVoice's API works with on-disk wav files; we use temp files inside the
event loop's worker thread so the main loop never blocks on I/O.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import tempfile
import types
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import numpy as np

from app.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

# Speed mapping: combine emotion preset with intensity (0..1).
# `delta` is added to base 1.0 then scaled by intensity.
_EMOTION_SPEED: dict[str, float] = {
    "neutral": 0.00,
    "calm": -0.10,
    "friendly": 0.05,
    "excited": 0.15,
    "serious": -0.05,
    "urgent": 0.25,
}


def _resolve_speed(emotion: str, intensity: float) -> float:
    delta = _EMOTION_SPEED.get(emotion, 0.0)
    intensity = max(0.0, min(1.0, intensity))
    return float(max(0.7, min(1.4, 1.0 + delta * (0.5 + intensity))))


def _se_filename_for(speaker_key: str) -> str:
    """Map a MeloTTS speaker_key to its OpenVoice SE filename (no extension).

    OpenVoice v2 checkpoints ship the per-speaker tone embeddings as
    ``base_speakers/ses/<name>.pth``. The filenames are lowercase with hyphens
    while MeloTTS uses a mix of hyphens and underscores in ``spk2id`` keys,
    e.g. ``EN_INDIA`` <-> ``en-india.pth``.  Normalising both directions keeps
    voice cloning faithful to the chosen base accent.
    """
    return speaker_key.replace("_", "-").lower()


# All MeloTTS speakers we know how to advertise to the client, grouped by
# language code.  Only the configured primary language is loaded at startup
# (see ``_load_blocking``); the other languages are loaded lazily the first
# time a speaker from them is requested.  This keeps cold-start cheap while
# letting the UI offer the full range of MeloTTS predefined voices.
#
# JP is intentionally listed but only exposed when ``openvoice_language == "JP"``
# because the rest of the time we install a stub for ``melo.text.japanese`` to
# avoid MeCab/pyopenjtalk DLL issues on Windows (see
# ``_install_melo_japanese_shim_for_non_jp``).
_MELO_SPEAKERS_BY_LANGUAGE: dict[str, tuple[str, ...]] = {
    "EN": ("EN-Default", "EN-US", "EN-BR", "EN_INDIA", "EN-AU", "EN-NEWEST"),
    "ES": ("ES",),
    "FR": ("FR",),
    "ZH": ("ZH",),
    "KR": ("KR",),
    "JP": ("JP",),
}


def _language_for_speaker_key(speaker_key: str) -> str | None:
    """Return the MeloTTS language code that owns ``speaker_key`` (or None)."""
    for lang, speakers in _MELO_SPEAKERS_BY_LANGUAGE.items():
        if speaker_key in speakers:
            return lang
    return None


def _exposed_speaker_keys() -> list[str]:
    """All MeloTTS speaker keys safe to expose on this install.

    JP requires MeCab/pyopenjtalk which we stub out unless JP is the
    configured primary language, so we only advertise it in that case.
    """
    out: list[str] = []
    for lang, speakers in _MELO_SPEAKERS_BY_LANGUAGE.items():
        if lang == "JP" and settings.openvoice_language != "JP":
            continue
        out.extend(speakers)
    return out


def _install_melo_japanese_shim_for_non_jp() -> None:
    """Melo ``english.py`` does ``from .japanese import distribute_phone``.

    ``japanese.py`` imports MeCab at module level; on Windows the MeCab wheel
    often fails with ``DLL load failed``. For non-JP Melo languages we only need
    ``distribute_phone`` (pure Python). Register a minimal submodule before any
    ``melo`` import.

    When ``openvoice_language`` is ``JP``, skip so real MeCab can load (if
    installed and working).
    """
    if settings.openvoice_language == "JP":
        return
    key = "melo.text.japanese"
    existing = sys.modules.get(key)
    if existing is not None and getattr(existing, "g2p", None) is not None:
        return  # full japanese already imported

    def distribute_phone(n_phone: int, n_word: int) -> list[int]:
        phones_per_word = [0] * n_word
        for _ in range(n_phone):
            min_tasks = min(phones_per_word)
            min_index = phones_per_word.index(min_tasks)
            phones_per_word[min_index] += 1
        return phones_per_word

    mod = types.ModuleType(key)
    mod.distribute_phone = distribute_phone
    sys.modules[key] = mod


def _install_lazy_melo_cleaner() -> None:
    """Avoid eager imports of JP/KR/FR/... in upstream ``melo.text.cleaner``."""
    import app.tts.melo_cleaner_lazy as melo_cleaner_lazy

    sys.modules["melo.text.cleaner"] = melo_cleaner_lazy


# Maps Melo language codes to the submodule that exposes ``get_bert_feature``.
_BERT_MODULE_FOR_LANGUAGE: dict[str, str] = {
    "ZH": "melo.text.chinese_bert",
    "EN": "melo.text.english_bert",
    "JP": "melo.text.japanese_bert",
    "ZH_MIX_EN": "melo.text.chinese_mix",
    "FR": "melo.text.french_bert",
    "SP": "melo.text.spanish_bert",
    "ES": "melo.text.spanish_bert",
    "KR": "melo.text.korean",
}


def _install_lazy_melo_get_bert() -> None:
    """Replace ``melo.text.get_bert`` with a lazy per-language dispatcher.

    Upstream ``melo.text.__init__.get_bert`` does::

        from .chinese_bert import get_bert_feature as zh_bert
        from .english_bert import get_bert_feature as en_bert
        from .japanese_bert import get_bert_feature as jp_bert    # MeCab / pyopenjtalk
        from .chinese_mix import get_bert_feature as zh_mix_en_bert
        from .spanish_bert import get_bert_feature as sp_bert
        from .french_bert import get_bert_feature as fr_bert
        from .korean import get_bert_feature as kr_bert            # jamo

    on every call, even when synthesising plain English. Each of those drags
    in heavy / OS-specific deps (``jamo``, ``pyopenjtalk``, MeCab, ...). We
    swap in a dispatcher that imports only the language we actually use, so
    English-only users don't need any other language's runtime stack.

    Must run AFTER ``melo.text`` itself is importable but BEFORE
    ``melo.api`` / ``melo.utils`` does ``from .text import get_bert`` --
    otherwise ``melo.utils`` would capture the original eager version.
    """
    import importlib

    melo_text = importlib.import_module("melo.text")

    def _lazy_get_bert(norm_text, word2ph, language, device):
        mod_name = _BERT_MODULE_FOR_LANGUAGE.get(language)
        if not mod_name:
            raise KeyError(
                f"No bert extractor configured for Melo language {language!r}"
            )
        mod = importlib.import_module(mod_name)
        return mod.get_bert_feature(norm_text, word2ph, device)

    melo_text.get_bert = _lazy_get_bert


# NLTK resources required by MeloTTS English G2P (``g2p_en``):
#   - ``averaged_perceptron_tagger_eng`` for POS tagging (NLTK >= 3.9 layout)
#   - ``averaged_perceptron_tagger``     legacy alias used by older NLTK
#   - ``cmudict``                        CMU pronouncing dictionary
# We probe each one and only download missing pieces, so warmup stays fast on
# subsequent runs and works offline once installed.
_NLTK_REQUIREMENTS: tuple[tuple[str, str], ...] = (
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("corpora/cmudict", "cmudict"),
)


def _ensure_nltk_data() -> None:
    """Download MeloTTS' English NLTK corpora once, on demand.

    Without these, ``melo.text.english.g2p`` raises ``LookupError: Resource
    'averaged_perceptron_tagger_eng' not found.`` on the very first synthesis
    and keeps failing for every subsequent call.
    """
    if settings.openvoice_language != "EN":
        return
    try:
        import nltk
    except ImportError:
        log.warning(
            "nltk not installed; MeloTTS English G2P will fail. "
            "Install it with `pip install nltk`."
        )
        return

    for resource_path, package in _NLTK_REQUIREMENTS:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            log.info("Downloading NLTK resource '%s'...", package)
            try:
                nltk.download(package, quiet=True)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Failed to download NLTK resource '%s': %s. "
                    "MeloTTS English G2P may fail.", package, exc,
                )


@dataclass
class _Models:
    tts: object  # melo.api.TTS
    converter: object  # openvoice.api.ToneColorConverter
    source_se: object  # torch.Tensor (default fallback SE)
    # Per-speaker source SE tensors keyed by Melo speaker_key.  Voice cloning
    # only works correctly when the source SE matches the actual base voice
    # rendered by MeloTTS — otherwise the tone-color conversion leaves a
    # noticeable accent mismatch (e.g. EN_INDIA base + en-default source SE
    # smears the Indian accent toward US/UK).
    speaker_se: dict[str, object]
    spk2id: dict[str, int]
    melo_sample_rate: int
    device: str


class OpenVoiceTTS:
    """Lazy-loaded local TTS + voice cloner."""

    def __init__(self) -> None:
        self._models: _Models | None = None
        self._init_lock = asyncio.Lock()
        self._infer_lock = asyncio.Lock()  # serialise GPU/CPU inference
        self._embedding_cache: dict[str, object] = {}  # voice_id -> torch.Tensor
        # Lazy-loaded MeloTTS instances for non-primary languages, keyed by
        # the MeloTTS language code (e.g. "ES", "FR").  Loaded on demand the
        # first time a speaker from that language is selected so startup
        # stays cheap.
        self._extra_tts: dict[str, object] = {}
        self._extra_tts_locks: dict[str, asyncio.Lock] = {}

    # ----------------------------- lifecycle -----------------------------

    async def warmup(self) -> None:
        """Load checkpoints up front so the first request is snappy."""
        try:
            await self._ensure_loaded()
        except Exception as exc:  # noqa: BLE001
            log.warning("OpenVoice warmup skipped: %s", exc)

    async def aclose(self) -> None:
        # MeloTTS / Converter don't expose explicit close; rely on GC.
        self._models = None
        self._embedding_cache.clear()
        self._extra_tts.clear()
        self._extra_tts_locks.clear()

    @property
    def is_ready(self) -> bool:
        return self._models is not None

    async def list_speakers(self) -> list[str]:
        """Return every MeloTTS speaker key the user can pick.

        Includes both the primary-language speakers (already loaded) and the
        speakers from all other supported MeloTTS languages, which are loaded
        lazily on first use.  Falling back to just the configured key keeps
        the client UI alive when checkpoints are missing.
        """
        try:
            await self._ensure_loaded()
        except Exception:  # noqa: BLE001
            return [settings.openvoice_speaker_key]
        return _exposed_speaker_keys()

    def resolve_speaker_id(self, speaker_key: str | None) -> int | None:
        """Map a speaker key to an id, returning None if invalid/unloaded."""
        if not speaker_key:
            return None
        if self._models is None:
            return None
        if speaker_key in self._models.spk2id:
            return self._models.spk2id[speaker_key]
        # Lazy-loaded language already in the cache?
        lang = _language_for_speaker_key(speaker_key)
        if lang is None:
            return None
        tts = self._extra_tts.get(lang)
        if tts is None:
            return None
        try:
            return tts.hps.data.spk2id.get(speaker_key)  # type: ignore[attr-defined]
        except AttributeError:
            return None

    async def warm_speaker(self, speaker_key: str) -> bool:
        """Best-effort: pre-load the language MeloTTS for ``speaker_key``.

        Used by the API right after the user picks a non-primary speaker so
        the actual first synthesis doesn't pay the model-download cost.
        Returns True iff the speaker is now resolvable.
        """
        try:
            resolved = await self._get_tts_for_speaker_key(speaker_key)
        except Exception as exc:  # noqa: BLE001
            log.warning("Speaker warm-up for '%s' failed: %s", speaker_key, exc)
            return False
        return resolved is not None

    async def _ensure_loaded(self) -> _Models:
        if self._models is not None:
            return self._models
        async with self._init_lock:
            if self._models is not None:
                return self._models
            self._models = await asyncio.to_thread(self._load_blocking)
            return self._models

    def _resolve_device(self) -> str:
        device = settings.openvoice_device
        if device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:  # noqa: BLE001
                return "cpu"
        return device

    def _load_blocking(self) -> _Models:
        # Imports kept local so the module can be imported even when the
        # heavy deps aren't installed yet.
        import torch

        _install_melo_japanese_shim_for_non_jp()
        _install_lazy_melo_cleaner()
        _install_lazy_melo_get_bert()
        _ensure_nltk_data()
        from melo.api import TTS
        from openvoice.api import ToneColorConverter

        ckpt_root = settings.openvoice_checkpoints
        converter_cfg = ckpt_root / "converter" / "config.json"
        converter_pth = ckpt_root / "converter" / "checkpoint.pth"
        ses_dir = ckpt_root / "base_speakers" / "ses"
        source_se_pth = ses_dir / f"{settings.openvoice_source_se_name}.pth"
        for required in (converter_cfg, converter_pth, source_se_pth):
            if not required.exists():
                raise FileNotFoundError(
                    f"OpenVoice checkpoint missing: {required}. "
                    f"Download `checkpoints_v2_0417.zip` from the OpenVoice "
                    f"repo and extract into {ckpt_root}."
                )

        device = self._resolve_device()
        log.info("Loading OpenVoice v2 (device=%s)", device)

        converter = ToneColorConverter(str(converter_cfg), device=device)
        converter.load_ckpt(str(converter_pth))

        tts = TTS(language=settings.openvoice_language, device=device)
        spk2id = tts.hps.data.spk2id  # type: ignore[attr-defined]
        if settings.openvoice_speaker_key not in spk2id:
            raise ValueError(
                f"Speaker '{settings.openvoice_speaker_key}' not in MeloTTS "
                f"language '{settings.openvoice_language}'. Available: {list(spk2id)}"
            )
        source_se = torch.load(source_se_pth, map_location=device)

        # Pre-load every per-speaker SE we can find.  The OpenVoice v2
        # checkpoints ship one SE per base voice (en-us.pth, en-br.pth,
        # en-india.pth, es.pth, fr.pth, zh.pth, kr.pth, ...).  We cache them
        # for ALL exposed speakers — not just the currently loaded primary
        # language — because non-primary languages are loaded lazily on first
        # synthesis and we don't want to do disk I/O on the audio thread.
        speaker_se: dict[str, object] = {}
        for spk in _exposed_speaker_keys():
            se_path = ses_dir / f"{_se_filename_for(spk)}.pth"
            if not se_path.exists():
                log.debug("OpenVoice SE missing for speaker %s (%s)", spk, se_path)
                continue
            try:
                speaker_se[spk] = torch.load(se_path, map_location=device)
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to load SE for %s: %s", spk, exc)

        melo_sr = int(tts.hps.data.sampling_rate)  # type: ignore[attr-defined]

        log.info(
            "OpenVoice ready (lang=%s, speaker=%s, ses=%d, melo_sr=%d, target_sr=%d)",
            settings.openvoice_language,
            settings.openvoice_speaker_key,
            len(speaker_se),
            melo_sr,
            settings.openvoice_output_sample_rate,
        )
        return _Models(
            tts=tts,
            converter=converter,
            source_se=source_se,
            speaker_se=speaker_se,
            spk2id=dict(spk2id),
            melo_sample_rate=melo_sr,
            device=device,
        )

    # --------------------------- voice cloning ---------------------------

    async def clone_voice(
        self,
        name: str,
        audio_bytes: bytes,
        filename: str = "sample.wav",
    ) -> str:
        """Save the reference audio + pre-extract its tone embedding.

        Returns a generated voice_id; the audio file lives at
        `<voices_dir>/<voice_id><ext>`. Pre-extracting the embedding here
        validates the audio and warms the cache for the first synthesis.
        """
        models = await self._ensure_loaded()

        ext = (Path(filename).suffix or ".wav").lower()
        # Browser MediaRecorder default is webm/opus; OpenVoice's se_extractor
        # uses ffmpeg under the hood and accepts those.
        if ext not in {".wav", ".mp3", ".webm", ".ogg", ".m4a", ".flac"}:
            ext = ".wav"

        voice_id = f"voice_{uuid.uuid4().hex[:12]}"
        target_path = settings.openvoice_voices_dir / f"{voice_id}{ext}"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target_path.write_bytes, audio_bytes)

        try:
            embedding = await asyncio.to_thread(
                self._extract_embedding_blocking, str(target_path), models
            )
        except Exception as exc:  # noqa: BLE001
            with contextlib.suppress(Exception):
                target_path.unlink()
            raise RuntimeError(f"voice extraction failed: {exc}") from exc

        self._embedding_cache[voice_id] = embedding
        log.info("Cloned voice '%s' -> %s (%s)", name, voice_id, target_path.name)
        return voice_id

    def _extract_embedding_blocking(self, audio_path: str, models: _Models):
        """Extract a speaker embedding from a reference audio file.

        We deliberately avoid ``openvoice.se_extractor.get_se`` because that
        path imports ``whisper_timestamped`` at module top-level — a heavy dep
        (and the only thing pulling Whisper into this server now that STT
        runs on Vosk) we don't want just for cloning a 6-second clip.

        Instead we trim silence with librosa and call the converter's
        ``extract_se`` directly. For short, user-recorded samples this matches
        what se_extractor does internally.
        """
        prepared = self._prepare_reference_wav(audio_path, models)
        try:
            return models.converter.extract_se(  # type: ignore[attr-defined]
                [str(prepared)]
            )
        finally:
            if str(prepared) != str(audio_path):
                with contextlib.suppress(Exception):
                    Path(prepared).unlink()

    def _prepare_reference_wav(self, audio_path: str, models: _Models) -> Path:
        """Decode the upload and write a clean mono wav at the converter's SR.

        Browsers sometimes send wav with non-standard sample rates; we
        normalise so ``ToneColorConverter`` always sees a well-formed file.
        """
        import librosa
        import soundfile as sf

        target_sr = int(getattr(models.converter, "hps").data.sampling_rate)  # type: ignore[attr-defined]
        audio, _sr = librosa.load(audio_path, sr=target_sr, mono=True)
        if audio.size == 0:
            raise RuntimeError("the recording is empty or could not be decoded")

        # Trim leading/trailing silence (≤ -40 dBFS). top_db is a generous
        # threshold so we don't accidentally chop a quiet voice.
        trimmed, _ = librosa.effects.trim(audio, top_db=40)
        if trimmed.size < int(0.5 * target_sr):
            trimmed = audio  # fall back to the full clip if trimming nuked it

        out_path = Path(audio_path).with_name(Path(audio_path).stem + "__ref.wav")
        sf.write(str(out_path), trimmed, target_sr, subtype="PCM_16")
        return out_path

    async def _get_embedding(self, voice_id: str) -> object | None:
        """Return cached embedding, re-extracting from disk if needed."""
        if voice_id in self._embedding_cache:
            return self._embedding_cache[voice_id]

        # Look for any file in voices_dir matching <voice_id>.*
        candidates = [
            p
            for p in settings.openvoice_voices_dir.glob(f"{voice_id}.*")
            if p.is_file()
        ]
        if not candidates:
            return None
        models = await self._ensure_loaded()
        embedding = await asyncio.to_thread(
            self._extract_embedding_blocking, str(candidates[0]), models
        )
        self._embedding_cache[voice_id] = embedding
        return embedding

    # --------------------------- multi-language --------------------------

    async def _get_tts_for_speaker_key(
        self, speaker_key: str
    ) -> tuple[object, int] | None:
        """Resolve a speaker key to ``(MeloTTS instance, speaker_id)``.

        Lazy-loads the MeloTTS model for the speaker's language on first
        request.  Returns ``None`` if the key is unknown or its language
        couldn't be loaded (e.g. missing language deps), so callers can fall
        back to the configured default speaker.
        """
        models = await self._ensure_loaded()

        # Fast path: speaker belongs to the primary (already-loaded) MeloTTS.
        if speaker_key in models.spk2id:
            return models.tts, models.spk2id[speaker_key]

        lang = _language_for_speaker_key(speaker_key)
        if lang is None or lang == settings.openvoice_language:
            return None

        # JP requires MeCab + pyopenjtalk, which we shim out unless JP is the
        # configured primary language; refuse to lazy-load it to avoid a
        # confusing import-time crash on Windows.
        if lang == "JP" and settings.openvoice_language != "JP":
            log.warning(
                "Refusing to lazy-load JP MeloTTS: requires MeCab/pyopenjtalk. "
                "Set OPENVOICE_LANGUAGE=JP at startup if you really want JP."
            )
            return None

        # Per-language lock so concurrent synthesis requests for the same
        # new language don't race on the heavy load.  Other languages and
        # the primary inference loop stay unblocked.
        lock = self._extra_tts_locks.setdefault(lang, asyncio.Lock())
        async with lock:
            tts = self._extra_tts.get(lang)
            if tts is None:
                try:
                    tts = await asyncio.to_thread(
                        self._load_extra_tts_blocking, lang, models.device
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "Failed to lazy-load MeloTTS language '%s' "
                        "(missing dep?): %s",
                        lang,
                        exc,
                    )
                    return None
                self._extra_tts[lang] = tts
                log.info(
                    "MeloTTS language '%s' loaded lazily for speaker '%s'",
                    lang,
                    speaker_key,
                )

        try:
            speaker_id = tts.hps.data.spk2id.get(speaker_key)  # type: ignore[attr-defined]
        except AttributeError:
            speaker_id = None
        if speaker_id is None:
            log.warning(
                "Speaker '%s' not present in lazy-loaded language '%s'.",
                speaker_key,
                lang,
            )
            return None
        return tts, int(speaker_id)

    @staticmethod
    def _load_extra_tts_blocking(language: str, device: str):
        """Load a non-primary MeloTTS language model on the worker thread."""
        from melo.api import TTS

        log.info("Loading MeloTTS language '%s' (device=%s)", language, device)
        return TTS(language=language, device=device)

    # ----------------------------- synthesis -----------------------------

    async def synthesize_stream(
        self,
        text_iter: AsyncIterator[str],
        voice_id: str | None = None,
        speaker_key: str | None = None,
        emotion: str = "neutral",
        intensity: float = 0.5,
    ) -> AsyncIterator[bytes]:
        """Stream PCM bytes generated from a streaming text source."""
        voice_id = voice_id or settings.openvoice_default_voice_id

        try:
            models = await self._ensure_loaded()
        except Exception as exc:  # noqa: BLE001
            log.warning("OpenVoice not loaded; emitting silence (%s).", exc)
            async for _ in text_iter:
                pass
            async for chunk in _silent_pcm():
                yield chunk
            return

        resolved_key = speaker_key or settings.openvoice_speaker_key
        resolved = await self._get_tts_for_speaker_key(resolved_key)
        if resolved is None:
            log.warning(
                "speaker '%s' not available; falling back to '%s'.",
                resolved_key,
                settings.openvoice_speaker_key,
            )
            resolved_key = settings.openvoice_speaker_key
            resolved = await self._get_tts_for_speaker_key(resolved_key)
            if resolved is None:
                log.error(
                    "Default speaker '%s' is also unavailable; aborting synthesis.",
                    resolved_key,
                )
                async for _ in text_iter:
                    pass
                return
        tts_instance, speaker_id = resolved

        target_se = None
        if voice_id != settings.openvoice_default_voice_id:
            target_se = await self._get_embedding(voice_id)
            if target_se is None:
                log.warning("voice_id '%s' not found; falling back to default.", voice_id)

        # Match the source SE to the actual base voice MeloTTS will render.
        # Without this, cloning a custom voice on top of a non-default speaker
        # (e.g. EN_INDIA) would always blend through `en-default.pth`, washing
        # out the chosen accent. Falls back to the configured default SE if a
        # matching per-speaker SE wasn't shipped with the checkpoints.
        source_se = models.speaker_se.get(resolved_key, models.source_se)

        speed = _resolve_speed(emotion, intensity)

        async for sentence in _coalesce_text(text_iter):
            try:
                pcm = await self._synthesize_sentence(
                    sentence,
                    tts_instance,
                    models,
                    speaker_id,
                    source_se,
                    target_se,
                    speed,
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("OpenVoice synthesis failed: %s", exc)
                continue
            # Stream out in 4 KB frames so the WS pump can interleave them.
            for off in range(0, len(pcm), 4096):
                yield pcm[off : off + 4096]

    async def synthesize_to_pcm(
        self,
        text: str,
        *,
        voice_id: str | None = None,
        speaker_key: str | None = None,
        emotion: str = "neutral",
        intensity: float = 0.5,
    ) -> bytes:
        """One-shot: full text → mono 16-bit PCM @ ``openvoice_output_sample_rate``."""

        t = (text or "").strip()
        if not t:
            return b""

        async def one_shot() -> AsyncIterator[str]:
            yield t

        chunks: list[bytes] = []
        async for chunk in self.synthesize_stream(
            one_shot(),
            voice_id=voice_id,
            speaker_key=speaker_key,
            emotion=emotion,
            intensity=intensity,
        ):
            chunks.append(chunk)
        return b"".join(chunks)

    async def _synthesize_sentence(
        self,
        text: str,
        tts_instance: object,
        models: _Models,
        speaker_id: int,
        source_se: object,
        target_se: object | None,
        speed: float,
    ) -> bytes:
        async with self._infer_lock:
            return await asyncio.to_thread(
                self._synthesize_blocking,
                text,
                tts_instance,
                models,
                speaker_id,
                source_se,
                target_se,
                speed,
            )

    def _synthesize_blocking(
        self,
        text: str,
        tts_instance: object,
        models: _Models,
        speaker_id: int,
        source_se: object,
        target_se: object | None,
        speed: float,
    ) -> bytes:
        with tempfile.TemporaryDirectory(prefix="openvoice_") as tmp:
            tmpdir = Path(tmp)
            src_path = tmpdir / "src.wav"
            out_path = tmpdir / "out.wav"

            # Step 1: MeloTTS base synthesis (using the language-specific
            # TTS instance — primary or lazy-loaded extra).
            tts_instance.tts_to_file(  # type: ignore[attr-defined]
                text,
                speaker_id,
                str(src_path),
                speed=speed,
            )

            final_path = src_path
            if target_se is not None:
                # Step 2: ToneColorConverter applies the cloned timbre on top
                # of the chosen base voice.  ``source_se`` MUST correspond to
                # the speaker that produced ``src_path`` for the conversion to
                # preserve the intended accent.
                models.converter.convert(  # type: ignore[attr-defined]
                    audio_src_path=str(src_path),
                    src_se=source_se,
                    tgt_se=target_se,
                    output_path=str(out_path),
                    message="@MyShell",
                )
                final_path = out_path

            return _wav_to_pcm16(
                final_path,
                target_sr=settings.openvoice_output_sample_rate,
            )


# --------------------------- helpers ---------------------------

async def _coalesce_text(
    text_iter: AsyncIterator[str], min_chars: int = 24
) -> AsyncIterator[str]:
    """Buffer LLM tokens into roughly sentence-sized chunks for TTS quality."""
    buf = ""
    async for piece in text_iter:
        buf += piece
        if len(buf) >= min_chars and any(buf.endswith(p) for p in ".!?,;\n"):
            yield buf.strip()
            buf = ""
    if buf.strip():
        yield buf.strip()


async def _silent_pcm(ms: int = 250, sample_rate: int = 16000) -> AsyncIterator[bytes]:
    """Emit a short period of silence so the WS path is exercised in dev mode."""
    n_samples = int(sample_rate * ms / 1000)
    yield b"\x00\x00" * n_samples
    await asyncio.sleep(0)


def _wav_to_pcm16(path: Path, target_sr: int) -> bytes:
    """Read a wav file and return mono int16 PCM bytes at `target_sr`."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        # 8-bit unsigned
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    if framerate != target_sr:
        audio = _resample(audio, framerate, target_sr)

    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16).tobytes()
    return pcm


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or audio.size == 0:
        return audio
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(src_sr, dst_sr)
        return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)
    except Exception:  # noqa: BLE001
        # Fallback: linear interpolation.
        n_out = int(round(audio.size * dst_sr / src_sr))
        x_old = np.linspace(0, 1, audio.size, endpoint=False)
        x_new = np.linspace(0, 1, n_out, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)


# Module-level singleton constructor for FastAPI lifespan.
_singleton: OpenVoiceTTS | None = None


def get_tts() -> OpenVoiceTTS:
    global _singleton
    if _singleton is None:
        _singleton = OpenVoiceTTS()
    return _singleton
