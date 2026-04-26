"""Voice cloning + profile selection endpoints (OpenVoice v2)."""

from __future__ import annotations

import asyncio
import io
import wave
from dataclasses import asdict

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from app.config import settings
from app.core.logging import get_logger
from app.core.state import VoiceProfile, get_state
from app.ws.manager import reception_manager

log = get_logger(__name__)
router = APIRouter(prefix="/api/voice", tags=["voice"])

_MAX_SAMPLE_SECONDS = 6
_MAX_SAMPLE_BYTES = 6 * 1024 * 1024  # 6 MB hard cap (~6s at 1 MB/s)


@router.get("/profiles")
async def list_profiles() -> dict:
    state = get_state()
    return {
        "active_voice_id": state.data.active_voice_id,
        "default_voice_id": settings.openvoice_default_voice_id,
        "active_speaker_key": state.data.tts_speaker_key,
        "tts_language": settings.openvoice_language,
        "custom_voices": [asdict(v) for v in state.data.custom_voices],
    }


@router.put("/profiles/active")
async def set_active(payload: dict) -> dict:
    voice_id = (payload.get("voice_id") or "").strip()
    if not voice_id:
        raise HTTPException(400, "voice_id is required")

    # Validate against the known voices: ``default`` and any user-cloned
    # ``voice_<hex>`` profile.  Unknown IDs would silently fall back to
    # ``default`` at synthesis time, which is confusing in the UI.
    state = get_state()
    known: set[str] = {settings.openvoice_default_voice_id}
    known.update(v.voice_id for v in state.data.custom_voices)
    if voice_id not in known:
        raise HTTPException(
            422,
            f"Unknown voice_id '{voice_id}'. Known: {sorted(known)[:20]}",
        )

    state.data.active_voice_id = voice_id
    await state.save()
    return {"active_voice_id": voice_id}


@router.get("/presets")
async def list_presets(request: Request) -> dict:
    """List available MeloTTS speaker presets for the configured language."""
    state = get_state()
    tts = getattr(request.app.state, "tts", None)
    speakers: list[str]
    if tts is None:
        speakers = [settings.openvoice_speaker_key]
    else:
        speakers = await tts.list_speakers()
    return {
        "tts_language": settings.openvoice_language,
        "active_speaker_key": state.data.tts_speaker_key,
        "speakers": speakers,
    }


@router.put("/presets/active")
async def set_active_preset(request: Request, payload: dict) -> dict:
    """Set the active MeloTTS speaker preset used by the server TTS."""
    speaker_key = (payload.get("speaker_key") or "").strip()
    if not speaker_key:
        raise HTTPException(400, "speaker_key is required")

    tts = getattr(request.app.state, "tts", None)
    if tts is None:
        raise HTTPException(503, "TTS engine not initialized")

    # Validate against the full set of speakers we expose (primary language
    # plus every other MeloTTS language we know how to lazy-load).
    speakers = await tts.list_speakers()
    if speaker_key not in speakers:
        raise HTTPException(
            422,
            f"Unknown speaker_key '{speaker_key}'. Available: {speakers[:50]}",
        )

    state = get_state()
    state.data.tts_speaker_key = speaker_key
    await state.save()

    # Best-effort: pre-warm the language MeloTTS in the background so the
    # very next synthesis doesn't pay the lazy model-load cost.  We
    # deliberately don't await it -- the user gets an immediate "saved"
    # response and the load happens while they're typing/signing.
    warm = getattr(tts, "warm_speaker", None)
    if callable(warm):
        asyncio.create_task(warm(speaker_key))

    return {"active_speaker_key": speaker_key}


def _pcm16_mono_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


@router.post("/synthesize")
async def synthesize(request: Request, payload: dict) -> Response:
    """Synthesise text with OpenVoice; uses app state (active voice) unless overridden.

    Request JSON:
      * ``text`` (required) — sentence to speak
      * ``voice_id`` — optional; default ``active_voice_id``
      * ``speaker_key`` — optional; default active MeloTTS preset
      * ``emotion`` — optional; default from app state
      * ``intensity`` or ``emotion_intensity`` — optional; 0..1, default from state

    Returns ``audio/wav`` (16-bit mono PCM at ``openvoice_output_sample_rate``).
    """
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "text is required")

    tts = getattr(request.app.state, "tts", None)
    if tts is None:
        raise HTTPException(503, "TTS engine not initialized")

    state = get_state()
    voice = (payload.get("voice_id") or "").strip() or state.data.active_voice_id
    speaker = (payload.get("speaker_key") or "").strip() or state.data.tts_speaker_key
    emo = (payload.get("emotion") or state.data.emotion_preset or "neutral").strip()
    raw_int = payload.get("intensity", payload.get("emotion_intensity"))
    if raw_int is None:
        intensity = float(state.data.emotion_intensity)
    else:
        try:
            intensity = float(raw_int)
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, "intensity must be a number") from exc
    intensity = max(0.0, min(1.0, intensity))

    try:
        pcm = await tts.synthesize_to_pcm(
            text,
            voice_id=voice,
            speaker_key=speaker,
            emotion=emo,
            intensity=intensity,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Synthesis failed")
        raise HTTPException(500, f"synthesis failed: {exc}") from exc

    wav = _pcm16_mono_to_wav(pcm, settings.openvoice_output_sample_rate)
    return Response(content=wav, media_type="audio/wav")


@router.post("/speak")
async def speak(request: Request, payload: dict) -> dict:
    """Synthesize text and stream it to an already-connected device on /ws/reception/{device_id}.

    Request JSON:
      * ``device_id`` (required) — target WS device id (ESP32)
      * ``text`` (required) — sentence to speak
      * ``voice_id`` / ``speaker_key`` / ``emotion`` / ``intensity`` — same as /synthesize

    Response JSON:
      * ``ok`` — True if the request was accepted and queued for streaming
      * ``chunks`` — number of binary frames sent
    """
    device_id = (payload.get("device_id") or "").strip()
    if not device_id:
        raise HTTPException(400, "device_id is required")
    if not reception_manager.is_connected(device_id):
        raise HTTPException(
            409,
            f"device '{device_id}' is not connected on /ws/reception/{device_id}",
        )

    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "text is required")

    tts = getattr(request.app.state, "tts", None)
    if tts is None:
        raise HTTPException(503, "TTS engine not initialized")

    state = get_state()
    voice = (payload.get("voice_id") or "").strip() or state.data.active_voice_id
    speaker = (payload.get("speaker_key") or "").strip() or state.data.tts_speaker_key
    emo = (payload.get("emotion") or state.data.emotion_preset or "neutral").strip()
    raw_int = payload.get("intensity", payload.get("emotion_intensity"))
    if raw_int is None:
        intensity = float(state.data.emotion_intensity)
    else:
        try:
            intensity = float(raw_int)
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, "intensity must be a number") from exc
    intensity = max(0.0, min(1.0, intensity))

    try:
        pcm = await tts.synthesize_to_pcm(
            text,
            voice_id=voice,
            speaker_key=speaker,
            emotion=emo,
            intensity=intensity,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Synthesis failed")
        raise HTTPException(500, f"synthesis failed: {exc}") from exc

    chunk_count = 0
    # Match OpenVoice stream framing (4 KB) so the client side stays low-latency.
    for off in range(0, len(pcm), 4096):
        chunk = pcm[off : off + 4096]
        if not chunk:
            continue
        chunk_count += 1
        await reception_manager.send_bytes(device_id, chunk)
    await reception_manager.send_json(device_id, {"type": "audio_end", "chunks": chunk_count})

    return {"ok": True, "device_id": device_id, "chunks": chunk_count}


@router.post("/clone")
async def clone(
    request: Request,
    name: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    name = name.strip()
    if not name:
        raise HTTPException(400, "name is required")
    audio = await file.read()
    if not audio:
        raise HTTPException(400, "empty audio sample")
    if len(audio) > _MAX_SAMPLE_BYTES:
        raise HTTPException(
            413,
            f"sample too large; expected <= {_MAX_SAMPLE_SECONDS}s",
        )

    tts = getattr(request.app.state, "tts", None)
    if tts is None:
        raise HTTPException(503, "TTS engine not initialized")

    try:
        voice_id = await tts.clone_voice(
            name, audio, filename=file.filename or "sample.wav"
        )
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Voice clone failed")
        raise HTTPException(500, f"clone failed: {exc}") from exc

    state = get_state()
    profile = VoiceProfile(voice_id=voice_id, name=name, is_default=False)
    state.data.custom_voices.append(profile)
    state.data.active_voice_id = voice_id
    await state.save()

    return {"voice_id": voice_id, "name": name}
