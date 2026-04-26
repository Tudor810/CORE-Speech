"""Voice management — presets, custom voices, cloning.

Wraps :class:`~app.tts.openvoice_client.OpenVoiceTTS` and the persisted
:class:`~app.core.state.AppState` so the GUI never has to touch
asyncio. All async calls are routed through
:func:`app.runtime.run_coro_blocking` / :func:`app.runtime.submit_coro`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# IMPORTANT: ``..runtime`` must come first — its module-level side-effect
# inserts ``server/`` onto ``sys.path`` so the ``app.*`` imports below
# resolve to the server package (not our companion package).
from ..runtime import app_settings, app_state, run_coro_blocking, submit_coro, tts_engine
from app.core.state import VoiceProfile  # noqa: E402  (runtime side-effect)


@dataclass(frozen=True)
class VoiceSnapshot:
    """Read-only snapshot of the current voice config."""

    active_voice_id: str
    default_voice_id: str
    active_speaker_key: str
    tts_language: str
    custom_voices: list[VoiceProfile]
    speakers: list[str]


def snapshot(*, refresh_speakers: bool = True) -> VoiceSnapshot:
    """Build a fresh :class:`VoiceSnapshot` from disk + the loaded TTS engine."""
    settings = app_settings()
    state = app_state().data

    speakers: list[str] = [settings.openvoice_speaker_key]
    if refresh_speakers:
        try:
            speakers = run_coro_blocking(
                tts_engine().list_speakers(), timeout=120.0
            )
        except Exception:
            pass

    return VoiceSnapshot(
        active_voice_id=state.active_voice_id,
        default_voice_id=settings.openvoice_default_voice_id,
        active_speaker_key=state.tts_speaker_key,
        tts_language=settings.openvoice_language,
        custom_voices=list(state.custom_voices),
        speakers=speakers,
    )


def set_active_voice(voice_id: str) -> None:
    """Persist a different ``active_voice_id`` (default or one of the clones)."""
    voice_id = (voice_id or "").strip()
    if not voice_id:
        raise ValueError("voice_id is required")
    state = app_state()
    state.data.active_voice_id = voice_id
    run_coro_blocking(state.save(), timeout=10.0)


def set_active_speaker(speaker_key: str) -> None:
    """Persist a different MeloTTS base speaker preset."""
    speaker_key = (speaker_key or "").strip()
    if not speaker_key:
        raise ValueError("speaker_key is required")

    speakers = run_coro_blocking(tts_engine().list_speakers(), timeout=120.0)
    if speaker_key not in speakers:
        raise ValueError(
            f"unknown speaker '{speaker_key}'. Available: {speakers[:10]}"
        )

    state = app_state()
    state.data.tts_speaker_key = speaker_key
    run_coro_blocking(state.save(), timeout=10.0)


def clone_voice_from_wav(
    name: str,
    wav_path: Path,
    *,
    on_done: Optional[Callable[[Optional[str], Optional[str]], None]] = None,
) -> None:
    """Run the OpenVoice clone pipeline on a recorded WAV.

    Returns immediately. ``on_done(voice_id, error)`` fires later on the
    runtime loop's thread (caller is responsible for hopping back to the
    UI thread, e.g. via :func:`app.runtime.call_in_ui`).
    """

    name = (name or "").strip()
    if not name:
        raise ValueError("name is required")
    if not wav_path.is_file():
        raise FileNotFoundError(f"recording not found: {wav_path}")

    audio_bytes = wav_path.read_bytes()
    if not audio_bytes:
        raise ValueError("recording is empty")

    async def _go() -> tuple[Optional[str], Optional[str]]:
        try:
            voice_id = await tts_engine().clone_voice(
                name, audio_bytes, filename=wav_path.name
            )
        except Exception as exc:
            return None, str(exc)

        state = app_state()
        profile = VoiceProfile(voice_id=voice_id, name=name, is_default=False)
        state.data.custom_voices.append(profile)
        state.data.active_voice_id = voice_id
        await state.save()
        return voice_id, None

    fut = submit_coro(_go())

    def _on_done(_fut) -> None:
        if on_done is None:
            return
        try:
            voice_id, err = fut.result()
        except Exception as exc:
            voice_id, err = None, str(exc)
        on_done(voice_id, err)

    fut.add_done_callback(_on_done)


def remove_custom_voice(voice_id: str) -> None:
    """Drop a cloned voice from the persisted profile list (file kept on disk)."""
    state = app_state()
    state.data.custom_voices = [
        v for v in state.data.custom_voices if v.voice_id != voice_id
    ]
    if state.data.active_voice_id == voice_id:
        state.data.active_voice_id = app_settings().openvoice_default_voice_id
    run_coro_blocking(state.save(), timeout=10.0)


def warmup_engine_async() -> None:
    """Kick off OpenVoice loading in the background (non-blocking)."""
    submit_coro(tts_engine().warmup())
