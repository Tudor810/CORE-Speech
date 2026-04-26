"""In-memory + JSON-backed app state (active voice, emotion, training session).

A single `AppState` instance is attached to the FastAPI app at startup. Mutations
are persisted to disk on each write; loads happen only at boot. This keeps the
companion-app config sticky across server restarts without pulling in a DB.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from app.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

EmotionPreset = Literal["neutral", "friendly", "serious", "urgent", "calm", "excited"]


@dataclass
class VoiceProfile:
    voice_id: str
    name: str
    is_default: bool = False


@dataclass
class TrainingSession:
    label: str | None = None
    active: bool = False
    samples_collected: int = 0


@dataclass
class AppStateData:
    active_voice_id: str = settings.openvoice_default_voice_id
    # MeloTTS base speaker preset. This affects the *base* voice before any
    # OpenVoice tone conversion (voice cloning) is applied.
    tts_speaker_key: str = settings.openvoice_speaker_key
    custom_voices: list[VoiceProfile] = field(default_factory=list)
    emotion_preset: EmotionPreset = "neutral"
    emotion_intensity: float = 0.5  # 0..1
    training: TrainingSession = field(default_factory=TrainingSession)


class AppState:
    """Thread-safe wrapper around AppStateData with JSON persistence."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        self.data: AppStateData = self._load()

    def _load(self) -> AppStateData:
        if not self._path.exists():
            return AppStateData()
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            voices = [VoiceProfile(**v) for v in raw.get("custom_voices", [])]
            training = TrainingSession(**raw.get("training", {}))
            return AppStateData(
                active_voice_id=raw.get(
                    "active_voice_id", settings.openvoice_default_voice_id
                ),
                tts_speaker_key=raw.get("tts_speaker_key", settings.openvoice_speaker_key),
                custom_voices=voices,
                emotion_preset=raw.get("emotion_preset", "neutral"),
                emotion_intensity=raw.get("emotion_intensity", 0.5),
                training=training,
            )
        except Exception as exc:
            log.warning("Failed to load app state, starting fresh: %s", exc)
            return AppStateData()

    async def save(self) -> None:
        async with self._lock:
            payload = {
                "active_voice_id": self.data.active_voice_id,
                "tts_speaker_key": self.data.tts_speaker_key,
                "custom_voices": [asdict(v) for v in self.data.custom_voices],
                "emotion_preset": self.data.emotion_preset,
                "emotion_intensity": self.data.emotion_intensity,
                "training": asdict(self.data.training),
            }
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self._path)


_state: AppState | None = None


def get_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState(settings.state_file)
    return _state
