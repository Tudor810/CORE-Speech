"""Centralized runtime configuration loaded from environment + .env file.

All API keys, model paths, and tunable parameters live here so other modules
never read os.environ directly. Use `from app.config import settings`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVER_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_SERVER_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- General ---
    app_name: str = "AR Glasses Sign-Speech Bridge"
    environment: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"

    # --- HTTP / CORS ---
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    # --- Camera / Vision ---
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    mediapipe_max_hands: int = 2
    mediapipe_min_detection_confidence: float = 0.5
    mediapipe_min_presence_confidence: float = 0.5
    mediapipe_min_tracking_confidence: float = 0.5
    # Path to the Tasks-API HandLandmarker model file. The legacy
    # ``mp.solutions.hands`` API is no longer shipped on Python 3.13+, so we
    # always go through the Tasks API even at runtime; the same .task file
    # used by ``scripts/extract_dataset.py`` and ``scripts/live_predict.py``.
    mediapipe_model_path: Path = _SERVER_ROOT / "models" / "hand_landmarker.task"

    # --- LSTM classifier ---
    lstm_window_size: int = 30
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_input_size: int = 63  # 21 landmarks * (x, y, z) for one dominant hand
    lstm_confidence_threshold: float = 0.75
    lstm_checkpoint_path: Path = _SERVER_ROOT / "models" / "sign_lstm.pt"
    lstm_labels_path: Path = _SERVER_ROOT / "models" / "labels.json"

    # --- LLM (local, via Ollama) ---
    # Ollama auto-detects CUDA / Metal / ROCm at launch and offloads as
    # many model layers as fit on the GPU; ``ollama_num_gpu`` lets you
    # override that (-1 = auto / use GPU if available, 0 = CPU only).
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_temperature: float = 0.4
    # Headroom for one sentence of output; in practice the model emits
    # ~30-80 tokens for a sign-gloss rewrite.
    ollama_max_tokens: int = 512
    ollama_num_gpu: int = -1
    # Duration string (e.g. "5m", "1h") — how long Ollama keeps the
    # model resident in RAM/VRAM after the last request.
    ollama_keep_alive: str = "5m"
    # Total per-request timeout (seconds). Cold-load on CPU can take a
    # while on the very first call.
    ollama_request_timeout: float = 60.0

    # --- TTS (OpenVoice v2 + MeloTTS, fully local) ---
    # Path to the directory that contains the v2 checkpoints, i.e. the folder
    # extracted from `checkpoints_v2_0417.zip` (https://github.com/myshell-ai/OpenVoice).
    # Layout expected:
    #   <openvoice_checkpoints>/converter/{config.json,checkpoint.pth}
    #   <openvoice_checkpoints>/base_speakers/ses/<speaker>.pth
    openvoice_checkpoints: Path = _SERVER_ROOT / "models" / "openvoice_v2"
    openvoice_language: str = "EN"  # MeloTTS language: EN, ES, FR, ZH, JP, KR
    openvoice_speaker_key: str = "EN-Default"  # MeloTTS speaker id within language
    openvoice_source_se_name: str = "en-default"  # base_speakers/ses/<name>.pth
    openvoice_device: Literal["cpu", "cuda", "auto"] = "auto"
    openvoice_output_sample_rate: int = 16000  # resample target for the ESP32
    openvoice_default_voice_id: str = "default"  # special id = no tone-conversion
    openvoice_voices_dir: Path = _SERVER_ROOT / "data" / "voices"

    # --- STT (Vosk, fully local) ---
    # Wire format the pipeline accumulates before forwarding a chunk to the
    # recogniser. These are backend-neutral; if/when we swap Vosk for another
    # local engine these still apply.
    stt_sample_rate: int = 16000
    # Rolling window forwarded to STT. 0.8 s is a good push-to-talk default --
    # short enough that brief utterances don't sit in the buffer waiting for a
    # full window, long enough that Vosk has real context to chew on. The
    # bridge also sends ``{"action":"flush"}`` on button release, so any
    # trailing fragment shorter than this still gets transcribed.
    stt_chunk_seconds: float = 0.8

    # Pick exactly one of:
    #   * VOSK_MODEL_PATH -- absolute or repo-relative folder containing an
    #     unpacked model from https://alphacephei.com/vosk/models.
    #   * VOSK_MODEL_NAME -- model id Vosk will auto-download into ~/.cache/vosk
    #     on first use (e.g. "vosk-model-small-en-us-0.15", "vosk-model-en-us-0.22").
    # If both are set, VOSK_MODEL_PATH wins.
    vosk_model_path: str = ""
    vosk_model_name: str = "vosk-model-small-en-us-0.15"
    # BCP-47 hint included in transcript JSON; Vosk itself is single-language
    # per model so this is metadata only (e.g. "en", "es", "ru").
    vosk_language: str = "en"

    # --- Storage paths ---
    models_dir: Path = _SERVER_ROOT / "models"
    data_dir: Path = _SERVER_ROOT / "data"
    samples_dir: Path = _SERVER_ROOT / "data" / "samples"
    recordings_dir: Path = _SERVER_ROOT / "data" / "recordings"
    state_file: Path = _SERVER_ROOT / "data" / "app_state.json"

    def ensure_dirs(self) -> None:
        for path in (
            self.models_dir,
            self.data_dir,
            self.samples_dir,
            self.recordings_dir,
            self.openvoice_voices_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


settings = get_settings()
