"""FastAPI entry point.

Boots the Edge server, mounts REST + WebSocket routers, and manages the
lifecycle of long-lived components (camera, MediaPipe, LSTM, Vosk STT).

Run with:  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import emotion, health, training, voice
from app.config import settings
from app.core.logging import get_logger, setup_logging
from app.core.state import get_state
from app.ml.classifier import GestureClassifier
from app.stt.vosk_client import VoskClient
from app.tts.openvoice_client import OpenVoiceTTS
from app.vision.capture import CameraStream
from app.vision.landmarks import HandLandmarker
from app.ws import (
    emission_ws,
    reception_browser_ws,
    reception_ws,
    training_ws,
    transcripts_ws,
)

setup_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize heavy components on startup; release them on shutdown."""
    log.info("Starting %s (env=%s)", settings.app_name, settings.environment)

    # Persistent state (voice/emotion/training) loaded from disk.
    app.state.app_state = get_state()

    # Vision: camera stream + MediaPipe Hands. Both lazy-start on first subscriber.
    app.state.camera = CameraStream(
        index=settings.camera_index,
        width=settings.camera_width,
        height=settings.camera_height,
        fps=settings.camera_fps,
    )
    app.state.landmarker = HandLandmarker()

    # Sign-language classifier (LSTM + sliding window).
    app.state.classifier = GestureClassifier()
    app.state.classifier.load()

    # Speech-to-text (lazy: Vosk model loads on first transcription).
    app.state.stt = VoskClient()

    # Text-to-speech: OpenVoice v2 + MeloTTS. Warm up so the first synthesis
    # doesn't pay the model-load cost; failure is non-fatal (the adapter
    # falls back to silent audio so the WS path is still observable).
    app.state.tts = OpenVoiceTTS()
    await app.state.tts.warmup()

    log.info("Server ready on %s:%s", settings.host, settings.port)
    try:
        yield
    finally:
        log.info("Shutting down...")
        await app.state.camera.aclose()
        app.state.landmarker.close()
        await app.state.tts.aclose()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Edge server for Deaf/Mute AR Glasses bidirectional communication.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(voice.router)
app.include_router(emotion.router)
app.include_router(training.router)

app.include_router(reception_ws.router)
app.include_router(reception_browser_ws.router)
app.include_router(emission_ws.router)
app.include_router(training_ws.router)
app.include_router(transcripts_ws.router)


@app.get("/")
async def root() -> dict:
    return {"service": settings.app_name, "docs": "/docs", "health": "/api/health"}
