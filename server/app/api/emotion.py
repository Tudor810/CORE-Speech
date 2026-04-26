"""Emotion / tone preset endpoints."""

from __future__ import annotations

from typing import get_args

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.state import EmotionPreset, get_state

router = APIRouter(prefix="/api/emotion", tags=["emotion"])

_VALID_PRESETS: tuple[str, ...] = get_args(EmotionPreset)


class EmotionPayload(BaseModel):
    preset: str = Field(..., description="One of the supported emotion presets")
    intensity: float = Field(0.5, ge=0.0, le=1.0)


@router.get("")
async def get_emotion() -> dict:
    state = get_state()
    return {
        "preset": state.data.emotion_preset,
        "intensity": state.data.emotion_intensity,
        "available": list(_VALID_PRESETS),
    }


@router.put("")
async def set_emotion(payload: EmotionPayload) -> dict:
    if payload.preset not in _VALID_PRESETS:
        raise HTTPException(400, f"preset must be one of {_VALID_PRESETS}")
    state = get_state()
    state.data.emotion_preset = payload.preset  # type: ignore[assignment]
    state.data.emotion_intensity = payload.intensity
    await state.save()
    return {"preset": payload.preset, "intensity": payload.intensity}
