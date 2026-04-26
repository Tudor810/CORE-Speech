"""Liveness + readiness probe."""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.config import settings

router = APIRouter(tags=["health"])


@router.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": __version__,
        "environment": settings.environment,
    }
