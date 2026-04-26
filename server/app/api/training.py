"""Training Studio endpoints: capture sessions, sample listing, training trigger."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import settings
from app.core.logging import get_logger
from app.core.state import get_state
from app.ml import trainer
from app.ml.classifier import GestureClassifier

log = get_logger(__name__)
router = APIRouter(prefix="/api/training", tags=["training"])

_train_lock = asyncio.Lock()


class StartCapturePayload(BaseModel):
    label: str = Field(..., min_length=1, max_length=64)


class TrainPayload(BaseModel):
    epochs: int = Field(25, ge=1, le=500)


@router.post("/start")
async def start_capture(payload: StartCapturePayload) -> dict:
    state = get_state()
    state.data.training.label = payload.label.strip()
    state.data.training.active = True
    state.data.training.samples_collected = 0
    await state.save()
    return {"active": True, "label": state.data.training.label}


@router.post("/stop")
async def stop_capture() -> dict:
    state = get_state()
    state.data.training.active = False
    await state.save()
    return {"active": False, "label": state.data.training.label}


@router.get("/samples")
async def list_samples() -> dict:
    """Return per-label sample counts on disk."""
    counts: dict[str, int] = {}
    for path in sorted(settings.samples_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            counts[path.stem] = sum(1 for _ in fh)
    return {"labels": counts, "total": sum(counts.values())}


@router.post("/train")
async def train(
    payload: TrainPayload,
    background: BackgroundTasks,
    request: Request,
) -> dict:
    """Kick off training. Subsequent requests are rejected while one is running."""
    if _train_lock.locked():
        raise HTTPException(409, "training already in progress")

    # Capture the *shared* classifier instance held by the running reception
    # pipeline so that hot-reloading after training updates live inference
    # without a server restart.
    classifier: GestureClassifier | None = getattr(
        request.app.state, "classifier", None
    )

    async def _run() -> None:
        async with _train_lock:
            try:
                result = await trainer.train_from_dataset(epochs=payload.epochs)
                log.info("Training finished: %s", result)
                if classifier is not None:
                    await asyncio.to_thread(classifier.load)
                    log.info(
                        "Classifier hot-reloaded: %d label(s)",
                        len(getattr(classifier, "_labels", [])),
                    )
            except Exception as exc:  # noqa: BLE001
                log.exception("Training failed: %s", exc)

    background.add_task(_run)
    return {"status": "started", "epochs": payload.epochs}


@router.post("/load-default-pack")
async def load_default_pack() -> dict:
    return trainer.load_default_pack()
