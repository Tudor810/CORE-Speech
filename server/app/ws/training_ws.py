"""WS endpoint: dashboard <-> server for the Sign Language Training Studio.

Server -> client:
    {"type": "landmarks", "frame": {hands, handedness, has_hand}, "ts": float}
    {"type": "captured", "label": str, "samples": int}
    {"type": "status",   "msg": str}

Client -> server:
    {"action": "start_capture", "label": "<label>"}
    {"action": "stop_capture"}
    {"action": "save_sample"}    # snapshot the current rolling window
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections import deque
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import settings
from app.core.logging import get_logger
from app.ws.manager import training_manager

log = get_logger(__name__)
router = APIRouter()


@router.websocket("/ws/training/{session_id}")
async def training_ws(websocket: WebSocket, session_id: str) -> None:
    await training_manager.connect(session_id, websocket)
    app_state = websocket.app.state.app_state
    camera = websocket.app.state.camera
    landmarker = websocket.app.state.landmarker

    rolling: deque[list[list[float]]] = deque(maxlen=settings.lstm_window_size)
    capture_label: str | None = None

    async def producer() -> None:
        nonlocal capture_label
        async for frame in camera.subscribe():
            hand_frame = await landmarker.process(frame)
            if hand_frame.has_hand:
                rolling.append(hand_frame.hands[0])
            await training_manager.send_json(
                session_id,
                {
                    "type": "landmarks",
                    "frame": hand_frame.to_json(),
                    "ts": time.time(),
                },
            )

    producer_task = asyncio.create_task(producer(), name=f"training-{session_id}")

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            text = msg.get("text")
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue

            action = payload.get("action")
            if action == "start_capture":
                label = (payload.get("label") or "").strip()
                if not label:
                    await training_manager.send_json(
                        session_id, {"type": "status", "msg": "missing label"}
                    )
                    continue
                capture_label = label
                app_state.data.training.label = label
                app_state.data.training.active = True
                app_state.data.training.samples_collected = 0
                await app_state.save()
                await training_manager.send_json(
                    session_id, {"type": "status", "msg": f"capturing: {label}"}
                )
            elif action == "stop_capture":
                capture_label = None
                app_state.data.training.active = False
                await app_state.save()
                await training_manager.send_json(
                    session_id, {"type": "status", "msg": "stopped"}
                )
            elif action == "save_sample":
                if not capture_label:
                    await training_manager.send_json(
                        session_id, {"type": "status", "msg": "not capturing"}
                    )
                    continue
                if len(rolling) < settings.lstm_window_size:
                    await training_manager.send_json(
                        session_id,
                        {"type": "status", "msg": "buffer not yet full, hold steady"},
                    )
                    continue
                samples = await _persist_sample(capture_label, list(rolling))
                app_state.data.training.samples_collected = samples
                await app_state.save()
                await training_manager.send_json(
                    session_id,
                    {"type": "captured", "label": capture_label, "samples": samples},
                )
    except WebSocketDisconnect:
        pass
    finally:
        producer_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await producer_task
        await training_manager.disconnect(session_id)


async def _persist_sample(label: str, frames: list[list[list[float]]]) -> int:
    """Append one window to <samples_dir>/<label>.jsonl. Returns sample count."""
    path: Path = settings.samples_dir / f"{label}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"frames": frames}) + "\n"
    await asyncio.to_thread(_append_line, path, line)
    return await asyncio.to_thread(_count_lines, path)


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)
