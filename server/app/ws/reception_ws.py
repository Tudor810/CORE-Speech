"""WS endpoint: server -> ES audio stream (Pipeline I, sign -> speech).

The ESP32 connects here and immediately starts receiving binary audio frames
plus JSON control frames. Frames are produced by `pipelines.reception.run`
which fans out from camera -> MediaPipe -> LSTM -> LLM -> ElevenLabs.

Wire format:
  binary  -> raw audio bytes (PCM_16000 by default)
  text    -> JSON: {"type": "status"|"transcript"|"error", ...}
"""

from __future__ import annotations

import asyncio
import contextlib

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.logging import get_logger
from app.pipelines.reception import ReceptionPipeline
from app.ws.manager import reception_manager

log = get_logger(__name__)
router = APIRouter()


@router.websocket("/ws/reception/{device_id}")
async def reception_ws(websocket: WebSocket, device_id: str) -> None:
    await reception_manager.connect(device_id, websocket)
    pipeline = ReceptionPipeline(
        device_id=device_id,
        app_state=websocket.app.state.app_state,
        landmarker=websocket.app.state.landmarker,
        classifier=websocket.app.state.classifier,
        tts=websocket.app.state.tts,
        send_bytes=lambda b: reception_manager.send_bytes(device_id, b),
        send_json=lambda j: reception_manager.send_json(device_id, j),
        camera=websocket.app.state.camera,
    )
    pipeline_task = asyncio.create_task(pipeline.run(), name=f"reception-{device_id}")

    try:
        while True:
            # ES rarely sends frames here, but keep the socket alive and
            # listen for control messages (e.g. {"action": "stop"}).
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            text = msg.get("text")
            if text:
                log.debug("[reception] %s -> %s", device_id, text)
                if text.strip().lower() == '{"action":"stop"}':
                    await pipeline.stop()
    except WebSocketDisconnect:
        pass
    finally:
        await pipeline.stop()
        pipeline_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pipeline_task
        await reception_manager.disconnect(device_id)
