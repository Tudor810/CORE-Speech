"""WS endpoint: ES mic -> server -> text on ES (Pipeline II, speech -> text).

ESP32 sends binary frames (16-bit PCM @ 16 kHz). The server buffers them into
~1.5s windows and pipes each chunk through Vosk (local STT), then sends back
JSON {"type":"transcript", "text": "...", "final": true} for the OLED.

Every JSON event is also fanned out through :data:`transcripts_hub` so the
browser dashboard's /live page can subscribe at ``/ws/transcripts/{device_id}``
and render the same captions.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.logging import get_logger
from app.pipelines.emission import EmissionPipeline
from app.ws.manager import emission_manager
from app.ws.transcripts_ws import transcripts_hub

log = get_logger(__name__)
router = APIRouter()


# Control messages the bridge / dashboard can send over the same socket.
# Kept tiny on purpose — the wire is mostly raw PCM frames.
_ACTION_FLUSH = "flush"
_ACTION_RESET = "reset"


def _parse_action(text: str) -> str | None:
    """Best-effort extraction of ``{"action": "<verb>"}`` from a text frame.

    Tolerates whitespace, casing, and minor JSON typos so a noisy serial bridge
    can't accidentally drop a flush. Anything we don't understand is ignored.
    """
    raw = text.strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    action = payload.get("action")
    if not isinstance(action, str):
        return None
    return action.strip().lower() or None


@router.websocket("/ws/emission/{device_id}")
async def emission_ws(websocket: WebSocket, device_id: str) -> None:
    await emission_manager.connect(device_id, websocket)

    async def _send_and_broadcast(payload: dict[str, Any]) -> None:
        # Fan out every emission JSON event to dashboard subscribers in
        # addition to the originating device, so the /live page mirrors
        # whatever the OLED is showing.
        await emission_manager.send_json(device_id, payload)
        await transcripts_hub.publish(device_id, payload)

    pipeline = EmissionPipeline(
        device_id=device_id,
        stt=websocket.app.state.stt,
        send_json=_send_and_broadcast,
    )
    pipeline_task = asyncio.create_task(pipeline.run(), name=f"emission-{device_id}")

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if (data := msg.get("bytes")) is not None:
                await pipeline.feed_audio(data)
            elif (text := msg.get("text")) is not None:
                action = _parse_action(text)
                if action == _ACTION_FLUSH:
                    await pipeline.flush()
                elif action == _ACTION_RESET:
                    await pipeline.reset()
    except WebSocketDisconnect:
        pass
    finally:
        await pipeline.stop()
        pipeline_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pipeline_task
        await emission_manager.disconnect(device_id)
