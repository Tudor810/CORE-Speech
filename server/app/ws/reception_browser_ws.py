"""WS endpoint: dashboard browser -> server (sign-to-speech with browser camera).

Wire format:
  client -> server  (binary)  raw JPEG frame from <canvas>
  client -> server  (text)    JSON control: {"action": "stop"}
  server -> client  (binary)  raw 16-bit LE PCM @ openvoice_output_sample_rate
  server -> client  (text)    JSON: {"type": "status"|"gesture"|"phrase"|
                                      "translation"|"audio_end"|"landmarks"|"error", ...}

Frames are decoded with OpenCV in a worker thread and pushed onto a bounded
queue that the standard ``ReceptionPipeline`` consumes via its async-iterator
frame source. Slow-decoder frames are dropped instead of buffered so latency
stays low even if the browser uploads bursts.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import AsyncIterator

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.logging import get_logger
from app.pipelines.reception import ReceptionPipeline
from app.ws.manager import ConnectionManager

log = get_logger(__name__)
router = APIRouter()

# Channel used purely for outbound (server -> browser) backpressure.
browser_reception_manager = ConnectionManager("reception-browser")

_FRAME_QUEUE_MAX = 2  # keep at most ~2 frames pending (~150 ms @ 12 fps)


@router.websocket("/ws/reception_browser/{device_id}")
async def reception_browser_ws(websocket: WebSocket, device_id: str) -> None:
    await browser_reception_manager.connect(device_id, websocket)

    frame_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(
        maxsize=_FRAME_QUEUE_MAX
    )

    async def frame_iter() -> AsyncIterator[np.ndarray]:
        while True:
            item = await frame_queue.get()
            if item is None:
                return
            yield item

    pipeline = ReceptionPipeline(
        device_id=device_id,
        app_state=websocket.app.state.app_state,
        landmarker=websocket.app.state.landmarker,
        classifier=websocket.app.state.classifier,
        tts=websocket.app.state.tts,
        send_bytes=lambda b: browser_reception_manager.send_bytes(device_id, b),
        send_json=lambda j: browser_reception_manager.send_json(device_id, j),
        frames=frame_iter(),
        emit_landmarks=True,
    )
    pipeline_task = asyncio.create_task(
        pipeline.run(), name=f"reception-browser-{device_id}"
    )

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            data = msg.get("bytes")
            if data:
                # Drop the oldest frame if the decoder is behind so we keep
                # latency bounded.
                if frame_queue.full():
                    with contextlib.suppress(asyncio.QueueEmpty):
                        frame_queue.get_nowait()
                frame = await asyncio.to_thread(_decode_jpeg, data)
                if frame is not None:
                    with contextlib.suppress(asyncio.QueueFull):
                        frame_queue.put_nowait(frame)
                continue

            text = msg.get("text")
            if not text:
                continue
            t = text.strip().lower()
            if t == '{"action":"stop"}':
                await pipeline.stop()
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        log.exception("[reception-browser] %s crashed: %s", device_id, exc)
    finally:
        await pipeline.stop()
        with contextlib.suppress(asyncio.QueueFull):
            frame_queue.put_nowait(None)
        pipeline_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pipeline_task
        await browser_reception_manager.disconnect(device_id)


def _decode_jpeg(buf: bytes) -> np.ndarray | None:
    """Decode a JPEG byte string into a BGR numpy frame, or None on failure."""
    try:
        import cv2

        arr = np.frombuffer(buf, dtype=np.uint8)
        if arr.size == 0:
            return None
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame if frame is not None else None
    except Exception as exc:  # noqa: BLE001
        log.debug("JPEG decode failed: %s", exc)
        return None
