"""WS endpoint: dashboard browser <- server (live STT transcripts).

The ESP32's microphone audio arrives on ``/ws/emission/{device_id}`` and is
transcribed by Vosk (local STT) inside ``EmissionPipeline``. Those JSON events
are normally sent back to the ESP32 only; this module adds a tiny per-device
pub/sub bus so any number of browsers (e.g. the /live page) can subscribe to
the same stream and render the captions in real time.

Wire format (server -> client, JSON only):

    {"type": "status",     "msg": "ready" | "subscribed", ...}
    {"type": "transcript", "text": "...", "language": "en",
                            "duration": 1.5, "final": true}
    {"type": "error",      "msg": "..."}

The hub deliberately uses a bounded per-subscriber queue so a slow browser
tab can't apply backpressure to the STT pipeline; the oldest pending event
is dropped first.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.logging import get_logger

log = get_logger(__name__)
router = APIRouter()


class TranscriptHub:
    """Per-device pub/sub bus for STT events from ``EmissionPipeline``."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, device_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        async with self._lock:
            self._subscribers.setdefault(device_id, set()).add(q)
        log.info(
            "[transcripts] +subscriber for %s (total=%d)",
            device_id,
            len(self._subscribers[device_id]),
        )
        return q

    async def unsubscribe(self, device_id: str, q: asyncio.Queue) -> None:
        async with self._lock:
            subs = self._subscribers.get(device_id)
            if not subs:
                return
            subs.discard(q)
            remaining = len(subs)
            if not subs:
                self._subscribers.pop(device_id, None)
        log.info(
            "[transcripts] -subscriber for %s (remaining=%d)",
            device_id,
            remaining,
        )

    async def publish(self, device_id: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            subs = list(self._subscribers.get(device_id, ()))
        if not subs:
            return
        for q in subs:
            if q.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(payload)


transcripts_hub = TranscriptHub()


@router.websocket("/ws/transcripts/{device_id}")
async def transcripts_ws(websocket: WebSocket, device_id: str) -> None:
    """Browser subscribes here to receive STT events for ``device_id``."""
    await websocket.accept()
    q = await transcripts_hub.subscribe(device_id)
    try:
        await websocket.send_json(
            {"type": "status", "msg": "subscribed", "device_id": device_id}
        )
        # Two cooperating tasks: deliver hub events; drain any client text
        # (we accept ping/keepalive but ignore the payload).
        deliver_task = asyncio.create_task(
            _deliver(websocket, q), name=f"transcripts-deliver-{device_id}"
        )
        recv_task = asyncio.create_task(
            _drain_recv(websocket), name=f"transcripts-recv-{device_id}"
        )
        done, pending = await asyncio.wait(
            {deliver_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t
        for t in done:
            exc = t.exception()
            if exc is not None and not isinstance(exc, WebSocketDisconnect):
                log.warning("[transcripts:%s] task error: %s", device_id, exc)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        log.warning("[transcripts:%s] error: %s", device_id, exc)
    finally:
        await transcripts_hub.unsubscribe(device_id, q)
        with contextlib.suppress(Exception):
            await websocket.close()


async def _deliver(websocket: WebSocket, q: "asyncio.Queue[dict[str, Any]]") -> None:
    while True:
        payload = await q.get()
        await websocket.send_json(payload)


async def _drain_recv(websocket: WebSocket) -> None:
    """Discard anything the client sends; just keeps the WS healthy."""
    while True:
        msg = await websocket.receive()
        if msg.get("type") == "websocket.disconnect":
            return
