"""Per-device WebSocket connection registry with bounded send queues.

Each connected device (ESP32 / dashboard) gets:
  * a slot in `self._connections` keyed by `device_id`
  * a bounded outbound queue so a slow consumer can't pile up audio frames

Consumers should call `send_bytes` / `send_json` -- never write to the WS
directly -- so back-pressure is uniform.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import WebSocket

from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class _Connection:
    websocket: WebSocket
    out_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=64))
    pump_task: Optional[asyncio.Task] = None


class ConnectionManager:
    """Tracks connected WebSockets per channel."""

    def __init__(self, channel: str) -> None:
        self._channel = channel
        self._connections: dict[str, _Connection] = {}
        self._lock = asyncio.Lock()

    async def connect(self, device_id: str, websocket: WebSocket) -> _Connection:
        await websocket.accept()
        async with self._lock:
            if device_id in self._connections:
                # Replace any prior session for the same device.
                await self._teardown(device_id)
            conn = _Connection(websocket=websocket)
            conn.pump_task = asyncio.create_task(
                self._pump(device_id, conn), name=f"{self._channel}-pump-{device_id}"
            )
            self._connections[device_id] = conn
            log.info("[%s] connected: %s", self._channel, device_id)
            return conn

    async def disconnect(self, device_id: str) -> None:
        async with self._lock:
            await self._teardown(device_id)

    async def _teardown(self, device_id: str) -> None:
        conn = self._connections.pop(device_id, None)
        if not conn:
            return
        if conn.pump_task:
            conn.pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await conn.pump_task
        with contextlib.suppress(Exception):
            await conn.websocket.close()
        log.info("[%s] disconnected: %s", self._channel, device_id)

    async def _pump(self, device_id: str, conn: _Connection) -> None:
        """Drain the outbound queue to the WebSocket, one frame at a time."""
        try:
            while True:
                kind, payload = await conn.out_queue.get()
                if kind == "bytes":
                    await conn.websocket.send_bytes(payload)
                elif kind == "json":
                    await conn.websocket.send_json(payload)
                elif kind == "text":
                    await conn.websocket.send_text(payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] pump error for %s: %s", self._channel, device_id, exc)

    async def send_bytes(self, device_id: str, data: bytes) -> None:
        await self._enqueue(device_id, ("bytes", data))

    async def send_json(self, device_id: str, payload: dict[str, Any]) -> None:
        await self._enqueue(device_id, ("json", payload))

    async def send_text(self, device_id: str, text: str) -> None:
        await self._enqueue(device_id, ("text", text))

    def is_connected(self, device_id: str) -> bool:
        """Return True if a websocket is currently registered for device_id."""
        return device_id in self._connections

    async def _enqueue(self, device_id: str, item: tuple[str, Any]) -> None:
        conn = self._connections.get(device_id)
        if not conn:
            return
        if conn.out_queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                conn.out_queue.get_nowait()
        await conn.out_queue.put(item)


# Channel-specific singletons (one per WS endpoint kind).
reception_manager = ConnectionManager("reception")
emission_manager = ConnectionManager("emission")
training_manager = ConnectionManager("training")
