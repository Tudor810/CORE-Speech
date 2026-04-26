"""Async OpenCV camera stream with multi-subscriber fan-out.

A single `CameraStream` owns the `cv2.VideoCapture` device. Multiple consumers
(Reception pipeline, Training WebSocket) `subscribe()` and pull frames from
their own bounded queue. The producer drops frames for slow consumers instead
of blocking, so a stalled subscriber never starves the inference path.

Frame grabs run inside `asyncio.to_thread` because OpenCV's read() is blocking.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import AsyncIterator, Optional

import numpy as np

from app.core.logging import get_logger

log = get_logger(__name__)


class CameraStream:
    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        queue_size: int = 2,
    ) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._fps = fps
        self._queue_size = queue_size

        self._capture = None  # Lazily created cv2.VideoCapture
        self._task: Optional[asyncio.Task] = None
        self._subscribers: list[asyncio.Queue[np.ndarray]] = []
        self._lock = asyncio.Lock()
        self._running = False

    async def _ensure_started(self) -> None:
        if self._running:
            return
        async with self._lock:
            if self._running:
                return
            await asyncio.to_thread(self._open_capture)
            self._running = True
            self._task = asyncio.create_task(self._produce(), name="camera-producer")
            log.info("Camera stream started (index=%s)", self._index)

    def _open_capture(self) -> None:
        import cv2  # imported lazily so dev tooling doesn't require it

        cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self._index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._capture = cap

    def _read_blocking(self) -> Optional[np.ndarray]:
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        return frame if ok else None

    async def _produce(self) -> None:
        """Background loop: read frames and broadcast to subscribers."""
        try:
            while self._running:
                frame = await asyncio.to_thread(self._read_blocking)
                if frame is None:
                    await asyncio.sleep(0.02)
                    continue
                for q in list(self._subscribers):
                    if q.full():
                        # Drop oldest to keep latency bounded.
                        with contextlib.suppress(asyncio.QueueEmpty):
                            q.get_nowait()
                    with contextlib.suppress(asyncio.QueueFull):
                        q.put_nowait(frame)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            log.exception("Camera producer crashed: %s", exc)

    async def subscribe(self) -> AsyncIterator[np.ndarray]:
        """Yield BGR frames as a numpy array. Caller must consume promptly."""
        await self._ensure_started()
        q: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.append(q)
        try:
            while True:
                frame = await q.get()
                yield frame
        finally:
            self._subscribers.remove(q)
            if not self._subscribers:
                # No more consumers: stop the camera to free the device.
                await self.aclose()

    async def aclose(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._capture is not None:
            cap = self._capture
            self._capture = None
            await asyncio.to_thread(cap.release)
        log.info("Camera stream stopped")
