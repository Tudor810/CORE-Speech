"""Threaded webcam capture.

Wraps ``cv2.VideoCapture`` in a background thread that pushes the most
recent BGR frame into a single-slot buffer. The GUI grabs frames at its
own pace by calling :meth:`Camera.read_latest`; back-pressure is impossible
because we always overwrite the slot.

Multiple consumers (the live view, the training view) typically take turns
— starting a session in one tab automatically stops the camera in the
other via :meth:`Camera.stop`.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class CameraInfo:
    index: int
    width: int
    height: int


class CameraError(RuntimeError):
    """Raised when the requested camera index can't be opened."""


class Camera:
    """Background-thread cv2 webcam reader."""

    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._info: Optional[CameraInfo] = None
        self._frame_count = 0

    # --------------------------- lifecycle ----------------------------------
    def start(
        self,
        index: int,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> CameraInfo:
        """Open the camera and start the background reader thread."""
        self.stop()

        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            raise CameraError(f"could not open camera index {index}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)

        self._cap = cap
        self._info = CameraInfo(index=index, width=actual_w, height=actual_h)
        self._stop.clear()
        self._frame = None
        self._frame_count = 0

        self._thread = threading.Thread(
            target=self._run, name=f"camera-{index}", daemon=True
        )
        self._thread.start()
        return self._info

    def stop(self) -> None:
        """Stop the reader thread and release the camera handle."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        with self._lock:
            self._frame = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def info(self) -> Optional[CameraInfo]:
        return self._info

    # ---------------------------- producer ----------------------------------
    def _run(self) -> None:
        assert self._cap is not None
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                # Brief backoff so we don't hot-spin if the device disappeared.
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame
                self._frame_count += 1

    # ---------------------------- consumer ----------------------------------
    def read_latest(self) -> Optional[np.ndarray]:
        """Return a copy of the latest BGR frame, or None if none arrived yet."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count


# ---------------------------------------------------------------------------
# Camera enumeration helpers (mirrors live_predict.list_cameras).
# ---------------------------------------------------------------------------
def list_cameras(max_index: int = 6) -> list[int]:
    """Return the indices of cameras that opened successfully."""
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    found: list[int] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
        cap.release()
    return found
