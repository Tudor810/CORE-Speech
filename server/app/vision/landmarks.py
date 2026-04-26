"""MediaPipe Hands wrapper (Tasks API).

Produces a JSON-serializable dict per frame so the same payload can drive both
the LSTM classifier (numpy view) and the Training/Live Studio preview
(WebSocket).

We use the modern ``mediapipe.tasks.python.vision.HandLandmarker`` for two
reasons:

* The legacy ``mediapipe.solutions.hands`` API is *not* shipped in newer
  MediaPipe wheels (Python 3.13+, which includes 3.14 used in dev). Importing
  it raises ``AttributeError`` at runtime, which is exactly what was breaking
  the live page (no landmarks ever reached the browser, no signs ever
  registered).

* ``scripts/extract_dataset.py`` and ``scripts/live_predict.py`` both use the
  Tasks API, so going through the same code path keeps train-time and
  inference-time landmark distributions byte-identical.

The Tasks API is *not* thread-safe and ``detect_for_video`` requires
monotonically increasing timestamps. We serialise calls with an ``asyncio.Lock``
and offload CPU-bound ``detect_for_video`` to a worker thread.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3
HAND_VECTOR_SIZE = NUM_LANDMARKS * COORDS_PER_LANDMARK   # 63
FRAME_VECTOR_SIZE = 2 * HAND_VECTOR_SIZE                  # 126 (Left | Right)


@dataclass
class HandFrame:
    """Single-frame landmark payload."""

    hands: list[list[list[float]]]   # [[ [x,y,z]*21 ], ...]
    handedness: list[str]            # ["Left" | "Right"] aligned with `hands`
    has_hand: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "hands": self.hands,
            "handedness": self.handedness,
            "has_hand": self.has_hand,
        }

    def to_vector(self) -> np.ndarray:
        """Return a fixed-length 63-dim vector for the dominant hand.

        Mirrors the train-time projection used by ``scripts/train_from_dataset.
        py`` (``to_dominant_hand``): build a ``[Left|Right]`` 126-dim vector
        from MediaPipe's handedness labels, then pick Left if non-zero, else
        Right, else zeros.

        Picking ``hands[0]`` (the previous behaviour) was wrong because
        MediaPipe's detection order is not stable, so the LSTM occasionally
        saw the "Right" block in slots it was trained to expect "Left" in.
        """
        if not self.has_hand:
            return np.zeros(settings.lstm_input_size, dtype=np.float32)

        full = np.zeros(FRAME_VECTOR_SIZE, dtype=np.float32)
        for i, hand in enumerate(self.hands):
            label = self.handedness[i] if i < len(self.handedness) else "Left"
            offset = 0 if label == "Left" else HAND_VECTOR_SIZE
            flat = np.asarray(hand, dtype=np.float32).flatten()
            if flat.size == HAND_VECTOR_SIZE:
                full[offset : offset + HAND_VECTOR_SIZE] = flat

        left = full[:HAND_VECTOR_SIZE]
        right = full[HAND_VECTOR_SIZE:]
        if np.any(left):
            dominant = left
        elif np.any(right):
            dominant = right
        else:
            dominant = np.zeros(HAND_VECTOR_SIZE, dtype=np.float32)

        # Allow the runtime to override the LSTM input size (e.g. 126 for the
        # rare two-hand model); otherwise we always return a 63-dim slice.
        target = settings.lstm_input_size
        if target == HAND_VECTOR_SIZE:
            return dominant
        if target == FRAME_VECTOR_SIZE:
            return full
        if dominant.size >= target:
            return dominant[:target]
        out = np.zeros(target, dtype=np.float32)
        out[: dominant.size] = dominant
        return out


class HandLandmarker:
    def __init__(self) -> None:
        self._landmarker = None
        self._lock = asyncio.Lock()
        self._last_ts_ms = -1  # monotonic timestamp counter for VIDEO mode

    def _ensure_loaded(self) -> None:
        if self._landmarker is not None:
            return

        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        model_path = settings.mediapipe_model_path
        if not model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe HandLandmarker model not found at {model_path}. "
                f"Download it from https://storage.googleapis.com/mediapipe-models/"
                f"hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task "
                f"and drop it in server/models/."
            )

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=settings.mediapipe_max_hands,
            min_hand_detection_confidence=settings.mediapipe_min_detection_confidence,
            min_hand_presence_confidence=settings.mediapipe_min_presence_confidence,
            min_tracking_confidence=settings.mediapipe_min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        log.info(
            "MediaPipe HandLandmarker (Tasks API) loaded from %s", model_path
        )

    def _process_blocking(self, frame_bgr: np.ndarray) -> HandFrame:
        import cv2  # lazy
        import mediapipe as mp

        self._ensure_loaded()

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # ``detect_for_video`` requires strictly increasing timestamps.
        ts_ms = max(self._last_ts_ms + 1, time.monotonic_ns() // 1_000_000)
        self._last_ts_ms = ts_ms
        result = self._landmarker.detect_for_video(mp_image, ts_ms)  # type: ignore[union-attr]

        hands: list[list[list[float]]] = []
        handedness: list[str] = []
        if result.hand_landmarks:
            handedness_lists = result.handedness or []
            for i, lm_list in enumerate(result.hand_landmarks):
                hands.append([[lm.x, lm.y, lm.z] for lm in lm_list])
                if i < len(handedness_lists) and handedness_lists[i]:
                    handedness.append(handedness_lists[i][0].category_name)
                else:
                    handedness.append("Left")
        return HandFrame(
            hands=hands, handedness=handedness, has_hand=bool(hands)
        )

    async def process(self, frame_bgr: np.ndarray) -> HandFrame:
        async with self._lock:
            return await asyncio.to_thread(self._process_blocking, frame_bgr)

    def close(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:  # noqa: BLE001
                pass
            self._landmarker = None
