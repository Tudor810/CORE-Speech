"""Training Studio backend.

A thin facade around two existing pieces of the server:

* The MediaPipe HandLandmarker — wrapped here in a synchronous wrapper
  so the GUI can keep its rolling 30-frame window of *single-hand*
  21x3 landmarks (the format ``samples_dir/<label>.jsonl`` expects).

* :func:`app.ml.trainer.train_from_dataset` — kicked off through the
  background asyncio runtime so the UI never blocks on training.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# Runtime import side-effect: ``server/`` joins ``sys.path`` so the
# subsequent ``app.*`` imports resolve to the server package.
from ..runtime import submit_coro
from app.config import settings  # noqa: E402
from app.ml import trainer  # noqa: E402
from app.ml.classifier import GestureClassifier  # noqa: E402

from .camera import Camera


@dataclass
class TrainingFrame:
    """Per-frame payload for the on-screen overlay."""

    bgr: np.ndarray
    hands: list[list[list[float]]]
    handedness: list[str]
    has_hand: bool


@dataclass
class CapturedSample:
    label: str
    samples_for_label: int
    total_samples: int


# ---------------------------------------------------------------------------
# Training session: camera -> landmarks -> rolling 30-frame window
# ---------------------------------------------------------------------------
EventCallback = Callable[[TrainingFrame], None]


class TrainingSession:
    """Runs MediaPipe landmark extraction off the GUI thread."""

    def __init__(
        self,
        camera: Camera,
        on_frame: EventCallback,
        *,
        window_size: int = settings.lstm_window_size,
    ) -> None:
        self._camera = camera
        self._on_frame = on_frame
        self._window_size = window_size

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._buffer: deque[list[list[float]]] = deque(maxlen=window_size)
        self._buf_lock = threading.Lock()

    # --------------------------- lifecycle ----------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        with self._buf_lock:
            self._buffer.clear()
        self._thread = threading.Thread(
            target=self._run, name="training-session", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def buffer_progress(self) -> tuple[int, int]:
        with self._buf_lock:
            return len(self._buffer), self._window_size

    def snapshot_buffer(self) -> Optional[list[list[list[float]]]]:
        """Return a deep-copied window if it's full, else None."""
        with self._buf_lock:
            if len(self._buffer) < self._window_size:
                return None
            return [list(map(list, frame)) for frame in self._buffer]

    # --------------------------- worker -------------------------------------
    def _run(self) -> None:
        model_path = settings.mediapipe_model_path
        if not Path(model_path).is_file():
            print(f"  [training] hand model missing at {model_path}")
            return

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=settings.mediapipe_max_hands,
            min_hand_detection_confidence=settings.mediapipe_min_detection_confidence,
            min_hand_presence_confidence=settings.mediapipe_min_presence_confidence,
            min_tracking_confidence=settings.mediapipe_min_tracking_confidence,
        )

        last_seen_frame_id = -1
        ts_ms = 0
        with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
            while not self._stop.is_set():
                frame_id = self._camera.frame_count
                if frame_id == last_seen_frame_id or not self._camera.is_running:
                    time.sleep(0.005)
                    continue
                last_seen_frame_id = frame_id

                bgr = self._camera.read_latest()
                if bgr is None:
                    continue

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms += 1
                result = landmarker.detect_for_video(mp_image, ts_ms)

                hands: list[list[list[float]]] = []
                handedness: list[str] = []
                if result.hand_landmarks:
                    handedness_lists = result.handedness or []
                    for i, lm_list in enumerate(result.hand_landmarks):
                        hands.append([[lm.x, lm.y, lm.z] for lm in lm_list])
                        if i < len(handedness_lists) and handedness_lists[i]:
                            handedness.append(
                                handedness_lists[i][0].category_name
                            )
                        else:
                            handedness.append("Left")

                if hands:
                    with self._buf_lock:
                        # Match server/app/ws/training_ws.py: store the *first*
                        # hand only — the trainer expects a single-hand
                        # 21x3 landmark window per sample.
                        self._buffer.append(hands[0])

                try:
                    self._on_frame(
                        TrainingFrame(
                            bgr=bgr,
                            hands=hands,
                            handedness=handedness,
                            has_hand=bool(hands),
                        )
                    )
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Sample persistence + counts (mirrors server/app/api/training.py logic)
# ---------------------------------------------------------------------------
def list_sample_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(settings.samples_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            counts[path.stem] = sum(1 for _ in fh)
    return counts


def save_sample(label: str, frames: list[list[list[float]]]) -> CapturedSample:
    """Append one window to ``samples_dir/<label>.jsonl`` and return the count."""
    label = (label or "").strip()
    if not label:
        raise ValueError("label is required")
    if not frames:
        raise ValueError("no frames to save")

    settings.samples_dir.mkdir(parents=True, exist_ok=True)
    path: Path = settings.samples_dir / f"{label}.jsonl"
    line = json.dumps({"frames": frames}) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)

    counts = list_sample_counts()
    return CapturedSample(
        label=label,
        samples_for_label=counts.get(label, 0),
        total_samples=sum(counts.values()),
    )


# ---------------------------------------------------------------------------
# Training trigger
# ---------------------------------------------------------------------------
def train(
    epochs: int = 25,
    *,
    on_done: Optional[Callable[[Optional[dict], Optional[str]], None]] = None,
) -> None:
    """Kick off LSTM training in the background.

    ``on_done`` runs on the runtime loop's thread when training finishes.
    The classifier is *not* automatically reloaded here because the live
    engine creates a fresh predictor every time the user starts a session
    — but if a classifier instance is passed via :func:`set_live_classifier`
    we'll hot-reload that one too.
    """

    async def _go() -> tuple[Optional[dict], Optional[str]]:
        try:
            result = await trainer.train_from_dataset(epochs=epochs)
        except Exception as exc:
            return None, str(exc)
        if _shared_classifier is not None:
            try:
                _shared_classifier.load()
            except Exception as exc:
                return result, f"trained ok but reload failed: {exc}"
        return result, None

    fut = submit_coro(_go())

    def _on_done(_fut) -> None:
        if on_done is None:
            return
        try:
            result, err = fut.result()
        except Exception as exc:
            result, err = None, str(exc)
        on_done(result, err)

    fut.add_done_callback(_on_done)


def load_default_pack() -> dict:
    """Suggest a starter set of labels (server-side stub)."""
    return trainer.load_default_pack()


# Optional shared classifier hook (currently unused — the live engine
# instantiates its own predictor per session — but kept for symmetry with
# the FastAPI server which hot-reloads its long-lived classifier).
_shared_classifier: Optional[GestureClassifier] = None


def set_live_classifier(classifier: Optional[GestureClassifier]) -> None:
    global _shared_classifier
    _shared_classifier = classifier
