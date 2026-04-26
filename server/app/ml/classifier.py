"""Real-time gesture classifier with sliding-window buffer and debouncing."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from app.config import settings
from app.core.logging import get_logger
from app.ml.lstm import SignLSTM, load_checkpoint

log = get_logger(__name__)


@dataclass
class Prediction:
    label: str
    confidence: float
    is_new: bool  # True only when a new label is emitted (not a duplicate)


class GestureClassifier:
    """Buffers per-frame landmark vectors and emits gesture labels."""

    def __init__(
        self,
        window_size: int = settings.lstm_window_size,
        confidence_threshold: float = settings.lstm_confidence_threshold,
    ) -> None:
        self._window_size = window_size
        self._threshold = confidence_threshold
        self._buffer: deque[np.ndarray] = deque(maxlen=window_size)
        self._model: Optional[SignLSTM] = None
        self._labels: list[str] = []
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = asyncio.Lock()
        self._last_label: str | None = None

    def load(self) -> None:
        """Load weights from disk; safe to call multiple times.

        Concurrent ``_infer`` calls may run while we swap. We assemble the
        new model + labels locally first, then publish them in an order that
        keeps ``len(self._labels) >= model_outputs`` at all times so an
        in-flight prediction can never index past the end of ``self._labels``.
        """
        model, labels = load_checkpoint()
        if model is None:
            # Drop the model first so any concurrent _infer short-circuits
            # via the is_ready check on its next call.
            self._model = None
            self._labels = []
            self._last_label = None
            return
        new_model = model.to(self._device)
        new_model.eval()
        # Publish labels first so they are at least as wide as the new model.
        self._labels = list(labels)
        self._model = new_model
        self._last_label = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None and len(self._labels) > 0

    def push(self, vector: np.ndarray) -> None:
        """Add a single 63-dim landmark vector to the rolling window."""
        if vector.size != settings.lstm_input_size:
            raise ValueError(
                f"Expected vector of size {settings.lstm_input_size}, "
                f"got {vector.size}"
            )
        self._buffer.append(vector.astype(np.float32))

    def reset(self) -> None:
        self._buffer.clear()
        self._last_label = None

    async def predict(self) -> Optional[Prediction]:
        """Run inference if the buffer is full and confidence passes threshold."""
        if not self.is_ready:
            return None
        if len(self._buffer) < self._window_size:
            return None

        seq = np.stack(list(self._buffer), axis=0)  # (window, 63)
        async with self._lock:
            return await asyncio.to_thread(self._infer, seq)

    def _infer(self, seq: np.ndarray) -> Optional[Prediction]:
        assert self._model is not None
        with torch.no_grad():
            x = torch.from_numpy(seq).unsqueeze(0).to(self._device)
            logits = self._model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        if conf < self._threshold:
            return None
        label = self._labels[idx]
        is_new = label != self._last_label
        self._last_label = label
        return Prediction(label=label, confidence=conf, is_new=is_new)
