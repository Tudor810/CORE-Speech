"""LSTM trainer stub.

Reads JSONL samples produced by the Training Studio, splits train/val,
and runs a short fine-tune. The full training loop is intentionally minimal:
the goal here is the *plumbing* (background task wiring, checkpointing) so
the dashboard's "Train" button has something real to call.

Sample file layout (one file per label, written by training API):
    server/data/samples/<label>.jsonl
where each line is {"frames": [[x,y,z]*21]*window}.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.config import settings
from app.core.logging import get_logger
from app.ml.lstm import SignLSTM, save_checkpoint

log = get_logger(__name__)


class _SignDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        self.sequences = torch.from_numpy(sequences).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def _load_samples(samples_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Walk samples_dir/*.jsonl into (X, y, labels)."""
    label_files = sorted(samples_dir.glob("*.jsonl"))
    labels: list[str] = []
    X_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []

    for label_idx, file in enumerate(label_files):
        label = file.stem
        labels.append(label)
        seqs: list[np.ndarray] = []
        with file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                frames = payload.get("frames")
                if not frames:
                    continue
                arr = np.array(frames, dtype=np.float32)
                if arr.ndim == 3:  # (window, 21, 3) -> flatten last two
                    arr = arr.reshape(arr.shape[0], -1)
                if arr.shape[-1] != settings.lstm_input_size:
                    continue
                if arr.shape[0] != settings.lstm_window_size:
                    continue
                seqs.append(arr)
        if seqs:
            stack = np.stack(seqs, axis=0)
            X_chunks.append(stack)
            y_chunks.append(np.full(len(stack), label_idx, dtype=np.int64))

    if not X_chunks:
        return (
            np.empty((0, settings.lstm_window_size, settings.lstm_input_size), np.float32),
            np.empty((0,), np.int64),
            labels,
        )
    return np.concatenate(X_chunks, axis=0), np.concatenate(y_chunks, axis=0), labels


def _train_blocking(epochs: int = 25, batch_size: int = 16, lr: float = 1e-3) -> dict:
    X, y, labels = _load_samples(settings.samples_dir)
    if len(X) == 0 or len(labels) < 2:
        msg = "Not enough samples or labels (need >=2 labels) to train."
        log.warning(msg)
        return {"status": "skipped", "reason": msg, "labels": labels}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignLSTM(num_classes=len(labels)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    ds = _SignDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    last_loss = float("nan")
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            total += loss.item() * xb.size(0)
        last_loss = total / max(len(ds), 1)
        if epoch % 5 == 0:
            log.info("epoch=%d loss=%.4f", epoch, last_loss)

    save_checkpoint(model, labels)
    return {
        "status": "ok",
        "labels": labels,
        "samples": int(len(ds)),
        "final_loss": float(last_loss),
    }


async def train_from_dataset(epochs: int = 25) -> dict:
    """Run training off the event loop so the API stays responsive."""
    log.info("Starting LSTM training (epochs=%d)", epochs)
    return await asyncio.to_thread(_train_blocking, epochs)


def load_default_pack() -> dict:
    """Hook for shipping a curated 'default sign pack'.

    For now this is a stub returning a manifest the dashboard can display.
    Replace with a real download/extract step when a pack is available.
    """
    return {
        "status": "stub",
        "message": "Default sign pack loader not yet implemented.",
        "suggested_labels": [
            "hello",
            "thanks",
            "yes",
            "no",
            "please",
            "help",
            "water",
            "food",
        ],
    }
