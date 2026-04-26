"""PyTorch LSTM classifier for sign-language gesture sequences."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from app.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


class SignLSTM(nn.Module):
    """Two-layer LSTM with a fully-connected classification head.

    Input  : (batch, window, features=63)
    Output : (batch, num_classes) logits
    """

    def __init__(
        self,
        input_size: int = settings.lstm_input_size,
        hidden_size: int = settings.lstm_hidden_size,
        num_layers: int = settings.lstm_num_layers,
        num_classes: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # take final timestep
        return self.fc(last)


def save_checkpoint(
    model: SignLSTM,
    labels: list[str],
    path: Path = settings.lstm_checkpoint_path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": labels,
            "input_size": model.input_size,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "num_classes": model.num_classes,
        },
        path,
    )
    log.info("Saved LSTM checkpoint -> %s", path)


def load_checkpoint(
    path: Path = settings.lstm_checkpoint_path,
) -> tuple[SignLSTM | None, list[str]]:
    """Return (model, labels) or (None, []) if no checkpoint exists."""
    if not path.exists():
        log.warning(
            "No LSTM checkpoint at %s; classifier will return 'unknown'.", path
        )
        return None, []
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    labels: list[str] = ckpt.get("labels", [])
    model = SignLSTM(
        input_size=ckpt.get("input_size", settings.lstm_input_size),
        hidden_size=ckpt.get("hidden_size", settings.lstm_hidden_size),
        num_layers=ckpt.get("num_layers", settings.lstm_num_layers),
        num_classes=ckpt.get("num_classes", len(labels) or 1),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    log.info("Loaded LSTM checkpoint with %d labels", len(labels))
    return model, labels
