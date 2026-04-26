"""Train the SignLSTM from the .npy dataset produced by ``extract_dataset.py``.

Walks ``--dataset-root/<label>/*.npy`` (default ``server/data/dataset``), turns
each (seq_len, 126) sequence into a (seq_len, 63) "dominant hand" sequence, and
fits the same ``SignLSTM`` architecture used by the runtime classifier. The
resulting checkpoint is saved to ``settings.lstm_checkpoint_path`` so the
FastAPI server can hot-load it on next boot.

Why 63 instead of 126?
----------------------
``extract_dataset.py`` faithfully captures both hands in a 126-dim vector
(``[Left 21*(x,y,z) | Right 21*(x,y,z)]``). The live pipeline
(``app.vision.landmarks.HandFrame.to_vector``) however feeds a single dominant
hand (63 dims) to the classifier. To keep the trained model byte-compatible
with inference, we project each frame to its dominant hand here:

    for frame f in (seq_len, 126):
        if Left  block (f[0:63])  is non-zero -> use it
        elif Right block (f[63:126]) is non-zero -> use it
        else: zeros(63)

If you ever switch the runtime to a 126-dim two-hand vector, set
``--keep-both-hands`` and bump ``settings.lstm_input_size`` to 126.

Usage
-----
    python -m scripts.train_from_dataset
    python -m scripts.train_from_dataset --epochs 60 --batch-size 16
    python -m scripts.train_from_dataset --dataset-root server/data/dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# Allow `python scripts/train_from_dataset.py` from the server/ folder *and*
# `python -m scripts.train_from_dataset` to both work.
_HERE = Path(__file__).resolve()
_SERVER_ROOT = _HERE.parents[1]
if str(_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVER_ROOT))

from app.config import settings  # noqa: E402
from app.ml.lstm import SignLSTM, save_checkpoint  # noqa: E402

LEFT_BLOCK = slice(0, 63)
RIGHT_BLOCK = slice(63, 126)


def to_dominant_hand(seq: np.ndarray) -> np.ndarray:
    """Project a (seq_len, 126) two-hand sequence to (seq_len, 63) per-frame.

    Picks the Left block when present; otherwise the Right block; otherwise
    zeros. Matches what the live ``HandLandmarker`` feeds the classifier.
    """
    if seq.ndim != 2 or seq.shape[1] != 126:
        raise ValueError(f"Expected (T, 126) sequence, got {seq.shape}")
    out = np.zeros((seq.shape[0], 63), dtype=np.float32)
    left = seq[:, LEFT_BLOCK]
    right = seq[:, RIGHT_BLOCK]
    has_left = np.any(left != 0.0, axis=1)
    has_right = np.any(right != 0.0, axis=1)
    out[has_left] = left[has_left]
    use_right = (~has_left) & has_right
    out[use_right] = right[use_right]
    return out


def load_dataset(
    dataset_root: Path,
    seq_len: int,
    keep_both_hands: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Walk ``<root>/<label>/*.npy`` into (X, y, labels).

    Returns
    -------
    X : (N, seq_len, F)  float32  -- F is 63 (dominant) or 126 (both)
    y : (N,)             int64
    labels : list[str]            -- alphabetical, index aligned with y
    """
    label_dirs = sorted(p for p in dataset_root.iterdir() if p.is_dir())
    labels: list[str] = []
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    feat_dim = 126 if keep_both_hands else 63

    for idx, lbl_dir in enumerate(label_dirs):
        npy_files = sorted(lbl_dir.glob("*.npy"))
        if not npy_files:
            continue
        labels.append(lbl_dir.name)
        bucket: list[np.ndarray] = []
        for f in npy_files:
            arr = np.load(f).astype(np.float32)
            if arr.shape != (seq_len, 126):
                print(f"  skip {f}: shape {arr.shape} != ({seq_len}, 126)")
                continue
            if not keep_both_hands:
                arr = to_dominant_hand(arr)
            bucket.append(arr)
        if not bucket:
            labels.pop()  # nothing usable here
            continue
        stack = np.stack(bucket, axis=0)
        Xs.append(stack)
        ys.append(np.full(len(stack), len(labels) - 1, dtype=np.int64))
        print(f"  label '{lbl_dir.name}': {len(stack)} sequences")

    if not Xs:
        return (
            np.empty((0, seq_len, feat_dim), np.float32),
            np.empty((0,), np.int64),
            labels,
        )
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0), labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SignLSTM from extract_dataset.py .npy files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        default=str(settings.data_dir / "dataset"),
        help="Folder that contains <label>/<n>.npy subfolders.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Fraction of data held out for validation (0 disables).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=settings.lstm_window_size,
        help="Expected per-sequence length (must match extract_dataset.py).",
    )
    parser.add_argument(
        "--keep-both-hands",
        action="store_true",
        help="Train on the full 126-dim two-hand vector instead of the "
             "dominant-hand 63-dim projection (will NOT match the live "
             "classifier unless you also bump settings.lstm_input_size).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the train/val split + model init.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_dir():
        print(f"Error: dataset root not found: {dataset_root}", file=sys.stderr)
        return 1

    print(f"Loading dataset from: {dataset_root}")
    X, y, labels = load_dataset(dataset_root, args.seq_len, args.keep_both_hands)
    if len(X) == 0:
        print("Error: no usable .npy sequences found.", file=sys.stderr)
        return 1
    if len(labels) < 2:
        print(
            f"Error: need at least 2 labels to train, found {len(labels)}: "
            f"{labels}",
            file=sys.stderr,
        )
        return 1

    feat_dim = X.shape[-1]
    print(
        f"Total sequences: {len(X)}  "
        f"shape={X.shape}  labels={labels}"
    )

    if feat_dim != settings.lstm_input_size:
        print(
            f"WARNING: feature dim {feat_dim} != settings.lstm_input_size "
            f"{settings.lstm_input_size}. The runtime classifier will reject "
            f"this checkpoint until that setting is updated.",
            file=sys.stderr,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).long()
    full_ds = TensorDataset(Xt, yt)

    if 0 < args.val_split < 1 and len(full_ds) >= 4:
        n_val = max(1, int(len(full_ds) * args.val_split))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(
            full_ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_ds, val_ds = full_ds, None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        if val_ds is not None
        else None
    )

    model = SignLSTM(input_size=feat_dim, num_classes=len(labels)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_n = 0
        running_correct = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            running_loss += loss.item() * xb.size(0)
            running_correct += (logits.argmax(-1) == yb).sum().item()
            running_n += xb.size(0)
        train_loss = running_loss / max(running_n, 1)
        train_acc = running_correct / max(running_n, 1)

        msg = f"epoch {epoch:>3}/{args.epochs}  loss={train_loss:.4f}  acc={train_acc:.3f}"

        if val_loader is not None:
            model.eval()
            v_correct = 0
            v_total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    v_correct += (logits.argmax(-1) == yb).sum().item()
                    v_total += xb.size(0)
            val_acc = v_correct / max(v_total, 1)
            msg += f"  val_acc={val_acc:.3f}"
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(model, labels)
                msg += "  [checkpoint]"
        else:
            save_checkpoint(model, labels)

        print(msg)

    if val_loader is None:
        save_checkpoint(model, labels)

    print("\nDone.")
    print(f"  labels:     {labels}")
    print(f"  checkpoint: {settings.lstm_checkpoint_path}")
    if val_loader is not None:
        print(f"  best val_acc: {best_val_acc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
