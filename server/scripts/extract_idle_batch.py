"""Batch-extract idle (background) windows from many short clips.

Reads every video listed in ``--list`` (one path per line) -- or every
``*.mp4`` under ``--video-dir`` -- and chops each one into fixed-length
landmark windows using MediaPipe Hands. All windows land in a single
``<output>/<label>/<n>.npy`` folder so ``train_from_dataset.py`` picks them
up as a regular class.

Why a dedicated script (instead of looping ``extract_dataset.py --mode
windows`` 67 times)?
- MediaPipe HandLandmarker is initialized **once** instead of per-video.
- Output indices are continuous across all source videos.
- Per-video caps (``--max-windows-per-video``) and a global cap
  (``--max-windows-total``) keep the idle class balanced against the
  sign-rep classes (typically 60-100 sequences each).

Window layout, dtype and channel ordering match ``extract_dataset.py`` so
the resulting ``.npy`` files are byte-compatible with the training
pipeline.

Usage
-----
    python -m scripts.extract_idle_batch \
        --video-dir ../idle \
        --output data/dataset_filtered \
        --label idle \
        --seq-len 30 --stride 30 \
        --max-windows-per-video 2 --max-windows-total 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

_HERE = Path(__file__).resolve()
_SERVER_ROOT = _HERE.parents[1]
if str(_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVER_ROOT))

from scripts.extract_dataset import (  # noqa: E402  (after sys.path tweak)
    landmarks_to_frame_vector,
    next_sequence_index,
    standardize_sequence,
)

_DEFAULT_MODEL_PATH = _SERVER_ROOT / "models" / "hand_landmarker.task"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-extract fixed-length windows from many videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video-dir", help="Folder containing *.mp4 files.")
    src.add_argument("--list", help="Text file with one absolute path per line.")
    p.add_argument("--output", default="data/dataset_filtered",
                   help="Dataset root; saves to <output>/<label>/<n>.npy.")
    p.add_argument("--label", default="idle", help="Class label folder name.")
    p.add_argument("--seq-len", type=int, default=30,
                   help="Frames per window (must match extract_dataset.py).")
    p.add_argument("--stride", type=int, default=30,
                   help="Step between consecutive windows in a single video.")
    p.add_argument("--max-windows-per-video", type=int, default=2,
                   help="Cap on windows kept per source video.")
    p.add_argument("--max-windows-total", type=int, default=120,
                   help="Global cap across all videos (0 = unlimited).")
    p.add_argument("--max-hands", type=int, default=2)
    p.add_argument("--detection-confidence", type=float, default=0.5)
    p.add_argument("--presence-confidence", type=float, default=0.5)
    p.add_argument("--tracking-confidence", type=float, default=0.5)
    p.add_argument("--model", default=str(_DEFAULT_MODEL_PATH),
                   help="Path to hand_landmarker.task.")
    return p.parse_args()


def collect_videos(args: argparse.Namespace) -> list[Path]:
    if args.list:
        list_path = Path(args.list)
        if not list_path.is_file():
            print(f"Error: --list file not found: {list_path}", file=sys.stderr)
            sys.exit(1)
        paths = [
            Path(line.strip())
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    else:
        d = Path(args.video_dir)
        if not d.is_dir():
            print(f"Error: --video-dir not found: {d}", file=sys.stderr)
            sys.exit(1)
        paths = sorted(d.glob("*.mp4"))
    paths = [p for p in paths if p.is_file()]
    if not paths:
        print("Error: no input videos found.", file=sys.stderr)
        sys.exit(1)
    return paths


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"Error: hand_landmarker.task not found at {model_path}",
              file=sys.stderr)
        return 1

    out_dir = Path(args.output) / args.label
    next_idx = next_sequence_index(out_dir)
    print(f"Saving to {out_dir}/  (starting index: {next_idx})")

    videos = collect_videos(args)
    print(f"Processing {len(videos)} video(s).")

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=args.max_hands,
        min_hand_detection_confidence=args.detection_confidence,
        min_hand_presence_confidence=args.presence_confidence,
        min_tracking_confidence=args.tracking_confidence,
    )

    saved_total = 0
    seen = 0
    cap_global = args.max_windows_total if args.max_windows_total > 0 else None
    # MediaPipe VIDEO mode requires strictly monotonic timestamps across all
    # detect_for_video() calls -- even across different source videos sharing
    # the same landmarker instance. We track a global ms cursor and bump it
    # by a safety gap (1s) between videos.
    global_ts_ms = 0
    INTER_VIDEO_GAP_MS = 1000

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        for video_path in videos:
            if cap_global is not None and saved_total >= cap_global:
                print(f"Reached global cap ({cap_global}); stopping.")
                break

            seen += 1
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"  [{seen}/{len(videos)}] skip (cannot open): "
                      f"{video_path.name}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if fps <= 1e-3:
                fps = 30.0
            ms_per_frame = 1000.0 / fps

            frames: list[np.ndarray] = []
            video_start_ms = global_ts_ms + INTER_VIDEO_GAP_MS
            frame_idx = 0
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts = int(video_start_ms + frame_idx * ms_per_frame)
                if ts <= global_ts_ms:
                    ts = global_ts_ms + 1
                global_ts_ms = ts
                result = landmarker.detect_for_video(mp_image, ts)
                frames.append(landmarks_to_frame_vector(result))
                frame_idx += 1
            cap.release()

            saved_here = 0
            if len(frames) >= args.seq_len:
                start = 0
                while start + args.seq_len <= len(frames):
                    if saved_here >= args.max_windows_per_video:
                        break
                    if cap_global is not None and saved_total >= cap_global:
                        break
                    window = frames[start : start + args.seq_len]
                    seq_arr = standardize_sequence(window, args.seq_len)
                    out_path = out_dir / f"{next_idx}.npy"
                    np.save(out_path, seq_arr)
                    next_idx += 1
                    saved_here += 1
                    saved_total += 1
                    start += args.stride

            print(f"  [{seen}/{len(videos)}] {video_path.name}: "
                  f"frames={len(frames)} saved={saved_here} "
                  f"(total={saved_total})")

    print(f"\nDone. Saved {saved_total} window(s) for label "
          f"'{args.label}' in {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
