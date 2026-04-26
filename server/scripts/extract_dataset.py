"""Auto-extract sign language repetitions from a continuous webcam recording.

Reads a long video where the same sign is repeated many times (with the hands
returning to a resting / out-of-frame position between reps) and chops it into
fixed-length landmark sequences ready for LSTM training.

Pipeline
--------
1. Frames are read with ``cv2.VideoCapture``.
2. **MediaPipe Hands** (``mediapipe.tasks.python.vision.HandLandmarker`` in
   ``VIDEO`` running mode) extracts the 21 (x, y, z) landmarks for up to two
   hands per frame. The legacy ``mediapipe.solutions.hands`` API is not
   shipped in newer wheels (incl. Python 3.14), so we use the Tasks API and
   require a local ``hand_landmarker.task`` model file.
3. A small two-state machine segments the video into individual repetitions:

       IDLE       --(hands appear)-->     RECORDING
       RECORDING  --(hands missing for `--patience` frames)--> IDLE

   A "patience" counter (default 5 frames) tolerates the brief tracking drops
   MediaPipe occasionally produces while the hands are still visible. Frames
   inside the patience window are kept in the sequence as zeros so the
   timeline stays continuous; the trailing missing frames at the end of a
   recording are discarded.
4. Each captured rep is standardized to exactly ``--seq-len`` frames
   (default 30) by zero-padding (right) or truncating (keep first N).
5. Sequences are saved as ``<output>/<label>/<sequence_number>.npy``. The
   numbering continues from any existing files in that folder, so you can run
   the script repeatedly on different source videos for the same label.

Resulting numpy array shape (per saved file)
--------------------------------------------
    arr.shape == (seq_len, 126)         # default: (30, 126)
    arr.dtype == float32

Layout of the 126-dim per-frame vector::

        index   0 .. 62    -> Left  hand: 21 landmarks * (x, y, z) = 63 floats
        index  63 .. 125   -> Right hand: 21 landmarks * (x, y, z) = 63 floats

Hands are slotted by MediaPipe's handedness classification. A frame with no
visible hand contains all zeros; a frame with only one hand has zeros in the
other hand's block.

Modes
-----
* ``--mode segment`` (default) -- the IDLE/RECORDING state machine described
  above. Use this for sign-repetition videos.
* ``--mode windows`` -- skip the state machine entirely and chop the whole
  video into fixed-length windows spaced by ``--stride`` frames. Use this for
  the *idle / background* class so the LSTM has examples of "the user is not
  signing anything right now" (hands resting, fidgeting, out of frame, etc.).

Usage
-----
    python extract_dataset.py --video hello_60_reps.mp4 --label hello
    python extract_dataset.py --video data.mp4 --label thanks --output dataset \
        --seq-len 30 --patience 5

    # idle / background class:
    python extract_dataset.py --video idle.mp4 --label idle \
        --output dataset --mode windows --stride 15
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

NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3                                  # x, y, z
HAND_VECTOR_SIZE = NUM_LANDMARKS * COORDS_PER_LANDMARK   # 63
FRAME_VECTOR_SIZE = 2 * HAND_VECTOR_SIZE                 # 126 (Left | Right)

# Default location of the Tasks API model file. Override with --model.
_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"
)

# Standard 21-point hand connection graph (matches mp.solutions.hands.HAND_CONNECTIONS).
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm
)


def landmarks_to_frame_vector(result) -> np.ndarray:
    """Flatten a Tasks-API HandLandmarker result to a fixed-length ``(126,)`` vector.

    Layout: ``[Left 21*(x,y,z) | Right 21*(x,y,z)]``. Missing hands -> zeros.
    """
    vec = np.zeros(FRAME_VECTOR_SIZE, dtype=np.float32)
    if not result.hand_landmarks:
        return vec
    handedness = result.handedness or []

    for i, lm_list in enumerate(result.hand_landmarks):
        # Tasks API returns handedness from the camera's POV. For a non-mirrored
        # video this matches MediaPipe's "Left"/"Right" convention.
        label = "Left"
        if i < len(handedness) and handedness[i]:
            label = handedness[i][0].category_name  # "Left" or "Right"
        offset = 0 if label == "Left" else HAND_VECTOR_SIZE
        flat = np.array(
            [[lm.x, lm.y, lm.z] for lm in lm_list],
            dtype=np.float32,
        ).flatten()
        if flat.size == HAND_VECTOR_SIZE:
            vec[offset : offset + HAND_VECTOR_SIZE] = flat
    return vec


def standardize_sequence(frames: list[np.ndarray], seq_len: int) -> np.ndarray:
    """Zero-pad (right) or truncate (keep first N) to exactly ``seq_len`` rows."""
    if not frames:
        return np.zeros((seq_len, FRAME_VECTOR_SIZE), dtype=np.float32)

    arr = np.stack(frames, axis=0).astype(np.float32)
    n = arr.shape[0]
    if n >= seq_len:
        return arr[:seq_len]
    pad = np.zeros((seq_len - n, FRAME_VECTOR_SIZE), dtype=np.float32)
    return np.concatenate([arr, pad], axis=0)


def next_sequence_index(out_dir: Path) -> int:
    """Resume numbering from existing ``<n>.npy`` files in ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [int(p.stem) for p in out_dir.glob("*.npy") if p.stem.isdigit()]
    return (max(existing) + 1) if existing else 0


def draw_hand_landmarks(
    frame_bgr: np.ndarray,
    result,
) -> None:
    """Lightweight replacement for the legacy ``mp.solutions.drawing_utils``."""
    if not result.hand_landmarks:
        return
    h, w = frame_bgr.shape[:2]
    for lm_list in result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame_bgr, pts[a], pts[b], (0, 255, 0), 2, cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(frame_bgr, (x, y), 3, (0, 0, 255), -1, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-extract sign language repetitions from a video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Path to input video file.")
    parser.add_argument("--label", required=True, help="Class label, e.g. 'hello'.")
    parser.add_argument("--output", default="dataset", help="Root output directory.")
    parser.add_argument(
        "--mode",
        choices=("segment", "windows"),
        default="segment",
        help=(
            "segment: IDLE/RECORDING state machine, one .npy per sign rep "
            "(use for sign repetitions). "
            "windows: chop the whole video into fixed-length windows with "
            "--stride spacing (use for the 'idle' / background class)."
        ),
    )
    parser.add_argument("--seq-len", type=int, default=30,
                        help="Frames per saved sequence (pad/truncate target).")
    parser.add_argument("--patience", type=int, default=5,
                        help="Consecutive missing-hand frames before ending a rep.")
    parser.add_argument("--min-frames", type=int, default=5,
                        help="Discard reps shorter than this many real frames.")
    parser.add_argument(
        "--stride",
        type=int,
        default=15,
        help="windows mode only: step between windows (in frames).",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=200,
        help="windows mode only: cap the number of windows saved per video.",
    )
    parser.add_argument("--max-hands", type=int, default=2,
                        help="MediaPipe num_hands.")
    parser.add_argument("--detection-confidence", type=float, default=0.5,
                        help="MediaPipe min_hand_detection_confidence.")
    parser.add_argument("--presence-confidence", type=float, default=0.5,
                        help="MediaPipe min_hand_presence_confidence.")
    parser.add_argument("--tracking-confidence", type=float, default=0.5,
                        help="MediaPipe min_tracking_confidence.")
    parser.add_argument("--model", default=str(_DEFAULT_MODEL_PATH),
                        help="Path to hand_landmarker.task (Tasks API model).")
    parser.add_argument("--no-display", action="store_true",
                        help="Don't open the cv2 preview window (headless).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"Error: video file not found: {video_path}", file=sys.stderr)
        return 1

    model_path = Path(args.model)
    if not model_path.is_file():
        print(
            f"Error: hand_landmarker.task not found at {model_path}.\n"
            "Download it from "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task",
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.output) / args.label
    next_idx = next_sequence_index(out_dir)
    print(f"Saving sequences to {out_dir}/  (starting index: {next_idx})")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: could not open video: {video_path}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1e-3:
        fps = 30.0
    ms_per_frame = 1000.0 / fps

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=args.max_hands,
        min_hand_detection_confidence=args.detection_confidence,
        min_hand_presence_confidence=args.presence_confidence,
        min_tracking_confidence=args.tracking_confidence,
    )

    saved = 0
    state = "IDLE"                     # "IDLE" or "RECORDING" (segment mode)
    current_seq: list[np.ndarray] = []
    missing_streak = 0                 # consecutive frames with no hands
    all_frames: list[np.ndarray] = []  # used by windows mode

    def flush_sequence(trailing_missing: int) -> None:
        """Save the current buffer as a standardized .npy if it's long enough."""
        nonlocal saved, next_idx
        captured = (
            current_seq[:-trailing_missing] if trailing_missing else list(current_seq)
        )
        real_frames = len(captured)
        if real_frames < args.min_frames:
            return
        seq_arr = standardize_sequence(captured, args.seq_len)
        out_path = out_dir / f"{next_idx}.npy"
        np.save(out_path, seq_arr)
        print(
            f"  saved {out_path}  "
            f"(real frames: {real_frames}, shape: {seq_arr.shape})"
        )
        saved += 1
        next_idx += 1

    def save_window(window_frames: list[np.ndarray]) -> None:
        """Windows mode: persist a fixed-length window verbatim."""
        nonlocal saved, next_idx
        seq_arr = standardize_sequence(window_frames, args.seq_len)
        out_path = out_dir / f"{next_idx}.npy"
        np.save(out_path, seq_arr)
        if saved % 25 == 0:
            print(f"  saved {out_path}  (shape: {seq_arr.shape})")
        saved += 1
        next_idx += 1

    aborted = False
    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * ms_per_frame)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            has_hand = bool(result.hand_landmarks)
            frame_vec = landmarks_to_frame_vector(result)

            if args.mode == "segment":
                # ---- State machine -----------------------------------------
                if state == "IDLE":
                    if has_hand:
                        state = "RECORDING"
                        current_seq = [frame_vec]
                        missing_streak = 0
                else:  # RECORDING
                    # Always append: brief gaps inside the patience window
                    # become zero rows so the temporal alignment stays intact.
                    current_seq.append(frame_vec)
                    if has_hand:
                        missing_streak = 0
                    else:
                        missing_streak += 1
                        if missing_streak >= args.patience:
                            flush_sequence(trailing_missing=missing_streak)
                            state = "IDLE"
                            current_seq = []
                            missing_streak = 0
            else:  # windows mode: just stash every frame for chunking later.
                all_frames.append(frame_vec)
                state = "WINDOWS"

            # ---- UI ----------------------------------------------------------
            if not args.no_display:
                draw_hand_landmarks(frame_bgr, result)

                color = (0, 0, 255) if state == "RECORDING" else (0, 200, 0)
                h, w = frame_bgr.shape[:2]
                cv2.rectangle(frame_bgr, (0, 0), (w, 34), (0, 0, 0), -1)
                buf_len = (
                    len(current_seq) if args.mode == "segment" else len(all_frames)
                )
                cv2.putText(
                    frame_bgr,
                    f"Mode: {args.mode}  |  State: {state}  |  Label: {args.label}  |  "
                    f"Saved: {saved}  |  Buf: {buf_len}",
                    (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("extract_dataset", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Aborted by user (q).")
                    aborted = True
                    break

            frame_idx += 1

        # End-of-video flush: if the file ended mid-recording, save what we have.
        if args.mode == "segment":
            if not aborted and state == "RECORDING":
                flush_sequence(trailing_missing=missing_streak)
        else:
            # Windows mode: chunk the stashed frames into seq_len windows
            # spaced by --stride.
            if not aborted and len(all_frames) >= args.seq_len:
                start = 0
                while start + args.seq_len <= len(all_frames):
                    if saved >= args.max_windows:
                        print(
                            f"  reached --max-windows={args.max_windows}, stopping."
                        )
                        break
                    window = all_frames[start : start + args.seq_len]
                    save_window(window)
                    start += args.stride
                print(
                    f"  windows mode: {saved} window(s) cut from "
                    f"{len(all_frames)} frames (stride={args.stride})."
                )

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved {saved} sequence(s) for label '{args.label}' in {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
