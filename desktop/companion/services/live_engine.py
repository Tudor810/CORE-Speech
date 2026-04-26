"""In-process sign-to-speech engine.

This is a port of the main loop in ``server/scripts/live_predict.py``,
restructured so the GUI can drive it instead of an OpenCV ``imshow``
window. The state machine, motion-pause segmentation, sentence-timeout
dispatch and the LLM + TTS hand-off are all preserved bit-for-bit so
behaviour matches what users already trust.

Architecture:

* :class:`LiveEngine` runs a worker thread that pulls frames from the
  shared :class:`~app.services.camera.Camera`, runs MediaPipe Hands and
  the trained LSTM, and emits :class:`LiveEvent` records via a callback.
* Sentence dispatch (LLM rewrite + TTS playback) happens in a second
  thread so a slow Ollama call never holds up the live preview.
* The GUI thread receives events via ``call_in_ui`` and updates labels
  / overlays.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# IMPORTANT: ``..runtime`` first so ``server/`` is on ``sys.path`` before
# we resolve ``app.*``.
from .. import runtime as _runtime  # noqa: F401
from app.config import settings  # noqa: E402
from app.ml.lstm import load_checkpoint  # noqa: E402

from .camera import Camera

NUM_LANDMARKS = 21
COORDS = 3
HAND_VECTOR_SIZE = NUM_LANDMARKS * COORDS         # 63
FRAME_VECTOR_SIZE = 2 * HAND_VECTOR_SIZE          # 126

LEFT_BLOCK = slice(0, HAND_VECTOR_SIZE)
RIGHT_BLOCK = slice(HAND_VECTOR_SIZE, FRAME_VECTOR_SIZE)


# ---------------------------------------------------------------------------
# Public event types
# ---------------------------------------------------------------------------
@dataclass
class LandmarkPayload:
    """Per-frame payload pushed to the GUI for the on-screen overlay."""

    hands: list[list[list[float]]]
    handedness: list[str]
    has_hand: bool


@dataclass
class LiveEvent:
    """All possible events the engine can emit, multiplexed on ``kind``."""

    kind: str  # "frame" | "gesture" | "phrase" | "translation" | "audio_end" | "status" | "error"
    frame: Optional[np.ndarray] = None
    landmarks: Optional[LandmarkPayload] = None
    label: Optional[str] = None
    confidence: float = 0.0
    tokens: list[str] = field(default_factory=list)
    text: Optional[str] = None
    msg: Optional[str] = None
    fps: float = 0.0
    is_idle: bool = False
    state: str = "IDLE"


@dataclass
class LiveConfig:
    """Tunables surfaced to the GUI (mirrors live_predict.py CLI flags)."""

    seq_len: int = settings.lstm_window_size
    threshold: float = settings.lstm_confidence_threshold
    stride: int = 5
    min_frames: int = 5
    patience: int = 5
    motion_threshold: float = 0.004
    motion_smooth: int = 5
    pause_frames: int = 6
    stable_time: float = 0.2
    sentence_timeout: float = 1.0
    cooldown: float = 1.5
    max_sentence: int = 12
    idle_label: str = "idle"
    allow_repeats: bool = False


# ---------------------------------------------------------------------------
# Helpers (lifted from live_predict.py)
# ---------------------------------------------------------------------------
def _landmarks_to_frame_vector(result) -> tuple[np.ndarray, LandmarkPayload]:
    vec = np.zeros(FRAME_VECTOR_SIZE, dtype=np.float32)
    hands_payload: list[list[list[float]]] = []
    handedness_payload: list[str] = []

    if not result.hand_landmarks:
        return vec, LandmarkPayload(hands_payload, handedness_payload, False)

    handedness = result.handedness or []
    for i, lm_list in enumerate(result.hand_landmarks):
        label = "Left"
        if i < len(handedness) and handedness[i]:
            label = handedness[i][0].category_name
        offset = 0 if label == "Left" else HAND_VECTOR_SIZE
        flat = np.array(
            [[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32,
        ).flatten()
        if flat.size == HAND_VECTOR_SIZE:
            vec[offset : offset + HAND_VECTOR_SIZE] = flat
        hands_payload.append([[lm.x, lm.y, lm.z] for lm in lm_list])
        handedness_payload.append(label)

    return vec, LandmarkPayload(hands_payload, handedness_payload, True)


def _to_dominant_hand_seq(seq_126: np.ndarray) -> np.ndarray:
    out = np.zeros((seq_126.shape[0], HAND_VECTOR_SIZE), dtype=np.float32)
    left = seq_126[:, LEFT_BLOCK]
    right = seq_126[:, RIGHT_BLOCK]
    has_left = np.any(left != 0.0, axis=1)
    has_right = np.any(right != 0.0, axis=1)
    out[has_left] = left[has_left]
    use_right = (~has_left) & has_right
    out[use_right] = right[use_right]
    return out


def _standardize_sequence(frames: list[np.ndarray], seq_len: int) -> np.ndarray:
    if not frames:
        return np.zeros((seq_len, FRAME_VECTOR_SIZE), dtype=np.float32)
    arr = np.stack(frames, axis=0).astype(np.float32)
    n = arr.shape[0]
    if n >= seq_len:
        return arr[:seq_len]
    pad = np.zeros((seq_len - n, FRAME_VECTOR_SIZE), dtype=np.float32)
    return np.concatenate([arr, pad], axis=0)


def _hand_motion_energy(curr: np.ndarray, prev: np.ndarray) -> float:
    motions: list[float] = []
    for blk in (LEFT_BLOCK, RIGHT_BLOCK):
        a, b = curr[blk], prev[blk]
        if np.any(a) and np.any(b):
            diff = a.reshape(NUM_LANDMARKS, COORDS) - b.reshape(NUM_LANDMARKS, COORDS)
            motions.append(float(np.linalg.norm(diff, axis=1).mean()))
    return float(np.mean(motions)) if motions else 0.0


# ---------------------------------------------------------------------------
# Predictor (loads the trained LSTM from settings.lstm_checkpoint_path)
# ---------------------------------------------------------------------------
class Predictor:
    def __init__(self) -> None:
        model, labels = load_checkpoint()
        if model is None or not labels:
            raise RuntimeError(
                f"No trained LSTM checkpoint at {settings.lstm_checkpoint_path}. "
                "Open the Teach signs tab and train at least two labels first."
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.labels = list(labels)

    @torch.no_grad()
    def predict(self, seq_126: np.ndarray) -> tuple[str, float]:
        if self.model.input_size == HAND_VECTOR_SIZE:
            seq = _to_dominant_hand_seq(seq_126)
        else:
            seq = seq_126
        x = torch.from_numpy(seq).float().unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])


# ---------------------------------------------------------------------------
# LiveEngine
# ---------------------------------------------------------------------------
EventCallback = Callable[[LiveEvent], None]


class LiveEngine:
    """Per-frame sign engine + sentence dispatch."""

    def __init__(
        self,
        camera: Camera,
        on_event: EventCallback,
        *,
        config: Optional[LiveConfig] = None,
        emotion: str = "neutral",
        intensity: float = 0.5,
        do_llm: bool = True,
        tts_worker=None,  # app.services.tts_worker.TtsWorker
    ) -> None:
        self._camera = camera
        self._on_event = on_event
        self._config = config or LiveConfig()
        self._emotion = emotion
        self._intensity = intensity
        self._do_llm = do_llm
        self._tts = tts_worker

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._predictor: Optional[Predictor] = None
        self._dispatch_q: "queue.Queue[tuple[str, object]]" = queue.Queue()

    # --------------------------- knobs --------------------------------------
    def set_emotion(self, emotion: str, intensity: float) -> None:
        self._emotion = emotion
        self._intensity = max(0.0, min(1.0, intensity))

    def set_do_llm(self, do_llm: bool) -> None:
        self._do_llm = do_llm

    def set_config(self, config: LiveConfig) -> None:
        self._config = config

    @property
    def labels(self) -> list[str]:
        return list(self._predictor.labels) if self._predictor else []

    # --------------------------- lifecycle ----------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        # Build the predictor on the calling thread so model-load errors
        # surface synchronously to the view (instead of being swallowed in
        # the worker thread).
        self._predictor = Predictor()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="live-engine", daemon=True
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

    # ---------------------------- main loop --------------------------------
    def _run(self) -> None:
        cfg = self._config
        model_path = settings.mediapipe_model_path
        if not Path(model_path).is_file():
            self._emit(
                LiveEvent(
                    kind="error",
                    msg=(
                        f"Hand model not found at {model_path}. Download "
                        "hand_landmarker.task into server/models/."
                    ),
                )
            )
            return

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=settings.mediapipe_max_hands,
            min_hand_detection_confidence=settings.mediapipe_min_detection_confidence,
            min_hand_presence_confidence=settings.mediapipe_min_presence_confidence,
            min_tracking_confidence=settings.mediapipe_min_tracking_confidence,
        )

        sliding_buf: deque[np.ndarray] = deque(maxlen=cfg.seq_len)
        motion_buf: deque[float] = deque(maxlen=max(1, cfg.motion_smooth))
        prev_frame_vec = np.zeros(FRAME_VECTOR_SIZE, dtype=np.float32)
        sentence: list[str] = []

        candidate_label: Optional[str] = None
        candidate_first_seen = 0.0
        candidate_best_conf = 0.0
        candidate_committed = False

        last_label = ""
        last_conf = 0.0
        last_was_idle = False
        last_commit_time = 0.0
        cooldown_until = 0.0
        missing_streak = 0
        low_motion_streak = 0
        frames_since_last_pred = 0

        pending_dispatch = False

        last_seen_frame_id = -1
        ts_ms = 0
        fps_t0 = time.monotonic()
        fps_count = 0
        fps_value = 0.0

        self._emit(LiveEvent(kind="status", msg="ready"))

        with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
            while not self._stop.is_set():
                # --- pull the latest camera frame ----------------------------
                frame_id = self._camera.frame_count
                if frame_id == last_seen_frame_id or not self._camera.is_running:
                    time.sleep(0.005)
                    continue
                last_seen_frame_id = frame_id
                frame_bgr = self._camera.read_latest()
                if frame_bgr is None:
                    continue

                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms += 1
                result = landmarker.detect_for_video(mp_image, ts_ms)

                has_hand = bool(result.hand_landmarks)
                frame_vec, landmarks = _landmarks_to_frame_vector(result)
                now_s = time.monotonic()
                in_cooldown = now_s < cooldown_until

                # --- motion-energy bookkeeping (sliding mode) ---------------
                inst_motion = _hand_motion_energy(frame_vec, prev_frame_vec)
                motion_buf.append(inst_motion)
                smoothed_motion = float(np.mean(motion_buf)) if motion_buf else 0.0
                prev_frame_vec = frame_vec

                if has_hand:
                    sliding_buf.append(frame_vec)
                    missing_streak = 0
                    state = "RECORDING"
                    if smoothed_motion >= cfg.motion_threshold:
                        low_motion_streak = 0
                    else:
                        low_motion_streak += 1
                else:
                    missing_streak += 1
                    low_motion_streak = 0
                    state = "IDLE"

                # --- live prediction every --stride frames -------------------
                frames_since_last_pred += 1
                if (
                    has_hand
                    and len(sliding_buf) >= max(cfg.min_frames, 1)
                    and frames_since_last_pred >= cfg.stride
                    and self._predictor is not None
                ):
                    frames_since_last_pred = 0
                    seq = _standardize_sequence(list(sliding_buf), cfg.seq_len)
                    label, conf = self._predictor.predict(seq)
                    is_idle = label == cfg.idle_label
                    if is_idle:
                        last_label, last_conf = "", 0.0
                        last_was_idle = True
                        candidate_label = None
                        candidate_committed = False
                    else:
                        last_label, last_conf = label, conf
                        last_was_idle = False
                        if conf >= cfg.threshold:
                            if candidate_label != label:
                                candidate_label = label
                                candidate_first_seen = now_s
                                candidate_best_conf = conf
                                candidate_committed = False
                            else:
                                if conf > candidate_best_conf:
                                    candidate_best_conf = conf
                            if (
                                not candidate_committed
                                and (now_s - candidate_first_seen) >= cfg.stable_time
                                and not in_cooldown
                            ):
                                self._commit_sign(
                                    sentence,
                                    candidate_label,
                                    candidate_best_conf,
                                    cfg,
                                )
                                last_commit_time = time.monotonic()
                                candidate_committed = True
                                cooldown_until = now_s + cfg.cooldown

                # --- buffer reset on hand-gone or motion-pause -----------------
                end_by_missing = missing_streak >= cfg.patience and sliding_buf
                end_by_pause = (
                    has_hand and low_motion_streak >= cfg.pause_frames and sliding_buf
                )
                if end_by_missing or end_by_pause:
                    sliding_buf.clear()
                    candidate_label = None
                    candidate_committed = False
                    low_motion_streak = 0
                    frames_since_last_pred = 0
                    last_label, last_conf = "", 0.0
                    if end_by_missing:
                        last_was_idle = False

                # --- sentence dispatch (LLM + TTS) ----------------------------
                if (
                    sentence
                    and not pending_dispatch
                    and last_commit_time > 0.0
                    and (now_s - last_commit_time) >= cfg.sentence_timeout
                ):
                    pending_dispatch = True
                    tokens_to_send = list(sentence)
                    sentence.clear()
                    last_commit_time = 0.0
                    self._emit(
                        LiveEvent(kind="phrase", tokens=tokens_to_send)
                    )
                    threading.Thread(
                        target=self._sentence_worker,
                        args=(tokens_to_send,),
                        daemon=True,
                    ).start()

                # --- drain dispatch results ----------------------------------
                while True:
                    try:
                        kind, payload = self._dispatch_q.get_nowait()
                    except queue.Empty:
                        break
                    if kind == "text":
                        self._emit(LiveEvent(kind="translation", text=str(payload)))
                    elif kind == "audio_end":
                        self._emit(LiveEvent(kind="audio_end"))
                    elif kind == "error":
                        self._emit(LiveEvent(kind="error", msg=str(payload)))
                    elif kind == "done":
                        pending_dispatch = False

                # --- emit a frame event for the GUI overlay -------------------
                fps_count += 1
                if (now_s - fps_t0) >= 0.5:
                    fps_value = fps_count / (now_s - fps_t0)
                    fps_t0 = now_s
                    fps_count = 0

                self._emit(
                    LiveEvent(
                        kind="frame",
                        frame=frame_bgr,
                        landmarks=landmarks,
                        label=last_label,
                        confidence=last_conf,
                        tokens=list(sentence),
                        is_idle=last_was_idle and has_hand,
                        state=state,
                        fps=fps_value,
                    )
                )

    # --------------------------- helpers ------------------------------------
    def _emit(self, event: LiveEvent) -> None:
        try:
            self._on_event(event)
        except Exception:
            # The view layer is allowed to be picky; engine survival > UI errors.
            pass

    def _commit_sign(
        self,
        sentence: list[str],
        label: str,
        conf: float,
        cfg: LiveConfig,
    ) -> None:
        if not cfg.allow_repeats and sentence and sentence[-1] == label:
            return
        sentence.append(label)
        if len(sentence) > cfg.max_sentence:
            del sentence[: len(sentence) - cfg.max_sentence]
        self._emit(LiveEvent(kind="gesture", label=label, confidence=conf))

    def _sentence_worker(self, tokens: list[str]) -> None:
        """Run the LLM rewrite (optional) + queue text to the TTS worker."""
        natural_text = " ".join(tokens)
        try:
            if self._do_llm:
                try:
                    natural_text = (
                        self._collect_translation_blocking(tokens) or natural_text
                    )
                except Exception as exc:
                    self._dispatch_q.put(("error", f"LLM: {exc}"))
            self._dispatch_q.put(("text", natural_text))
            if self._tts is not None and self._tts.enabled and natural_text:
                self._tts.say(
                    natural_text,
                    emotion=self._emotion,
                    intensity=self._intensity,
                    on_done=lambda: self._dispatch_q.put(("audio_end", None)),
                )
            else:
                self._dispatch_q.put(("audio_end", None))
        finally:
            self._dispatch_q.put(("done", None))

    def _collect_translation_blocking(self, tokens: list[str]) -> str:
        """Drive ``app.llm.local_client.translate_signs`` synchronously."""
        from app.llm.local_client import translate_signs

        async def _go() -> str:
            pieces: list[str] = []
            async for chunk in translate_signs(
                tokens, emotion=self._emotion, intensity=self._intensity
            ):
                pieces.append(chunk)
            return "".join(pieces).strip()

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_go())
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()
