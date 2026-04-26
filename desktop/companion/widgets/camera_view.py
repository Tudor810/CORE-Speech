"""Video display widget with hand-landmark overlay.

Shows a BGR frame from the camera and (optionally) draws the same
21-point hand topology as ``server/scripts/live_predict.py``. The
underlying widget is a :class:`customtkinter.CTkLabel`; we just keep
swapping its ``image`` reference with a new :class:`CTkImage`.
"""

from __future__ import annotations

from typing import Optional

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageDraw

from .. import theme

# Same 21-point topology as live_predict.py / TrainingStudio.tsx.
HAND_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


class CameraView(ctk.CTkFrame):
    """Resizable image surface with optional hand overlay."""

    def __init__(
        self,
        master,
        *,
        width: int = 640,
        height: int = 480,
        placeholder: str = "Camera off",
    ) -> None:
        super().__init__(
            master,
            fg_color="#000000",
            corner_radius=theme.CARD_RADIUS,
            width=width,
            height=height,
        )
        self._target_w = width
        self._target_h = height
        self._placeholder_text = placeholder

        self._label = ctk.CTkLabel(self, text="", fg_color="transparent")
        self._label.pack(expand=True, fill="both", padx=2, pady=2)

        self._placeholder = ctk.CTkLabel(
            self,
            text=placeholder,
            text_color=theme.TEXT_DIM,
            font=ctk.CTkFont(size=13),
            fg_color="transparent",
        )
        self._placeholder.place(relx=0.5, rely=0.5, anchor="center")

        self._current_ctkimg: Optional[ctk.CTkImage] = None

    # --------------------------- public API ---------------------------------
    def set_target_size(self, width: int, height: int) -> None:
        self._target_w = max(1, int(width))
        self._target_h = max(1, int(height))

    def show_placeholder(self, text: Optional[str] = None) -> None:
        if text is not None:
            self._placeholder.configure(text=text)
        self._placeholder.place(relx=0.5, rely=0.5, anchor="center")
        self._label.configure(image="")
        self._current_ctkimg = None

    def render(
        self,
        bgr: np.ndarray,
        *,
        landmarks: Optional[list[list[list[float]]]] = None,
    ) -> None:
        """Resize ``bgr``, optionally draw landmarks, and push it to the label."""
        if bgr is None or bgr.size == 0:
            return

        h, w = bgr.shape[:2]
        if w == 0 or h == 0:
            return

        # Fit-to-frame (preserve aspect ratio).
        target_w = max(self._target_w, 1)
        target_h = max(self._target_h, 1)
        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        if (new_w, new_h) != (w, h):
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        if landmarks:
            draw = ImageDraw.Draw(img)
            self._draw_landmarks(draw, landmarks, new_w, new_h)

        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(new_w, new_h))
        self._label.configure(image=ctk_img)
        self._current_ctkimg = ctk_img

        # Hide the placeholder once we have a real frame.
        self._placeholder.place_forget()

    # --------------------------- helpers ------------------------------------
    @staticmethod
    def _draw_landmarks(
        draw: ImageDraw.ImageDraw,
        hands: list[list[list[float]]],
        width: int,
        height: int,
    ) -> None:
        line_color = (34, 255, 85)
        joint_color = (255, 59, 59)
        line_w = max(2, width // 240)
        joint_r = max(3, width // 180)

        for hand in hands:
            if not hand or len(hand) < 21:
                continue
            for a, b in HAND_EDGES:
                if a >= len(hand) or b >= len(hand):
                    continue
                ax, ay = hand[a][0] * width, hand[a][1] * height
                bx, by = hand[b][0] * width, hand[b][1] * height
                draw.line((ax, ay, bx, by), fill=line_color, width=line_w)
            for px, py, *_ in hand:
                cx, cy = px * width, py * height
                draw.ellipse(
                    (cx - joint_r, cy - joint_r, cx + joint_r, cy + joint_r),
                    fill=joint_color,
                )
