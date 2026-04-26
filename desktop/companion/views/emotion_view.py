"""Mood / emotion picker."""

from __future__ import annotations

import customtkinter as ctk

from .. import theme
from ..runtime import app_state, run_coro_blocking

PRESETS: list[tuple[str, str, str]] = [
    ("neutral", "Neutral", "Balanced, default delivery."),
    ("calm", "Calm", "Soft and steady."),
    ("friendly", "Friendly", "Warm and inviting."),
    ("excited", "Excited", "Energetic, upbeat."),
    ("serious", "Serious", "Measured and direct."),
    ("urgent", "Urgent", "High intensity, attention-getting."),
]


class EmotionView(ctk.CTkFrame):
    """Pick an emotional tone preset + intensity."""

    def __init__(self, master) -> None:
        super().__init__(master, fg_color="transparent")

        self._build_header()

        self._save_status = ctk.CTkLabel(
            self,
            text="",
            text_color=theme.OK,
            font=ctk.CTkFont(size=12),
            anchor="w",
        )
        self._save_status.pack(fill="x", padx=theme.PAD_L)

        self._preset_buttons: dict[str, ctk.CTkButton] = {}
        self._build_presets()
        self._build_intensity()
        self.refresh()

    # ---------------------------- layout ------------------------------------
    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=theme.PAD_L, pady=(theme.PAD_L, 0))
        ctk.CTkLabel(
            header,
            text="Mood",
            text_color=theme.TEXT,
            font=ctk.CTkFont(size=22, weight="bold"),
            anchor="w",
        ).pack(fill="x")
        ctk.CTkLabel(
            header,
            text="Choose how your voice should feel — calm, friendly, urgent, and more.",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(size=13),
            anchor="w",
        ).pack(fill="x", pady=(2, 0))

    def _build_presets(self) -> None:
        card = ctk.CTkFrame(self, fg_color=theme.PANEL, corner_radius=theme.CARD_RADIUS)
        card.pack(fill="x", padx=theme.PAD_L, pady=theme.PAD_L)

        ctk.CTkLabel(
            card,
            text="How should the voice feel?",
            text_color=theme.TEXT,
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=theme.PAD_L, pady=(theme.PAD_M, theme.PAD_S))

        grid = ctk.CTkFrame(card, fg_color="transparent")
        grid.pack(fill="x", padx=theme.PAD_L, pady=(0, theme.PAD_M))

        for col in range(3):
            grid.grid_columnconfigure(col, weight=1, uniform="emo")

        for i, (key, label, hint) in enumerate(PRESETS):
            row, col = divmod(i, 3)
            btn = ctk.CTkButton(
                grid,
                text=f"{label}\n{hint}",
                fg_color=theme.PANEL_ALT,
                hover_color=theme.BORDER,
                text_color=theme.TEXT,
                anchor="w",
                height=58,
                command=lambda k=key: self._pick(k),
            )
            btn.grid(
                row=row,
                column=col,
                padx=theme.PAD_S // 2,
                pady=theme.PAD_S // 2,
                sticky="nsew",
            )
            self._preset_buttons[key] = btn

    def _build_intensity(self) -> None:
        card = ctk.CTkFrame(self, fg_color=theme.PANEL, corner_radius=theme.CARD_RADIUS)
        card.pack(fill="x", padx=theme.PAD_L, pady=(0, theme.PAD_L))

        head = ctk.CTkFrame(card, fg_color="transparent")
        head.pack(fill="x", padx=theme.PAD_L, pady=(theme.PAD_M, theme.PAD_S))

        ctk.CTkLabel(
            head,
            text="How strong",
            text_color=theme.TEXT,
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).pack(side="left")

        self._intensity_label = ctk.CTkLabel(
            head,
            text="50%",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(size=13),
        )
        self._intensity_label.pack(side="right")

        self._intensity_slider = ctk.CTkSlider(
            card,
            from_=0,
            to=1,
            number_of_steps=20,
            command=self._on_slider_drag,
        )
        self._intensity_slider.pack(fill="x", padx=theme.PAD_L, pady=(0, theme.PAD_M))
        self._intensity_slider.bind("<ButtonRelease-1>", lambda _e: self._save())

    # ---------------------------- behaviour ---------------------------------
    def on_show(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        state = app_state().data
        for key, btn in self._preset_buttons.items():
            if key == state.emotion_preset:
                btn.configure(fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER)
            else:
                btn.configure(fg_color=theme.PANEL_ALT, hover_color=theme.BORDER)

        self._intensity_slider.set(state.emotion_intensity)
        self._intensity_label.configure(
            text=f"{int(state.emotion_intensity * 100)}%"
        )

    def _pick(self, key: str) -> None:
        state = app_state()
        state.data.emotion_preset = key  # type: ignore[assignment]
        try:
            run_coro_blocking(state.save(), timeout=10.0)
        except Exception as exc:
            self._save_status.configure(text=f"Couldn't save: {exc}", text_color=theme.ERR)
            return
        self.refresh()
        self._flash_save()

    def _on_slider_drag(self, value) -> None:
        v = float(value)
        self._intensity_label.configure(text=f"{int(v * 100)}%")

    def _save(self) -> None:
        state = app_state()
        state.data.emotion_intensity = float(self._intensity_slider.get())
        try:
            run_coro_blocking(state.save(), timeout=10.0)
        except Exception as exc:
            self._save_status.configure(text=f"Couldn't save: {exc}", text_color=theme.ERR)
            return
        self._flash_save()

    def _flash_save(self) -> None:
        self._save_status.configure(text="Saved.", text_color=theme.OK)
        self.after(1500, lambda: self._save_status.configure(text=""))
