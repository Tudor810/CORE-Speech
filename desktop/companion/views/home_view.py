"""Home dashboard: at-a-glance status + quick navigation."""

from __future__ import annotations

import customtkinter as ctk

from .. import theme
from ..runtime import app_state
from ..services import voice as voice_service

EMOTION_LABELS: dict[str, str] = {
    "neutral": "Neutral",
    "calm": "Calm",
    "friendly": "Friendly",
    "excited": "Excited",
    "serious": "Serious",
    "urgent": "Urgent",
}


class HomeView(ctk.CTkFrame):
    """Landing page with status cards + 'Start a live session' CTA."""

    def __init__(self, master, *, navigate) -> None:
        super().__init__(master, fg_color="transparent")
        self._navigate = navigate

        self._build_header()
        self._build_status_cards()
        self._build_cta()
        self._build_features()
        self.refresh()

    # --------------------------- layout -------------------------------------
    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=theme.PAD_L, pady=(theme.PAD_L, 0))

        ctk.CTkLabel(
            header,
            text="Welcome back",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=theme.TEXT,
            anchor="w",
        ).pack(fill="x")

        ctk.CTkLabel(
            header,
            text="A quick look at your glasses and how they sound right now.",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(size=13),
            anchor="w",
        ).pack(fill="x", pady=(2, 0))

    def _build_status_cards(self) -> None:
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=theme.PAD_L, pady=theme.PAD_L)
        for i in range(3):
            row.grid_columnconfigure(i, weight=1, uniform="status")

        self._voice_card = self._make_card(
            row, "Voice", "Loading…", action="Change voice →", action_target="voice"
        )
        self._voice_card.grid(row=0, column=0, sticky="nsew", padx=(0, theme.PAD_S))

        self._mood_card = self._make_card(
            row, "Mood", "Loading…", action="Adjust mood →", action_target="emotion"
        )
        self._mood_card.grid(
            row=0, column=1, sticky="nsew", padx=(theme.PAD_S, theme.PAD_S)
        )

        self._sign_card = self._make_card(
            row,
            "Signs",
            "Loading…",
            action="Teach a new sign →",
            action_target="training",
        )
        self._sign_card.grid(row=0, column=2, sticky="nsew", padx=(theme.PAD_S, 0))

    def _build_cta(self) -> None:
        cta = ctk.CTkFrame(
            self, fg_color=theme.PANEL, corner_radius=theme.CARD_RADIUS
        )
        cta.pack(fill="x", padx=theme.PAD_L, pady=(0, theme.PAD_L))

        text_box = ctk.CTkFrame(cta, fg_color="transparent")
        text_box.pack(side="left", fill="both", expand=True, padx=theme.PAD_L, pady=theme.PAD_L)

        ctk.CTkLabel(
            text_box,
            text="Try it now",
            text_color=theme.TEXT,
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w",
        ).pack(fill="x")
        ctk.CTkLabel(
            text_box,
            text="Open the live session, sign in front of the camera, and hear it spoken back.",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(size=12),
            anchor="w",
            justify="left",
            wraplength=480,
        ).pack(fill="x", pady=(theme.PAD_S, 0))

        ctk.CTkButton(
            cta,
            text="● Start a live session",
            fg_color=theme.ACCENT,
            hover_color=theme.ACCENT_HOVER,
            corner_radius=theme.BUTTON_RADIUS,
            command=lambda: self._navigate("live"),
        ).pack(side="right", padx=theme.PAD_L, pady=theme.PAD_L)

    def _build_features(self) -> None:
        card = ctk.CTkFrame(
            self, fg_color=theme.PANEL, corner_radius=theme.CARD_RADIUS
        )
        card.pack(fill="x", padx=theme.PAD_L, pady=(0, theme.PAD_L))

        ctk.CTkLabel(
            card,
            text="What your glasses can do",
            text_color=theme.TEXT,
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=theme.PAD_L, pady=(theme.PAD_L, theme.PAD_S))

        items = [
            (
                "Sign → speech.",
                "Sign in front of the camera and your glasses speak it out loud.",
            ),
            (
                "Speech → captions.",
                "When someone talks to you, the words show up on your lenses.",
            ),
            (
                "Teach new signs.",
                "Show a sign a few times and your glasses will learn it.",
            ),
        ]
        for title, body in items:
            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(fill="x", padx=theme.PAD_L, pady=(0, theme.PAD_S))

            ctk.CTkLabel(
                row,
                text=title,
                text_color=theme.ACCENT,
                font=ctk.CTkFont(size=12, weight="bold"),
                anchor="w",
            ).pack(side="left")
            ctk.CTkLabel(
                row,
                text=" " + body,
                text_color=theme.TEXT_MUTED,
                font=ctk.CTkFont(size=12),
                anchor="w",
                justify="left",
                wraplength=520,
            ).pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(card, text="", height=theme.PAD_S).pack()

    # --------------------------- helpers ------------------------------------
    def _make_card(
        self,
        parent,
        title: str,
        value: str,
        *,
        action: str,
        action_target: str,
    ) -> ctk.CTkFrame:
        card = ctk.CTkFrame(
            parent, fg_color=theme.PANEL, corner_radius=theme.CARD_RADIUS
        )
        ctk.CTkLabel(
            card,
            text=title.upper(),
            text_color=theme.TEXT_DIM,
            font=ctk.CTkFont(size=10, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=theme.PAD_L, pady=(theme.PAD_L, 0))

        value_label = ctk.CTkLabel(
            card,
            text=value,
            text_color=theme.TEXT,
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w",
            justify="left",
        )
        value_label.pack(fill="x", padx=theme.PAD_L, pady=(theme.PAD_S, theme.PAD_S))

        action_btn = ctk.CTkButton(
            card,
            text=action,
            fg_color="transparent",
            text_color=theme.ACCENT,
            hover_color=theme.PANEL_ALT,
            anchor="w",
            command=lambda: self._navigate(action_target),
        )
        action_btn.pack(fill="x", padx=theme.PAD_L - 4, pady=(0, theme.PAD_L))

        # Stash the value label so refresh() can update it later.
        card._value_label = value_label  # type: ignore[attr-defined]
        return card

    # --------------------------- refresh ------------------------------------
    def on_show(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        snap = voice_service.snapshot(refresh_speakers=False)
        active_voice_label = "Default"
        if snap.active_voice_id != snap.default_voice_id:
            for v in snap.custom_voices:
                if v.voice_id == snap.active_voice_id:
                    active_voice_label = v.name
                    break

        self._voice_card._value_label.configure(text=active_voice_label)  # type: ignore[attr-defined]

        state = app_state().data
        mood_text = EMOTION_LABELS.get(state.emotion_preset, state.emotion_preset)
        mood_text = f"{mood_text} · {int(state.emotion_intensity * 100)}%"
        self._mood_card._value_label.configure(text=mood_text)  # type: ignore[attr-defined]

        from ..services.training import list_sample_counts

        try:
            counts = list_sample_counts()
        except Exception:
            counts = {}
        n_labels = len(counts)
        n_samples = sum(counts.values())
        if n_labels == 0:
            sign_text = "No signs yet"
        else:
            sign_text = f"{n_labels} sign{'s' if n_labels != 1 else ''} · {n_samples} clips"
        self._sign_card._value_label.configure(text=sign_text)  # type: ignore[attr-defined]
