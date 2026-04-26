"""Top-level window: sidebar nav + view stack.

The window owns the long-lived services that views share:

* :class:`Camera` — the single ``cv2.VideoCapture`` handle
* :class:`TtsWorker` — the background OpenVoice / SAPI synth + playback worker

Views are instantiated lazily on first navigation and reused across visits.
"""

from __future__ import annotations

from typing import Optional

import customtkinter as ctk

from . import theme
from .runtime import app_settings, shutdown_runtime
from .services import voice as voice_service
from .services.camera import Camera
from .services.tts_worker import TtsWorker
from .views import EmotionView, HomeView, LiveView, TrainingView, VoiceView


NAV_ITEMS: list[tuple[str, str, str]] = [
    ("home", "Home", "▮▮ Welcome dashboard"),
    ("live", "Live", "● Sign-to-speech"),
    ("voice", "Voice", "♪ Pick / clone voices"),
    ("emotion", "Mood", "✿ Emotional tone"),
    ("training", "Teach signs", "✎ Add new signs"),
]


class MainWindow(ctk.CTk):
    """Application root. Holds shared services + view stack."""

    def __init__(
        self,
        *,
        camera_index: Optional[int] = None,
        tts_disabled: bool = False,
    ) -> None:
        super().__init__()
        self.title("AR Glasses Companion")
        self.geometry("1200x780")
        self.minsize(1000, 680)
        self.configure(fg_color=theme.BG)

        ctk.set_default_color_theme("blue")

        # ------- shared services ---------------------------------------------
        self._camera = Camera()
        self._tts = TtsWorker(disabled=tts_disabled)
        if camera_index is not None:
            self._camera_index = int(camera_index)
        else:
            self._camera_index = int(app_settings().camera_index or 0)

        # Pre-load OpenVoice in the background so the first sentence isn't
        # penalised by model-cold-start.
        try:
            voice_service.warmup_engine_async()
        except Exception:
            pass

        # ------- layout: sidebar | content -----------------------------------
        self.grid_columnconfigure(0, weight=0, minsize=theme.SIDEBAR_WIDTH)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._sidebar = self._build_sidebar()
        self._sidebar.grid(row=0, column=0, sticky="nsew")

        self._content = ctk.CTkFrame(self, fg_color=theme.BG)
        self._content.grid(row=0, column=1, sticky="nsew")
        self._content.grid_rowconfigure(0, weight=1)
        self._content.grid_columnconfigure(0, weight=1)

        self._views: dict[str, ctk.CTkFrame] = {}
        self._current: Optional[str] = None

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.navigate("home")

    # --------------------------- layout -------------------------------------
    def _build_sidebar(self) -> ctk.CTkFrame:
        bar = ctk.CTkFrame(
            self,
            fg_color=theme.PANEL,
            corner_radius=0,
            width=theme.SIDEBAR_WIDTH,
        )
        bar.grid_propagate(False)

        brand = ctk.CTkFrame(bar, fg_color="transparent")
        brand.pack(fill="x", padx=theme.PAD_L, pady=(theme.PAD_XL, theme.PAD_L))
        ctk.CTkLabel(
            brand,
            text="AR Glasses",
            text_color=theme.TEXT,
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
        ).pack(fill="x")
        ctk.CTkLabel(
            brand,
            text="Sign · Voice · Mood",
            text_color=theme.TEXT_DIM,
            font=ctk.CTkFont(size=11),
            anchor="w",
        ).pack(fill="x", pady=(2, 0))

        self._nav_buttons: dict[str, ctk.CTkButton] = {}
        for key, label, hint in NAV_ITEMS:
            btn = ctk.CTkButton(
                bar,
                text=f"  {label}",
                anchor="w",
                fg_color="transparent",
                text_color=theme.TEXT_MUTED,
                hover_color=theme.PANEL_ALT,
                corner_radius=theme.BUTTON_RADIUS,
                height=40,
                font=ctk.CTkFont(size=13, weight="bold"),
                command=lambda k=key: self.navigate(k),
            )
            btn.pack(fill="x", padx=theme.PAD_M, pady=2)
            self._nav_buttons[key] = btn

        # Footer.
        ctk.CTkFrame(bar, fg_color="transparent").pack(fill="both", expand=True)
        ctk.CTkLabel(
            bar,
            text="Local mode · server still serves the glasses.",
            text_color=theme.TEXT_DIM,
            font=ctk.CTkFont(size=10),
            anchor="w",
            justify="left",
            wraplength=theme.SIDEBAR_WIDTH - 2 * theme.PAD_L,
        ).pack(fill="x", padx=theme.PAD_L, pady=(0, theme.PAD_M))

        return bar

    # --------------------------- navigation ---------------------------------
    def navigate(self, key: str) -> None:
        if key == self._current:
            return

        prev = self._current
        if prev is not None and prev in self._views:
            view = self._views[prev]
            view.grid_forget()
            on_hide = getattr(view, "on_hide", None)
            if callable(on_hide):
                try:
                    on_hide()
                except Exception:
                    pass

        view = self._views.get(key)
        if view is None:
            view = self._build_view(key)
            self._views[key] = view

        view.grid(row=0, column=0, sticky="nsew")
        on_show = getattr(view, "on_show", None)
        if callable(on_show):
            try:
                on_show()
            except Exception:
                pass

        self._current = key
        self._update_nav_styles()

    def _update_nav_styles(self) -> None:
        for key, btn in self._nav_buttons.items():
            if key == self._current:
                btn.configure(fg_color=theme.ACCENT, text_color=theme.TEXT)
            else:
                btn.configure(fg_color="transparent", text_color=theme.TEXT_MUTED)

    def _build_view(self, key: str) -> ctk.CTkFrame:
        if key == "home":
            return HomeView(self._content, navigate=self.navigate)
        if key == "live":
            return LiveView(
                self._content,
                camera=self._camera,
                tts=self._tts,
                camera_index=self._camera_index,
            )
        if key == "voice":
            return VoiceView(self._content)
        if key == "emotion":
            return EmotionView(self._content)
        if key == "training":
            return TrainingView(
                self._content,
                camera=self._camera,
                camera_index=self._camera_index,
            )
        raise ValueError(f"unknown view: {key}")

    # --------------------------- shutdown -----------------------------------
    def _on_close(self) -> None:
        # Make sure background workers stop before Tk tears down.
        for view in self._views.values():
            on_hide = getattr(view, "on_hide", None)
            if callable(on_hide):
                try:
                    on_hide()
                except Exception:
                    pass
        try:
            self._camera.stop()
        except Exception:
            pass
        try:
            self._tts.shutdown()
        except Exception:
            pass
        try:
            shutdown_runtime()
        except Exception:
            pass
        self.destroy()
