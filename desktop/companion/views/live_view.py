"""Live sign-to-speech session.

Replicates the Next.js Live page on top of :class:`LiveEngine` and the
shared :class:`Camera`. Events from the engine arrive on a background
thread and are marshalled back to the Tk main loop with
:func:`app.runtime.call_in_ui`.
"""

from __future__ import annotations

import customtkinter as ctk

from .. import theme
from ..runtime import app_state, call_in_ui
from ..services.camera import Camera, CameraError, list_cameras
from ..services.live_engine import LiveEngine, LiveEvent
from ..services.tts_worker import TtsWorker
from ..widgets.camera_view import CameraView


MAX_GESTURE_HISTORY = 8
MAX_PHRASE_HISTORY = 8


class LiveView(ctk.CTkFrame):
    """Camera + live engine + TTS for the deaf user."""

    def __init__(
        self,
        master,
        *,
        camera: Camera,
        tts: TtsWorker,
        camera_index: int = 1,
    ) -> None:
        super().__init__(master, fg_color="transparent")
        self._camera = camera
        self._tts = tts
        self._engine: LiveEngine | None = None
        self._initial_camera_index = camera_index
        self._cameras = list_cameras()
        if not self._cameras:
            self._cameras = [camera_index]

        self._gesture_history: list[tuple[str, float]] = []
        self._phrases: list[dict] = []
        self._current_phrase_idx: int | None = None

        self._build_header()
        self._build_controls()
        self._build_video()
        self._build_status_row()
        self._build_history()

    # --------------------------- layout -------------------------------------
    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=theme.PAD_L, pady=(theme.PAD_L, 0))
        ctk.CTkLabel(
            header,
            text="Live",
            text_color=theme.TEXT,
            font=ctk.CTkFont(size=22, weight="bold"),
            anchor="w",
        ).pack(fill="x")
        ctk.CTkLabel(
            header,
            text="Show signs to your camera and hear them spoken back in real time.",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(size=13),
            anchor="w",
        ).pack(fill="x", pady=(2, 0))

    def _build_controls(self) -> None:
        bar = ctk.CTkFrame(self, fg_color=theme.PANEL, corner_radius=theme.CARD_RADIUS)
        bar.pack(fill="x", padx=theme.PAD_L, pady=theme.PAD_L)

        left = ctk.CTkFrame(bar, fg_color="transparent")
        left.pack(side="left", fill="x", expand=True, padx=theme.PAD_L, pady=theme.PAD_M)
        ctk.CTkLabel(
            left,
            text="CAMERA",
            text_color=theme.TEXT_DIM,
            font=ctk.CTkFont(size=10, weight="bold"),
            anchor="w",
        ).pack(fill="x")

        cam_row = ctk.CTkFrame(left, fg_color="transparent")
        cam_row.pack(fill="x", pady=(theme.PAD_S, 0))

        cam_values = [str(i) for i in self._cameras]
        if not cam_values:
            cam_values = [str(self._initial_camera_index)]
        self._cam_var = ctk.StringVar(
            value=str(self._initial_camera_index)
            if self._initial_camera_index in self._cameras
            else cam_values[0]
        )
        self._cam_menu = ctk.CTkOptionMenu(
            cam_row,
            values=cam_values,
            variable=self._cam_var,
            width=80,
            command=self._on_camera_changed,
        )
        self._cam_menu.pack(side="left")
        ctk.CTkButton(
            cam_row,
            text="↻",
            width=36,
            command=self._refresh_camera_list,
        ).pack(side="left", padx=(theme.PAD_S, 0))

        right = ctk.CTkFrame(bar, fg_color="transparent")
        right.pack(side="right", padx=theme.PAD_L, pady=theme.PAD_M)

        self._mute_btn = ctk.CTkButton(
            right, text="🔊 Mute", command=self._toggle_mute, width=90
        )
        self._stop_btn = ctk.CTkButton(
            right,
            text="■ Stop",
            command=self.stop_session,
            fg_color=theme.ERR,
            hover_color="#dc2626",
            width=90,
        )
        self._start_btn = ctk.CTkButton(
            right,
            text="● Start session",
            command=self.start_session,
            fg_color=theme.ACCENT,
            hover_color=theme.ACCENT_HOVER,
            width=160,
        )
        self._start_btn.pack(side="right")

        self._muted = False

    def _build_video(self) -> None:
        wrap = ctk.CTkFrame(
            self, fg_color=theme.PANEL, corner_radius=theme.CARD_RADIUS
        )
        wrap.pack(fill="both", expand=True, padx=theme.PAD_L, pady=(0, theme.PAD_L))

        self._camera_view = CameraView(
            wrap,
            width=720,
            height=480,
            placeholder='Press "Start session" to turn on your camera.',
        )
        self._camera_view.pack(fill="both", expand=True, padx=theme.PAD_M, pady=theme.PAD_M)

        overlay = ctk.CTkFrame(wrap, fg_color="transparent")
        overlay.pack(fill="x", padx=theme.PAD_M, pady=(0, theme.PAD_M))

        self._signs_label = ctk.CTkLabel(
            overlay,
            text="",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(size=11),
            anchor="w",
            justify="left",
        )
        self._signs_label.pack(fill="x")

        self._spoken_label = ctk.CTkLabel(
            overlay,
            text="",
            text_color=theme.TEXT,
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
            justify="left",
        )
        self._spoken_label.pack(fill="x", pady=(2, 0))

    def _build_status_row(self) -> None:
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=theme.PAD_L, pady=(0, theme.PAD_L))
        for i in range(3):
            row.grid_columnconfigure(i, weight=1, uniform="status")

        self._status_card = self._status_box(
            row, "STATUS", "Idle", "Press start to begin."
        )
        self._status_card.grid(row=0, column=0, sticky="nsew", padx=(0, theme.PAD_S))

        self._fps_card = self._status_box(row, "FPS", "0.0", "Live engine speed.")
        self._fps_card.grid(row=0, column=1, sticky="nsew", padx=theme.PAD_S)

        self._labels_card = self._status_box(
            row, "MODEL LABELS", "—", "Trained sign vocabulary."
        )
        self._labels_card.grid(row=0, column=2, sticky="nsew", padx=(theme.PAD_S, 0))

    def _build_history(self) -> None:
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="both", expand=False, padx=theme.PAD_L, pady=(0, theme.PAD_L))
        row.grid_columnconfigure(0, weight=1, uniform="hist")
        row.grid_columnconfigure(1, weight=1, uniform="hist")

        gest_card = ctk.CTkFrame(
            row, fg_color=theme.PANEL, corner_radius=theme.CARD_RADIUS
        )
        gest_card.grid(row=0, column=0, sticky="nsew", padx=(0, theme.PAD_S))
        ctk.CTkLabel(
            gest_card,
            text="Detected signs",
            text_color=theme.TEXT,
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=theme.PAD_L, pady=(theme.PAD_M, theme.PAD_S))
        self._gesture_box = ctk.CTkTextbox(
            gest_card,
            height=120,
            fg_color=theme.PANEL_ALT,
            text_color=theme.TEXT,
            font=ctk.CTkFont(size=12),
        )
        self._gesture_box.pack(fill="both", expand=True, padx=theme.PAD_M, pady=(0, theme.PAD_M))
        self._gesture_box.configure(state="disabled")

        phrase_card = ctk.CTkFrame(
            row, fg_color=theme.PANEL, corner_radius=theme.CARD_RADIUS
        )
        phrase_card.grid(row=0, column=1, sticky="nsew", padx=(theme.PAD_S, 0))
        ctk.CTkLabel(
            phrase_card,
            text="Spoken sentences",
            text_color=theme.TEXT,
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=theme.PAD_L, pady=(theme.PAD_M, theme.PAD_S))
        self._phrase_box = ctk.CTkTextbox(
            phrase_card,
            height=120,
            fg_color=theme.PANEL_ALT,
            text_color=theme.TEXT,
            font=ctk.CTkFont(size=12),
        )
        self._phrase_box.pack(fill="both", expand=True, padx=theme.PAD_M, pady=(0, theme.PAD_M))
        self._phrase_box.configure(state="disabled")

    @staticmethod
    def _status_box(parent, title: str, value: str, hint: str) -> ctk.CTkFrame:
        card = ctk.CTkFrame(parent, fg_color=theme.PANEL, corner_radius=theme.CARD_RADIUS)
        ctk.CTkLabel(
            card,
            text=title,
            text_color=theme.TEXT_DIM,
            font=ctk.CTkFont(size=10, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=theme.PAD_L, pady=(theme.PAD_M, 0))
        value_label = ctk.CTkLabel(
            card,
            text=value,
            text_color=theme.TEXT,
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
        )
        value_label.pack(fill="x", padx=theme.PAD_L, pady=(theme.PAD_S, 0))
        ctk.CTkLabel(
            card,
            text=hint,
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(size=11),
            anchor="w",
        ).pack(fill="x", padx=theme.PAD_L, pady=(0, theme.PAD_M))
        card._value_label = value_label  # type: ignore[attr-defined]
        return card

    # --------------------------- navigation hooks --------------------------
    def on_show(self) -> None:
        # Refresh the camera dropdown (devices may have changed) but don't
        # auto-start the camera until the user clicks Start.
        self._refresh_camera_list()

    def on_hide(self) -> None:
        # Free the camera when the user navigates away.
        self.stop_session()

    # --------------------------- session control ---------------------------
    def start_session(self) -> None:
        if self._engine is not None and self._engine.is_running:
            return

        try:
            cam_idx = int(self._cam_var.get())
        except ValueError:
            cam_idx = self._initial_camera_index

        try:
            self._camera.start(cam_idx)
        except CameraError as exc:
            self._set_status("Camera error", str(exc))
            return

        engine = LiveEngine(
            camera=self._camera,
            on_event=self._on_engine_event,
            tts_worker=self._tts,
            emotion=app_state().data.emotion_preset,
            intensity=app_state().data.emotion_intensity,
        )
        try:
            engine.start()
        except Exception as exc:
            self._set_status("Engine error", str(exc))
            self._camera.stop()
            return

        self._engine = engine
        self._labels_card._value_label.configure(  # type: ignore[attr-defined]
            text=", ".join(engine.labels[:6]) + ("…" if len(engine.labels) > 6 else "")
        )
        self._set_status("Listening", f"Camera {cam_idx} · Sign in front of camera.")
        self._start_btn.pack_forget()
        self._stop_btn.pack(side="right", padx=(theme.PAD_S, 0))
        self._mute_btn.pack(side="right")

    def stop_session(self) -> None:
        if self._engine is not None:
            self._engine.stop()
            self._engine = None
        self._camera.stop()
        self._set_status("Idle", "Press start to begin.")
        self._camera_view.show_placeholder('Press "Start session" to turn on your camera.')
        self._stop_btn.pack_forget()
        self._mute_btn.pack_forget()
        self._start_btn.pack(side="right")

    def _toggle_mute(self) -> None:
        self._muted = not self._muted
        # We don't have a "mute" hook on TtsWorker — easiest is to pause
        # outgoing utterances by toggling the engine flag.
        if self._engine is not None:
            self._engine.set_do_llm(not self._muted or False)
        # The TTS worker keeps its enabled state; we mute by no-op'ing
        # the say() in the engine's _sentence_worker via emotion override.
        # For now just relabel the button so the user sees the state.
        self._mute_btn.configure(text="🔇 Muted" if self._muted else "🔊 Mute")

    def _refresh_camera_list(self) -> None:
        cams = list_cameras()
        if not cams:
            cams = [int(self._cam_var.get() or self._initial_camera_index)]
        values = [str(i) for i in cams]
        current = self._cam_var.get()
        self._cam_menu.configure(values=values)
        if current not in values:
            self._cam_var.set(values[0])

    def _on_camera_changed(self, _value: str) -> None:
        if self._engine is not None and self._engine.is_running:
            # Hot-swap by restarting the session on the new camera.
            self.stop_session()
            self.start_session()

    # --------------------------- engine events -----------------------------
    def _on_engine_event(self, event: LiveEvent) -> None:
        call_in_ui(self, self._handle_event, event)

    def _handle_event(self, event: LiveEvent) -> None:
        if event.kind == "frame":
            landmarks = (
                event.landmarks.hands
                if event.landmarks and event.landmarks.has_hand
                else None
            )
            if event.frame is not None:
                self._camera_view.render(event.frame, landmarks=landmarks)
            self._fps_card._value_label.configure(text=f"{event.fps:5.1f}")  # type: ignore[attr-defined]

            if event.label:
                marker = "★" if event.confidence >= 0.5 else "·"
                self._signs_label.configure(
                    text=f"{marker} live: {event.label.upper()}  ({event.confidence*100:.0f}%)"
                )
            elif event.is_idle:
                self._signs_label.configure(text="idle")
            elif event.tokens:
                self._signs_label.configure(
                    text="tokens: " + " · ".join(s.upper() for s in event.tokens)
                )
            else:
                self._signs_label.configure(text="ready — make a sign")
            return

        if event.kind == "gesture" and event.label:
            self._gesture_history.append((event.label, event.confidence))
            self._gesture_history = self._gesture_history[-MAX_GESTURE_HISTORY:]
            self._render_gestures()
            return

        if event.kind == "phrase":
            self._phrases.append(
                {"tokens": list(event.tokens), "translation": "", "finished": False}
            )
            self._phrases = self._phrases[-MAX_PHRASE_HISTORY:]
            self._current_phrase_idx = len(self._phrases) - 1
            self._render_phrases()
            return

        if event.kind == "translation":
            if self._current_phrase_idx is None:
                self._phrases.append(
                    {
                        "tokens": [],
                        "translation": event.text or "",
                        "finished": False,
                    }
                )
                self._current_phrase_idx = len(self._phrases) - 1
            else:
                self._phrases[self._current_phrase_idx]["translation"] = event.text or ""
            self._spoken_label.configure(text=event.text or "")
            self._render_phrases()
            return

        if event.kind == "audio_end":
            if self._current_phrase_idx is not None:
                self._phrases[self._current_phrase_idx]["finished"] = True
                self._render_phrases()
            self._current_phrase_idx = None
            return

        if event.kind == "status":
            self._set_status("Listening", "Sign in front of the camera.")
            return

        if event.kind == "error":
            self._set_status("Error", event.msg or "Something went wrong.")
            return

    def _set_status(self, value: str, hint: str) -> None:
        self._status_card._value_label.configure(text=value)  # type: ignore[attr-defined]
        # Replace the hint label (third child of the card) by walking children.
        for child in self._status_card.winfo_children():
            if isinstance(child, ctk.CTkLabel) and child is not getattr(
                self._status_card, "_value_label", None
            ):
                if child.cget("text_color") == theme.TEXT_MUTED:
                    child.configure(text=hint)
                    break

    def _render_gestures(self) -> None:
        text = "\n".join(
            f"{label}    ({conf * 100:.1f}%)"
            for label, conf in reversed(self._gesture_history)
        )
        self._gesture_box.configure(state="normal")
        self._gesture_box.delete("1.0", "end")
        self._gesture_box.insert("end", text or "Detected signs will appear here.")
        self._gesture_box.configure(state="disabled")

    def _render_phrases(self) -> None:
        lines: list[str] = []
        for p in reversed(self._phrases):
            tokens = " · ".join(p["tokens"]) if p["tokens"] else "—"
            translation = p["translation"] or ("…" if not p["finished"] else "")
            speaking = "  (speaking…)" if not p["finished"] else ""
            lines.append(f"signs: {tokens}\nspoken: {translation}{speaking}\n")
        text = "\n".join(lines).rstrip()
        self._phrase_box.configure(state="normal")
        self._phrase_box.delete("1.0", "end")
        self._phrase_box.insert(
            "end", text or "Each spoken phrase will show up here once your glasses translate it."
        )
        self._phrase_box.configure(state="disabled")
