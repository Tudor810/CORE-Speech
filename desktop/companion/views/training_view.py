"""Teach signs (Training Studio).

Camera preview with hand landmarks, plus controls to:

* start/stop a labelled capture session,
* persist the current 30-frame rolling window as one sample,
* trigger an LSTM fine-tune (background).

All work runs through :mod:`app.services.training` so the same code paths
the FastAPI server uses are exercised here.
"""

from __future__ import annotations

import customtkinter as ctk

from .. import theme
from ..runtime import call_in_ui
from ..services import training as training_service
from ..services.camera import Camera, CameraError, list_cameras
from ..widgets.camera_view import CameraView


class TrainingView(ctk.CTkFrame):
    """Capture training samples + start training."""

    def __init__(self, master, *, camera: Camera, camera_index: int = 1) -> None:
        super().__init__(master, fg_color="transparent")
        self._camera = camera
        self._initial_camera_index = camera_index
        self._cameras = list_cameras() or [camera_index]

        self._session: training_service.TrainingSession | None = None
        self._capture_label: str | None = None

        self._build_header()
        self._status = ctk.CTkLabel(
            self,
            text="",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(size=12),
            anchor="w",
            justify="left",
        )
        self._status.pack(fill="x", padx=theme.PAD_L)

        self._build_grid()
        self.refresh_samples()

    # ---------------------------- layout ------------------------------------
    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=theme.PAD_L, pady=(theme.PAD_L, 0))
        ctk.CTkLabel(
            header,
            text="Teach signs",
            text_color=theme.TEXT,
            font=ctk.CTkFont(size=22, weight="bold"),
            anchor="w",
        ).pack(fill="x")
        ctk.CTkLabel(
            header,
            text="Show a sign in front of the camera a few times to teach it.",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(size=13),
            anchor="w",
        ).pack(fill="x", pady=(2, 0))

    def _build_grid(self) -> None:
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=theme.PAD_L, pady=theme.PAD_L)
        body.grid_columnconfigure(0, weight=3, uniform="train")
        body.grid_columnconfigure(1, weight=2, uniform="train")
        body.grid_rowconfigure(0, weight=1)

        # --- Left: camera preview --------------------------------------------------
        left = ctk.CTkFrame(body, fg_color=theme.PANEL, corner_radius=theme.CARD_RADIUS)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, theme.PAD_S))

        head = ctk.CTkFrame(left, fg_color="transparent")
        head.pack(fill="x", padx=theme.PAD_L, pady=(theme.PAD_M, theme.PAD_S))
        ctk.CTkLabel(
            head,
            text="Camera",
            text_color=theme.TEXT,
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).pack(side="left")
        self._connection_label = ctk.CTkLabel(
            head,
            text="● off",
            text_color=theme.TEXT_DIM,
            font=ctk.CTkFont(size=12),
        )
        self._connection_label.pack(side="right")

        cam_row = ctk.CTkFrame(left, fg_color="transparent")
        cam_row.pack(fill="x", padx=theme.PAD_L, pady=(0, theme.PAD_S))

        self._cam_var = ctk.StringVar(
            value=str(
                self._initial_camera_index
                if self._initial_camera_index in self._cameras
                else self._cameras[0]
            )
        )
        self._cam_menu = ctk.CTkOptionMenu(
            cam_row,
            values=[str(i) for i in self._cameras],
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

        self._camera_view = CameraView(
            left,
            width=640,
            height=480,
            placeholder='Press "Start camera" below to start tracking your hands.',
        )
        self._camera_view.pack(fill="both", expand=True, padx=theme.PAD_M, pady=(0, theme.PAD_M))

        self._buffer_label = ctk.CTkLabel(
            left,
            text="window: 0 / 30",
            text_color=theme.TEXT_DIM,
            font=ctk.CTkFont(size=12),
            anchor="w",
        )
        self._buffer_label.pack(fill="x", padx=theme.PAD_M, pady=(0, theme.PAD_M))

        ctk.CTkButton(
            left,
            text="Start camera",
            command=self._toggle_camera,
        ).pack(fill="x", padx=theme.PAD_M, pady=(0, theme.PAD_M))

        # --- Right: controls + sample list -----------------------------------
        right = ctk.CTkFrame(body, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(theme.PAD_S, 0))

        self._build_capture_card(right)
        self._build_samples_card(right)
        self._build_train_card(right)

    def _build_capture_card(self, parent) -> None:
        card = ctk.CTkFrame(parent, fg_color=theme.PANEL, corner_radius=theme.CARD_RADIUS)
        card.pack(fill="x", pady=(0, theme.PAD_M))

        ctk.CTkLabel(
            card,
            text="Add a new sign",
            text_color=theme.TEXT,
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=theme.PAD_L, pady=(theme.PAD_M, theme.PAD_S))

        self._label_entry = ctk.CTkEntry(
            card, placeholder_text="What does this sign mean? (e.g. hello)"
        )
        self._label_entry.pack(fill="x", padx=theme.PAD_L, pady=(0, theme.PAD_M))

        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.pack(fill="x", padx=theme.PAD_L, pady=(0, theme.PAD_M))

        self._start_btn = ctk.CTkButton(
            btn_row,
            text="Start",
            fg_color=theme.ACCENT,
            hover_color=theme.ACCENT_HOVER,
            command=self._start_capture,
        )
        self._start_btn.pack(side="left")

        self._stop_btn = ctk.CTkButton(
            btn_row,
            text="Done",
            fg_color=theme.ERR,
            hover_color="#dc2626",
            command=self._stop_capture,
        )

        self._save_btn = ctk.CTkButton(
            btn_row,
            text="+ Save example",
            command=self._save_sample,
            state="disabled",
        )
        self._save_btn.pack(side="right")

        ctk.CTkLabel(
            card,
            text=(
                "Show the sign in front of the camera for about a second, then click "
                "“Save example”. Save it 20–30 times for the best results."
            ),
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(size=11),
            anchor="w",
            justify="left",
            wraplength=360,
        ).pack(fill="x", padx=theme.PAD_L, pady=(0, theme.PAD_M))

    def _build_samples_card(self, parent) -> None:
        card = ctk.CTkFrame(parent, fg_color=theme.PANEL, corner_radius=theme.CARD_RADIUS)
        card.pack(fill="both", expand=True, pady=(0, theme.PAD_M))

        head = ctk.CTkFrame(card, fg_color="transparent")
        head.pack(fill="x", padx=theme.PAD_L, pady=(theme.PAD_M, theme.PAD_S))
        ctk.CTkLabel(
            head,
            text="Your signs",
            text_color=theme.TEXT,
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).pack(side="left")
        ctk.CTkButton(
            head, text="Refresh", width=80, command=self.refresh_samples
        ).pack(side="right")

        self._samples_box = ctk.CTkTextbox(
            card,
            height=160,
            fg_color=theme.PANEL_ALT,
            text_color=theme.TEXT,
            font=ctk.CTkFont(size=12),
        )
        self._samples_box.pack(
            fill="both", expand=True, padx=theme.PAD_M, pady=(0, theme.PAD_M)
        )
        self._samples_box.configure(state="disabled")

    def _build_train_card(self, parent) -> None:
        card = ctk.CTkFrame(parent, fg_color=theme.PANEL, corner_radius=theme.CARD_RADIUS)
        card.pack(fill="x", pady=(0, theme.PAD_M))

        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=theme.PAD_L, pady=theme.PAD_M)

        self._train_btn = ctk.CTkButton(
            row,
            text="Teach my glasses",
            fg_color=theme.ACCENT,
            hover_color=theme.ACCENT_HOVER,
            command=self._train,
        )
        self._train_btn.pack(side="left")

        ctk.CTkButton(
            row,
            text="Load starter signs",
            command=self._load_starter,
        ).pack(side="left", padx=(theme.PAD_S, 0))

    # ---------------------------- behaviour ---------------------------------
    def on_show(self) -> None:
        self.refresh_samples()

    def on_hide(self) -> None:
        self._stop_camera()

    def _toggle_camera(self) -> None:
        if self._session is not None and self._session.is_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self) -> None:
        try:
            cam_idx = int(self._cam_var.get())
        except ValueError:
            cam_idx = self._initial_camera_index
        try:
            self._camera.start(cam_idx)
        except CameraError as exc:
            self._set_status(f"Camera error: {exc}")
            return

        self._session = training_service.TrainingSession(
            self._camera, on_frame=self._on_frame
        )
        self._session.start()
        self._connection_label.configure(text="● live", text_color=theme.OK)
        self._set_status("Camera live. Show your hand to the camera.")
        self._save_btn.configure(
            state="normal" if self._capture_label else "disabled"
        )

    def _stop_camera(self) -> None:
        if self._session is not None:
            self._session.stop()
            self._session = None
        self._camera.stop()
        self._connection_label.configure(text="● off", text_color=theme.TEXT_DIM)
        self._camera_view.show_placeholder('Press "Start camera" below to start tracking your hands.')
        self._save_btn.configure(state="disabled")

    def _refresh_camera_list(self) -> None:
        cams = list_cameras() or [int(self._cam_var.get() or self._initial_camera_index)]
        values = [str(i) for i in cams]
        self._cam_menu.configure(values=values)
        if self._cam_var.get() not in values:
            self._cam_var.set(values[0])

    def _on_camera_changed(self, _value: str) -> None:
        if self._session is not None and self._session.is_running:
            self._stop_camera()
            self._start_camera()

    def _on_frame(self, frame) -> None:
        call_in_ui(self, self._render_frame, frame)

    def _render_frame(self, frame) -> None:
        landmarks = frame.hands if frame.has_hand else None
        self._camera_view.render(frame.bgr, landmarks=landmarks)
        if self._session is not None:
            cur, total = self._session.buffer_progress
            self._buffer_label.configure(text=f"window: {cur} / {total}")

    # ---------------------------- capture API -------------------------------
    def _start_capture(self) -> None:
        label = self._label_entry.get().strip()
        if not label:
            self._set_status("Type a name for the sign first.")
            return
        if self._session is None or not self._session.is_running:
            self._start_camera()
        if self._session is None:
            return

        self._capture_label = label
        self._start_btn.pack_forget()
        self._stop_btn.pack(side="left")
        self._save_btn.configure(state="normal")
        self._set_status(f"Capturing samples for “{label}”. Hold a sign and click + Save.")

    def _stop_capture(self) -> None:
        self._capture_label = None
        self._stop_btn.pack_forget()
        self._start_btn.pack(side="left", before=self._save_btn)
        self._save_btn.configure(state="disabled")
        self._set_status("Stopped capturing.")

    def _save_sample(self) -> None:
        if self._capture_label is None or self._session is None:
            return
        snapshot = self._session.snapshot_buffer()
        if snapshot is None:
            self._set_status("Window not yet full — hold the pose for a moment longer.")
            return
        try:
            captured = training_service.save_sample(self._capture_label, snapshot)
        except Exception as exc:
            self._set_status(f"Couldn't save: {exc}")
            return
        self._set_status(
            f"Saved “{captured.label}” — you have {captured.samples_for_label} "
            f"example{'' if captured.samples_for_label == 1 else 's'}."
        )
        self.refresh_samples()

    # ---------------------------- training ----------------------------------
    def _train(self) -> None:
        self._train_btn.configure(state="disabled", text="Training…")
        self._set_status("Training started. This can take a few moments.")

        def _on_done(result, err: str | None) -> None:
            call_in_ui(self, self._after_train, result, err)

        try:
            training_service.train(epochs=25, on_done=_on_done)
        except Exception as exc:
            self._after_train(None, str(exc))

    def _after_train(self, result, err: str | None) -> None:
        self._train_btn.configure(state="normal", text="Teach my glasses")
        if err:
            self._set_status(f"Training failed: {err}")
            return
        if result and result.get("status") == "ok":
            labels = ", ".join(result.get("labels", []))
            loss = result.get("final_loss", 0.0)
            self._set_status(
                f"Training complete. Labels: {labels} (final loss {loss:.4f}). "
                "Open Live to try it out."
            )
        else:
            reason = (result or {}).get("reason", "unknown")
            self._set_status(f"Training skipped: {reason}")

    def _load_starter(self) -> None:
        try:
            info = training_service.load_default_pack()
        except Exception as exc:
            self._set_status(f"Couldn't load starter pack: {exc}")
            return
        labels = info.get("suggested_labels", []) or []
        if labels:
            self._set_status(
                "Starter pack: " + ", ".join(labels) + ". Capture each one to begin."
            )
        else:
            self._set_status(info.get("message") or "Starter pack not available.")

    # ---------------------------- helpers -----------------------------------
    def refresh_samples(self) -> None:
        try:
            counts = training_service.list_sample_counts()
        except Exception as exc:
            self._set_samples_text(f"Couldn't load samples: {exc}")
            return
        if not counts:
            self._set_samples_text("You haven’t saved any signs yet.")
            return
        lines = [
            f"{label:<24} {count} example{'s' if count != 1 else ''}"
            for label, count in sorted(counts.items())
        ]
        lines.append("")
        lines.append(f"total: {sum(counts.values())} samples")
        self._set_samples_text("\n".join(lines))

    def _set_samples_text(self, text: str) -> None:
        self._samples_box.configure(state="normal")
        self._samples_box.delete("1.0", "end")
        self._samples_box.insert("end", text)
        self._samples_box.configure(state="disabled")

    def _set_status(self, text: str) -> None:
        self._status.configure(text=text)
