"""Voice page: pick a preset, manage cloned voices, record a new clone."""

from __future__ import annotations

import threading
from pathlib import Path

import customtkinter as ctk

from .. import theme
from ..runtime import call_in_ui, submit_coro
from ..services import voice as voice_service
from ..services.audio_recorder import MAX_SECONDS, AudioCaptureError, MicRecorder

# Same friendly-name table the Next.js page uses for MeloTTS keys.
SPEAKER_LABELS: dict[str, str] = {
    "EN-Default": "Default",
    "EN-US": "American",
    "EN-BR": "British",
    "EN_INDIA": "Indian",
    "EN-INDIA": "Indian",
    "EN-AU": "Australian",
    "EN_NEWEST": "Newest",
    "EN-NEWEST": "Newest",
    "EN": "English",
    "ES": "Spanish",
    "FR": "French",
    "ZH": "Chinese",
    "JP": "Japanese",
    "KR": "Korean",
}

SPEAKER_HINTS: dict[str, str] = {
    "EN-Default": "Neutral English",
    "EN-US": "US accent",
    "EN-BR": "British accent",
    "EN_INDIA": "Indian accent",
    "EN-INDIA": "Indian accent",
    "EN-AU": "Australian accent",
    "EN_NEWEST": "Latest model voice",
    "EN-NEWEST": "Latest model voice",
}


def _pretty(name: str) -> str:
    if name in SPEAKER_LABELS:
        return SPEAKER_LABELS[name]
    stripped = name
    for prefix in ("EN-", "EN_", "ES-", "FR-", "ZH-", "JP-", "KR-"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    return stripped.replace("_", " ").replace("-", " ").title() or name


class VoiceView(ctk.CTkFrame):
    """Configure the voice your glasses speak in."""

    def __init__(self, master) -> None:
        super().__init__(master, fg_color="transparent")

        self._recorder = MicRecorder()
        self._wav_path: Path | None = None
        self._recording = False

        self._build_header()
        self._error_label = ctk.CTkLabel(
            self,
            text="",
            text_color=theme.ERR,
            font=ctk.CTkFont(size=12),
            anchor="w",
            justify="left",
        )
        self._error_label.pack(fill="x", padx=theme.PAD_L)

        self._build_presets()
        self._build_custom_voices()
        self._build_clone()

        self.refresh(refresh_speakers=False)

    # ---------------------------- layout ------------------------------------
    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=theme.PAD_L, pady=(theme.PAD_L, 0))
        ctk.CTkLabel(
            header,
            text="Voice",
            text_color=theme.TEXT,
            font=ctk.CTkFont(size=22, weight="bold"),
            anchor="w",
        ).pack(fill="x")
        ctk.CTkLabel(
            header,
            text="Choose how your glasses sound when they speak for you.",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(size=13),
            anchor="w",
        ).pack(fill="x", pady=(2, 0))

    def _build_presets(self) -> None:
        self._presets_card = ctk.CTkFrame(
            self, fg_color=theme.PANEL, corner_radius=theme.CARD_RADIUS
        )
        self._presets_card.pack(fill="x", padx=theme.PAD_L, pady=theme.PAD_L)

        ctk.CTkLabel(
            self._presets_card,
            text="Pick a voice",
            text_color=theme.TEXT,
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=theme.PAD_L, pady=(theme.PAD_M, theme.PAD_S))

        self._presets_grid = ctk.CTkFrame(self._presets_card, fg_color="transparent")
        self._presets_grid.pack(fill="x", padx=theme.PAD_L, pady=(0, theme.PAD_M))

        self._presets_loading = ctk.CTkLabel(
            self._presets_grid,
            text="Loading voices…",
            text_color=theme.TEXT_DIM,
            font=ctk.CTkFont(size=12),
            anchor="w",
        )
        self._presets_loading.grid(row=0, column=0, sticky="w")

    def _build_custom_voices(self) -> None:
        self._custom_card = ctk.CTkFrame(
            self, fg_color=theme.PANEL, corner_radius=theme.CARD_RADIUS
        )
        # Pack only when there are voices to show.
        ctk.CTkLabel(
            self._custom_card,
            text="Your voices",
            text_color=theme.TEXT,
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=theme.PAD_L, pady=(theme.PAD_M, theme.PAD_S))

        self._custom_list = ctk.CTkFrame(self._custom_card, fg_color="transparent")
        self._custom_list.pack(fill="x", padx=theme.PAD_L, pady=(0, theme.PAD_M))

    def _build_clone(self) -> None:
        card = ctk.CTkFrame(self, fg_color=theme.PANEL, corner_radius=theme.CARD_RADIUS)
        card.pack(fill="x", padx=theme.PAD_L, pady=(0, theme.PAD_L))

        ctk.CTkLabel(
            card,
            text="Use your own voice",
            text_color=theme.TEXT,
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=theme.PAD_L, pady=(theme.PAD_M, 0))

        ctk.CTkLabel(
            card,
            text=f"Record a {int(MAX_SECONDS)}-second sample and your glasses will speak in your voice.",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(size=12),
            anchor="w",
            justify="left",
            wraplength=600,
        ).pack(fill="x", padx=theme.PAD_L, pady=(2, theme.PAD_M))

        ctk.CTkLabel(
            card,
            text="Name this voice",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(size=12),
            anchor="w",
        ).pack(fill="x", padx=theme.PAD_L)
        self._name_entry = ctk.CTkEntry(card, placeholder_text="My voice")
        self._name_entry.pack(fill="x", padx=theme.PAD_L, pady=(2, theme.PAD_M))

        controls = ctk.CTkFrame(card, fg_color="transparent")
        controls.pack(fill="x", padx=theme.PAD_L, pady=(0, theme.PAD_S))

        self._record_btn = ctk.CTkButton(
            controls,
            text="● Start recording",
            fg_color=theme.ACCENT,
            hover_color=theme.ACCENT_HOVER,
            command=self._toggle_recording,
        )
        self._record_btn.pack(side="left")

        self._record_status = ctk.CTkLabel(
            controls,
            text="",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(size=12),
            anchor="w",
        )
        self._record_status.pack(side="left", padx=(theme.PAD_M, 0))

        self._progress = ctk.CTkProgressBar(card, height=8)
        self._progress.pack(fill="x", padx=theme.PAD_L, pady=(0, theme.PAD_M))
        self._progress.set(0.0)

        self._save_btn = ctk.CTkButton(
            card,
            text="Save and use this voice",
            fg_color=theme.ACCENT,
            hover_color=theme.ACCENT_HOVER,
            state="disabled",
            command=self._save_clone,
        )
        self._save_btn.pack(fill="x", padx=theme.PAD_L, pady=(0, theme.PAD_M))

        self._clone_msg = ctk.CTkLabel(
            card,
            text="",
            text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(size=12),
            anchor="w",
            justify="left",
        )
        self._clone_msg.pack(fill="x", padx=theme.PAD_L, pady=(0, theme.PAD_M))

    # ---------------------------- refresh -----------------------------------
    def on_show(self) -> None:
        self.refresh()

    def refresh(self, *, refresh_speakers: bool = True) -> None:
        self._set_error("")
        # Speakers might require loading the TTS engine — kick off in a worker
        # so the UI stays responsive.
        def _load() -> None:
            try:
                snap = voice_service.snapshot(refresh_speakers=refresh_speakers)
            except Exception as exc:
                call_in_ui(self, self._set_error, str(exc))
                return
            call_in_ui(self, self._render_snapshot, snap)

        threading.Thread(target=_load, daemon=True).start()

    def _render_snapshot(self, snap) -> None:
        # Clear preset grid.
        for child in self._presets_grid.winfo_children():
            child.destroy()

        speakers = snap.speakers or []
        if not speakers:
            ctk.CTkLabel(
                self._presets_grid,
                text="No voice presets available — is OpenVoice / MeloTTS installed?",
                text_color=theme.TEXT_DIM,
                font=ctk.CTkFont(size=12),
                anchor="w",
            ).grid(row=0, column=0, sticky="w")
        else:
            cols = 3
            for col in range(cols):
                self._presets_grid.grid_columnconfigure(col, weight=1, uniform="sp")
            for i, key in enumerate(speakers):
                row, col = divmod(i, cols)
                is_active = key == snap.active_speaker_key
                btn = ctk.CTkButton(
                    self._presets_grid,
                    text=f"{_pretty(key)}\n{SPEAKER_HINTS.get(key, 'Tap to use')}",
                    fg_color=theme.ACCENT if is_active else theme.PANEL_ALT,
                    hover_color=theme.ACCENT_HOVER if is_active else theme.BORDER,
                    text_color=theme.TEXT,
                    state="disabled" if is_active else "normal",
                    command=lambda k=key: self._pick_preset(k),
                    height=58,
                    anchor="w",
                )
                btn.grid(
                    row=row,
                    column=col,
                    padx=theme.PAD_S // 2,
                    pady=theme.PAD_S // 2,
                    sticky="nsew",
                )

        # Custom voices.
        for child in self._custom_list.winfo_children():
            child.destroy()
        if snap.custom_voices:
            self._custom_card.pack(
                fill="x", padx=theme.PAD_L, pady=(0, theme.PAD_L), after=self._presets_card
            )

            self._custom_row(
                "Default",
                snap.default_voice_id,
                snap.active_voice_id == snap.default_voice_id,
                allow_delete=False,
            )
            for v in snap.custom_voices:
                self._custom_row(
                    v.name,
                    v.voice_id,
                    snap.active_voice_id == v.voice_id,
                    allow_delete=True,
                )
        else:
            self._custom_card.pack_forget()

    def _custom_row(
        self,
        name: str,
        voice_id: str,
        is_active: bool,
        *,
        allow_delete: bool,
    ) -> None:
        row = ctk.CTkFrame(self._custom_list, fg_color="transparent")
        row.pack(fill="x", pady=2)

        ctk.CTkLabel(
            row,
            text=name,
            text_color=theme.TEXT,
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        ).pack(side="left", fill="x", expand=True, padx=theme.PAD_S)

        if allow_delete:
            ctk.CTkButton(
                row,
                text="Remove",
                fg_color="transparent",
                text_color=theme.ERR,
                hover_color=theme.PANEL_ALT,
                width=80,
                command=lambda vid=voice_id: self._remove_voice(vid),
            ).pack(side="right", padx=(theme.PAD_S, 0))

        ctk.CTkButton(
            row,
            text="In use" if is_active else "Use",
            state="disabled" if is_active else "normal",
            width=70,
            command=lambda vid=voice_id: self._pick_profile(vid),
        ).pack(side="right")

    # ---------------------------- callbacks ---------------------------------
    def _pick_preset(self, key: str) -> None:
        self._set_error("")
        try:
            voice_service.set_active_speaker(key)
        except Exception as exc:
            self._set_error(f"Couldn't switch voice: {exc}")
            return
        self.refresh(refresh_speakers=False)

    def _pick_profile(self, voice_id: str) -> None:
        self._set_error("")
        try:
            voice_service.set_active_voice(voice_id)
        except Exception as exc:
            self._set_error(f"Couldn't switch voice: {exc}")
            return
        self.refresh(refresh_speakers=False)

    def _remove_voice(self, voice_id: str) -> None:
        self._set_error("")
        try:
            voice_service.remove_custom_voice(voice_id)
        except Exception as exc:
            self._set_error(f"Couldn't remove voice: {exc}")
            return
        self.refresh(refresh_speakers=False)

    # ---------------------------- recording ---------------------------------
    def _toggle_recording(self) -> None:
        if self._recording:
            self._recorder.stop()
            return

        try:
            self._recorder.start(
                duration=MAX_SECONDS,
                on_tick=lambda elapsed: call_in_ui(self, self._on_tick, elapsed),
                on_done=lambda path, err: call_in_ui(self, self._on_recording_done, path, err),
            )
        except AudioCaptureError as exc:
            self._set_error(str(exc))
            return

        self._recording = True
        self._wav_path = None
        self._save_btn.configure(state="disabled")
        self._progress.set(0.0)
        self._record_btn.configure(
            text="■ Stop", fg_color=theme.ERR, hover_color="#dc2626"
        )
        self._record_status.configure(text="Recording…")
        self._clone_msg.configure(text="")

    def _on_tick(self, elapsed: float) -> None:
        pct = min(1.0, elapsed / MAX_SECONDS)
        self._progress.set(pct)
        self._record_status.configure(text=f"Recording… {elapsed:.1f}s / {MAX_SECONDS:.0f}s")

    def _on_recording_done(self, path, err: str | None) -> None:
        self._recording = False
        self._record_btn.configure(
            text="● Re-record" if path else "● Start recording",
            fg_color=theme.ACCENT,
            hover_color=theme.ACCENT_HOVER,
        )
        if err:
            self._set_error(f"Microphone error: {err}")
            self._record_status.configure(text="")
            return
        self._wav_path = path
        self._record_status.configure(text="Recording ready")
        self._save_btn.configure(state="normal")
        self._progress.set(1.0)

    # ---------------------------- cloning -----------------------------------
    def _save_clone(self) -> None:
        if self._wav_path is None or not self._name_entry.get().strip():
            return

        name = self._name_entry.get().strip()
        wav_path = self._wav_path
        self._save_btn.configure(state="disabled", text="Saving…")
        self._set_error("")
        self._clone_msg.configure(
            text="Extracting your voice timbre — this can take a few seconds.",
            text_color=theme.TEXT_MUTED,
        )

        def _on_done(voice_id: str | None, err: str | None) -> None:
            call_in_ui(self, self._after_clone, voice_id, err)

        try:
            voice_service.clone_voice_from_wav(name, wav_path, on_done=_on_done)
        except Exception as exc:
            self._after_clone(None, str(exc))

    def _after_clone(self, voice_id: str | None, err: str | None) -> None:
        if err:
            self._save_btn.configure(state="normal", text="Save and use this voice")
            self._set_error(f"Couldn't save your voice: {err}")
            self._clone_msg.configure(text="")
            return
        self._save_btn.configure(state="disabled", text="Save and use this voice")
        self._wav_path = None
        self._name_entry.delete(0, "end")
        self._record_status.configure(text="")
        self._record_btn.configure(text="● Start recording")
        self._progress.set(0.0)
        self._clone_msg.configure(
            text=f"Saved as voice {voice_id}. Your glasses will use this voice now.",
            text_color=theme.OK,
        )
        self.refresh(refresh_speakers=False)

    def _set_error(self, text: str) -> None:
        self._error_label.configure(text=text)
        if text:
            self._error_label.pack(fill="x", padx=theme.PAD_L)
        else:
            self._error_label.pack_forget()
