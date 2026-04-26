"""AR Glasses Companion — desktop entry point.

Usage::

    cd server
    .venv\\Scripts\\activate
    pip install -r ../desktop/requirements.txt
    python ../desktop/main.py

The app reuses ``server/app/*`` in-process — sign LSTM, MediaPipe Hands,
the OpenVoice/MeloTTS engine, and the local-LLM translator — without
relying on the FastAPI HTTP server. The same ``app_state.json`` is
shared with the server so the ESP32 / AR-glasses firmware keeps working
when you launch ``uvicorn app.main:app`` separately.
"""

from __future__ import annotations

import argparse
import sys


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ar-glasses-companion",
        description="Local desktop companion for the AR sign-glasses.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="OpenCV camera index to default to (overrides settings.camera_index).",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Probe cv2.VideoCapture for indices 0..5 and exit.",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable OpenVoice / fallback TTS (silent UI).",
    )
    parser.add_argument(
        "--theme",
        choices=["dark", "light", "system"],
        default="dark",
        help="customtkinter appearance mode.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if args.list_cameras:
        # Importing the camera module pulls cv2 in but no GUI / Torch.
        from companion.services.camera import list_cameras

        cams = list_cameras()
        if not cams:
            print("no cameras found.")
        else:
            print("available camera indices:")
            for i in cams:
                print(f"  {i}")
        return 0

    # Importing :mod:`companion.main_window` boots the runtime (sys.path
    # tweak, background asyncio loop, AppState load) — keep the import
    # inside the function so any errors surface with a clean traceback
    # rather than at interpreter startup.
    import customtkinter as ctk

    from companion.main_window import MainWindow

    ctk.set_appearance_mode(args.theme)

    window = MainWindow(camera_index=args.camera, tts_disabled=args.no_tts)
    try:
        window.mainloop()
    except KeyboardInterrupt:
        window._on_close()  # type: ignore[attr-defined]
    return 0


if __name__ == "__main__":
    sys.exit(main())
