# AR Glasses Companion — Desktop App

A fully-local, in-process replacement for the Next.js `client/` dashboard.
Built with [customtkinter](https://github.com/TomSchimansky/CustomTkinter)
and modelled directly on `server/scripts/live_predict.py` — the camera,
MediaPipe Hands, the trained sign LSTM, the local-LLM translator and the
OpenVoice + MeloTTS TTS engine all run inside the same Python process as
the GUI. No HTTP, no WebSocket, no second server needed.

The original FastAPI server (`server/app/main.py`) is **unchanged** so the
ESP32 / AR-glasses path keeps working.

## Features

- **Home** — live status of the cameras, current voice, current mood.
- **Live** — one-tap sign-to-speech session: webcam preview with hand
  landmarks, rolling sign tokens, the local-LLM rewrite, and spoken audio
  through your active voice. Same state machine as `live_predict.py`
  (sliding window, motion-pause segmentation, sentence-timeout dispatch).
- **Voice** — pick any installed MeloTTS speaker preset (US / British /
  Indian / Australian / Spanish / French / ...) **or** clone your own
  voice from a 6-second microphone recording.
- **Mood** — six emotional tone presets with an intensity slider that
  feed the LLM prompt and adjust the TTS speed exactly like the server.
- **Teach signs** — capture 30-frame sample windows for a new sign label
  and trigger an in-process LSTM fine-tune. The trained checkpoint hot-
  reloads into the live engine on the next session.

State (active voice id, MeloTTS speaker key, emotion, custom-voice list)
is persisted to `server/data/app_state.json` — the same file the FastAPI
server reads — so changes here are picked up by the AR-glasses path on
the next request.

## Install

The desktop app reuses every heavy dependency from `server/requirements.txt`
(torch, mediapipe, opencv, librosa, OpenVoice, MeloTTS, faster-whisper,
…). Install it inside the **same virtualenv**:

```powershell
cd server
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# OpenVoice / MeloTTS one-shot setup (only needed once):
python -m unidic download
# ...and unzip the OpenVoice v2 checkpoints into server/models/openvoice_v2/
# (see server/SETUP.md for details).

# Then add the desktop-only GUI deps:
pip install -r ..\desktop\requirements.txt
```

You also need:

- A webcam (cv2 DirectShow on Windows, V4L2 on Linux, AVFoundation on macOS).
- A trained sign LSTM checkpoint at `server/models/sign_lstm.pt` (built by
  `scripts/train_from_dataset.py`, or by clicking **Teach my glasses** in
  the Teach signs tab).
- [Ollama](https://ollama.com) running locally with the model from
  `server/.env` pulled (e.g. `ollama pull llama3.2`). The translator falls
  back to a tiny offline stub if Ollama is unreachable.

## Run

From the repo root, with the venv active:

```powershell
python desktop\main.py
```

Optional CLI flags:

| Flag | Default | Purpose |
| ---- | ------- | ------- |
| `--camera N` | from `server/.env` (`camera_index`) | OpenCV index (0 = built-in laptop cam, 1 = first external). |
| `--list-cameras` | — | Probe `cv2.VideoCapture` indices 0..5 and exit. |
| `--no-tts` | — | Skip OpenVoice / pyttsx3 entirely (silent UI). |
| `--theme dark\|light\|system` | `dark` | Force the customtkinter color mode. |

## Project layout

```
desktop/
├── main.py                  Entry point (argparse + customtkinter mainloop)
├── requirements.txt         GUI-only deps (server reqs are reused)
├── README.md
└── companion/                  (NB: not "app" — that name is reserved
    │                           for the server's package we import from)
    ├── theme.py             Color + spacing constants
    ├── runtime.py           Background asyncio loop + service singletons
    ├── main_window.py       Main window with sidebar nav
    ├── services/
    │   ├── camera.py            Threaded cv2 webcam capture
    │   ├── live_engine.py       Sliding-window sign engine (port of live_predict.py)
    │   ├── tts_worker.py        OpenVoice in-process synth + pygame playback
    │   ├── voice.py             Voice presets + cloning (wraps OpenVoiceTTS)
    │   ├── training.py          Sample capture + LSTM trainer wrapper
    │   └── audio_recorder.py    Microphone capture (sounddevice)
    ├── widgets/
    │   └── camera_view.py       CTk video display with hand-landmark overlay
    └── views/
        ├── home_view.py
        ├── live_view.py
        ├── voice_view.py
        ├── emotion_view.py
        └── training_view.py
```

## Differences vs. the old web client

| Old (Next.js) | New (CTk desktop) |
| ------------- | ----------------- |
| Browser MediaRecorder for cloning | `sounddevice` mic capture, written straight to WAV |
| WS frame upload to `/ws/reception_browser` | In-process cv2 → MediaPipe → LSTM → TTS, no network hop |
| WS landmark stream to `/ws/training` | In-process landmark stream into the training view |
| REST `/api/voice/*`, `/api/emotion`, `/api/training/*` | Direct calls into `app.tts.openvoice_client`, `app.core.state`, `app.ml.trainer` |
| Audio decoded with `AudioContext` | PCM played by `pygame.mixer` |

The end result is one window, one process — same UX, dramatically lower
latency, and no need to keep the FastAPI server running just to use the
companion.
