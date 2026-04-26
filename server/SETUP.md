# Server setup (Windows)

This server requires **Python 3.11 or 3.12**.

Python 3.13+ (including 3.14) is currently incompatible with several native dependencies used here (notably `mediapipe` and parts of the `numpy`/`torch` stack) and will fail to install.

## 1) Create a virtualenv (recommended)

From `D:\projects\full_project\server`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2) Install server dependencies

```powershell
python -m pip install -r requirements.txt
```

## 3) Local STT (Vosk)

The server uses [Vosk](https://alphacephei.com/vosk/) for speech-to-text — fully local, Apache-2.0, CPU-friendly, and tiny compared to Whisper.

**Easiest path (auto-download):** leave `VOSK_MODEL_PATH=` empty in `.env`. On first transcription Vosk will fetch `VOSK_MODEL_NAME` (default `vosk-model-small-en-us-0.15`, ~40 MB) into `~/.cache/vosk` and reuse it forever.

**Manual / offline path:** download a model from https://alphacephei.com/vosk/models, unzip it, and point `VOSK_MODEL_PATH` at the resulting folder. Useful options:

| Model | Size | Notes |
|---|---|---|
| `vosk-model-small-en-us-0.15` | 40 MB | Default. Fast, decent accuracy. |
| `vosk-model-en-us-0.22` | 1.8 GB | Much higher accuracy (production-grade). |
| `vosk-model-small-es-0.42` | 39 MB | Spanish. |
| `vosk-model-small-fr-0.22` | 41 MB | French. |

```powershell
# Example: pre-stage the small EN model into the repo so production deploys
# don't need outbound network.
mkdir models\vosk
Invoke-WebRequest https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip -OutFile vosk-en.zip
Expand-Archive vosk-en.zip -DestinationPath models\vosk
# Then in .env:
#   VOSK_MODEL_PATH=models/vosk/vosk-model-small-en-us-0.15
```

## 4) Install Ollama (local LLM)

The sign-gloss → natural-sentence step now runs against a local
[Ollama](https://ollama.com/) server instead of Google Gemini. No API
key is required and inference automatically uses the GPU if one is
available (CUDA on Windows / Linux, Metal on macOS).

```powershell
# Download & install: https://ollama.com/download
# Then pull the default 3B-parameter instruction model:
ollama pull llama3.2

# Sanity-check that the daemon is up:
curl http://localhost:11434/api/tags
```

The server reads `OLLAMA_BASE_URL` / `OLLAMA_MODEL` from `.env`. Swap
to `qwen2.5:3b-instruct` (`ollama pull qwen2.5:3b-instruct`) for an
often-stronger rewrite model at the same size class. To force CPU even
on a GPU box, set `OLLAMA_NUM_GPU=0`.

## 5) Optional: install OpenVoice + MeloTTS (local TTS)

Upstream repos pin strict dependency versions (e.g. `numpy==1.22.0`) that conflict with this server.
Install them without pulling their declared deps:

```powershell
python -m pip install --no-deps git+https://github.com/myshell-ai/OpenVoice.git
python -m pip install --no-deps git+https://github.com/myshell-ai/MeloTTS.git
```

Then download UniDic (used by Japanese text processing):

```powershell
python -m unidic download
```

Finally, download `checkpoints_v2_0417.zip` from the OpenVoice repo and unzip
into:

`server/models/openvoice_v2/`

