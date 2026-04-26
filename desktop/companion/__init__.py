"""AR Glasses Companion desktop app.

A customtkinter GUI that drives the same pipeline as
``server/scripts/live_predict.py`` (camera → MediaPipe → sign LSTM →
local-LLM → OpenVoice TTS) plus the voice / emotion / training panels
that used to live in the Next.js dashboard.

Heavy work lives in :mod:`companion.services`; the views
(:mod:`companion.views`) are thin and only marshal data between the
services and the GUI. The desktop package is intentionally NOT named
``app`` so it doesn't shadow ``server/app/`` when both are on
``sys.path``.
"""
