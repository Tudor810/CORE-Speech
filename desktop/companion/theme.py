"""Color + spacing tokens shared by every view.

Mirrors the dark palette used by the previous Next.js dashboard so the
desktop app feels like the same product.
"""

from __future__ import annotations

# Background / surface colors (used as ``fg_color`` on customtkinter frames).
BG = "#0b1020"          # window background
PANEL = "#10172a"       # raised panel ("card")
PANEL_ALT = "#162038"   # subtle surface inside a panel
BORDER = "#1f2a44"      # divider / outline color

# Text.
TEXT = "#e2e8f0"
TEXT_MUTED = "#94a3b8"
TEXT_DIM = "#64748b"

# Accents and status.
ACCENT = "#5b8cff"      # primary brand blue
ACCENT_HOVER = "#3b6cff"
OK = "#22c55e"
WARN = "#f59e0b"
ERR = "#ef4444"

# Numeric tokens for consistent spacing.
PAD_S = 6
PAD_M = 12
PAD_L = 18
PAD_XL = 28

CARD_RADIUS = 12
BUTTON_RADIUS = 8

SIDEBAR_WIDTH = 220
