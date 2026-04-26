"""Process-wide async runtime + service singletons.

Most of the heavy lifting (OpenVoice, the local-LLM HTTP client, the
trainer) is written as ``async def`` — but customtkinter's mainloop is
strictly synchronous. We solve that by spinning up a single asyncio
event loop in a background thread at import time and exposing two
helpers:

* :func:`run_coro_blocking` — for "fire-and-wait" calls that the UI is
  prepared to block on (e.g. switching the active speaker).

* :func:`submit_coro` — for "fire-and-forget" calls that should not
  freeze the UI; the caller gets a :class:`concurrent.futures.Future`
  it can poll or attach a done-callback to.

The runtime also caches the :class:`OpenVoiceTTS` engine and the
``AppState`` instance so every view sees the same objects.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

# Make the server's ``app.*`` package importable. The desktop app is a
# sibling of ``server/`` and intentionally piggybacks on every model + helper
# the FastAPI server already ships with.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVER_ROOT = _REPO_ROOT / "server"
if str(_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVER_ROOT))

from app.config import settings  # noqa: E402  (import after sys.path tweak)
from app.core.state import AppState, get_state  # noqa: E402

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Background asyncio loop
# ---------------------------------------------------------------------------
_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_ready = threading.Event()


def _loop_main() -> None:
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop_ready.set()
    try:
        _loop.run_forever()
    finally:
        try:
            pending = asyncio.all_tasks(_loop)
            for task in pending:
                task.cancel()
            if pending:
                _loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            _loop.run_until_complete(_loop.shutdown_asyncgens())
        finally:
            _loop.close()


def start_runtime() -> None:
    """Idempotent: spin up the background loop if it isn't running yet."""
    global _loop_thread
    if _loop_thread is not None and _loop_thread.is_alive():
        return
    _loop_thread = threading.Thread(
        target=_loop_main, name="async-runtime", daemon=True
    )
    _loop_thread.start()
    _loop_ready.wait(timeout=2.0)


def shutdown_runtime() -> None:
    """Stop the background loop. Safe to call multiple times."""
    global _loop, _loop_thread
    if _loop is None:
        return
    loop = _loop
    loop.call_soon_threadsafe(loop.stop)
    if _loop_thread is not None:
        _loop_thread.join(timeout=2.0)
    _loop = None
    _loop_thread = None


def submit_coro(coro: Awaitable[T]) -> Future[T]:
    """Schedule ``coro`` on the runtime loop and return a futures.Future."""
    if _loop is None:
        start_runtime()
    assert _loop is not None
    return asyncio.run_coroutine_threadsafe(coro, _loop)


def run_coro_blocking(coro: Awaitable[T], timeout: float | None = None) -> T:
    """Run ``coro`` on the runtime loop and synchronously wait for the result."""
    return submit_coro(coro).result(timeout=timeout)


# ---------------------------------------------------------------------------
# UI-thread bridging
# ---------------------------------------------------------------------------
def call_in_ui(widget, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Schedule ``fn(*args, **kwargs)`` on the Tk main thread.

    Tk widgets are not thread-safe; any update from a background thread (the
    camera, the live engine, the LLM dispatch) must come back through
    ``widget.after(0, ...)``.
    """
    try:
        widget.after(0, lambda: fn(*args, **kwargs))
    except Exception:
        # Widget destroyed during teardown — silently drop the update.
        pass


# ---------------------------------------------------------------------------
# Shared service singletons
# ---------------------------------------------------------------------------
_state: AppState | None = None
_tts_engine = None  # OpenVoiceTTS, lazily imported


def app_state() -> AppState:
    """Return the process-wide :class:`AppState` (shared with the server)."""
    global _state
    if _state is None:
        _state = get_state()
    return _state


def tts_engine():
    """Return the lazily-loaded :class:`OpenVoiceTTS` singleton.

    The engine itself is async; pair calls to its methods with
    :func:`run_coro_blocking` / :func:`submit_coro`.
    """
    global _tts_engine
    if _tts_engine is None:
        from app.tts.openvoice_client import OpenVoiceTTS

        _tts_engine = OpenVoiceTTS()
    return _tts_engine


def app_settings():
    """Re-export :mod:`app.config.settings` for views that need paths/tunables."""
    return settings


# Eagerly start the loop on import — the import order in :mod:`main` gives
# us a healthy thread before any view code runs.
start_runtime()
