"""USB-CDC bridge between an ESP32 sketch and the FastAPI server.

The ESP32 sketch streams 8-bit unsigned PCM @ 16 kHz over its USB-CDC
serial port (default 460800 baud) and accepts the same format back to
drive ``dacWrite`` on the speaker pin. ASCII control lines like
``[AUDIO_START]`` / ``[AUDIO_STOP]`` are interleaved on the same TX pipe
to mark when the microphone is live.

This script bridges that serial stream to the existing WebSocket
endpoints provided by ``app.main``:

    /ws/emission/{device_id}   mic-in  -- 16-bit PCM @ 16 kHz (binary)
                                          → transcript JSON out
                                          (Vosk local STT).
    /ws/reception/{device_id}  TTS-out -- 16-bit PCM @ 16 kHz (binary)
                                          driven by the gesture pipeline
                                          (MediaPipe → LSTM → LLM →
                                          OpenVoice).

Conversion details:

  * Mic-in (uint8 → int16):  ``s16 = (u8 - 128) << 8``  (centers around
    zero and rescales to fill the int16 range).
  * Speaker-out (int16 → uint8):  ``u8 = (s16 + 32768) >> 8``  (mirror
    of the sketch's ``map(s16, -3000, 3000, 0, 255)`` for full-range
    audio).

Speaker writes are paced to ~16 KB/s (one byte per sample @ 16 kHz) so
we don't overflow the ESP32's small USB-CDC RX buffer.

Usage::

    python -m scripts.serial_bridge                      # auto-detect ESP32 port
    python -m scripts.serial_bridge --port COM5
    python -m scripts.serial_bridge --port /dev/ttyACM0 --no-tts -v
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import signal
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np
import serial.tools.list_ports
import serial_asyncio
import websockets
from websockets.exceptions import ConnectionClosed

# ``InvalidStatusCode`` was renamed to ``InvalidStatus`` in websockets 14
# (the old alias still exists but is deprecated and prints a warning on
# import in 16.x). Accept whichever the installed version provides.
try:
    from websockets.exceptions import InvalidStatus as _InvalidStatus
except ImportError:  # pragma: no cover -- websockets < 14
    from websockets.exceptions import InvalidStatusCode as _InvalidStatus  # type: ignore[attr-defined,no-redef]

# ---------- constants ----------

SAMPLE_RATE = 16000          # must match the sketch's I2S sample_rate
DEFAULT_BAUD = 460800

# Sentinels emitted by ``Serial.println("\\n[AUDIO_START]")``. We match
# only the bracketed token so leading "\n" or trailing "\r\n" line
# endings produced by ``println`` don't matter.
SENTINEL_START = b"[AUDIO_START]"
SENTINEL_STOP = b"[AUDIO_STOP]"
SENTINEL_MAXLEN = max(len(SENTINEL_START), len(SENTINEL_STOP))

# USB Vendor IDs we treat as "probably an ESP32 dev board" for
# best-effort auto-discovery. ESP32-S2/S3/C3/... boards either expose
# the chip's native USB peripheral (Espressif VID 0x303A) or route
# UART0 through one of the cheap USB-to-UART bridge chips listed below.
ESP32_USB_VIDS: dict[int, str] = {
    0x303A: "Espressif (native USB)",
    0x10C4: "Silicon Labs CP210x",
    0x1A86: "WCH CH340 / CH341",
    0x0403: "FTDI",
}

# 1024 uint8 samples == 64 ms @ 16 kHz; small enough for snappy STT
# windowing, large enough to keep WS frame rate well under 100 fps.
MIC_FRAME_BYTES = 1024
# 256 uint8 samples == 16 ms; pacing target so we stay near 16 KB/s.
SPK_CHUNK_BYTES = 256
SPK_CHUNK_PERIOD = SPK_CHUNK_BYTES / SAMPLE_RATE  # seconds

LOG = logging.getLogger("serial_bridge")


# ---------- conversions ----------


def u8_to_s16(u8: bytes) -> bytes:
    """Convert mic samples (uint8, biased around 128) into int16 PCM."""
    if not u8:
        return b""
    arr = np.frombuffer(u8, dtype=np.uint8).astype(np.int32)
    s16 = ((arr - 128) << 8).clip(-32768, 32767).astype("<i2")
    return s16.tobytes()


def s16_to_u8(s16: bytes) -> bytes:
    """Convert TTS samples (int16 PCM) into 8-bit unsigned for ``dacWrite``."""
    if not s16:
        return b""
    n = (len(s16) // 2) * 2
    if n == 0:
        return b""
    arr = np.frombuffer(s16[:n], dtype="<i2").astype(np.int32)
    u8 = ((arr + 32768) >> 8).clip(0, 255).astype(np.uint8)
    return u8.tobytes()


# ---------- serial demux ----------


class SerialDemux:
    """Splits the ESP32's mixed ASCII-debug + binary-audio TX stream.

    The sketch uses the same serial port for two things:

    * ``Serial.println("...")``  --  ASCII debug + the
      ``[AUDIO_START]`` / ``[AUDIO_STOP]`` toggle markers.
    * ``Serial.write(byte)``     --  raw 8-bit mic samples while the
      "Sistem: PORNIT" state is active.

    The two streams are not framed; we use a tiny state machine that
    flips between TEXT and AUDIO whenever it sees a sentinel and forwards
    the active payload to the matching handler. A trailing carryover keeps
    sentinels that straddle a serial-read boundary intact.

    ``on_mode_change`` (optional) is invoked with the *new* mode string
    (``"AUDIO"`` or ``"TEXT"``) every time the sentinel toggles. The bridge
    uses it to send ``{"action":"reset"}`` to the server when the user
    presses the push-to-talk button and ``{"action":"flush"}`` when they
    release it, so trailing audio shorter than one STT window still gets
    transcribed.
    """

    def __init__(
        self,
        on_audio: Callable[[bytes], Awaitable[None]],
        on_text: Callable[[bytes], Awaitable[None]],
        on_mode_change: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._on_audio = on_audio
        self._on_text = on_text
        self._on_mode_change = on_mode_change
        self._audio_mode = False
        self._tail = b""

    @property
    def audio_mode(self) -> bool:
        return self._audio_mode

    async def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        buf = self._tail + chunk
        self._tail = b""
        i = 0
        while i < len(buf):
            target = SENTINEL_STOP if self._audio_mode else SENTINEL_START
            idx = buf.find(target, i)
            if idx == -1:
                # No sentinel found in the rest of the buffer. Emit
                # everything except the last (SENTINEL_MAXLEN - 1) bytes
                # so a sentinel split across the next read still matches.
                safe_end = len(buf) - (SENTINEL_MAXLEN - 1)
                if safe_end > i:
                    await self._emit(buf[i:safe_end])
                    self._tail = buf[safe_end:]
                else:
                    self._tail = buf[i:]
                return

            if idx > i:
                await self._emit(buf[i:idx])
            old = "AUDIO" if self._audio_mode else "TEXT"
            self._audio_mode = not self._audio_mode
            new = "AUDIO" if self._audio_mode else "TEXT"
            LOG.info("ESP32 mode: %s -> %s", old, new)
            if self._on_mode_change is not None:
                await self._on_mode_change(new)
            i = idx + len(target)

    async def _emit(self, payload: bytes) -> None:
        if not payload:
            return
        if self._audio_mode:
            await self._on_audio(payload)
        else:
            await self._on_text(payload)


# ---------- port discovery ----------


def find_esp32_port() -> str | None:
    """Return the first plausible ESP32 COM port, or ``None``.

    Native-USB Espressif chips (VID 0x303A) win over generic USB-UART
    bridges (CH340 / CP210x / FTDI) when both are present, since the
    native one is almost always the intended connection. Within each
    tier, port enumeration order decides; pass ``--port`` to override.
    """
    by_vid: dict[int, list[tuple[str, str]]] = {}
    for p in serial.tools.list_ports.comports():
        vid = getattr(p, "vid", None)
        if vid in ESP32_USB_VIDS:
            by_vid.setdefault(vid, []).append((p.device, ESP32_USB_VIDS[vid]))

    # Prefer native-USB Espressif ports, then bridge chips in dict order.
    for vid in (0x303A, 0x10C4, 0x1A86, 0x0403):
        candidates = by_vid.get(vid)
        if not candidates:
            continue
        device, label = candidates[0]
        if len(candidates) > 1:
            LOG.warning(
                "Multiple %s ports found (%s); using %s. Pass --port to override.",
                label, ", ".join(d for d, _ in candidates), device,
            )
        else:
            LOG.info("Auto-detected ESP32 on %s (%s)", device, label)
        return device
    return None


def _describe_available_ports() -> str:
    """Format the list of currently visible serial ports for error messages."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return "no serial ports detected on this system"
    lines = []
    for p in ports:
        vid = getattr(p, "vid", None)
        pid = getattr(p, "pid", None)
        ids = f" [VID:PID={vid:04X}:{pid:04X}]" if vid and pid else ""
        desc = (p.description or "").strip()
        lines.append(f"  {p.device}{ids}  {desc}".rstrip())
    return "available serial ports:\n" + "\n".join(lines)


# ---------- WebSocket workers ----------


# Sentinel values that travel on the same queue as PCM frames so a single
# consumer can preserve mic-vs-control ordering. We use plain strings instead
# of dicts/dataclasses because they're trivially identity-checked in the
# sender hot loop.
CTRL_FLUSH = "__ctrl_flush__"
CTRL_RESET = "__ctrl_reset__"

# Type alias for clarity. Items on the audio queue are either:
#   * bytes  -- a uint8 mic frame to be converted + sent as binary,
#   * str    -- one of the CTRL_* sentinels above (control message),
#   * None   -- shutdown signal from the serial reader.
EmissionItem = bytes | str | None


async def emission_worker(
    server: str,
    device_id: str,
    audio_q: "asyncio.Queue[EmissionItem]",
    *,
    transcript_log: Path | None,
) -> None:
    """Push mic frames to ``/ws/emission`` and print transcripts."""
    url = f"{server.rstrip('/')}/ws/emission/{device_id}"
    backoff = 1.0
    while True:
        LOG.info("Connecting to %s", url)
        try:
            async with websockets.connect(url, max_size=None) as ws:
                LOG.info("Emission WS connected")
                backoff = 1.0
                send_task = asyncio.create_task(
                    _send_audio(ws, audio_q), name="emission-send"
                )
                recv_task = asyncio.create_task(
                    _recv_transcripts(ws, transcript_log), name="emission-recv"
                )
                done, pending = await asyncio.wait(
                    [send_task, recv_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await t
                for t in done:
                    exc = t.exception()
                    if exc is not None and not isinstance(
                        exc, (ConnectionClosed, asyncio.CancelledError)
                    ):
                        raise exc
        except asyncio.CancelledError:
            raise
        except (OSError, ConnectionClosed, _InvalidStatus) as exc:
            LOG.warning(
                "Emission WS error: %s -- retrying in %.1fs", exc, backoff
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 15.0)


async def _send_audio(ws, audio_q: "asyncio.Queue[EmissionItem]") -> None:
    while True:
        item = await audio_q.get()
        if item is None:
            return
        if isinstance(item, str):
            # Control verb -- forward as a small JSON action frame. Matches
            # the server's parser in app/ws/emission_ws.py.
            if item == CTRL_FLUSH:
                await ws.send('{"action":"flush"}')
            elif item == CTRL_RESET:
                await ws.send('{"action":"reset"}')
            else:
                LOG.debug("Unknown emission control verb: %r", item)
            continue
        s16 = u8_to_s16(item)
        if s16:
            await ws.send(s16)


async def _recv_transcripts(ws, transcript_log: Path | None) -> None:
    async for msg in ws:
        if isinstance(msg, (bytes, bytearray)):
            continue
        try:
            payload = json.loads(msg)
        except json.JSONDecodeError:
            LOG.debug("Non-JSON emission text: %r", msg)
            continue
        kind = payload.get("type")
        if kind == "transcript":
            text = (payload.get("text") or "").strip()
            if text:
                LOG.info("Transcript: %s", text)
                if transcript_log is not None:
                    try:
                        with transcript_log.open("a", encoding="utf-8") as f:
                            f.write(text + "\n")
                    except OSError as exc:
                        LOG.warning("Could not write transcript log: %s", exc)
        elif kind == "status":
            LOG.debug("Emission status: %s", payload.get("msg"))
        elif kind == "error":
            LOG.warning("Emission error: %s", payload.get("msg"))
        else:
            LOG.debug("Emission msg: %s", payload)


async def reception_worker(
    server: str,
    device_id: str,
    spk_q: "asyncio.Queue[bytes]",
) -> None:
    """Subscribe to ``/ws/reception`` and forward TTS audio to the speaker."""
    url = f"{server.rstrip('/')}/ws/reception/{device_id}"
    backoff = 1.0
    while True:
        LOG.info("Connecting to %s", url)
        try:
            async with websockets.connect(url, max_size=None) as ws:
                LOG.info("Reception WS connected")
                backoff = 1.0
                async for msg in ws:
                    if isinstance(msg, (bytes, bytearray)):
                        u8 = s16_to_u8(bytes(msg))
                        if u8:
                            await spk_q.put(u8)
                        continue
                    try:
                        payload = json.loads(msg)
                    except json.JSONDecodeError:
                        LOG.debug("Non-JSON reception text: %r", msg)
                        continue
                    kind = payload.get("type")
                    if kind == "phrase":
                        LOG.info("Phrase: %s", " ".join(payload.get("tokens", [])))
                    elif kind == "translation":
                        LOG.info("Translation: %s", payload.get("text", ""))
                    elif kind == "gesture":
                        LOG.debug(
                            "Gesture: %s (%.2f)",
                            payload.get("label"),
                            payload.get("confidence", 0.0),
                        )
                    elif kind == "audio_end":
                        LOG.debug(
                            "Audio end (chunks=%s)", payload.get("chunks")
                        )
                    elif kind == "status":
                        LOG.debug("Reception status: %s", payload.get("msg"))
                    elif kind == "error":
                        LOG.warning("Reception error: %s", payload.get("msg"))
                    else:
                        LOG.debug("Reception msg: %s", payload)
        except asyncio.CancelledError:
            raise
        except (OSError, ConnectionClosed, _InvalidStatus) as exc:
            LOG.warning(
                "Reception WS error: %s -- retrying in %.1fs", exc, backoff
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 15.0)


# ---------- serial workers ----------


async def serial_reader_loop(
    reader: asyncio.StreamReader,
    audio_q: "asyncio.Queue[EmissionItem]",
) -> None:
    """Read serial bytes, demux text vs audio, batch audio into WS frames."""
    audio_buf = bytearray()

    async def on_audio(payload: bytes) -> None:
        audio_buf.extend(payload)
        while len(audio_buf) >= MIC_FRAME_BYTES:
            frame = bytes(audio_buf[:MIC_FRAME_BYTES])
            del audio_buf[:MIC_FRAME_BYTES]
            try:
                audio_q.put_nowait(frame)
            except asyncio.QueueFull:
                # Drop the oldest mic frame so we don't lag behind realtime.
                with contextlib.suppress(asyncio.QueueEmpty):
                    audio_q.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    audio_q.put_nowait(frame)

    async def on_text(payload: bytes) -> None:
        try:
            text = payload.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                LOG.info("ESP32: %s", stripped)

    async def on_mode_change(new_mode: str) -> None:
        # Push-to-talk semantics: clear any pre-recorded buffer on press,
        # commit any trailing fragment on release. Both are best-effort --
        # if the queue is full we drop the verb rather than block the
        # serial reader (the audio in-flight is more important).
        if new_mode == "AUDIO":
            audio_buf.clear()
            verb: EmissionItem = CTRL_RESET
        else:
            # Flush whatever uint8 leftover hasn't filled MIC_FRAME_BYTES yet
            # so the server gets every last sample of this utterance.
            if audio_buf:
                tail = bytes(audio_buf)
                audio_buf.clear()
                with contextlib.suppress(asyncio.QueueFull):
                    audio_q.put_nowait(tail)
            verb = CTRL_FLUSH
        with contextlib.suppress(asyncio.QueueFull):
            audio_q.put_nowait(verb)

    demux = SerialDemux(
        on_audio=on_audio, on_text=on_text, on_mode_change=on_mode_change
    )

    while True:
        chunk = await reader.read(2048)
        if not chunk:
            LOG.warning("Serial EOF -- COM port closed?")
            await audio_q.put(None)
            return
        await demux.feed(chunk)


async def serial_writer_loop(
    writer: asyncio.StreamWriter,
    spk_q: "asyncio.Queue[bytes]",
) -> None:
    """Drain ``spk_q`` to serial at ~16 KB/s using a deadline-paced loop."""
    pending = bytearray()
    loop = asyncio.get_running_loop()
    next_write = loop.time()
    while True:
        if not pending:
            chunk = await spk_q.get()
            pending.extend(chunk)
            # Reset the pacing clock at the start of every burst so a
            # long silent gap doesn't make us splat the next sentence.
            next_write = loop.time()
        out_len = min(SPK_CHUNK_BYTES, len(pending))
        out = bytes(pending[:out_len])
        del pending[:out_len]
        writer.write(out)
        with contextlib.suppress(ConnectionResetError):
            await writer.drain()
        next_write += out_len / SAMPLE_RATE
        delay = next_write - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)


# ---------- entry point ----------


async def run(args: argparse.Namespace) -> None:
    port = args.port or find_esp32_port()
    if not port:
        raise SystemExit(
            "Could not find an ESP32 COM port automatically.\n"
            f"{_describe_available_ports()}\n"
            "Pass --port (e.g. --port COM3) or check the USB cable."
        )
    LOG.info("Opening serial %s @ %d baud", port, args.baud)
    try:
        reader, writer = await serial_asyncio.open_serial_connection(
            url=port, baudrate=args.baud
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Could not open {port}: {exc}\n{_describe_available_ports()}"
        ) from exc

    audio_q: asyncio.Queue[EmissionItem] = asyncio.Queue(maxsize=64)
    spk_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=128)

    transcript_log: Path | None = None
    if args.log_transcripts:
        transcript_log = Path(args.log_transcripts).expanduser().resolve()
        transcript_log.parent.mkdir(parents=True, exist_ok=True)
        LOG.info("Appending transcripts to %s", transcript_log)

    tasks: list[asyncio.Task] = [
        asyncio.create_task(serial_reader_loop(reader, audio_q),
                            name="serial-reader"),
    ]

    if not args.no_stt:
        tasks.append(asyncio.create_task(
            emission_worker(args.server, args.device_id, audio_q,
                            transcript_log=transcript_log),
            name="ws-emission",
        ))
    else:
        # Drain the queue so the reader never deadlocks on backpressure.
        async def _drain() -> None:
            while True:
                item = await audio_q.get()
                if item is None:
                    return
        tasks.append(asyncio.create_task(_drain(), name="ws-emission-disabled"))

    if not args.no_tts:
        tasks.append(asyncio.create_task(
            serial_writer_loop(writer, spk_q), name="serial-writer"
        ))
        tasks.append(asyncio.create_task(
            reception_worker(args.server, args.device_id, spk_q),
            name="ws-reception",
        ))

    stop = asyncio.Event()

    def _on_signal(*_: object) -> None:
        LOG.info("Signal received, shutting down...")
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, AttributeError, ValueError):
            loop.add_signal_handler(sig, _on_signal)

    stop_task = asyncio.create_task(stop.wait(), name="stop-waiter")
    try:
        done, _ = await asyncio.wait(
            [stop_task, *tasks], return_when=asyncio.FIRST_COMPLETED
        )
        for t in done:
            if t is stop_task:
                continue
            exc = t.exception()
            if exc is not None and not isinstance(exc, asyncio.CancelledError):
                LOG.error("Task %s exited: %s", t.get_name(), exc)
    finally:
        for t in [stop_task, *tasks]:
            t.cancel()
        for t in [stop_task, *tasks]:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        with contextlib.suppress(Exception):
            writer.close()
        LOG.info("Bye.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--port",
        help=(
            "Serial port (e.g. COM5 on Windows, /dev/ttyACM0 on Linux). "
            "Auto-detected by Espressif USB VID if omitted."
        ),
    )
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD,
                   help=f"Baud rate (default {DEFAULT_BAUD}).")
    p.add_argument("--server", default="ws://localhost:8000",
                   help="FastAPI base URL for WS endpoints (default ws://localhost:8000).")
    p.add_argument("--device-id", default="glasses-01",
                   help="Logical device id used in the WS path (default glasses-01).")
    p.add_argument("--no-stt", action="store_true",
                   help="Don't connect /ws/emission (mic samples are dropped).")
    p.add_argument("--no-tts", action="store_true",
                   help="Don't connect /ws/reception (no audio to speaker).")
    p.add_argument("--log-transcripts", default=None,
                   help="Optional file path; transcripts are appended one per line.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Enable debug logging.")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
