"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ReconnectingWS } from "@/lib/ws";
import {
  api,
  type EmotionResponse,
  type VoiceProfilesResponse,
} from "@/lib/api";

const DEFAULT_DEVICE_ID = "glasses-01";
const MAX_TRANSCRIPTS = 50;
// Show "listening…" badge if the most recent transcript landed within this
// many ms — gives the UI a heartbeat without us needing a separate VAD event.
const ACTIVE_WINDOW_MS = 4000;

type ServerEvent =
  | { type: "status"; msg: string; device_id?: string }
  | {
      type: "transcript";
      text: string;
      language?: string | null;
      duration?: number;
      final?: boolean;
    }
  | { type: "error"; msg: string };

interface TranscriptEntry {
  id: string;
  text: string;
  language: string | null;
  duration: number;
  final: boolean;
  ts: number;
}

function isServerEvent(value: unknown): value is ServerEvent {
  return (
    !!value &&
    typeof value === "object" &&
    "type" in (value as Record<string, unknown>) &&
    typeof (value as { type: unknown }).type === "string"
  );
}

function makeId(): string {
  return Math.random().toString(36).slice(2, 10);
}

function formatTime(ts: number): string {
  return new Date(ts).toLocaleTimeString();
}

function statusToFriendly(msg: string, deviceId: string): string {
  switch (msg) {
    case "subscribed":
      return `Connected — listening for ${deviceId}.`;
    case "ready":
      return "Glasses are ready. Speak into the mic.";
    default:
      return msg;
  }
}

export default function LiveSession() {
  const [pendingDeviceId, setPendingDeviceId] = useState(DEFAULT_DEVICE_ID);
  const [activeDeviceId, setActiveDeviceId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [connected, setConnected] = useState(false);
  const [transcripts, setTranscripts] = useState<TranscriptEntry[]>([]);
  const [voices, setVoices] = useState<VoiceProfilesResponse | null>(null);
  const [emotion, setEmotion] = useState<EmotionResponse | null>(null);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());

  const wsRef = useRef<ReconnectingWS | null>(null);

  const refreshConfig = useCallback(async () => {
    try {
      const [v, e] = await Promise.all([api.listVoices(), api.getEmotion()]);
      setVoices(v);
      setEmotion(e);
    } catch (err) {
      setError(`Couldn’t load your voice / mood (${String(err)})`);
    }
  }, []);

  useEffect(() => {
    refreshConfig();
  }, [refreshConfig]);

  // Tick once a second so the "listening…" badge fades out on its own.
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  const teardown = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setConnected(false);
  }, []);

  useEffect(() => {
    return () => teardown();
  }, [teardown]);

  const handleEvent = useCallback(
    (deviceId: string) =>
      (ev: ServerEvent) => {
        if (ev.type === "status") {
          setStatusMsg(statusToFriendly(ev.msg, deviceId));
          return;
        }
        if (ev.type === "error") {
          setError(ev.msg);
          return;
        }
        if (ev.type === "transcript") {
          const text = (ev.text ?? "").trim();
          if (!text) return;
          const entry: TranscriptEntry = {
            id: makeId(),
            text,
            language: ev.language ?? null,
            duration: ev.duration ?? 0,
            final: ev.final ?? true,
            ts: Date.now(),
          };
          setTranscripts((prev) => [entry, ...prev].slice(0, MAX_TRANSCRIPTS));
        }
      },
    [],
  );

  const start = useCallback(() => {
    if (running) return;
    const id = pendingDeviceId.trim() || DEFAULT_DEVICE_ID;
    setActiveDeviceId(id);
    setError(null);
    setStatusMsg(`Connecting to ${id}…`);

    const ws = new ReconnectingWS(`/ws/transcripts/${id}`, {
      onOpen: () => {
        setConnected(true);
      },
      onClose: () => setConnected(false),
      onError: () => setError("Lost connection to the transcript stream."),
      onMessage: (data) => {
        if (isServerEvent(data)) handleEvent(id)(data);
      },
    });
    ws.connect();
    wsRef.current = ws;
    setRunning(true);
  }, [pendingDeviceId, running, handleEvent]);

  const stop = useCallback(() => {
    teardown();
    setRunning(false);
    setStatusMsg(null);
  }, [teardown]);

  const clearTranscripts = useCallback(() => {
    setTranscripts([]);
  }, []);

  const activeVoiceLabel = useMemo(() => {
    if (!voices) return "Default";
    if (voices.active_voice_id === voices.default_voice_id) return "Default";
    const found = voices.custom_voices.find(
      (v) => v.voice_id === voices.active_voice_id,
    );
    return found?.name ?? "Default";
  }, [voices]);

  const latest = transcripts[0] ?? null;
  const isHearingSomething =
    connected && !!latest && now - latest.ts < ACTIVE_WINDOW_MS;

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">Live captions</h1>
          <p className="text-slate-400 text-sm">
            Whatever your glasses’ microphone hears is transcribed locally
            with Vosk and shown here in real time.
          </p>
        </div>
        <div className="text-xs text-slate-500">
          Device{" "}
          <span className="text-slate-300">{activeDeviceId ?? "—"}</span>
        </div>
      </header>

      {error && (
        <div className="card border-err text-err text-sm">{error}</div>
      )}

      <section className="card space-y-4">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div className="flex-1 min-w-[200px]">
            <label className="block text-xs uppercase text-slate-500 mb-1">
              Device id
            </label>
            <input
              type="text"
              className="w-full"
              value={pendingDeviceId}
              onChange={(e) => setPendingDeviceId(e.target.value)}
              placeholder={DEFAULT_DEVICE_ID}
              disabled={running}
              spellCheck={false}
              autoCapitalize="off"
              autoCorrect="off"
            />
            <p className="text-xs text-slate-500 mt-1">
              Must match the <code>--device-id</code> the serial bridge is
              using ({DEFAULT_DEVICE_ID} by default).
            </p>
          </div>

          <div className="flex items-center gap-2">
            {!running ? (
              <button
                className="btn btn-primary"
                onClick={start}
                disabled={!pendingDeviceId.trim()}
              >
                ● Start listening
              </button>
            ) : (
              <button className="btn btn-danger" onClick={stop}>
                ■ Stop
              </button>
            )}
          </div>
        </div>

        <div className="relative w-full min-h-[260px] bg-bg/60 border border-border rounded-lg overflow-hidden p-6 flex flex-col">
          <div className="absolute top-3 left-3 right-3 flex flex-wrap items-center gap-2 text-xs">
            <span className="flex items-center gap-2 px-2 py-1 rounded-full bg-bg/70">
              <span
                className={`inline-block w-2 h-2 rounded-full ${
                  isHearingSomething
                    ? "bg-ok animate-pulse"
                    : connected
                    ? "bg-ok"
                    : running
                    ? "bg-yellow-400"
                    : "bg-slate-500"
                }`}
              />
              <span>
                {isHearingSomething
                  ? "Hearing speech…"
                  : connected
                  ? "Listening (silence)"
                  : running
                  ? "Connecting…"
                  : "Idle"}
              </span>
            </span>
            {latest?.language && (
              <span className="px-2 py-1 rounded-full bg-bg/70 uppercase tracking-wide text-slate-400">
                {latest.language}
              </span>
            )}
          </div>

          <div className="mt-10 mb-2 text-[10px] uppercase tracking-wide text-slate-500">
            Latest transcript
          </div>
          {latest ? (
            <div className="text-2xl sm:text-3xl leading-snug text-slate-100 break-words">
              {latest.text}
            </div>
          ) : (
            <div className="text-slate-500 text-sm">
              {running
                ? "Waiting for the first words…"
                : "Press “Start listening” to subscribe to the glasses’ mic."}
            </div>
          )}

          {statusMsg && (
            <div className="mt-auto pt-4 text-xs text-slate-400">
              {statusMsg}
            </div>
          )}
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        <div className="card">
          <div className="text-xs uppercase text-slate-500">Voice</div>
          <div className="mt-2 text-lg truncate">{activeVoiceLabel}</div>
          <Link className="text-accent text-xs mt-2 inline-block" href="/voice">
            Change voice →
          </Link>
        </div>
        <div className="card">
          <div className="text-xs uppercase text-slate-500">Mood</div>
          <div className="mt-2 text-lg capitalize">
            {emotion?.preset ?? "—"}
            {emotion && (
              <span className="text-slate-500 text-sm ml-2">
                {Math.round(emotion.intensity * 100)}%
              </span>
            )}
          </div>
          <Link
            className="text-accent text-xs mt-2 inline-block"
            href="/emotion"
          >
            Adjust mood →
          </Link>
        </div>
        <div className="card">
          <div className="text-xs uppercase text-slate-500">Captions stored</div>
          <div className="mt-2 text-lg">{transcripts.length}</div>
          <div className="text-xs text-slate-500 mt-1">
            Most recent {MAX_TRANSCRIPTS} are kept in this tab.
          </div>
        </div>
      </section>

      <section className="card">
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-medium">Transcript history</h2>
          {transcripts.length > 0 && (
            <button
              className="text-xs text-slate-500 hover:text-slate-300"
              onClick={clearTranscripts}
            >
              Clear
            </button>
          )}
        </div>
        {transcripts.length === 0 ? (
          <p className="text-sm text-slate-500">
            Each phrase your glasses hear will land here — newest on top.
          </p>
        ) : (
          <ul className="space-y-2">
            {transcripts.map((t) => (
              <li
                key={t.id}
                className="border border-border rounded-lg px-4 py-3 bg-bg/40"
              >
                <div className="flex items-center justify-between text-xs text-slate-500">
                  <span>{formatTime(t.ts)}</span>
                  <span className="flex items-center gap-2">
                    {t.language && (
                      <span className="uppercase tracking-wide">
                        {t.language}
                      </span>
                    )}
                    {t.duration > 0 && (
                      <span>{t.duration.toFixed(1)}s audio</span>
                    )}
                  </span>
                </div>
                <div className="mt-1 text-base text-slate-100 break-words">
                  {t.text}
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
