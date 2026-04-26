"use client";

import { useCallback, useEffect, useState } from "react";
import VoiceCloneRecorder from "@/components/VoiceCloneRecorder";
import {
  api,
  type VoicePresetsResponse,
  type VoiceProfilesResponse,
} from "@/lib/api";

// MeloTTS uses a mix of `-` and `_` separators in its `spk2id` keys
// (see ``models--myshell-ai--MeloTTS-English/config.json``). The Indian
// variant in particular is `EN_INDIA` (underscore), not `EN-INDIA`. Keeping
// both spellings here means the label still maps correctly even if the
// upstream key naming changes again.
const SPEAKER_LABELS: Record<string, string> = {
  "EN-Default": "Default",
  "EN-US": "American",
  "EN-BR": "British",
  EN_INDIA: "Indian",
  "EN-INDIA": "Indian",
  "EN-AU": "Australian",
  EN_NEWEST: "Newest",
  "EN-NEWEST": "Newest",
  EN: "English",
  ES: "Spanish",
  FR: "French",
  ZH: "Chinese",
  JP: "Japanese",
  KR: "Korean",
};

const SPEAKER_HINTS: Record<string, string> = {
  "EN-Default": "Neutral English",
  "EN-US": "US accent",
  "EN-BR": "British accent",
  EN_INDIA: "Indian accent",
  "EN-INDIA": "Indian accent",
  "EN-AU": "Australian accent",
  EN_NEWEST: "Latest model voice",
  "EN-NEWEST": "Latest model voice",
  ES: "Spanish voice",
  FR: "French voice",
  ZH: "Mandarin Chinese voice",
  JP: "Japanese voice",
  KR: "Korean voice",
};

function prettyVoiceName(key: string): string {
  if (SPEAKER_LABELS[key]) return SPEAKER_LABELS[key];
  const stripped = key.replace(/^[A-Za-z]{2,3}[-_]/, "").replace(/[-_]/g, " ").trim();
  const base = stripped || key;
  return base
    .split(" ")
    .map((w) => (w ? w.charAt(0).toUpperCase() + w.slice(1).toLowerCase() : w))
    .join(" ");
}

export default function VoicePage() {
  const [data, setData] = useState<VoiceProfilesResponse | null>(null);
  const [presets, setPresets] = useState<VoicePresetsResponse | null>(null);
  const [speakerKey, setSpeakerKey] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [voices, p] = await Promise.all([
        api.listVoices(),
        api.listVoicePresets(),
      ]);
      setData(voices);
      setPresets(p);
      setSpeakerKey(p.active_speaker_key);
    } catch (err) {
      setError(String(err));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function pickProfile(voice_id: string) {
    setBusy(true);
    try {
      await api.setActiveVoice(voice_id);
      await refresh();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  async function pickPreset(key: string) {
    setSpeakerKey(key);
    setBusy(true);
    setError(null);
    try {
      await api.setActiveSpeaker(key);
      await refresh();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-2xl font-semibold">Voice</h1>
        <p className="text-slate-400 text-sm">
          Choose how your glasses sound when they speak for you.
        </p>
      </header>

      {error && (
        <div className="card border-err text-err text-sm">
          Something went wrong: {error}
        </div>
      )}

      <section className="card space-y-3">
        <h2 className="font-medium">Pick a voice</h2>
        {!presets ? (
          <p className="text-slate-500 text-sm">Loading voices…</p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {presets.speakers.map((k) => {
              const isActive = k === presets.active_speaker_key;
              return (
                <button
                  key={k}
                  className={`btn justify-start ${isActive ? "btn-primary" : ""}`}
                  onClick={() => pickPreset(k)}
                  disabled={busy || isActive}
                >
                  <div className="text-left">
                    <div className="font-medium">{prettyVoiceName(k)}</div>
                    <div className="text-xs opacity-70">
                      {isActive
                        ? "In use"
                        : SPEAKER_HINTS[k] ?? "Tap to use"}
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </section>

      {data && data.custom_voices.length > 0 && (
        <section className="card">
          <h2 className="font-medium mb-3">Your voices</h2>
          <ul className="divide-y divide-border">
            <li className="py-3 flex items-center justify-between">
              <div className="font-medium">Default</div>
              <button
                className="btn"
                onClick={() => pickProfile(data.default_voice_id)}
                disabled={busy || data.active_voice_id === data.default_voice_id}
              >
                {data.active_voice_id === data.default_voice_id ? "In use" : "Use"}
              </button>
            </li>
            {data.custom_voices.map((v) => (
              <li
                key={v.voice_id}
                className="py-3 flex items-center justify-between"
              >
                <div className="font-medium">{v.name}</div>
                <button
                  className="btn"
                  onClick={() => pickProfile(v.voice_id)}
                  disabled={busy || data.active_voice_id === v.voice_id}
                >
                  {data.active_voice_id === v.voice_id ? "In use" : "Use"}
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="card">
        <h2 className="font-medium mb-1">Use your own voice</h2>
        <p className="text-xs text-slate-500 mb-4">
          Record a short sample and your glasses will speak in your voice.
        </p>
        <VoiceCloneRecorder onCloned={refresh} />
      </section>
    </div>
  );
}
