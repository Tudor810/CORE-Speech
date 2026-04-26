"use client";

import { useEffect, useState } from "react";
import { api, type EmotionPreset } from "@/lib/api";

const PRESETS: { id: EmotionPreset; label: string; hint: string }[] = [
  { id: "neutral",  label: "Neutral",  hint: "Balanced, default delivery." },
  { id: "calm",     label: "Calm",     hint: "Soft and steady." },
  { id: "friendly", label: "Friendly", hint: "Warm and inviting." },
  { id: "excited",  label: "Excited",  hint: "Energetic, upbeat." },
  { id: "serious",  label: "Serious",  hint: "Measured and direct." },
  { id: "urgent",   label: "Urgent",   hint: "High intensity, attention-getting." },
];

export default function EmotionSelector() {
  const [preset, setPreset] = useState<EmotionPreset>("neutral");
  const [intensity, setIntensity] = useState(0.5);
  const [saved, setSaved] = useState<{ at: number; ok: boolean } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const e = await api.getEmotion();
        setPreset(e.preset);
        setIntensity(e.intensity);
      } catch (err) {
        setError(String(err));
      }
    })();
  }, []);

  async function persist(next: { preset: EmotionPreset; intensity: number }) {
    setBusy(true);
    setError(null);
    try {
      await api.setEmotion(next.preset, next.intensity);
      setSaved({ at: Date.now(), ok: true });
    } catch (err) {
      setError(String(err));
      setSaved({ at: Date.now(), ok: false });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <label className="block text-sm text-slate-300 mb-2">How should the voice feel?</label>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          {PRESETS.map((p) => (
            <button
              key={p.id}
              className={`btn justify-start ${
                preset === p.id ? "btn-primary" : ""
              }`}
              onClick={() => {
                setPreset(p.id);
                persist({ preset: p.id, intensity });
              }}
              disabled={busy}
            >
              <div className="text-left">
                <div className="font-medium">{p.label}</div>
                <div className="text-xs opacity-70">{p.hint}</div>
              </div>
            </button>
          ))}
        </div>
      </div>

      <div>
        <label className="flex items-center justify-between text-sm text-slate-300 mb-2">
          <span>How strong</span>
          <span className="text-slate-400">
            {Math.round(intensity * 100)}%
          </span>
        </label>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={intensity}
          onChange={(e) => setIntensity(parseFloat(e.target.value))}
          onMouseUp={() => persist({ preset, intensity })}
          onTouchEnd={() => persist({ preset, intensity })}
          className="w-full !p-0 !border-0 bg-transparent"
        />
      </div>

      {saved && (
        <p className={`text-xs ${saved.ok ? "text-ok" : "text-err"}`}>
          {saved.ok ? "Saved." : "Couldn’t save. Please try again."}
        </p>
      )}
      {error && (
        <p className="text-err text-sm">
          Something went wrong. Please try again.
        </p>
      )}
    </div>
  );
}
