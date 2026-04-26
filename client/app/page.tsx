"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, type EmotionResponse, type VoiceProfilesResponse } from "@/lib/api";

type Health = { status: string; version: string } | null;

const EMOTION_LABELS: Record<string, string> = {
  neutral: "Neutral",
  calm: "Calm",
  friendly: "Friendly",
  excited: "Excited",
  serious: "Serious",
  urgent: "Urgent",
};

export default function DashboardPage() {
  const [health, setHealth] = useState<Health>(null);
  const [voices, setVoices] = useState<VoiceProfilesResponse | null>(null);
  const [emotion, setEmotion] = useState<EmotionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [h, v, e] = await Promise.all([
          api.health(),
          api.listVoices(),
          api.getEmotion(),
        ]);
        if (cancelled) return;
        setHealth(h);
        setVoices(v);
        setEmotion(e);
      } catch (err) {
        if (!cancelled) setError(String(err));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const activeVoiceLabel =
    voices?.custom_voices.find((v) => v.voice_id === voices.active_voice_id)
      ?.name ??
    (voices?.active_voice_id === voices?.default_voice_id ? "Default" : "Default");

  const isOnline = !!health && !error;
  const emotionLabel = emotion
    ? EMOTION_LABELS[emotion.preset] ?? emotion.preset
    : null;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Welcome back</h1>
        <p className="text-slate-400 text-sm">
          A quick look at your glasses and how they sound right now.
        </p>
      </header>

      {error && (
        <div className="card border-err text-err text-sm">
          We can&apos;t reach your glasses right now. Make sure they&apos;re turned on
          and connected to the same network.
        </div>
      )}

      <section className="grid gap-4 md:grid-cols-3">
        <div className="card">
          <div className="text-xs uppercase text-slate-500">Glasses</div>
          <div className="mt-2 text-lg">
            {health ? (
              <span className="text-ok">Online</span>
            ) : error ? (
              <span className="text-err">Offline</span>
            ) : (
              <span className="text-slate-500">Connecting…</span>
            )}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            {isOnline ? "Ready to translate signs and speech." : "Waiting for a connection."}
          </div>
        </div>

        <div className="card">
          <div className="text-xs uppercase text-slate-500">Voice</div>
          <div className="mt-2 text-lg truncate">
            {activeVoiceLabel ?? "Default"}
          </div>
          <Link className="text-accent text-xs mt-2 inline-block" href="/voice">
            Change voice →
          </Link>
        </div>

        <div className="card">
          <div className="text-xs uppercase text-slate-500">Mood</div>
          <div className="mt-2 text-lg capitalize">
            {emotionLabel ?? "—"}
            {emotion && (
              <span className="text-slate-500 text-sm ml-2">
                {Math.round(emotion.intensity * 100)}%
              </span>
            )}
          </div>
          <Link className="text-accent text-xs mt-2 inline-block" href="/emotion">
            Adjust mood →
          </Link>
        </div>
      </section>

      <section className="card flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="font-medium">See what your glasses hear</h2>
          <p className="text-sm text-slate-500">
            Open the live page to watch the microphone feed get transcribed in
            real time.
          </p>
        </div>
        <Link href="/live" className="btn btn-primary">
          ● Open live captions
        </Link>
      </section>

      <section className="card">
        <h2 className="font-medium mb-2">What your glasses can do</h2>
        <ul className="text-sm text-slate-300 space-y-2">
          <li>
            <span className="text-accent font-medium">Sign → speech.</span>
            <span className="text-slate-500">
              {" "}Sign in front of the camera and your glasses speak it out loud.
            </span>
          </li>
          <li>
            <span className="text-accent font-medium">Speech → captions.</span>
            <span className="text-slate-500">
              {" "}When someone talks to you, the words show up on your lenses.
            </span>
          </li>
          <li>
            <span className="text-accent font-medium">Teach new signs.</span>
            <span className="text-slate-500">
              {" "}Show a sign a few times and your glasses will learn it.
            </span>
          </li>
        </ul>
      </section>
    </div>
  );
}
