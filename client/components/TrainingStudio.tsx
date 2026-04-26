"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ReconnectingWS } from "@/lib/ws";
import { api, type SamplesResponse } from "@/lib/api";

type LandmarkFrame = {
  hands: number[][][]; // [[ [x,y,z]*21 ], ...]
  handedness: string[];
  has_hand: boolean;
};

type ServerEvent =
  | { type: "landmarks"; frame: LandmarkFrame; ts: number }
  | { type: "captured"; label: string; samples: number }
  | { type: "status"; msg: string };

const HAND_EDGES: Array<[number, number]> = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
];

export default function TrainingStudio() {
  // Generate the session id on the client only. Doing this during render
  // (e.g. with useMemo) makes the SSR-rendered HTML disagree with the client's
  // first render and triggers a React hydration error.
  const [sessionId, setSessionId] = useState<string | null>(null);
  const wsRef = useRef<ReconnectingWS | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const lastFrameRef = useRef<LandmarkFrame | null>(null);

  const [connected, setConnected] = useState(false);
  const [label, setLabel] = useState("");
  const [capturing, setCapturing] = useState(false);
  const [samples, setSamples] = useState<SamplesResponse | null>(null);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [training, setTraining] = useState(false);

  useEffect(() => {
    setSessionId(`studio-${Math.random().toString(36).slice(2, 8)}`);
  }, []);

  const refreshSamples = useCallback(async () => {
    try {
      setSamples(await api.listSamples());
    } catch {
      setStatusMsg("Couldn’t load your saved signs.");
    }
  }, []);

  useEffect(() => {
    refreshSamples();
  }, [refreshSamples]);

  useEffect(() => {
    if (!sessionId) return;
    const ws = new ReconnectingWS(`/ws/training/${sessionId}`, {
      onOpen: () => setConnected(true),
      onClose: () => setConnected(false),
      onMessage: (data) => {
        const ev = data as ServerEvent;
        if (!ev || typeof ev !== "object" || !("type" in ev)) return;
        if (ev.type === "landmarks") {
          lastFrameRef.current = ev.frame;
          drawFrame(canvasRef.current, ev.frame);
        } else if (ev.type === "captured") {
          setStatusMsg(
            `Saved “${ev.label}” — you have ${ev.samples} example${
              ev.samples === 1 ? "" : "s"
            } so far.`,
          );
          refreshSamples();
        }
      },
    });
    wsRef.current = ws;
    ws.connect();
    return () => ws.close();
  }, [sessionId, refreshSamples]);

  function startCapture() {
    if (!label.trim()) return;
    wsRef.current?.send({ action: "start_capture", label: label.trim() });
    setCapturing(true);
  }

  function stopCapture() {
    wsRef.current?.send({ action: "stop_capture" });
    setCapturing(false);
  }

  function saveSample() {
    wsRef.current?.send({ action: "save_sample" });
  }

  async function train() {
    setTraining(true);
    setStatusMsg(null);
    try {
      await api.trainModel(25);
      setStatusMsg("Your glasses are learning the new signs. This can take a moment.");
    } catch {
      setStatusMsg("Couldn’t start learning. Please try again.");
    } finally {
      setTraining(false);
    }
  }

  async function loadStarter() {
    try {
      await api.loadDefaultPack();
      setStatusMsg("Loaded a starter pack of common signs.");
      refreshSamples();
    } catch {
      setStatusMsg("Couldn’t load the starter signs.");
    }
  }

  return (
    <div className="grid lg:grid-cols-2 gap-6">
      <section className="card">
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-medium">Camera</h2>
          <span className={`text-xs ${connected ? "text-ok" : "text-warn"}`}>
            {connected ? "● connected" : "● connecting…"}
          </span>
        </div>
        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          className="w-full aspect-[4/3] bg-black rounded-lg border border-border"
        />
        <p className="text-xs text-slate-500 mt-2">
          We only track the position of your hands. No video is recorded.
        </p>
      </section>

      <section className="space-y-4">
        <div className="card space-y-3">
          <h2 className="font-medium">Add a new sign</h2>
          <input
            placeholder="What does this sign mean? (e.g. hello)"
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            disabled={capturing}
            className="w-full"
          />
          <div className="flex gap-2">
            {!capturing ? (
              <button
                className="btn btn-primary"
                onClick={startCapture}
                disabled={!label.trim() || !connected}
              >
                Start
              </button>
            ) : (
              <button className="btn btn-danger" onClick={stopCapture}>
                Done
              </button>
            )}
            <button
              className="btn"
              disabled={!capturing}
              onClick={saveSample}
              title="Save what you’re showing right now as one example"
            >
              + Save example
            </button>
          </div>
          <p className="text-xs text-slate-500">
            Show the sign in front of the camera for about a second, then click
            “Save example.” Save it 20–30 times for the best results.
          </p>
        </div>

        <div className="card space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="font-medium">Your signs</h2>
            <button className="btn" onClick={refreshSamples}>
              Refresh
            </button>
          </div>
          {!samples || samples.total === 0 ? (
            <p className="text-slate-500 text-sm">
              You haven’t saved any signs yet.
            </p>
          ) : (
            <ul className="text-sm divide-y divide-border">
              {Object.entries(samples.labels).map(([k, v]) => (
                <li key={k} className="py-2 flex justify-between text-slate-300">
                  <span className="font-medium">{k}</span>
                  <span className="text-slate-500">
                    {v} example{v === 1 ? "" : "s"}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="card flex flex-wrap gap-2">
          <button className="btn btn-primary" onClick={train} disabled={training}>
            {training ? "Starting…" : "Teach my glasses"}
          </button>
          <button className="btn" onClick={loadStarter}>
            Load starter signs
          </button>
        </div>

        {statusMsg && (
          <div className="card text-sm text-slate-300">{statusMsg}</div>
        )}
      </section>
    </div>
  );
}

function drawFrame(canvas: HTMLCanvasElement | null, frame: LandmarkFrame) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, W, H);

  if (!frame.has_hand) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "14px system-ui";
    ctx.fillText("Show your hand to the camera", 12, H - 12);
    return;
  }

  frame.hands.forEach((lm, idx) => {
    ctx.strokeStyle = idx === 0 ? "#5b8cff" : "#3fc28d";
    ctx.fillStyle = idx === 0 ? "#5b8cff" : "#3fc28d";
    ctx.lineWidth = 2;
    ctx.beginPath();
    HAND_EDGES.forEach(([a, b]) => {
      const pa = lm[a];
      const pb = lm[b];
      if (!pa || !pb) return;
      ctx.moveTo(pa[0] * W, pa[1] * H);
      ctx.lineTo(pb[0] * W, pb[1] * H);
    });
    ctx.stroke();

    lm.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(x * W, y * H, 3, 0, Math.PI * 2);
      ctx.fill();
    });
  });
}
