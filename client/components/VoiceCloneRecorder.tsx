"use client";

import { useEffect, useRef, useState } from "react";
import { api, HttpError } from "@/lib/api";

const MAX_SECONDS = 6;
const TARGET_SAMPLE_RATE = 16000;

type AudioCtxCtor = typeof AudioContext;

function getAudioContextCtor(): AudioCtxCtor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    AudioContext?: AudioCtxCtor;
    webkitAudioContext?: AudioCtxCtor;
  };
  return w.AudioContext ?? w.webkitAudioContext ?? null;
}

function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16 * 8 / 8, true); // bits per sample
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    s = s < 0 ? s * 0x8000 : s * 0x7fff;
    view.setInt16(offset, s, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

function downmixToMono(buffer: AudioBuffer): Float32Array {
  if (buffer.numberOfChannels === 1) {
    return buffer.getChannelData(0).slice(0);
  }
  const length = buffer.length;
  const out = new Float32Array(length);
  for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
    const data = buffer.getChannelData(ch);
    for (let i = 0; i < length; i++) out[i] += data[i];
  }
  for (let i = 0; i < length; i++) out[i] /= buffer.numberOfChannels;
  return out;
}

async function blobToWav(blob: Blob, targetRate = TARGET_SAMPLE_RATE): Promise<Blob> {
  const Ctor = getAudioContextCtor();
  if (!Ctor) throw new Error("AudioContext is not supported in this browser");

  const arrayBuf = await blob.arrayBuffer();

  // Decode at the input file's native rate, then resample with OfflineAudioContext.
  const decoder = new Ctor();
  let decoded: AudioBuffer;
  try {
    decoded = await decoder.decodeAudioData(arrayBuf.slice(0));
  } finally {
    void decoder.close();
  }

  const channels = decoded.numberOfChannels;
  const targetLength = Math.max(
    1,
    Math.round((decoded.length * targetRate) / decoded.sampleRate),
  );

  const offline = new OfflineAudioContext(channels, targetLength, targetRate);
  const src = offline.createBufferSource();
  src.buffer = decoded;
  src.connect(offline.destination);
  src.start(0);
  const rendered = await offline.startRendering();
  const mono = downmixToMono(rendered);
  return encodeWav(mono, targetRate);
}

interface Props {
  onCloned?: () => void;
}

export default function VoiceCloneRecorder({ onCloned }: Props) {
  const [name, setName] = useState("");
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [blob, setBlob] = useState<Blob | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const tickRef = useRef<number | null>(null);
  const stopAtRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      stopAll();
    };
  }, []);

  function stopAll() {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
    recorderRef.current?.stream.getTracks().forEach((t) => t.stop());
    if (tickRef.current != null) {
      window.clearInterval(tickRef.current);
      tickRef.current = null;
    }
  }

  async function startRecording() {
    setError(null);
    setBlob(null);
    setSeconds(0);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };
      mr.onstop = () => {
        const out = new Blob(chunksRef.current, { type: mr.mimeType || "audio/webm" });
        setBlob(out);
        setRecording(false);
        stream.getTracks().forEach((t) => t.stop());
        if (tickRef.current != null) {
          window.clearInterval(tickRef.current);
          tickRef.current = null;
        }
      };
      mr.start();
      recorderRef.current = mr;
      stopAtRef.current = Date.now() + MAX_SECONDS * 1000;
      setRecording(true);
      tickRef.current = window.setInterval(() => {
        const remaining = (stopAtRef.current ?? 0) - Date.now();
        const elapsed = MAX_SECONDS - Math.max(0, Math.ceil(remaining / 1000));
        setSeconds(Math.min(MAX_SECONDS, elapsed));
        if (remaining <= 0) {
          mr.stop();
        }
      }, 100);
    } catch (err) {
      setError(`microphone error: ${err}`);
      setRecording(false);
    }
  }

  function stopRecording() {
    recorderRef.current?.stop();
  }

  async function upload() {
    if (!blob || !name.trim()) return;
    setSubmitting(true);
    setError(null);
    try {
      let wav: Blob;
      try {
        wav = await blobToWav(blob);
      } catch (convErr) {
        throw new Error(
          `couldn't process the recording in your browser (${String(convErr)})`,
        );
      }
      await api.cloneVoice(name.trim(), wav, "sample.wav");
      setBlob(null);
      setName("");
      setSeconds(0);
      onCloned?.();
    } catch (err) {
      let msg: string;
      if (err instanceof HttpError) {
        const body = err.body as { detail?: unknown } | undefined;
        const detail =
          (typeof body?.detail === "string" && body.detail) ||
          (body?.detail != null ? JSON.stringify(body.detail) : "");
        msg = detail
          ? `${detail} (HTTP ${err.status})`
          : `server returned HTTP ${err.status}`;
      } else if (err instanceof Error) {
        msg = err.message;
      } else {
        msg = String(err);
      }
      setError(msg);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm text-slate-300 mb-1">Name this voice</label>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="My voice"
          className="w-full"
          disabled={recording || submitting}
        />
      </div>

      <div className="flex items-center gap-3">
        {!recording ? (
          <button className="btn btn-primary" onClick={startRecording} disabled={submitting}>
            ● Start recording
          </button>
        ) : (
          <button className="btn btn-danger" onClick={stopRecording}>
            ■ Stop ({seconds}s / {MAX_SECONDS}s)
          </button>
        )}

        {blob && !recording && (
          <span className="text-sm text-slate-400">Recording ready</span>
        )}
      </div>

      {recording && (
        <div className="h-1.5 bg-border rounded overflow-hidden">
          <div
            className="h-full bg-accent transition-all"
            style={{ width: `${(seconds / MAX_SECONDS) * 100}%` }}
          />
        </div>
      )}

      {blob && !recording && (
        <audio
          src={URL.createObjectURL(blob)}
          controls
          className="w-full mt-2 rounded"
        />
      )}

      <button
        className="btn btn-primary"
        disabled={!blob || !name.trim() || submitting}
        onClick={upload}
      >
        {submitting ? "Saving…" : "Save and use this voice"}
      </button>

      {error && (
        <p className="text-err text-sm">
          We couldn&apos;t save your voice: {error}
        </p>
      )}
      <p className="text-xs text-slate-500">
        Speak naturally for up to {MAX_SECONDS} seconds. Your recording stays on your own device.
      </p>
    </div>
  );
}
