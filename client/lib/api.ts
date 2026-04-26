// Typed fetch wrapper for the FastAPI backend.
// Reads NEXT_PUBLIC_API_BASE; defaults to http://localhost:8000.

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://localhost:8000";

export type EmotionPreset =
  | "neutral"
  | "friendly"
  | "serious"
  | "urgent"
  | "calm"
  | "excited";

export interface VoiceProfile {
  voice_id: string;
  name: string;
  is_default: boolean;
}

export interface VoiceProfilesResponse {
  active_voice_id: string;
  default_voice_id: string;
  active_speaker_key?: string;
  tts_language?: string;
  custom_voices: VoiceProfile[];
}

export interface VoicePresetsResponse {
  tts_language: string;
  active_speaker_key: string;
  speakers: string[];
}

export interface EmotionResponse {
  preset: EmotionPreset;
  intensity: number;
  available?: EmotionPreset[];
}

export interface SamplesResponse {
  labels: Record<string, number>;
  total: number;
}

class HttpError extends Error {
  constructor(
    public status: number,
    message: string,
    public body?: unknown,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(init?.body && !(init.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    let body: unknown = undefined;
    try {
      body = await res.json();
    } catch {
      /* ignore */
    }
    throw new HttpError(res.status, `HTTP ${res.status} on ${path}`, body);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  health: () => request<{ status: string; version: string }>("/api/health"),

  // Voice
  listVoices: () => request<VoiceProfilesResponse>("/api/voice/profiles"),
  setActiveVoice: (voice_id: string) =>
    request<{ active_voice_id: string }>("/api/voice/profiles/active", {
      method: "PUT",
      body: JSON.stringify({ voice_id }),
    }),
  listVoicePresets: () => request<VoicePresetsResponse>("/api/voice/presets"),
  setActiveSpeaker: (speaker_key: string) =>
    request<{ active_speaker_key: string }>("/api/voice/presets/active", {
      method: "PUT",
      body: JSON.stringify({ speaker_key }),
    }),
  cloneVoice: (name: string, file: Blob, filename = "sample.wav") => {
    const fd = new FormData();
    fd.append("name", name);
    fd.append("file", file, filename);
    return request<{ voice_id: string; name: string }>("/api/voice/clone", {
      method: "POST",
      body: fd,
    });
  },

  // Emotion
  getEmotion: () => request<EmotionResponse>("/api/emotion"),
  setEmotion: (preset: EmotionPreset, intensity: number) =>
    request<EmotionResponse>("/api/emotion", {
      method: "PUT",
      body: JSON.stringify({ preset, intensity }),
    }),

  // Training
  startCapture: (label: string) =>
    request<{ active: boolean; label: string }>("/api/training/start", {
      method: "POST",
      body: JSON.stringify({ label }),
    }),
  stopCapture: () =>
    request<{ active: boolean; label: string | null }>("/api/training/stop", {
      method: "POST",
    }),
  listSamples: () => request<SamplesResponse>("/api/training/samples"),
  trainModel: (epochs = 25) =>
    request<{ status: string; epochs: number }>("/api/training/train", {
      method: "POST",
      body: JSON.stringify({ epochs }),
    }),
  loadDefaultPack: () =>
    request<{ status: string; message: string; suggested_labels?: string[] }>(
      "/api/training/load-default-pack",
      { method: "POST" },
    ),
};

export { API_BASE, HttpError };
