// Gapless playback for streamed PCM audio.
//
// The reception WebSocket emits 16-bit signed little-endian mono PCM at the
// server's `openvoice_output_sample_rate` (default 16 kHz). We schedule each
// chunk on a shared AudioContext clock so back-to-back frames play without
// underruns or audible seams.

export interface PcmPlayerOptions {
  sampleRate?: number;
  /** Max queued audio (in seconds) before older chunks are dropped. */
  maxBufferSeconds?: number;
}

type AudioCtxCtor = typeof AudioContext;

function getAudioContextCtor(): AudioCtxCtor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    AudioContext?: AudioCtxCtor;
    webkitAudioContext?: AudioCtxCtor;
  };
  return w.AudioContext ?? w.webkitAudioContext ?? null;
}

export class PcmPlayer {
  private ctx: AudioContext | null = null;
  private gain: GainNode | null = null;
  private nextStartTime = 0;
  private muted = false;
  private readonly sampleRate: number;
  private readonly maxBufferSeconds: number;
  private active = new Set<AudioBufferSourceNode>();
  private startedAt = 0;

  constructor(opts: PcmPlayerOptions = {}) {
    this.sampleRate = opts.sampleRate ?? 16000;
    this.maxBufferSeconds = opts.maxBufferSeconds ?? 4;
  }

  /** Lazy-create the AudioContext on first user interaction. */
  async ensureRunning(): Promise<void> {
    if (!this.ctx) {
      const Ctor = getAudioContextCtor();
      if (!Ctor) throw new Error("Audio playback is not supported in this browser");
      // We pass the desired sample rate so no resampling is needed for our
      // 16 kHz PCM. Some browsers ignore this and use the device rate; that's
      // fine, the buffer's own sample rate metadata will trigger resampling.
      try {
        this.ctx = new Ctor({ sampleRate: this.sampleRate });
      } catch {
        this.ctx = new Ctor();
      }
      this.gain = this.ctx.createGain();
      this.gain.gain.value = this.muted ? 0 : 1;
      this.gain.connect(this.ctx.destination);
    }
    if (this.ctx.state === "suspended") {
      await this.ctx.resume();
    }
  }

  /** Push a binary PCM chunk into the playback queue. */
  enqueue(chunk: ArrayBuffer | Uint8Array): void {
    if (!this.ctx || !this.gain) return;
    const ctx = this.ctx;
    const buf = chunk instanceof ArrayBuffer ? chunk : chunk.buffer.slice(
      chunk.byteOffset,
      chunk.byteOffset + chunk.byteLength,
    );

    // Convert int16 LE -> float32 [-1, 1]
    const view = new DataView(buf);
    const sampleCount = Math.floor(view.byteLength / 2);
    if (sampleCount === 0) return;

    const audioBuffer = ctx.createBuffer(1, sampleCount, this.sampleRate);
    const channel = audioBuffer.getChannelData(0);
    for (let i = 0; i < sampleCount; i++) {
      const s = view.getInt16(i * 2, true);
      channel[i] = s < 0 ? s / 0x8000 : s / 0x7fff;
    }

    // Reset the clock if we've been idle.
    const now = ctx.currentTime;
    if (this.nextStartTime < now) this.nextStartTime = now + 0.02;

    // Drop the queue head if we've buffered way too much (e.g. tab in BG).
    if (this.nextStartTime - now > this.maxBufferSeconds) {
      this.nextStartTime = now + 0.02;
    }

    if (this.active.size === 0) this.startedAt = this.nextStartTime;

    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.gain);
    source.onended = () => {
      this.active.delete(source);
    };
    source.start(this.nextStartTime);
    this.active.add(source);
    this.nextStartTime += audioBuffer.duration;
  }

  /** Approximate seconds of audio still queued ahead of the playhead. */
  get queuedSeconds(): number {
    if (!this.ctx) return 0;
    return Math.max(0, this.nextStartTime - this.ctx.currentTime);
  }

  /** Stop all currently scheduled audio immediately. */
  stop(): void {
    for (const src of this.active) {
      try {
        src.stop();
      } catch {
        /* already ended */
      }
    }
    this.active.clear();
    this.nextStartTime = this.ctx?.currentTime ?? 0;
  }

  setMuted(muted: boolean): void {
    this.muted = muted;
    if (this.gain) this.gain.gain.value = muted ? 0 : 1;
  }

  async close(): Promise<void> {
    this.stop();
    if (this.ctx) {
      try {
        await this.ctx.close();
      } catch {
        /* ignore */
      }
      this.ctx = null;
      this.gain = null;
    }
  }
}
