// Reconnecting WebSocket helper used by the Training Studio.

const WS_BASE =
  process.env.NEXT_PUBLIC_WS_BASE?.replace(/\/$/, "") ||
  (typeof window !== "undefined"
    ? `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.hostname}:8000`
    : "ws://localhost:8000");

export type WsHandlers = {
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (err: Event) => void;
  onMessage: (data: unknown) => void;
};

export interface WsOptions {
  /** Default "blob"; pass "arraybuffer" for binary streaming consumers. */
  binaryType?: BinaryType;
}

export class ReconnectingWS {
  private ws: WebSocket | null = null;
  private url: string;
  private handlers: WsHandlers;
  private opts: WsOptions;
  private retry = 0;
  private closedByUser = false;
  private reconnectTimer: number | null = null;

  constructor(path: string, handlers: WsHandlers, opts: WsOptions = {}) {
    this.url = `${WS_BASE}${path}`;
    this.handlers = handlers;
    this.opts = opts;
  }

  connect() {
    this.closedByUser = false;
    this.open();
  }

  private open() {
    try {
      this.ws = new WebSocket(this.url);
    } catch (err) {
      this.scheduleReconnect();
      return;
    }
    if (this.opts.binaryType) {
      this.ws.binaryType = this.opts.binaryType;
    }
    this.ws.onopen = () => {
      this.retry = 0;
      this.handlers.onOpen?.();
    };
    this.ws.onmessage = (e) => {
      let payload: unknown = e.data;
      if (typeof e.data === "string") {
        try {
          payload = JSON.parse(e.data);
        } catch {
          /* keep raw */
        }
      }
      this.handlers.onMessage(payload);
    };
    this.ws.onerror = (e) => this.handlers.onError?.(e);
    this.ws.onclose = () => {
      this.handlers.onClose?.();
      this.ws = null;
      if (!this.closedByUser) this.scheduleReconnect();
    };
  }

  private scheduleReconnect() {
    if (this.reconnectTimer != null) return;
    const delay = Math.min(8000, 500 * 2 ** this.retry);
    this.retry += 1;
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      this.open();
    }, delay);
  }

  send(data: unknown) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
    this.ws.send(typeof data === "string" ? data : JSON.stringify(data));
    return true;
  }

  sendBinary(buf: ArrayBuffer | ArrayBufferView | Blob) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
    // Avoid unbounded buffering on slow networks (e.g. tab in background).
    if (this.ws.bufferedAmount > 4 * 1024 * 1024) return false;
    this.ws.send(buf as ArrayBuffer);
    return true;
  }

  close() {
    this.closedByUser = true;
    if (this.reconnectTimer != null) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
    this.ws = null;
  }
}

export { WS_BASE };
