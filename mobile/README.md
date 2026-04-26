# Mobile Companion App (Stub)

This folder is reserved for a future React Native (Expo) build of the
companion app. The web dashboard in `/client` is the primary configuration
surface today; the mobile app will mirror its three feature areas:

1. **Voice** — record / upload a 6-second sample and clone via ElevenLabs.
2. **Emotion** — preset + intensity slider for paralinguistic tone.
3. **Training Studio** — live landmark preview from the Edge server, capture
   custom signs, and trigger LSTM fine-tuning.

## Suggested setup (when implementation begins)

```bash
npx create-expo-app@latest .
npx expo install nativewind tailwindcss
npx expo install react-native-reanimated
```

Recommended stack:

- **Expo SDK 51+** with the new architecture enabled
- **NativeWind 4** for Tailwind-class styling
- **Expo AV** (`expo-av`) for microphone recording
- **Reusable API client**: copy `client/lib/api.ts` and adapt for `fetch` /
  `XMLHttpRequest` so the typed contract stays aligned with the web app.
- **Reusable WS client**: copy `client/lib/ws.ts`; React Native's WebSocket
  is API-compatible.

The Edge server URL should be configurable via Expo `extra` config so
end-users can pair the phone to a specific local server IP.
