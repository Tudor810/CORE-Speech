#include <Arduino.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

#include <driver/i2s.h>

// ------------------------
// Pin configuration
// ------------------------
#define I2C_SDA 33
#define I2C_SCL 35
#define BUTTON_PIN 4
#define SPEAKER_PIN 17
#define I2S_SCK 10
#define I2S_WS 11
#define I2S_SD 12

#define I2S_PORT I2S_NUM_0

// ------------------------
// User configuration
// ------------------------
static const char *WIFI_SSID = "HONOR Magic5 Lite 5G";
static const char *WIFI_PASS = "pupic1234";

// Server is FastAPI in this repo (see README WebSocket endpoints).
// Use ws:// (not wss://) unless you add TLS support.
static const char *WS_HOST = "10.103.228.176";
static const uint16_t WS_PORT = 8000;
static const char *DEVICE_ID = "glasses-01";

// Endpoints:
// - Mode 1 (mic -> server -> text): /ws/emission/{device_id}
// - Mode 2 (server audio -> speaker): /ws/reception/{device_id}

// ------------------------
// LCD (keep your init sequence)
// ------------------------
LiquidCrystal_I2C lcd(0x27, 16, 2);

void forceLCDInit() {
  for (int i = 0; i < 3; i++) {
    Wire.beginTransmission(0x27);
    Wire.write(0x30 | 0x08 | 0x04);
    Wire.endTransmission();
    delay(5);
    Wire.beginTransmission(0x27);
    Wire.write(0x30 | 0x08);
    Wire.endTransmission();
    delay(5);
  }
}

static void lcdPrint2(const String &l1, const String &l2) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(l1.substring(0, 16));
  lcd.setCursor(0, 1);
  lcd.print(l2.substring(0, 16));
}

// ------------------------
// Mode state machine
// ------------------------
enum class DeviceMode : uint8_t {
  Mode1_MicToServer_Text = 0,
  Mode2_ServerAudioToSpeaker = 1,
};

static DeviceMode g_mode = DeviceMode::Mode1_MicToServer_Text;

// ------------------------
// Debounced button (active-low)
// ------------------------
static bool g_btnStable = HIGH;
static bool g_btnLastRead = HIGH;
static uint32_t g_btnLastChangeMs = 0;
static const uint32_t BTN_DEBOUNCE_MS = 35;
static uint32_t g_ignoreButtonUntilMs = 0;

static bool buttonPressedEdge() {
  bool r = digitalRead(BUTTON_PIN);
  uint32_t now = millis();
  if (r != g_btnLastRead) {
    g_btnLastRead = r;
    g_btnLastChangeMs = now;
  }
  if ((now - g_btnLastChangeMs) >= BTN_DEBOUNCE_MS && r != g_btnStable) {
    bool prev = g_btnStable;
    g_btnStable = r;
    if (prev == HIGH && g_btnStable == LOW) {
      return true; // falling edge after debounce
    }
  }
  return false;
}

// ------------------------
// I2S microphone capture
// INMP441 outputs 24-bit in 32-bit container. We downshift to int16.
// ------------------------
static const uint32_t MIC_SAMPLE_RATE = 16000;
static const size_t MIC_WS_CHUNK_BYTES = 1024; // send in 1 KB frames (binary)
static uint8_t g_micChunk[MIC_WS_CHUNK_BYTES];
static size_t g_micChunkFill = 0;

static void setupI2S() {
  i2s_config_t i2s_config = {};
  i2s_config.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  i2s_config.sample_rate = MIC_SAMPLE_RATE;
  i2s_config.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
  i2s_config.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  i2s_config.communication_format = I2S_COMM_FORMAT_I2S;
  i2s_config.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  i2s_config.dma_buf_count = 8;
  i2s_config.dma_buf_len = 128;
  i2s_config.use_apll = false;
  i2s_config.tx_desc_auto_clear = false;
  i2s_config.fixed_mclk = 0;

  i2s_pin_config_t pin_config = {};
  pin_config.bck_io_num = I2S_SCK;
  pin_config.ws_io_num = I2S_WS;
  pin_config.data_out_num = -1;
  pin_config.data_in_num = I2S_SD;

  i2s_driver_install(I2S_PORT, &i2s_config, 0, nullptr);
  i2s_set_pin(I2S_PORT, &pin_config);
  i2s_start(I2S_PORT);
}

static inline int16_t inmp441_to_s16(int32_t raw32) {
  // Typical INMP441: valid MSBs are top 24 bits; shift down to 16-bit audio.
  // This shift keeps sign and gives reasonable amplitude.
  return (int16_t)(raw32 >> 14);
}

// ------------------------
// DAC playback (timer paced @ 16 kHz)
// Server sends int16 PCM @ 16 kHz. We convert to unsigned 8-bit for DAC.
// ------------------------
static const uint32_t PLAY_SAMPLE_RATE = 16000;
static const size_t PLAY_RING_BYTES = 32 * 1024; // enough for bursty WS frames
static volatile uint8_t g_playRing[PLAY_RING_BYTES];
static volatile uint32_t g_playW = 0;
static volatile uint32_t g_playR = 0;
static hw_timer_t *g_dacTimer = nullptr;
static portMUX_TYPE g_playMux = portMUX_INITIALIZER_UNLOCKED;

static inline uint32_t ringCount() {
  uint32_t w = g_playW, r = g_playR;
  return (w >= r) ? (w - r) : (PLAY_RING_BYTES - (r - w));
}

static void IRAM_ATTR onDacTimer() {
  uint8_t out = 128; // mid-rail silence
  portENTER_CRITICAL_ISR(&g_playMux);
  if (g_playR != g_playW) {
    out = g_playRing[g_playR];
    g_playR = (g_playR + 1) % PLAY_RING_BYTES;
  }
  portEXIT_CRITICAL_ISR(&g_playMux);
  dacWrite(SPEAKER_PIN, out);
}

static void startDacTimer() {
  if (g_dacTimer) return;
  g_dacTimer = timerBegin(0, 80, true); // 80 MHz / 80 = 1 MHz tick
  timerAttachInterrupt(g_dacTimer, &onDacTimer, true);
  timerAlarmWrite(g_dacTimer, 1000000UL / PLAY_SAMPLE_RATE, true);
  timerAlarmEnable(g_dacTimer);
}

static void stopDacTimer() {
  if (!g_dacTimer) return;
  timerAlarmDisable(g_dacTimer);
  timerDetachInterrupt(g_dacTimer);
  timerEnd(g_dacTimer);
  g_dacTimer = nullptr;
  dacWrite(SPEAKER_PIN, 128);
  portENTER_CRITICAL(&g_playMux);
  g_playR = g_playW = 0;
  portEXIT_CRITICAL(&g_playMux);
}

static inline uint8_t s16_to_u8(int16_t s) {
  // signed 16-bit -> unsigned 8-bit (with simple scaling)
  int32_t v = ((int32_t)s + 32768) >> 8; // 0..255
  if (v < 0) v = 0;
  if (v > 255) v = 255;
  return (uint8_t)v;
}

static void playRingPushU8(const uint8_t *data, size_t len) {
  portENTER_CRITICAL(&g_playMux);
  for (size_t i = 0; i < len; i++) {
    uint32_t nextW = (g_playW + 1) % PLAY_RING_BYTES;
    if (nextW == g_playR) {
      // buffer full: drop oldest byte to keep realtime
      g_playR = (g_playR + 1) % PLAY_RING_BYTES;
    }
    g_playRing[g_playW] = data[i];
    g_playW = nextW;
  }
  portEXIT_CRITICAL(&g_playMux);
}

// ------------------------
// WebSocket client
// ------------------------
static WebSocketsClient ws;
static bool g_wsConnected = false;
static String g_wsPath;

static String wsPathForMode(DeviceMode m) {
  String base = (m == DeviceMode::Mode1_MicToServer_Text) ? "/ws/emission/" : "/ws/reception/";
  return base + DEVICE_ID;
}

static void wsConnectForMode(DeviceMode m) {
  g_wsConnected = false;
  ws.disconnect();
  delay(20);

  g_wsPath = wsPathForMode(m);
  ws.begin(WS_HOST, WS_PORT, g_wsPath.c_str());
  ws.setReconnectInterval(1500);
  ws.enableHeartbeat(15000, 3000, 2);
}

static void onWsEvent(WStype_t type, uint8_t *payload, size_t length) {
  switch (type) {
    case WStype_CONNECTED: {
      g_wsConnected = true;
      Serial.printf("[WS] connected: ws://%s:%u%s\n", WS_HOST, (unsigned)WS_PORT, g_wsPath.c_str());
      if (g_mode == DeviceMode::Mode1_MicToServer_Text) {
        lcdPrint2("Mode 1: MIC->WS", "WS: connected");
      } else {
        lcdPrint2("Mode 2: WS->SPK", "WS: connected");
      }
      break;
    }
    case WStype_DISCONNECTED: {
      g_wsConnected = false;
      Serial.println("[WS] disconnected (reconnecting...)");
      if (g_mode == DeviceMode::Mode1_MicToServer_Text) {
        lcdPrint2("Mode 1: MIC->WS", "WS: reconnect");
      } else {
        lcdPrint2("Mode 2: WS->SPK", "WS: reconnect");
      }
      break;
    }
    case WStype_TEXT: {
      // Server sends JSON (e.g. {type:"transcript", text, final}) on /ws/emission
      // and also status JSON on /ws/reception. For LCD: show a trimmed preview.
      String msg;
      msg.reserve(length + 1);
      for (size_t i = 0; i < length; i++) msg += (char)payload[i];

      // Basic LCD friendly rendering: show first 16 chars on line2, mode on line1.
      if (g_mode == DeviceMode::Mode1_MicToServer_Text) {
        Serial.printf("[WS] text (%uB) on /ws/emission: %s\n", (unsigned)length, msg.c_str());
        lcdPrint2("Transcript:", msg);
      } else {
        Serial.printf("[WS] text (%uB) on /ws/reception: %s\n", (unsigned)length, msg.c_str());
        lcdPrint2("Status:", msg);
      }
      break;
    }
    case WStype_BIN: {
      // Binary audio frames are expected on /ws/reception (Mode 2).
      // Format expected: little-endian int16 PCM @ 16kHz (server README says that).
      if (g_mode != DeviceMode::Mode2_ServerAudioToSpeaker) break;
      if (length < 2) break;

      // Convert int16 LE -> u8 DAC stream.
      Serial.printf("[WS] audio frame: %u bytes\n", (unsigned)length);
      static uint8_t tmp[2048];
      size_t samples = length / 2;
      size_t offset = 0;
      while (samples) {
        const size_t batchSamples = (samples > sizeof(tmp)) ? sizeof(tmp) : samples; // 1 byte per output sample
        for (size_t i = 0; i < batchSamples; i++) {
          const size_t idx = offset + (2 * i);
          int16_t s = (int16_t)((uint16_t)payload[idx] | ((uint16_t)payload[idx + 1] << 8));
          tmp[i] = s16_to_u8(s);
        }
        playRingPushU8(tmp, batchSamples);
        offset += batchSamples * 2;
        samples -= batchSamples;
      }
      break;
    }
    default:
      break;
  }
}

// ------------------------
// WiFi
// ------------------------
static void wifiConnect() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.printf("[WiFi] connecting to \"%s\"...\n", WIFI_SSID);
  lcdPrint2("WiFi connecting", WIFI_SSID);
  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    if ((millis() - start) > 20000) break;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("[WiFi] connected, IP=%s, GW=%s\n",
                  WiFi.localIP().toString().c_str(),
                  WiFi.gatewayIP().toString().c_str());
    lcdPrint2("WiFi connected", WiFi.localIP().toString());
  } else {
    Serial.println("[WiFi] FAILED (timeout) — check SSID/PW");
    lcdPrint2("WiFi failed", "check SSID/PW");
  }
}

// ------------------------
// Mode switching
// ------------------------
static void applyMode(DeviceMode newMode) {
  if (newMode == g_mode) return;
  g_mode = newMode;

  Serial.printf("[MODE] switched to %s\n",
                (g_mode == DeviceMode::Mode1_MicToServer_Text) ? "Mode 1 (MIC->WS->TEXT)" : "Mode 2 (WS AUDIO->SPK)");

  // Stop / start peripherals for each mode.
  if (g_mode == DeviceMode::Mode1_MicToServer_Text) {
    stopDacTimer();
    g_micChunkFill = 0;
    lcdPrint2("Mode 1: MIC->WS", "connecting...");
  } else {
    startDacTimer();
    lcdPrint2("Mode 2: WS->SPK", "connecting...");
  }

  wsConnectForMode(g_mode);
}

// ------------------------
// Mode 1 loop: capture mic, buffer, send to WS
// Sends little-endian int16 PCM at 16kHz (matches server README).
// ------------------------
static void mode1_captureAndSend() {
  if (!g_wsConnected) return;

  int32_t raw = 0;
  size_t bytes_read = 0;

  // Read a single 32-bit sample. Small timeout to keep ws.loop responsive.
  esp_err_t ok = i2s_read(I2S_PORT, &raw, sizeof(raw), &bytes_read, 5 / portTICK_PERIOD_MS);
  if (ok != ESP_OK || bytes_read != sizeof(raw)) return;

  int16_t s16 = inmp441_to_s16(raw);
  g_micChunk[g_micChunkFill++] = (uint8_t)(s16 & 0xFF);
  g_micChunk[g_micChunkFill++] = (uint8_t)((s16 >> 8) & 0xFF);

  if (g_micChunkFill >= MIC_WS_CHUNK_BYTES) {
    ws.sendBIN(g_micChunk, MIC_WS_CHUNK_BYTES);
    g_micChunkFill = 0;
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n[BOOT] ESP32-S2 starting");

  pinMode(BUTTON_PIN, INPUT_PULLUP);
  delay(20);
  g_btnStable = g_btnLastRead = digitalRead(BUTTON_PIN);
  g_btnLastChangeMs = millis();
  g_ignoreButtonUntilMs = millis() + 400; // ignore boot-time bounce/glitches

  Wire.begin(I2C_SDA, I2C_SCL);
  delay(200);
  forceLCDInit();
  lcd.init();
  delay(100);
  lcd.backlight();
  lcd.clear();
  lcdPrint2("Booting...", "ESP32-S2");

  setupI2S();
  startDacTimer();  // we’ll stop it if Mode 1
  stopDacTimer();

  wifiConnect();

  ws.onEvent(onWsEvent);
  wsConnectForMode(g_mode);

  lcdPrint2("Mode 1: MIC->WS", "connecting...");
}

void loop() {
  // Button toggles between the two modes.
  if (millis() > g_ignoreButtonUntilMs && buttonPressedEdge()) {
    applyMode((g_mode == DeviceMode::Mode1_MicToServer_Text)
                ? DeviceMode::Mode2_ServerAudioToSpeaker
                : DeviceMode::Mode1_MicToServer_Text);
  }

  // Maintain websocket connection.
  ws.loop();

  // Mode-specific work
  if (g_mode == DeviceMode::Mode1_MicToServer_Text) {
    mode1_captureAndSend();
  } else {
    // Playback is timer-driven; loop just keeps WS pumping.
    // Optional: show buffer fill occasionally without spamming LCD.
  }
}

