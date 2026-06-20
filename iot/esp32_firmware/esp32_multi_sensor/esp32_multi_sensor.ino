#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <time.h>

// ---------- CONFIG ----------
const char* WIFI_SSID     = "Ishani";
const char* WIFI_PASSWORD = "07186556";
const char* LOCAL_URL     = "http://192.168.29.219:8000";
const char* CLOUD_URL     = "https://parkiq-glrk.onrender.com";
const char* DEVICE_ID     = "ESP32-NODE-01";

// ---------- SENSOR COUNT ----------
#define NUM_SENSORS 1

// ---------- PIN MAPPING ----------
const int TRIG_PINS[NUM_SENSORS] = {12};
const int ECHO_PINS[NUM_SENSORS] = {13};

// Traffic Light LEDs
const int LED_RED    = 25;
const int LED_GREEN  = 26;
const int LED_YELLOW = 27;

// Shared buzzer
const int BUZZER_PIN = 14;

// Zone names reported to server
const char* ZONE_IDS[NUM_SENSORS] = {"Zone-1"};

// ---------- THRESHOLDS ----------
const float DIST_DETECT_CM  = 80.0;  // max range to consider object present
const float DIST_FAR_CM     = 60.0;  // far   -> Green  (just detected)
const float DIST_MED_CM     = 35.0;  // medium -> Yellow (getting closer)
const float DIST_CLOSE_CM   = 15.0;  // close  -> Red + Buzzer (very close)
const int   DEBOUNCE_READINGS          = 5;
const unsigned long VIOLATION_THRESHOLD_MS = 10000;

// ---------- PER-ZONE STATE ----------
enum ZoneState { VACANT, OCCUPIED, VIOLATION };
ZoneState zoneState[NUM_SENSORS];
unsigned long occupiedSince[NUM_SENSORS];
int belowCount[NUM_SENSORS], aboveCount[NUM_SENSORS];
unsigned long lastHeartbeat[NUM_SENSORS];
float lastDistance[NUM_SENSORS];

void setLed(bool r, bool g, bool y) {
  digitalWrite(LED_RED, r); 
  digitalWrite(LED_GREEN, g); 
  digitalWrite(LED_YELLOW, y);
}

// ---------- DISTANCE-BASED LED (IoT mode) ----------
// OFF       = no object detected
// Green     = object far (> DIST_FAR_CM)
// Yellow    = object medium distance (DIST_MED..DIST_FAR)
// Red+Buzz  = object very close (< DIST_CLOSE_CM)
void updateIoTLed(float distance) {
  if (distance <= 0 || distance > DIST_DETECT_CM) {
    setLed(0, 0, 0);            // No object - all OFF
    digitalWrite(BUZZER_PIN, LOW);
  } else if (distance > DIST_FAR_CM) {
    setLed(0, 1, 0);            // Far - Green
    digitalWrite(BUZZER_PIN, LOW);
  } else if (distance > DIST_CLOSE_CM) {
    setLed(0, 0, 1);            // Medium - Yellow
    digitalWrite(BUZZER_PIN, LOW);
  } else {
    setLed(1, 0, 0);            // Very close - Red + Buzzer!
    digitalWrite(BUZZER_PIN, HIGH);
  }
}

// ---------- SENSOR READ (sequential, never simultaneous) ----------
float readDistanceCm(int idx) {
  int trig = TRIG_PINS[idx];
  int echo = ECHO_PINS[idx];

  digitalWrite(trig, LOW);
  delayMicroseconds(2);
  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW);

  long duration = pulseIn(echo, HIGH, 30000); // 30ms timeout (~5m max range)
  if (duration == 0) return -1.0;
  return duration * 0.0343 / 2.0;
}

void sendEventToUrl(String url, String payload, int idx) {
  HTTPClient http;
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(payload);
  if(code > 0) {
    Serial.printf("[Zone %s] HTTP %d (URL: %s)\n", ZONE_IDS[idx], code, url.c_str());
  } else {
    Serial.printf("[Zone %s] HTTP Error %d (URL: %s)\n", ZONE_IDS[idx], code, url.c_str());
  }
  http.end();
}

void sendEvent(int idx, const char* eventType, unsigned long durationSec, float distanceCm) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.printf("[Zone %s] WiFi not connected. Skipping send.\n", ZONE_IDS[idx]);
    return;
  }

  StaticJsonDocument<256> doc;
  doc["device_id"]    = DEVICE_ID;
  doc["zone_id"]      = ZONE_IDS[idx];
  doc["event"]        = eventType;
  doc["duration_sec"] = durationSec;
  doc["distance_cm"]  = distanceCm;
  doc["timestamp"]    = (long)time(nullptr);

  String payload;
  serializeJson(doc, payload);
  Serial.printf("[Zone %s] %s -> %s\n", ZONE_IDS[idx], eventType, payload.c_str());

  // Send to BOTH local and cloud servers
  sendEventToUrl(String(LOCAL_URL) + "/api/sensor-event", payload, idx);
  sendEventToUrl(String(CLOUD_URL) + "/api/sensor-event", payload, idx);
}

// ---------- UPDATE ONE ZONE ----------
void updateZone(int idx) {
  float distance  = readDistanceCm(idx);
  lastDistance[idx] = distance;
  bool occupiedNow = (distance > 0 && distance < DIST_DETECT_CM);

  if (occupiedNow) { belowCount[idx]++; aboveCount[idx] = 0; }
  else             { aboveCount[idx]++; belowCount[idx] = 0; }

  switch (zoneState[idx]) {

    case VACANT:
      updateIoTLed(distance);   // OFF if no object, green if far
      if (belowCount[idx] >= DEBOUNCE_READINGS) {
        zoneState[idx]    = OCCUPIED;
        occupiedSince[idx] = millis();
        Serial.printf("[Zone %s] -> OCCUPIED (%.1f cm)\n", ZONE_IDS[idx], distance);
        sendEvent(idx, "OCCUPANCY_START", 0, distance);
      }
      break;

    case OCCUPIED: {
      updateIoTLed(distance);   // Green/Yellow/Red based on proximity
      unsigned long elapsed = millis() - occupiedSince[idx];

      if (aboveCount[idx] >= DEBOUNCE_READINGS) {
        zoneState[idx] = VACANT;
        setLed(0, 0, 0);        // Object left — all OFF
        digitalWrite(BUZZER_PIN, LOW);
        Serial.printf("[Zone %s] -> VACANT\n", ZONE_IDS[idx]);
        sendEvent(idx, "VACATED", elapsed / 1000, distance);
      } else if (elapsed > VIOLATION_THRESHOLD_MS) {
        zoneState[idx] = VIOLATION;
        Serial.printf("[Zone %s] -> VIOLATION\n", ZONE_IDS[idx]);
        sendEvent(idx, "VIOLATION_CONFIRMED", elapsed / 1000, distance);
      }
      break;
    }

    case VIOLATION: {
      updateIoTLed(distance);   // Still distance-based even in violation
      unsigned long elapsed = millis() - occupiedSince[idx];

      if (aboveCount[idx] >= DEBOUNCE_READINGS) {
        zoneState[idx] = VACANT;
        setLed(0, 0, 0);        // Cleared — all OFF
        digitalWrite(BUZZER_PIN, LOW);
        Serial.printf("[Zone %s] -> VACANT (cleared)\n", ZONE_IDS[idx]);
        sendEvent(idx, "VACATED", elapsed / 1000, distance);
      } else if (millis() - lastHeartbeat[idx] > 30000) {
        lastHeartbeat[idx] = millis();
        sendEvent(idx, "VIOLATION_ONGOING", elapsed / 1000, distance);
      }
      break;
    }
  }

  delay(60); // 60ms gap between sensors to prevent cross-talk
}

// ---------- SETUP ----------
void setup() {
  Serial.begin(115200);

  for (int i = 0; i < NUM_SENSORS; i++) {
    pinMode(TRIG_PINS[i], OUTPUT);
    pinMode(ECHO_PINS[i], INPUT);
    zoneState[i]      = VACANT;
    occupiedSince[i]  = 0;
    belowCount[i]     = 0;
    aboveCount[i]     = 0;
    lastHeartbeat[i]  = 0;
    lastDistance[i]   = -1;
  }
  pinMode(LED_RED, OUTPUT);
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.printf("\nConnected! IP: %s\n", WiFi.localIP().toString().c_str());
  configTime(5 * 3600 + 1800, 0, "pool.ntp.org"); // IST
  
  setLed(0, 0, 0); // Start with all LEDs OFF
}

// ---------- DEVICE STATE POLLING ----------
String activeMode = "CCTV";           // Default to CCTV — updated by fetchDeviceState()
String cctvAlertLevel = "VACANT";  // VACANT / MEDIUM / HIGH / CRITICAL

void fetchDeviceState() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    // Try Cloud first
    http.begin(String(CLOUD_URL) + "/api/device-state");
    int code = http.GET();
    if (code <= 0) {
      http.end();
      // Fallback to local
      http.begin(String(LOCAL_URL) + "/api/device-state");
      code = http.GET();
    }

    if (code > 0) {
      String payload = http.getString();
      StaticJsonDocument<256> doc;
      deserializeJson(doc, payload);
      activeMode     = doc["mode"].as<String>();
      cctvAlertLevel = doc["alert_level"].as<String>();
    }
    http.end();
  }
}

// ---------- MAIN LOOP ----------
void loop() {
  // Poll server state every 2 seconds
  static unsigned long lastStateFetch = 0;
  if (millis() - lastStateFetch > 2000) {
    fetchDeviceState();
    lastStateFetch = millis();
  }

  if (activeMode == "CCTV") {
    // 🎥 CCTV MODE: 4-level alert driven by AI CIS score
    if (cctvAlertLevel == "CRITICAL") {
      // 🚨 Blinking Red + Buzzer — Immediate tow required!
      bool ledOn = (millis() / 300) % 2 == 0;  // toggle every 300ms
      setLed(ledOn, 0, 0);
      digitalWrite(BUZZER_PIN, HIGH);
    } else if (cctvAlertLevel == "HIGH") {
      // 🔴 Solid Red, No Buzzer — High severity violation
      setLed(1, 0, 0);
      digitalWrite(BUZZER_PIN, LOW);
    } else if (cctvAlertLevel == "MEDIUM") {
      // 🟡 Solid Yellow — Violation detected, monitoring
      setLed(0, 0, 1);
      digitalWrite(BUZZER_PIN, LOW);
    } else if (cctvAlertLevel == "LOW") {
      // 🟢 Solid Green — Minor violation, low risk
      setLed(0, 1, 0);
      digitalWrite(BUZZER_PIN, LOW);
    } else {
      // ⚫ VACANT — No violation, all LEDs OFF
      setLed(0, 0, 0);
      digitalWrite(BUZZER_PIN, LOW);
    }
  } else {
    // 📡 IOT MODE: Hardware acts autonomously using HC-SR04 sensors
    for (int i = 0; i < NUM_SENSORS; i++) {
      updateZone(i);
    }

    // Telemetry heartbeat for all zones every 5 seconds
    static unsigned long lastTelemetry = 0;
    if (millis() - lastTelemetry > 5000) {
      for (int i = 0; i < NUM_SENSORS; i++) {
        sendEvent(i, "TELEMETRY_UPDATE", 0, lastDistance[i]);
      }
      lastTelemetry = millis();
    }
  }

  delay(200);
}