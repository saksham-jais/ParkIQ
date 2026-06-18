# ParkIQ — AI-Powered Parking Impact Intelligence for Bengaluru
### Gridlock Hackathon 2.0 — Round 2 Prototype Documentation

**Problem Statement:** Poor Visibility on Parking-Induced Congestion
**Tagline:** From Parking Violations to Congestion Intelligence

---

## Table of Contents

1. Executive Summary
2. Problem Statement & Why Current Approaches Fail
3. Core Innovation — The Congestion Impact Score (CIS)
4. Dataset: Fields, Cleaning, Feature Engineering
5. System Architecture (End-to-End Data Flow)
6. Module 1 — Hotspot Detection & Clustering
7. Module 2 — Congestion Impact Score Engine
8. Module 3 — Hotspot Prediction (Forecasting)
9. Module 4 — Enforcement Recommendation Engine
10. Module 5 — IoT Smart Parking Occupancy Node (ESP32)
11. Module 6 — CCTV + YOLOv8 Computer Vision Pipeline *(implemented)*
12. Backend API Design
13. Database Schema
14. Dashboard (Frontend) — All Pages Implemented
15. High-Risk Area Analysis & AI Suggestions *(new)*
16. Stretch Goals: Digital Twin & Emergency Lane Protection
17. Tech Stack — Hackathon MVP vs. Production Vision
18. 4-Week Build Roadmap
19. Demo Script for Judges
20. Evaluation Metrics
21. Risks & Honest Limitations
22. Submission Checklist
23. Future Scope

---

## 1. Executive Summary

ParkIQ turns raw illegal-parking violation records into **prioritized, actionable enforcement intelligence** for Bengaluru Traffic Police. Instead of just flagging "this vehicle is illegally parked," ParkIQ answers the question that actually matters operationally: **"How much congestion is this specific violation causing, and which violations should officers clear first?"**

The system has three layers that work together:

- A **data intelligence layer** that scores every historical and incoming violation with a Congestion Impact Score (CIS) and clusters them into hotspots.
- A **physical IoT layer** (built on the ESP32 + HC-SR04 kit you already own) that demonstrates real-time occupancy detection at a single parking zone, proving the concept can scale to a city-wide sensor network.
- A **decision layer** (dashboard + recommendation engine) that turns scores and hotspots into a ranked action list for police.

This document is the single source of truth for what to build, what each piece does, and how the pieces connect — written so it can be followed task-by-task during the 4-week build window.

---

## 2. Problem Statement & Why Current Approaches Fail

On-street illegal and spillover parking near commercial areas, metro stations, and events chokes carriageways and intersections. Today:

- Enforcement is patrol-based and reactive — officers find violations by chance, not by data.
- There is no heatmap correlating *where* violations happen with *how much* traffic impact they actually cause.
- All violations are treated as equally urgent, which they are not. A scooter on a side lane and a tanker blocking a junction approach are not the same problem, but most systems log them identically.
- Existing CV-based "violation detector" solutions (the kind most other teams will submit) stop at *detection*. They don't help police decide what to do next, or in what order.

**The gap ParkIQ fills:** turning a flat list of violations into a ranked, explainable, geographically-aware priority queue.

---

## 3. Core Innovation — The Congestion Impact Score (CIS)

CIS is a 0–100 score assigned to every violation (historical or live), answering: *how disruptive is this specific parked vehicle, at this specific location, at this specific time?*

### Why a transparent formula first, not a black-box model first

Be upfront about this with judges: the provided dataset has **no ground-truth congestion measurement** (no actual road speed, queue length, or delay data tied to each violation). Training a supervised model (XGBoost/LightGBM) to predict "congestion impact" without a real label is the trap most teams will fall into quietly — they'll train a model against a self-invented label and call it AI, but it's just curve-fitting to their own assumptions.

ParkIQ's honest, defensible approach:

1. **Start with an explainable, hand-tunable heuristic formula** (below) — this is the actual deliverable for the hackathon. It's auditable, which is exactly what a police department evaluating a new system wants.
2. **Position ML (XGBoost/LightGBM) as the v2 enhancement** once real correlating signal is available — e.g., road-segment average speed from a maps/traffic API, or manual severity labels from a few traffic officers acting as domain experts. Say this explicitly in your pitch; it shows engineering maturity rather than overselling unverified accuracy numbers.

### CIS Formula (Hackathon MVP)

```
CIS = 100 × ( w1·VehicleWeight_norm
            + w2·JunctionProximity
            + w3·LocationViolationFrequency_norm
            + w4·TimeOfDayFactor )

Suggested weights: w1 = 0.35, w2 = 0.20, w3 = 0.25, w4 = 0.20  (sum = 1.0)
```

| Factor | Definition | Example values |
|---|---|---|
| Vehicle Weight | Road-space / blockage potential by vehicle type | Scooter=1, Auto=2, Car=3, Maxi-Cab=4, Bus=5, Tanker/Truck=6 |
| Junction Proximity | 1 if violation is near a named junction, 0 if "No Junction" | binary, can be refined to a distance-decay score later |
| Location Violation Frequency | How often this exact spot (or ~30m grid cell) shows up in the dataset | normalized 0–1 across the dataset |
| Time-of-Day Factor | Weight by rush-hour proximity | Peak (8–9am, 6–7pm) = 1.0, near-peak = 0.7, off-peak = 0.3 |

This single formula is the centerpiece of your pitch deck. Walk judges through one real example from the dataset and show the score build up term by term — that's far more convincing than an accuracy percentage.

---

## 4. Dataset: Fields, Cleaning, Feature Engineering

### 4.1 Field reference (from the Traffic Police dataset)

| Field | Use in ParkIQ |
|---|---|
| `latitude`, `longitude` | Hotspot clustering, map plotting |
| `location` | Human-readable label for dashboard |
| `vehicle_type` | Vehicle Weight factor in CIS |
| `violation_type`, `offence_code` | Filtering, violation-type breakdown analytics |
| `created_datetime` | Time-of-Day factor, peak-hour analysis |
| `junction_name` | Junction Proximity factor |
| `police_station`, `center_code` | Aggregation for "officers per zone" recommendation |
| `validation_status` | Filter to `approved` records for training/scoring (drop `rejected`, treat `NULL` as pending) |
| `device_id` | Traceability — which camera/device logged the violation |

### 4.2 Cleaning & feature engineering (Python)

```python
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

# ---- 1. Load & clean ----
df = pd.read_csv("violations.csv")
df["created_datetime"] = pd.to_datetime(df["created_datetime"])
df["hour"] = df["created_datetime"].dt.hour
df["vehicle_type"] = df["vehicle_type"].str.upper().str.strip()

# Keep approved + pending records, drop rejected
df = df[df["validation_status"] != "rejected"]

# ---- 2. Vehicle impact weight (tune against real road-width data if you get time) ----
VEHICLE_WEIGHT = {
    "SCOOTER": 1, "MOTORCYCLE": 1,
    "PASSENGER AUTO": 2, "AUTO": 2,
    "CAR": 3,
    "MAXI-CAB": 4, "MAXI CAB": 4,
    "BUS": 5,
    "TANKER": 6, "TRUCK": 6, "LORRY": 6,
}
df["vehicle_weight"] = df["vehicle_type"].map(VEHICLE_WEIGHT).fillna(2)

# ---- 3. Junction proximity ----
df["near_junction"] = (df["junction_name"] != "No Junction").astype(int)

# ---- 4. Violation frequency per ~30m grid cell ----
df["geo_cell"] = (df["latitude"].round(3).astype(str) + "_" +
                  df["longitude"].round(3).astype(str))
freq = df.groupby("geo_cell").size().rename("location_violation_count")
df = df.merge(freq, on="geo_cell", how="left")

# ---- 5. Time-of-day factor ----
def time_factor(hour):
    if hour in [8, 9, 18, 19]:
        return 1.0
    elif hour in [7, 10, 17, 20]:
        return 0.7
    return 0.3
df["time_factor"] = df["hour"].apply(time_factor)

# ---- 6. CIS ----
def normalize(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

w1, w2, w3, w4 = 0.35, 0.20, 0.25, 0.20
df["CIS"] = 100 * (
    w1 * normalize(df["vehicle_weight"]) +
    w2 * df["near_junction"] +
    w3 * normalize(df["location_violation_count"]) +
    w4 * df["time_factor"]
)

df.to_csv("violations_scored.csv", index=False)
```

---

## 5. System Architecture (End-to-End Data Flow)

```
                ┌──────────────────────┐
                │ Traffic Police CSV /  │
                │ Live Camera Feed       │
                └───────────┬───────────┘
                            │
                            ▼
                ┌──────────────────────┐        ┌──────────────────┐
                │ Feature Engineering   │◄───────│  ESP32 Sensor     │
                │ + CIS Scoring         │        │  Node (live demo) │
                └───────────┬───────────┘        └──────────────────┘
                            │
              ┌─────────────┼──────────────┐
              ▼             ▼              ▼
      ┌───────────────┐ ┌──────────┐ ┌───────────────┐
      │ DBSCAN Hotspot │ │ Time-of- │ │ Enforcement    │
      │ Clustering     │ │ day risk │ │ Recommendation │
      └───────┬────────┘ └────┬─────┘ └───────┬────────┘
              └───────────────┼───────────────┘
                              ▼
                     ┌──────────────────┐
                     │ FastAPI Backend  │
                     │ + SQLite/Postgres│
                     └─────────┬────────┘
                              ▼
                     ┌──────────────────┐
                     │ Streamlit         │
                     │ Dashboard (Police)│
                     └──────────────────┘
```

---

## 6. Module 1 — Hotspot Detection & Clustering

**Goal:** find the small set of locations responsible for a disproportionate share of congestion-causing violations.

```python
coords = df[["latitude", "longitude"]].to_numpy()
db = DBSCAN(eps=0.001, min_samples=4).fit(coords)   # eps ≈ 100m in lat/lon degrees
df["hotspot_cluster"] = db.labels_                   # -1 = noise, not a hotspot

hotspots = (
    df[df["hotspot_cluster"] != -1]
    .groupby("hotspot_cluster")
    .agg(
        violation_count=("id", "count"),
        avg_CIS=("CIS", "mean"),
        center_lat=("latitude", "mean"),
        center_lon=("longitude", "mean"),
        top_location=("location", "first"),
    )
    .sort_values("avg_CIS", ascending=False)
)
print(hotspots.head(10))   # Top 10 Congestion-Critical Zones
```

`eps` and `min_samples` are tuning knobs — start with `eps=0.001` (~100m) and `min_samples=4`, then adjust based on how tight/loose your clusters look on the map.

---

## 7. Module 2 — Congestion Impact Score Engine

Already covered in Section 3–4. Operationally, this runs:

- **Batch mode:** over the full historical CSV, to populate `violations_scored.csv` and seed the dashboard before the demo.
- **Live mode:** whenever a new violation event arrives (from a camera pipeline or the ESP32 node), the same scoring function runs on the single new record and updates the database.

---

## 8. Module 3 — Hotspot Prediction (Forecasting)

**Hackathon-realistic scope:** don't promise an LSTM/Temporal Fusion Transformer unless you have time left after the MVP is solid — that's a stretch goal, not the core deliverable.

A defensible, buildable version:

1. Group historical violations by `(geo_cell, hour_of_day, day_of_week)`.
2. Compute the historical average violation count and average CIS for each combination.
3. For "predicting the next 1 hour," look up the current `(geo_cell, hour, day_of_week)` bucket and return its historical average as the forecast, with a confidence interval from the standard deviation.

This is a **seasonal-average baseline forecast** — simple, explainable, and genuinely better than no forecast at all. If time allows, upgrade to a `Prophet` or `statsmodels` seasonal model per top-10 hotspot only (not city-wide) to keep compute manageable in a hackathon timeframe.

---

## 9. Module 4 — Enforcement Recommendation Engine

Maps CIS → priority → suggested action. Pure business logic, no ML needed:

| CIS Range | Priority | Suggested Action |
|---|---|---|
| 0–30 | LOW | Monitor only |
| 31–55 | MEDIUM | Issue notice / warning |
| 56–80 | HIGH | Dispatch officer for immediate clearance |
| 81–100 | CRITICAL | Tow vehicle / emergency clearance |

```python
def priority_and_action(cis):
    if cis <= 30:  return "LOW", "Monitor"
    if cis <= 55:  return "MEDIUM", "Issue Notice"
    if cis <= 80:  return "HIGH", "Immediate Enforcement"
    return "CRITICAL", "Tow Vehicle"

df[["priority", "action"]] = df["CIS"].apply(
    lambda x: pd.Series(priority_and_action(x))
)
```

This table is also what you show live on stage: pick a real row from the dataset, run it through this function, and the room sees how a raw violation becomes a dispatch instruction.

---

## 10. Module 5 — IoT Smart Parking Occupancy Node (ESP32)

This is your physical proof-of-concept: a single sensor node simulating one "smart curb" that a city-wide rollout would replicate thousands of times.

### 10.1 Bill of Materials (from your kit — nothing extra required)

| Component | Quantity | Notes |
|---|---|---|
| ESP32 Dev Board | 1 | Main controller, WiFi-enabled |
| HC-SR04 Ultrasonic Sensor | 1 | Distance/occupancy sensing |
| Red LED | 1 | Violation indicator |
| Green LED | 1 | Vacant indicator |
| Blue LED | 1 | Occupied (within grace period) indicator |
| Buzzer | 1 | Audible violation alert |
| Breadboard | 1 | Prototyping |
| Jumper wires | ~15 | Connections |
| Resistors (220–330Ω) | 3 | LED current limiting — check your kit, most starter kits include these |
| Resistors (1kΩ + 2kΩ) | 1 each | Voltage divider for the echo pin (see note below) |

**Important wiring note:** the HC-SR04's `ECHO` pin outputs at the same voltage as its `VCC` (5V if powered from the ESP32's `VIN`/5V pin). ESP32 GPIOs are only 3.3V-tolerant. Feed `ECHO` through a simple voltage divider (1kΩ from ECHO to the junction node, 2kΩ from that junction to GND, and take the signal for the ESP32 from the junction) to avoid damaging the GPIO pin over repeated use.

### 10.2 Pin mapping

| ESP32 Pin | Connects to |
|---|---|
| GPIO 5 | HC-SR04 `TRIG` |
| GPIO 18 | HC-SR04 `ECHO` (via voltage divider) |
| GPIO 25 | Red LED anode (→ 220Ω resistor) |
| GPIO 26 | Green LED anode (→ 220Ω resistor) |
| GPIO 27 | Blue LED anode (→ 220Ω resistor) |
| GPIO 14 | Buzzer positive |
| 5V (VIN) | HC-SR04 `VCC` |
| GND | HC-SR04 `GND`, LED cathodes (common cathode), buzzer negative |

### 10.3 Behavior logic

1. Sensor continuously measures distance to the curb/parking zone floor.
2. If distance drops below a calibrated threshold (something is parked), the zone enters **OCCUPIED** (blue LED) and a timer starts.
3. If the object stays past a grace period in a no-parking zone, the zone flips to **VIOLATION** (red LED + buzzer) and an event is sent to the backend.
4. When the object leaves, the zone resets to **VACANT** (green LED) and a "vacated" event with total duration is sent.
5. Debouncing (requiring several consecutive readings) prevents false triggers from sensor noise, pedestrians walking past, etc.

### 10.4 Firmware (Arduino/ESP32, C++)

```cpp
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <time.h>

// ---------- CONFIG ----------
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* SERVER_URL    = "http://<your-backend-ip>:8000/api/sensor-event";

const char* DEVICE_ID = "ESP32-NODE-01";
const char* ZONE_ID   = "A1";   // matches a zone/junction name in your dataset

// Pin mapping
const int TRIG_PIN   = 5;
const int ECHO_PIN   = 18;      // via voltage divider — see wiring notes
const int LED_RED    = 25;
const int LED_GREEN  = 26;
const int LED_BLUE   = 27;
const int BUZZER_PIN = 14;

// Tunable thresholds — calibrate to your mounting height/distance
const float OCCUPIED_DISTANCE_CM       = 80.0;
const int   DEBOUNCE_READINGS          = 5;
const unsigned long VIOLATION_THRESHOLD_MS = 120UL * 1000; // 2 min demo threshold

// ---------- STATE ----------
enum ZoneState { VACANT, OCCUPIED, VIOLATION };
ZoneState currentState = VACANT;
unsigned long occupiedSince = 0;
int belowCount = 0, aboveCount = 0;

float readDistanceCm() {
  digitalWrite(TRIG_PIN, LOW);  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH); delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long duration = pulseIn(ECHO_PIN, HIGH, 30000); // 30ms timeout
  if (duration == 0) return -1;
  return duration * 0.0343 / 2.0;
}

void setLed(bool r, bool g, bool b) {
  digitalWrite(LED_RED, r); digitalWrite(LED_GREEN, g); digitalWrite(LED_BLUE, b);
}

void sendEvent(const char* eventType, unsigned long durationSec, float distanceCm) {
  if (WiFi.status() != WL_CONNECTED) return;
  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");

  StaticJsonDocument<256> doc;
  doc["device_id"]   = DEVICE_ID;
  doc["zone_id"]      = ZONE_ID;
  doc["event"]        = eventType;
  doc["duration_sec"] = durationSec;
  doc["distance_cm"]  = distanceCm;
  doc["timestamp"]    = (long)time(nullptr);

  String payload;
  serializeJson(doc, payload);
  http.POST(payload);
  http.end();
}

void setup() {
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT); pinMode(ECHO_PIN, INPUT);
  pinMode(LED_RED, OUTPUT); pinMode(LED_GREEN, OUTPUT); pinMode(LED_BLUE, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) delay(500);
  configTime(5 * 3600 + 1800, 0, "pool.ntp.org"); // IST offset

  setLed(0, 1, 0); // start green / vacant
}

void loop() {
  float distance = readDistanceCm();
  bool occupiedNow = (distance > 0 && distance < OCCUPIED_DISTANCE_CM);

  if (occupiedNow) { belowCount++; aboveCount = 0; }
  else              { aboveCount++; belowCount = 0; }

  switch (currentState) {
    case VACANT:
      if (belowCount >= DEBOUNCE_READINGS) {
        currentState = OCCUPIED;
        occupiedSince = millis();
        setLed(0, 0, 1);
        sendEvent("OCCUPANCY_START", 0, distance);
      }
      break;

    case OCCUPIED: {
      unsigned long elapsed = millis() - occupiedSince;
      if (aboveCount >= DEBOUNCE_READINGS) {
        currentState = VACANT;
        setLed(0, 1, 0);
        sendEvent("VACATED", elapsed / 1000, distance);
      } else if (elapsed > VIOLATION_THRESHOLD_MS) {
        currentState = VIOLATION;
        setLed(1, 0, 0);
        digitalWrite(BUZZER_PIN, HIGH);
        sendEvent("VIOLATION_CONFIRMED", elapsed / 1000, distance);
      }
      break;
    }

    case VIOLATION: {
      unsigned long elapsed = millis() - occupiedSince;
      if (aboveCount >= DEBOUNCE_READINGS) {
        currentState = VACANT;
        setLed(0, 1, 0);
        digitalWrite(BUZZER_PIN, LOW);
        sendEvent("VACATED", elapsed / 1000, distance);
      } else {
        static unsigned long lastHeartbeat = 0;
        if (millis() - lastHeartbeat > 30000) {
          sendEvent("VIOLATION_ONGOING", elapsed / 1000, distance);
          lastHeartbeat = millis();
        }
      }
      break;
    }
  }
  delay(200); // ~5 readings/sec
}
```

### 10.5 Demo data payload sent to backend

```json
{
  "device_id": "ESP32-NODE-01",
  "zone_id": "A1",
  "event": "VIOLATION_CONFIRMED",
  "duration_sec": 132,
  "distance_cm": 42.7,
  "timestamp": 1750039200
}
```

---

## 11. Module 6 — CCTV + YOLOv8 Computer Vision Pipeline

> **Status: ✅ Fully Implemented** (`src/cctv_detector.py`)

The pipeline continuously processes frames from any video source and produces structured violation events:

```
Video Frame
    ↓
YOLOv8 Nano (vehicle detection, COCO classes: car/motorcycle/bus/truck)
    ↓
ByteTrack multi-object tracker (persistent track IDs across frames)
    ↓
Zone Checker (cv2.pointPolygonTest against NO_PARKING_ZONES polygons)
    ↓
Dwell Timer (per track_id, seconds inside zone)
    ↓
CIS Engine (base vehicle score + time penalty + zone bonus → 0-100)
    ↓
FastAPI POST → /api/cctv-event → SQLite → Dashboard
    ↓
ESP32 Buzzer trigger (via /api/set-buzzer) on VIOLATION_CONFIRMED
```

**Key parameters (tunable in `cctv_detector.py`):**

| Parameter | Default | Description |
|---|---|---|
| `VIOLATION_SEC` | 8s | Dwell time before violation is confirmed |
| `TELEMETRY_SEC` | 3s | Periodic telemetry POST interval |
| `CONF_THRESHOLD` | 0.35 | YOLOv8 confidence threshold |
| `FRAME_SKIP` | 2 | Process every Nth frame (speed vs. accuracy) |

**Zone calibration:**
```bash
python src/cctv_detector.py --calibrate
# Click 4 corners of each no-parking zone → coordinates printed for copy-paste
```

---

## 12. Backend API Design

> **Status: ✅ Fully Implemented** (`src/backend_api.py`)

**Complete endpoint table:**

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/sensor-event` | POST | Receives live ESP32 occupancy events |
| `/api/cctv-event` | POST | Receives YOLOv8 detection events |
| `/api/cctv-events/recent` | GET | Latest N CCTV detections (paginated) |
| `/api/cctv-events/stats` | GET | Totals, avg CIS, top vehicle, by-zone breakdown |
| `/api/cctv-events/high-risk` | GET | High-risk zone table: avg/max CIS + avg duration |
| `/api/video_feed` | GET | MJPEG stream from `data/current_frame.jpg` |
| `/api/device-state` | GET | Current mode + buzzer state for ESP32 polling |
| `/api/set-mode` | POST | Switch mode: `IOT` or `CCTV` |
| `/api/set-buzzer` | POST | Toggle ESP32 hardware buzzer from CCTV pipeline |

---

## 12. Database Schema

For the hackathon, **SQLite is enough** — don't burn build time setting up Postgres/AWS unless you have spare days. The schema below works identically in both.

```sql
CREATE TABLE violations (
    id                TEXT PRIMARY KEY,
    latitude          REAL,
    longitude         REAL,
    location          TEXT,
    vehicle_number    TEXT,
    vehicle_type      TEXT,
    violation_type    TEXT,
    offence_code      TEXT,
    created_datetime  TIMESTAMP,
    junction_name     TEXT,
    police_station    TEXT,
    validation_status TEXT,
    cis_score         REAL,
    hotspot_cluster   INTEGER,
    priority          TEXT,
    action            TEXT
);

CREATE TABLE sensor_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id     TEXT,
    zone_id       TEXT,
    event         TEXT,
    duration_sec  INTEGER,
    distance_cm   REAL,
    timestamp     INTEGER
);

CREATE TABLE hotspots (
    cluster_id      INTEGER PRIMARY KEY,
    center_lat      REAL,
    center_lon      REAL,
    violation_count INTEGER,
    avg_cis         REAL,
    top_location    TEXT,
    last_updated    TIMESTAMP
);

CREATE TABLE enforcement_actions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    violation_id     TEXT REFERENCES violations(id),
    action           TEXT,
    priority         TEXT,
    assigned_officer TEXT,
    status           TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 13. Dashboard (Frontend)

> **Status: ✅ Fully Implemented** (`src/dashboard.py`)

All five dashboard pages are live. PyDeck (WebGL) replaces Folium for performance on the full 248k-record dataset.

### Implemented Pages

| Page | Status | Key Features |
|---|---|---|
| 🗺️ **Live Violations Map** | ✅ Live | WebGL ScatterplotLayer + HexagonLayer, 5 KPI metrics, vehicle/hour charts |
| ⚠️ **High-Risk Area Analysis** | ✅ Live | Hotspot table, heatmap, CIS distribution, AI suggestions (7 types) |
| 📋 **Enforcement Queue** | ✅ Live | Priority-filtered table, colour-coded rows, tow/action counts |
| 📡 **IoT Sensor Monitor** | ✅ Live | Auto-refresh every 2s, event counts, live DB query |
| 🎥 **CCTV AI Monitor** | ✅ Live | MJPEG stream embed, real-time event table, zone bar chart, zone suggestions |

### Auto-refresh Architecture

Streamlit's `@st.fragment(run_every=N)` decorator is used for the IoT and CCTV pages so only those fragments re-run on each refresh cycle — the rest of the page stays static, avoiding full rerenders.

---

## 14. High-Risk Area Analysis & AI Suggestions

> **Status: ✅ Implemented** — Page 2 of the dashboard

The **⚠️ High-Risk Area Analysis** page goes beyond a simple list of violations. It:

1. **Identifies top-10 hotspot locations** by grouping `violations_scored.csv` on `location` field for records with `CIS > 70`, computing `violation_count`, `avg_cis`, and `max_cis`.
2. **Renders a HeatmapLayer** (PyDeck) showing high-risk concentration geographically.
3. **Shows CIS distribution** across 5 bands (LOW / MEDIUM / HIGH / HIGH+ / CRITICAL) as a bar chart.
4. **Generates 7 data-driven AI suggestions** based on actual dataset patterns:

| Suggestion | Trigger Condition |
|---|---|
| 🚨 Immediate Tow Operations | CRITICAL violations (CIS > 80) exist |
| 🚛 Vehicle-Specific Enforcement | Top vehicle type in high-risk zone |
| ⏰ Peak-Hour Patrol Scheduling | Peak violation hour identified |
| 🚦 Junction Clearance Priority | Junction-adjacent violations detected |
| 📱 Digital Notice Dispatch | MEDIUM violations present (e-challan automation) |
| 📡 IoT Sensor Expansion | Recommends sensor deployment at top-5 coords |
| 🧠 Predictive Pre-positioning | Historical peak-hour pattern detected |

Each suggestion is colour-coded (red/orange/green) and labelled HIGH / MEDIUM / LOW priority.

---

## 14. Stretch Goals: Digital Twin & Emergency Lane Protection

Only attempt these once the core (Sections 6–13) is working end-to-end. They are differentiators, not requirements.

**Digital Twin (simplified, achievable version):** rather than a full traffic simulation engine, build a simple "what-if" calculator: given a violation's CIS and an estimated road-capacity-reduction percentage for its vehicle type, output an estimated percentage improvement in flow if the vehicle is removed. This is a heuristic, not a true simulation — present it honestly as a first-order estimate, which is still a meaningful and demoable feature.

**Emergency Vehicle Lane Protection:** take a hardcoded ambulance route (a list of lat/lon waypoints for the demo) and check it against your hotspot table for overlapping high-CIS zones within a buffer distance. Surface a simple alert: "Zone A1 on Route X has a CRITICAL violation — recommend clearance before dispatch." This can be a single SQL/GeoPandas query, not a routing engine.

---

## 15. Tech Stack — Hackathon MVP vs. Production Vision

Be explicit in your pitch about which is which — judges respect a team that scoped realistically.

| Layer | Hackathon MVP (build this) | Production Vision (pitch this) |
|---|---|---|
| Database | SQLite | PostgreSQL + PostGIS |
| Backend | FastAPI (single instance) | FastAPI on Docker/Kubernetes |
| ML | Heuristic CIS formula + DBSCAN | CIS formula calibrated with real traffic-speed data; XGBoost for refined scoring |
| CV (optional) | Skip, or a pretrained YOLOv8 demo on a few sample images | Full YOLOv8 pipeline on live CCTV feeds |
| IoT | 1 ESP32 + HC-SR04 demo node | Fleet of nodes over LoRaWAN/NB-IoT for city coverage |
| Maps | Folium (free, offline-friendly) | MapMyIndia APIs (per problem statement's data partner) |
| Dashboard | Streamlit | React + dedicated ops dashboard |
| Deployment | Local laptop for the demo | AWS/Render with CI/CD |

---

## 16. 4-Week Build Roadmap

**Week 1 — Data foundation**
Clean the dataset, build the feature engineering pipeline (Section 4), implement the CIS formula, run DBSCAN clustering, get a static map visualization working with real data.

**Week 2 — Decision layer**
Build the enforcement priority/action logic, stand up the FastAPI backend with the violations/hotspots/enforcement-queue endpoints, wire the Streamlit dashboard to real data (Live Map + Enforcement Queue pages).

**Week 3 — IoT integration**
Wire and flash the ESP32 + HC-SR04 node, test the occupancy/violation state machine on the bench, connect it to the FastAPI `/api/sensor-event` endpoint, add the IoT Sensor Monitor dashboard page.

**Week 4 — Polish & narrative**
Add the seasonal-average hotspot forecast (Section 8), build one stretch goal if time allows (digital twin calculator or emergency lane check), record the demo video, finalize the pitch deck and this documentation, rehearse the live demo end-to-end at least three times.

---

## 17. Demo Script for Judges

A suggested 4-minute flow:

1. **Hook (30s):** "Police already collect thousands of parking violations. The problem isn't detection — it's knowing which ones actually matter."
2. **Show the map (60s):** open the Live Violations Map, point out a real hotspot from the dataset (e.g., Coles Road/Frazer Town), show the color-coded CIS.
3. **Walk through one CIS calculation live (45s):** pick one row, show the formula breakdown, land on the final score and priority.
4. **Physical demo (60s):** place a toy car/object near the ESP32+HC-SR04 node, show the LED transition green → blue → red and buzzer activation, show the event landing in the dashboard's IoT Sensor Monitor in real time.
5. **Close (15s):** state the operational outcome — "this turns a flat violation log into a ranked dispatch list any officer can act on today."

---

## 18. Evaluation Metrics

Since there's no ground-truth congestion measurement available, frame your evaluation honestly:

- **Clustering quality:** silhouette score for the DBSCAN hotspots, and a sanity check that top hotspots correspond to genuinely busy areas (Frazer Town, Koramangala, Shivajinagar — all visible in the sample data).
- **CIS sensitivity:** show that CIS responds correctly to each input (e.g., a tanker near a junction during rush hour scores higher than a scooter on a side road at midnight) — a few worked examples is the right level of rigor here.
- **System metrics:** sensor node detection latency, end-to-end event-to-dashboard latency, API response time.
- **Operational metric (illustrative, not measured live):** estimated percentage of violations that would be reprioritized away from a naive "first-come-first-served" enforcement order.

---

## 19. Risks & Honest Limitations

- **No real congestion ground truth in the dataset.** Mitigation: explainable heuristic CIS instead of an unverified supervised model (see Section 3).
- **ESP32 WiFi dependency for the live demo.** Mitigation: test the venue's WiFi/hotspot beforehand; have a local-buffer fallback (log to serial/SD if WiFi drops) so the demo doesn't fail live.
- **Single sensor node ≠ city-wide system.** Be explicit that the IoT node is a proof-of-concept for one zone, with the production vision (Section 15) explaining how it scales.
- **Time constraints.** Treat Sections 6–13 as the required MVP. Treat Section 14 (digital twin, emergency lane) as optional — don't let stretch goals threaten the core build.

---

## 20. Submission Checklist

- [x] Cleaned dataset + feature engineering script (`src/data_pipeline.py`)
- [x] CIS scoring function, tested against sample rows
- [x] DBSCAN hotspot clustering, top-10 hotspot table
- [x] FastAPI backend running — 9 endpoints live (`src/backend_api.py`)
- [x] SQLite database populated and queryable (`data/parkiq.db`)
- [x] Streamlit dashboard: all **5 pages** working (`src/dashboard.py`)
- [x] **High-Risk Area Analysis** page with AI suggestions (⚠️ new)
- [x] **YOLOv8 CCTV pipeline** with ByteTrack, CIS, buzzer trigger (`src/cctv_detector.py`)
- [x] MJPEG live feed embedded directly in Streamlit (no popup window)
- [x] ESP32 + HC-SR04 node assembled, firmware flashed, tested on the bench
- [x] IoT Sensor Monitor dashboard page receiving live events (auto-refresh 2s)
- [x] `/api/cctv-events/high-risk` endpoint returning zone-level CIS analytics
- [x] `README.md` created with setup guide, architecture overview, and demo script
- [x] `ParkIQ_Architecture.md` updated to reflect fully implemented state
- [ ] Demo video recorded (backup in case live demo fails)
- [ ] Pitch deck summarizing problem → CIS innovation → architecture → demo → impact

---

## 21. Future Scope

- Calibrate CIS weights against real traffic-speed data (e.g., correlating violation locations with road-segment average speed from a maps API) once available, then revisit supervised ML scoring.
- Expand the IoT node fleet using LoRaWAN/NB-IoT for low-power, low-bandwidth city-wide coverage instead of WiFi-only nodes.
- Integrate with live CCTV feeds via YOLOv8 for automatic violation logging, feeding directly into the CIS pipeline.
- Build a true emergency-vehicle routing integration (not just a static route check) once a real routing API is available.
- Move from SQLite/Folium to PostgreSQL+PostGIS/MapMyIndia for a production-scale deployment with Bengaluru Traffic Police.

---

*Project: ParkIQ — AI-Powered Parking Impact Intelligence for Bengaluru*
*From Parking Violations to Congestion Intelligence*
