# 🚦 ParkIQ — AI-Powered Parking Impact Intelligence

> **Gridlock Hackathon 2.0 — Round 2**
> *From Parking Violations to Congestion Intelligence*

---

## 🎯 What is ParkIQ?

ParkIQ transforms raw illegal-parking violation records into **prioritised, actionable enforcement intelligence** for Bengaluru Traffic Police. It answers the question that actually matters:

> *"How much congestion is this specific violation causing, and which ones should officers clear first?"*

---

## 🏗️ System Architecture

```
CCTV / Webcams (Multi-Node) ──► YOLOv8 Detector ─────────►┐
BTP Jan-May Dataset ──────────► Data Pipeline ────────────►│
ESP32 + HC-SR04 Sensor ───────► FastAPI Backend ◄──────────┘
                                        │
                              ┌─────────┴─────────┐
                              │   SQLite DB        │
                              └─────────┬─────────┘
                              ┌─────────▼─────────┐
                              │ Streamlit Dashboard│
                              │ (Police Console)   │
                              └───────────────────┘
```

---

## 📦 Project Structure

```
GridLock/
├── src/
│   ├── dashboard.py       # Streamlit Dashboard (Map, Analytics, CCTV Network)
│   ├── backend_api.py     # FastAPI backend (REST + MJPEG stream)
│   ├── cctv_detector.py   # Distributed Multi-Cam YOLOv8 + CIS engine
│   └── data_pipeline.py   # Feature Eng. + DBSCAN Hotspots + CIS Scoring
├── iot/
│   └── esp32_firmware/
│       └── esp32_multi_sensor/
│           └── esp32_multi_sensor.ino   # Arduino hardware firmware
├── data/
│   ├── violations.csv           # Raw Jan-May Police dataset (248k records)
│   ├── violations_scored.csv    # CIS-scored output
│   └── parkiq.db                # Real-time SQLite database
├── yolov8n.pt                   # YOLOv8 Nano weights
├── requirements.txt
├── ParkIQ_Architecture.md       # Architecture spec
└── README.md                    # This file
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Score the Dataset (AI Processing)
```bash
python src/data_pipeline.py
```
*Parses 5 months of Bangalore Police data (248k records), clusters hotspots using DBSCAN, and calculates Congestion Impact Scores (CIS).*

### 3. Start the Backend API
```bash
python src/backend_api.py
```

### 4. Launch the AI Dashboard
```bash
streamlit run src/dashboard.py
```

### 5. Launch Distributed CCTV AI Cameras
You can run multiple independent cameras tracking different specific physical zones simultaneously!

**Run Camera 1:**
```bash
python src/cctv_detector.py --device-id "CCTV-CAM-01"
```

**Run Camera 2 (using your 2nd Phone IP):**
```bash
python src/cctv_detector.py --source "http://192.168.29.100:8080/video" --device-id "CCTV-CAM-02"
```

### 6. Calibrate Road Width & No-Parking Zones
To precisely map physical road dimensions and custom no-parking polygons for your hardware setup:

1. Run the calibration mode for Camera 1:
   ```bash
   python src/cctv_detector.py --device-id "CCTV-CAM-01" --calibrate
   ```
2. A window will open. **Click 2 points** on the LEFT and RIGHT edges of the road to set the road width.
3. **Click the 4 corners** of each no-parking zone. Press `ENTER`.
4. The terminal will print out a dictionary string.
5. Copy that string and paste it into the `ROAD_CONFIG` variable inside `src/cctv_detector.py` (around line 66).
6. Repeat the exact same process for Camera 2 by changing the device-id to `"CCTV-CAM-02"` and specifying the source!

---

## 🖥️ Dashboard Features

| Page | Description |
|---|---|
| 🧠 **AI Predictive Impact Map** | Predictive mapping using historical data. Features dynamic time-range sliders, real-time dispatch directives, and **Live Zone Analytics** (Vehicle type and violation category breakdown charts). |
| ⚠️ **High-Risk Area Analysis** | Top-10 congestion hotspots, AI-generated resolution policies, and macro-level city metrics. |
| 🌐 **City-Wide CCTV Network** | Real-time live feed monitoring, displaying current active violations across the distributed multi-camera nodes. |
| 📡 **IoT Sensor Monitor** | Live hardware feed from ESP32 nodes indicating Vacant/Occupied/Violation states. |

*(A global `🚨 Live Alert` threat monitor runs constantly in the background, simulating a massive network of 200,000 cameras to push critical tow-truck dispatch toasts to the user).*

---

## 🧠 Core Innovation — Congestion Impact Score (CIS)

```
CIS = ( BaseVehicleWeight + ZonePenalty + LaneBlockagePenalty ) × TimeOfDayFactor
```
*Note: The **Lane Blockage Penalty** dynamically calculates the vehicle's bounding-box width relative to the physical road width. The **TimeOfDayFactor** applies a 1.3x multiplier during rush hour and 0.5x during night hours!*

| CIS Range | Hardware Trigger | Action |
|---|---|---|
| 0–40 | None | Monitor only |
| 41–70 | 🟡 Solid Yellow LED | Dispatch patrol for e-challan |
| 71–100 | 🔴 Solid Red + Buzzer | Emergency heavy-tow clearance |

---

## 🚀 Hackathon Pitch Script (3 Minutes)

1. **The Hook (30s):** "Currently, traffic police patrol blindly. They waste time giving tickets on quiet backstreets while a single truck completely blocks Brigade Road. We built ParkIQ, powered by 5 months of real Bangalore Traffic Police data, to fix this."
2. **The Macro View (30s):** Go to *High-Risk Analysis*. "By analyzing 248,000 records, our AI automatically found the top 10 worst streets in the city. But parking is dynamic..."
3. **The Micro View (60s):** Go to *AI Predictive Impact Map*. Slide the Time Range to `08:00 - 11:00`. "When we simulate morning rush hour, look at the Live Analytics charts below. We see exactly which vehicles cause the gridlock. The system generates an immediate dispatch directive telling tow trucks exactly where to go."
4. **The Live Demo (60s):** Run the CCTV detector / ESP32. "And it works in real-time. This camera calculates CIS dynamically. If a heavy truck parks in our custom-drawn polygon during rush hour, the CIS breaks 80, the dashboard alerts globally, and the hardware buzzer goes off to initiate towing."

---

## ⚠️ Honest Limitations
- No real congestion ground truth in dataset → explainable heuristic CIS (not black-box ML).
- YOLOv8 Nano struggles with very small/toy vehicles → bounding-box heuristic applied specifically for the tabletop demo model.
- ESP32 demo node = 1 zone proof-of-concept (production: LoRaWAN fleet).

---
*ParkIQ — Gridlock Hackathon 2.0 | Team Submission*
