<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:05080d,50:0e1620,100:eab308&height=180&section=header&text=ParkIQ&fontSize=64&fontColor=fef08a&fontAlignY=40&desc=AI-Powered%20Parking%20Impact%20Intelligence&descSize=18&descAlignY=62&animation=fadeIn" />

<br/>

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com/?font=JetBrains+Mono&size=20&duration=2800&pause=900&color=FDE047&center=true&vCenter=true&width=620&lines=Transform+parking+violations+into+intelligence.;Prioritise+enforcement+actions+with+AI.;Clear+gridlock+before+it+cascades.;Respond+with+data%2C+not+guesswork." alt="Typing SVG" /></a>

<br/>

<p>
  <img src="https://img.shields.io/badge/STATUS-OPERATIONAL-eab308?style=for-the-badge&labelColor=05080d" />
  <img src="https://img.shields.io/badge/BUILD-PASSING-eab308?style=for-the-badge&labelColor=05080d" />
  <img src="https://img.shields.io/badge/GRIDLOCK-HACKATHON_2.0-eab308?style=for-the-badge&labelColor=05080d" />
</p>

<p>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLOv8-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/C++-00599C?style=flat-square&logo=c%2B%2B&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" />
</p>

<br/>

**[Dashboard](#-platform-modules)** &nbsp;·&nbsp;
**[Architecture](#-architecture)** &nbsp;·&nbsp;
**[Quick Start](#-quick-start)**

</div>

<br/>

---

## 🎯 Mission Brief

Currently, traffic police patrol blindly. They waste time giving tickets on quiet backstreets while a single truck completely blocks a main road.

**ParkIQ** transforms raw illegal-parking violation records into **prioritised, actionable enforcement intelligence** for Traffic Police. It answers the question that actually matters:

> *"How much congestion is this specific violation causing, and which ones should officers clear first?"*

Built for **GridLock Hackathon 2.0**, it functions as a command-center decision support tool powered by historical data and real-time YOLOv8 multi-camera inference.

<table>
<tr>
<td width="25%" align="center"><b>Track</b><br/><sub>Monitor physical zones with YOLOv8</sub></td>
<td width="25%" align="center"><b>Score</b><br/><sub>Calculate Congestion Impact Score (CIS)</sub></td>
<td width="25%" align="center"><b>Sync</b><br/><sub>Evaluate global state & trigger IoT alerts</sub></td>
<td width="25%" align="center"><b>Act</b><br/><sub>Dispatch tow trucks to critical hotspots</sub></td>
</tr>
</table>

---

## 🏗️ Architecture

```mermaid
flowchart TD
    %% Custom Styles
    classDef input fill:#0e1620,stroke:#eab308,stroke-width:2px,color:#fef08a,rx:5,ry:5
    classDef ai fill:#0e1620,stroke:#06b6d4,stroke-width:2px,color:#e0f7fa,rx:5,ry:5
    classDef core fill:#06243a,stroke:#3b82f6,stroke-width:2px,color:#eff6ff,rx:10,ry:10
    classDef db fill:#0e1620,stroke:#f59e0b,stroke-width:2px,color:#fef3c7
    classDef ui fill:#eab308,stroke:#a16207,stroke-width:3px,color:#05080d,rx:5,ry:5
    classDef hardware fill:#450a0a,stroke:#ef4444,stroke-width:2px,color:#fca5a5,rx:10,ry:10

    subgraph Inputs ["📡 Data Sources"]
        A("📹 Multi-Node CCTV Feeds"):::input
        C("📊 Historical Police Data (248k Records)"):::input
    end

    subgraph AI ["🧠 Processing Engine"]
        B("👁️ YOLOv8 and CIS Engine"):::ai
        D("⚙️ DBSCAN Data Pipeline"):::ai
        E("🌍 Global State Evaluator"):::core
    end

    subgraph Infrastructure ["⚙️ Backend & Storage"]
        G("⚡ FastAPI Backend"):::core
        F[("🗄️ SQLite Database")]:::db
    end

    subgraph Endpoints ["🚨 Action & Interface"]
        I("🖥️ Streamlit Police Console"):::ui
        H("🔴 ESP32 IoT Hardware Node"):::hardware
    end

    %% Routing
    A --> B
    C --> D
    
    B --> E
    D --> F
    
    E ==>|"REST and MJPEG stream"| G
    E -.->|"Physical Feedback"| H
    
    G --> F
    F ==> I

    %% Subgraph Styles
    style Inputs fill:transparent,stroke:#334155,stroke-width:2px,stroke-dasharray: 5 5
    style AI fill:transparent,stroke:#334155,stroke-width:2px,stroke-dasharray: 5 5
    style Infrastructure fill:transparent,stroke:#334155,stroke-width:2px,stroke-dasharray: 5 5
    style Endpoints fill:transparent,stroke:#334155,stroke-width:2px,stroke-dasharray: 5 5
```

**Data flow:** Live multi-node CCTV feeds pass through a YOLOv8 engine which tracks, scores, and measures time. Historical police data goes through a feature engineering pipeline for DBSCAN hotspot clustering. The outputs fuse into a FastAPI backend and SQLite DB, instantly updating the Streamlit police console and signalling physical ESP32 IoT nodes in the real world.

---

## 🧠 Core Innovation — Congestion Impact Score (CIS)

Instead of a binary "illegal/legal" status, every violation receives a real-time **Congestion Impact Score**:

```
CIS = ( BaseVehicleWeight + ZonePenalty + LaneBlockagePenalty ) × TimeOfDayFactor
```

*Note: The **Lane Blockage Penalty** dynamically calculates the vehicle's bounding-box width relative to the physical road width. The **TimeOfDayFactor** applies a 1.3x multiplier during rush hour and 0.5x during night hours!*

| CIS Range | Hardware Trigger (Global Highest Priority) | Action |
|---|---|---|
| **0–40** | None / 🟢 Green LED (Vacant/Low) | Monitor only |
| **41–70** | 🟡 Solid Yellow LED | Dispatch patrol for e-challan |
| **71–100** | 🔴 Solid Red + Buzzer | Emergency heavy-tow clearance |

---

## 🖥️ Platform Modules

| Module | Function |
|---|---|
| 🧠 **AI Predictive Impact Map** | Predictive mapping using historical data. Features dynamic time-range sliders, real-time dispatch directives, and **Live Zone Analytics**. |
| ⚠️ **High-Risk Area Analysis** | Top-10 congestion hotspots, AI-generated resolution policies, and macro-level city metrics. |
| 🌐 **City-Wide CCTV Network** | Real-time live feed monitoring across distributed nodes. Stabilized responsive video grids. |
| 📡 **IoT Sensor Monitor** | Live hardware feed from ESP32 nodes indicating Vacant/Occupied/Violation states. |

*(A global `🚨 Live Alert` threat monitor runs constantly in the background, simulating a massive network of 200,000 cameras to push critical tow-truck dispatch toasts to the user).*

---

## 📸 Screenshots

<div align="center">

<table>
<tr>
<td width="50%" align="center" valign="top">
<b>Predictive Impact Map</b><br/><br/>
<img src="assets/AI%20Predictive%20Impact%20Map.png" width="100%"/>
</td>
<td width="50%" align="center" valign="top">
<b>High-Risk Hotspots</b><br/><br/>
<img src="assets/High-Risk%20Area%20Analysis.png" width="100%"/>
</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">
<b>Live CCTV Network</b><br/><br/>
<img src="assets/City-Wide%20CCTV%20Network.png" width="100%"/>
</td>
<td width="50%" align="center" valign="top">
<b>IoT Sensor Monitor</b><br/><br/>
<img src="assets/IoT%20Sensor%20Monitor.png" width="100%"/>
</td>
</tr>
</table>

</div>

---

## 🛠️ Tech Stack

<div align="center">

<table>
<tr><th>Layer</th><th>Stack</th></tr>
<tr>
<td><b>Frontend (Dashboard)</b></td>
<td>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white"/>
</td>
</tr>
<tr>
<td><b>Backend</b></td>
<td>
<img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white"/>
<img src="https://img.shields.io/badge/Python_3.11-3776AB?style=flat-square&logo=python&logoColor=white"/>
</td>
</tr>
<tr>
<td><b>Computer Vision & AI</b></td>
<td>
<img src="https://img.shields.io/badge/YOLOv8-blue?style=flat-square"/>
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white"/>
<img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white"/>
</td>
</tr>
<tr>
<td><b>Hardware / IoT</b></td>
<td>
<img src="https://img.shields.io/badge/ESP32-E7352C?style=flat-square&logo=espressif&logoColor=white"/>
<img src="https://img.shields.io/badge/C++-00599C?style=flat-square&logo=c%2B%2B&logoColor=white"/>
<img src="https://img.shields.io/badge/Arduino-00979D?style=flat-square&logo=arduino&logoColor=white"/>
</td>
</tr>
</table>

</div>

---

## 📂 Repository Structure

```
GridLock/ParkIQ/
│
├── src/
│   ├── dashboard.py       # Streamlit Dashboard (Map, Analytics, CCTV Network)
│   ├── backend_api.py     # FastAPI backend (REST + MJPEG stream)
│   ├── cctv_detector.py   # Distributed Multi-Cam YOLOv8 + CIS engine
│   └── data_pipeline.py   # Feature Eng. + DBSCAN Hotspots + CIS Scoring
│
├── iot/
│   └── esp32_firmware/
│       └── esp32_multi_sensor/
│           └── esp32_multi_sensor.ino   # Arduino hardware firmware
│
├── data/
│   ├── violations.csv           # Raw Jan-May Police dataset (248k records)
│   ├── violations_scored.csv    # CIS-scored output
│   └── parkiq.db                # Real-time SQLite database
│
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

## ⚠️ Honest Limitations
- No real congestion ground truth in dataset → explainable heuristic CIS (not black-box ML).
- YOLOv8 Nano has been optimized with lowered confidence thresholds and strict IoU tracking for small vehicles, but extreme occlusion in dense traffic may still cause ID swaps.
- ESP32 demo node = 1 zone proof-of-concept (production: LoRaWAN fleet).

<div align="center">

### From Parking Violations to Congestion Intelligence.

<br/>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:eab308,50:0e1620,100:05080d&height=120&section=footer" />

<sub>Built for GridLock Hackathon 2.0</sub>

</div>
