from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sqlite3
import os

app = FastAPI(title="ParkIQ API")

# Allow Streamlit (and any local origin) to call the API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data", exist_ok=True)
DB_FILE = "data/parkiq.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    # IoT HC-SR04 events
    conn.execute(
        """CREATE TABLE IF NOT EXISTS sensor_events (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id     TEXT,
            zone_id       TEXT,
            event         TEXT,
            duration_sec  INTEGER,
            distance_cm   REAL,
            timestamp     INTEGER
        )"""
    )
    # CCTV / YOLOv8 events
    conn.execute(
        """CREATE TABLE IF NOT EXISTS cctv_events (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id        TEXT,
            track_id         INTEGER,
            vehicle_type     TEXT,
            zone_id          TEXT,
            event            TEXT,
            duration_sec     INTEGER,
            cis              INTEGER,
            priority         TEXT,
            camera_location  TEXT,
            latitude         REAL,
            longitude        REAL,
            timestamp        INTEGER
        )"""
    )
    conn.commit()
    conn.close()

init_db()

# ── IoT Sensor Model ──────────────────────────────────────────────────────────
class SensorEvent(BaseModel):
    device_id: str
    zone_id: str
    event: str
    duration_sec: int
    distance_cm: float
    timestamp: int

@app.post("/api/sensor-event")
def receive_sensor_event(payload: SensorEvent):
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        """INSERT INTO sensor_events
           (device_id, zone_id, event, duration_sec, distance_cm, timestamp)
           VALUES (?,?,?,?,?,?)""",
        (payload.device_id, payload.zone_id, payload.event,
         payload.duration_sec, payload.distance_cm, payload.timestamp),
    )
    conn.commit()
    conn.close()
    print(f"[IoT]  {payload.event} | Zone:{payload.zone_id}")
    return {"status": "received"}

@app.get("/api/sensor-events/recent")
def get_recent_sensor_events(limit: int = 50):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM sensor_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/api/sensor-events/stats")
def get_sensor_stats():
    conn = sqlite3.connect(DB_FILE)
    stats = {}
    total_events = conn.execute("SELECT COUNT(*) FROM sensor_events").fetchone()[0]
    violations = conn.execute("SELECT COUNT(*) FROM sensor_events WHERE event='VIOLATION_CONFIRMED'").fetchone()[0]
    live_nodes_count = conn.execute("SELECT COUNT(DISTINCT device_id) FROM sensor_events WHERE timestamp > strftime('%s','now') - 60").fetchone()[0]
    
    stats["total_events"] = total_events
    stats["violations"] = violations
    stats["live_nodes_count"] = live_nodes_count
    conn.close()
    return stats

# ── CCTV / YOLOv8 Model ───────────────────────────────────────────────────────
class CCTVEvent(BaseModel):
    device_id: str
    track_id: int
    vehicle_type: str
    zone_id: str
    event: str
    duration_sec: int
    cis: int
    priority: str
    camera_location: Optional[str] = ""
    latitude: Optional[float] = 0.0
    longitude: Optional[float] = 0.0
    timestamp: int

@app.post("/api/cctv-event")
def receive_cctv_event(payload: CCTVEvent):
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        """INSERT INTO cctv_events
           (device_id, track_id, vehicle_type, zone_id, event,
            duration_sec, cis, priority, camera_location, latitude, longitude, timestamp)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (payload.device_id, payload.track_id, payload.vehicle_type,
         payload.zone_id, payload.event, payload.duration_sec,
         payload.cis, payload.priority, payload.camera_location,
         payload.latitude, payload.longitude, payload.timestamp),
    )
    conn.commit()
    conn.close()
    print(f"[CCTV] {payload.event} | Track#{payload.track_id} {payload.vehicle_type} "
          f"Zone:{payload.zone_id} CIS:{payload.cis} [{payload.priority}]")
    return {"status": "received"}

@app.get("/api/cctv-events/recent")
def get_recent_cctv_events(limit: int = 50):
    """Dashboard polling endpoint — returns latest CCTV detections."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM cctv_events ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/api/cctv-events/stats")
def get_cctv_stats():
    """Aggregated violation stats for the dashboard — LIVE only (last 90s, non-vacated tracks)."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    stats = {}

    LIVE_WINDOW = 7  # seconds — telemetry fires every 3s, so 7s is enough to drop dead tracks quickly

    # Fetch the latest event for each track that was active in the LIVE_WINDOW
    live_rows = conn.execute(
        """
        SELECT track_id, event, priority, zone_id, duration_sec
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER(PARTITION BY track_id ORDER BY timestamp DESC, id DESC) as rn
            FROM cctv_events
        )
        WHERE rn = 1
          AND timestamp > strftime('%s','now') - ?
          AND event != 'VACATED'
        """,
        (LIVE_WINDOW,)
    ).fetchall()

    active_vehicles = len(live_rows)
    # VIOLATION_SEC is 8 seconds in the detector
    live_violations = sum(1 for r in live_rows if r['duration_sec'] >= 8)
    live_critical   = sum(1 for r in live_rows if r['duration_sec'] >= 8 and r['priority'] == "CRITICAL")
    live_high       = sum(1 for r in live_rows if r['duration_sec'] >= 8 and r['priority'] == "HIGH")
    live_medium     = sum(1 for r in live_rows if r['duration_sec'] >= 8 and r['priority'] == "MEDIUM")
    live_low        = sum(1 for r in live_rows if r['duration_sec'] >= 8 and r['priority'] == "LOW")

    stats["active_vehicles"]  = active_vehicles
    stats["total_violations"] = live_violations
    stats["critical_priority"]= live_critical
    stats["high_priority"]    = live_high
    stats["medium_priority"]  = live_medium
    stats["low_priority"]     = live_low

    # By-zone live count
    zone_counts = {}
    for r in live_rows:
        if r['duration_sec'] >= 8 and r['zone_id']:
            zone_counts[r['zone_id']] = zone_counts.get(r['zone_id'], 0) + 1
    stats["by_zone"] = [{"zone_id": z, "count": c} for z, c in sorted(zone_counts.items(), key=lambda x: -x[1])]

    # All-time historical totals kept for trend/reference
    stats["all_time_violations"] = conn.execute(
        "SELECT COUNT(*) FROM cctv_events WHERE event='VIOLATION_CONFIRMED'"
    ).fetchone()[0]
    avg_cis_row = conn.execute(
        "SELECT AVG(cis) FROM cctv_events WHERE event='VIOLATION_CONFIRMED'"
    ).fetchone()[0]
    stats["avg_cis"] = round(avg_cis_row, 1) if avg_cis_row else 0
    top_vehicle_row = conn.execute(
        "SELECT vehicle_type, COUNT(*) as cnt FROM cctv_events "
        "WHERE event='VIOLATION_CONFIRMED' GROUP BY vehicle_type ORDER BY cnt DESC LIMIT 1"
    ).fetchone()
    stats["top_vehicle"] = top_vehicle_row[0] if top_vehicle_row else "N/A"
    avg_dur_row = conn.execute(
        "SELECT AVG(duration_sec) FROM cctv_events WHERE event='VIOLATION_CONFIRMED'"
    ).fetchone()[0]
    stats["avg_duration_sec"] = int(avg_dur_row) if avg_dur_row else 0

    conn.close()
    return stats



@app.get("/api/cctv-events/high-risk")
def get_high_risk_zones():
    """Returns zones with the most VIOLATION_CONFIRMED events, with avg CIS."""
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute(
        """SELECT zone_id,
                  COUNT(*) as violation_count,
                  AVG(cis)  as avg_cis,
                  MAX(cis)  as max_cis,
                  AVG(duration_sec) as avg_duration
           FROM cctv_events
           WHERE event='VIOLATION_CONFIRMED'
           GROUP BY zone_id
           ORDER BY avg_cis DESC"""
    ).fetchall()
    conn.close()
    return [
        {
            "zone_id":        r[0],
            "violation_count": r[1],
            "avg_cis":        round(r[2], 1) if r[2] else 0,
            "max_cis":        r[3] or 0,
            "avg_duration":   int(r[4]) if r[4] else 0,
        }
        for r in rows
    ]

# ── Video Streaming Endpoint ──────────────────────────────────────────────────
import time
from fastapi.responses import StreamingResponse

# 1×1 black JPEG — served until the detector writes the first real frame
_BLANK_JPEG = (
    b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
    b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t'
    b'\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a'
    b'\x1f\x1e\x1d\x1a\x1c\x1c $.\'"\x1c\x1c!,# \x1c\x1c\x1e%.\'"\xd5'
    b'\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00'
    b'\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00'
    b'\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00'
    b'\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q\x142'
    b'\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18'
    b'\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84'
    b'\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a'
    b'\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7'
    b'\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4'
    b'\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9'
    b'\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08'
    b'\x01\x01\x00\x00?\x00\xfb\xd4\xff\xd9'
)


def get_offline_jpeg():
    import cv2, numpy as np
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (17, 17, 17)  # #111 hex
    cv2.putText(img, "Camera Offline / Not Available", (110, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (136, 136, 136), 2, cv2.LINE_AA)
    ret, jpeg = cv2.imencode('.jpg', img)
    return jpeg.tobytes()

_OFFLINE_JPEG = get_offline_jpeg()

def generate_mjpeg_frames(cam_id: str):
    """Yield MJPEG frames indefinitely. If the camera drops, yield an offline image."""
    frame_path = f"data/current_frame_{cam_id}.jpg"
    while True:
        try:
            if not os.path.exists(frame_path) or (time.time() - os.path.getmtime(frame_path) > 5):
                # Camera is offline, stream the offline placeholder
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + _OFFLINE_JPEG + b'\r\n'
                )
                time.sleep(1.0) # Check less frequently if offline
                continue

            with open(frame_path, "rb") as f:
                data = f.read()
            
            if data:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n'
                )
        except Exception:
            pass
            
        time.sleep(0.05)     # ~20 FPS cap

@app.get("/api/video_feed")
def video_feed(cam_id: str = "CCTV-CAM-01"):
    return StreamingResponse(generate_mjpeg_frames(cam_id), media_type="multipart/x-mixed-replace; boundary=frame")


# ── Device State for ESP32 Mode Switching ─────────────────────────────────────
# Global state to share between Dashboard, CCTV, and ESP32
class AppState:
    mode = "IOT"  # "IOT" or "CCTV"
    cctv_buzzer_active = False
    cctv_zone_id = ""
    cctv_alert_level = "VACANT"

class ModeUpdate(BaseModel):
    mode: str

class BuzzerUpdate(BaseModel):
    active: bool
    zone_id: str = ""
    level: str = "VACANT"

@app.get("/api/device-state")
def get_device_state():
    if AppState.mode == "CCTV":
        stats = get_cctv_stats()
        if stats.get("critical_priority", 0) > 0:
            AppState.cctv_alert_level = "CRITICAL"
            AppState.cctv_buzzer_active = True
        elif stats.get("high_priority", 0) > 0:
            AppState.cctv_alert_level = "HIGH"
            AppState.cctv_buzzer_active = False
        elif stats.get("medium_priority", 0) > 0:
            AppState.cctv_alert_level = "MEDIUM"
            AppState.cctv_buzzer_active = False
        elif stats.get("low_priority", 0) > 0:
            AppState.cctv_alert_level = "LOW"
            AppState.cctv_buzzer_active = False
        else:
            AppState.cctv_alert_level = "VACANT"
            AppState.cctv_buzzer_active = False

    return {
        "mode": AppState.mode,
        "buzzer": AppState.cctv_buzzer_active,
        "zone_id": AppState.cctv_zone_id,
        "alert_level": AppState.cctv_alert_level
    }

@app.post("/api/set-mode")
def set_mode(payload: ModeUpdate):
    AppState.mode = payload.mode
    # Reset buzzer when switching modes
    AppState.cctv_buzzer_active = False
    print(f"[SYSTEM] Mode switched to: {AppState.mode}")
    return {"status": "success", "mode": AppState.mode}

@app.post("/api/set-buzzer")
def set_buzzer(payload: BuzzerUpdate):
    AppState.cctv_buzzer_active = payload.active
    AppState.cctv_zone_id = payload.zone_id
    AppState.cctv_alert_level = payload.level
    print(f"[SYSTEM] Hardware Trigger -> LED: {payload.level} | Buzzer: {payload.active}")
    return {"status": "success"}

@app.post("/api/save-calibration")
async def save_calibration(request: Request):
    """Save zone calibration JSON from the browser canvas tool."""
    import json
    try:
        data = await request.json()
        os.makedirs("data", exist_ok=True)
        existing = {}
        if os.path.exists("data/calibration.json"):
            try:
                with open("data/calibration.json") as f:
                    existing = json.load(f)
            except Exception:
                pass
        cam_id      = data.get("cam_id", "CCTV-CAM-01")
        road_pts    = data.get("road_pts", [])
        saved_zones = data.get("saved_zones", [])
        if not saved_zones or len(road_pts) != 2:
            return JSONResponse({"status": "error", "msg": "Invalid data"}, status_code=400)
        existing[cam_id] = {"zones": {}}
        for i, z in enumerate(saved_zones):
            existing[cam_id]["zones"][f"Zone-{i+1}"] = {
                "road_width_px": z["road_width_px"],
                "polygon": z["polygon"]
            }
        with open("data/calibration.json", "w") as f:
            json.dump(existing, f, indent=2)
        return {"status": "saved", "cam": cam_id, "zones": len(saved_zones)}
    except Exception as e:
        return JSONResponse({"status": "error", "msg": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_api:app", host="0.0.0.0", port=8000, reload=True)
