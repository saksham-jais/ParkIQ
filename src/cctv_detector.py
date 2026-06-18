"""
ParkIQ — CCTV + YOLOv8 Illegal Parking Detector
=================================================
Pipeline:
  CCTV / Webcam / Video file
       ↓
  YOLOv8 (vehicle detection)
       ↓
  Parking Zone Checker (are they in a no-parking polygon?)
       ↓
  Dwell Timer (how long have they been there?)
       ↓
  CIS Engine (Congestion Impact Score)
       ↓
  FastAPI POST → Dashboard

Run:
  python src/cctv_detector.py                     # use webcam (cam index 0)
  python src/cctv_detector.py --source video.mp4  # use a video file
  python src/cctv_detector.py --source 0 --show   # show live annotated window
"""

import cv2
import time
import argparse
import requests
import datetime
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SERVER_URL       = "http://127.0.0.1:8000/api/cctv-event"
# These will be overridden by argparse if provided
DEVICE_ID        = "CCTV-CAM-01"
CAMERA_LOCATION  = "MG Road Junction, Bengaluru"
CAMERA_LAT       = 12.9716
CAMERA_LON       = 77.5946

# ── Phone Cameras (IP Webcam app) ─────────────────────────────────────────────
# Install "IP Webcam" from Play Store, tap "Start Server"
# Replace the IPs below with what the app shows on your phone screens
PHONE_CAMERA_1_URL = "http://192.168.29.247:8080/video"   # ← CAM 1 IP
PHONE_CAMERA_2_URL = "http://192.168.29.41:8080/video"   # ← CAM 2 IP

MODEL_WEIGHTS    = "yolov8n.pt"          # auto-downloads on first run (~6MB)
CONF_THRESHOLD   = 0.35                  # slightly lower threshold for toy vehicles
VIOLATION_SEC    = 8                     # 8 seconds for demo (toy car stays still)
TELEMETRY_SEC    = 3                     # POST telemetry every 3 seconds
FRAME_SKIP       = 2                     # process every Nth frame (speed vs accuracy)

# COCO class IDs that are vehicles
VEHICLE_CLASSES  = {
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
}

# ROAD_CONFIG: Per-camera road boundary + no-parking zones
# road_width_px = pixel distance between LEFT and RIGHT edges of the road
#                 (calibrated by clicking 2 points on opposite road edges)
# zones         = no-parking polygons drawn INSIDE the road
ROAD_CONFIG = {
    "CCTV-CAM-01": {
        "road_width_px": 1500,  # ← calibrate with --calibrate
        "zones": {
            "Zone-1": np.array([[259, 435], [6, 699], [619, 767], [775, 296]], dtype=np.int32),
            "Zone-2": np.array([[1118, 191], [1374, 941], [1904, 705], [1511, 292]], dtype=np.int32),
        }
    },
    "CCTV-CAM-02": {
        "road_width_px": 1300,  # ← calibrate with --calibrate
        "zones": {
            "Zone-1": np.array([[650, 503], [391, 708], [757, 809], [946, 439]], dtype=np.int32),
            "Zone-2": np.array([[1174, 405], [1255, 772], [1667, 733], [1432, 446]], dtype=np.int32),
        }
    },
}
# Keep backward-compat alias
NO_PARKING_ZONES = {cam: cfg["zones"] for cam, cfg in ROAD_CONFIG.items()}

# ── Buzzer trigger via ESP32 backend ──────────────────────────────────────────
BUZZER_API_URL = "http://127.0.0.1:8000/api/sensor-event"  # same FastAPI server
SET_BUZZER_URL = "http://127.0.0.1:8000/api/set-buzzer"

def trigger_buzzer_alert(zone_id: str, priority: str):
    """Tell backend to trigger ESP32 hardware LEDs and buzzer."""
    import requests
    
    # Send the exact alert level to the backend API
    # Buzzer is ONLY active if priority is CRITICAL
    try:
        requests.post(SET_BUZZER_URL, json={
            "active": priority == "CRITICAL", 
            "zone_id": zone_id, 
            "level": priority
        }, timeout=0.5)
    except Exception:
        pass
    
    # 3. Update dashboard database
    try:
        requests.post(BUZZER_API_URL, json={
            "device_id": DEVICE_ID,
            "zone_id": zone_id,
            "event": "VIOLATION_CONFIRMED",
            "duration_sec": int(VIOLATION_SEC),
            "distance_cm": 0.0,
            "timestamp": int(time.time())
        }, timeout=1)
    except Exception:
        pass

def stop_buzzer_alert():
    """Tell backend to turn OFF the ESP32 buzzer when vehicle leaves."""
    import requests
    try:
        requests.post(SET_BUZZER_URL, json={"active": False, "zone_id": "", "level": "NONE"}, timeout=0.5)
    except Exception:
        pass

# ─────────────────────────────────────────────
# CIS ENGINE
# ─────────────────────────────────────────────
def compute_cis(vehicle_type: str, dwell_seconds: float, zone_id: str,
                bbox_width: int = 100, frame_width: int = 640) -> int:
    """Compute Congestion Impact Score (0-100).

    CIS measures the CURRENT congestion threat — it does NOT increase
    over time. A car at 6 AM has the same low impact whether it has
    been parked 10 seconds or 10 minutes.
    """
    import datetime
    current_hour = datetime.datetime.now().hour

    # 1. Base: how physically obstructing is the vehicle?
    base = {"car": 40, "motorcycle": 20, "bus": 70, "truck": 80}.get(vehicle_type, 30)

    # 2. Zone penalty: parked inside a designated no-parking zone?
    zone_penalty = 20 if zone_id else 0

    # 3. Lane blockage: vehicle width vs calibrated real road width
    road_width = ROAD_CONFIG.get(DEVICE_ID, {}).get("road_width_px", frame_width)
    blockage_ratio = min(bbox_width / max(road_width, 1), 1.0)
    lane_penalty = int(blockage_ratio * 20)  # max +20 pts

    # 4. Time-of-day multiplier (rush hour amplifies impact)
    if current_hour in [8, 9, 10, 17, 18, 19]:
        time_multiplier = 1.3   # Rush hour
    elif current_hour >= 22 or current_hour <= 6:
        time_multiplier = 0.5   # Night — low traffic impact
    else:
        time_multiplier = 1.0   # Normal daytime

    final_cis = int((base + zone_penalty + lane_penalty) * time_multiplier)
    return min(final_cis, 100)


def priority_label(cis: int) -> str:
    if cis > 70: return "HIGH"
    if cis > 40: return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────
# ZONE CHECKER
# ─────────────────────────────────────────────
def bbox_center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def point_in_zone(cx, cy, zone_poly) -> bool:
    return cv2.pointPolygonTest(zone_poly, (cx, cy), False) >= 0


def get_vehicle_zone(cx, cy) -> str | None:
    zones = NO_PARKING_ZONES.get(DEVICE_ID, {})
    for zone_id, poly in zones.items():
        if point_in_zone(cx, cy, poly):
            return zone_id
    return None


# ─────────────────────────────────────────────
# API SENDER
# ─────────────────────────────────────────────
def send_event(track_id, vehicle_type, zone_id, event_type, dwell_sec, cis, priority):
    payload = {
        "device_id":      DEVICE_ID,
        "track_id":       int(track_id),
        "vehicle_type":   vehicle_type,
        "zone_id":        zone_id,
        "event":          event_type,
        "duration_sec":   int(dwell_sec),
        "cis":            cis,
        "priority":       priority,
        "camera_location": CAMERA_LOCATION,
        "latitude":       CAMERA_LAT,
        "longitude":      CAMERA_LON,
        "timestamp":      int(time.time()),
    }
    try:
        r = requests.post(SERVER_URL, json=payload, timeout=2)
        print(f"[{event_type}] Track#{track_id} {vehicle_type} Zone:{zone_id} CIS:{cis} → HTTP {r.status_code}")
    except Exception as e:
        print(f"[WARN] Could not reach API: {e}")


# ─────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────
COLOUR_MAP = {
    "LOW":    (0, 255, 0),      # green
    "MEDIUM": (0, 165, 255),    # orange
    "HIGH":   (0, 0, 255),      # red
}

def draw_zones(frame):
    overlay = frame.copy()
    cam_cfg = ROAD_CONFIG.get(DEVICE_ID, {})
    road_w  = cam_cfg.get("road_width_px", 0)
    zones   = cam_cfg.get("zones", {})
    
    # Draw road width indicator at top of frame
    if road_w > 0:
        fh, fw = frame.shape[:2]
        mid_x = fw // 2
        half  = road_w // 2
        lx, rx = max(0, mid_x - half), min(fw, mid_x + half)
        cv2.line(frame, (lx, 18), (rx, 18), (0, 200, 255), 2)
        cv2.arrowedLine(frame, (mid_x, 18), (lx, 18), (0, 200, 255), 1, tipLength=0.05)
        cv2.arrowedLine(frame, (mid_x, 18), (rx, 18), (0, 200, 255), 1, tipLength=0.05)
        cv2.putText(frame, f"Road: {road_w}px", (lx + 4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    
    # Draw no-parking zones
    for zone_id, poly in zones.items():
        cv2.fillPoly(overlay, [poly], (0, 0, 180))
        cv2.polylines(frame, [poly], True, (0, 0, 255), 2)
        cx = int(poly[:, 0].mean())
        cy = int(poly[:, 1].mean())
        cv2.putText(frame, f"NO PARK: {zone_id}", (cx - 50, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)


def draw_vehicle(frame, x1, y1, x2, y2, label, colour):
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ─────────────────────────────────────────────
# MAIN DETECTOR
# ─────────────────────────────────────────────
def run(source=0, show=True):
    import os
    os.makedirs("data", exist_ok=True)
    print(f"[ParkIQ] Loading YOLOv8 model: {MODEL_WEIGHTS}")
    model = YOLO(MODEL_WEIGHTS)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    # Per-track state
    track_entry_time   = {}   # track_id → timestamp when first entered zone
    track_zone         = {}   # track_id → zone_id
    track_vehicle_type = {}   # track_id → vehicle type string
    track_violation_sent = set()   # track_ids for which VIOLATION was already sent
    last_telemetry     = defaultdict(float)  # track_id → last telemetry time

    frame_idx = 0
    print(f"[ParkIQ] Starting detection on source: {source}")
    print(f"[ParkIQ] Violation threshold: {VIOLATION_SEC}s | Server: {SERVER_URL}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ParkIQ] Stream ended.")
            break

        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue

        # ── YOLOv8 Tracking ──
        results = model.track(
            frame,
            persist=True,
            conf=CONF_THRESHOLD,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False,
        )

        if show:
            draw_zones(frame)

        now = time.time()
        active_track_ids = set()

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes   = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = box
                vehicle_type = VEHICLE_CLASSES.get(cls_id, "vehicle")
                
                # ── HACKATHON DEMO OVERRIDE ──
                # YOLOv8 Nano struggles with tiny toys viewed from the front and often 
                # misclassifies them as trucks. We use the bounding box width to fix this!
                # If it's a small box (width < 200 pixels), it's one of the toy cars.
                # If it's a large box (width > 200 pixels), it's the actual blue truck.
                w = x2 - x1
                if vehicle_type == "truck" and w < 200:
                    vehicle_type = "car"
                
                cx, cy = bbox_center(x1, y1, x2, y2)
                zone_id = get_vehicle_zone(cx, cy)

                active_track_ids.add(track_id)
                track_vehicle_type[track_id] = vehicle_type

                if zone_id:
                    # First time entering this zone
                    if track_id not in track_entry_time or track_zone.get(track_id) != zone_id:
                        track_entry_time[track_id] = now
                        track_zone[track_id] = zone_id
                        track_violation_sent.discard(track_id)
                        send_event(track_id, vehicle_type, zone_id, "OCCUPANCY_START", 0, 0, "LOW")

                    dwell_sec = now - track_entry_time[track_id]
                    # Pass bbox width and frame width for lane blockage calculation
                    frame_w = frame.shape[1]
                    cis = compute_cis(vehicle_type, dwell_sec, zone_id,
                                      bbox_width=(x2 - x1), frame_width=frame_w)
                    priority = priority_label(cis)

                    # Violation threshold crossed
                    if dwell_sec >= VIOLATION_SEC and track_id not in track_violation_sent:
                        track_violation_sent.add(track_id)
                        send_event(track_id, vehicle_type, zone_id,
                                   "VIOLATION_CONFIRMED", dwell_sec, cis, priority)
                        trigger_buzzer_alert(zone_id, priority)

                    # Periodic telemetry
                    if now - last_telemetry[track_id] > TELEMETRY_SEC:
                        send_event(track_id, vehicle_type, zone_id,
                                   "TELEMETRY_UPDATE", dwell_sec, cis, priority)
                        last_telemetry[track_id] = now

                    if show:
                        colour = COLOUR_MAP[priority]
                        label = f"#{track_id} {vehicle_type} | {zone_id} | CIS:{cis} [{priority}] {int(dwell_sec)}s"
                        draw_vehicle(frame, x1, y1, x2, y2, label, colour)
                else:
                    # Vehicle left zone — send VACATED
                    if track_id in track_zone:
                        dwell_sec = now - track_entry_time.get(track_id, now)
                        cis = compute_cis(vehicle_type, dwell_sec, track_zone[track_id])
                        send_event(track_id, vehicle_type, track_zone[track_id],
                                   "VACATED", dwell_sec, cis, priority_label(cis))
                        if track_id in track_violation_sent:
                            stop_buzzer_alert()
                        del track_zone[track_id]
                        del track_entry_time[track_id]

                    if show:
                        label = f"#{track_id} {vehicle_type}"
                        draw_vehicle(frame, x1, y1, x2, y2, label, (180, 180, 180))

        # Write the fully-annotated frame once per processed loop tick.
        # Skipped frames are not written — last good frame stays on disk.
        cv2.imwrite(f"data/current_frame_{DEVICE_ID}.jpg", frame)

        # Clean up lost tracks
        lost = set(track_zone.keys()) - active_track_ids
        for tid in lost:
            if tid in track_entry_time:
                dwell_sec = now - track_entry_time[tid]
                vt = track_vehicle_type.get(tid, "vehicle")
                cis = compute_cis(vt, dwell_sec, track_zone[tid])
                send_event(tid, vt, track_zone[tid], "VACATED", dwell_sec, cis, priority_label(cis))
                if tid in track_violation_sent:
                    stop_buzzer_alert()
                del track_zone[tid]
                del track_entry_time[tid]

        if show:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, f"ParkIQ CCTV | {ts} | Active: {len(active_track_ids)}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("ParkIQ — Illegal Parking Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[ParkIQ] Quit by user.")
                break

    cap.release()
    if show:
        cv2.destroyAllWindows()
    print("[ParkIQ] Detector stopped.")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ParkIQ CCTV Detector")
    parser.add_argument("--cam", type=int, choices=[1, 2], default=1,
                        help="Which predefined camera to use (1 or 2)")
    parser.add_argument("--source", default=None,
                        help="Override video source (0=webcam, or custom IP URL)")
    parser.add_argument("--show", action="store_true", default=True,
                        help="Show annotated live window")
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.add_argument("--calibrate", action="store_true",
                        help="Open calibration mode to click your no-parking zone corners")
    parser.add_argument("--device-id", default="CCTV-CAM-01",
                        help="Override the camera device ID for multi-camera setups.")
    args = parser.parse_args()

    DEVICE_ID = args.device_id
    
    # Determine the source
    if args.source is not None:
        src_raw = args.source
    else:
        src_raw = PHONE_CAMERA_1_URL if args.cam == 1 else PHONE_CAMERA_2_URL
        
    src = int(src_raw) if str(src_raw).isdigit() else src_raw

    if args.calibrate:
        print(f"[CALIBRATE] Opening camera feed: {src}")
        print("STEP 1: Click 2 points on the LEFT and RIGHT edges of the full road.")
        print("STEP 2: Click 4 corners for each No-Parking Zone. Press ENTER when done.")
        cap = cv2.VideoCapture(src)
        ret, frame = cap.read()
        cap.release()
        if ret:
            state   = {"phase": 1, "road_pts": [], "zone_pts": []}
            clone   = frame.copy()

            def click(event, x, y, flags, param):
                if event != cv2.EVENT_LBUTTONDOWN:
                    return
                if state["phase"] == 1:
                    state["road_pts"].append([x, y])
                    cv2.circle(clone, (x, y), 6, (0, 200, 255), -1)
                    if len(state["road_pts"]) == 1:
                        cv2.putText(clone, "Now click RIGHT edge of road",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
                    elif len(state["road_pts"]) == 2:
                        p1, p2 = state["road_pts"]
                        cv2.line(clone, tuple(p1), tuple(p2), (0, 200, 255), 2)
                        road_w = abs(p2[0] - p1[0])
                        cv2.putText(clone, f"Road width: {road_w}px — Now draw No-Park zones (4 pts each)",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        state["phase"] = 2
                elif state["phase"] == 2:
                    pts = state["zone_pts"]
                    pts.append([x, y])
                    cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
                    if len(pts) % 4 != 1:
                        cv2.line(clone, tuple(pts[-2]), tuple(pts[-1]), (0,255,0), 2)
                    if len(pts) % 4 == 0:
                        cv2.line(clone, tuple(pts[-1]), tuple(pts[-4]), (0,255,0), 2)
                cv2.imshow("Calibrate", clone)

            cv2.putText(clone, "STEP 1: Click LEFT edge of road",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
            cv2.imshow("Calibrate", clone)
            cv2.setMouseCallback("Calibrate", click)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            road_pts  = state["road_pts"]
            zone_pts  = state["zone_pts"]
            road_w    = abs(road_pts[1][0] - road_pts[0][0]) if len(road_pts) == 2 else 640

            if len(zone_pts) > 0 and len(zone_pts) % 4 == 0:
                print(f"\n\u2705 Paste this into ROAD_CONFIG in cctv_detector.py:")
                print(f'    "{DEVICE_ID}": {{')
                print(f'        "road_width_px": {road_w},  # calibrated road width')
                print(f'        "zones": {{')
                for i in range(0, len(zone_pts), 4):
                    zp = zone_pts[i:i+4]
                    print(f'            "Zone-{i//4 + 1}": np.array({zp}, dtype=np.int32),')
                print(f'        }}')
                print(f'    }},')
            else:
                print(f"\n\u274c Error: Got {len(zone_pts)} zone points. You must click exactly 4 points per zone.")
    else:
        run(source=src, show=args.show)
