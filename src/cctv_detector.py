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
import platform
IS_LOCAL = platform.system() == "Windows"
BASE_API_URL = "http://127.0.0.1:8000" if IS_LOCAL else "https://parkiq-glrk.onrender.com"
# These will be overridden by argparse if provided
DEVICE_ID        = "CCTV-CAM-01"
CAMERA_LOCATION  = "MG Road Junction, Bengaluru"
CAMERA_LAT       = 12.9716
CAMERA_LON       = 77.5946

# ── Phone Cameras (IP Webcam app) ─────────────────────────────────────────────
# Install "IP Webcam" from Play Store, tap "Start Server"
# Replace the IPs below with what the app shows on your phone screens
PHONE_CAMERA_1_URL = "http://192.168.29.247:8080/video"   # ← CAM 1 IP
PHONE_CAMERA_2_URL = "http://192.168.29.131:8080/video"   # ← CAM 2 IP

MODEL_WEIGHTS    = "yolov8n.pt"          # auto-downloads on first run (~6MB)
CONF_THRESHOLD   = 0.15                  # lowered to catch blurry bikes/scooters
VIOLATION_SEC    = 8                     # 8 seconds for demo (toy car stays still)
TELEMETRY_SEC    = 3                     # POST telemetry every 3 seconds
FRAME_SKIP       = 2                     # process every Nth frame (speed vs accuracy)

# COCO class IDs that are vehicles
VEHICLE_CLASSES  = {
    1:  "bicycle",
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
        "zones": {
            "Zone-1": {
                "road_width_px": 748,
                "polygon": np.array([[1179, 174], [1050, 171], [1291, 562], [1560, 489]], dtype=np.int32)      
            },
        }
    },
    "CCTV-CAM-02": {
        "type": "school",      # e.g., School / Residential area
        "zones": {
            "Zone-1": {
                "road_width_px": 908,
                "polygon": np.array([[189, 737], [675, 767], [754, 406], [427, 473]], dtype=np.int32)
            },
            "Zone-2": {
                "road_width_px": 908,
                "polygon": np.array([[1000, 348], [1273, 391], [1677, 747], [1211, 819]], dtype=np.int32)
            },
        }
    },
}
# ── Override with user calibration if exists ──
import os, json
if os.path.exists("data/calibration.json"):
    try:
        with open("data/calibration.json", "r") as f:
            calib = json.load(f)
            for cam, data in calib.items():
                if cam not in ROAD_CONFIG:
                    ROAD_CONFIG[cam] = {"zones": {}}
                else:
                    ROAD_CONFIG[cam]["zones"] = {}
                for z_id, z_data in data.get("zones", {}).items():
                    ROAD_CONFIG[cam]["zones"][z_id] = {
                        "road_width_px": z_data["road_width_px"],
                        "polygon": np.array(z_data["polygon"], dtype=np.int32)
                    }
    except Exception as e:
        print(f"[WARN] Failed to load calibration.json: {e}")

# Keep backward-compat alias
NO_PARKING_ZONES = {}
for cam, cfg in ROAD_CONFIG.items():
    NO_PARKING_ZONES[cam] = {}
    for z_id, z_data in cfg.get("zones", {}).items():
        NO_PARKING_ZONES[cam][z_id] = z_data["polygon"] if isinstance(z_data, dict) else z_data

def trigger_buzzer_alert(zone_id: str, priority: str):
    """Tell backend to trigger ESP32 hardware LEDs and buzzer."""
    import requests
    
    # Send the exact alert level to the backend API
    # Buzzer is ONLY active if priority is CRITICAL
    try:
        r = requests.post(f"{BASE_API_URL}/api/set-buzzer", json={
            "active": priority == "CRITICAL", 
            "zone_id": zone_id, 
            "level": priority
        }, timeout=2.0)
        print(f"[LED] Triggered {priority} -> HTTP {r.status_code}")
    except Exception as e:
        print(f"[LED] Failed to trigger {priority}: {e}")
    
    # 3. Update dashboard database
    try:
        requests.post(f"{BASE_API_URL}/api/sensor-event", json={
            "device_id": DEVICE_ID,
            "zone_id": zone_id,
            "event": "VIOLATION_CONFIRMED",
            "duration_sec": int(VIOLATION_SEC),
            "distance_cm": 0.0,
            "timestamp": int(time.time())
        }, timeout=2.0)
    except Exception:
        pass

def update_hardware_leds(active_track_ids, track_entry_time, track_vehicle_type, track_zone, track_violation_sent, ghost_tracks=None):
    """Evaluate all current active violations and update the ESP32 LEDs to the highest priority, or VACANT if none."""
    import requests
    highest_priority = "VACANT"
    if ghost_tracks is None: ghost_tracks = {}
    
    # If there are any violating tracks, calculate the highest priority among them
    if track_violation_sent:
        import time
        now = time.time()
        for tid in track_violation_sent:
            p = "LOW"
            if tid in active_track_ids and tid in track_entry_time:
                vt = track_vehicle_type.get(tid, "vehicle")
                dwell_sec = now - track_entry_time[tid]
                cis = compute_cis(vt, dwell_sec, track_zone[tid])
                p = priority_label(cis)
            elif tid in ghost_tracks:
                vt = ghost_tracks[tid]["vtype"]
                dwell_sec = now - ghost_tracks[tid]["entry_time"]
                cis = compute_cis(vt, dwell_sec, ghost_tracks[tid]["zone_id"])
                p = priority_label(cis)
            else:
                continue

            if p == "CRITICAL": highest_priority = "CRITICAL"
            elif p == "HIGH" and highest_priority != "CRITICAL": highest_priority = "HIGH"
            elif p == "MEDIUM" and highest_priority not in ["CRITICAL", "HIGH"]: highest_priority = "MEDIUM"
            elif p == "LOW" and highest_priority == "VACANT": highest_priority = "LOW"
                
    try:
        r = requests.post(f"{BASE_API_URL}/api/set-buzzer", json={"active": highest_priority == "CRITICAL", "zone_id": "", "level": highest_priority}, timeout=2.0)
        print(f"[LED] Global State updated to {highest_priority} -> HTTP {r.status_code}")
    except Exception as e:
        print(f"[LED] Failed to update global state to {highest_priority}: {e}")

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
    cam_cfg = ROAD_CONFIG.get(DEVICE_ID, {})
    if zone_id and zone_id in cam_cfg.get("zones", {}):
        z_data = cam_cfg["zones"][zone_id]
        road_width = z_data.get("road_width_px", frame_width) if isinstance(z_data, dict) else cam_cfg.get("road_width_px", frame_width)
    else:
        road_width = cam_cfg.get("road_width_px", frame_width)

    blockage_ratio = min(bbox_width / max(road_width, 1), 1.0)
    lane_penalty = int(blockage_ratio * 20)  # max +20 pts

    # 4. Time-of-day multiplier (location-aware rush hour amplifies impact)
    cam_type = cam_cfg.get("type", "commercial")
    time_multiplier = 1.0   # Normal daytime default

    if cam_type == "commercial":
        # Rush Hour: 8-10 AM, 5-8 PM
        if current_hour in [8, 9, 10, 17, 18, 19]:
            time_multiplier = 1.3
        # Off-peak daytime: 11 AM - 4 PM
        elif current_hour in [11, 12, 13, 14, 15, 16]:
            time_multiplier = 0.5
        # Night time: 8 PM - 7 AM
        else:
            time_multiplier = 0.3
    elif cam_type == "school":
        # School Rush: 7-9 AM, 2-4 PM
        if current_hour in [7, 8, 14, 15]:
            time_multiplier = 1.4
        # Off-peak daytime
        elif current_hour in [9, 10, 11, 12, 13]:
            time_multiplier = 0.5
        # Evening/Night
        else:
            time_multiplier = 0.3
            
    final_cis = int((base + zone_penalty + lane_penalty) * time_multiplier)
    return min(final_cis, 100)


def priority_label(cis: int) -> str:
    if cis >= 80: return "CRITICAL"   # Red  — severe obstruction
    if cis >= 55: return "HIGH"       # Red  — significant obstruction
    if cis >= 35: return "MEDIUM"     # Yellow — moderate risk
    return "LOW"                      # Green — minimal risk


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
        r = requests.post(f"{BASE_API_URL}/api/cctv-event", json=payload, timeout=2)
        print(f"[{event_type}] Track#{track_id} {vehicle_type} Zone:{zone_id} CIS:{cis} → HTTP {r.status_code}")
    except Exception as e:
        print(f"[WARN] Could not reach API: {e}")


# ─────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────
COLOUR_MAP = {
    "LOW":      (50, 220, 50),    # green  — minimal risk
    "MEDIUM":   (0, 230, 230),   # yellow — moderate risk  (BGR: yellow = 0,255,255)
    "HIGH":     (0, 80, 255),    # red-orange — significant
    "CRITICAL": (0, 0, 230),     # bright red — severe
}

def draw_zones(frame):
    overlay = frame.copy()
    cam_cfg = ROAD_CONFIG.get(DEVICE_ID, {})
    zones   = cam_cfg.get("zones", {})
    
    # Draw no-parking zones and road width indicators
    for zone_id, z_data in zones.items():
        if isinstance(z_data, dict):
            poly = z_data["polygon"]
            road_w = z_data.get("road_width_px", 0)
        else:
            poly = z_data
            road_w = cam_cfg.get("road_width_px", 0)
            
        cv2.fillPoly(overlay, [poly], (0, 0, 180))
        cv2.polylines(frame, [poly], True, (0, 0, 255), 2)
        cx = int(poly[:, 0].mean())
        cy = int(poly[:, 1].mean())
        cv2.putText(frame, f"NO PARK: {zone_id} (Lane Width: {road_w}px)", (cx - 50, cy),
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
    track_entry_time     = {}   # track_id → timestamp when first entered zone
    track_zone           = {}   # track_id → zone_id
    track_vehicle_type   = {}   # track_id → LOCKED vehicle type string
    track_type_votes     = defaultdict(lambda: defaultdict(int))  # track_id → {type: count}
    track_type_locked    = set()  # track_ids whose type is locked and won't flip
    track_violation_sent = set()  # track_ids for which VIOLATION was already sent
    track_occupancy_sent = set()  # track_ids for which OCCUPANCY_START was sent
    last_telemetry       = defaultdict(float)  # track_id → last telemetry time
    pending_exit         = defaultdict(int)    # track_id → consecutive out-of-zone frame count

    EXIT_GRACE_FRAMES = 15  # must be outside zone for this many frames to confirm exit (prevents jitter resets)
    STATIONARY_SEC    = 5.0 # ignore passing cars until they dwell for this long

    # Ghost-track resurrection: when tracker drops a stationary car for a frame
    # and re-detects it with a NEW id, we match by position and inherit state.
    ghost_tracks         = {}   # track_id → {cx, cy, box, zone_id, entry_time, violation_sent, vtype, locked}
    track_last_pos       = {}   # track_id → (cx, cy) — updated every frame for ghost position
    track_last_box       = {}   # track_id → [x1, y1, x2, y2]
    GHOST_TTL            = 3.0  # seconds to keep ghost before truly considering it gone (handles short occlusions)
    GHOST_DIST_PX        = 300  # max centroid distance to match a ghost (handles large centroid shifts for buses)

    VOTE_WINDOW  = 10   # keep last N votes per track
    LOCK_VOTES   = 6    # need this many votes for one class to lock

    frame_idx = 0
    last_calib_mtime = 0
    print(f"[ParkIQ] Starting detection on source: {source}")
    print(f"[ParkIQ] Violation threshold: {VIOLATION_SEC}s | Server: {BASE_API_URL}")

    while True:
        # ── Dynamic calibration reload ──
        try:
            if os.path.exists("data/calibration.json"):
                mtime = os.path.getmtime("data/calibration.json")
                if mtime > last_calib_mtime:
                    last_calib_mtime = mtime
                    import json
                    with open("data/calibration.json", "r") as f:
                        calib = json.load(f)
                        for cam, data in calib.items():
                            if cam not in ROAD_CONFIG:
                                ROAD_CONFIG[cam] = {"zones": {}}
                            else:
                                ROAD_CONFIG[cam]["zones"] = {}
                            for z_id, z_data in data.get("zones", {}).items():
                                ROAD_CONFIG[cam]["zones"][z_id] = {
                                    "road_width_px": z_data["road_width_px"],
                                    "polygon": np.array(z_data["polygon"], dtype=np.int32)
                                }
                    print("[ParkIQ] \U0001f504 Live-reloaded calibration config from disk!")
        except Exception:
            pass

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

        draw_zones(frame)

        now = time.time()
        active_track_ids = set()

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes     = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes   = results[0].boxes.cls.cpu().numpy().astype(int)
            confs     = results[0].boxes.conf.cpu().numpy()

            # ── Cross-class IoU NMS ──────────────────────────────────────────
            # YOLO's built-in NMS suppresses duplicates within the same class.
            # This suppresses overlapping boxes ACROSS different classes so
            # the same physical car can't be counted as both "car" and "bus".
            def box_iou(b1, b2):
                ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
                ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
                a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
                return inter / (a1 + a2 - inter + 1e-6)

            CROSS_NMS_IOU = 0.85
            order = confs.argsort()[::-1]   # highest confidence first
            keep  = []
            suppressed = set()
            for i in order:
                if i in suppressed:
                    continue
                keep.append(i)
                for j in order:
                    if j != i and j not in suppressed:
                        if box_iou(boxes[i], boxes[j]) > CROSS_NMS_IOU:
                            suppressed.add(j)
            boxes     = boxes[keep]
            track_ids = track_ids[keep]
            classes   = classes[keep]
            confs     = confs[keep]
            seen_ids  = set(track_ids.tolist())


            # ── Ghost-track resurrection pass ────────────────────────────────
            # For every brand-new track_id, check if it's near a ghost.
            for box, track_id, cls_id in zip(boxes, track_ids, classes):
                if track_id in track_entry_time:
                    continue  # already a known track
                
                best_ghost_id, best_score = None, -1
                
                # Trust YOLO: if YOLO kept the same ID and we have it as a ghost, use it instantly!
                if track_id in ghost_tracks:
                    best_ghost_id = track_id
                else:
                    x1, y1, x2, y2 = box
                    cx, cy = bbox_center(x1, y1, x2, y2)
                    raw_type = VEHICLE_CLASSES.get(cls_id, "vehicle")
                    
                    for g_id, g in list(ghost_tracks.items()):
                        if now - g["ts"] > GHOST_TTL:
                            del ghost_tracks[g_id]
                            continue
                            
                        # Prevent a car from stealing a locked bus's ghost
                        if g["locked"] and g["vtype"] != raw_type:
                            continue
                            
                        dist = ((cx - g["cx"])**2 + (cy - g["cy"])**2) ** 0.5
                        if dist > GHOST_DIST_PX:
                            continue
                            
                        # Score matches using IoU + proximity
                        giou = box_iou(box, g["box"])
                        score = (giou * 500) + (GHOST_DIST_PX - dist)
                        
                        # STRICT MATCHING: If bounding boxes don't overlap by at least 20%, 
                        # it's a completely different vehicle. Do NOT let it steal the ghost!
                        if giou < 0.2:
                            continue
                            
                        if score > best_score:
                            best_score, best_ghost_id = score, g_id
                        
                if best_ghost_id is not None:
                    g = ghost_tracks.pop(best_ghost_id)
                    # Inherit everything from the ghost
                    track_entry_time[track_id]  = g["entry_time"]
                    track_zone[track_id]        = g["zone_id"]
                    track_vehicle_type[track_id]= g["vtype"]
                    if g["locked"]:
                        track_type_locked.add(track_id)
                    if g["violation_sent"]:
                        track_violation_sent.add(track_id)
                    if g.get("occupancy_sent", False):
                        track_occupancy_sent.add(track_id)
                    pending_exit[track_id] = 0   # clear any pending exit for the resurrected track
                    print(f"[GHOST] Track#{best_ghost_id} → #{track_id} resurrected (score={best_score:.0f}, dwell={now-g['entry_time']:.1f}s)")


            for box, track_id, cls_id in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = box
                raw_type = VEHICLE_CLASSES.get(cls_id, "vehicle")
                w = x2 - x1

                # ── Majority-vote type locking ──────────────────────────────────
                # Accumulate per-track votes; once a class dominates, lock it.
                if track_id not in track_type_locked:
                    votes = track_type_votes[track_id]
                    votes[raw_type] += 1
                    # Keep only the most recent VOTE_WINDOW assessments
                    total = sum(votes.values())
                    if total > VOTE_WINDOW:
                        # Trim oldest: reduce the least-voted class by 1
                        min_type = min(votes, key=votes.get)
                        votes[min_type] -= 1
                        if votes[min_type] == 0:
                            del votes[min_type]
                    # Lock if any single class reaches LOCK_VOTES
                    top_type = max(votes, key=votes.get)
                    if votes[top_type] >= LOCK_VOTES:
                        track_vehicle_type[track_id] = top_type
                        track_type_locked.add(track_id)
                        print(f"[LOCK] Track#{track_id} locked as '{top_type}' (votes: {dict(votes)})")
                    else:
                        # Not locked yet — use current best-guess
                        track_vehicle_type[track_id] = top_type

                vehicle_type = track_vehicle_type.get(track_id, raw_type)
                cx, cy = bbox_center(x1, y1, x2, y2)
                zone_id = get_vehicle_zone(cx, cy)

                active_track_ids.add(track_id)
                track_vehicle_type[track_id] = vehicle_type
                track_last_pos[track_id] = (cx, cy)   # always update last known position
                track_last_box[track_id] = box        # save bounding box for IoU matching

                if zone_id:
                    pending_exit[track_id] = 0   # car is in zone — reset any pending exit

                    # First time entering this zone
                    if track_id not in track_entry_time or track_zone.get(track_id) != zone_id:
                        track_entry_time[track_id] = now
                        track_zone[track_id] = zone_id
                        track_violation_sent.discard(track_id)
                        track_occupancy_sent.discard(track_id)

                    dwell_sec = now - track_entry_time[track_id]
                    
                    if dwell_sec >= STATIONARY_SEC:
                        if track_id not in track_occupancy_sent:
                            track_occupancy_sent.add(track_id)
                            send_event(track_id, vehicle_type, zone_id, "OCCUPANCY_START", dwell_sec, 0, "LOW")

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
                            update_hardware_leds(active_track_ids, track_entry_time, track_vehicle_type, track_zone, track_violation_sent, ghost_tracks)

                        # Periodic telemetry
                        if now - last_telemetry[track_id] > TELEMETRY_SEC:
                            send_event(track_id, vehicle_type, zone_id,
                                       "TELEMETRY_UPDATE", dwell_sec, cis, priority)
                            last_telemetry[track_id] = now

                        colour = COLOUR_MAP[priority]
                        label = f"#{track_id} {vehicle_type} | {zone_id} | CIS:{cis} [{priority}] {int(dwell_sec)}s"
                        draw_vehicle(frame, x1, y1, x2, y2, label, colour)
                    else:
                        label = f"#{track_id} {vehicle_type}"
                        draw_vehicle(frame, x1, y1, x2, y2, label, (180, 180, 180))
                else:
                    if track_id in track_zone:
                        # Increment grace counter — must be out of zone for EXIT_GRACE_FRAMES
                        pending_exit[track_id] += 1
                        if pending_exit[track_id] >= EXIT_GRACE_FRAMES:
                            if track_id in track_occupancy_sent:
                                dwell_sec = now - track_entry_time.get(track_id, now)
                                cis = compute_cis(vehicle_type, dwell_sec, track_zone[track_id])
                                send_event(track_id, vehicle_type, track_zone[track_id],
                                           "VACATED", dwell_sec, cis, priority_label(cis))
                                
                                track_violation_sent.discard(track_id)
                                update_hardware_leds(active_track_ids, track_entry_time, track_vehicle_type, track_zone, track_violation_sent, ghost_tracks)
                                
                            track_occupancy_sent.discard(track_id)
                            del track_zone[track_id]
                            del track_entry_time[track_id]
                            pending_exit[track_id] = 0
                        else:
                            # Still in grace period. Draw it as grey since it's physically out of the zone.
                            label = f"#{track_id} {vehicle_type}"
                            draw_vehicle(frame, x1, y1, x2, y2, label, (180, 180, 180))
                            continue # skip drawing the default grey box below

                    label = f"#{track_id} {vehicle_type}"
                    draw_vehicle(frame, x1, y1, x2, y2, label, (180, 180, 180))


        # ── Ghost expiry & lost-track cleanup ───────────────────────────────
        lost = set(track_zone.keys()) - active_track_ids
        for tid in lost:
            if tid in track_entry_time:
                dwell_sec = now - track_entry_time[tid]
                vt = track_vehicle_type.get(tid, "vehicle")
                cis = compute_cis(vt, dwell_sec, track_zone[tid])
                lx, ly = track_last_pos.get(tid, (-1, -1))
                # Store as ghost instead of immediately vacating
                ghost_tracks[tid] = {
                    "cx":            lx,
                    "cy":            ly,
                    "box":           track_last_box.get(tid, [0,0,0,0]),
                    "zone_id":       track_zone[tid],
                    "entry_time":    track_entry_time[tid],
                    "violation_sent": tid in track_violation_sent,
                    "occupancy_sent": tid in track_occupancy_sent,
                    "vtype":         vt,
                    "locked":        tid in track_type_locked,
                    "ts":            now,
                }
                # Don't send VACATED yet — wait for GHOST_TTL to expire
                del track_zone[tid]
                del track_entry_time[tid]

        # ── Expire old ghosts and send VACATED ───────────────────────────────
        for g_id in list(ghost_tracks.keys()):
            g = ghost_tracks[g_id]
            if now - g["ts"] > GHOST_TTL:
                dwell_sec = now - g["entry_time"]
                if g.get("occupancy_sent", False):
                    cis = compute_cis(g["vtype"], dwell_sec, g["zone_id"])
                    send_event(g_id, g["vtype"], g["zone_id"], "VACATED",
                               dwell_sec, cis, priority_label(cis))
                track_violation_sent.discard(g_id)
                track_occupancy_sent.discard(g_id)
                update_hardware_leds(active_track_ids, track_entry_time, track_vehicle_type, track_zone, track_violation_sent, ghost_tracks)
                del ghost_tracks[g_id]
                if g.get("occupancy_sent", False):
                    print(f"[GHOST] Track#{g_id} expired after {dwell_sec:.1f}s — VACATED sent")

        ts = datetime.datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"ParkIQ CCTV | {ts} | Active: {len(active_track_ids)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Camera: {DEVICE_ID}", 
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Write the fully-annotated frame once per processed loop tick.
        # Skipped frames are not written — last good frame stays on disk.
        cv2.imwrite(f"data/current_frame_{DEVICE_ID}.jpg", frame)

        if show:
            cv2.imshow("ParkIQ — Illegal Parking Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[ParkIQ] Quit by user.")
                break

    cap.release()
    if show:
        cv2.destroyAllWindows()
    
    # ── Clean up hardware state when stream ends ──
    print("[ParkIQ] Stream ended. Clearing all active tracks.")
    now = time.time()
    
    # Send VACATED for all active tracks
    for tid, etime in track_entry_time.items():
        if tid in track_occupancy_sent:
            dwell_sec = now - etime
            vt = track_vehicle_type.get(tid, "vehicle")
            cis = compute_cis(vt, dwell_sec, track_zone[tid])
            send_event(tid, vt, track_zone[tid], "VACATED", dwell_sec, cis, priority_label(cis))
            
    # Send VACATED for all ghost tracks
    for g_id, g in ghost_tracks.items():
        if g.get("occupancy_sent", False):
            dwell_sec = now - g["entry_time"]
            cis = compute_cis(g["vtype"], dwell_sec, g["zone_id"])
            send_event(g_id, g["vtype"], g["zone_id"], "VACATED", dwell_sec, cis, priority_label(cis))

    print("[ParkIQ] Resetting hardware LEDs to VACANT.")
    try:
        import requests
        requests.post(f"{BASE_API_URL}/api/set-buzzer", json={"active": False, "zone_id": "", "level": "VACANT"}, timeout=2.0)
    except Exception:
        pass

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
    parser.add_argument("--cloud", action="store_true",
                        help="Point to the hosted Render backend instead of localhost.")
    args = parser.parse_args()

    if args.cloud:
        BASE_API_URL = "https://parkiq-glrk.onrender.com"

    DEVICE_ID = args.device_id
    
    # Determine the source
    if args.source is not None:
        src_raw = args.source
    else:
        src_raw = PHONE_CAMERA_1_URL if args.cam == 1 else PHONE_CAMERA_2_URL
        
    src = int(src_raw) if str(src_raw).isdigit() else src_raw

    if args.calibrate:
        print(f"[CALIBRATE] Opening camera feed: {src}")
        print("STEP 1: For each lane/zone, click 2 points for the lane width (LEFT then RIGHT).")
        print("STEP 2: Then click 4 points for the NO-PARKING ZONE. Repeat for multiple zones.")
        print("Press ENTER when you have marked all zones.")
        cap = cv2.VideoCapture(src)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"\n❌ ERROR: Could not open video source: '{src}'")
            print("Please make sure the file exists and the path is correct!")
        else:
            state   = {"phase": 1, "zones": [], "current_road_pts": [], "current_zone_pts": []}
            clone   = frame.copy()

            def click(event, x, y, flags, param):
                if event != cv2.EVENT_LBUTTONDOWN:
                    return
                if state["phase"] == 1:
                    state["current_road_pts"].append([x, y])
                    cv2.circle(clone, (x, y), 6, (0, 200, 255), -1)
                    if len(state["current_road_pts"]) == 1:
                        cv2.putText(clone, "Now click RIGHT edge of lane/road",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
                    elif len(state["current_road_pts"]) == 2:
                        p1, p2 = state["current_road_pts"]
                        cv2.line(clone, tuple(p1), tuple(p2), (0, 200, 255), 2)
                        road_w = abs(p2[0] - p1[0])
                        cv2.putText(clone, f"Width: {road_w}px — Now click 4 points for NO-PARKING ZONE",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        state["phase"] = 2
                elif state["phase"] == 2:
                    pts = state["current_zone_pts"]
                    pts.append([x, y])
                    cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
                    if len(pts) > 1:
                        cv2.line(clone, tuple(pts[-2]), tuple(pts[-1]), (0,255,0), 2)
                    if len(pts) == 4:
                        cv2.line(clone, tuple(pts[-1]), tuple(pts[0]), (0,255,0), 2)
                        road_w = abs(state["current_road_pts"][1][0] - state["current_road_pts"][0][0])
                        state["zones"].append({
                            "road_width_px": road_w,
                            "polygon": pts.copy()
                        })
                        state["current_zone_pts"] = []
                        cv2.putText(clone, f"Zone {len(state['zones'])} saved. Click 4 pts for another zone here, 'n' for new lane, or ENTER to finish.",
                                    (10, 90 + 30*len(state['zones'])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                        # Keep phase 2 so they can keep adding zones with the same lane width
                cv2.imshow("Calibrate", clone)

            cv2.putText(clone, "STEP 1: Click LEFT edge of lane/road",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
            cv2.imshow("Calibrate", clone)
            cv2.setMouseCallback("Calibrate", click)
            
            while True:
                key = cv2.waitKey(10) & 0xFF
                if key == 13 or key == 10:  # Enter
                    break
                elif key == ord('n') or key == ord('N'):
                    if state["phase"] == 2 and len(state["current_zone_pts"]) == 0:
                        state["phase"] = 1
                        state["current_road_pts"] = []
                        cv2.putText(clone, "STEP 1: Click LEFT edge of NEW lane/road",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
                        cv2.imshow("Calibrate", clone)
            cv2.destroyAllWindows()

            zones = state["zones"]
            if len(zones) > 0:
                import json
                print(f"\n✅ Saving calibration data for {DEVICE_ID} automatically...")
                calib = {}
                if os.path.exists("data/calibration.json"):
                    try:
                        with open("data/calibration.json", "r") as f:
                            calib = json.load(f)
                    except: pass
                
                if DEVICE_ID not in calib:
                    calib[DEVICE_ID] = {"zones": {}}
                
                calib[DEVICE_ID]["zones"] = {}
                for i, z in enumerate(zones):
                    calib[DEVICE_ID]["zones"][f"Zone-{i+1}"] = {
                        "road_width_px": int(z["road_width_px"]),
                        "polygon": [list(int(p) for p in pt) for pt in z["polygon"]]
                    }
                
                with open("data/calibration.json", "w") as f:
                    json.dump(calib, f, indent=4)
                    
                print("✅ Successfully auto-updated calibration! You can now start the live detector.")
            else:
                print("\n\u274c Error: No complete zones (road width + 4 points) were marked.")
    else:
        run(source=src, show=args.show)
