"""
HADO Smart Court IoT — Raspberry Pi 4 Edge Streamer
=====================================================
Pipeline:
  Mokose HDMI capture → cv2.VideoCapture (MJPEG)
    → YOLOv8n-pose (NCNN, imgsz=320, FP16)
    → Vest hue sampling (torso HSV)
    → Socket.IO 'hado_stream' to MacBook server

Option A optimizations:
  1) NCNN model format (ARM-optimized, ~3x faster than ONNX)
  2) imgsz=320 (vs 640) — ~3x faster
  3) Multi-threaded capture (decouples I/O from inference)

Hybrid role classification:
  Pi extracts torso vest HUE → sends to server
  Server combines hue (cold start) + behavior (after 30 s)
"""

import cv2
import time
import math
import threading
import numpy as np
import socketio
from ultralytics import YOLO

# ════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════
MACBOOK_SERVER_URL = "http://192.168.0.15:8080"   # ⚠️ Set to your Mac IP
MODEL_PATH         = "yolov8n-pose_ncnn_model"    # NCNN export folder
IMG_SIZE           = 320                          # inference resolution
TARGET_FPS         = 15                           # max FPS cap
CONF_THRESHOLD     = 0.35
CAM_INDEX          = 0                            # /dev/video0
COURT_CENTER_X     = 320                          # match server's midline


# ════════════════════════════════════════════════════════════════
# Mokose HDMI Capture Setup (UVC, MJPEG forced)
# ════════════════════════════════════════════════════════════════
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)               # keep latest frame only

if not cap.isOpened():
    raise RuntimeError("❌ Camera not found. Check Mokose USB connection.")


# ════════════════════════════════════════════════════════════════
# Threaded Capture (always serves the freshest frame)
# ════════════════════════════════════════════════════════════════
_latest_frame = None
_frame_lock   = threading.Lock()
_running      = True


def _capture_loop():
    global _latest_frame
    while _running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.005)
            continue
        with _frame_lock:
            _latest_frame = frame


def get_latest_frame():
    with _frame_lock:
        return None if _latest_frame is None else _latest_frame.copy()


threading.Thread(target=_capture_loop, daemon=True).start()


# ════════════════════════════════════════════════════════════════
# Vest Hue Sampling (torso HSV)
# COCO keypoints: 5=L-shoulder, 6=R-shoulder, 11=L-hip, 12=R-hip
# Returns: median hue (0–179) or -1 if torso region is unreliable.
# ════════════════════════════════════════════════════════════════
def sample_vest_hue(frame, kp_list):
    if len(kp_list) < 13:
        return -1
    ls, rs = kp_list[5], kp_list[6]
    lh, rh = kp_list[11], kp_list[12]

    # All four torso keypoints must be confidently visible
    if min(ls[2], rs[2], lh[2], rh[2]) < 0.30:
        return -1

    h, w = frame.shape[:2]
    x1 = int(max(0, min(ls[0], rs[0], lh[0], rh[0])))
    y1 = int(max(0, min(ls[1], rs[1])))
    x2 = int(min(w, max(ls[0], rs[0], lh[0], rh[0])))
    y2 = int(min(h, max(lh[1], rh[1])))
    if x2 - x1 < 6 or y2 - y1 < 10:
        return -1

    torso = frame[y1:y2, x1:x2]
    hsv   = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    # Use only saturated, well-lit pixels (ignore black/shadow/highlight)
    sat_mask = (hsv[..., 1] > 70) & (hsv[..., 2] > 60) & (hsv[..., 2] < 240)
    if sat_mask.sum() < 25:
        return -1

    # Circular-aware median (hue wraps 0↔180)
    hues = hsv[..., 0][sat_mask].astype(np.float32)
    rad  = hues * (2 * math.pi / 180.0)
    cs, sn = float(np.cos(rad).mean()), float(np.sin(rad).mean())
    mean_rad = math.atan2(sn, cs)
    mean_hue = (mean_rad * 180.0 / (2 * math.pi)) % 180.0
    return int(round(mean_hue))


# ════════════════════════════════════════════════════════════════
# Pose Classification (HADO-specific stances)
# ════════════════════════════════════════════════════════════════
#   CHARGE  — arm raised UPWARD (preparing to attack)
#   ATTACK  — arm extended FORWARD toward opponent team
#   SHIELD  — arm lowered toward ground (defensive ready)
#   NEUTRAL — none of the above
#
# Facing direction is inferred from player's X position:
#   Team A (x < center) → faces +x (right)
#   Team B (x ≥ center) → faces -x (left)
# COCO keypoints: 5,6=shoulders · 7,8=elbows · 9,10=wrists · 11,12=hips
# ════════════════════════════════════════════════════════════════
def classify_pose(kp_list, player_center_x):
    if len(kp_list) < 17:
        return "unknown"

    ls, rs = kp_list[5],  kp_list[6]
    lw, rw = kp_list[9],  kp_list[10]
    lh, rh = kp_list[11], kp_list[12]

    # Reference scale (normalize for camera distance)
    sw = abs(rs[0] - ls[0])
    th = abs((rs[1] + ls[1]) / 2 - (rh[1] + lh[1]) / 2)
    scale = max(sw, th * 0.6, 30.0)

    # +1 = facing right (team A), -1 = facing left (team B)
    facing = +1 if player_center_x < COURT_CENTER_X else -1

    def arm_state(shoulder, wrist):
        if shoulder[2] < 0.30 or wrist[2] < 0.30:
            return None
        dx_raw = (wrist[0] - shoulder[0]) / scale
        dy     = (wrist[1] - shoulder[1]) / scale
        forward = dx_raw * facing                    # +ve = toward opponent

        # 1) CHARGE — wrist clearly above shoulder, near vertical
        if dy < -0.55 and abs(dx_raw) < 0.70:
            return "charge"

        # 2) ATTACK — wrist extended FORWARD (toward opponent), near shoulder height
        if forward > 0.65 and abs(dy) < 0.50:
            return "attack"

        # 3) SHIELD — wrist below shoulder (arm down, ready to raise)
        if dy > 0.35 and abs(dx_raw) < 0.90:
            return "shield"

        return "neutral"

    right_state = arm_state(rs, rw)
    left_state  = arm_state(ls, lw)

    # Priority: ATTACK > CHARGE > SHIELD > NEUTRAL
    states = [s for s in (right_state, left_state) if s is not None]
    for label in ("attack", "charge", "shield"):
        if label in states:
            return label
    return "neutral" if states else "unknown"


# ════════════════════════════════════════════════════════════════
# Load Model & Connect to Server
# ════════════════════════════════════════════════════════════════
print(f"📦 Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH, task="pose")

sio = socketio.Client()


@sio.event
def connect():
    print("✅ Connected to MacBook server")


@sio.event
def disconnect():
    print("⚠️  Disconnected from MacBook server")


print(f"🌐 Connecting to {MACBOOK_SERVER_URL} ...")
try:
    sio.connect(MACBOOK_SERVER_URL)
except Exception as e:
    print(f"❌ Connection failed: {e}")
    raise SystemExit


# ════════════════════════════════════════════════════════════════
# Main Inference Loop
# ════════════════════════════════════════════════════════════════
def build_payload(frame, results):
    """Convert YOLO results → JSON payload for the MacBook server."""
    payload = []
    for result in results:
        boxes     = result.boxes
        keypoints = result.keypoints
        if boxes is None or keypoints is None or len(boxes) == 0:
            continue

        for i, kp in enumerate(keypoints.data):
            pid = int(boxes.id[i].item()) if boxes.id is not None else i
            xy  = boxes.xyxy[i].tolist()
            cx  = (xy[0] + xy[2]) / 2
            cy  = (xy[1] + xy[3]) / 2
            kp_list = kp.tolist()

            vest_hue = sample_vest_hue(frame, kp_list)
            pose     = classify_pose(kp_list, cx)

            payload.append({
                "playerId":       pid,
                "x":              round(cx, 1),
                "y":              round(cy, 1),
                "rightWristY":    round(kp_list[10][1], 1) if len(kp_list) > 10 else 0,
                "rightShoulderY": round(kp_list[6][1],  1) if len(kp_list) > 6  else 0,
                "vestHue":        vest_hue,    # hybrid role input (-1 if invalid)
                "pose":           pose,        # "attack" | "charge" | "shield" | "neutral" | "unknown"
            })
    return payload


def main():
    global _running
    fps_t0, fps_count = time.time(), 0

    print(f"🚀 Streaming at target {TARGET_FPS} FPS, imgsz={IMG_SIZE}")
    try:
        while True:
            loop_start = time.time()

            frame = get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Track mode → boxes.id is populated (ByteTrack)
            results = model.track(
                frame,
                imgsz=IMG_SIZE,
                persist=True,
                verbose=False,
                conf=CONF_THRESHOLD,
            )

            payload = build_payload(frame, results)
            if payload:
                sio.emit('hado_stream', payload)

            # ── FPS report every 2 sec ──
            fps_count += 1
            if time.time() - fps_t0 >= 2.0:
                fps = fps_count / (time.time() - fps_t0)
                print(f"[FPS] {fps:.1f} · players: {len(payload)}")
                fps_count, fps_t0 = 0, time.time()

            # ── Enforce target FPS cap ──
            delay = max(1.0 / TARGET_FPS - (time.time() - loop_start), 0)
            time.sleep(delay)

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")

    finally:
        _running = False
        cap.release()
        sio.disconnect()
        print("✅ Clean shutdown complete")


if __name__ == '__main__':
    main()
