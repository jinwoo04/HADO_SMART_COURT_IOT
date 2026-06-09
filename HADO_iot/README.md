# HADO Smart Court IoT

Real-time player tracking and tactical advisory system for the AR sport HADO.

**Pipeline:** Mokose HDMI capture → Raspberry Pi 4 (YOLOv8n-pose) → MacBook server (tactical analysis) → Live advice broadcast.

---

## Project Structure

```
HADO_iot/
├── server/
│   ├── server.js              # Node.js backend (Express + Socket.IO)
│   └── package.json
├── pi/
│   ├── edge_streamer.py       # Pi edge streamer (NCNN + threaded)
│   ├── export_ncnn.py         # one-time NCNN export (run on Mac)
│   └── requirements.txt
├── run_server.sh              # launch MacBook server
├── run_pi.sh                  # launch Pi streamer
└── README.md
```

---

## Architecture — Option A Optimizations

| Layer | Optimization | Effect |
|---|---|---|
| Model format | **NCNN** (ARM-optimized) | ~3× faster than ONNX |
| Inference size | **imgsz=320** (vs 640) | ~3× fewer pixels |
| Capture I/O | **Threaded capture** | Decouples camera I/O from inference |
| Network | Smartphone hotspot Wi-Fi | Bypasses school P2P blocks |
| Camera | Mokose HDMI → USB (UVC) | Plug-and-play on Pi |
| Role ID | **Hybrid: color + behavior** | Instant cold-start, adapts over time |

**Expected on Pi 4 CPU:** 1.5 FPS (baseline) → **10–15 FPS** (optimized).

---

## Setup — MacBook (Server)

### 1) Install Node.js

```bash
brew install node
```

### 2) Check your MacBook IP

```bash
ipconfig getifaddr en0
# example: 192.168.0.15
```

Record this IP — you will paste it into the Pi script.

### 3) Launch the server

```bash
cd HADO_iot
./run_server.sh
# Server listens on port 8080
```

When the Pi connects for the first time, **macOS will show a firewall popup → click Allow**.

---

## Setup — MacBook (One-time NCNN export)

YOLOv8 → NCNN export is much faster on Mac than Pi.

```bash
cd HADO_iot/pi
pip install ultralytics
python export_ncnn.py
# → creates yolov8n-pose_ncnn_model/
```

Copy the folder to the Pi:

```bash
scp -r yolov8n-pose_ncnn_model pi@<PI_IP>:~/HADO_iot/pi/
```

---

## Setup — Raspberry Pi 4

### 1) Connect hardware
- Plug **Mokose HDMI capture** into Pi's **blue USB 3.0 port**
- Plug source camera HDMI into the Mokose
- Ensure source camera outputs **clean HDMI** (no overlays)

### 2) Install dependencies

```bash
cd HADO_iot/pi
pip install -r requirements.txt
```

### 3) Configure server IP

Edit `pi/edge_streamer.py`:

```python
MACBOOK_SERVER_URL = "http://192.168.0.15:8080"   # ← your Mac IP here
```

### 4) Launch

```bash
cd HADO_iot
./run_pi.sh
```

You should see:
```
📦 Loading model: yolov8n-pose_ncnn_model
✅ Connected to MacBook server
🚀 Streaming at target 15 FPS, imgsz=320
[FPS] 12.4 · players: 4
[FPS] 13.1 · players: 6
```

---

## Verification Checklist

| Step | Command | Expected |
|---|---|---|
| Camera detected on Pi | `ls /dev/video*` | `/dev/video0` |
| MJPEG support | `v4l2-ctl --list-formats-ext -d /dev/video0` | Includes `MJPG` |
| Pi → Mac network | `ping <MAC_IP>` from Pi | Reply < 10 ms |
| Server alive | Open `http://<MAC_IP>:8080` in browser | (404 OK — server responds) |
| FPS target met | Watch `[FPS] X.X` in Pi logs | ≥ 10 FPS |

---

## Partial Visibility Handling

The system **never blocks** — it adapts to however many players the camera sees:

| Detected | Team A | Team B | What runs |
|---|---|---|---|
| **6 (full)**   | 3 | 3 | Full role classification + all scenarios with strict thresholds |
| **4–5**        | any | any | Same as above, with `[N/6 visible]` note appended to advice |
| **2–3**        | ≥1 | ≥1 | **Small-sample mode** — relaxed ratio thresholds + close-range duel scenario |
| **1 vs 1**     | 1 | 1 | Small-sample mode; distance-based engagement detection |
| **1 only (our)** | 1 | 0 | **Single-team mode** — individual positional/pose feedback |
| **1 only (enemy)** | 0 | 1 | "Reposition into frame" advice |
| **0**          | 0 | 0 | Idle status only |

The role classifier still runs on whoever is visible, but **confidence is penalized** when fewer than 3 players are detected on a team:

| Team size | Confidence multiplier |
|---|---|
| 3 players | × 1.0 |
| 2 players | × 0.7 |
| 1 player  | × 0.4 |

### Output: `detection_status` (every 2 s)

```
{ detected: 4, expected: 6, teamA: 2, teamB: 2, quality: "partial" }
```

`quality` ∈ `"good"` (≥5) · `"partial"` (3–4) · `"low"` (1–2) · `"none"` (0).

---

## Tactical Scenarios

The MacBook server runs analysis every **2 seconds** over a **40-second sliding window**.

### Scenario A — Enemy Right-Side Concentration
```
enemyRightRatio > 0.7 && ourRightRatio < 0.4
→ "🚨 Enemy focused right side — reinforce right-side defense!"
```

### Scenario B — Enemy Forward Pressure
```
enemyAvgX < 380 && ourAvgX < 100
→ "⚠️ Heavy forward pressure — deploy shields and push formation up!"
```

### Default
```
→ "✅ Formation match is stable."
```

---

## Pose Classification (HADO Stances)

The Pi labels each detected player every frame with one of:

| Pose | Trigger | Body Cue |
|---|---|---|
| 🟦 **charge**  | `dy < -0.55 * scale` and arm near-vertical | Arm raised straight up — preparing to attack |
| 🟥 **attack**  | `forward > 0.65 * scale` and near shoulder height | Arm extended **toward opponent team** |
| 🟩 **shield**  | `dy > 0.35 * scale` and arm near body | Arm lowered toward ground — defensive ready |
| ⬜ **neutral** | none of the above | Standing / moving |
| ❔ **unknown** | low keypoint confidence | Occluded or off-screen |

**Facing direction is team-aware.** Team A (left half) faces +x, Team B (right half) faces -x — so "arm extended forward" only counts when the wrist projects toward the opponent's half. Sideways or backward arm motion never triggers `attack`.

**Reference scale** uses shoulder width and torso height — thresholds normalize across camera distance so the same logic works whether the player is 50 px or 200 px tall in the frame.

**Priority** when both arms detect different states: `attack > charge > shield > neutral`.

---

## Hybrid Role Classification

Each player is classified into **Attacker / Technician / Defender** using a two-stage hybrid scheme.

| Stage | Time | Primary Signal | Confidence |
|---|---|---|---|
| 🧊 **Cold start** | 0–30 s   | Vest hue (color)         | 0.3 – 1.0 |
| 🔥 **Warm**       | 30 s +   | Behavior (rank in team)  | 0.65 – 0.95 |

### How it works

**Pi side** — for each detected player, sample the torso region defined by COCO keypoints 5, 6, 11, 12 (shoulders → hips), convert BGR → HSV, and report the **median hue** (`vestHue` field, 0–179, or `-1` if unreliable).

**Server side** — every 2 s, group recent samples by `playerId`:
- Count **attack motions** (`rightShoulderY - rightWristY > 30 px`)
- Count **forward-zone occupancy** (player on enemy half)
- Compute **mobility** (positional standard deviation)
- Tally **color votes** across the 30-second window

During the cold-start span, each player is assigned the role matching the most-voted color. After 30 seconds, players are **ranked within their team** by attack score; mobility breaks ties. If behavior and color agree, confidence rises to **0.95**.

### Vest color configuration

Edit `HUE_ROLE_TABLE` in `server/server.js`:

```javascript
const HUE_ROLE_TABLE = [
  { role: "Attacker",   ranges: [[0, 10], [170, 180]] },  // red
  { role: "Technician", ranges: [[20, 40]] },              // yellow
  { role: "Defender",   ranges: [[90, 130]] },             // blue
];
```

Calibrate on the actual court — lighting shifts HSV hue by ±10. Use a quick capture and `cv2.cvtColor(..., COLOR_BGR2HSV)` to check real values before the match.

### Output event

```
io.emit('player_roles', {
  roles: {
    "1": { team: "A", role: "Attacker",   confidence: 0.95, source: "behavior+color" },
    "2": { team: "A", role: "Technician", confidence: 0.65, source: "behavior" },
    "3": { team: "A", role: "Defender",   confidence: 0.95, source: "behavior+color" },
    ...
  },
  coldStart: false,
  elapsedMs: 47123
})
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Camera not found` | Wrong USB port / cable | Use Pi's blue USB 3.0 port |
| `Connection failed` | Wrong server IP | Re-check `ipconfig getifaddr en0` |
| FPS < 5 | Pi CPU throttling | `vcgencmd measure_temp` → add fan |
| `ncnn` import error | Missing package | `pip install ncnn` |
| Player IDs flicker | `conf` too low | Raise `CONF_THRESHOLD` to 0.4 |
| Lag accumulates | Buffer fills up | Confirm `CAP_PROP_BUFFERSIZE=1` |
| macOS rejects connection | Firewall popup missed | System Settings → Network → Firewall |

---

## Team

Jinu Park (박진우) · Junhyeok Jung (정준혁)

IoT Term Project · 2026
