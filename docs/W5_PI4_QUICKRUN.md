# W5 Pi4 현장 실측 빠른 실행 가이드

> 2026-06-11~17 코트 현장용. 이 파일만 열면 모든 명령어가 있음.

---

## 사전 준비 (집에서 미리)

```bash
# Pi4에서 NCNN 모델 준비 (Mac에서 export 후 SCP)
# Mac:
python HADO_iot/pi/export_ncnn.py    # yolov8n-pose_ncnn_model/ 생성
scp -r yolov8n-pose_ncnn_model pi@<PI_IP>:~/hado-smart-court-iot/

# Pi4에서 TTS 설치
sudo apt install espeak-ng

# Pi4에서 의존성 확인
source ~/hado_venv/bin/activate
pip install -r requirements.txt
```

---

## 현장 실행 순서

### Step 1 — 캘리브레이션 (코트 코너 마커 부착 후)

```bash
./run.sh calibrate
# SPACE → 코너 4점 클릭 (좌상→우상→우하→좌하) → ENTER
# "MEAN ROUND-TRIP ERROR: X.X mm" 확인 → 50mm 이하가 목표
```

### Step 2 — Level 2 라이브 데모

```bash
./run.sh main --level 2 --voice
# 키: ESC 종료 / r CSV 녹화 시작/중지 / t 팀 초기화
```

### Step 3 — 동작 인식 데모

```bash
./run.sh action --threaded
# ESC로 종료 (Pi4 headless: Ctrl+C)
```

---

## W5 실측 명령어 (QA_PREP.md [TODO] 채우기)

### FPS 측정 (NCNN)

```bash
./run.sh bench --frames 200 --imgsz 320 --model yolov8n-pose_ncnn_model
```

출력에서 기록:
- `Wall-clock FPS` → QA_PREP.md Q5, Q7 `[TODO: NCNN fps]`
- `피크 RAM` → Q5 `[TODO: RAM]`
- `최고 CPU 온도` → Q9 `[TODO: 온도]`

### FPS 측정 (ONNX 비교용)

```bash
./run.sh bench --frames 200 --imgsz 320 --model yolov8n-pose.onnx
```

출력에서 기록: `Wall-clock FPS` → `[TODO: ONNX fps]`

### 위치 오차 측정 (10점 마커법)

```bash
# 코트 위 알려진 위치 10곳에 마커 부착 (예: 모서리 2곳 + 라인 교차 8곳)
# 마커 실제 좌표(m) vs 시스템 추정 좌표(m) 수동 비교
# CSV 녹화 후 분석:
./run.sh main --level 1
# 'r' 키 → CSV 시작 → 10 위치 차례로 서기 → 'r' 키 → CSV 중지
# data/position_logs/positions_*.csv 확인
```

---

## 기록 체크리스트 (현장에서 채울 것)

| 항목 | 측정값 | 목표 |
|------|--------|------|
| 평균 FPS (NCNN, threaded) | | ≥15 |
| 평균 FPS (ONNX) | | ≥6 |
| 피크 RAM (NCNN 모드) | | <1.5 GB |
| CPU 온도 (5분 실행) | | <70°C |
| 위치 오차 RMS (10점 평균) | | <10 cm |
| 캘리브레이션 오차 | | <50 mm |
| TTS 지연 (규칙 발화~음성) | | <500 ms |

---

## 트러블슈팅 (현장)

| 문제 | 즉각 해결책 |
|------|------------|
| 카메라 안 잡힘 | `ls /dev/video*` → source 번호 확인 후 `--source 2` 등 |
| FPS < 10 | `--imgsz 256` 또는 `NCNN` 모델 확인 |
| 팀 배정 잘못됨 | 실행 중 `t` 키 눌러 초기화 |
| 음성 안 나옴 | `espeak-ng "테스트"` → 소리 확인 |
| 캘리브레이션 오차 > 50mm | 마커 재부착 후 재캘리브레이션 |
| 녹화 파일 없음 | `--record` 옵션 확인, 경로 미리 생성 |

---

## 촬영용 명령어 (데모 영상 녹화)

```bash
# 90초 데모 영상 (Level 2, headless 녹화)
./run.sh main --level 2 --voice --headless --record data/field_demo.mp4 --max-frames 2700

# 동작 인식 데모 녹화
./run.sh action --threaded --record data/action_demo_field.mp4 --max-frames 1800
```

---

*이 파일을 Pi4 바탕화면에 단축키로 등록해두면 현장에서 빠르게 참조 가능.*
