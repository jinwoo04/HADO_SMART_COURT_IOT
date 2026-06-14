# W5 현장 테스트 가이드 — 2026-06-14

> 이 문서 한 장이면 내일 현장에서 처음부터 끝까지 진행 가능.
> Pi4 세팅 → 캘리브레이션 → Level 1/2/3 데모 → W5 실측 → 녹화

---

## 시스템 아키텍처 (발표 기준)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         HADO Smart Court IoT                        │
│                   "단일 카메라로 코트 전체를 읽는다"                      │
└─────────────────────────────────────────────────────────────────────┘

  HADO 코트 (10.0 × 6.0 m)
  ┌──────────────────────────────────────────────────────────────────┐
  │   [Fixed Camera / GoPro]                                         │
  │        │ USB/CSI                                                 │
  │        ▼                                                         │
  │  ┌─────────────┐    Wi-Fi hotspot    ┌─────────────────────┐     │
  │  │ Raspberry   │ ─────────────────▶  │   MacBook Server    │     │
  │  │   Pi 4      │   player state      │                     │     │
  │  │             │   (bbox + kpts)     │  ① Position analysis│     │
  │  │ ① YOLOv8n  │                     │  ② Bird-eye view    │     │
  │  │    -pose    │                     │  ③ Tactic engine    │     │
  │  │ ② IoU Trk  │                     │  ④ TTS / Arrow      │     │
  │  │ ③ Homogr.  │                     │                     │     │
  │  └─────────────┘                     └─────────────────────┘     │
  │   Edge Layer                          Server Layer                │
  └──────────────────────────────────────────────────────────────────┘

  ※ 독립 실행 모드: Pi4 단독으로 전체 파이프라인 실행 가능 (No Wi-Fi 필요)
```

---

## 파이프라인 플로우 (중간발표 기준)

```
카메라 입력
    │
    ▼
[YOLOv8n-pose]  ── imgsz=320 (NCNN ARM FP16) ──▶  bbox + 17 keypoints
    │
    ▼
[IoU Tracker]  ──▶  player_id 부여 및 유지 (max_lost=15 frames)
    │
    ▼
[Homography]  ──▶  pixel (x_px, y_px) → court meter (x_m, y_m)
    │              foot point = bbox 하단 중앙 ((x1+x2)/2, y2)
    ▼
[Bird-Eye View]  ──▶  10.0×6.0m 코트 위에 선수 위치 실시간 표시
    │
    ├──── Level 1 여기서 끝 (./run.sh main --level 1)
    │
    ▼
[Tactic Engine]  ──▶  R1~R6 규칙 판단 → TacticAdvice
    │
    ├── R1 Spacing   : 팀원 1.0m 이내 → 분산
    ├── R2 Coverage  : 한쪽 코트 비움 → 커버
    ├── R3 Counter   : 정면 2.5m 이내 적 → 측면 회피
    ├── R4 Gap Attack: 적 사이 2.0m 공간 → 전진
    ├── R5 Backline  : 적 진영 침투 → 후방 수비
    └── R6 Lane Cover: 적 레인 집중 → 해당 레인 커버
    │
    ▼
[시각화 + TTS]  ──▶  화살표 오버레이 + 한국어 음성 안내
    │
    └──── Level 2 여기서 끝 (./run.sh main --level 2 --voice)

                         ┌──────────────────────┐
별도 데모:               │  [Pose Estimator]    │
Action Recognition  ──▶  │  17 keypoints → 각도 │──▶ 7동작 분류
(demo_pose / action)      │  charge / shoot /    │    + confidence bar
                          │  shield / dodge L/R  │    + history log
                          │  crouch / ready      │
                          └──────────────────────┘
```

---

## 코트 구역 레이아웃

```
         ◀───────── 10.0 m ─────────▶
         팀A 영역 (x=0~5)   팀B 영역 (x=5~10)

    ┌────┬─────┬──┬──────┬────┐   y=0 (상단)
    │    │     │  │      │    │
    │ 1  │  2  │  │  2'  │ 1' │   레인: 상단 (y=0~2m)
    │ 선 │  선 │  │  선  │ 선 │
    │    │     │  │      │    │
  6m├────┼─────┼──┼──────┼────┤   y=2.0m ── 레인경계
    │    │     │  │      │    │
    │    │     │  │      │    │   레인: 중단 (y=2~4m)
    │    │     │  │      │    │
    ├────┼─────┼──┼──────┼────┤   y=4.0m ── 레인경계
    │    │     │  │      │    │
    │    │     │  │      │    │   레인: 하단 (y=4~6m)
    │    │     │  │      │    │
    └────┴─────┴──┴──────┴────┘   y=6.0 (하단)
     x=0  x=1.5 x=3 x=5  x=7  x=8.5 x=10
            (팀A)  (중앙) (팀B)

  존 구분 (팀A 기준):
    후방 (x=0~1.5m)  │  중간 (x=1.5~3.0m)  │  전방 (x=3.0~5.0m)
    ← 1선 →         ← 2선 →              ← 3선(전진금지) →
  ※ 중앙선(x=5.0m) 넘기 불가 — HADO 규칙

  측정 기준점 (위치오차 측정용 10개):
    P1:(0.0, 0.0)  P2:(5.0, 0.0)  P3:(10.0, 0.0)
    P4:(0.0, 3.0)  P5:(5.0, 3.0)  P6:(10.0, 3.0)
    P7:(0.0, 6.0)  P8:(5.0, 6.0)  P9:(10.0, 6.0)
    P10:(5.0, 1.5)  ← 추가 검증점
```

---

## 현장 테스트 절차 (순서대로 따라하기)

### STEP 0 — 도착 직후 (5분)

```bash
# Pi4 전원 ON → SSH 접속 or 모니터 연결
cd ~/hado-smart-court-iot
source ~/hado_venv/bin/activate

# 7항목 자동 점검
./run.sh preflight
```

**출력 확인**: 모든 항목 ✅ 이어야 함. ❌ 있으면 아래 표 참조.

| ❌ 항목 | 즉각 처치 |
|--------|----------|
| ultralytics/cv2/numpy | `pip install -r requirements.txt` |
| NCNN 모델 없음 | Mac에서 `scp -r yolov8n-pose_ncnn_model pi@<IP>:~/hado-smart-court-iot/` |
| espeak-ng 없음 | `sudo apt install espeak-ng` |
| calibration.json 없음 | Step 1 캘리브레이션 먼저 진행 |
| 카메라 없음 | `ls /dev/video*` → `--source 번호` 지정 |

---

### STEP 1 — 캘리브레이션 (8분)

코트 4 코너에 체커보드 마커(또는 A4 색지) 부착 후:

```bash
./run.sh calibrate
```

**조작**:
1. 영상 창이 뜨면 `SPACE` → 코트가 선명히 보이는 순간 눌러 캡처
2. 마우스로 4점 클릭: **좌상→우상→우하→좌하** 순서
3. `ENTER` → 호모그래피 저장

**합격 기준**: `MEAN ROUND-TRIP ERROR: X.X mm` → **50 mm 이하**

기록: `캘리브레이션 오차 = ___ mm` → TECH_REPORT §5, w5_measurements.md

---

### STEP 2 — Level 1 라이브 확인 (5분)

```bash
./run.sh main --level 1
```

**확인 항목**:
- [ ] 선수 bbox가 안정적으로 그려짐
- [ ] player_id (숫자)가 움직임 중에 유지됨 (잠깐 놓쳐도 15프레임 내 재획득)
- [ ] 버드아이뷰 창에 코트 좌표가 찍힘
- [ ] 좌상단 FPS 표시 → **10 fps 이상** (NCNN이면 15+ 목표)

**종료**: `ESC`

---

### STEP 3 — Level 2 전술 엔진 데모 (10분)

```bash
./run.sh main --level 2 --voice
```

**확인 항목**:
- [ ] 선수 옆에 화살표가 표시됨
- [ ] 화살표 색: HIGH=빨강, MID=노랑, BASE=초록
- [ ] 화면 상단에 규칙 레이블 (R1~R6) 표시
- [ ] 한국어 TTS 음성 출력됨 ("측면 회피", "거리 확보" 등)
- [ ] 선수 움직임에 따라 화살표 방향 변경됨

**테스트 시나리오 (직접 이동하며)**:

| 동작 | 기대 규칙 | 기대 음성 |
|------|----------|----------|
| 팀원에게 1m 이내 접근 | R1 | "거리 확보" |
| 중앙선(x=5m)에 붙어서 이동 | R3 | "측면 회피" |
| 한쪽 레인에 모두 집중 | R6 | "위쪽/아래쪽 레인 커버" |
| 상대 진영 중간에 공간 생성 | R4 | "공격 전진" |

**조작 키**:
- `r` : CSV 위치 녹화 시작/중지 (위치오차 측정용)
- `t` : 팀 배정 초기화
- `ESC` : 종료

---

### STEP 4 — 동작 인식 데모 (5분)

```bash
./run.sh action_live
```

**7동작 직접 시연**:

| 동작 | 시연 방법 |
|------|----------|
| charge | 양팔 앞으로 모아서 충전 자세 |
| shoot | 한 팔 앞으로 쏘는 동작 |
| shield | 양팔 교차해서 방어 자세 |
| dodge left | 몸 왼쪽으로 기울이기 |
| dodge right | 몸 오른쪽으로 기울이기 |
| crouch | 몸 숙이기 |
| ready | 기본 서있는 자세 |

**확인 항목**:
- [ ] 스켈레톤 17점 키포인트 overlay 표시됨
- [ ] 상단 confidence bar 표시됨
- [ ] action history 로그 갱신됨
- [ ] 정확도: 각 동작별 1~2번 시연 시 80%+ 예상

---

### STEP 5 — W5 성능 측정 (20분)

#### 5-A. FPS + RAM + CPU 온도 자동 측정

```bash
./run.sh w5_measure
# → data/w5_measurements.md 자동 생성
```

실패 시 수동 대체:

```bash
# NCNN FPS (실측 핵심)
./run.sh bench --frames 200 --imgsz 320 --model yolov8n-pose_ncnn_model

# ONNX FPS (비교용)
./run.sh bench --frames 200 --imgsz 320 --model yolov8n-pose.onnx
```

**기록할 값**:

| 측정 항목 | 결과 | 목표 | 비고 |
|----------|------|------|------|
| FPS (NCNN, threaded) | ____ | ≥15 | Q5 답변값 |
| FPS (ONNX) | ____ | ≥6  | 비교 기준 |
| 피크 RAM | ____MB | <1500 | Q5 답변값 |
| CPU 온도 (5분) | ____°C | <70 | Q9 답변값 |
| TTS 지연 | ____ms | <500 | Q4 답변값 |

#### 5-B. 위치 오차 측정 (10점 마커법)

```bash
# 1) Level 1 실행 후 'r' 키로 CSV 녹화 시작
./run.sh main --level 1
# 마커 위치 P1~P10을 각 3~5초씩 차례로 서기
# 'r' 키로 녹화 중지 → ESC 종료

# 2) RMS 오차 계산 (아래 좌표는 P1~P10 기준)
./run.sh measure-error \
    --csv data/position_logs/positions_XXXXXX.csv \
    --ref 0.0,0.0 5.0,0.0 10.0,0.0 \
          0.0,3.0 5.0,3.0 10.0,3.0 \
          0.0,6.0 5.0,6.0 10.0,6.0 \
          5.0,1.5
```

**기록**: `위치 오차 RMS = ____ cm` → Q6 답변값. 목표 < 10 cm.

---

### STEP 6 — 데모 영상 녹화 (Pi4 카메라 실제 영상)

```bash
# 90초 최종 데모 영상 (Level 2 + 음성)
./run.sh main --level 2 --voice --headless --record data/demo.mp4 --max-frames 2700

# 동작 인식 데모 영상 (60초)
./run.sh action --threaded --record data/action_demo_field.mp4 --max-frames 1800
```

> **주의**: `demo.py`의 PatternPlayer 합성 데이터로 녹화하지 말 것.
> 반드시 Pi4 카메라 실제 영상 (`./run.sh main --record`)으로 교체.

---

## 측정 결과 → 어디에 기입하는가

```
W5 측정 완료
     │
     ├──▶ data/w5_measurements.md        (자동 생성 or 수동 기입)
     │
     ├──▶ docs/QA_PREP.md
     │         Q5: FPS(NCNN/ONNX), RAM
     │         Q6: 위치오차 RMS
     │         Q9: CPU 온도
     │
     └──▶ docs/TECH_REPORT_*.md §5
               표 [TODO] 칸 전부 실측값으로 교체
```

---

## 트러블슈팅

| 문제 | 원인 | 즉각 해결책 |
|------|------|------------|
| 카메라 안 잡힘 | 번호 불일치 | `ls /dev/video*` → `--source 2` 시도 |
| FPS < 10 | NCNN 미적용 | `./run.sh preflight` → 모델 재확인 |
| FPS < 10 지속 | 해상도 문제 | `--imgsz 256` 으로 낮추기 |
| 한글 깨짐 | 폰트 없음 | `sudo apt install fonts-nanum` |
| TTS 없음 | espeak 미설치 | `sudo apt install espeak-ng` |
| 팀 배정 이상 | 초기화 필요 | 실행 중 `t` 키 |
| 오차 > 50mm | 마커 흔들림 | 마커 재부착 + 재캘리브레이션 |
| SSH 화면 없음 | 헤드리스 | `--headless --record` 옵션 사용 |

---

## 발표 내용과 현재 코드 일치 여부 체크

| 중간발표에서 말한 것 | 구현 파일 | 상태 |
|-------------------|----------|------|
| YOLOv8n-pose, imgsz=320 | `src/detector.py`, `src/main.py` | ✅ |
| IoU Tracker, max_lost=15 | `src/tracker.py` | ✅ |
| foot point 기반 homography | `src/homography.py` | ✅ |
| Bird-eye view 실시간 | `src/visualizer.py` | ✅ |
| R1 Spacing | `src/tactic_engine.py` | ✅ |
| R2 Coverage | `src/tactic_engine.py` | ✅ |
| R3 Counter (2.5m) | `src/tactic_engine.py` | ✅ |
| R4 Gap Attack (2.0m 공간) | `src/tactic_engine.py` | ✅ |
| R5 Backline | `src/tactic_engine.py` | ✅ |
| R6 Lane Cover | `src/tactic_engine.py` | ✅ |
| 한국어 TTS (pyttsx3/espeak) | `src/guide.py` | ✅ |
| 7동작 분류 (charge/shoot/…) | `src/pose.py`, `src/action_demo.py` | ✅ |
| NCNN 자동 선택 | `src/main.py` `_resolve_model_path()` | ✅ |
| Pi4 threaded capture | `src/camera.py` | ✅ |
| Pi→MacBook 아키텍처 (슬라이드) | 시연 = Pi 단독 모드 | ✅ (독립 실행) |

---

## 핵심 명령어 요약 (현장 빠른 참조)

```bash
# 사전 점검
./run.sh preflight

# 캘리브레이션
./run.sh calibrate

# Level 1 (추적 + 버드아이뷰)
./run.sh main --level 1

# Level 2 (전술 엔진 + 음성)
./run.sh main --level 2 --voice

# 동작 인식 데모
./run.sh action_live

# W5 자동 측정
./run.sh w5_measure

# FPS 수동 측정 (NCNN)
./run.sh bench --frames 200 --imgsz 320 --model yolov8n-pose_ncnn_model

# 90초 데모 녹화
./run.sh main --level 2 --voice --headless --record data/demo.mp4 --max-frames 2700
```

---

*마지막 업데이트: 2026-06-13. W5 테스트 전날 준비.*
