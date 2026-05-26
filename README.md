# HADO Smart Court IoT

AI 기반 HADO 선수 실시간 위치 추적 + 전술 가이드 시스템.

> 박진우 (JINU) · piaojinu@hufs.ac.kr · HUFS IoT Spring 2026

---

## 🎯 무엇을 하는 시스템인가

- 카메라 1대로 HADO 코트(6m × 2.66m) 위의 선수를 실시간 감지
- 픽셀 좌표를 실제 코트 좌표(meter)로 변환 (Homography)
- Bird-eye view(코트 탑뷰) 위에 선수 위치/궤적 실시간 표시 **← Level 1**
- 5가지 전술 규칙으로 추천 위치 화살표 + 한국어 음성 안내 **← Level 2**

## 📦 하드웨어

- Raspberry Pi 4 (4GB / 8GB)
- Pi Camera v2 또는 USB 웹캠 (Logitech C920 등)
- (선택) Raspberry Pi AI Kit (Hailo-8L) — fps 30+ 달성
- (Level 2 음성용) 소형 스피커 또는 블루투스 이어폰

## 🚀 빠른 시작

### 1. 환경 설치

```bash
python3 -m venv hado_venv
source hado_venv/bin/activate
pip install -r requirements.txt
```

### 2. 카메라 캘리브레이션 (최초 1회)

```bash
python -m src.calibrate
```

화면 안내에 따라 SPACE → 4점 시계방향 클릭 → ENTER.

### 3. 실행

```bash
./run.sh main                          # Level 1 (위치 추적)
./run.sh main --level 2                # Level 2 (가이드 포함)
./run.sh main --level 2 --voice        # + 음성 안내
./run.sh main --headless --record demo.mp4   # SSH 환경: 영상 저장
```

키 단축키:
- `ESC` — 종료
- `s` — 스냅샷  ·  `r` — CSV 녹화  ·  `c` — 궤적 초기화
- `v` — 음성 토글  ·  `t` — 팀 배정 초기화

### 4. FPS 벤치마크

```bash
./run.sh bench --frames 200 --imgsz 320
```

### 5. 테스트

```bash
./run.sh test
```

## 🗂 디렉토리 구조

```
hado-smart-court-iot/
├── README.md
├── requirements.txt
├── run.sh
├── config/
│   ├── court_config.yaml
│   └── calibration.json        # (자동 생성)
├── src/
│   ├── camera.py               # 카메라 추상화
│   ├── calibrate.py            # 4점 클릭 캘리브레이션
│   ├── homography.py           # 좌표 변환
│   ├── detector.py             # YOLOv8 감지
│   ├── tracker.py              # IoU 트래커
│   ├── visualizer.py           # Bird-eye view
│   ├── tactic_engine.py        # Level 2 — 규칙 5개
│   ├── guide.py                # 화살표 + TTS
│   ├── benchmark.py            # FPS 측정
│   └── main.py                 # 통합 실행
├── tests/
│   └── test_homography.py
└── data/
    ├── birdeye_preview.png
    ├── guide_preview.png
    └── position_logs/
```

## 🧠 Level 2 — 5가지 전술 규칙

| 규칙 | 이름 | 발동 조건 | 권고 |
|------|------|----------|------|
| R1 | Spacing | 팀원 < 1.0m | 분산 |
| R2 | Coverage | 팀 좌우 쏠림 (spread < 0.8m) | 빈쪽 커버 |
| R3 | Counter | 정면 적 < 2.5m | 측면 회피 |
| R4 | Gap Attack | 적 간 갭 > 2.0m | 전진 |
| R5 | Backline | 적 2명+ 진영 깊이 침투 | 후방 수비 |

## 🛠 트러블슈팅

| 증상 | 해결 |
|------|------|
| `cv2.imshow` 안 뜸 (SSH) | `--headless --record out.mp4` 사용 |
| `picamera2` 임포트 실패 | `pip install picamera2`, Bookworm OS 필요 |
| FPS 10 미만 | `--imgsz 256` 또는 AI Kit 사용 |
| 음성 안 나옴 | `apt install espeak`, 또는 `--voice` 제거 |
| 캘리브레이션 부정확 | 코너 마커 재고정 후 다시 클릭 |

## 📅 개발 일정

- **Week 1** (5/14–20): 환경 + 캘리브레이션 ✅
- **Week 2** (5/21–27): YOLOv8 + 트래킹 ✅
- **Week 3** (5/28–6/3): Bird-eye view + 1차 발표 🚧
- **Week 4** (6/4–10): 전술 분석 (코드 준비됨, 튜닝 필요)
- **Week 5** (6/11–17): 실환경 테스트
- **Week 6** (6/18–22): 최종 발표

## 📝 라이선스

학술 프로젝트용. HADO는 meleap Inc.의 상표.
