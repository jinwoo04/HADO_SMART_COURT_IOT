# HADO Smart Court IoT

AI 기반 HADO 선수 실시간 위치 추적 + 전술 가이드 + 동작 인식 시스템.

> 박진우 (JINU) · piaojinu@hufs.ac.kr · HUFS IoT Spring 2026

---

## 🎯 무엇을 하는 시스템인가

- 카메라 1대로 HADO 코트(10m × 6m) 위의 선수를 실시간 감지
- 픽셀 좌표를 실제 코트 좌표(meter)로 변환 (Homography)
- Bird-eye view(코트 탑뷰) 위에 선수 위치/궤적 실시간 표시 **← Level 1**
- 6가지 전술 규칙으로 추천 위치 화살표 + 한국어 음성 안내 **← Level 2**
- YOLOv8n-pose 키포인트로 7가지 HADO 동작 실시간 분류 **← Level 3 (데모)**

## 📦 하드웨어

- Raspberry Pi 4 (4GB / 8GB)
- Pi Camera v2 또는 USB 웹캠
- (선택) Raspberry Pi AI Kit (Hailo-8L) — fps 30+ 달성
- (Level 2 음성용) 소형 스피커 또는 블루투스 이어폰

## 🚀 빠른 시작

### 1. 환경 설치

```bash
python3 -m venv hado_venv
source hado_venv/bin/activate
pip install -r requirements.txt
# Pi4 TTS: sudo apt install espeak-ng
```

### 2. 카메라 캘리브레이션 (최초 1회)

```bash
./run.sh calibrate
```

화면 안내에 따라 SPACE → 4점 시계방향 클릭 → ENTER.

### 3. 실행

```bash
./run.sh main                          # Level 1 (위치 추적)
./run.sh main --level 2                # Level 2 (가이드 포함)
./run.sh main --level 2 --voice        # + 음성 안내
./run.sh main --headless --record demo.mp4   # SSH 환경: 영상 저장

./run.sh action                        # Level 3 — 동작 인식 데모
./run.sh action --threaded             # Pi4 최적 (스레드 캡처)
```

키 단축키 (main):
- `ESC` — 종료
- `s` — 스냅샷  ·  `r` — CSV 녹화  ·  `c` — 궤적 초기화
- `v` — 음성 토글  ·  `t` — 팀 배정 초기화

### 4. FPS 벤치마크 (W5 실측용)

```bash
./run.sh bench --frames 200 --imgsz 320
# → 추론 FPS, Wall-clock FPS, 피크 RAM, CPU 온도(Pi4) 출력
```

### 5. 테스트

```bash
./run.sh test   # 180개 단위 테스트
```

## 🗂 주요 모듈

```
src/
├── camera.py          # 카메라 추상화 (picamera2 / OpenCV, 스레드 캡처)
├── calibrate.py       # 4점 클릭 호모그래피 캘리브레이션
├── homography.py      # 픽셀 ↔ 코트 좌표 변환
├── detector.py        # YOLOv8n-pose 감지 (NCNN / ONNX / PT 자동선택)
├── tracker.py         # IoU 트래커
├── pose.py            # 7동작 분류 + 조끼 hue 샘플링
├── tactic_engine.py   # 전술 규칙 엔진 (R1–R6 + MovementModel 블렌딩)
├── guide.py           # pyttsx3 한국어 TTS + 방향 화살표
├── visualizer.py      # Bird-eye view 렌더링
├── movement_model.py  # 전문가 패턴 기반 이동 예측
├── benchmark.py       # FPS / RAM / CPU 온도 측정
├── action_demo.py     # Level 3 — 실시간 동작 인식 데모
└── main.py            # Level 1/2 통합 실행
```

## 🧠 Level 2 — 6가지 전술 규칙

| 규칙 | 이름 | 발동 조건 | 긴급도 |
|------|------|----------|--------|
| R5 | Backline | 적 2명+ 자기 진영 깊이 침투 | HIGH |
| R3 | Counter | 정면 3m 이내 적 | HIGH |
| R1 | Spacing | 팀원 1.5m 이내 | MID |
| R6 | Lane Cover | 적 ≥60% 상/하단 레인 집중 | MID |
| R4 | Gap Attack | 적 y-라인 갭 > 2.5m | MID |
| R2 | Coverage | 팀 y축 쏠림 < 1.5m | MID |

우선순위: R5 > R3 > R1 > R6 > R4 > R2

## 🤸 Level 3 — 7가지 HADO 동작

| 기호 | 동작 | 판정 조건 |
|------|------|----------|
| ↑ | 차지 준비 | 손목 어깨보다 scale×0.55 이상 위, 수직 방향 |
| ⚡ | 공격 발사 | 손목 팀 전방으로 scale×0.65 이상 뻗음 |
| 🛡 | 쉴드 방어 | 양팔 bbox폭 55%+ 벌림 + 손목 팔꿈치 위 |
| ← | 회피 좌 | 어깨 중점이 엉덩이 대비 왼쪽으로 30%+ |
| → | 회피 우 | 위와 반대 방향 |
| ↓ | 슬라이딩 | 무릎-어깨 수직 간격 < bbox 높이 × 0.45 |
| ● | 준비 자세 | 이 외 모든 경우 |

## 🛠 트러블슈팅

| 증상 | 해결 |
|------|------|
| `cv2.imshow` 안 뜸 (SSH) | `--headless --record out.mp4` 사용 |
| `picamera2` 임포트 실패 | `pip install picamera2`, Bookworm OS 필요 |
| FPS 10 미만 | `--threaded` 추가 또는 `--imgsz 256` |
| 음성 안 나옴 | `sudo apt install espeak-ng` (Pi4) |
| 캘리브레이션 부정확 | 코너 마커 재고정 후 `./run.sh calibrate` |
| action_demo 즉시 종료 | `--threaded` 없이 실행 (첫 프레임 대기 불필요) |

## 📅 개발 일정

- **Week 1** (5/14–20): 환경 + 캘리브레이션 ✅
- **Week 2** (5/21–27): YOLOv8 + 트래킹 ✅
- **Week 3** (5/28–6/3): Bird-eye view + 중간발표 ✅ (발표 6/8)
- **Week 4** (6/4–10): 전술 엔진 R1–R6 + TTS + action_demo ✅
- **Week 5** (6/11–17): Pi4 실환경 테스트 + 성능 측정
- **Week 6** (6/18–22): 최종 보고서 + 데모 영상 + 발표

## 📝 라이선스

학술 프로젝트용. HADO는 meleap Inc.의 상표.
