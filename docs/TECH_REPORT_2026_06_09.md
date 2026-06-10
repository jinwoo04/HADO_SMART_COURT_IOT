# HADO Smart Court IoT — 기술 보고서
**작성일**: 2026-06-09  
**작성자**: 박진우 (IoT Term Project, HUFS 컴퓨터공학과)  
**팀원**: 정준혁

---

## 1. 프로젝트 개요

HADO는 실내에서 플레이하는 AR 기반 체육 e스포츠로, 10.0 × 6.0 m 코트에서 3 vs 3으로 진행된다. 이 프로젝트는 단일 Raspberry Pi 4 + 고정 카메라만으로 선수 실시간 추적, 전술 분석, 음성 가이드를 구현하는 오프라인 IoT 시스템이다.

### 시스템 목표

| 지표 | 목표 | 하드 한계 |
|------|------|-----------|
| 엔드-투-엔드 FPS | ≥15 | 10 |
| 피크 RAM | <1.5 GB | 2.5 GB |
| CPU 온도 | <70 °C | 80 °C |
| 위치 오차 | <10 cm | 25 cm |

---

## 2. 전체 아키텍처

```
[카메라 (Mokose HDMI / USB 웹캠)]
        ↓ MJPEG (640×480)
[Camera 클래스] ← 스레드 캡처 (최신 프레임 유지)
        ↓
[PersonDetector] — YOLOv8n-pose NCNN/ONNX (imgsz=320)
        ↓ Detection (bbox + 17 keypoints)
[IoU Tracker] → Track ID 유지
        ↓
[Homography] → 픽셀 좌표 → 코트 좌표 (m)
        ↓
[TacticEngine] → TacticAdvice (규칙 R1~R6)
        ↓
[VoiceGuide] → pyttsx3 한국어 TTS (오프라인)
        ↓
[Visualizer] → 버드아이뷰 + 화살표 오버레이
```

---

## 3. 구현 모듈 상세

### 3.1 Camera (`src/camera.py`)

**Pi Camera (picamera2) / USB 웹캠 자동 fallback** + 공통 인터페이스.

핵심 최적화:
- `CAP_PROP_BUFFERSIZE=1`: OpenCV 내부 버퍼를 1프레임으로 제한 → 지연 누적 방지
- `threaded=True` 모드: 백그라운드 스레드가 항상 최신 프레임 유지. 추론 루프가 `read()` 시 즉시 반환 → 카메라 I/O 대기 시간 제거

```python
cam = Camera(source=0, width=640, height=480, fps=30, threaded=True)
cam.open()
ok, frame = cam.read()   # 항상 최신 프레임, 블로킹 없음
```

### 3.2 PersonDetector (`src/detector.py`)

YOLOv8n-pose 기반 인물 감지. 모델 형식 우선순위:

| 형식 | 속도 (Pi4) | 비고 |
|------|-----------|------|
| NCNN | ~15 FPS | ARM 최적화, FP16 |
| ONNX | ~6 FPS | 범용 |
| PyTorch (.pt) | ~3 FPS | 개발/디버그용 |

`Detection` dataclass: `x1, y1, x2, y2, confidence, keypoints(17×3)`

### 3.3 HADO 동작 분류기 (`src/pose.py` — `classify_hado_action()`)

17개 COCO 키포인트로 7가지 HADO 동작 실시간 분류.

**스케일 정규화** (이번 업데이트 핵심):
```
scale = max(어깨폭, 몸통높이 × 0.6, 30px)
```
카메라 2m vs 5m 거리에서도 동일 임계값 적용.

| 동작 | 판정 조건 (scale 정규화) | 우선순위 |
|------|------------------------|---------|
| 슬라이딩 | (무릎y - 어깨y) / bbox_h < 0.45×0.55 | 1위 |
| 차지 준비 | dy/scale < -0.55 && \|dx/scale\| < 0.70 | 2위 |
| 공격 발사 | forward/scale > 0.65 && \|dy/scale\| < 0.50 | 3위 |
| 쉴드 방어 | 팔 벌림 > 55% bbox폭 + 양 손목 팔꿈치 위 | 4위 |
| 회피 좌/우 | (sh_cx - hi_cx) / bbox_w > 0.30 | 5위 |
| 준비 자세 | 이 외 | 6위 |

**팀 방향 인식**: `frame_center_x` 기준으로 팀 A(좌반)는 +x 방향이 전방. 팔을 뒤로 뻗어도 "공격 발사"로 오인하지 않음.

**조끼 색 샘플링** (`sample_vest_hue()`):
- 토르소 bbox (어깨+엉덩이 키포인트) → HSV 변환
- 채도>70, 명도 60~240 필터 → 원형평균 hue 반환
- 역할 cold-start 분류에 활용 (팀원 코드 `server.js` 로직 이식)

### 3.4 전술 엔진 (`src/tactic_engine.py`)

규칙 기반 분석. 선수당 매 프레임 1개 `TacticAdvice` 반환.

| 규칙 | 조건 | 긴급도 |
|------|------|--------|
| R5 | 적 2명+ 자기 진영 깊이 침투 | HIGH |
| R3 | 정면 3m 이내 적 | HIGH |
| R6 *(신규)* | 적 ≥60% 상단/하단 레인 집중 | MID |
| R4 | 적 y-라인 사이 갭 2.5m+ | MID |
| R2 | 팀 y축 쏠림 | MID |
| R1 | 팀원 1.5m 이내 | MID |

**R6 설계 (팀원 server.js Scenario A 이식)**:
코트 y축을 3등분. 적의 60% 이상이 상단(y < H/3) 또는 하단(y > 2H/3)에 집중 시, 해당 레인에 없는 아군에게 커버 이동 지시.

### 3.5 실시간 동작 인식 데모 (`src/action_demo.py`)

```bash
# Mac 웹캠 개발
python -m src.action_demo

# Pi4 최적 (NCNN + 스레드 캡처)
python -m src.action_demo --threaded

# 녹화
python -m src.action_demo --threaded --record demo_action.mp4
```

HUD 구성:
- 상단: 최근 30프레임 동작 히스토리 타임라인
- 우측: 6가지 동작 실시간 점수 막대
- 하단: 확정 동작 레이블 (6프레임 최다 투표 smoothing) + 신뢰도 바

---

## 4. 테스트 현황

```
전체: 239개 통과 (0 실패)
  test_tactic_engine.py: 52개 (R1~R6 규칙 + R6 신규 6개)
  test_tracker.py:       26개
  test_guide.py:         27개 (_RULE_KO 커버리지 3개 추가)
  test_pose.py:          21개 (쉴드 3개 + draw_skeleton 3개 포함)
  test_movement_model.py:20개 (coverage, intent필터 3개 추가)
  test_detector.py:      15개
  test_visualizer.py:    17개 (recording, distinct색상, 빈트랙 추가)
  test_measure_w5.py:    13개 (W5 실측 헬퍼)
  test_camera.py:        16개 (backend, threaded 버그수정 회귀 3개 추가)
  test_vest.py:          13개 (draw_vest_label 3개 추가)
  test_benchmark.py:     11개 (benchmark() 반환 dict 5개 추가)
  test_homography.py:     8개 (존 라인·마진·배치 검증 추가)
```

## 4.1 새벽 코드 리뷰 수정 사항 (2026-06-09)

| 파일 | 버그 | 심각도 | 수정 내용 |
|------|------|--------|-----------|
| `camera.py` | `threaded=True` 시 첫 `read()` → `(False, zeros)` 반환 → 메인 루프 즉시 종료 | **High** | `(True, zeros)` 반환으로 변경 |
| `action_demo.py` | 모듈 docstring "6동작" (실제 7동작) | Low | 7동작 + 차지 준비 항목 추가 |
| `action_demo.py` | `bar_w = (w-200)*conf` — width < 200 시 음수 | Low | `max(0, ...)` guard 추가 |
| `action_demo.py` | 점수 막대 배경 rect가 "ready" 행 포함해 1칸 과대 | Low | `visible_count` 계산으로 정확히 처리 |
| `pose.py` | "HADO 6동작 분류기" 주석 오기 | Low | "7동작"으로 정정 |

**스모크 테스트**: `python -m src.action_demo --max-frames 5 --headless` → 5프레임 36.5 FPS 정상 종료

---

## 5. 실측 데이터 (W5 예정)

> 아래 항목은 Pi4 실기 테스트 후 채울 것 (2026-06-11~17)

| 항목 | 측정값 | 목표 |
|------|--------|------|
| 평균 FPS (NCNN, threaded) | [TODO] | ≥15 |
| 평균 FPS (ONNX) | [TODO] | ≥6 |
| 피크 RAM (NCNN 모드) | [TODO] | <1.5 GB |
| CPU 온도 (5분 실행) | [TODO] | <70 °C |
| 위치 오차 RMS | [TODO] | <10 cm |
| TTS 지연 (한국어) | [TODO] | <500 ms |

---

## 6. Pi4 배포 체크리스트

```bash
# 1. venv 활성화
source ~/hado_venv/bin/activate

# 2. espeak-ng 설치 (한국어 TTS)
sudo apt install espeak-ng

# 3. NCNN 모델 준비 (Mac에서 export 후 복사)
# Mac에서:
python HADO_iot/pi/export_ncnn.py
scp -r yolov8n-pose_ncnn_model pi@<PI_IP>:~/hado-smart-court-iot/

# 4. 실행 (Pi4 최적)
python -m src.action_demo --threaded
```

---

## 7. 남은 과제

| 주차 | 과제 |
|------|------|
| W5 (6/11-17) | Pi4 실측, QA_PREP.md 채우기, 온코트 테스트 |
| W6 (6/18-22) | Final report §5 실측 데이터 기입, Q&A 영어 리허설 ×5 |
| 최종 제출 | 2026-06-22 |

---

---

## 8. 참고 자료

### 관련 오픈소스 구현체

| 이름 | 관련성 | URL |
|------|--------|-----|
| YoloV8-ncnn-Raspberry-Pi-4 (Qengineering) | Pi4 NCNN 배포 벤치마크 — 본 시스템 FPS 추정 근거 | https://github.com/Qengineering/YoloV8-ncnn-Raspberry-Pi-4 |
| Football-Analysis-using-CV-YOLOv8 | YOLOv8 + 호모그래피 버드아이뷰 → 본 시스템 좌표 변환과 동일 접근 | https://github.com/Rijo-1/Football-Analysis-using-Computer-Vision-with-Yolov8-OpenCV |
| Football-Players-Tracking (Darkmyter) | YOLOv8 + ByteTrack 멀티 선수 추적 — 향후 IoU→ByteTrack 업그레이드 참조 | https://github.com/Darkmyter/Football-Players-Tracking |
| ehpi_action_recognition | 스켈레톤 기반 경량 동작 인식, 20–60 FPS @ 엣지 기기 — 본 7동작 분류기 설계 참조 | https://github.com/noboevbo/ehpi_action_recognition |
| Ultralytics NCNN Export | 공식 YOLOv8 → NCNN FP16 변환 파이프라인 문서 | https://docs.ultralytics.com/integrations/ncnn |

### 관련 논문

| 논문 | 관련성 |
|------|--------|
| "AI-Driven Soccer Analysis" (ArXiv 2025) | 단일 카메라 3D 선수 재구성 + 대형 감지 — 호모그래피 접근 타당성 검증 |
| "Soccer Vision Challenges" (ArXiv 2020) | 폐색·카메라 캘리브레이션 문제 체계화 — 본 시스템 IoU tracker 한계와 연계 |

*이 문서는 CLAUDE.md 기준 자동 생성되었습니다.*
