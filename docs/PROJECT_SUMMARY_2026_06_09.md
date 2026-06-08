# HADO Smart Court IoT — 프로젝트 요약 보고서

**HUFS 컴퓨터공학과 IoT 텀프로젝트 (Spring 2026)**  
**팀**: 박진우 (piaojinu@hufs.ac.kr) · 정준혁  
**최종 제출**: 2026-06-22

---

## 프로젝트 소개

HADO는 AR 기기를 착용하고 실내 코트(10m × 6m)에서 팀으로 플레이하는 체육 e스포츠입니다. 선수들은 팔 동작으로 에너지구를 발사하고 쉴드를 펼쳐 막습니다.

**이 프로젝트는 Raspberry Pi 4 한 대와 일반 카메라만으로 선수 실시간 추적 + 전술 분석 + 음성 가이드를 구현합니다. 클라우드나 인터넷 없이 완전 오프라인 동작.**

---

## 핵심 기능

### Level 1 — 선수 위치 추적
- USB 카메라 → YOLOv8n-pose로 선수 감지
- IoU 트래커로 각 선수 ID 유지
- 호모그래피 변환으로 픽셀 → 코트 좌표(m) 변환
- 버드아이뷰(Bird-Eye View) 실시간 렌더링

### Level 2 — 전술 분석 & 가이드
- 6가지 규칙 기반 전술 분석 (간격 유지, 공격 기회, 레인 커버 등)
- 한국어 TTS 음성 안내 (pyttsx3, 오프라인)
- 방향 화살표 오버레이

### Level 3 — 동작 인식 데모
- 카메라 앞에서 HADO 동작을 취하면 실시간으로 동작을 인식
- 7가지 동작: 차지 준비 / 공격 발사 / 쉴드 방어 / 회피 좌 / 회피 우 / 슬라이딩 / 준비
- 스켈레톤 오버레이 + 동작 히스토리 + 신뢰도 시각화

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| 하드웨어 | Raspberry Pi 4 (4GB), USB 카메라 (Mokose HDMI 캡처 또는 웹캠) |
| AI 모델 | YOLOv8n-pose (NCNN 형식, ARM 최적화) |
| 비전 | OpenCV 4.x, NumPy |
| TTS | pyttsx3 (오프라인, 한국어 Yuna 음성) |
| 언어 | Python 3.11 단일 |

---

## 시스템 성능 (Pi4 목표값)

| 지표 | 목표 | 방법 |
|------|------|------|
| FPS | ≥15 | NCNN + 스레드 캡처 + imgsz=320 |
| RAM | <1.5 GB | 경량 모델 (YOLOv8n) |
| 위치 오차 | <10 cm | 4점 호모그래피 캘리브레이션 |
| TTS 지연 | <0.5 s | 비동기 백그라운드 발화 |

---

## 주요 최적화 포인트

### 1. NCNN 모델 형식
YOLOv8n-pose를 ARM 최적화 NCNN 형식으로 변환(FP16). ONNX 대비 약 3배 빠른 추론 속도를 Pi4 CPU에서 달성.

### 2. 스레드 분리 캡처
카메라 I/O와 AI 추론 루프를 별도 스레드로 분리. 추론이 느릴 때도 카메라 버퍼가 항상 최신 프레임을 유지하여 지연 누적 없음.

### 3. 스케일 정규화 포즈 분류
어깨폭 + 몸통 높이를 기준으로 키포인트 임계값을 정규화. 선수가 카메라 가까이 서든 멀리 서든 동일한 동작 인식 정확도 유지.

### 4. 팀 방향 인식
팀A(왼쪽)와 팀B(오른쪽)의 플레이 방향을 인식하여 "공격 발사" 동작을 팀 전방 방향으로만 판정. 뒤를 보며 팔을 뻗어도 오인식하지 않음.

---

## 어노테이션 데이터

실제 경기 영상("윈터컵 결승 2경기 RGB vs MAJOR")을 수동 어노테이션하여 전문가 이동 패턴 635개를 데이터로 구축. 이를 토대로 MovementModel(이동 예측 모델) 학습.

---

## 프로젝트 구조

```
hado-smart-court-iot/
├── src/
│   ├── camera.py          # 카메라 (picamera2/OpenCV 통합)
│   ├── detector.py        # YOLOv8n-pose 감지
│   ├── tracker.py         # IoU 트래커
│   ├── homography.py      # 좌표 변환
│   ├── pose.py            # 동작 분류 + 조끼 색상 샘플링
│   ├── tactic_engine.py   # 전술 규칙 엔진
│   ├── guide.py           # 한국어 TTS 가이드
│   ├── visualizer.py      # 시각화
│   └── action_demo.py     # 실시간 동작 인식 데모
├── tests/                 # 180개 단위 테스트
├── data/movement_data.csv # 전문가 어노테이션 패턴 635개
└── config/court_config.yaml
```

---

## 실행 방법

```bash
# 환경 활성화
source ~/hado_venv/bin/activate

# 동작 인식 데모 (Pi4, NCNN + 스레드 캡처)
python -m src.action_demo --threaded

# 전술 분석 데모 (시뮬레이션)
python -m src.demo --headless --frames 2700

# 레벨 2 (실카메라 + 음성)
python -m src.main --level 2 --voice
```

---

## 제출 현황

- [x] 소스코드 (`src/`) + 테스트 180개 통과
- [x] README.md
- [x] 최종 보고서 `.docx` (§5 실측 데이터 미기입 — W5 이후)
- [x] 발표 PPT `.pptx`
- [x] 데모 영상 `data/demo.mp4` (90초)
- [x] 백업 영상 `data/demo_5min.mp4` (5분)
- [ ] 실측 데이터 (W5: 2026-06-11~17)
- [ ] Q&A 영어 리허설 ×5

---

*HUFS Computer Engineering — IoT Term Project 2026*
