# HADO Smart Court IoT — W3 세션 작업 보고서

**작성일**: 2026-05-30  
**작성자**: Jinu Park (박진우)  
**저장소**: https://github.com/jinwoo04/HADO_SMART_COURT_IOT  
**대상**: IoT Term Project + 개인 프로젝트(경기영상 분석) 병행 진행

---

## 1. 이번 세션 목표

| 구분 | 목표 |
|------|------|
| 이번 학기 프로젝트 | YOLOv8-pose를 이용한 자세 분석 파이프라인 추가 |
| 개인 프로젝트 | 경기 영상(AR 이펙트 포함) 기반 파인튜닝 데이터 준비 |

---

## 2. 완료된 작업

### 2-1. YOLOv8-pose 파이프라인 (이번 학기 프로젝트)

**추가 파일**: `src/pose.py`, `src/demo_pose.py`  
**수정 파일**: `src/detector.py`, `run.sh`

#### Detection 클래스 확장

```python
@dataclass
class Detection:
    x1, y1, x2, y2: float
    confidence: float
    keypoints: np.ndarray | None  # (17, 3) — 신규 추가
```

YOLOv8-pose 모델 사용 시 COCO 17개 키포인트(x, y, confidence)가 자동으로 채워진다.

#### 자세 분류 로직 (`src/pose.py`)

3가지 자세를 규칙 기반으로 분류:

| 자세 | 조건 | 설명 |
|------|------|------|
| `attack` (공격 자세) | crouch_score > 0.45 | 어깨~발목 수직 간격이 좁음 → 낮은 자세 |
| `shield` (쉴드 준비) | arm_spread > 0.50 | 손목~어깨 수평 거리가 넓음 → 팔 벌림 |
| `neutral` (중립) | 그 외 | 일반 대기 자세 |

```
crouch_score = 1 - (ankle_y - shoulder_y) / (0.60 × bbox_height)
arm_spread   = max(|wrist_x - shoulder_x|) / bbox_width
```

#### demo_pose.py 파이프라인

```
카메라/영상 → YOLOv8-pose 감지 → IoU 트래커 → 호모그래피
    → 코트 좌표 변환 → TacticEngine 전술 분석
    → 카메라 뷰(스켈레톤+자세 라벨) + 버드아이뷰(전술 화살표) 출력
```

실행 방법:
```bash
./run.sh demo_pose --video data/영상.mp4 --imgsz 640 --conf 0.25 --out data/out.mp4
./run.sh pose --video data/영상.mp4 --headless  # 화면 출력 없이 저장
```

#### 실제 경기영상 테스트 결과

영상: `5경기 DINOS vs SSP 블루코트.mp4` (1920×1080, 5884프레임)

| 설정 | 결과 |
|------|------|
| imgsz=320, conf=0.35 | 감지율 ~0명/프레임 |
| imgsz=640, conf=0.25 | 최대 6명/프레임, 키포인트 14-17/17개 유효 |
| Mac CPU 처리속도 | 평균 12.6fps |

자세 통계 (전체 영상):
- neutral: 10,042회 (95%)
- shield: 534회 (5%)
- attack: 15회 (<1%) — 임계값 튜닝 필요

---

### 2-2. 시각화 개선

**수정 파일**: `src/guide.py`, `src/demo.py`, `src/annotate.py`

#### 전술 화살표 개선 (`src/guide.py`)

- 출발점(●) → 화살표 → 목표 X 형태로 이동 방향 명시
- 각 화살표 옆에 한글 규칙 라벨 표시 (팀원 분산 / 코트 커버 / 측면 회피 등)
- HIGH 긴급도 → 주황색 강조원 추가
- 우하단 범례 패널 추가

#### 패턴 기반 시뮬레이션 (`src/demo.py`)

기존 사인파 더미 이동 → `movement_data.csv` 152개 실제 패턴 기반 `PatternPlayer`로 교체:
- 포지션별 이동 패턴 (테크니션 / 디펜더 / 어태커)
- 팀B는 x축 미러링으로 반대 진영 재현

#### 한글 폰트 수정 (`src/annotate.py`)

macOS Sequoia/Sonoma에서 `/Library/Fonts/AppleSDGothicNeo.ttc` 경로로 수정.  
(기존 `AppleGothic.ttf` 경로 변경으로 한글이 `?`로 출력되던 문제 해결)

---

### 2-3. YOLOv8 파인튜닝 준비 (개인 프로젝트)

**추가 파일**: `src/extract_frames.py`, `notebooks/hado_finetune.ipynb`

#### 배경

HADO 경기 영상에는 AR 이펙트(쉴드, 볼, 파티클, HUD)가 포함되어 있어 표준 YOLOv8n으로 감지율이 낮다. 커스텀 파인튜닝으로 이를 해결한다.

#### 프레임 추출 (`src/extract_frames.py`)

16개 경기 영상에서 951장 추출:

| 카테고리 | 장수 | 설명 |
|----------|------|------|
| clean | 626장 | 선수 완전 노출 |
| partial | 227장 | 쉴드/볼 부분 가림 |
| heavy | 98장 | AR 이펙트 강한 가림 |

사용 영상:
- DINOS vs SSP 블루코트/레드코트 (5경기)
- 윈터컵 결승 1·2경기 RGB vs MAJOR
- 윈터컵 4강 3경기 APEX vs MAJOR 블루/레드코트
- 단편 클립 1~10.mp4

```bash
python -m src.extract_frames --total 600
```

#### Colab 학습 노트북 (`notebooks/hado_finetune.ipynb`)

| 단계 | 내용 |
|------|------|
| 데이터 | Roboflow에서 YOLOv8 포맷으로 다운로드 |
| 모델 | yolov8n.pt 사전학습 가중치 |
| 학습 | 60에폭, imgsz=640, batch=16, augmentation 포함 |
| 결과 | best.pt → Google Drive 저장 |
| 소요시간 | Colab T4 기준 약 20~30분 |

파인튜닝 후 적용:
```bash
./run.sh demo_pose --video data/경기영상.mp4 \
  --model hado_yolov8n_v1.pt --imgsz 640 --conf 0.25
```

---

## 3. 프로젝트 범위 구분

```
이번 학기 프로젝트 (제출 2026-06-22)
├── 입력: 일반 카메라 영상 (AR 이펙트 없음)
├── 모델: yolov8n-pose.pt (기존 그대로)
└── 핵심: 위치 + 자세 → 의도 예측 → 전술 조언

개인 프로젝트 (학기 후 진행)
├── 입력: 경기 방송 영상 (AR 이펙트 포함)
├── 모델: yolov8n 파인튜닝 (hado_yolov8n_v1.pt)
└── 핵심: 이펙트 무시하고 선수 감지 → 게임 상태 + 움직임 예측
```

---

## 4. 파인튜닝 다음 단계

1. **Roboflow 라벨링** (수동 작업 필요)
   - `data/frames_for_labeling/` 951장 업로드
   - SAM AI Assist로 bbox 라벨링
   - Export → YOLOv8 포맷

2. **Colab 학습 실행**
   - `notebooks/hado_finetune.ipynb` 실행
   - Roboflow API Key 입력

3. **모델 평가**
   - mAP50 > 0.80 목표
   - heavy 프레임 감지율 확인

---

## 5. 커밋 목록

| 해시 | 내용 |
|------|------|
| `6cc27c0` | feat: YOLOv8-pose 파이프라인 추가 |
| `18b3350` | fix/feat: 한글 폰트 수정 + 전술 화살표 개선 + demo 패턴 시뮬레이션 |
| `f3d7ebd` | feat: YOLOv8 파인튜닝 준비 (프레임 추출 + Colab 노트북) |

---

## 6. 현재 파일 구조 (신규/변경)

```
src/
├── pose.py          # 신규 — 자세 분류, 스켈레톤 시각화
├── demo_pose.py     # 신규 — 실제 영상 통합 파이프라인
├── extract_frames.py # 신규 — 라벨링용 프레임 추출
├── detector.py      # 수정 — keypoints 필드 추가
├── guide.py         # 수정 — 화살표 라벨 + 범례
├── demo.py          # 수정 — PatternPlayer 시뮬레이션
└── annotate.py      # 수정 — macOS 폰트 경로

notebooks/
└── hado_finetune.ipynb  # 신규 — Colab 파인튜닝 노트북

data/frames_for_labeling/  # gitignore — 951장 (Roboflow로 관리)
```
