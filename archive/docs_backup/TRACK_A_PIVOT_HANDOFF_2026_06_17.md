# Track A 방향 전환 인계서
**작성일**: 2026-06-17  
**브랜치**: `presentation/demo-finalization`  
**최종발표**: 2026-06-22  
**작성 목적**: 다른 Claude 세션이 현재 상황을 처음부터 이해하고 개발을 이어갈 수 있도록 하는 완전한 인계서

---

## 1. 프로젝트 기본 정보

**HADO Smart Court IoT**  
- 라즈베리파이 4 + 고정 카메라 1대로 HADO AR 스포츠를 지원하는 **오프라인 임베디드 시스템**
- Python 3.11 단일 언어, OpenCV + YOLOv8n-pose + pyttsx3
- 인터넷 연결 없음, 클라우드 없음, 단일 기기에서 완결
- 로컬 경로: `/Users/jinu/iot project/hado-smart-court-iot`
- GitHub: `https://github.com/jinwoo04/HADO_SMART_COURT_IOT.git`

**HADO 코트**: 10.0 × 6.0 m, 3 vs 3, AR 헤드셋 착용, 가상의 공을 쏘거나 막는 게임

---

## 2. 중간발표 때 발표한 내용 (2026-06-08/15)

중간발표에서 Track A로 발표한 내용은 **코트 전체를 내려다보는 시스템**이었다.

### 발표 기준 파이프라인 (3단계)

```
카메라 (코트 상단 고정) → YOLOv8n-pose → IoU 트래커 → Homography
    ↓
Level 1: 버드아이뷰 (선수 위치 실시간 표시)
Level 2: 전술 엔진 R1~R6 → 화살표 + 음성 안내
Level 3: 7동작 분류 (charge/shoot/shield/dodge/crouch/ready)
```

### 발표에서 강조한 핵심 기술들

| 내용 | 파일 |
|------|------|
| YOLOv8n-pose로 bbox + 17 keypoints 추출 | `src/detector.py` |
| IoU 트래커 (선수 ID 유지) | `src/tracker.py` |
| Homography — 픽셀 → 코트 미터 변환 | `src/homography.py` |
| 버드아이뷰 렌더링 | `src/visualizer.py` |
| 전술 규칙 엔진 R1~R6 | `src/tactic_engine.py` |
| 화살표 + 한국어 TTS | `src/guide.py` |
| 7동작 분류 (스케일 정규화) | `src/pose.py` |
| 라이브 동작 인식 데모 | `src/action_demo.py` |

발표 발표 대본: `docs/FINAL_PRESENTATION_SCRIPT_EN.md`  
발표 범위 브리프: `docs/TRACK_A_PRESENTATION_SCOPE_BRIEF_2026_06_10.md`

---

## 3. 방향 전환 — 무엇이 왜 바뀌었나 (2026-06-17)

### 변경 결정

> **"선수들이 코트에서 어디에 위치하는지를 알려주는 내용은 빼고,  
> 선수의 관절값을 17개의 key point로 불러오고  
> 그 선수가 취하는 동작이 어떤 동작인지를 파악해서  
> 그 다음 동작을 추천해주는 형태로 가자"**  
> — 박진우 (2026-06-17)

### 핵심 차이

| 항목 | 중간발표 (이전) | 최종발표 (현재) |
|------|----------------|----------------|
| 카메라 위치 | 코트 상단 고정 필수 | 어느 각도나 가능 |
| 주요 출력 | 버드아이뷰 + 전술 화살표 | 동작 분류 + 다음 동작 추천 |
| 핵심 입력 | 코트 좌표 (x, y 미터) | 17 키포인트 상대 비율 |
| 전술 기준 | 코트 공간 위치 기반 규칙 | 동작 시퀀스 기반 전환 테이블 |
| 의존 모듈 제거 | — | homography, tactic_engine, guide, visualizer (bird-eye) |
| 신규 기능 | — | `recommend_next_action()`, `draw_keypoint_ids()` |

### 방향 전환의 논리적 근거

- 기존 시스템은 카메라가 코트 상단 특정 위치에 고정되어야 calibration.json이 유효함
- 동작 인식 기반은 카메라 거리·각도에 무관하게 동작 (스케일 정규화)
- "지금 어디 있는가" → "지금 무슨 동작인가, 다음엔 무엇을 해야 하는가" 로 피드백의 수준이 올라감
- 개인 선수 훈련 도구로도 활용 가능해짐 (1인 + 일반 카메라)

---

## 4. 현재 구현 상태 (2026-06-17 기준)

### 4-1. 새로 추가된 것들

#### `src/pose.py`에 추가됨

```python
# 데이터클래스
@dataclass
class NextActionRec:
    action: str       # ACTION_KO 키 (예: "shoot")
    reason: str       # 한글 설명 (예: "차지 완료 → 발사")
    priority: float   # 0.0~1.0

# 전환 테이블 (현재 동작 → 다음 추천 동작)
NEXT_ACTION_RECS: dict[str, list[tuple[str, str, float]]] = {
    "charge":  [("shoot",   "차지 완료 → 발사",      1.0),
                ("dodge_l", "발사 포기 → 좌측 회피", 0.3)],
    "shoot":   [("dodge_r", "발사 후 우측 회피",     0.8),
                ("dodge_l", "발사 후 좌측 회피",     0.7),
                ("shield",  "발사 후 방어 준비",      0.4)],
    "shield":  [("charge",  "방어 해제 → 역공 차지", 0.9), ...],
    "dodge_l": [("charge",  "회피 후 역공 차지",     0.8), ...],
    "dodge_r": [("charge",  "회피 후 역공 차지",     0.8), ...],
    "crouch":  [("charge",  "일어나서 차지 시작",    0.7), ...],
    "ready":   [("charge",  "차지 시작",             0.7), ...],
}

def recommend_next_action(
    current_action: str,
    history: list[str] | None = None,
) -> list[NextActionRec]:
    """현재 동작 + 히스토리 → 다음 동작 추천 (최대 2개, 우선순위 내림차순).
    같은 동작 4프레임 이상 지속 시 전환 추천 가중치 +20%.
    """

def draw_keypoint_ids(img, det, conf_min=0.25) -> None:
    """스켈레톤 위에 키포인트 번호 0~16 표시 (발표용 --show-kp-ids 플래그)."""
```

#### `src/action_demo.py`에 추가됨

```python
_PANEL_H = 130   # 90px → 130px (추천 영역 확장)

# 하단 패널 레이아웃:
#   y0+12 : ↑  차지 준비                   FPS 29.3
#   y0+50 : [신뢰도 바━━━━━━━━░] 87%
#   ────────────── separator ──────────────
#   y0+74 : 다음 추천  ⚡ 공격 발사   차지 완료 → 발사
#   y0+102:            ← 회피 좌     발사 포기 → 좌측 회피

# 새 플래그
--show-kp-ids    # 키포인트 번호 화면에 표시 (발표용)
```

### 4-2. 실행 명령

```bash
source hado_venv/bin/activate

# 기본 동작 인식 (외부 카메라 source=1)
python -m src.action_demo --source 1

# 17 키포인트 번호 표시 (발표 시연용)
python -m src.action_demo --source 1 --show-kp-ids

# Pi4에서 (NCNN 자동 선택 + threaded 캡처)
./run.sh action_live --show-kp-ids

# 녹화
python -m src.action_demo --source 1 --show-kp-ids --record data/demo_track_a_new.mp4
```

### 4-3. 테스트

```bash
python -m pytest tests/ -q   # 272개 통과
python -m pytest tests/test_pose.py -v   # 36개 (새 테스트 포함)
```

새로 추가된 테스트 클래스:
- `TestRecommendNextAction` (12개): 추천 우선순위, 히스토리 가중치, 유효성 등
- `TestDrawKeypointIds` (3개): 크래시 없음, 픽셀 변화 확인

---

## 5. 파일 구조 — Track A 관련 핵심 파일만

```
hado-smart-court-iot/
├── src/
│   ├── pose.py           ★ 핵심: 17kp 분류 + 추천 (완전 독립 모듈)
│   ├── action_demo.py    ★ 핵심: 라이브 데모 메인 루프
│   ├── detector.py       ★ YOLOv8n-pose 래퍼 (Detection 데이터클래스)
│   ├── camera.py         ★ picamera2/OpenCV 자동 전환, threaded 캡처
│   ├── demo_pose.py      (버드아이뷰 + 스켈레톤 병합 데모 — 이전 스타일)
│   ├── benchmark.py      (FPS/RAM 측정)
│   └── measure_w5.py     (W5 실측 헬퍼)
├── tests/
│   └── test_pose.py      ★ 36개 테스트 (분류 + 추천 + 시각화)
├── config/
│   └── court_config.yaml (임계값 설정 — pose 분류 임계값은 pose.py 내 상수)
├── docs/
│   ├── FINAL_PRESENTATION_SCRIPT_EN.md   (이전 발표 대본 — 업데이트 필요)
│   ├── QA_PREP.md                        (Q&A 준비 — 업데이트 필요)
│   └── TRACK_A_PIVOT_HANDOFF_2026_06_17.md  ← 이 파일
└── data/
    └── w5_measurements.md  (Mac 사전 측정 완료, Pi4 [TODO])
```

---

## 6. 최종발표까지 남은 작업 (2026-06-22)

### 필수 (발표에 직접 영향)

#### 6-1. 발표 대본 업데이트
`docs/FINAL_PRESENTATION_SCRIPT_EN.md`의 내용이 이전 방향(버드아이뷰 + 전술엔진) 기준이다.
새 방향(17kp → 동작 분류 → 다음 동작 추천)으로 슬라이드 내러티브를 재구성해야 한다.

**새 대본 흐름 (제안)**:
```
Slide 1 — Title
Slide 2 — Problem: 선수가 실시간으로 "지금 뭘 해야 하는지" 모른다
Slide 3 — System: 카메라 → YOLOv8n-pose → 17 keypoints → 동작 분류 → 다음 추천
Slide 4 — 17 Keypoints: COCO 17kp 설명, 어떻게 추출하는지
Slide 5 — Action Classification: 7동작 + 스케일 정규화 + 팀방향 인식
Slide 6 — Next Action Recommendation: NEXT_ACTION_RECS 전환 테이블 + 히스토리 가중치
Slide 7 — Pi4 Optimization: NCNN, threaded 캡처, imgsz=320
Slide 8 — Demo: 라이브 or 녹화 영상
Slide 9 — Validation: 272 tests, 동작별 precision
Slide 10 — Results & Next Steps
```

#### 6-2. QA_PREP.md 업데이트
기존 Q2~Q9의 일부가 "왜 전술 엔진을 ML 대신 룰 기반으로 했는가" 등 이전 방향 중심.
새 방향에 맞는 예상 질문 추가가 필요하다.

**새로 추가가 필요한 Q&A 예시**:
- "Why transition table instead of ML for next action?"
- "How is the recommendation different from just a lookup table?"
- "Can you combine action rec with court position in the future?"
- "How does the history-based weight boost work?"

#### 6-3. 데모 영상 재생성
현재 `data/demo.mp4`는 시뮬레이션 기반 버드아이뷰 영상이다.
새 방향의 데모 영상(라이브 카메라 → 17kp → 동작 분류 → 추천 패널)이 필요하다.

```bash
# 실제 카메라로 녹화 (연결된 카메라 source=1)
python -m src.action_demo --source 1 --show-kp-ids --record data/demo_track_a_pivot.mp4 --max-frames 1800
# → 60초 @ 30fps
```

#### 6-4. Pi4 실측 데이터 (W5)
`data/w5_measurements.md` B섹션의 [TODO] 슬롯이 비어있다.
Pi4에서 `./run.sh w5_measure` 실행 후 채워야 한다.
- NCNN FPS, ONNX FPS (목표 ≥15)
- RAM (목표 <1.5GB)
- CPU 온도 (목표 <70°C)
- TTS 지연 (목표 <500ms)

---

### 선택 (발표 품질 향상)

#### 6-5. 키포인트 레전드 오버레이 (선택)
`--show-kp-ids` 모드에서 화면 구석에 키포인트 번호 → 부위 이름 레전드를 표시하면
발표 시청자가 17개 keypoint 의미를 바로 이해할 수 있다.

```python
# pose.py에 추가할 수 있는 상수
KP_NAMES = {
    0: "코", 5: "왼어깨", 6: "오른어깨",
    7: "왼팔꿈치", 8: "오른팔꿈치",
    9: "왼손목", 10: "오른손목",
    11: "왼엉덩이", 12: "오른엉덩이",
    13: "왼무릎", 14: "오른무릎",
    15: "왼발목", 16: "오른발목",
}
```

#### 6-6. 동작 시퀀스 시각화 (선택)
현재 히스토리 타임라인은 컬러 바로만 표시된다.
동작명 텍스트를 위에 표시하면 발표 중 청중이 읽기 쉽다.

#### 6-7. 영어 Q&A 리허설 ×5
`docs/QA_PREP.md` 기준으로 소리내어 영어 답변 연습.
업데이트된 내용으로 rehearsal_checklist에 체크 필요.

---

## 7. 개발 이어가는 방법 — Claude에게 주는 지침

### 7-1. 해도 되는 것

```
- src/pose.py         수정 가능 (추천 로직, 분류 임계값, 시각화)
- src/action_demo.py  수정 가능 (UI, 패널 레이아웃, 플래그)
- src/detector.py     수정 가능 (모델 경로, conf 임계값)
- src/camera.py       수정 가능 (해상도, FPS, threaded 설정)
- tests/test_pose.py  테스트 추가 자유롭게
- docs/               문서 업데이트 자유롭게
- data/w5_measurements.md  Pi4 측정값 기입
```

### 7-2. 절대 건드리면 안 되는 것

```
- config/calibration.json   카메라 캘리브레이션 (덮어쓰면 homography 파괴)
- data/movement_data.csv    전문가 어노테이션 원본
- data/movement_data_annotated.csv  동일
- Track B 관련 파일 일체    (research/* 브랜치, tools/, data/track_b_*)
```

### 7-3. 새 기능 추가 전 체크리스트

1. `python -m pytest tests/ -q` 통과 확인
2. 관련 모듈 `python -m src.<module>` standalone 실행 확인
3. 테스트 없이 로직 추가하지 말 것
4. config에 없는 매직 넘버를 코드에 박지 말 것

### 7-4. 커밋 컨벤션

```
feat:     새 기능
fix:      버그 수정
refactor: 동작 변경 없는 구조 개선
test:     테스트만
docs:     문서만
```

커밋 전 반드시: `python -m pytest tests/ -q` 통과

---

## 8. 현재 동작 인식 로직 상세

### 8-1. COCO 17 키포인트 인덱스

```
0:코  1:왼눈  2:오른눈  3:왼귀  4:오른귀
5:왼어깨  6:오른어깨
7:왼팔꿈치  8:오른팔꿈치
9:왼손목  10:오른손목
11:왼엉덩이  12:오른엉덩이
13:왼무릎  14:오른무릎
15:왼발목  16:오른발목
```

### 8-2. 7동작 판정 우선순위 및 조건

| 동작 | 조건 | 우선순위 |
|------|------|---------|
| `crouch` | 어깨~무릎 수직 압축 > 0.55 | 1 (최우선) |
| `charge` | 손목이 어깨보다 scale×0.55 이상 위, 수직 방향 | 2 |
| `shoot`  | 손목이 팀 방향으로 scale×0.65 이상 앞으로 뻗음 | 3 |
| `shield` | 양팔 벌림 > 0.55 + 양 손목 팔꿈치 위 | 4 |
| `dodge_l`| 어깨 중점이 엉덩이 중점보다 왼쪽 치우침 > 0.30 | 5 |
| `dodge_r`| 어깨 중점이 엉덩이 중점보다 오른쪽 치우침 > 0.30 | 6 |
| `ready`  | 위 조건 모두 미충족 | 7 (기본) |

스케일 정규화: `scale = max(어깨폭, 몸통높이×0.6, 30px)`  
팀 방향: `frame_center_x` 기준, 왼쪽 팀은 오른쪽이 전방, 오른쪽 팀은 왼쪽이 전방

### 8-3. 다음 동작 추천 흐름

```python
# action_demo.py 메인 루프 내
result = classify_hado_action(target, frame_center_x=args.width / 2)
if result:
    smooth_buf.append(result.action)
    smoothed = max(set(smooth_buf), key=list(smooth_buf).count)   # 최근 6프레임 최다
    history.append(smoothed)
    last_recs = recommend_next_action(smoothed, list(history))    # 최대 2개 반환

# recommend_next_action 내부:
# 1. NEXT_ACTION_RECS에서 현재 동작에 해당하는 추천 목록 조회
# 2. history[-4:]가 모두 같은 동작이면 priority × 1.2 (전환 강조)
# 3. priority 내림차순 정렬 후 최대 2개 반환
```

---

## 9. 이전 중간발표 내용과의 호환성

기존에 구현된 Level 1~2 코드(버드아이뷰, 전술엔진)는 **삭제하지 않는다**.
`main.py`, `tactic_engine.py`, `visualizer.py`, `guide.py`, `homography.py`는 모두 유지.

이유:
- 기말 최종보고서에 Level 1~2 구현 내용을 포함해야 함
- Track B 연구와 공유되는 코드가 있음
- `./run.sh main --level 2` 로 언제든 이전 시스템 실행 가능

**발표에서 어떻게 이어가느냐**: 발표는 새 방향(동작 인식 + 추천)을 중심으로 하되,
Level 1~2를 "이전에 구현한 더 큰 시스템의 일부"로 자연스럽게 언급하면 된다.

```
"We built a full court tracking system (Level 1~2), and now we're taking it further —
instead of just tracking WHERE players are, we recognize WHAT they're doing
and recommend WHAT to do NEXT."
```

---

## 10. 빠른 현황 체크 명령어

```bash
cd "/Users/jinu/iot project/hado-smart-court-iot"
source hado_venv/bin/activate

# 현재 브랜치 확인
git branch

# 테스트 전체 통과 확인
python -m pytest tests/ -q

# 동작 인식 데모 즉시 실행 (카메라 source=1)
python -m src.action_demo --source 1 --show-kp-ids

# 모듈 standalone 테스트
python -m src.pose        # 카메라 소스로 자세 분석 단독 실행
python -m src.action_demo  # 동작 인식 데모

# W5 측정 (Mac 사전 검증)
python -m src.measure_w5 --source 1 --frames 100
```

---

## 11. 발표까지 남은 날짜별 권장 작업

| 날짜 | 작업 |
|------|------|
| **6/18 (오늘+1)** | 발표 대본 재작성 (새 방향 기준), 데모 영상 녹화 |
| **6/19** | QA_PREP.md 새 방향 Q&A 추가 (4~5개), 영어 리허설 ×2 |
| **6/20** | Pi4 현장 실측 (`./run.sh w5_measure`), w5_measurements.md B섹션 채우기 |
| **6/21** | 영어 리허설 ×3, 최종보고서 §5 실측 데이터 기입 |
| **6/22** | **최종발표** |

---

*이 인계서는 `/Users/jinu/iot project/hado-smart-court-iot/docs/TRACK_A_PIVOT_HANDOFF_2026_06_17.md`에 저장됨*  
*GitHub 브랜치: `presentation/demo-finalization` (최신 커밋: `e658bc8`)*
