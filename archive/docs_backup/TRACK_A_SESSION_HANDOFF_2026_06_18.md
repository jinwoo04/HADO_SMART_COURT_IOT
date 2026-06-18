# Track A 세션 인계서 — 2026-06-18 (v2)
**작성일**: 2026-06-18
**최종 갱신**: 2026-06-18 (스코프 축소 결정 반영)
**이전 인계서**: `docs/TRACK_A_PIVOT_HANDOFF_2026_06_17.md`
**최종발표**: 🔴 **2026-06-20 (토)** — 6/22에서 앞당겨짐. **남은 일수: D-2**
**작성 목적**: 2026-06-18 Cowork 세션에서 결정·생성·미해결된 모든 내용을 다음 Claude 세션이 콜드스타트로 이어받을 수 있도록 정리

---

## ★ 최우선 변경 — 발표 단순화 결정 (2026-06-18 후반)

> 박진우의 결정:
> 1. 발표일 **6/22 → 6/20 (토)** 로 당겨짐 → **D-2**.
> 2. 최종발표 스코프를 **단 하나의 기능**으로 축소.
> 3. 그 외 모든 기능은 **삭제하지 말고 `archive/` 폴더로 격리 보관** (개인 프로젝트로 이어갈 예정).

### 최종발표에 남기는 단 하나의 기능

```
카메라 → YOLOv8n-pose → 17 keypoints
    → 하도 기본 동작 분류 (PPT + 207장 기반 도메인 어휘)
    → "지금 ○○ 동작을 취하고 있음" 표시
```

- 핵심 차별점: **일반 동작(charge/shoot/shield)이 아니라, 박진우가 하도 월드컵에서 시연한 하도리듬 도메인 어휘로 분류**.
- 데모 1인 기준, 카메라 1대, 도메인 어휘 자체가 발표의 메인 스토리.

### 격리(archive)할 것들 — 발표 시나리오에서 제외, 코드는 보존

| 카테고리 | 파일 | archive 안의 위치 |
|---------|------|-----------------|
| Level 1 추적 | `src/homography.py`, `src/tracker.py`, `src/visualizer.py`, `src/calibrate.py`, `src/aruco_calibrate.py`, `src/recorder.py` | `archive/level1_tracking/` |
| Level 2 전술 | `src/tactic_engine.py`, `src/guide.py`, `src/movement_model.py`, `src/analyzer.py`, `src/extract_patterns.py`, `src/visualize_patterns.py`, `src/label_intents.py` | `archive/level2_tactical/` |
| Level 3 부가 | `pose.py`의 `NEXT_ACTION_RECS`, `recommend_next_action()`, `analyze_pose()`, `PoseFeatures`, `draw_movement_arrow()` | `archive/level3_extras/pose_extras.py` 로 분리 |
| 통합 진입 | `src/main.py`(Level 1+2 오케스트레이션), `src/demo.py`, `src/demo_pose.py` | `archive/integrated/` |
| 측정/벤치 (발표 직접 사용 안 함) | `src/benchmark.py`, `src/measure_error.py`, `src/measure_w5.py`, `src/pi_preflight.py` | `archive/measurement/` |
| 관련 테스트 | `tests/test_tactic_engine.py`, `tests/test_homography.py`, `tests/test_visualizer.py`, `tests/test_guide.py`, `tests/test_movement_model.py`, `tests/test_tracker.py`, `tests/test_benchmark.py`, `tests/test_measure_w5.py`, `tests/test_pi_preflight.py` | `archive/tests/` |
| 보조 도구 | `src/upload.py`, `src/annotate_tool.py`, `src/auto_label.py`, `src/extract_frames.py`, `src/vest.py` | `archive/tools/` |

### 발표용 코어 (남기는 것) — `src/` 최소 구성

```
src/
├── camera.py             ← 카메라 입력 (threaded)
├── detector.py           ← YOLOv8n-pose 래퍼 (bbox + 17 keypoints)
├── pose.py               ← classify_hado_action만 남김 (그 외는 archive로 분리)
├── action_demo.py        ← 메인 발표 데모 (다음 동작 추천 패널 제거 또는 단순화)
└── annotate.py           ← 한글 텍스트 표시 (put_text_kr)
```

`pose.py`는 **분리 작업이 필요**:
- 남길 것: `classify_hado_action`, `SKELETON_PAIRS`, `ACTION_KO`, `ACTION_COLOR`, `ACTION_EMOJI`, `draw_skeleton`, `draw_keypoint_ids`, `sample_vest_hue`, `_KP_CONF_MIN`, `_LIMB_COLORS`, 키포인트 인덱스 상수들
- 단, `ACTION_KO`/`ACTION_COLOR`/`ACTION_EMOJI` 의 키 7개는 **PPT 받은 후 하도 도메인 어휘로 교체**.
- 분리할 것 (→ `archive/level3_extras/pose_extras.py`): `NEXT_ACTION_RECS`, `NextActionRec`, `recommend_next_action`, `PoseFeatures`, `analyze_pose`, `draw_posture_label`, `draw_movement_arrow`, `POSTURE_*` 상수들.

`action_demo.py` 도 단순화 필요:
- 제거: 다음 동작 추천 패널 (`_draw_action_panel`의 하단 절반), `last_recs`, `recommend_next_action` 호출
- 유지: 스켈레톤, 현재 동작 + 신뢰도 표시, FPS, `--show-kp-ids`

### 격리 작업의 원칙

- **삭제 금지**: 모든 코드는 `archive/` 아래로 이동만, `git rm` 금지.
- **archive/README.md 작성**: 무엇이 왜 격리됐는지, 원본 위치, 어떻게 다시 활용할지 1페이지 정리.
- **import 경로 정리**: 격리 후 `action_demo.py` 가 archive를 import 하지 않도록 확인.
- **테스트 분리**: archive로 가는 모듈의 테스트도 함께 archive로. 발표 코어 테스트만 `tests/` 에 남김.
- **git 커밋**: `refactor: archive level1/2/extras for presentation simplification` 단일 커밋.

---

## 0. 5초 요약 (v2)

- 발표일 🔴 **6/20(토)** 로 당겨짐 → D-2.
- 발표 스코프를 **단 하나의 기능**으로 축소: 17 keypoints → 하도 기본 동작 분류.
- 그 외 Level 1·2 코드, 다음 동작 추천, 공격/수비 의도 추론은 모두 **`archive/` 폴더로 격리 보관** (개인 프로젝트로 이어갈 자산).
- 발표 자료 13장 PPTX는 작성됐으나 (`docs/HADO_Final_Presentation.pptx`) **이번 단순화에 맞춰 재작성 필요** — 슬라이드 13장 → 약 7장으로 축소.
- 도메인 자료(하도 기본 동작 PPT + 시연 사진 207장 HEIC)는 `/Users/jinu/Desktop/하도 기본 동작/` 에 존재. PPT는 마운트 deadlock으로 못 읽음 → 박진우가 **PDF로 export 후 채팅 업로드 필수**.

---

## 1. 이 세션에서 결정·생성된 것

### 1-1. 발표 자료 신규 작성 (완료)

- 파일: `docs/HADO_Final_Presentation.pptx` (462KB), `.pdf` (213KB)
- 빌드 스크립트: 이 세션 임시 워크스페이스. 재빌드가 필요하면 `pptxgenjs` 기반으로 다시 짜야 함.
- 구성 13장:

| # | 슬라이드 | 핵심 |
|---|---------|------|
| 1 | Title (다크) | HADO Smart Court IoT · 박진우 · 2026.06.22 |
| 2 | Problem | 게임은 실시간, 분석은 그렇지 않다 + 10×6m / 3v3 / <200ms |
| 3 | System Overview | 6블록 파이프라인 + L1/L2/L3 카드 |
| 4 | Hardware | Pi4 / Camera / Output 3카드 + "Fully offline" 강조 |
| 5 | **L1 Tracking** | YOLOv8n-pose @320 · IoU · 호모그래피 + Mac 98fps / Pi ≥15fps |
| 6 | **L2 Tactical Engine** | R1~R6 6룰 + "Why rules, not ML?" |
| 7 | **L3 Action Recognition** | 17 키포인트 스켈레톤 + 7동작 + 스케일 정규화 공식 |
| 8 | **L3 Next Action Recommendation ★NEW** | `NEXT_ACTION_RECS` 전환 테이블 + `recommend_next_action` 3단 로직 |
| 9 | Pi 4 Optimization | NCNN ≈3× · Threaded 0ms · imgsz=320 −75% |
| 10 | Validation | 272 tests / 15 modules / 7 action sets + W5 [TODO] 측정표 |
| 11 | Demo (다크) | 영상 placeholder + 3가지 run 명령어 |
| 12 | Results & Future Work | 7 완료 / 4 향후 과제 |
| 13 | Thank you (다크) | Q&A + GitHub URL |

디자인: deep navy `#0A1A3E` + cyan `#00C2D7` + coral `#FF6B35`. Helvetica. 다크/라이트 샌드위치.

### 1-2. 채워야 할 [TODO] (발표 전 필수)

**Slide 10 W5 측정표** — Pi4에서 `./run.sh w5_measure` 실행 후 채워야 함:
- NCNN inference FPS (목표 ≥15)
- ONNX inference FPS (목표 ≥10)
- RAM usage (목표 <1.5 GB)
- CPU temperature (30 min, 목표 <70 °C)
- TTS guidance latency (목표 <500 ms)

**Slide 11 데모 영상** — 현재 `data/demo_track_a_pivot.mp4` placeholder. 실제 영상 녹화 후 임베드 또는 외부 재생 링크.

### 1-3. Cowork 세션에서의 부가 작업 (실제 프로젝트와 분리)

- 사용자가 처음에 Claude Code 설치를 진행했음 (`claude --version` → `2.1.140`, Sonnet 4.6, Claude Pro). 이건 이 프로젝트와 별개로 **본인 Mac**에 설치된 것.
- 외부 카메라(GoPro/Sony) + USB 3.0 캡처보드 호환성·실전 함정 정리. 결론: **기존 Arducam IMX519 CSI로 충분**, 캡처보드 사용 시 MJPG 강제 + 공식 5.1V/3A 어댑터 필수.
- LMS용 영문 제출문 작성 (참고문헌 4개: TFLite OD / ByteTrack / Voronoi(Fonseca 2012) / CV for Sports(Thomas 2017)).
- 클라우드 백업 전략 정리: **rclone + Google Drive** 권장 (iCloud는 Pi에서 불가). Strategy = `csv_and_clips` 추천. 이건 발표 무관, 운영 편의용.

---

## 2. 새 방향 제안 — "도메인 동작 + 공격/수비 의도 추론"

### 2-1. 사용자가 새로 제시한 내용

> 「단순히 차징, 공격, 쉴드 이런 단순한 동작만이 아니라 하도의 기본동작에서 어떤 동작을 하고 있는지 파악하고 그 내용을 기반으로 공격을 하는건지 수비를 하는건지도 같이 파악하는걸로 갔으면 좋겠다」
> — 박진우 (2026-06-18)

근거:
- 본인은 **하도 월드컵에서 "하도리듬" 주제로 전세계 대상 시연 경력**. 도메인 지식이 코드에 녹아들지 않음.
- 시연 영상: https://www.youtube.com/watch?v=htic7uFaEFE (제목: "하도 리듬")
- 자료 폴더: `/Users/jinu/Desktop/하도 기본 동작/`
  - 본인이 직접 동작 취한 사진 **207장** (전부 **HEIC** 포맷, iPhone 원본)
  - `하도리듬운동트레이닝-1편.pptx` (11.2MB)

### 2-2. 제안한 3-Layer 구조

```
17 keypoints
    ↓
[Layer A] 하도 도메인 동작 분류  (← PPT/사진으로 정의되는 어휘)
    예: "차징", "샷", "쉴드", "스텝백", "사이드스텝 L/R", "페이크",
        "리듬 발놀림", "준비자세" 등 — PPT 보고 확정 필요
    ↓
[Layer B] 의도 추론 (Attack / Defense / Neutral)
    매핑 예:
        ATTACK   ← 차징, 샷, 카운터, 페이크샷
        DEFENSE  ← 쉴드, 스텝백, 사이드스텝, 회피
        NEUTRAL  ← 리듬 발놀림, 준비자세
    + 직전 1초 시퀀스 가중치 (현재 history 메커니즘 재활용)
    ↓
[Layer C] 다음 동작 추천  (← 기존 `NEXT_ACTION_RECS` 확장)
```

이 설계는 기존 `src/pose.py`의 7동작 로직을 버리는 게 아니라 **상위 어휘로 일반화**하는 형태. 임계값 튜닝과 매핑 테이블만 바꾸면 됨.

### 2-3. 발표 슬라이드 반영 안 (PPT 자료 확정 후)

- Slide 7: 7동작 → **하도 도메인 어휘 N동작** 로 교체
- Slide 8: 다음 동작 추천 → **공격/수비 의도 레이어 추가**
- 신규 Slide 9-1: "하도 월드컵 시연 — 직접 정의한 도메인 어휘" (본인 경력 + 유튜브 QR)
- 신규 Slide 9-2: ATTACK/DEFENSE 분류 결과 + 207장으로 측정한 정확도

---

## 3. 미해결 과제 (다음 Claude가 받아야 할 입력)

### 3-1. 🔴 PPT 내용 추출 — **사용자 액션 필요**

- 파일: `/Users/jinu/Desktop/하도 기본 동작/하도리듬운동트레이닝-1편.pptx`
- 이 세션에서 못 읽은 이유: 마운트 read에서 `OSError: [Errno 35] Resource deadlock avoided` 반복. 작은 파일은 됨, 11MB짜리는 deadlock.
- 다음 Claude는 **시도하지 말 것**. 같은 이유로 또 실패함.
- **사용자가 해야 할 일**: Keynote/PowerPoint에서 `파일 → PDF로 보내기` → **채팅창에 직접 업로드** (Cowork는 PDF의 페이지 이미지를 컨텍스트에 넣어줌 → 다음 Claude가 보고 동작 어휘 추출 가능).

### 3-2. 🟡 HEIC → JPG 일괄 변환

- 207장 모두 HEIC라 `cv2.imread()` 못 함.
- 변환 스크립트는 작성 안 됐음. 다음 Claude가 작성하면 됨:
  ```python
  # tools/convert_heic_to_jpg.py
  from pillow_heif import register_heif_opener
  register_heif_opener()
  from PIL import Image
  from pathlib import Path
  src = Path("/Users/jinu/Desktop/하도 기본 동작")
  dst = Path("/Users/jinu/iot project/hado-smart-court-iot/data/reference_actions/_raw_jpg")
  dst.mkdir(parents=True, exist_ok=True)
  for heic in src.glob("*.HEIC"):
      Image.open(heic).save(dst / (heic.stem + ".jpg"), "JPEG", quality=90)
  ```
- 변환 후 사용자가 동작별 폴더(`charging/`, `shooting/`, `shield/` ...)에 분류해 넣어야 함. 자동 분류는 PPT 어휘 확정 후 검토.

### 3-3. 🟡 도메인 어휘 확정

PPT PDF 받은 뒤 다음 Claude가 결정해야 할 것:
- HADO 기본 동작 정확한 명칭과 개수 (영문/한글 매핑)
- 동작 간 위계 (상위/하위, 공격/수비/중립 분류)
- 키포인트로 식별 가능한 동작과 그렇지 않은 동작의 구분

---

## 4. 다음 Claude에게 주는 작업 지침

### 4-1. 기존 인계서(`TRACK_A_PIVOT_HANDOFF_2026_06_17.md`) 의 모든 §7 규칙 그대로 유지

- 건드리지 말아야 할 파일 동일
- 코딩 컨벤션, 테스트 정책, 커밋 컨벤션 동일
- Track B 일체 금지 동일

### 4-2. 이 인계서가 우선

`TRACK_A_PIVOT_HANDOFF_2026_06_17.md` 와 충돌 시 **이 문서가 최신**.

### 4-3. 첫 행동 순서 (v2 — D-2 기준, 토요일 발표)

**선결 조건**: 박진우가 `하도리듬운동트레이닝-1편.pptx`를 PDF로 export해 채팅 업로드 완료.

**Day 1 (6/18 저녁 ~ 6/19 오전)**
1. **archive/ 격리 작업** — 위 표대로 Level 1·2 + 다음 동작 추천 모두 격리.
   - `archive/README.md` 작성 (격리 이유, 원본 위치, 부활 방법).
   - `pose.py` 분리: 도메인 동작 분류만 남기고 나머지는 `archive/level3_extras/pose_extras.py`.
   - `action_demo.py` 단순화: 다음 동작 추천 패널 제거.
   - 테스트 분리.
   - 격리 후 `python -m pytest tests/ -q` 통과 확인 (축소된 테스트셋).
   - 격리 후 `python -m src.action_demo --source 1 --show-kp-ids` 동작 확인.
   - Single commit: `refactor: archive level1/2/extras for presentation simplification`.

2. **PDF에서 도메인 어휘 추출** — 박진우가 PDF 올리면 즉시 분석.
   - 동작 N개 (이름, 한글/영문, 설명, 분류 우선순위) 표로 정리.
   - 박진우 확인 후 확정.

3. **HEIC → JPG 변환** — 207장 → `data/reference_actions/_raw_jpg/`.
   ```python
   # tools/convert_heic_to_jpg.py
   from pillow_heif import register_heif_opener
   register_heif_opener()
   from PIL import Image
   from pathlib import Path
   import unicodedata, os
   for d in os.listdir(os.path.expanduser("~/Desktop")):
       if unicodedata.normalize('NFC', d) == '하도 기본 동작':
           src = Path("~/Desktop").expanduser() / d; break
   dst = Path("data/reference_actions/_raw_jpg"); dst.mkdir(parents=True, exist_ok=True)
   for heic in src.glob("*.HEIC"):
       Image.open(heic).save(dst / (heic.stem + ".jpg"), "JPEG", quality=90)
   ```

**Day 2 (6/19 오후)**
4. **207장 동작 라벨링** — 박진우가 폴더 분류해줘야 함. 예:
   ```
   data/reference_actions/
   ├── basic_step/        (하도 동작 1)
   ├── side_step_l/       (하도 동작 2)
   ├── ...
   ```
   - 박진우가 직접 분류 → 다음 Claude는 17kp 추출 + 동작별 통계만.

5. **`pose.py` 임계값 튜닝** — 207장의 키포인트 통계로 자동 보정.
   - `training/tune_thresholds.py` 작성.
   - 튜닝 전/후 정확도 측정 → 정량 지표 확보 (발표 슬라이드 용도).

**Day 3 (6/20 오전 — 발표 당일)**
6. **발표 자료 단순화 재작성** — `docs/HADO_Final_Presentation.pptx`를 v2로:
   - 13장 → **약 7장**으로 축소.
   - 제안 구성:
     | # | 슬라이드 |
     |---|---------|
     | 1 | Title |
     | 2 | Problem — "동작 학습이 어렵다 / 하도리듬을 정확히 익히기 어렵다" |
     | 3 | Solution overview — 카메라 → 17kp → 도메인 동작 분류 (한 줄 흐름) |
     | 4 | 하도리듬 도메인 — 하도 월드컵 시연 경력 + 동작 어휘 N개 |
     | 5 | 분류 로직 — 스케일 정규화 + 키포인트 기반 판정 |
     | 6 | Demo — 라이브 또는 녹화 + 207장 정확도 지표 |
     | 7 | Results & Future — 격리한 Level 1·2를 "확장 가능성"으로 소개 |
   - Level 1·2 슬라이드 다 제거, 다음 동작 추천 슬라이드 제거.
   - 단, **Future Work에서 한 줄로 언급**: "코트 전체 추적·전술 가이드·다음 동작 추천 등은 이미 prototype 완료, 향후 통합 예정". (= archive 폴더 활용 ↑)

7. **발표 대본 재작성** — `docs/FINAL_PRESENTATION_SCRIPT_EN.md` v2.
   - 약 3분 (슬라이드당 25~30초).
   - 핵심 메시지 한 줄: *"We built an action recognizer that speaks HADO's own vocabulary, learned from World Cup demonstration footage."*

8. **QA_PREP.md 단순화** — 5~6개 핵심 Q&A.
   - "Why HADO domain vocab vs generic actions?"
   - "What was the World Cup demo, and how does it relate?"
   - "How did 207 photos translate into tuned thresholds?"
   - "What's in the archive folder, and why isn't it part of the demo?"
   - "If you had more time, what would you add first?"

### 4-4. 발표용 데모 영상

`python -m src.action_demo --source 1 --show-kp-ids --record data/demo_track_a_pivot.mp4 --max-frames 1800` 실행해서 60초 영상. Slide 11에 임베드.

### 4-5. Pi4 실측

`./run.sh w5_measure` → `data/w5_measurements.md` B섹션 채우기 → Slide 10 측정표에 반영.

---

## 5. 발표까지 남은 일별 권장 작업 (v2 — 토요일 발표)

| 날짜 | 작업 | 책임 |
|------|------|-----|
| **6/18 (오늘 저녁)** | PPT를 PDF로 export 후 채팅 업로드 | 박진우 |
| 6/18 (저녁) | `archive/` 폴더 격리 작업 + 테스트 통과 확인 | 다음 Claude |
| 6/19 (오전) | PDF에서 도메인 어휘 추출 + 박진우 확인 | 다음 Claude + 박진우 |
| 6/19 (오전) | HEIC → JPG 변환 (207장) | 다음 Claude |
| 6/19 (오후) | **박진우가 207장을 동작별 폴더로 분류** | 박진우 |
| 6/19 (저녁) | 17kp 통계 + 임계값 자동 튜닝 + 정확도 측정 | 다음 Claude |
| 6/20 (오전) | 발표 자료 v2 (7장) 재작성 + 대본 + QA_PREP | 다음 Claude |
| 6/20 (오전) | 데모 영상 녹화 (또는 라이브 시연 리허설) | 박진우 |
| 6/20 (오전) | 영어 리허설 ×2 | 박진우 |
| **6/20 (오후/저녁)** | **🎤 최종발표** | 박진우 |

⚠️ Pi4 W5 실측은 **이번 발표에서는 우선순위 낮음** (단순화 결정으로 슬라이드에서 측정표 빠짐). 박진우가 데모를 라이브로 할지 녹화로 할지에 따라 Pi4 사용 여부 결정.

---

## 6. 절대 하지 말 것 (v2 재강조)

1. `/Users/jinu/Desktop/하도 기본 동작/*.pptx`를 직접 읽으려 시도 → 마운트 deadlock. PDF를 받을 때까지 PPT 분석 작업 보류.
2. **Level 1·2 코드 삭제 절대 금지** → `archive/`로 이동만. 박진우가 개인 프로젝트로 이어갈 자산.
3. `config/calibration.json`, `data/movement_data*.csv` 건드리기 금지.
4. Track B 관련 파일 일체 금지.
5. 임계값 튜닝 결과를 **검증 없이** 푸시 금지 → 반드시 정확도 비교 후 적용.
6. `docs/HADO_Final_Presentation.pptx`(현재 13장 버전)을 그대로 사용 금지 → **단순화 결정에 따라 7장 v2로 재작성**. 단, 13장 버전 파일은 백업으로 `docs/archive/HADO_Final_Presentation_v1_13slides.pptx`로 보존.
7. `archive/` 안에 import 의존성 두기 금지 → `src/` 발표 코어는 archive를 import하지 않아야 함.

---

## 7. 상태 빠른 확인 명령어

```bash
cd "/Users/jinu/iot project/hado-smart-court-iot"
source hado_venv/bin/activate

# 발표 자료 확인
open docs/HADO_Final_Presentation.pdf
open docs/HADO_Final_Presentation.pptx

# 기존 테스트 통과 확인
python -m pytest tests/ -q                # 272 통과 기대

# 데모 즉시 실행
python -m src.action_demo --source 1 --show-kp-ids

# 이전 시스템(Level 1+2)도 살아있는지
python -m src.main --level 2

# 폴더 확인 (Korean 인코딩 NFD 주의)
python3 -c "
import os, unicodedata
mnt = '/Users/jinu/Desktop'
for d in os.listdir(mnt):
    if unicodedata.normalize('NFC', d) == '하도 기본 동작':
        print(os.path.join(mnt, d))
        print('files:', len(os.listdir(os.path.join(mnt, d))))
        break
"
```

---

## 8. 이 인계서의 의미

이 인계서는 다음 Claude가 **이 문서 하나만 읽고도**:
- 6/17 pivot 결정 + 6/18 새 방향 확장 둘 다 이해
- 발표 자료가 어디 있고 무엇이 들어있는지 파악
- 남은 [TODO]가 무엇인지 파악
- 어떤 순서로 무엇을 하면 되는지 파악
- 어떤 함정을 피해야 하는지 파악

할 수 있도록 작성됨. 6/17 인계서와 함께 읽으면 전체 맥락이 완성됨.

---

*이 인계서는 `/Users/jinu/iot project/hado-smart-court-iot/docs/TRACK_A_SESSION_HANDOFF_2026_06_18.md` 에 저장됨.*
*이전 인계서: `docs/TRACK_A_PIVOT_HANDOFF_2026_06_17.md`*
*발표 자료: `docs/HADO_Final_Presentation.pptx` / `.pdf`*
