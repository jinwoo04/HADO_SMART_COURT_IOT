# Track A Presentation Scope Brief — 2026-06-10

이 문서는 **발표 때 실제로 보여준 내용만** 기준으로,
Claude Code에게 넘겨도 되는 구현 범위를 정리한 Track A 전용 브리프다.

중요:

- 이 문서는 `Track A` 전용이다.
- `Track B`의 Roboflow relabel / effect-occlusion / 개인 연구 루프는 포함하지 않는다.
- 최근에 만든 Track B skeleton review export도 여기에는 넣지 않는다.

## 1. 발표에서 실제로 다룬 핵심 범위

발표 내용은 크게 아래 5개였다.

1. **실시간 스마트 코트 전술 분석 시스템**
2. **단일 카메라 + Raspberry Pi 4 + MacBook 서버 아키텍처**
3. **선수 추적 + 버드아이뷰 + 전술 규칙 엔진**
4. **스켈레톤 기반 HADO 동작 인식 데모**
5. **오프라인 IoT 시스템으로서의 구현 가능성**

즉, Claude Code에 넘겨야 하는 것도 이 범위까지만이다.

## 2. Claude Code가 이어서 구현해도 되는 범위

### A. 시스템 아키텍처 정리

발표 슬라이드 기준 시스템 흐름:

- Fixed/GoPro camera
- Raspberry Pi 4
- Wi-Fi hotspot
- MacBook server

역할 분리:

- **Edge layer (Pi)**:
  - 카메라 입력
  - YOLOv8n-pose 추론
  - bbox + 17 keypoints 추출
  - 필요한 최소 정보 전송

- **Server layer (MacBook)**:
  - player state 수신
  - 위치 분석
  - 버드아이뷰 렌더링
  - tactical engine 실행

- **Tactical layer**:
  - spacing / lane / defense / attack rule 판단
  - 시각화 또는 음성 가이드 출력

### B. Level 1 — Tracking

발표 기준 구현 범위:

- YOLOv8n-pose 기반 선수 감지
- IoU tracker 기반 player ID 유지
- bbox 하단 중앙 foot point 사용
- homography로 pixel -> court meter 변환
- bird-eye view 표시

관련 코드:

- `src/detector.py`
- `src/tracker.py`
- `src/homography.py`
- `src/visualizer.py`

### C. Level 2 — Tactical Engine

발표 기준 구현 범위:

- 규칙 기반 tactical engine
- spacing / lane cover / backline defense / counter threat 등
- bird-eye view 위 화살표 시각화
- 설명 가능한 rule label

관련 코드:

- `src/tactic_engine.py`
- `src/guide.py`
- `src/demo.py`
- `src/main.py`

### D. Level 3 — Action Recognition

발표 기준 구현 범위:

- YOLOv8-pose keypoints 기반 7동작 분류
- action demo 화면
- skeleton overlay
- score bars / action history / confidence HUD

발표에서 다룬 7동작:

- charge
- shoot
- shield
- dodge left
- dodge right
- crouch
- ready

관련 코드:

- `src/pose.py`
- `src/action_demo.py`
- `src/demo_pose.py`

### E. Pi4 최적화

발표 수준에서 다룬 구현 포인트:

- NCNN 우선
- threaded camera capture
- imgsz 320
- offline TTS

관련 코드/문서:

- `src/camera.py`
- `HADO_iot/pi/export_ncnn.py`
- `docs/W5_PI4_QUICKRUN.md`
- `docs/TRACK_A_FIELD_TEST_PACKET_2026_06_13.md`
- `docs/TRACK_A_FIELD_TEST_VISUAL_2026_06_13.html`

## 3. Claude Code에게 넘기면 안 되는 범위

아래는 전부 Track B 또는 발표 범위 밖이다.

- Roboflow relabeling
- player-only retrain loop
- effect / shield / projectile 멀티클래스 확장
- AR effect occlusion robustness 연구
- `tools/retrain_track_b_export.py`
- `tools/prepare_track_b_batch_review.py`
- `tools/export_track_b_pose_review.py`
- `docs/TRACK_B_*`

즉, Claude Code에게는 **발표용 구현 수준까지만** 넘겨야 한다.

## 4. 발표 기준으로 Claude Code가 실제로 해도 되는 일

아래는 발표 내용과 직접 연결되는 구현 작업들이다.

### 우선순위 높음

1. `src/demo.py` 발표용 시연 안정화
2. `src/action_demo.py` 동작 인식 데모 안정화
3. `src/demo_pose.py` 버드아이뷰 + skeleton overlay 정리
4. `src/tactic_engine.py` 규칙 설명성/시각화 정리
5. `src/camera.py` Pi4 threaded capture 안정화

### 우선순위 중간

1. 발표 슬라이드 수준에 맞는 HUD 문구/시각화 정리
2. `run.sh` 기반 실행 흐름 단순화
3. Pi4 현장 테스트용 quick run 정리
4. action demo 녹화/저장 흐름 정리

### 우선순위 낮음

1. 코드 리팩터링
2. 문서 polish
3. small test additions

## 5. 발표 수준에서의 “구현 완료 기준”

Claude Code는 아래가 되면 Track A 범위에서 충분하다.

- 단일 카메라 입력이 안정적으로 들어온다
- YOLOv8n-pose로 선수 bbox + keypoints가 나온다
- player ID가 어느 정도 유지된다
- bird-eye view에 실시간 위치가 그려진다
- tactical rule이 설명 가능한 형태로 출력된다
- action demo에서 7동작이 시연 가능하다
- Pi4에서 돌아갈 최적화 방향이 명확하다

반대로, 아래는 Track A 완료 기준이 아니다.

- AR effect가 겹친 실제 경기영상에서 robust detection
- Roboflow 반복학습
- effect-aware skeleton verification

이건 전부 Track B다.

## 6. Claude Code에게 전달할 짧은 메시지

아래 내용을 그대로 넘기면 된다.

---

This task is **Track A only**, based strictly on what was presented in the IoT presentation.

Please continue only the presentation-level implementation:

- single-camera HADO player tracking
- YOLOv8n-pose based bbox + 17 keypoints
- IoU tracking
- homography to bird-eye court coordinates
- rule-based tactical engine
- real-time action recognition demo with 7 actions
- Raspberry Pi 4 optimization (NCNN, threaded capture, imgsz=320)

Relevant files:

- `src/detector.py`
- `src/tracker.py`
- `src/homography.py`
- `src/tactic_engine.py`
- `src/demo.py`
- `src/pose.py`
- `src/action_demo.py`
- `src/demo_pose.py`
- `src/camera.py`
- `docs/FINAL_PRESENTATION_SCRIPT_EN.md`
- `docs/PROJECT_SUMMARY_2026_06_09.md`
- `docs/TECH_REPORT_2026_06_09.md`

Do **not** continue Track B work here.
Do not include:

- Roboflow relabeling
- player-only retraining loops
- effect/occlusion research
- Track B skeleton review exports

Keep the implementation aligned with the presentation scope only.

---

## 7. 지금 당장 가장 자연스러운 다음 작업

발표 기준에서 Claude가 이어서 하기 가장 좋은 작업은 아래다.

1. `src/demo.py` / `src.action_demo.py` / `src.demo_pose.py` 시연 흐름 정리
2. Pi4 실행 경로를 더 단순하게 정리
3. action demo와 tactical demo의 시각화 polish
4. 발표에서 말한 시스템 아키텍처와 실제 코드 구조를 더 잘 맞추기

즉, 지금 Claude에게 넘겨야 하는 건
`발표 수준의 Track A 구현 정리`
이고,
`Track B 연구 확장`
은 아니다.
