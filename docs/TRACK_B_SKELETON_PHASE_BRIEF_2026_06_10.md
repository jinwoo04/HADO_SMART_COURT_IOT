# Track B Skeleton Phase Brief — 2026-06-10

이 문서는 Track B에서 `player-only relabel/retrain` 다음 단계로 넘어갈 때,
실제 선수 위치값과 동작을 스켈레톤 기반으로 확인하는 작업을
Codex나 Claude Code가 바로 이어받을 수 있도록 정리한 브리프다.

## 1. 이 단계의 목적

Track B의 현재 1차 목표는 `player-only detector` 안정화다.

그 다음 단계는 아래 두 가지를 확인하는 것이다.

1. 실제 경기영상에서 선수 위치를 프레임 단위로 안정적으로 추적할 수 있는가
2. 스켈레톤(keypoints) 기반으로 동작을 어느 정도 일관되게 읽을 수 있는가

즉, 이 단계의 목적은 “바로 멀티클래스 effect 인식으로 가는 것”이 아니라,
`bbox -> track -> keypoints -> action/position` 파이프라인이 실제 선수 영상에서
성립하는지 확인하는 것이다.

## 2. 왜 player-only 다음에 skeleton phase로 가는가

pose model은 detector보다 더 민감하다.

- player bbox가 흔들리면 keypoints도 같이 흔들린다
- AR effect가 torso/arm을 가리면 action 판정이 급격히 불안정해진다
- 따라서 pose/action은 detector가 어느 정도 안정된 뒤에 얹는 것이 맞다

정리하면:

- `player-only V5/V6 안정화`가 먼저
- 그 다음 `skeleton-based verification`

## 3. 이미 있는 코드 자산

이 프로젝트에는 pose/action 관련 코드가 이미 존재한다.

### 핵심 모듈

- `src/detector.py`
  - YOLOv8 detect / pose 공용 래퍼
  - pose 모델이면 `Detection.keypoints`까지 채움

- `src/tracker.py`
  - IoU 기반 multi-person tracking

- `src/homography.py`
  - pixel -> court meter 좌표 변환

- `src/pose.py`
  - keypoints 기반 자세/동작 해석
  - `classify_hado_action()`
  - `analyze_pose()`
  - `draw_skeleton()`
  - `sample_vest_hue()`

- `src/action_demo.py`
  - 실시간 action demo
  - `charge / shoot / shield / dodge_l / dodge_r / crouch / ready`

- `src/demo_pose.py`
  - 실제 영상에서 detection + tracking + bird-eye + skeleton overlay demo

### Pi/Edge 참고 자산

- `HADO_iot/pi/edge_streamer.py`
  - pose 추론 + vest hue + action label 송신 예시

## 4. 이번에 추가한 Track B용 도구

새 스크립트:

- `tools/export_track_b_pose_review.py`

역할:

- 실제 경기영상을 입력으로 받음
- pose detector 실행
- tracker로 track id 유지
- 대략적인 court 좌표 계산
- action / posture 추정
- 아래 산출물 생성

출력:

- `*_pose_tracks.csv`
- `*_pose_tracks.jsonl`
- `*_pose_overlay.mp4` (선택)
- `*_pose_summary.json`

즉, 이 스크립트는 “스켈레톤이 보였는가”를 사람이 검토할 수 있는
review artifact를 만드는 도구다.

## 5. 바로 실행하는 방법

### 가벼운 smoke

```bash
./hado_venv/bin/python tools/export_track_b_pose_review.py \
  --video "data/drive_imports/batch02_raw/match01.mp4" \
  --out-dir "data/track_b_pose_review/smoke_match01" \
  --pt \
  --device mps \
  --frame-stride 60 \
  --max-frames 10 \
  --no-video
```

### 실제 review용

```bash
./hado_venv/bin/python tools/export_track_b_pose_review.py \
  --video "data/drive_imports/batch02_raw/match01.mp4" \
  --out-dir "data/track_b_pose_review/match01_full_review" \
  --pt \
  --device mps \
  --frame-stride 3
```

설명:

- `--pt`는 `yolov8n-pose.pt` 강제 사용
- `--no-video`를 빼면 skeleton overlay mp4도 같이 저장
- `frame-stride`는 작을수록 촘촘하지만 느려짐

## 6. 이 단계에서 먼저 봐야 할 지표

### 위치 쪽

- track id가 몇 프레임 유지되는가
- court_x / court_y가 급격히 튀는가
- 실제 플레이 장면에서 선수 위치가 코트 내부에 합리적으로 들어오는가

### 동작 쪽

- `charge`와 `shoot`가 과도하게 섞이지 않는가
- effect overlap이 심할 때 `ready`로 너무 많이 무너지지 않는가
- crouch / dodge 방향 판정이 최소한 대략 맞는가

### 스켈레톤 쪽

- torso/shoulder/arm keypoints가 끊기는 패턴이 어디서 많은가
- 실드나 이펙트가 arm keypoint를 얼마나 자주 망가뜨리는가

## 7. Claude Code가 다음에 해야 할 일

Claude Code는 아래 순서로 진행하면 된다.

1. 최신 `player-only` 베스트 모델 상태와 Track B 문서 확인
2. `tools/export_track_b_pose_review.py`를 실제 경기영상 몇 개에 실행
3. CSV / JSONL / overlay mp4를 바탕으로 실패 패턴 분류
4. pose 단계에서 잘 되는 장면 / 안 되는 장면을 구분
5. 필요하면 action confusion review용 샘플 클립 폴더 생성
6. skeleton phase가 Track B에서 실용적인지 판단

## 8. Claude Code에게 넘길 핵심 판단 기준

Claude가 봐야 할 핵심은 아래다.

- 현재는 effect 멀티클래스보다 player-only detector 안정화가 우선
- skeleton phase는 detector가 어느 정도 안정된 뒤의 검증 레이어
- clean/no-effect 장면과 heavy-effect 장면을 분리해서 비교해야 함
- pose가 안 잡히는 이유가 detector miss인지, keypoint miss인지 분리해야 함

## 9. Claude Code용 전달 문안

아래 내용을 그대로 Claude Code에 전달해도 된다.

---

Track B is now moving from player-only relabel/retrain into a skeleton-based verification phase.

Current goal:
- keep Track B separate from Track A
- continue using the current player-only baseline first
- verify whether real-player location and action can be extracted reliably from pose keypoints

Existing relevant code:
- `src/detector.py`
- `src/tracker.py`
- `src/homography.py`
- `src/pose.py`
- `src/action_demo.py`
- `src/demo_pose.py`
- `tools/export_track_b_pose_review.py`

What you should do next:
1. run `tools/export_track_b_pose_review.py` on a few real match videos
2. inspect the generated CSV / JSONL / overlay outputs
3. separate failure causes into:
   - detector miss
   - unstable track
   - missing keypoints
   - unstable action classification
4. summarize which clips are good candidates for skeleton-based action verification
5. keep the project player-only for now; do not expand to effect multi-class yet

Important context:
- pose/action is the next validation layer after player-only detection, not a replacement for the current relabel/retrain loop
- heavy AR/effect overlap will probably break arm/torso keypoints more than clean footage
- compare clean/no-effect scenes vs heavy-effect scenes explicitly

---

## 10. 현재 가장 현실적인 다음 작업

가장 먼저 할 일은 아래 둘 중 하나다.

1. Roboflow relabel이 끝난 새 export로 `player-only` 재학습을 먼저 완료
2. 동시에 소수의 실제 경기영상에 대해 skeleton review artifact를 뽑아서
   “이 단계가 지금 실용적인지” 빠르게 판단

즉, 순서는 완전히 `pose 먼저`가 아니라,
`player-only stabilization + pose verification`을 병행하는 쪽이 맞다.
