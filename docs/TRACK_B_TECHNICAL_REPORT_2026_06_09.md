# Track B Technical Report — 2026-06-10

## 1. 연구 목적

Track B의 목적은 HADO 경기 영상에서 AR 이펙트가 선수와 겹치는 상황에서도 선수 검출 성능을 유지하는 player detector를 만드는 것이다.

기존 문제는 아래와 같았다.

- Roboflow로 학습한 모델이 실제 경기영상에서 선수와 이펙트가 겹치면 confidence가 급격히 낮아짐
- intro 화면, 스코어보드, 전환 장면이 학습 데이터에 섞이며 detector를 혼란시킴
- 실제 중요한 실패 장면과 Roboflow 내부 split 간 분포 차이가 큼

따라서 이번 단계에서는 모델 구조 변경보다 먼저, 실제 경기 hard frame을 체계적으로 다시 수집하고 relabeling 루프를 만드는 것을 목표로 했다.

## 2. 기준 데이터와 모델

### 2.1 Roboflow V4 정규화

원본 V4 export에서 클래스는 `['0', 'object', 'player']` 형태였다.

검토 결과:

- class `0`와 class `2`가 모두 시각적으로 player 의미
- class `1`은 object였지만 실제 사용 가능한 라벨이 거의 없었음

정규화 정책:

- `0 -> player`
- `2 -> player`
- `1 -> drop`

### 2.2 V4 player-only baseline

정규화 후 현재 leaderboard 기준 peak 성능:

- Precision `0.676`
- Recall `0.648`
- mAP50 `0.653`
- mAP50-95 `0.353`

이 값은 Track B의 현재 baseline이다.

## 3. 실제 영상 분석 파이프라인

이번에 만든 파이프라인은 세 단계다.

### 3.1 sampled inference

스크립트:

- `tools/run_player_model_on_video.py`

역할:

- 실제 경기영상을 일정 간격으로 샘플링
- 현재 모델 추론 수행
- 프레임별 detection count와 average confidence를 CSV로 기록

핵심 출력:

- `*_player_only_detections.csv`

### 3.2 hard frame mining

스크립트:

- `tools/mine_video_hard_frames.py`

선정 조건:

- `low_count`
- `high_count`
- `low_conf`

역할:

- suspicious frame만 잘라 JPG로 저장
- `hard_frame_manifest.csv` 생성
- contact sheet 생성

### 3.3 multi-video batch intake

스크립트:

- `tools/process_track_b_video_batch.py`

역할:

- 여러 경기 영상을 한 번에 처리
- detector CSV + hard frame mining + batch manifest를 연속 수행

### 3.4 export retrain orchestration

스크립트:

- `tools/retrain_track_b_export.py`

역할:

- Roboflow export를 player-only detect dataset으로 정규화
- YOLOv8 재학습 수행
- `clean/partial/heavy` 태그별 평가 수행
- hard player sample을 다시 추출해 다음 relabeling으로 연결

### 3.5 run leaderboard + best checkpoint selection

스크립트:

- `tools/compare_track_b_runs.py`

역할:

- `outputs/track_b_retrain_runs/*/run_summary.json`
- `results.csv`

를 다시 읽어서 버전별 leaderboard를 만든다.

주요 산출물:

- `outputs/track_b_retrain_runs/track_b_run_leaderboard.md`
- `outputs/track_b_retrain_runs/track_b_run_leaderboard.csv`
- `outputs/track_b_retrain_runs/track_b_current_best_checkpoint.txt`

의미:

- 다음 batch review나 hard mining에서 기본 checkpoint를 사람이 직접 고르지 않아도 됨
- 현재 기준 추천 run을 자동으로 유지할 수 있음

### 3.6 one-command batch review orchestration

스크립트:

- `tools/prepare_track_b_batch_review.py`

역할:

- batch video intake
- hard frame mining
- summary 생성
- priority CSV 생성
- Roboflow upload bundle 생성

을 한 번의 명령으로 묶는다.

### 3.7 skeleton verification preparation

스크립트:

- `tools/export_track_b_pose_review.py`

역할:

- 실제 경기영상에서 skeleton / action / court position review artifact를 생성
- CSV / JSONL / optional overlay mp4를 저장

이 단계의 목적:

- Track B가 `player-only detector 안정화` 다음에
- 실제 선수 위치와 action을 skeleton 기반으로 어느 정도 확인할 수 있는지 검증

## 4. 10경기 batch01 결과

### 4.1 입력

- source: Google Drive 10경기
- local path: `data/drive_imports/batch01`

### 4.2 처리 설정

- detector confidence: `0.10`
- frame stride: `30`
- max samples per match: `18`
- device: `auto` (`mps` available 시 자동 선택, 아니면 `cpu/cuda`)

### 4.3 결과 요약

- 총 처리 경기 수: `10`
- 총 추출 hard frame: `175`

confidence가 가장 낮은 경기:

1. `match02` — `0.2718`
2. `match09` — `0.2827`
3. `match01` — `0.2932`
4. `match06` — `0.2955`
5. `match010` — `0.3151`

해석:

- `match02`, `match09`는 relabeling 우선순위가 매우 높다.
- `match08`은 평균 confidence는 상대적으로 높지만 shield overlap과 high-count 장면이 많아 별도 가치가 있다.

## 5. 관측된 실패 유형

contact sheet 검토 결과 반복적으로 확인된 실패 패턴은 아래와 같다.

### 5.1 non-playable scene contamination

- roster
- scoreboard
- overtime title
- intro transition

이 장면들은 detector miss처럼 보일 수 있지만 실제로는 학습에 꼭 넣을 필요가 없다.

### 5.2 effect overlap under-confidence

- shield가 선수 torso를 가릴 때 confidence가 크게 감소
- projectile, UI, 광원 효과가 함께 겹치면 box가 흔들림

### 5.3 over-detection

- shield edge나 효과 경계가 player 후보로 잘못 검출되는 경우가 있음
- high-count 장면은 중복 box 정리용 relabeling 가치가 큼

## 6. Roboflow relabeling 전략

이번 라운드 전략은 의도적으로 단순하다.

### 6.1 유지할 것

- `player-only`
- tight and consistent box policy
- playable frame 우선

### 6.2 아직 하지 않을 것

- effect/object 멀티클래스 확장
- segmentation 전환
- 지나치게 많은 negative frame 포함

이유:

- 현재 병목은 class taxonomy보다 player visibility consistency에 있다.
- 따라서 먼저 player-only V5를 안정화해야 한다.

## 7. 업로드 번들 구성

### 7.1 phase1

- 파일:
  - `data/track_b_batch_review/roboflow_upload_bundle_batch01_phase1.zip`
  - `data/track_b_batch_review/roboflow_upload_bundle_batch01_phase1_manifest.csv`
- 프레임 수:
  - `36`
- 구성:
  - `low_conf` `30`
  - `high_count` `6`

이 묶음은 가장 먼저 Roboflow에 올릴 compact batch다.

### 7.2 priority

- 파일:
  - `data/track_b_batch_review/roboflow_upload_bundle_batch01_priority.zip`
  - `data/track_b_batch_review/roboflow_upload_bundle_batch01_priority_manifest.csv`
- 프레임 수:
  - `90`
- 구성:
  - `label_priority` `36`
  - `review_or_skip` `54`

이 묶음은 phase1 뒤에 확장용으로 사용한다.

## 8. 다음 실험 설계

권장 다음 단계:

1. batch02 playable-bias 40장 relabeling
2. Roboflow V5 export
3. `tools/retrain_track_b_export.py`로 V5 player-only retrain
4. leaderboard에서 V4 대비 정량 비교
5. 실제 경기영상 재평가
6. detector가 어느 정도 안정되면 skeleton verification smoke 진행

참고:

- 2026-06-09에 V4 export 기준 `--skip-train` smoke와 `--epochs 1` training smoke를 모두 통과시켰다.
- `1 epoch` 결과는 학습 경로 검증용이므로 성능 판단 근거로 쓰지 않는다.

정량 비교 시 최소 확인 항목:

- Precision
- Recall
- mAP50
- mAP50-95
- leaderboard score
- 실제 영상 average nonzero confidence 변화
- low-count frame 감소 여부
- over-detection 장면 감소 여부

## 9. 결론

이번 단계의 성과는 “모델 성능 개선” 자체보다 “실제 실패 장면을 재학습 루프로 다시 연결하는 데이터 파이프라인”을 만든 데 있다.

현재 상태에서 가장 합리적인 기술 방향은 아래와 같다.

- Track B는 player-only V5를 먼저 안정화한다.
- 실제 영상 hard frame 기반 relabeling을 반복한다.
- 그 다음에야 segmentation 또는 effect class 실험으로 넘어간다.

즉, 지금 Track B의 병목은 모델보다 데이터 루프이며, 이번 작업은 그 병목을 줄이는 기반 작업으로 의미가 있다.
