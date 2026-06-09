# Track B Operator Playbook — 2026-06-10

이 문서는 Track B를 실제로 굴리는 작업자 기준 운영 문서다.

목표는 단순하다.

- 실제 경기영상에서 실패 장면을 찾는다.
- Roboflow에서 `player-only`로 다시 라벨링한다.
- 로컬에서 재학습한다.
- 이전 버전보다 진짜 좋아졌는지 비교한다.

Track A 발표용 내용은 여기서 완전히 제외한다.

## 1. 현재 원칙

- 현재 단계는 `player-only`
- `effect`, `shield`, `projectile` 멀티클래스는 아직 하지 않음
- playable frame 우선
- intro / scoreboard / roster / full-screen UI는 억지로 positive로 살리지 않음
- 실제 영상 domain gap을 줄이는 것이 우선

## 2. 현재 기준선

현재 자동 leaderboard 기준 추천 checkpoint:

- pointer:
  `outputs/track_b_retrain_runs/track_b_current_best_checkpoint.txt`
- leaderboard:
  `outputs/track_b_retrain_runs/track_b_run_leaderboard.md`

2026-06-10 기준 추천 run:

- `trackb_v4_player_only`

## 3. 반복 루프

Track B는 아래 6단계를 반복한다.

1. 새 경기영상 intake
2. hard frame mining
3. Roboflow relabeling
4. export download
5. local retrain
6. leaderboard 갱신 + 다음 실패 장면 재수집

## 4. 새 경기영상이 들어왔을 때

영상 폴더 예시:

- `data/drive_imports/batch02_raw`
- `data/drive_imports/batch03_raw`

가장 쉬운 실행 명령:

```bash
./hado_venv/bin/python tools/prepare_track_b_batch_review.py \
  --video-dir "data/drive_imports/<batch_name>" \
  --out-dir "data/track_b_batch_review/<batch_eval_name>" \
  --frame-stride 30 \
  --max-samples 18 \
  --bundle-make-zip
```

설명:

- 현재 추천 checkpoint는 자동으로 선택됨
- batch summary, priority CSV, upload bundle, zip이 함께 생성됨

주요 출력:

- `batch_manifest.csv`
- `batch_summary.md`
- `batch_priority.csv`
- `roboflow_upload_bundle/`
- `roboflow_upload_bundle.zip`

## 5. Roboflow에 올릴 때

업로드 대상은 항상 bundle의 `frames/` 폴더 또는 zip이다.

원칙:

- 이번 단계는 `player`만 라벨링
- shield에 가려져도 보이는 선수 몸 기준으로 박스
- effect 경계선까지 크게 감싸지 않기
- non-playable frame은 스킵하거나 매우 보수적으로 제외

한 라운드에서 너무 많이 올리지 않는다.

권장량:

- 첫 pass: `30~40장`
- follow-up: `40~90장`

## 6. Roboflow export를 받았을 때

다운로드 폴더 예시:

- `/Users/jinu/Downloads/hado-track-b-player-only.v5.yolov8`

재학습 명령:

```bash
./hado_venv/bin/python tools/retrain_track_b_export.py \
  --export-src "/Users/jinu/Downloads/<export_folder>" \
  --run-name "trackb_v5_player_only" \
  --epochs 30 \
  --batch 16
```

이 명령이 하는 일:

- export를 player-only detect dataset으로 정규화
- YOLOv8 재학습
- hard sample mining
- run summary 저장
- run leaderboard 자동 갱신
- 추천 checkpoint pointer 자동 갱신

## 7. 재학습 후 반드시 볼 것

### 7.1 leaderboard

- `outputs/track_b_retrain_runs/track_b_run_leaderboard.md`

확인할 것:

- 새 run이 기존 V4보다 위에 올라왔는지
- `mAP50-95`가 실제로 개선됐는지
- recall만 높고 precision이 무너진 것은 아닌지

### 7.2 개별 run summary

- `outputs/track_b_retrain_runs/<run_name>/run_summary.md`
- `outputs/track_b_retrain_runs/<run_name>/run_summary.json`

확인할 것:

- dataset 크기
- 사용한 export source
- hard mining에서 다시 어떤 프레임이 걸렸는지

### 7.3 실제 영상 재평가

새 모델이 leaderboard에서 좋아 보여도 실제 영상 재평가는 다시 해야 한다.

가장 쉬운 방법:

```bash
./hado_venv/bin/python tools/prepare_track_b_batch_review.py \
  --video-dir "data/drive_imports/<same_or_new_batch>" \
  --out-dir "data/track_b_batch_review/<new_eval_name>" \
  --bundle-make-zip
```

## 8. 의사결정 규칙

### 계속 player-only로 갈 때

아래 조건이면 계속 `player-only`가 맞다.

- 실제 영상에서 여전히 선수를 놓침
- over-detection이 effect overlap에서 자주 발생
- 라벨 정책이 아직 완전히 일관되지 않음

### effect 멀티클래스로 넓힐 수 있는 시점

아래 조건이 충족될 때만 고려한다.

- player-only baseline이 실제 영상에서 꽤 안정적
- hard frame 중 상당수가 “선수 miss”가 아니라 “effect 해석 부족” 문제로 남음
- effect taxonomy를 일관되게 정의할 수 있음

지금은 아직 여기까지 가지 않는다.

## 9. 우선순위 기준

새 batch를 봤을 때 우선순위는 아래 순서로 잡는다.

1. playable + heavy overlap
2. high-count over-detection
3. low-confidence player visibility
4. low-count miss
5. intro / scoreboard / end screen

즉, “무조건 miss 프레임”이 아니라 “학습 가치가 높은 실제 플레이 장면”이 먼저다.

## 10. 실수 방지 메모

- Track A 발표 파일과 섞지 않기
- export zip, 원본 영상, 대용량 모델은 git에 올리지 않기
- `run_summary.json`만 보고 끝내지 말고 leaderboard도 같이 보기
- Roboflow에서 클래스가 늘어나지 않았는지 매번 확인하기
- 새 버전 이름은 `trackb_v5_player_only`, `trackb_v6_player_only`처럼 고정하기

## 11. 지금 바로 다음에 할 일

현재 기준으로 가장 가까운 다음 작업:

1. batch02 upload bundle을 Roboflow에 업로드
2. `player-only` relabel
3. 새 export 다운로드
4. `retrain_track_b_export.py` 실행
5. leaderboard에서 V4 대비 비교

핵심은 이거다.

Track B는 지금 “새 데이터를 더 많이 넣는 단계”라기보다,
“실제 실패 장면을 더 정확히 다시 학습시키는 루프를 안정화하는 단계”다.
