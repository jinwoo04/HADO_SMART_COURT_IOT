# Track B Current Action Packet — 2026-06-13

이 문서는 Track A 현장 테스트 패킷을 마친 뒤, 바로 이어서 진행할 **Track B 전용 작업 패킷**이다.

Track B 목표:

- 실제 HADO 경기영상에서
- AR effect / shield / projectile이 선수와 겹치는 상황에서도
- `player` 검출을 안정화하고
- 이후 skeleton/action verification으로 넘어갈 수 있는 기반을 만든다.

Track A 발표용 Pi4 tactical demo와 섞지 않는다.

## 1. 현재 기준선

자동 leaderboard를 다시 생성한 결과, 현재 best는 여전히 `trackb_v4_player_only`다.

| Run | Score | Precision | Recall | mAP50 | mAP50-95 | Images |
|---|---:|---:|---:|---:|---:|---:|
| `trackb_v4_player_only` | 53.3439 | 0.6761 | 0.6479 | 0.6533 | 0.3530 | 87 |
| `trackb_v3_player_only` | 50.7705 | 0.7121 | 0.6596 | 0.6225 | 0.3328 | 87 |
| `trackb_v1_player_only` | 37.0888 | 0.0151 | 0.8378 | 0.4954 | 0.2227 | 36 |
| `trackb_v2_player_only` | 36.2957 | 0.8219 | 0.4324 | 0.5183 | 0.2295 | 41 |

현재 best checkpoint:

```text
outputs/track_b_retrain_runs/trackb_v4_player_only/train/yolov8n_player_only/weights/best.pt
```

자동 pointer:

```text
outputs/track_b_retrain_runs/track_b_current_best_checkpoint.txt
```

재생성 명령:

```bash
./hado_venv/bin/python tools/compare_track_b_runs.py
```

## 2. 현재 가장 중요한 실패 패턴

batch02 평가 결과:

- 영상: 10개
- 처리 프레임: 994개
- hard frame: 130개
- 핵심 실패: **over-detection**

즉, 지금 문제는 선수를 못 찾는 것보다,
AR effect / shield / projectile / glow를 `player`로 잘못 잡는 쪽이 더 크다.

가장 많이 나온 hard-frame reason:

| Reason | Count | 의미 |
|---|---:|---|
| `high_count_10` | 60 | 실제보다 player를 너무 많이 잡음 |
| `high_count_11` | 27 | effect/shield duplicate 가능성 높음 |
| `high_count_12` | 21 | 심한 over-detection |
| `high_count_13` | 7 | 최악 케이스 |
| `high_count_14` | 4 | 최악 케이스 |

따라서 v5의 relabel 목표는:

> "선수를 더 많이 찾기"가 아니라 "effect를 player로 착각하는 박스를 줄이기"다.

## 3. 지금 Roboflow에서 relabel할 대상

업로드 대상:

```text
data/track_b_batch_review/batch02_v4_eval/roboflow_upload_bundle_v2_playable_bias.zip
```

또는 폴더:

```text
data/track_b_batch_review/batch02_v4_eval/roboflow_upload_bundle_v2_playable_bias/frames
```

구성:

- 총 40장
- match01~match10에서 각 4장씩 균등 선발
- 전부 `label_priority`
- 대부분 high-count/effect-overlap 케이스

Roboflow 프로젝트:

- `hado-track-b-player-only`
- Object Detection
- 클래스는 `player` 하나만 사용
- 이 단계에서 `effect`, `shield`, `projectile` 클래스를 만들지 않는다

## 4. 생성된 review packet

Roboflow에 들어가기 전에 볼 로컬 검토 자료를 생성했다.

```text
outputs/track_b_relabel_review_packet/batch02_v4_playable_bias/relabel_contact_sheet.jpg
outputs/track_b_relabel_review_packet/batch02_v4_playable_bias/relabel_checklist.csv
outputs/track_b_relabel_review_packet/batch02_v4_playable_bias/README.md
```

재생성 명령:

```bash
./hado_venv/bin/python tools/build_track_b_relabel_review_packet.py \
  --manifest "data/track_b_batch_review/batch02_v4_eval/roboflow_upload_bundle_v2_playable_bias/upload_manifest.csv" \
  --out-dir "outputs/track_b_relabel_review_packet/batch02_v4_playable_bias" \
  --cols 4
```

## 5. contact sheet 1차 육안 검토 결과

전체적으로 v5 relabel에 적합하다.

주요 패턴:

- shield/effect 주변에 가짜 `player`가 많이 생김
- 중앙 HADO 로고/타이틀이 선수처럼 잡히는 장면이 있음
- projectile/glow가 작은 사람 박스처럼 검출되는 장면이 있음
- 실제 선수 몸이 effect 뒤에 일부만 보이는 장면이 많음

Roboflow에서 특히 주의할 review order:

| Review order | 판단 | 작업 |
|---:|---|---|
| 1~6 | high-value playable hard frame | effect/shield fake player 삭제, 실제 선수만 tight box |
| 7 | HADO title overlay 가능성 | playable인지 확인 후 skip 후보 |
| 12 | 큰 shield overlap | shield 경계는 box 금지, 보이는 선수 몸만 |
| 17 | HADO title overlay 가능성 | playable인지 확인 후 skip 후보 |
| 21 | yellow particles 많음 | particles는 player box 금지 |
| 27 | dual shield overlap | 가짜 shield box 삭제, real player만 |
| 29~30 | HADO title overlay 가능성 | skip 후보 |
| 36 | 낮은 confidence/high-count 혼재 | 실제 선수만 보수적으로 |
| 39 | scoreboard/full-screen UI | skip 후보 |

주의:

- 위 skip 후보도 완전 삭제 명령이 아니라, Roboflow에서 실제 이미지를 열어 playable 여부를 확인하라는 뜻이다.
- 플레이 장면이고 사람이 명확하면 `player`를 유지한다.
- full-screen UI나 HADO 타이틀 중심이면 학습 positive로 억지로 살리지 않는다.

## 6. v5 relabel 규칙

반드시 지킬 것:

1. 클래스는 `player` 하나만 쓴다.
2. effect, shield, projectile, glow에는 box를 만들지 않는다.
3. 실제 선수는 보이는 몸 기준으로 tight box를 만든다.
4. 몸 일부만 보여도 사람임이 명확하면 보이는 부분 기준으로 box를 유지한다.
5. 화면 전환, scoreboard, title-only 장면은 skip 후보로 둔다.
6. 같은 장면군에서는 box 크기 기준을 일관되게 유지한다.

하면 안 되는 것:

- effect까지 크게 감싸는 player box
- shield 외곽선을 player로 라벨링
- projectile을 player로 라벨링
- 애매한 UI 장면을 억지 positive로 유지
- 이번 단계에서 effect class 추가

## 7. v5 export 후 로컬 재학습

Roboflow에서 relabel이 끝나면 YOLOv8 format으로 export한다.

다운로드 폴더 예시:

```text
/Users/jinu/Downloads/hado-track-b-player-only.v5-trackb_v5_batch02_40.yolov8
```

재학습:

```bash
./hado_venv/bin/python tools/retrain_track_b_export.py \
  --export-src "/Users/jinu/Downloads/hado-track-b-player-only.v5-trackb_v5_batch02_40.yolov8" \
  --run-name "trackb_v5_player_only" \
  --epochs 35 \
  --imgsz 416 \
  --batch 16
```

학습 후 반드시 실행:

```bash
./hado_venv/bin/python tools/compare_track_b_runs.py
```

성공 기준:

- `trackb_v5_player_only`가 leaderboard에서 v4보다 위로 올라오면 성공
- 최소 목표: mAP50 `0.68+`
- precision이 크게 무너지면 실패로 본다
- recall만 오르고 high-count가 늘면, effect를 더 많이 player로 착각한 것이므로 실패다

## 8. v5 후 실제 영상 재평가

v5가 좋아 보이면 같은 batch02를 다시 평가한다.

```bash
./hado_venv/bin/python tools/prepare_track_b_batch_review.py \
  --video-dir "data/drive_imports/batch02_raw" \
  --out-dir "data/track_b_batch_review/batch02_v5_eval" \
  --frame-stride 30 \
  --max-samples 18 \
  --bundle-make-zip
```

비교할 항목:

- hard frame 130개보다 줄었는가
- high_count 계열이 줄었는가
- match08/match02 severity가 줄었는가
- skip 후보 UI 장면이 줄었는가

## 9. skeleton phase로 넘어가는 조건

바로 skeleton/action으로 넘어가도 되지만,
판단 기준은 아래처럼 둔다.

skeleton verification을 본격화해도 되는 조건:

- player-only v5/v6가 v4보다 high-count를 줄인다
- 실제 경기영상에서 선수 bbox가 대략 안정적이다
- over-detection이 줄어 pose crop이 덜 흔들린다

실행 예시:

```bash
./hado_venv/bin/python tools/export_track_b_pose_review.py \
  --video "data/drive_imports/batch02_raw/match08.mp4" \
  --out-dir "data/track_b_pose_review/batch02_match08_v5_probe" \
  --pt \
  --device mps \
  --frame-stride 6 \
  --max-frames 120
```

먼저 볼 영상:

1. `match08.mp4` — severity 최고, effect overlap 많음
2. `match02.mp4` — high-count + under-detect 혼재
3. `match07.mp4` — 상대적으로 양호한 baseline 비교용

## 10. 다음 작업 순서

현재 가장 자연스러운 순서:

1. Roboflow에서 batch02 40장 v5 relabel
2. v5 YOLOv8 export 다운로드
3. `tools/retrain_track_b_export.py`로 `trackb_v5_player_only` 학습
4. leaderboard 비교
5. batch02를 v5로 다시 평가
6. match08/match02/match07 skeleton probe 실행
7. player-only 안정화가 부족하면 v6 relabel 루프 반복

한 줄로 요약하면:

> 지금 Track B는 effect class 확장 단계가 아니라, player-only detector가 AR/effect overlap에 덜 속도록 만드는 v5 relabel/retrain 단계다.
