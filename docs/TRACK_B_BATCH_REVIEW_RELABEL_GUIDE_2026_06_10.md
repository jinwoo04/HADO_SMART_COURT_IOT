# Track B Batch Review + Roboflow Relabel Guide — 2026-06-10

이 문서는 Track B를 처음 이어받는 협업자가
`batch review가 정확히 뭔지`, `Roboflow에서 무엇을 relabel해야 하는지`
바로 이해하고 실행할 수 있도록 만든 실무 가이드다.

Track A 발표용 내용은 포함하지 않는다.

## 1. 한 줄 요약

Track B의 현재 목표는 이것이다.

- 실제 경기영상에서
- AR effect / shield / projectile이 선수와 겹쳐도
- `player` 박스를 더 안정적으로 검출하게 만들기

그래서 지금 하는 일은:

1. 현재 모델이 경기영상을 한번 본다.
2. 이상하게 본 프레임만 다시 뽑는다.
3. 그 프레임들만 Roboflow에서 `player-only`로 수정한다.
4. 그 수정본으로 다시 재학습한다.

## 2. batch review가 의미하는 것

batch review는
“경기영상 여러 개를 현재 player-only 모델로 자동 검사해서,
어떤 프레임을 다시 라벨링해야 효과가 큰지 고르는 단계”다.

사람이 경기영상 전체를 처음부터 끝까지 다 보는 대신,
모델이 먼저 아래 같은 실패 장면을 뽑아준다.

- `high_count_*`
  - 실제 선수 수보다 훨씬 많이 잡은 장면
  - 보통 effect / shield / duplicate box 문제
- `low_count_*`
  - 실제로는 선수가 보여야 하는데 적게 잡은 장면
  - miss 가능성
- `low_conf_*`
  - 박스는 떴지만 confidence가 너무 낮은 장면
  - 흐림, partial visibility, effect overlap 가능성

즉 batch review의 목적은
“지금 다시 수정해야 할 프레임만 추려내는 것”이다.

## 3. 현재 협업자가 가장 먼저 볼 폴더

현재 기준 batch02 평가 결과는 아래 폴더에 있다.

- [batch02_v4_eval](/Users/jinu/iot project/hado-smart-court-iot/data/track_b_batch_review/batch02_v4_eval)

여기서 가장 중요한 파일:

- 요약 문서:
  - [batch02_summary.md](/Users/jinu/iot project/hado-smart-court-iot/data/track_b_batch_review/batch02_v4_eval/batch02_summary.md)
- 우선순위 CSV:
  - [batch02_priority.csv](/Users/jinu/iot project/hado-smart-court-iot/data/track_b_batch_review/batch02_v4_eval/batch02_priority.csv)
- Roboflow 업로드용 묶음:
  - [roboflow_upload_bundle_v2_playable_bias](/Users/jinu/iot project/hado-smart-court-iot/data/track_b_batch_review/batch02_v4_eval/roboflow_upload_bundle_v2_playable_bias)
- 실제 업로드할 이미지:
  - [frames](/Users/jinu/iot project/hado-smart-court-iot/data/track_b_batch_review/batch02_v4_eval/roboflow_upload_bundle_v2_playable_bias/frames)
- 각 이미지 설명:
  - [upload_manifest.csv](/Users/jinu/iot project/hado-smart-court-iot/data/track_b_batch_review/batch02_v4_eval/roboflow_upload_bundle_v2_playable_bias/upload_manifest.csv)

## 4. Roboflow에서 실제로 relabel할 대상

이번 라운드에서 협업자가 Roboflow에서 수정해야 하는 것은
경기영상 전체가 아니라 아래 `40장`이다.

- [frames](/Users/jinu/iot project/hado-smart-court-iot/data/track_b_batch_review/batch02_v4_eval/roboflow_upload_bundle_v2_playable_bias/frames)

중요:

- 클래스는 이번 단계에서 `player` 하나만 유지한다.
- `effect`, `shield`, `projectile` 클래스는 만들지 않는다.

## 5. Roboflow에서 무엇을 수정해야 하는가

각 프레임에서 협업자가 보는 포인트는 단순하다.

### 5.1 가짜 player 박스 삭제

아래 경우는 `player`가 아니다.

- effect 광원
- shield 경계
- projectile
- 겹쳐진 이펙트가 사람처럼 보인 가짜 detection

이런 박스는 지운다.

### 5.2 실제 선수인데 박스가 없으면 추가

실제 보이는 선수인데 box가 없으면
새 `player` 박스를 추가한다.

### 5.3 박스가 너무 크면 줄이기

effect나 shield까지 같이 감싼 box는 좋지 않다.

원칙:

- 보이는 선수 몸 기준
- 너무 공격적으로 크게 잡지 않기
- 같은 유형 장면끼리 box 스타일을 일관되게 유지

### 5.4 일부만 보여도 사람임이 명확하면 유지

선수 몸 일부만 보여도
“명확히 사람이다”라고 판단되면
보이는 범위 기준으로 `player` 박스를 유지한다.

## 6. 어떤 장면은 스킵해도 된다

아래 장면은 굳이 좋은 positive example로 살리지 않아도 된다.

- intro
- roster
- scoreboard-only
- full-screen UI
- 경기 장면이 아닌 전환 화면

즉,
“무조건 다 라벨링”이 아니라
“실제로 학습 가치가 있는 playable frame 우선”이 원칙이다.

## 7. 왜 이 40장을 먼저 보나

이 40장은 임의 샘플이 아니라,
현재 모델이 가장 헷갈린 hard frame 중
playable bias 기준으로 추린 우선 묶음이다.

현재 batch02에서 특히 중요한 패턴은:

- miss보다
- `over-detection`

즉 이번 relabel의 핵심 목적은
“선수를 더 많이 찾게 하기”보다
“effect를 player로 착각하는 문제를 줄이는 것”에 더 가깝다.

## 8. 협업자 작업 순서

### Step 0. review packet 확인

Roboflow를 열기 전에 아래 명령으로 review packet을 생성한다.

```bash
./hado_venv/bin/python tools/build_track_b_relabel_review_packet.py \
  --manifest "data/track_b_batch_review/batch02_v4_eval/roboflow_upload_bundle_v2_playable_bias/upload_manifest.csv" \
  --out-dir "outputs/track_b_relabel_review_packet/batch02_v4_playable_bias" \
  --cols 4
```

먼저 확인할 파일:

- `outputs/track_b_relabel_review_packet/batch02_v4_playable_bias/relabel_contact_sheet.jpg`
- `outputs/track_b_relabel_review_packet/batch02_v4_playable_bias/relabel_checklist.csv`

contact sheet의 `#order` 번호와 checklist의 `review_order`가 대응된다.

주의:

- contact sheet에서 scoreboard, roster, intro처럼 non-playable로 보이는 이미지는 Roboflow에서 skip 후보로 본다.
- high-count 장면은 대부분 effect/shield가 player로 잘못 잡힌 케이스이므로 가짜 player box 삭제가 핵심이다.

### Step 1. 업로드

아래 폴더 또는 zip을 Roboflow 프로젝트에 업로드한다.

- [frames](/Users/jinu/iot project/hado-smart-court-iot/data/track_b_batch_review/batch02_v4_eval/roboflow_upload_bundle_v2_playable_bias/frames)
- 또는
- [roboflow_upload_bundle_v2_playable_bias.zip](/Users/jinu/iot project/hado-smart-court-iot/data/track_b_batch_review/batch02_v4_eval/roboflow_upload_bundle_v2_playable_bias.zip)

### Step 2. 클래스 확인

반드시 `player` 하나만 쓰고 있는지 확인한다.

### Step 3. 이미지별 수정

각 이미지에서:

- 가짜 player 삭제
- 실제 선수 추가
- player box를 visible body 기준으로 정리

### Step 4. 새 버전 export

수정 완료 후 새 Roboflow version을 export한다.

권장 이름 예시:

- `trackb_v5_player_only`
- `trackb_v6_player_only`

### Step 5. 로컬 재학습 전달

export 폴더나 ZIP을 로컬로 가져와 아래 명령으로 재학습한다.

```bash
./hado_venv/bin/python tools/retrain_track_b_export.py \
  --export-src "/Users/jinu/Downloads/<export_folder>" \
  --run-name "trackb_v5_player_only" \
  --epochs 30 \
  --batch 16
```

## 9. 애매할 때 의사결정 기준

판단이 애매하면 아래 순서로 본다.

1. 실제 사람 몸인가?
2. playable scene인가?
3. effect 때문에 생긴 가짜 box인가?
4. visible body 기준으로 보수적으로 box를 줄일 수 있는가?

간단 기준:

- 선수로 보인다 -> `player`
- effect / shield / 빛이다 -> 박스 없음
- 선수 일부만 보여도 명확하다 -> 보이는 몸 기준 `player`
- 경기 장면이 아니다 -> 스킵 가능

## 10. 협업자에게 바로 전달할 짧은 설명

아래 설명을 그대로 전달해도 된다.

---

We are not relabeling full match videos from scratch.

For Track B, the current workflow is:

1. run the current player-only model on real match videos
2. mine hard frames where the detector fails
3. upload only those selected frames to Roboflow
4. relabel `player` boxes only
5. export the new version
6. retrain locally and compare against the current baseline

For the current round, please relabel only the images inside:

- `data/track_b_batch_review/batch02_v4_eval/roboflow_upload_bundle_v2_playable_bias/frames`

Rules:

- keep this pass player-only
- remove fake player boxes caused by AR effects or shields
- add missing player boxes when the player is clearly visible
- keep boxes tight around the visible player body
- non-playable scenes can be skipped

---

## 11. 관련 문서

- [TRACK_B_OPERATOR_PLAYBOOK_2026_06_10.md](/Users/jinu/iot project/hado-smart-court-iot/docs/TRACK_B_OPERATOR_PLAYBOOK_2026_06_10.md)
- [TRACK_B_HANDOFF_REPORT_2026_06_09.md](/Users/jinu/iot project/hado-smart-court-iot/docs/TRACK_B_HANDOFF_REPORT_2026_06_09.md)
- [TRACK_B_TECHNICAL_REPORT_2026_06_09.md](/Users/jinu/iot project/hado-smart-court-iot/docs/TRACK_B_TECHNICAL_REPORT_2026_06_09.md)
