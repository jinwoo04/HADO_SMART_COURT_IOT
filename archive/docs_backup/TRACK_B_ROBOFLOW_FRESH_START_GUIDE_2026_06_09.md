# Track B Roboflow Fresh Start Guide — 2026-06-09

이 문서는 Track B용 Roboflow 프로젝트를 처음부터 새로 만드는 기준 문서다.

목표는 하나다.

- Track A 발표/데모용 내용과 완전히 분리
- Track B AR/effect occlusion 연구만 위한 데이터셋 운영
- relabeling -> export -> retrain 루프를 다시 깔끔하게 시작

---

## 1. 왜 새 프로젝트로 다시 시작하는가

현재 상태에서는 아래 문제가 섞여 있다.

- 발표용 Track A와 개인 연구용 Track B 목적이 다름
- playable frame과 non-playable frame 기준이 혼재됨
- class 의미가 일관되지 않았던 이력(V4의 `0`, `object`, `player`)이 남아 있음
- 앞으로는 Roboflow 프로젝트 내부에서부터 정책을 더 단순하게 가져가는 편이 유리함

따라서 새 프로젝트에서는 처음부터 아래 원칙으로 간다.

- 프로젝트 목적: `player-only`
- 현재 단계에서는 `effect` 클래스 만들지 않음
- 실제 플레이 장면 중심
- hard frame relabeling 품질을 최우선

---

## 2. 새 프로젝트 이름 권장안

아래처럼 바로 목적이 드러나는 이름으로 만드는 것을 권장한다.

### 권장 프로젝트명

- `hado-track-b-player-only`

### 선택 가능한 대안

- `hado-occlusion-player-only`
- `hado-track-b-v5-reset`

### 설명 문구

- `Player detection for HADO match footage under AR/effect occlusion`

프로젝트명에 `track-b`, `player-only`, `occlusion` 셋 중 최소 두 개는 반드시 들어가게 하는 것이 좋다.

---

## 3. Roboflow에서 처음 만들 때 선택값

새 프로젝트 생성 시 아래처럼 맞춘다.

### Project Type

- `Object Detection`

### What are you detecting

- `player`

### Visibility / Privacy

- 유료 플랜이면 `private`
- 무료 플랜이면 public 노출 가능성을 반드시 확인

### 초기 클래스 정책

- `player` 하나만 사용

지금 단계에서는 아래는 하지 않는다.

- `effect`
- `object`
- `shield`
- `projectile`
- segmentation

---

## 4. 새 프로젝트를 만든 직후 해야 할 폴더/배치 규칙

Roboflow 안에서 데이터가 다시 섞이지 않게 하려면 배치 이름부터 고정해야 한다.

### 배치 이름 규칙

- `trackb_phase1_2026_06`
- `trackb_priority_2026_06`
- `trackb_match02_followup`
- `trackb_match09_followup`

### 하지 말아야 할 배치 이름

- `test`
- `new`
- `upload1`
- `hado`
- `ppt`

배치명만 보고도 “무슨 라운드인지” 알 수 있어야 한다.

---

## 5. 첫 업로드는 무엇으로 할지

새 프로젝트의 첫 업로드는 이미 만들어 둔 작은 묶음부터 가는 것이 가장 안전하다.

### 1차 권장 업로드

- `data/track_b_batch_review/roboflow_upload_bundle_batch01_phase1.zip`

이유:

- 36장이라 관리가 쉬움
- 지금 가장 중요한 low-confidence / high-count 장면 위주
- 라벨 정책을 다시 바로잡기 좋음

### 2차 확장 업로드

- `data/track_b_batch_review/roboflow_upload_bundle_batch01_priority.zip`

이건 1차 정책이 안정화된 뒤 넣는 것이 좋다.

---

## 6. 라벨링 규칙

이 문서에서 가장 중요한 부분이다.

새 프로젝트에서는 라벨링 규칙을 아래처럼 단순하게 고정한다.

### 6.1 클래스

- `player` 하나만 쓴다

### 6.2 박스 기준

- 실제로 보이는 선수 몸 기준으로 박스를 준다
- shield, projectile, AR effect와 겹쳐도 선수 torso/몸통 중심으로 일관되게 잡는다
- 박스를 너무 크게 잡아서 effect까지 포함하지 않는다
- 박스를 너무 작게 잡아서 팔/몸 일부만 따지지 않는다

### 6.3 포함할 프레임

- 실제 플레이 장면
- 선수 위치와 자세가 의미 있는 장면
- occlusion이 있더라도 사람이 분명히 보이는 장면

### 6.4 제외 또는 매우 보수적으로 다룰 프레임

- roster 화면
- intro transition
- scoreboard full-screen
- overtime title
- 경기 준비 전/후 UI 위주 장면

이런 프레임은 positive training 예시로 억지로 살리는 것보다 빼는 편이 낫다.

---

## 7. Review 기준

라벨링 후 review할 때는 아래만 체크하면 된다.

### 체크리스트

1. 클래스가 `player` 외에 섞이지 않았는가
2. 한 선수에 중복 박스가 들어가지 않았는가
3. effect 경계선을 player로 잘못 잡지 않았는가
4. non-playable frame을 억지로 살리지 않았는가
5. 같은 장면군에서 박스 크기 기준이 일관적인가

### 가장 흔한 실수

- effect 빛 번짐까지 player box에 포함
- 화면 전환 장면을 굳이 training positive로 유지
- 아주 멀리 있는 선수를 프레임마다 기준 다르게 처리

---

## 8. 버전 관리 규칙

버전 이름도 처음부터 규칙을 고정한다.

### 권장 버전 흐름

- `V1`: phase1 relabel only
- `V2`: phase1 + priority
- `V3`: match02, match09 follow-up 포함

### 버전 메모에 남길 것

- 어떤 배치를 넣었는지
- 라벨 정책에서 달라진 점이 있는지
- non-playable frame을 얼마나 제외했는지

예:

- `phase1 36 frames relabeled with strict player-only boxes`
- `priority 90 frames added, scoreboard-like negatives excluded`

---

## 9. preprocessing / augmentation 권장 시작점

처음 새로 시작하는 버전에서는 과하게 건드리지 않는 것이 좋다.

### 권장 시작점

- Resize: 기본 사용
- Auto-orient: 기본값 유지
- Augmentation: 아주 강하게 주지 않기

### 처음 버전에서 피할 것

- 공격적인 crop
- 과한 mosaic 성격 augmentation
- 색 변화 augmentation 과다 사용

지금 단계에서는 augmentation으로 문제를 덮기보다, hard frame 라벨 품질을 올리는 편이 더 중요하다.

---

## 10. 로컬 쪽 폴더 운영 규칙

Roboflow 프로젝트를 새로 시작해도 로컬 파일이 섞이면 다시 무너진다.

따라서 로컬도 아래처럼 나눈다.

### 유지할 핵심 입력

- `data/track_b_batch_review/roboflow_upload_bundle_batch01_phase1.zip`
- `data/track_b_batch_review/roboflow_upload_bundle_batch01_priority.zip`

### 새 export 저장 권장 위치

- `/Users/jinu/Downloads/hado-track-b-player-only.v1i.yolov8`
- `/Users/jinu/Downloads/hado-track-b-player-only.v2i.yolov8`

### 저장 규칙

- export 폴더명에 `track-b-player-only`를 넣기
- `v1`, `v2` 같은 버전 번호를 반드시 넣기
- 기존 `hado-player.v4i.yolov8 (1)` 같은 모호한 이름은 다시 만들지 않기

---

## 11. export를 받은 뒤 로컬에서 할 일

새 프로젝트에서 export를 받으면 아래 명령으로 바로 다음 루프를 탄다.

```bash
./hado_venv/bin/python tools/retrain_track_b_export.py \
  --export-src "<새 export 폴더>" \
  --run-name "trackb_v1_player_only" \
  --epochs 30 \
  --batch 16
```

다음 버전이면:

```bash
./hado_venv/bin/python tools/retrain_track_b_export.py \
  --export-src "<새 export 폴더>" \
  --run-name "trackb_v2_player_only" \
  --epochs 30 \
  --batch 16
```

이 스크립트는 아래를 이어서 수행한다.

- export를 player-only detect dataset으로 정규화
- YOLOv8 retrain
- `clean/partial/heavy` 태그별 평가
- 다음 relabeling용 hard sample 추출

---

## 12. 새 프로젝트 운영 순서

실제 작업 순서는 아래 그대로 가면 된다.

1. Roboflow에서 새 Object Detection 프로젝트 생성
2. 프로젝트 이름을 `hado-track-b-player-only` 계열로 설정
3. 클래스는 `player` 하나만 사용
4. `phase1` ZIP 36장 업로드
5. 라벨링 + review
6. `V1` version 생성
7. export
8. 로컬에서 `retrain_track_b_export.py` 실행
9. `run_summary.md`, `tag_eval`, `hard_mining` 확인
10. 필요한 경우 `priority` 90장 추가 업로드
11. `V2` 생성
12. 다시 export + retrain

---

## 13. 새 프로젝트에서 절대 하지 말아야 할 것

- Track A 발표용 이미지/영상 업로드
- `player`, `object`, `effect`를 섞은 상태로 다시 시작
- 애매한 non-playable frame을 training positive로 억지 유지
- export 폴더 이름을 모호하게 저장
- 버전 설명 없이 그냥 `V2`, `V3`만 계속 누적

---

## 14. 첫 세팅 직후 바로 확인할 것

새 프로젝트를 만든 직후 아래 5개만 확인하면 된다.

1. Project Type이 `Object Detection`인가
2. 클래스가 `player` 하나만 보이는가
3. 업로드 첫 배치가 `phase1`인가
4. Track A 관련 파일이 하나도 안 들어갔는가
5. 프로젝트 설명과 이름만 보고도 Track B 전용임이 분명한가

---

## 15. 지금 시점 추천 결론

현재 가장 좋은 시작 방식은 아래다.

- Roboflow 새 프로젝트 생성
- 이름: `hado-track-b-player-only`
- 타입: `Object Detection`
- 클래스: `player`
- 첫 업로드: `roboflow_upload_bundle_batch01_phase1.zip`
- 첫 목표: `V1` 생성 후 로컬 retrain

즉, 이번 리셋의 목적은 “더 많은 데이터를 빨리 넣는 것”이 아니라 “Track B의 기준 정책을 다시 깨끗하게 고정하는 것”이다.
