# Track B Handoff Report — 2026-06-09

이 문서는 HADO AR/effect occlusion 연구 Track B를 다른 작업자나 다음 세션의 Codex/Claude Code가 바로 이어받을 수 있도록 정리한 실무 인계서다.

## 1. 작업 범위

Track B의 목표는 HADO 경기 영상에서 가상 이펙트가 선수 몸과 겹치는 상황에서도 선수 검출을 안정적으로 유지하는 것이다.

이 문서는 아래 범위만 다룬다.

- Roboflow 기반 player-only 데이터셋 정리
- 실제 경기영상 hard frame 추출
- relabeling 우선순위 선정
- 재학습 직전 상태 정리

Track A 발표용 데모, PPT, IoT 시연 안정화 작업과는 분리해서 다뤄야 한다.

## 2. 현재 기준선

현재 Track B의 기준선 모델은 Roboflow V4 export를 player-only로 정규화해 다시 학습한 YOLOv8n 모델이다.

- 체크포인트:
  - `/Users/jinu/Documents/Codex/2026-06-08/codex-codex/outputs/track_b_roboflow_export_review/hado_player_v4_player_only_yolov8n_mps_e30_best.pt`
- V4 player-only test 성능:
  - Precision `0.619`
  - Recall `0.686`
  - mAP50 `0.672`
  - mAP50-95 `0.430`

핵심 해석:

- Roboflow 내부 test set 기준 성능은 이전 버전보다 개선됐다.
- 하지만 실제 경기영상에서는 여전히 domain gap이 남아 있다.
- 다음 개선 포인트는 모델 구조 변경보다도 실제 경기 hard frame relabeling에 더 가깝다.

## 3. 이번에 완료한 작업

### 3.1 실제 경기 10개 intake

Google Drive에서 10개 경기 영상을 받아 아래 경로에 저장했다.

- `data/drive_imports/batch01`
- 파일: `match01.mp4` ~ `match010.mp4`

### 3.2 배치 처리 파이프라인 추가

아래 스크립트를 만들고 검증했다.

- `tools/process_track_b_video_batch.py`
  - 여러 경기 영상을 한 번에 처리
  - detector CSV 생성
  - hard frame mining 실행
  - 배치 manifest 생성

- `tools/prepare_roboflow_upload_bundle.py`
  - hard frame manifest를 다시 읽어서
  - 업로드용 프레임 폴더
  - 정렬된 CSV 큐
  - README
  - ZIP 파일
  - 형태로 재구성

기존 활용 스크립트:

- `tools/run_player_model_on_video.py`
- `tools/mine_video_hard_frames.py`

추가 자동화:

- `tools/retrain_track_b_export.py`
  - Roboflow export를 player-only detect dataset으로 정규화
  - 선택적으로 YOLOv8 재학습 수행
  - `clean/partial/heavy` 태그별 성능 평가 수행
  - 다음 relabeling용 hard player sample 추출

- `tools/mine_hard_player_samples.py`
  - 이제 `--device auto`와 Track B 기본 checkpoint 자동 탐색을 지원

### 3.3 10경기 1차 분석 완료

산출물:

- 통합 manifest:
  - `data/track_b_batch_review/batch01_combined_manifest.csv`

요약 결과:

- 총 10경기 처리 완료
- 총 hard frame `175장` 추출
- 우선 검토 대상 경기:
  - `match02`
  - `match09`
  - `match01`
  - `match06`
  - `match08`

### 3.4 Roboflow 업로드용 번들 생성 완료

두 종류의 업로드 묶음을 만들었다.

1. `phase1` 소형 묶음
- `data/track_b_batch_review/roboflow_upload_bundle_batch01_phase1.zip`
- `data/track_b_batch_review/roboflow_upload_bundle_batch01_phase1_manifest.csv`
- 총 `36장`
- `low_conf` + `high_count` 중심
- 가장 먼저 업로드할 묶음

2. `priority` 확장 묶음
- `data/track_b_batch_review/roboflow_upload_bundle_batch01_priority.zip`
- `data/track_b_batch_review/roboflow_upload_bundle_batch01_priority_manifest.csv`
- 총 `90장`
- `label_priority` + `review_or_skip` 포함
- 1차 라벨링 후 추가 확장용

## 4. 지금 가장 중요한 파일

바로 다음 작업에 필요한 파일만 추리면 아래와 같다.

- Claude/Codex 인계 브리프:
  - `docs/CLAUDE_CODE_CONTINUATION_BRIEF_2026_06_08.md`
- phase1 업로드 ZIP:
  - `data/track_b_batch_review/roboflow_upload_bundle_batch01_phase1.zip`
- phase1 업로드 manifest:
  - `data/track_b_batch_review/roboflow_upload_bundle_batch01_phase1_manifest.csv`
- phase1 업로드 안내:
  - `data/track_b_batch_review/roboflow_upload_bundle_batch01_phase1_README.md`
- priority 업로드 ZIP:
  - `data/track_b_batch_review/roboflow_upload_bundle_batch01_priority.zip`
- priority 업로드 manifest:
  - `data/track_b_batch_review/roboflow_upload_bundle_batch01_priority_manifest.csv`

## 5. 다음 작업자가 해야 할 일

우선순위는 아래 순서가 가장 좋다.

1. `roboflow_upload_bundle_batch01_phase1.zip` 36장을 Roboflow에 업로드한다.
2. 이번 라운드는 반드시 `player-only`로 유지한다.
3. 라벨링 규칙:
   - 실제 플레이 장면은 player 박스를 보수적으로 정확히 수정
   - shield/projectile과 겹쳐도 보이는 선수 몸 기준으로 일관되게 박스 부여
   - intro, roster, scoreboard 장면은 positive 학습 예시로 억지 사용하지 않음
4. Roboflow에서 `V5` 버전으로 export한다.
5. export ZIP 또는 export 폴더를 다시 로컬에 전달한다.
6. 그 다음 로컬에서 아래 명령으로 재학습 루프를 실행한다.

```bash
./hado_venv/bin/python tools/retrain_track_b_export.py \
  --export-src "<V5 export folder>" \
  --run-name "v5_player_only" \
  --epochs 30 \
  --batch 16
```

7. 생성된 `run_summary.json`, `run_summary.md`, `tag_eval/`, `hard_mining/`를 보고 V4 대비 성능 비교와 실제 영상 재평가를 진행한다.

시간이 적으면:

- `phase1` 36장만 먼저 라벨링

시간이 더 있으면:

- `priority` 90장까지 확장

## 6. 현재 남아 있는 리스크

- 실제 영상에서 low-confidence 장면이 많다.
- non-playable 화면이 hard frame에 일부 섞여 있다.
- heavy occlusion 예시는 더 늘려야 한다.
- effect class를 같이 학습하는 것은 아직 이르다.

따라서 다음 라운드 목표는 “멀티클래스 확장”이 아니라 “player-only V5 안정화”다.

## 7. 실행 환경 메모

- 사용 python:
  - `./hado_venv/bin/python`
  - 또는 `~/hado_venv/bin/python3`
- 참고:
  - repo 내부 python 링크와 `pyvenv.cfg`를 `/opt/anaconda3/bin/python3.13` 기준으로 정리했다.
  - Track B export smoke run 산출물은 저장소 바깥 경로에도 둘 수 있다. 예:
    - `/Users/jinu/Documents/Codex/2026-06-08/codex-codex/outputs/track_b_retrain_runs`

## 8. 자동화 검증 메모

2026-06-09 기준 아래 검증을 완료했다.

- `tools/retrain_track_b_export.py --skip-train`
  - V4 export를 실제로 변환
  - 태그별 평가
  - hard sample mining
  - 모두 통과
- `tools/retrain_track_b_export.py --epochs 1`
  - 학습 분기까지 end-to-end 실행 확인
  - 이 결과는 배선 smoke test 용도이며 성능 비교 기준으로 사용하지 않는다.

## 9. 권장 커밋 기준

아직은 데이터 준비 단계이므로, 아래 조건을 만족할 때만 커밋한다.

- Track B 관련 파일만 묶일 수 있을 것
- 새 스크립트 또는 문서가 검증됐을 것
- 대용량 원본 영상, 모델 가중치, export ZIP 자체는 커밋하지 않을 것

권장 커밋 예시:

- `feat: add track-b batch video intake pipeline`
- `docs: add track-b relabeling handoff reports`
- `feat: prepare roboflow upload bundles for occlusion relabeling`
