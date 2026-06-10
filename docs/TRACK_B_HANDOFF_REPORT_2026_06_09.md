# Track B Handoff Report — 2026-06-10

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
  - `outputs/track_b_retrain_runs/track_b_current_best_checkpoint.txt`
- 현재 leaderboard 기준 추천 run:
  - `trackb_v4_player_only`
- V4 player-only peak 성능:
  - Precision `0.676`
  - Recall `0.648`
  - mAP50 `0.653`
  - mAP50-95 `0.353`

핵심 해석:

- Roboflow 내부 test set 기준 성능은 이전 버전보다 개선됐다.
- 하지만 실제 경기영상에서는 여전히 domain gap이 남아 있다.
- 다음 개선 포인트는 모델 구조 변경보다도 실제 경기 hard frame relabeling에 더 가깝다.

## 3. 이번에 완료한 작업

### 3.1 실제 경기 batch01 + batch02 intake

Google Drive에서 실제 경기 영상을 두 batch로 받아 아래 경로에 저장했다.

- `data/drive_imports/batch01`
- `data/drive_imports/batch02_raw`
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

- `tools/compare_track_b_runs.py`
  - `outputs/track_b_retrain_runs/*/run_summary.json`과 `results.csv`를 다시 읽어
  - 버전별 precision / recall / mAP50 / mAP50-95를 leaderboard로 정리
  - 현재 추천 checkpoint를 자동으로 뽑아
    - `outputs/track_b_retrain_runs/track_b_current_best_checkpoint.txt`
    - `outputs/track_b_retrain_runs/track_b_current_best_checkpoint.json`
    - 형태로 저장

- `tools/prepare_track_b_batch_review.py`
  - batch video intake
  - summary 생성
  - Roboflow 업로드 bundle 생성
  - 을 한 번에 묶어 실행

- `tools/retrain_track_b_export.py`
  - 새 export 재학습이 끝나면
  - `tools/compare_track_b_runs.py`를 자동 호출해
  - 최신 leaderboard와 추천 checkpoint pointer를 갱신

- `tools/export_track_b_pose_review.py`
  - 실제 경기영상에서 skeleton / action / court position review artifact를 생성
  - CSV / JSONL / optional overlay mp4를 내보냄
  - Track B의 다음 단계인 skeleton-based verification 준비용

- `tools/build_track_b_openai_batch.py`
  - Track B upload manifest를 OpenAI Batch `/v1/responses` 입력 jsonl로 변환
  - hard frame image QA 자동화 준비용

- `tools/summarize_track_b_openai_batch.py`
  - OpenAI Batch 출력 jsonl을 merged CSV / priority CSV / summary로 정리
  - relabel triage 보조용

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

### 3.5 batch02 실제 영상 평가 완료

batch02는 새 경기 10개를 현재 V4 best checkpoint로 다시 훑은 결과다.

핵심 산출물:

- `data/track_b_batch_review/batch02_v4_eval/batch_manifest.csv`
- `data/track_b_batch_review/batch02_v4_eval/batch02_summary.md`
- `data/track_b_batch_review/batch02_v4_eval/batch02_priority.csv`
- `data/track_b_batch_review/batch02_v4_eval/roboflow_upload_bundle_v2_playable_bias.zip`

요약:

- 총 10경기 처리
- sampled frames `994`
- hard frames `130`
- 가장 우선순위가 높은 경기:
  - `match08`
  - `match02`
  - `match03`
  - `match04`
  - `match09`

해석:

- batch02에서는 miss보다 over-detection이 더 주요 문제였다.
- 즉, 다음 relabeling은 “못 잡은 선수”보다 “이펙트/실드 겹침으로 player box가 과하게 많이 뜨는 장면” 정리에 가깝다.

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
- 현재 추천 checkpoint pointer:
  - `outputs/track_b_retrain_runs/track_b_current_best_checkpoint.txt`
- 현재 leaderboard:
  - `outputs/track_b_retrain_runs/track_b_run_leaderboard.md`
- 작업자 운영 문서:
  - `docs/TRACK_B_OPERATOR_PLAYBOOK_2026_06_10.md`
- 협업용 relabel 가이드:
  - `docs/TRACK_B_BATCH_REVIEW_RELABEL_GUIDE_2026_06_10.md`
- skeleton phase 브리프:
  - `docs/TRACK_B_SKELETON_PHASE_BRIEF_2026_06_10.md`
- batch02 평가 문서:
  - `docs/TRACK_B_BATCH02_EVAL_2026_06_09.md`

## 5. 다음 작업자가 해야 할 일

우선순위는 아래 순서가 가장 좋다.

1. `data/track_b_batch_review/batch02_v4_eval/roboflow_upload_bundle_v2_playable_bias.zip` 또는 `frames/`를 Roboflow에 업로드한다.
2. 이번 라운드는 반드시 `player-only`로 유지한다.
3. 라벨링 규칙:
   - 실제 플레이 장면은 player 박스를 보수적으로 정확히 수정
   - shield/projectile과 겹쳐도 보이는 선수 몸 기준으로 일관되게 박스 부여
   - intro, roster, scoreboard 장면은 positive 학습 예시로 억지 사용하지 않음
4. Roboflow에서 다음 버전 export를 생성한다.
5. export ZIP 또는 export 폴더를 다시 로컬에 전달한다.
6. 그 다음 로컬에서 아래 명령으로 재학습 루프를 실행한다.

```bash
./hado_venv/bin/python tools/retrain_track_b_export.py \
  --export-src "<V5 export folder>" \
  --run-name "trackb_v5_player_only" \
  --epochs 30 \
  --batch 16
```

7. 생성된 `run_summary.json`, `run_summary.md`, `tag_eval/`, `hard_mining/`, `track_b_run_leaderboard.md`를 보고 V4 대비 성능 비교와 실제 영상 재평가를 진행한다.
8. player-only detector가 충분히 안정되면 `docs/TRACK_B_SKELETON_PHASE_BRIEF_2026_06_10.md`에 따라 skeleton verification으로 넘어간다.

시간이 적으면:

- `phase1` 36장만 먼저 라벨링

시간이 더 있으면:

- `priority` 90장까지 확장

## 6. 현재 남아 있는 리스크

- 실제 영상에서 low-confidence 장면이 많다.
- non-playable 화면이 hard frame에 일부 섞여 있다.
- heavy occlusion 예시는 더 늘려야 한다.
- effect class를 같이 학습하는 것은 아직 이르다.

따라서 다음 라운드 목표는 “멀티클래스 확장”이 아니라 “player-only V5/V6 안정화”다.

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

- `tools/compare_track_b_runs.py`
  - V1 ~ V4 run summary + results.csv 비교 검증 완료
  - 현재 추천 run은 `trackb_v4_player_only`
  - `resolve_model_path(None)`가 이 pointer를 우선적으로 읽는 것까지 확인

- `tools/prepare_track_b_batch_review.py --skip-process`
  - 기존 batch02 결과를 다시 읽어
  - summary / priority CSV / upload bundle / zip 재생성 검증 완료

- `tools/export_track_b_pose_review.py`
  - effect-heavy 실제 경기영상 smoke 실행 완료
  - 일반 샘플 영상 `data/1.mp4`에서는 CSV / JSONL row 생성 검증 완료
  - skeleton phase용 review artifact 형식이 실제로 작동하는 것 확인

## 9. 권장 커밋 기준

아직은 데이터 준비 단계이므로, 아래 조건을 만족할 때만 커밋한다.

- Track B 관련 파일만 묶일 수 있을 것
- 새 스크립트 또는 문서가 검증됐을 것
- 대용량 원본 영상, 모델 가중치, export ZIP 자체는 커밋하지 않을 것

권장 커밋 예시:

- `feat: add track-b batch video intake pipeline`
- `docs: add track-b relabeling handoff reports`
- `feat: prepare roboflow upload bundles for occlusion relabeling`
