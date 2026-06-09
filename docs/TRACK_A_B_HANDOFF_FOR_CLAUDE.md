# Track A / Track B Handoff for Claude Code

작성일: 2026-06-10

이 프로젝트는 현재 두 가지 목적의 작업이 같은 로컬 저장소에서 진행되고 있다. 앞으로 커밋, 브랜치, PR, 자동화 작업을 만들 때 반드시 아래 두 트랙을 구분한다.

상세 최신 인계:
- `docs/CLAUDE_CODE_CONTINUATION_BRIEF_2026_06_08.md`
  - 중간발표 3분 영어 대본
  - Track A 발표용 설명 흐름
  - Track B Roboflow V4 player-only 모델 결과
  - 실제 영상 smoke test 결과
  - 10경기 batch intake 결과
  - Claude Code 다음 작업 지시 포함
- `docs/TRACK_B_HANDOFF_REPORT_2026_06_09.md`
  - Track B 진행 현황과 다음 작업자를 위한 handoff 보고서
- `docs/TRACK_B_TECHNICAL_REPORT_2026_06_09.md`
  - Track B 기술적 배경, 파이프라인, 결과 요약
- `docs/TRACK_B_COLLABORATION_REPORT_2026_06_09.md`
  - 협업자 공유용 요약 보고서
- `docs/TRACK_A_PRESENTATION_SCOPE_BRIEF_2026_06_10.md`
  - 발표 때 실제로 다룬 Track A 구현 범위만 정리한 Claude 전달용 브리프

## Track A: IoT 발표용 버전

목적:
- 이번 IoT 발표에서 사용할 안정적인 데모 버전.
- 카메라 기반으로 선수 위치와 움직임을 추적하고, 다음 동작/전술 흐름을 보여준다.

중요 전제:
- 가상 이펙트가 없는 입력을 기준으로 한다.
- 발표 안정성, 데모 영상, 히트맵, 전술 시각화, 테스트 통과가 우선이다.
- AR/가상 이펙트 occlusion 대응 로직을 발표용 커밋에 섞지 않는다.

현재 Track A로 분류되는 주요 작업 파일:
- `src/demo.py`
  - 팀A 3명 중심 발표용 시연 화면.
  - 선수별 이동 방향, 의도 표시, 팀A 반쪽 코트 확대 표시.
  - `--frames`, `--max-frames` 둘 다 지원.
- `src/guide.py`
  - bird-eye 위 전술 화살표 + urgency 링 + 텍스트 패널 정리.
  - 발표 중 설명하기 쉬운 형태로 시각화 단순화.
- `src/demo_pose.py`
  - 카메라/비디오 입력을 모두 지원.
  - NCNN -> ONNX -> PyTorch 자동 모델 선택.
  - `--frames`, `--max-frames`, `--threaded`, `--headless` 지원.
  - bird-eye + skeleton + posture HUD 통합 데모.
- `src/action_demo.py`
  - NCNN -> ONNX -> PyTorch 자동 모델 선택.
  - `--frames`, `--max-frames`, `--threaded`, `--headless`, `--record` 지원.
  - 7-action skeleton 기반 발표용 동작 인식 데모.

권장 브랜치:
- `presentation/demo-finalization`

권장 커밋 예시:
- `feat: polish presentation demo overlays`
- `fix: improve presentation demo readability`

Claude Code가 우선 진행할 작업:
1. `src/demo.py`, `src/guide.py`, `src/demo_pose.py`, `src/action_demo.py`를 발표 목적에 맞게 검토한다.
2. 아래 짧은 스모크 테스트부터 다시 돌려 정상 종료를 확인한다.
3. 그 다음 Pi4 / 현장 카메라 기준으로 HUD 가독성과 FPS를 다듬는다.
4. 발표용 커밋에는 Track B 파일을 포함하지 않는다.

검증된 스모크 테스트:

```bash
./hado_venv/bin/python -m src.demo --headless --max-frames 30
./hado_venv/bin/python -m src.demo_pose --video data/1.mp4 --headless --frames 5
./hado_venv/bin/python -m src.demo_pose --video data/1.mp4 --headless --max-frames 5
./hado_venv/bin/python -m src.action_demo --source data/1.mp4 --headless --frames 5
```

확인된 결과:

- `demo_pose`는 정확히 5프레임 처리 후 정상 종료됨
- `action_demo`는 headless 5프레임 정상 종료됨
- `demo.py`는 `--max-frames` 별칭으로 headless 30프레임 정상 종료됨
- NCNN runtime이 로컬에서 로드되어 발표용 기본 경로가 동작함

## Track B: 개인 장기 프로젝트 버전

목적:
- 실제 HADO 경기 영상처럼 AR/가상 이펙트가 선수 위에 겹치는 상황에서도 선수 detection/tracking을 견고하게 만든다.

중요 전제:
- 이번 IoT 발표용 요구사항과 분리한다.
- 가상 이펙트 occlusion, short detection miss, ID 유지, 재식별, synthetic augmentation, occlusion 평가 지표가 핵심이다.
- Pi 4 호환성과 오프라인 실행 원칙을 계속 고려한다.

현재 Track B로 분류되는 미커밋 파일:
- `src/tracker.py`
  - `reid_distance_px` 추가.
  - 짧은 occlusion 뒤 bbox IoU가 낮거나 0이어도, lost track이 가까운 발 위치에 재등장하면 같은 ID로 복구.
- `tests/test_tracker.py`
  - 가까운 재등장 시 같은 ID를 유지하는 테스트 추가.
  - 멀리 재등장한 감지는 새 ID로 처리하는 테스트 추가.

검증 완료:
- `~/hado_venv/bin/python3 -m pytest tests/test_tracker.py -q`
- 결과: `28 passed`
- 참고: `.pytest_cache` 쓰기 권한 경고가 있었지만 테스트 실패는 아님.

권장 브랜치:
- `research/occlusion-robust-tracking`

권장 커밋 예시:
- `fix: recover track ids after short occlusion`
- `test: cover tracker reidentification after occlusion`

Codex가 앞으로 우선 진행할 작업:
1. Track B를 중심으로 발전시킨다.
2. synthetic AR/effect occlusion augmentation 유틸을 추가한다.
3. occlusion 전용 평가 체크리스트와 테스트셋 구조를 만든다.
4. 실제 경기 영상에서 detection miss, recovery rate, ID switch를 측정할 수 있는 평가 스크립트를 준비한다.

최근 Track B 업데이트:
- V4 Roboflow export를 player-only로 정규화해 YOLOv8n 30 epoch 재학습 완료.
- 새 checkpoint test 성능: precision 0.619, recall 0.686, mAP50 0.672, mAP50-95 0.430.
- 이전 checkpoint의 V4 test mAP50-95 0.316 대비 개선.
- 새 실제 영상 리뷰 도구: `tools/run_player_model_on_video.py`.
- 새 batch intake 도구: `tools/process_track_b_video_batch.py`.
- Drive 기반 10경기(`data/drive_imports/batch01`) 1차 샘플링/하드프레임 추출 완료.
- 실제 영상 smoke test 결과는 `docs/CLAUDE_CODE_CONTINUATION_BRIEF_2026_06_08.md` 참고.

## 절대 섞지 말 것

- Track A 발표용 커밋에 `src/tracker.py`, `tests/test_tracker.py`의 occlusion 연구 변경을 섞지 않는다.
- Track B 연구 커밋에 발표용 `src/demo.py`, `src/guide.py` 시각화 변경을 섞지 않는다.
- `config/calibration.json`, `data/movement_data.csv`, position log CSV, snapshots, model weights는 보호한다.
- force-push 금지.

## 현재 권장 진행

1. Claude Code는 Track A 발표용 변경을 별도 브랜치/커밋으로 정리한다.
2. Codex는 Track B 개인 프로젝트 변경을 별도 브랜치/커밋으로 정리하고 계속 발전시킨다.
3. 두 트랙 사이에서 공유할 수 있는 일반 개선이 생기면, 먼저 어느 트랙에 필요한지 명확히 분류한 뒤 별도 커밋으로 다룬다.
