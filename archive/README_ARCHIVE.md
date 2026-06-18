# HADO Smart Court IoT — Archive

이 폴더는 최종 발표(2026-06-21)에서 제외된 코드들을 보관한다.
개인 프로젝트로 이어서 진행할 예정이므로 삭제하지 않고 격리 보관함.

## 보관된 이유

최종 발표 방향 변경: 코트 전술 분석 → HADO 8가지 기본동작 keypoint 인식으로 축소

## 보관 내용

### `src/` — 전술/트래킹/호모그래피 관련 모듈

| 파일 | 기능 |
|------|------|
| `tactic_engine.py` | 전술 엔진 (R1~R6 규칙) |
| `tracker.py` | IoU 기반 선수 트래킹 |
| `homography.py` | 카메라 → 코트 좌표계 변환 |
| `visualizer.py` | 새눈뷰 시각화 |
| `guide.py` | 화면 전술 가이드 오버레이 |
| `main.py` | 메인 파이프라인 (Level 1/2) |
| `calibrate.py` | 코트 캘리브레이션 |
| `aruco_calibrate.py` | ArUco 마커 캘리브레이션 |
| `vest.py` | 조끼 색상 감지 |
| `demo.py` | 전술 데모 |
| `analyzer.py` | 패턴 분석기 |
| `measure_error.py` | 위치 오차 측정 |
| `movement_model.py` | 이동 예측 모델 |
| `benchmark.py` | FPS 벤치마크 |
| `demo_pose.py` | 포즈 단독 데모 |
| `recorder.py` | 영상 녹화 |
| `upload.py` | 결과 업로드 유틸 |
| `pi_preflight.py` | Pi4 사전 점검 |
| 기타 | annotate_tool, label_intents, auto_label, extract_*, visualize_* |

### `tests_backup/` — 전체 테스트 백업 (272개)

### `docs_backup/` — 기존 문서 백업

## 재개 시작점

이 코드들을 다시 활용하려면:
1. `src/` 에서 원하는 파일 복원
2. `requirements.txt` 의존성 재확인
3. `config/court_config.yaml` 설정 확인
4. `./run.sh test` 로 기존 테스트 재실행
