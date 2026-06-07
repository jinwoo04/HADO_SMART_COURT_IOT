# HADO Smart Court IoT — Codex Handoff

> 작성일: 2026-06-08 (중간발표 당일)  
> 목적: Claude Code → Codex 인계서. 자동화 작업 전 반드시 숙지할 것.

---

## 1. 로컬 경로 & GitHub Repo

| 항목 | 값 |
|------|-----|
| 로컬 경로 | `/Users/jinu/iot project/hado-smart-court-iot` |
| GitHub | `https://github.com/jinwoo04/HADO_SMART_COURT_IOT.git` |
| Python venv | `~/hado_venv` (python3 → `/opt/anaconda3/bin/python3.13`) |
| 실행 prefix | `~/hado_venv/bin/python3 -m src.<module>` |

---

## 2. 브랜치 전략

- **`main`** 단일 브랜치 운영 (현재까지 feature branch 없음)
- PR 없이 main에 직접 커밋해왔음
- 향후 Codex 작업 시 `feat/<topic>` 브랜치 생성 → PR → main 병합 권장
- 절대 force-push 금지

---

## 3. 최근 Claude Code 작업 요약 (2026-06-08)

### 오늘 수정한 내용
| 파일 | 변경 내용 |
|------|-----------|
| `src/demo.py` | **PatternPlayer 절대좌표 전환**: 기존 델타 누적 → CSV의 `to_x/to_y` 절대 좌표 사용. 선수가 코트 전역을 표류하는 문제 수정 |
| `src/demo.py` | **context 사이클 추가**: attack(9s) → transition(2s) → defend(8s) → transition(2s) 페이즈 자동 전환. PatternPlayer가 context에 맞는 패턴만 선택 |
| `src/demo.py` | **`_generate_demo_heatmap`**: 팀A 선수 포지션 로그(워밍업 300f 제외) → 역할별 히트맵(Technician/Attacker/Defender) 자동 생성. 구역선·Zone% 표시 포함 |
| `src/demo.py` | **KeyError 방어**: IoU tracker가 새 ID를 발급할 때 `pos` 딕셔너리에 없는 track_id 접근 시 crash 수정 |
| `src/analyzer.py` | **`_generate_heatmap` 분리**: 전체 합산 `heatmap.png` 유지 + 선수별 패널 `heatmap_by_player.png` 신규 생성 |
| venv | `~/hado_venv/bin/python3` symlink가 삭제된 miniconda를 가리키던 것을 `/opt/anaconda3/bin/python3.13`으로 수정 |

### 미커밋 상태
`src/demo.py`, `src/analyzer.py` 수정 완료, 아직 커밋 안 됨.

---

## 4. 남아있는 버그 / 이슈 / TODO

### 버그
| 심각도 | 설명 | 위치 |
|--------|------|------|
| 중 | Technician 히트맵이 너무 희미 — 이동 경로가 넓어서 밀도 낮음 | `demo.py:_generate_demo_heatmap` |
| 중 | Defender zone 점유율 18% — attack 페이즈에서 전진 패턴이 많아 후방 유지 약함 | `movement_data.csv` 패턴 분포 문제 |
| 하 | `PatternPlayer`가 context 전환 시 현재 스텝을 즉시 버림 — 자연스러운 마무리 없이 끊김 | `demo.py:PatternPlayer.set_context` |

### 미완성 기능 / TODO
- `data/matches/test_20260608_120000/positions.csv` — 구버전 델타 방식으로 기록된 비정상 데이터 (이동속도 4.5m/s). 새 demo 실행으로 덮어써야 함
- `src/recorder.py` — positions.csv에 `role` 컬럼 없음. 분석 시 track_id → role 매핑이 수동
- 데모 영상 길이: 현재 2400프레임(80초)이나 제출 목표는 90–120초(2700–3600프레임)
- 5분 무편집 백업 영상 미생성
- 최종 보고서/PPT 실측 데이터 셀 미기입 (§5 실험 결과 표)

---

## 5. Codex가 자동으로 수정해도 되는 범위

- `src/demo.py` — 시뮬레이션 로직, 시각화, 히트맵 생성
- `src/analyzer.py` — 통계 계산, 히트맵 개선
- `src/visualizer.py` — 렌더링, 색상, 오버레이
- `src/tactic_engine.py` — 전술 규칙 조정 (튜닝 수치는 `config/court_config.yaml` 참조)
- `src/guide.py` — 가이드 화살표 시각화
- `tests/` — 기존 테스트 유지 + 새 테스트 추가
- `README.md` — 문서 업데이트

---

## 6. 수정 전 반드시 확인해야 하는 위험 영역

| 파일/경로 | 이유 |
|-----------|------|
| `config/calibration.json` | 카메라 고유 캘리브레이션. 절대 수정 금지 |
| `data/movement_data.csv` | 전문가 어노테이션 원본 데이터. 절대 수정 금지 |
| `config/court_config.yaml` | 모든 튜닝 파라미터. 변경 시 테스트 필수 |
| `src/movement_model.py` | k-NN 모델 로직. 변경 시 `test_movement_model.py` 확인 |
| `src/homography.py` | 좌표계 핵심. 변경 시 `test_homography.py` 확인 |
| 네트워크/클라우드 호출 | 절대 추가 금지 — 오프라인 전용 시스템 |
| `requirements.txt` | 의존성 추가 시 Pi 4 호환성 확인 필요 |

---

## 7. 테스트 / 린트 / 빌드 명령어

```bash
# venv 활성화 (항상 먼저)
source ~/hado_venv/bin/activate
# 또는 prefix 방식:
~/hado_venv/bin/python3 -m pytest tests/ -q

# 전체 테스트 (158개 통과 기준)
./run.sh test

# 모듈별 단독 실행 (디버깅)
~/hado_venv/bin/python3 -m src.demo --headless --frames 900
~/hado_venv/bin/python3 -m src.movement_model
~/hado_venv/bin/python3 -m src.tactic_engine

# 데모 영상 생성 (90초 = 2700프레임)
~/hado_venv/bin/python3 -m src.demo --headless --frames 2700

# 벤치마크
./run.sh bench --frames 200 --imgsz 320
```

---

## 8. 커밋 / PR 선호 방식

- **Conventional Commits**: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`
- 한 커밋 = 한 관심사. 무관한 변경 혼합 금지
- 커밋 전 `./run.sh test` 통과 필수
- **절대 커밋 금지 파일**:
  - `data/position_logs/*.csv`
  - `data/snapshots/`
  - `models/*.pt` (50MB 초과)
  - `config/calibration.json`
- PR 제목: 70자 이내, 영어 or 한국어 무방
- main 브랜치 직접 push보다 PR 권장 (자동화 시)

---

## 9. Codex 처리 우선순위 작업 Top 5

### P1 — 데모 영상 최종본 생성 (90–120초)
```bash
~/hado_venv/bin/python3 -m src.demo --headless --frames 3000
```
- `data/demo.mp4` (100초 분량) 생성
- 상태바 의도 텍스트가 잘 보이는지 확인
- 프리뷰 프레임이 attack/transition/defend 세 페이즈를 모두 포함하는지 확인

### P2 — Technician 히트맵 밀도 개선
- `src/demo.py:_generate_demo_heatmap` 수정
- 테크니션은 코트를 횡단하므로 이동 경로(line segment)를 점이 아닌 선분으로 채워야 함
- 인접 프레임 두 점 사이를 `cv2.line`으로 그리는 방식으로 변경
- 기존 점 방식 대비 밀도 2–4배 향상 예상

### P3 — `src/recorder.py`에 role 컬럼 추가
- positions.csv 헤더에 `role` 컬럼 추가
- `tactic_engine.analyze()` 결과 또는 track_id → role 매핑으로 채움
- 이렇게 하면 실제 경기 기록에서도 `heatmap_by_player.png`에 역할명이 정확히 표시됨

### P4 — 전술 엔진 통계 HUD 추가
- 데모 영상 오른쪽 하단에 실시간 통계 오버레이 추가
- 표시 항목: 현재 게임 페이즈(ATTACK/DEFEND/TRANSITION), 각 팀 구역 점유율(%)
- `src/visualizer.py`에 `draw_phase_hud(img, phase, zone_ratio)` 함수 추가

### P5 — `tests/` 커버리지 보강
- `PatternPlayer` context 전환 테스트 (`test_demo.py` 신규)
  - `set_context("defend")` 호출 후 defender 패턴만 선택되는지 검증
  - 절대좌표: to_x가 team A면 0.1≤x≤4.9 범위인지 검증
- `analyzer.py:_generate_heatmap` — per-player 패널 수가 track 수와 같은지 검증

---

## 프로젝트 현재 상태 (W3, 6/8 기준)

```
- [x] W1: Setup + calibration
- [x] W2: YOLOv8 + tracking
- [~] W3: Bird-eye view + 1st draft presentation  ← 오늘 (중간발표)
- [ ] W4 (6/4-10): Tactic engine integration + voice
- [ ] W5 (6/11-17): On-court testing + measurements
- [ ] W6 (6/18-22): Final report + demo video + presentation
```

**최종 제출 마감: 2026-06-22**
