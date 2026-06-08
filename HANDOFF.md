# HADO Smart Court IoT — Codex Handoff

> 최종 업데이트: 2026-06-08 (중간발표 당일)  
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

| 브랜치 | 목적 |
|--------|------|
| `presentation/demo-finalization` | **발표용 (Track A)** — 팀A 3명 반코트, 오버레이 화살표·의도 레이블 |
| `research/occlusion-robust-tracking` | Track B — 폐색 재추적 연구 |
| `main` | 안정 베이스라인 |

**발표 전 체크**: `git checkout presentation/demo-finalization` 후 데모 실행.

---

## 3. 최근 수정 요약 (2026-06-08)

### 중요 버그 수정 (**팀A 선수 경계선 집중 현상 해결**)

**증상**: 모든 선수가 x≈4.9m(팀 경계선)에 몰려 전방에만 위치.

**원인**: `_load_pattern_library()`가 team A(x=0-5m)와 team B(x=5-10m) 패턴을 혼합 로드.
팀A 선수가 팀B 패턴(`to_x=9.7m`)을 선택하면 `_load_step()`에서
`max(0.1, min(4.9, 9.7)) = 4.9`로 클램핑 → 항상 최전방.

**수정**: `_load_pattern_library(team="A")` — team='A' 필터 추가.
`presentation/demo-finalization`과 `research/occlusion-robust-tracking` 양 브랜치에 적용.

### 기타 수정
| 파일 | 변경 |
|------|------|
| `src/demo.py` | team 필터 + Technician 히트맵 Zone%→Travel 거리 지표 변경 |
| `src/extract_movement.py` | 신규: 경기 영상에서 선수 궤적 추출 도구 |

---

## 4. 현재 버그 / 이슈

| 심각도 | 설명 | 위치 |
|--------|------|------|
| 중 | Defender zone 30% — 패턴 데이터가 수비 시에도 전방 이동 포함 | `movement_data.csv` 패턴 편향 |
| 하 | PatternPlayer context 전환 시 자연스러운 마무리 없이 끊김 | `demo.py:PatternPlayer.set_context` |
| 하 | 영상 추출(`extract_movement.py`) — AR 이펙트로 YOLOv8 감지율 낮음 | `src/extract_movement.py` |

---

## 5. Codex가 수정해도 되는 범위

- `src/demo.py` — 시뮬레이션·시각화·히트맵
- `src/analyzer.py` — 통계·히트맵 개선
- `src/visualizer.py` — 렌더링·색상·오버레이
- `src/tactic_engine.py` — 전술 규칙 (`config/court_config.yaml` 참조)
- `src/guide.py` — 가이드 화살표 시각화
- `src/extract_movement.py` — 영상 기반 패턴 추출
- `tests/` — 테스트 추가/유지

---

## 6. 절대 수정 금지

| 파일/경로 | 이유 |
|-----------|------|
| `config/calibration.json` | 카메라 고유 캘리브레이션 |
| `data/movement_data.csv` | 전문가 어노테이션 원본 |
| `config/court_config.yaml` | 변경 시 테스트 필수 |
| 네트워크/클라우드 호출 | 오프라인 전용 시스템 |

---

## 7. 테스트 / 실행 명령어

```bash
# venv 활성화
source ~/hado_venv/bin/activate

# 전체 테스트 (158개 기준)
./run.sh test

# 발표용 데모 (90초)
~/hado_venv/bin/python3 -m src.demo --headless --frames 2700
# → data/demo.mp4 (28MB, 1884×720, 팀A 반코트)
# → data/heatmap_role.png (역할별 히트맵)

# 5분 백업 영상
~/hado_venv/bin/python3 -m src.demo --headless --frames 9000

# 경기 영상에서 패턴 추출 (실험적)
~/hado_venv/bin/python3 -m src.extract_movement --video "data/5경기 DINOS vs SSP 블루코트.mp4" --no-save
```

---

## 8. 커밋 / PR 선호 방식

- Conventional Commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`
- 커밋 전 `./run.sh test` 통과 필수
- **커밋 금지**: `data/position_logs/*.csv`, `data/snapshots/`, `models/*.pt` (50MB+), `config/calibration.json`

---

## 9. 완료된 작업 (2026-06-08 야간)

### ✅ P1 — Defender Zone% 개선
- PatternPlayer `role_name` + `_ZONE_CLAMP` 추가
  - defender defend=(0.1,1.8) attack=(0.3,3.8) transition=(0.2,2.5)
  - main_attacker attack=(2.5,4.9) defend=(0.5,3.5)
- 히트맵 지표를 **페이즈별 분리 통계**로 개선
  - Defender: "Def Zone(def phase): 94%" (수비 페이즈 중 x≤2.0m 비율)
  - Attacker: "Atk Zone(atk phase): 51%" (공격 페이즈 중 x≥3.0m 비율)

### ✅ P2 — 5분 백업 영상
- `data/demo_5min.mp4` 생성 완료 (95MB)

### ✅ P3 — recorder.py role 컬럼
- `positions.csv` 헤더에 `role` 컬럼 추가
- `MatchRecorder.write(track_roles={...})` 파라미터 추가

### ✅ P4 — Zone HUD 추가
- `_draw_zone_hud()`: 버드아이뷰 우상단에 1·2·3선 실시간 선수 카운트

### ✅ P5 — PatternPlayer context 전환 개선
- `set_context()` → `_pending_context` 저장, 스텝 경계에서 적용

---

## 10. 완료된 작업 (2026-06-08 야간 — W4 TTS)

### ✅ W4 — pyttsx3 한국어 TTS 통합
- pyttsx3 설치 + requirements.txt 활성화
- `VoiceGuide._select_korean_voice()`: Yuna(Mac) > ko-KR compact > ko 포함 우선순위
- 발화 속도 180 → 200 wpm 개선
- Pi4 설치 안내: `sudo apt install espeak-ng`
- 테스트 추가: `test_voice_guide_korean_voice_selected` (pyttsx3 없으면 자동 skip)
- 전체 테스트 159개 통과

### ✅ W4 — demo --csv / player_id 분리
- `demo.py --csv <path>`: 어노테이션 CSV 지정 가능
- `_load_pattern_library(player_id=N)`: 특정 선수 패턴만 로드
- 윈터컵 결승 2경기 어노테이션 3패턴 `movement_data.csv` 병합 완료

---

## 11. 남은 작업 (W5–W6)

- W5 (6/11-17): Pi 4 실측 (fps, RAM, 온도, 위치 오차) → `docs/QA_PREP.md` [TODO] 채우기
- W6 (6/18-22): Final report §5 실측 데이터 기입, Q&A 영어 리허설 5회

Claude Code 최신 상세 인계:
- `docs/CLAUDE_CODE_CONTINUATION_BRIEF_2026_06_08.md`
  - 중간발표 3분 영어 대본
  - Track A 발표용 설명 흐름
  - Track B V4 player-only 모델 결과 및 실제 영상 smoke test
  - 다음 Claude Code 작업 지시

---

## 프로젝트 현재 상태 (W3, 6/8 기준)

```
- [x] W1: Setup + calibration
- [x] W2: YOLOv8 + tracking
- [x] W3: Bird-eye view + 중간발표 ← 오늘
- [x] W4 (6/4-10): Tactic engine + voice ← 완료
- [ ] W5 (6/11-17): On-court testing + measurements
- [ ] W6 (6/18-22): Final report + demo video + presentation
```

### 제출 체크리스트

- [x] Code in `src/` + tests (159개 통과)
- [x] README.md
- [x] Final report `.docx` — §5 실측 데이터 미기입
- [x] Final PPT `.pptx` — 실측 데이터 미기입
- [x] Demo video `data/demo.mp4` (90초, 28MB)
- [x] Backup demo `data/demo_5min.mp4` (5분)
- [ ] Q&A prep in English

**최종 제출 마감: 2026-06-22**
