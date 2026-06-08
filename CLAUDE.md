# HADO Smart Court IoT — Project Rules for Claude Code

This file is read automatically at the start of every Claude Code session.
Treat it as standing orders. Keep it concise — long files get partially ignored.

---

## What this project is

A real-time player tracking and tactical guidance system for HADO, an AR-based
physical e-sport on a 10.0 × 6.0 m court (높이 2.66 m). Runs entirely on a single
Raspberry Pi 4 with a fixed camera. No cloud, no internet.

**Owner**: Jinu Park (박진우), HUFS Computer Engineering, piaojinu@hufs.ac.kr
**Spec**: IoT Term Project, Spring 2026. Final submission June 22, 2026.

Two-level pipeline:
- **Level 1**: camera → YOLOv8n detect → IoU tracker → homography → bird-eye view
- **Level 2**: + rule-based tactic engine → visual arrows + Korean TTS

---

## Tech stack

- Python 3.11 (single language across modules)
- OpenCV 4.x for vision
- Ultralytics YOLOv8n for person detection (imgsz=320 on Pi 4 CPU)
- NumPy for math
- PyYAML for config (`config/court_config.yaml`)
- picamera2 / cv2.VideoCapture for camera (auto-fallback)
- pyttsx3 for offline Korean TTS

---

## How to run things

```bash
# Activate venv (always)
source ~/hado_venv/bin/activate

# Test
./run.sh test                      # all unit tests + module smoke tests
python -m tests.test_homography    # just homography tests

# Calibrate (once per camera position)
./run.sh calibrate

# Run main pipeline
./run.sh main                      # Level 1
./run.sh main --level 2            # Level 2
./run.sh main --level 2 --voice    # + voice
./run.sh main --headless --record demo.mp4   # SSH or recording

# Benchmark fps
./run.sh bench --frames 200 --imgsz 320
```

---

## Code conventions

- **One module = one concern**. `camera.py` only knows camera. `tracker.py` only knows tracking.
- **Each module must support `python -m src.<name>` standalone** with a `main()` function and dummy data. Used for debugging.
- **dataclasses for inter-module interfaces**: `Detection`, `Track`, `PlayerState`, `TacticAdvice`. Never pass dicts.
- **No magic numbers in code** — all tunables in `config/court_config.yaml`.
- **Comments in Korean are fine** for domain logic (HADO rules); docstrings in English.
- **Type hints required** on new functions.

---

## Coordinate convention

- **Pixel space**: (x_px, y_px) — origin top-left, y grows downward
- **Court space**: (x_m, y_m) in meters — origin top-left of court, x grows right (along the 10.0 m axis, team boundary at x=5.0), y grows down (along the 6.0 m axis)
- **Zone lines (x)**: 1.5 m, 3.0 m | 7.0 m, 8.5 m — 팀A: 3선/2선/1선, 팀B 미러
- **Lane lines (y)**: 2.0 m, 4.0 m — 코트를 3개 레인(선수별 담당 구역)으로 분할
- **Foot point**: bottom-center of bounding box = `((x1+x2)/2, y2)`. Use this for homography projection — NOT the center of the box.

---

## Testing rules

- New functions need a unit test under `tests/`.
- Before committing: `./run.sh test` must pass.
- For new tactical rules in `tactic_engine.py`: add a dummy player scenario in the module's `main()` that demonstrates the rule firing.

---

## Git workflow

- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`.
- One concern per commit. If `git diff --cached` mentions multiple unrelated changes, split.
- Never commit:
  - `data/position_logs/*.csv` (logged at runtime)
  - `data/snapshots/`
  - `models/*.pt` over 50 MB
  - `config/calibration.json` (camera-specific)

---

## What Claude should do automatically

- Read related modules before suggesting changes (don't suggest in isolation).
- Run `./run.sh test` after non-trivial edits.
- Use `/commit` to draft commit messages, but show the message before committing.
- Update `CLAUDE.md` itself when a new convention emerges that Claude keeps missing.

## What Claude should NOT do automatically

- **Don't delete files** without explicit confirmation.
- **Don't push to remote** without confirmation.
- **Don't modify `config/calibration.json`** — it's camera-specific.
- **Don't add new dependencies** to `requirements.txt` without discussing first.
- **Don't add cloud / network calls** — this project is explicitly offline.

---

## Performance budget (Pi 4 target)

| Metric | Target | Hard floor |
|--------|--------|------------|
| End-to-end fps | 15+ | 10 |
| Peak RAM | < 1.5 GB | 2.5 GB |
| CPU temp | < 70 °C | 80 °C (throttling) |
| Position error | < 10 cm | 25 cm |

If a change pushes any metric past the hard floor, flag it before merging.

---

## Project status (update each week)

- **W1** (5/14–20): Setup + calibration → DONE
- **W2** (5/21–27): YOLOv8 + tracking → DONE
- **W3** (5/28–6/3): Bird-eye view + 중간발표 → DONE (발표 6/8)
- **W4** (6/4–10): Tactic engine (R1–R6) + voice + action_demo → DONE
  - R6 Lane Cover 추가, charge 동작 추가, threaded 캡처 버그 수정
  - 테스트 180개 통과, demo.mp4 90초 생성, 5분 백업 완료
- **W5** (6/11–17): On-court testing + Pi4 실측 → UPCOMING
  - 측정 항목: FPS(NCNN/ONNX), RAM, CPU 온도, 위치오차 RMS
- **W6** (6/18–22): Final report §5 실측 기입 + Q&A 영어 리허설 ×5

---

## Useful context for Claude

- The author has 7 years of HADO playing experience (Korean national team).
  Tactical decisions should respect HADO-specific spacing (court is small, ~3m between teams).
- HADO court dimensions are fixed: 6.0 × 2.66 m. Don't generalize.
- Voice prompts must be in Korean (selected via pyttsx3 voice ID containing "ko").
- The system targets gym deployment — offline operation is non-negotiable.

---

## Submission deliverables (track these)

- [x] Code in `src/` + tests passing
- [x] README.md
- [x] Final report `.docx` (10+ pages) — [TODO] cells in §5 need real data
- [x] Final PPT `.pptx` (16 slides) — [TODO] cells need real data
- [ ] Demo video `.mp4` (90–120 s)
- [ ] Backup demo video (5-minute uncut)
- [ ] Q&A prep rehearsed in English

---

End of CLAUDE.md. Keep this file under ~150 lines.
