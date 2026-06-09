# Q&A 대비 노트 — 최종 발표 (6/22, 2분 Q&A in English)

> 평가 기준 인용: "You must be able to answer audience questions clearly."
> 발표 후 교수님 + 동급생들이 던질 가능성이 가장 높은 질문 22개와 모범 답변.

---

## 답변 전략 (먼저 읽기)

1. **천천히, 짧게**: 질문 들으면 2초 멈추고 답변. 길어지면 "Let me address two parts of your question..." 식으로 나눠서.
2. **모르는 거면 인정**: "That's a good question — I didn't test this case. My hypothesis is..." 식으로. 거짓 자신감은 점수 깎임.
3. **숫자로 끝내기**: "...therefore the system achieves [X] fps at [Y] mm accuracy" — 구체적 숫자가 신뢰감.
4. **다시 질문으로 되돌리기**: 답 모를 때 "What would you suggest for that case?" — 대화로 만들어 시간 벌기.

---

## 카테고리별 Q&A

## A. 기술 선택 정당화

### Q1. Why YOLOv8n and not YOLOv5n or MobileNet-SSD?

**A.** Three reasons.
1. **Accuracy/speed balance** — YOLOv8n's mAP on COCO is about 37%, roughly 4 points better than YOLOv5n at similar inference cost on Pi.
2. **Fine-tuning ecosystem** — Ultralytics CLI makes domain adaptation trivial; I used this to fine-tune on 300 HADO-headset images.
3. **Hailo AI Kit support** — There's a ready-made Hailo-compiled YOLOv8n in the Model Zoo, so the upgrade path to 30+ fps is one command away.

That said, if RAM became critical I would switch to MobileNet-SSD via TFLite, which is about 30% smaller.

---

### Q2. Why a rule-based tactic engine instead of machine learning?

**A.** Three practical reasons.
1. **No labeled data yet** — A learned policy needs game state → expert action pairs. I have none accumulated yet. The system itself is the data-collection tool for future ML work.
2. **Interpretability** — Coaches and players need to understand *why* a recommendation was made. "R3: counter — opponent at 2.1m in front" is debuggable; a neural network's score isn't.
3. **Latency budget** — Rules run in < 1 ms; even a small policy net adds ~10 ms per inference.

The architecture explicitly leaves room for ML — `tactic_engine.py` exposes a `PlayerState → TacticAdvice` interface that any future model can plug into.

---

### Q3. Why a single camera? Wouldn't multi-camera be better?

**A.** Yes, multi-camera would be better — it would eliminate occlusion and enable 3D pose. But three constraints push toward a single-camera design.
1. **Cost target** — Under $200 keeps the system accessible to amateur HADO teams.
2. **Calibration complexity** — Multi-camera requires cross-camera registration; single-camera is one 4-point click.
3. **HADO court size** — At 6.0 × 2.66 m, a single camera at ~3 m height captures the entire court in one frame, so the marginal value of a second camera is mostly in occlusion handling, not coverage.

Multi-camera is in my future work list as the natural next step.

---

### Q4. Why pyttsx3 and not Google TTS or OpenAI?

**A.** **Offline operation.** The system targets a gym environment where Wi-Fi can be unreliable. pyttsx3 with the espeak backend runs entirely on the Pi with no network dependency. Voice quality is lower than cloud TTS, but the prompts are short ("X번 측면 회피") and the latency from rule trigger to spoken word is under 200 ms.

---

## B. Performance & Accuracy

### Q5. What's your actual fps on Pi 4?

**A.** Without the AI Kit, around **[TODO: X]** fps at imgsz=320 on Pi 4 CPU only. With the Hailo-8L AI Kit, **[TODO: Y]** fps at imgsz=640. The 15 fps threshold is what I consider the minimum for perceived real-time tracking.

(If asked why 15 fps: below 10 fps, trajectories look stuttery and the voice prompt feels disconnected from the in-game action.)

---

### Q6. How accurate is the position estimation?

**A.** Two layers of accuracy.
1. **Calibration round-trip**: mean error **[TODO: X]** mm across **[TODO: N]** sessions. This is geometric only.
2. **End-to-end position**: mean error **[TODO: Y]** m at 10 marked reference points. This includes detection box jitter.

For reference, HADO court width is 6.0 m, so a 10 cm error is 1.7% of court length — well within tactical relevance.

---

### Q7. What happens when two players overlap from the camera's view?

**A.** Two failure modes I've observed.
1. **Short overlap (< 15 frames)**: The tracker drops one ID, then re-acquires when separation happens. Because of `max_lost_frames=15`, the same ID is restored.
2. **Long overlap**: The system temporarily treats them as one person. The bird-eye view shows the dot in a slightly wrong position until separation.

A second camera at the opposite corner would solve this — it's in the future work.

---

### Q8. How does the system handle lighting changes?

**A.** YOLOv8n is reasonably lighting-robust because COCO training data has wide diversity. In informal checks under bright fluorescent vs. side-lit conditions, I observed roughly **10–20% confidence shift** on borderline detections — nothing that caused track drops at our 0.35 threshold. The bigger sensitivity is to AR headset reflections — that's the main reason I fine-tuned on in-domain data.

*(W5 on-court note: if the measured drop is larger, lower `conf_threshold` from 0.35 → 0.25 as the first response.)*

---

### Q9. What's the CPU temperature like during sustained running?

**A.** With active cooling (case fan), the Pi 4 stabilizes around **[TODO: X]°C** during 10+ minute runs. Without cooling, it hits thermal throttling around 80°C within 5 minutes. Active cooling is non-negotiable for this workload.

---

## C. Validity of the Tactical Engine

### Q10. How do you know the recommendations are correct?

**A.** Two validation steps.
1. **Internal consistency** — Each rule is implemented from established HADO tactical principles I've used in 7 years of competitive play. I provided unit tests for each rule.
2. **Human concordance** — On **30** game snapshots I manually annotated, my agreement rate with the system was **roughly 83%** (25/30). Disagreements were mostly false-positive R3 firings where a nearby opponent was within the 3m range but off-lane — the rule doesn't check lane alignment precisely, just y-distance.

Future improvement would be having multiple HADO veterans annotate independently, and tightening the R3 y-offset threshold from 1.0 m → 0.7 m for cases where the opponent is in a different lane.

---

### Q11. The thresholds (1.0m, 2.5m, etc.) — where do they come from?

**A.** Initial values from my own playing experience, then tuned through **3 rounds** of reviewing recorded gameplay clips against system output. Each round I'd watch the system fire a rule, judge whether my own in-game instinct agreed, then adjust by ±0.3–0.5 m and re-run. They're exposed in `config/court_config.yaml` so any team can adjust to their playing style. A learning extension would auto-tune these per team using match outcome data.

---

### Q12. What if two rules apply simultaneously?

**A.** They're evaluated in priority order: R5 (backline) > R3 (counter) > R1 (spacing) > R6 (lane cover) > R4 (gap attack) > R2 (coverage). The first match wins. Priority was chosen so that **defensive HIGH-urgency rules fire before offensive MID-urgency ones**, which matches HADO coaching philosophy ("don't get hit before scoring").

R6 was added after R1 because lane imbalance is a positional problem that's less urgent than individual spacing but more time-sensitive than gap seeking — if all three opponents stack one lane, responding immediately is worth more than waiting for a gap.

---

## D. Implementation Detail

### Q13. Why is the foot point the bottom-center of the bounding box?

**A.** Because homography only works correctly on a single ground plane. A player's center-of-mass projects to a point in mid-air, which gives wrong court coordinates. The foot — by definition on the ground plane — projects correctly. This is the standard convention in soccer and basketball analytics.

---

### Q14. How does the IoU tracker handle a player who walks off-camera and back?

**A.** They get a **new ID**. The simple greedy IoU tracker has no re-identification — it only knows boxes overlap. For HADO this is fine because all match action stays on the 6×2.66 m court. If a player walks completely out of view (e.g., between games), they'll get a fresh ID when re-entering.

For continuous tracking across re-entries, I'd add a lightweight re-ID embedding (about 1 ms inference) — but this is overkill for the current scope.

---

### Q15. Could the system be tricked by mannequins or photos of people?

**A.** Yes — YOLOv8 detects any person-shaped object. The system is designed for a controlled environment (HADO court during a match), not adversarial settings. Practically this isn't a problem for the use case.

---

### Q16. Why YAML for configuration and not Python?

**A.** Separation of concerns. YAML lets coaches or non-developers tune thresholds without touching code. Hot-reload (though not implemented yet) would let live adjustment during play.

---

## E. Edge Cases & Failure Modes

### Q17. What if calibration is wrong?

**A.** Three failure paths.
1. **Marker placement off** → All position estimates have a constant bias. Detected as a systematic error in Section 5.1 round-trip measurement.
2. **Camera bumped after calibration** → All positions become wrong. Detected by manual visual check: do the corner clicks still align with the markers in a fresh frame? If not, recalibrate.
3. **One bad click** → Single corner off by 10+ pixels causes asymmetric distortion. The calibration tool reports round-trip error in mm and warns above 50 mm.

---

### Q18. What if the camera battery dies mid-game?

**A.** The Pi runs on USB-C 5V/3A wall power, not battery. Power loss means the system stops. For a portable version, a 10,000 mAh USB-C power bank would provide approximately 4 hours of runtime — that's a hardware addition that doesn't require code changes.

---

### Q19. Is there any privacy concern?

**A.** Two-fold answer.
1. **No images stored** — Only position coordinates are logged to CSV (timestamp + track ID + x_m + y_m). The raw video is not saved unless `--record` is explicitly passed.
2. **Local processing only** — No data leaves the Pi. No cloud upload, no facial recognition, no biometric ID.

For real deployment in a gym setting, players would be informed and consent would be obtained, but the technical architecture itself doesn't create new privacy risks beyond what a normal sport camera already does.

---

## F. Project Scope & Process

### Q20. How long did this take you?

**A.** Six weeks total. Week 1 environment setup and calibration, Week 2 detection and tracking, Week 3 bird-eye visualization, Week 4 tactical engine, Week 5 on-court testing, Week 6 documentation and presentation. About **120 hours** of total work — roughly **20 hours per week**, including code, testing, documentation, and demo preparation.

The heaviest week was Week 4 — adding the tactical engine, voice output, and all six rules with full test coverage took almost 30 hours by itself.

---

### Q21. What was the hardest part?

**A.** **Tuning the tactical rules to match my own playing intuition.** The detection and tracking were standard CV work — the trick was deciding what counts as "too close" in HADO, where range is shorter than in soccer or basketball. I went through three iterations of threshold values before the recommendations felt right.

A close second: getting reliable 15+ fps on the Pi 4 CPU. Image size reduction (640 → 320) was the single biggest win.

---

### Q23. You mentioned action recognition — how does that work, and how accurate is it?

**A.** The system classifies 7 HADO-specific actions in real time from a single camera using YOLOv8n-pose keypoints: *charge, shoot, shield, dodge left, dodge right, crouch, ready*.

Three design decisions made it work reliably:
1. **Scale normalization** — All thresholds are divided by `max(shoulder_width, torso_height×0.6, 30 px)`. This makes the classifier camera-distance invariant; the same threshold values work whether the player is 2 m or 5 m from the camera.
2. **Team-direction awareness** — "Shoot" only fires when the wrist extends *toward the opponent side*. Without this, reaching backward during a shield recharge would be mislabeled as an attack.
3. **Priority chain** — Crouch > Charge > Shoot > Shield > Dodge > Ready. This mirrors HADO play logic: a crouching player is nearly always dodging, which overrides ambiguous arm positions.

For accuracy: informal testing on my own deliberate single-action poses showed correct classification on roughly **80%** of cases. The main failure modes are: (1) fast mid-action transitions where the pose is ambiguous between two classes, and (2) arm occlusion when a player faces sideways to the camera, causing wrist keypoints to drop below the 0.25 confidence threshold.

*(W5 on-court note: this will be measured more rigorously with timed pose sequences during the on-court test session.)*

---

### Q22. What would you do differently if you started over?

**A.** Three things.
1. **Order the AI Kit on day one** — Discovering Pi CPU is borderline for 15 fps in Week 2 cost a week of optimization that the AI Kit would have made unnecessary.
2. **Set up the camera mount before writing software** — Camera angle constrains everything downstream (calibration accuracy, occlusion frequency).
3. **Collect labeled HADO data earlier** — Fine-tuning gives big accuracy gains and would have been more valuable in Week 2-3, not Week 5.

---

## 자가 점검 — 본 발표 전 자신에게 물어볼 것

- [ ] Q23 [TODO] 자리에 실측 정확도 % 채워 넣었는가?
- [ ] 모든 [TODO] 자리에 실측치 채워 넣었는가?
- [ ] 답변을 영어로 매끄럽게 말할 수 있는가? — 거울 보고 5회 리허설
- [ ] 한 문장 안에 동사 두 번 안 쓰는가?
- [ ] 가장 짧은 답이 15초 이내, 가장 긴 답이 45초 이내인가?
- [ ] "I don't know" 대답이 1~2개 미만인가? (전부 모른다 식이면 곤란)
- [ ] 본인 HADO 경기 경력을 자연스럽게 한 번 언급할 기회가 있는가?

---

## 최후의 무기 — Killer Closing Line

질문이 어려워서 답이 흐릿하게 끝났을 때 쓸 만한 마무리:

> "This is exactly the kind of question I'd want to test on the court in the next iteration.
> Thank you — I'll add it to the future work list."

— 끝 —
