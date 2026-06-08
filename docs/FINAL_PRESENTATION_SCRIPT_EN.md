# HADO Smart Court — Final Presentation Script (English)

> 최종발표 2026-06-15 용 영어 발표 대본  
> 예상 시간: 약 4분 (슬라이드당 30~45초)  
> 팁: 각 슬라이드 전환 전에 1초 정지. 숫자는 자신감 있게 또렷이.

---

## Slide 1 — Title

> "Good afternoon. My name is Jinu Park, and this is **HADO Smart Court IoT** —
> a real-time player tracking and tactical guidance system I built for the AR sport HADO."

---

## Slide 2 — Problem

> "HADO is played on a 10-meter-by-6-meter indoor court, 3 versus 3.
> Players move constantly, and the game happens too fast to analyze positioning live.
>
> Coaches can review footage after the match — but by then it is too late.
> Our question was: *can we give real-time tactical feedback with only a single camera and a Raspberry Pi?*"

---

## Slide 3 — System Overview

> "Here is the overall pipeline.
>
> A fixed camera captures the court.
> The Pi 4 runs YOLOv8n-pose, which gives us each player's bounding box and 17 body keypoints.
> An IoU tracker maintains consistent player IDs across frames.
> Homography transforms pixel coordinates to real court coordinates in meters.
>
> Everything then feeds into two outputs:
> a **bird-eye view** showing live positions, and a **tactical engine** that recommends where each player should move."

---

## Slide 4 — Level 1: Tracking

> "Level 1 is purely detection and tracking.
>
> We use YOLOv8n — the nano variant — at 320×320 input.
> The foot point, meaning the bottom center of each bounding box, is used for homography projection.
> That point is always on the ground plane, so the coordinate conversion is geometrically correct.
>
> On a Mac we reach about 98 fps inference.
> On the Raspberry Pi 4 with the NCNN ARM-optimized model, we target 15 frames per second,
> which is the minimum for real-time tactical relevance."

---

## Slide 5 — Level 2: Tactical Engine

> "Level 2 adds the tactical engine.
>
> Six rule-based rules are evaluated every frame per player, in priority order:
> - R5 and R3 are HIGH urgency — backline defense and counter-threat avoidance.
> - R1 through R6 are MID urgency — spacing, lane coverage, gap attack, and team formation.
>
> The reason I chose rules over machine learning here is twofold.
> First, interpretability: 'R3 — counter threat at 2.1 meters' is debuggable and explainable.
> Second, no labeled data yet. The system itself is the data-collection tool for future ML work.
>
> I also built a Movement Model that blends expert-annotated patterns from 635 movement records
> into the rule output, adding positional nuance beyond pure geometry."

---

## Slide 6 — Level 3: Action Recognition

> "Level 3 is a real-time action classifier built on top of the pose keypoints.
>
> It classifies 7 HADO-specific actions: charge, shoot, shield, two dodge directions, crouch, and ready.
>
> The key design challenge was camera-distance invariance.
> If I used pixel-based thresholds, a player 2 meters away and one 5 meters away would be classified differently.
>
> My solution is scale normalization:
>     scale = max(shoulder width, torso height × 0.6, 30 px)
> All thresholds divide by this scale, so the classifier behaves consistently at any camera distance.
>
> I also add team-direction awareness — 'shoot' only fires when the wrist extends *toward the opponent side*,
> preventing a charging motion from being mislabeled as an attack."

---

## Slide 7 — Optimization for Pi 4

> "The most critical engineering challenge was achieving 15+ fps on a Raspberry Pi 4 CPU.
>
> Three optimizations were key.
> First, NCNN model format — the ARM-optimized FP16 variant is about three times faster than ONNX on the Pi.
> Second, threaded camera capture — a background thread always holds the latest frame,
> so the inference loop never waits for the camera.
> Third, image size — reducing input from 640 to 320 pixels was the single largest win."

---

## Slide 8 — Validation

> "For validation, I wrote 180 unit tests covering all core modules.
>
> Each tactical rule has dedicated test cases, including edge cases such as:
> exactly two opponents where they split 50-50 across lanes — which should *not* trigger R6.
>
> For position accuracy, the target is under 10 centimeters RMS error,
> measured against 10 marked reference points on the court.
> I will fill in the actual measurements from on-court testing this week."

---

## Slide 9 — Demo

> *(데모 영상 재생 또는 라이브 시연)*
>
> "Let me show you the system in action.
>
> On the left is the raw camera view with skeleton overlays.
> On the right is the bird-eye view with real-time player positions.
>
> You can see the colored arrows — those are the tactical recommendations.
> Green arrows are MID urgency, red arrows are HIGH urgency.
> The text label shows which rule fired and why.
>
> *(If showing action demo)*
> And here is the action classifier — watch the action panel at the bottom as I move."

---

## Slide 10 — Results & Next Steps

> "To summarize:
>
> - We have a fully offline, single-camera system running on a Raspberry Pi 4.
> - Level 1 tracks players, Level 2 provides tactical guidance, Level 3 classifies actions.
> - 6 tactical rules, 7 action classes, 180 tests passing.
>
> The [TODO] slots — actual fps, RAM, and position error numbers —
> will be filled in after on-court testing this week.
>
> Future work includes multi-camera occlusion handling, model fine-tuning on HADO headset video,
> and auto-tuning rule thresholds from recorded match data.
>
> Thank you. I am happy to take questions."

---

## Buffer Lines (Q&A 전환 시 사용)

- *"That is a great question. Let me address two parts of it..."*
- *"I did not test that specific case, but my hypothesis is..."*
- *"The system currently handles this by... A more robust approach would be..."*
- *"This is exactly the kind of thing I would want to verify in the next on-court session."*

---

## 발표 전 자가 체크

- [ ] 각 슬라이드 번호와 대본이 매핑되는지 확인
- [ ] Slide 4의 fps 숫자를 실측 후 업데이트
- [ ] Slide 8의 [TODO]를 실측 후 업데이트
- [ ] 전체 소리내어 읽기 1회 (목표 4분 이내)
- [ ] demo 영상/라이브 시연 준비 (`./run.sh demo --headless` 사전 실행 확인)
- [ ] 긴장될 때: Slide 7 "three optimizations" — 숫자 3이 나오면 손가락 세기로 속도 조절

---

*이 대본은 템플릿입니다. 발표 당일 자신의 말투로 바꿔도 됩니다.*
*핵심은 숫자(15 fps, 10 cm, 180 tests, 635 patterns)와 이유(why rules not ML).*
