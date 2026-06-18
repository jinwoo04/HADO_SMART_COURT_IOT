# Claude Code Continuation Brief — 2026-06-08

This document is written for Claude Code so it can continue the HADO Smart Court work without mixing the presentation track and the long-term AR/effect research track.

## Current project split

### Track A — IoT mid/final presentation version

Use this for the school IoT presentation and demo.

Scope:
- Camera-based HADO player tracking and tactical visualization.
- Input assumption: normal camera video without intentional AR/effect occlusion experiments.
- Priority: stable demo, clear English explanation, dashboard/visual readability, tests passing.
- Do not mix Track B Roboflow/occlusion research into Track A commits.

Core narrative used in the current English presentation:

1. **Problem**
   - HADO players move quickly, so team positioning and tactical flow are hard to analyze during a live match.

2. **System idea**
   - Use one side-view camera to capture the whole court.
   - Process the video on Raspberry Pi 4 with YOLOv8n-pose.
   - Stream player position data through Socket.IO over local Wi-Fi.
   - Analyze the data on a MacBook server every 2 seconds.
   - Show tactical feedback on a real-time dashboard.

3. **Architecture**
   - Edge layer: camera + Raspberry Pi 4 + YOLOv8n-pose, extracting 17 keypoints per player.
   - Server layer: Node.js + Express + Socket.IO server, with a 40-second sliding buffer.
   - Tactical layer: checks spacing, formation balance, right-side concentration, and forward pressure.

4. **Hardware**
   - Raspberry Pi 4 for edge pose inference.
   - USB side-view camera at 1280x720.
   - MacBook server for tactical analysis and dashboard broadcasting.
   - Smartphone Wi-Fi hotspot for local low-latency networking.

5. **Progress**
   - Completed: pose detection pipeline, Raspberry Pi ONNX inference, MacBook server, real-time streaming, tactical rule engine, camera integration, frame extraction/dataset preparation.
   - Next: movement analysis improvement, model fine-tuning, dashboard refinement, performance optimization, final demo video.

### 3-minute English script used as the latest draft

Slide 1:

> Good morning. We are Jinwoo Park and Junhyeok Jung, and our project is **HADO Smart Court**.
>
> Our goal is to build an IoT-based system that tracks HADO players in real time and provides tactical analysis during a match.

Slide 2:

> In HADO, players move very quickly, so it is difficult to understand team positioning during the game.
>
> Our system uses one side-view camera to capture the court. The Raspberry Pi processes the video with YOLOv8n-pose, sends player position data through Wi-Fi, and the MacBook server analyzes the data every two seconds.
>
> Finally, the result is shown on a real-time tactical dashboard.

Slide 3:

> Our system has three layers.
>
> First, the **edge layer** captures video and extracts player keypoints on the Raspberry Pi.
>
> Second, the **server layer** receives the data, keeps a 40-second buffer, and runs analysis every two seconds.
>
> Third, the **tactical layer** checks player spacing, formation balance, right-side concentration, and forward pressure. Then it sends tactical advice to the dashboard.

Slide 4:

> For hardware, we use four main components.
>
> The Raspberry Pi 4 runs pose estimation at the edge. The USB camera captures all six players from the side view. The MacBook runs the server and tactical analysis engine. And the smartphone hotspot creates a local Wi-Fi network for low-latency communication.
>
> This makes the system portable and low-cost.

Slide 5:

> So far, we have completed the pose detection pipeline, Raspberry Pi inference, MacBook server, real-time streaming, tactical rule engine, camera integration, and dataset preparation.
>
> Next, we will improve movement analysis, fine-tune the model, refine the dashboard, optimize performance, and record the final demo video.
>
> Our final goal is a fully integrated real-time tactical coaching system for HADO matches.
>
> Thank you.

Recommended Claude Code work for Track A:
- Keep the 3-minute presentation script aligned with `/Users/jinu/Downloads/제목 추가.pptx`.
- Help fill final report/PPT TODO metrics after Pi 4 measurements are available.
- Do not change Track B model files or Roboflow research scripts in Track A commits.

## Track B — Long-term AR/effect occlusion player detection

Use this for the owner's personal project after the IoT presentation.

Scope:
- Player detection/tracking when HADO virtual effects overlap players.
- Roboflow dataset cleanup, hard-sample mining, effect/occlusion-specific evaluation, and model retraining.
- Do not mix this into presentation/demo-finalization commits.

### Latest Roboflow V4 player-only experiment

Source export:
- `/Users/jinu/Downloads/hado-player.v4i.yolov8 (1)`

Important label finding:
- Original classes: `['0', 'object', 'player']`
- Class `0` and class `2` both visually mean player.
- Class `1` (`object`) had only one label and was dropped.
- Normalized mapping: `0 -> player`, `2 -> player`, `1 -> drop`.

Normalized dataset:
- Path: `/Users/jinu/Documents/Codex/2026-06-08/codex-codex/work/track_b_roboflow_export/hado_player_v4_player_only_detect`
- Total player labels: 3598
- train: 945 images, 2564 player labels
- valid: 227 images, 627 player labels
- test: 134 images, 407 player labels

Latest best model:
- `/Users/jinu/Documents/Codex/2026-06-08/codex-codex/outputs/track_b_roboflow_export_review/hado_player_v4_player_only_yolov8n_mps_e30_best.pt`

Previous checkpoint on V4 test:
- Precision: 0.570
- Recall: 0.590
- mAP50: 0.559
- mAP50-95: 0.316

New V4 player-only test result:
- Precision: 0.619
- Recall: 0.686
- mAP50: 0.672
- mAP50-95: 0.430

Tag-level V4 test result:
- `clean`: 30 images, mAP50-95 0.443
- `partial`: 21 images, mAP50-95 0.435
- `heavy`: 8 images, mAP50-95 0.494

Interpretation:
- The hard-sample review and class normalization significantly improved recall and mAP.
- This is now the Track B player-only baseline.
- Effect detection should be a separate experiment later; current effect/object labels are not enough.
- Heavy-tag metrics are promising but based on only 8 images, so add more heavy/effect-overlap examples before making strong claims.

### Actual video smoke test added

New utility:
- `tools/run_player_model_on_video.py`

Purpose:
- Run the V4 player-only detector on actual match videos.
- Sample frames with `--frame-stride`.
- Export annotated preview video and per-frame detection counts CSV.

Example commands:

```bash
# Preview with annotated video
~/hado_venv/bin/python3 tools/run_player_model_on_video.py \
  --video data/dinos_preview.mp4 \
  --frame-stride 5 \
  --max-frames 300 \
  --out-dir data/track_b_reviews

# CSV-only low-confidence sweep
~/hado_venv/bin/python3 tools/run_player_model_on_video.py \
  --video data/5.mp4 \
  --frame-stride 10 \
  --max-frames 600 \
  --out-dir data/track_b_reviews/conf010 \
  --conf 0.10 \
  --no-video
```

Smoke-test observations:
- `data/dinos_preview.mp4`, conf 0.25:
  - 60 sampled frames
  - detections min/avg/max: 0 / 1.18 / 5
  - zero-detection frames: 16
  - average nonzero confidence: 0.501
- `data/dinos_preview.mp4`, conf 0.10:
  - detections min/avg/max: 0 / 2.62 / 9
  - zero-detection frames: 13
  - average nonzero confidence: 0.322
- `data/5.mp4`, conf 0.10:
  - 60 sampled frames
  - detections min/avg/max: 3 / 6.72 / 15
  - zero-detection frames: 0
  - average confidence: 0.203

Interpretation:
- The Roboflow test score improved, but real videos still show domain gaps.
- Some videos produce low-confidence duplicate/false candidates when confidence is lowered.
- Next Track B loop should mine hard frames directly from real videos, not only from the Roboflow split.

### Hard-frame mining utility added

New utility:
- `tools/mine_video_hard_frames.py`

Purpose:
- Read the CSV from `tools/run_player_model_on_video.py`.
- Extract suspicious frames for Roboflow relabeling.
- Suspicious means:
  - too few detections,
  - too many detections,
  - or low average confidence.
- Write individual frame JPGs, a `hard_frame_manifest.csv`, and a contact sheet.

Example commands already run:

```bash
~/hado_venv/bin/python3 tools/mine_video_hard_frames.py \
  --video data/dinos_preview.mp4 \
  --csv data/track_b_reviews/conf010/dinos_preview_player_only_detections.csv \
  --out-dir data/track_b_reviews/hard_frames/dinos_preview_conf010 \
  --low-count 1 --high-count 8 --low-conf 0.18 \
  --min-gap-frames 15 --max-samples 24

~/hado_venv/bin/python3 tools/mine_video_hard_frames.py \
  --video data/5.mp4 \
  --csv data/track_b_reviews/conf010/5_player_only_detections.csv \
  --out-dir data/track_b_reviews/hard_frames/5_conf010 \
  --low-count 1 --high-count 10 --low-conf 0.18 \
  --min-gap-frames 30 --max-samples 24
```

Generated review outputs:
- `data/track_b_reviews/hard_frames/dinos_preview_conf010/hard_frame_contact_sheet.jpg`
  - Mostly empty-court or player-entry frames. Useful for negative/background filtering.
- `data/track_b_reviews/hard_frames/5_conf010/hard_frame_contact_sheet.jpg`
  - Useful real hard frames with AR effects, low confidence, and duplicate/over-detection cases.
- `data/track_b_reviews/hard_frames/*/hard_frame_manifest.csv`
  - Frame index, timestamp, detection count, average confidence, reason, and image path.

### 10-match real-video batch intake added

Detailed note:
- `docs/TRACK_B_HANDOFF_REPORT_2026_06_09.md`

Batch source:
- Google Drive import saved to `data/drive_imports/batch01`
- Files: `match01.mp4` through `match010.mp4`

New batch wrapper:
- `tools/process_track_b_video_batch.py`
- Purpose:
  - process multiple match videos in one command,
  - run the current player-only detector,
  - mine hard frames,
  - and write a batch manifest.

Combined batch manifest:
- `data/track_b_batch_review/batch01_combined_manifest.csv`

Prepared Roboflow upload bundles:
- `data/track_b_batch_review/roboflow_upload_bundle_batch01_priority.zip`
  - structured 90-frame bundle with CSV queue and README
- `data/track_b_batch_review/roboflow_upload_bundle_batch01_priority_manifest.csv`
- `data/track_b_batch_review/roboflow_upload_bundle_batch01_phase1.zip`
  - compact 36-frame bundle with only `label_priority` samples
- `data/track_b_batch_review/roboflow_upload_bundle_batch01_phase1_manifest.csv`

First-pass summary across 10 matches:
- 10/10 videos imported and processed
- frame sampling: `frame_stride=30`
- confidence threshold: `0.10`
- total hard frames saved: `175`

Most important first-pass findings:
- `match02` had the lowest average nonzero confidence (`0.2718`) and many zero-player frames (`9`).
- `match09` was also weak (`avg_nonzero_conf 0.2827`, `avg_count 3.653`).
- `match08` had the highest average detections (`5.051`) but also many zero frames (`9`) and strong AR shield overlap.
- Contact sheets confirm that the next Roboflow pass should focus on:
  - intro/scoreboard exclusion,
  - shield/projectile overlap,
  - and duplicate/over-detection cleanup.

Recommended next Roboflow upload priority:
1. `match02`
2. `match09`
3. `match01`
4. `match06`
5. `match08`

Most practical next move:
- upload `roboflow_upload_bundle_batch01_phase1.zip` first
- complete a player-only V5 relabel pass on those 36 frames
- then expand to the 90-frame priority bundle if the first result still leaves large real-video gaps

Recommended next Roboflow action:
- Upload the selected frames from `data/track_b_reviews/hard_frames/5_conf010/frames/`.
- Correct player boxes carefully where AR effects overlap players.
- Keep some empty/transition frames as background or explicitly exclude them from training, depending on Roboflow workflow.
- After relabeling, export a new V5 dataset and repeat the player-only training loop.

Recommended Claude Code work for Track B:
1. Review the generated contact sheets and decide which frames should be uploaded to Roboflow.
2. Add more `heavy` and `partial` real-video frames to the validation/test set.
3. Export a V5 Roboflow dataset after relabeling and repeat the player-only training loop.
4. Keep a separate player-only baseline and do not force effect detection until effect labels are reliable.
5. Later, try segmentation training from polygon labels for overlap boundaries.

## Branch and commit discipline

Track A branch:
- `presentation/demo-finalization`
- Example commits:
  - `docs: add mid-presentation english script`
  - `fix: polish presentation demo overlays`

Track B branch:
- `research/occlusion-robust-tracking`
- Example commits:
  - `feat: add player-only video review utility`
  - `docs: document v4 player-only detection baseline`
  - `feat: mine hard frames from real hado videos`

Do not mix these in one commit:
- Track A presentation/demo code and Track B Roboflow/model research.
- Model weights and large generated videos unless explicitly requested.
- Camera-specific calibration files.

## Validation already run

```bash
./run.sh test
```

Result:
- All module tests passed.
