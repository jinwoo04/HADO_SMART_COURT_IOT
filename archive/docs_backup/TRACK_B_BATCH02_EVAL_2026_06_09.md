# Track B Batch02 Evaluation (2026-06-09)

## Scope

- Track: `B`
- Goal: evaluate the current `player-only` V4 local model on a fresh set of 10 new HADO match videos.
- Source videos: `data/drive_imports/batch02_raw`
- Model used:
  `outputs/track_b_retrain_runs/trackb_v4_player_only/train/yolov8n_player_only/weights/best.pt`

## Batch Evaluation Command

```bash
./hado_venv/bin/python tools/process_track_b_video_batch.py \
  --video-dir "data/drive_imports/batch02_raw" \
  --out-dir "data/track_b_batch_review/batch02_v4_eval" \
  --model "outputs/track_b_retrain_runs/trackb_v4_player_only/train/yolov8n_player_only/weights/best.pt" \
  --device auto \
  --conf 0.10 \
  --frame-stride 30 \
  --max-samples 18
```

## High-Level Result

- Videos processed: `10`
- Output manifest:
  `data/track_b_batch_review/batch02_v4_eval/batch_manifest.csv`
- Hard-frame contact sheets:
  `data/track_b_batch_review/batch02_v4_eval/hard_frames/match*/hard_frame_contact_sheet.jpg`

The dominant failure mode in this batch was not "missing every player" but
"over-detecting players" in heavy AR/effect overlap scenes. Most selected hard
frames were tagged as `high_count_10`, `high_count_11`, or `high_count_12`.

## Per-Match Hard Frame Counts

- `match01`: `12`
- `match02`: `16`
- `match03`: `14`
- `match04`: `14`
- `match05`: `11`
- `match06`: `10`
- `match07`: `7`
- `match08`: `18`
- `match09`: `14`
- `match10`: `14`

Priority review targets from this batch:

- `match08`: many shield/effect overlap scenes and dense multi-player layouts
- `match02`: many high-count scenes early in the match
- `match03`, `match04`, `match09`, `match10`: steady over-detection patterns

## Prepared Roboflow Upload Bundle

Recommended bundle for the next annotation pass:

- Folder:
  `data/track_b_batch_review/batch02_v4_eval/roboflow_upload_bundle_v2_playable_bias`
- ZIP:
  `data/track_b_batch_review/batch02_v4_eval/roboflow_upload_bundle_v2_playable_bias.zip`
- Size: about `16 MB`
- Frames included: `40`
- Distribution: `4` frames per match across `match01` to `match10`

This bundle keeps the pass `player-only` and biases toward playable court
frames with strong over-detection symptoms.

## Labeling Guidance For This Bundle

- Keep this pass `player` only.
- Do not add effect classes yet.
- Use tight, consistent player boxes around visible body regions.
- If a frame is clearly intro, scoreboard-only, or end-screen, skip it instead
  of forcing a label.
- In shield overlap scenes, label the player body you can actually see; do not
  expand boxes to cover the whole shield.

## Next Step After Annotation

1. Upload the `roboflow_upload_bundle_v2_playable_bias/frames` folder to the
   fresh Track B Roboflow project.
2. Annotate only the player boxes.
3. Generate the next dataset version.
4. Export YOLOv8 format.
5. Retrain locally with `tools/retrain_track_b_export.py`.

## One-Command Replay

For future batches, the full review loop can now be replayed with:

```bash
./hado_venv/bin/python tools/prepare_track_b_batch_review.py \
  --video-dir "data/drive_imports/batch02_raw" \
  --out-dir "data/track_b_batch_review/batch02_v4_eval" \
  --model "outputs/track_b_retrain_runs/trackb_v4_player_only/train/yolov8n_player_only/weights/best.pt" \
  --frame-stride 30 \
  --max-samples 18 \
  --bundle-make-zip
```

If detections already exist and only the summary/bundle should be regenerated:

```bash
./hado_venv/bin/python tools/prepare_track_b_batch_review.py \
  --skip-process \
  --out-dir "data/track_b_batch_review/batch02_v4_eval" \
  --bundle-make-zip
```
