# Claude Latest Track A / Track B Handoff — 2026-06-10

This is the latest compact handoff document for Claude Code.

Use this file first if you need a single starting point.

Important rule:

- Do not mix Track A presentation work and Track B long-term occlusion research in the same commit.

## 1. Project split

### Track A

Purpose:

- presentation/demo implementation only
- normal camera input
- no AR/effect-occlusion research in this track

Main goal:

- keep the demo stable
- keep the tactical visualization readable
- keep the action demo runnable on local / Pi4-oriented paths

### Track B

Purpose:

- long-term HADO real-match player detection under AR/effect overlap
- hard-frame mining
- Roboflow relabel loop
- retraining and real-video evaluation
- later skeleton/action verification after detector stabilization

Main goal right now:

- stabilize the `player-only` detector on real match footage
- especially reduce over-detection caused by effects and shields

## 2. Track A current state

Track A should stay limited to what was actually presented:

- single-camera player tracking
- YOLOv8n-pose based bbox + 17 keypoints
- IoU tracking
- homography to bird-eye court coordinates
- rule-based tactical engine
- 7-action real-time action demo
- Pi4-oriented runtime path (NCNN / ONNX / PT fallback)

Most important files:

- `src/demo.py`
- `src/guide.py`
- `src/demo_pose.py`
- `src/action_demo.py`
- `src/tactic_engine.py`
- `src/camera.py`
- `docs/TRACK_A_PRESENTATION_SCOPE_BRIEF_2026_06_10.md`

Recent validated Track A behavior:

- `src/demo.py` supports `--frames` and `--max-frames`
- `src/demo_pose.py` supports camera/video input and `--frames` / `--max-frames`
- `src/action_demo.py` supports `--frames` / `--max-frames`, `--threaded`, `--record`

Validated smoke commands:

```bash
./hado_venv/bin/python -m src.demo --headless --max-frames 30
./hado_venv/bin/python -m src.demo_pose --video data/1.mp4 --headless --max-frames 5
./hado_venv/bin/python -m src.action_demo --source data/1.mp4 --headless --frames 5
```

Observed result:

- each command exits normally
- NCNN runtime path is usable locally

Track A next work for Claude:

1. improve HUD readability and layout polish
2. verify the preferred presentation execution path
3. keep all work inside presentation/demo scope only

Do not do this in Track A:

- Roboflow relabel loop
- Track B retraining scripts
- effect-aware evaluation
- occlusion-focused tracker research

## 3. Track B current state

Track B is now beyond one-off experiments.

It already has:

- batch real-video intake
- hard-frame mining
- priority CSV generation
- Roboflow upload bundle generation
- retraining loop
- run leaderboard
- best checkpoint auto-selection
- skeleton review export preparation
- OpenAI Batch QA preparation scripts

Current recommended baseline:

- run name: `trackb_v4_player_only`
- checkpoint pointer:
  - `outputs/track_b_retrain_runs/track_b_current_best_checkpoint.txt`
- leaderboard:
  - `outputs/track_b_retrain_runs/track_b_run_leaderboard.md`

Current interpretation:

- the bigger current issue is not simple miss-only behavior
- the current batch review suggests over-detection from AR/effect overlap is a major failure mode

So the current Track B priority is:

- keep this phase `player-only`
- relabel useful playable hard frames
- reduce false player boxes on effects/shields

## 4. Track B most important docs

- `docs/TRACK_B_HANDOFF_REPORT_2026_06_09.md`
- `docs/TRACK_B_TECHNICAL_REPORT_2026_06_09.md`
- `docs/TRACK_B_COLLABORATION_REPORT_2026_06_09.md`
- `docs/TRACK_B_OPERATOR_PLAYBOOK_2026_06_10.md`
- `docs/TRACK_B_BATCH_REVIEW_RELABEL_GUIDE_2026_06_10.md`
- `docs/TRACK_B_SKELETON_PHASE_BRIEF_2026_06_10.md`
- `docs/TRACK_A_B_OPENAI_GITHUB_REFERENCE_2026_06_10.md`

## 5. Track B most important tools

- `tools/prepare_track_b_batch_review.py`
- `tools/process_track_b_video_batch.py`
- `tools/summarize_track_b_batch.py`
- `tools/prepare_roboflow_upload_bundle.py`
- `tools/retrain_track_b_export.py`
- `tools/compare_track_b_runs.py`
- `tools/export_track_b_pose_review.py`
- `tools/build_track_b_openai_batch.py`
- `tools/summarize_track_b_openai_batch.py`
- `tools/build_track_b_relabel_review_packet.py`

## 6. Track B current relabel target

The current recommended relabel bundle is:

- `data/track_b_batch_review/batch02_v4_eval/roboflow_upload_bundle_v2_playable_bias`

Most important files inside:

- frames to upload:
  - `data/track_b_batch_review/batch02_v4_eval/roboflow_upload_bundle_v2_playable_bias/frames`
- frame manifest:
  - `data/track_b_batch_review/batch02_v4_eval/roboflow_upload_bundle_v2_playable_bias/upload_manifest.csv`
- summary:
  - `data/track_b_batch_review/batch02_v4_eval/batch02_summary.md`
- priority CSV:
  - `data/track_b_batch_review/batch02_v4_eval/batch02_priority.csv`

Recommended local pre-review packet:

```bash
./hado_venv/bin/python tools/build_track_b_relabel_review_packet.py \
  --manifest "data/track_b_batch_review/batch02_v4_eval/roboflow_upload_bundle_v2_playable_bias/upload_manifest.csv" \
  --out-dir "outputs/track_b_relabel_review_packet/batch02_v4_playable_bias" \
  --cols 4
```

This creates:

- `relabel_contact_sheet.jpg`
- `relabel_checklist.csv`
- `README.md`

Relabel rules:

- `player` class only
- remove fake player boxes caused by AR effects / shields
- add missing player boxes when the player is clearly visible
- keep boxes tight around visible player body
- non-playable frames may be skipped

## 7. Track B OpenAI Batch QA flow

This is optional support, not the main detector training method.

Purpose:

- automatically review hard frames before human relabeling
- estimate playable vs skip candidates
- estimate overdetect / underdetect / low-confidence tendencies
- re-rank relabel priority

Build batch input:

```bash
./hado_venv/bin/python tools/build_track_b_openai_batch.py \
  --manifest "data/track_b_batch_review/batch02_v4_eval/roboflow_upload_bundle_v2_playable_bias/upload_manifest.csv" \
  --out-jsonl "outputs/track_b_openai_batch/batch02_frame_qa_requests.jsonl" \
  --out-manifest "outputs/track_b_openai_batch/batch02_frame_qa_manifest.csv" \
  --max-rows 40
```

Merge downloaded Batch output:

```bash
./hado_venv/bin/python tools/summarize_track_b_openai_batch.py \
  --request-manifest "outputs/track_b_openai_batch/batch02_frame_qa_manifest.csv" \
  --batch-output "outputs/track_b_openai_batch/batch02_frame_qa_output.jsonl" \
  --out-csv "outputs/track_b_openai_batch/batch02_frame_qa_merged.csv" \
  --out-md "outputs/track_b_openai_batch/batch02_frame_qa_summary.md" \
  --out-priority-csv "outputs/track_b_openai_batch/batch02_frame_qa_priority.csv"
```

This was smoke-tested locally with:

- manifest -> request jsonl generation
- synthetic batch output -> merged CSV / priority CSV / summary generation

## 8. What Claude should do next

### If working on Track A

Do only this:

1. polish demo readability
2. preserve presentation scope
3. validate smoke commands after each change

### If working on Track B

Do this order:

1. keep player-only relabel loop active
2. use the current batch02 playable-bias bundle as the main relabel target
3. retrain on the next Roboflow export
4. compare leaderboard and real-video behavior
5. only after detector stabilization, move into skeleton verification

## 9. What not to mix

Never mix these in one commit:

- Track A demo overlay polish
- Track B relabel / retrain / occlusion evaluation work

Do not commit:

- large raw videos
- exported model zips
- local weight artifacts unless explicitly intended

## 10. Recommended single sentence summary for Claude

Use Track A only for presentation-level demo stabilization.

Use Track B only for real-match player-only detector improvement under AR/effect overlap, with batch review -> Roboflow relabel -> retrain -> real-video re-evaluation as the main loop.
