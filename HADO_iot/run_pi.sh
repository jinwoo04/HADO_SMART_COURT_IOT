#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════
# Launch script — Raspberry Pi edge streamer
# ════════════════════════════════════════════════════════════════

set -e

cd "$(dirname "$0")/pi"

# Verify NCNN model is present
if [ ! -d yolov8n-pose_ncnn_model ]; then
  echo "❌ NCNN model not found: yolov8n-pose_ncnn_model/"
  echo "   Run export_ncnn.py on your Mac first, then SCP the folder here."
  exit 1
fi

# Verify camera
if [ ! -e /dev/video0 ]; then
  echo "❌ /dev/video0 not found — is the Mokose HDMI capture plugged in?"
  exit 1
fi

echo "✅ Camera detected: /dev/video0"
echo "▶  Starting edge streamer..."
echo ""

python3 edge_streamer.py
