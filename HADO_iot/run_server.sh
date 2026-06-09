#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════
# Launch script — MacBook backend server
# ════════════════════════════════════════════════════════════════

set -e

cd "$(dirname "$0")/server"

if [ ! -d node_modules ]; then
  echo "📦 Installing dependencies..."
  npm install
fi

echo ""
echo "🌐 Your MacBook IP address:"
ipconfig getifaddr en0 || echo "(en0 not found, try en1)"
echo ""
echo "👉 Set this IP in pi/edge_streamer.py → MACBOOK_SERVER_URL"
echo ""

node server.js
