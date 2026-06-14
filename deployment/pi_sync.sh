#!/usr/bin/env bash
# Pi4 코드 + 모델 동기화 스크립트
#
# 사용법 (Mac → Pi):
#   ./deployment/pi_sync.sh pi@192.168.x.x
#   ./deployment/pi_sync.sh pi@hado-pi.local   # mDNS 호스트명
#
# 하는 일:
#   1. Pi에서 git pull (src/ + movement_data.csv + docs/ 최신화)
#   2. NCNN 모델 scp (yolov8n-pose_ncnn_model/, yolov8n-pose.onnx)
#   3. Pi에서 preflight 자동 실행

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# ── 인수 확인 ──────────────────────────────────────────────────────────────────
PI_HOST="${1:-}"
if [ -z "$PI_HOST" ]; then
    echo "사용법: $0 <pi_user@pi_host>"
    echo "예시:   $0 pi@192.168.0.42"
    echo "예시:   $0 pi@hado-pi.local"
    exit 1
fi

REMOTE_DIR="~/hado-smart-court-iot"

echo "========================================================"
echo " HADO Pi 동기화: $PI_HOST"
echo "========================================================"

# ── Step 1: Pi에서 git pull ─────────────────────────────────────────────────
echo ""
echo "[1/3] Pi에서 git pull (presentation/demo-finalization 브랜치)"
ssh "$PI_HOST" "
    set -e
    cd $REMOTE_DIR
    git fetch origin
    git checkout presentation/demo-finalization 2>/dev/null || true
    git pull origin presentation/demo-finalization
    echo '  → git pull 완료'
"

# ── Step 2: 모델 파일 scp ──────────────────────────────────────────────────
echo ""
echo "[2/3] 모델 파일 전송 (NCNN 13MB + ONNX 12MB)"

NCNN_DIR="$PROJECT_ROOT/yolov8n-pose_ncnn_model"
ONNX_FILE="$PROJECT_ROOT/yolov8n-pose.onnx"

if [ -d "$NCNN_DIR" ]; then
    echo "  → NCNN 모델 전송 중..."
    scp -r "$NCNN_DIR" "$PI_HOST:$REMOTE_DIR/"
    echo "  → NCNN 완료"
else
    echo "  ⚠ NCNN 모델 없음 ($NCNN_DIR)"
fi

if [ -f "$ONNX_FILE" ]; then
    echo "  → ONNX 모델 전송 중..."
    scp "$ONNX_FILE" "$PI_HOST:$REMOTE_DIR/"
    echo "  → ONNX 완료"
else
    echo "  ⚠ ONNX 없음 ($ONNX_FILE)"
fi

# ── Step 3: Pi에서 preflight 실행 ──────────────────────────────────────────
echo ""
echo "[3/3] Pi preflight 점검"
ssh "$PI_HOST" "
    cd $REMOTE_DIR
    source ~/hado_venv/bin/activate 2>/dev/null || true
    ./run.sh preflight
"

echo ""
echo "========================================================"
echo " 동기화 완료! 내일 현장에서 아래 순서로 진행:"
echo "========================================================"
echo "  ssh $PI_HOST"
echo "  cd $REMOTE_DIR"
echo "  ./run.sh calibrate              # 코트 캘리브레이션"
echo "  ./run.sh main --level 2 --voice # Level 2 데모"
echo "  ./run.sh action_live            # 동작 인식 데모"
echo "  ./run.sh w5_measure             # W5 실측"
echo ""
echo "  자세한 절차: docs/W5_FIELD_TEST_GUIDE_2026_06_14.md"
