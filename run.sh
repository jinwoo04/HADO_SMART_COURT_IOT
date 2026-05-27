#!/usr/bin/env bash
# HADO Smart Court 원클릭 실행 스크립트
#
# 사용법:
#   ./run.sh              # main 실행 (Level 1)
#   ./run.sh calibrate    # 캘리브레이션 도구
#   ./run.sh bench        # FPS 벤치마크
#   ./run.sh test         # 단위 테스트

set -e

# 스크립트 위치를 기준으로 절대 경로
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 가상환경 자동 활성화
if [ -d "../hado_venv" ]; then
    source ../hado_venv/bin/activate
elif [ -d "$HOME/hado_venv" ]; then
    source "$HOME/hado_venv/bin/activate"
fi

CMD="${1:-main}"

case "$CMD" in
    main)
        python -m src.main "${@:2}"
        ;;
    calibrate|cal)
        python -m src.calibrate "${@:2}"
        ;;
    bench|benchmark)
        python -m src.benchmark "${@:2}"
        ;;
    detect)
        python -m src.detector "${@:2}"
        ;;
    demo)
        python -m src.demo "${@:2}"
        ;;
    test)
        python -m tests.test_homography
        python -m tests.test_detector
        python -m tests.test_tracker
        python -m tests.test_camera
        python -m tests.test_visualizer
        python -m tests.test_tactic_engine
        python -m tests.test_guide
        python -m src.homography
        ;;
    *)
        echo "사용법: ./run.sh [main|calibrate|bench|detect|test|demo]"
        exit 1
        ;;
esac
