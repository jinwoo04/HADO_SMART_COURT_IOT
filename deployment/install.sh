#!/usr/bin/env bash
# Pi 자동 실행 셋업 — 한 번만 실행하면 부팅 시 HADO 시스템 자동 시작
#
# 사용법:
#   chmod +x deployment/install.sh
#   ./deployment/install.sh

set -e

SERVICE_NAME="hado-smart-court"
SERVICE_FILE="$(dirname "$0")/${SERVICE_NAME}.service"

echo "=========================================="
echo " HADO Smart Court — Pi 자동 실행 셋업"
echo "=========================================="

# 1. systemd 서비스 파일 복사
echo "[1/4] systemd 서비스 파일 등록 중..."
sudo cp "$SERVICE_FILE" /etc/systemd/system/${SERVICE_NAME}.service

# 2. 로그 디렉토리 생성
mkdir -p ~/hado-smart-court-iot/data/auto_recordings

# 3. systemd 재로드 + 부팅 시 자동 시작 등록
echo "[2/4] 부팅 시 자동 시작 등록..."
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}.service

# 4. 지금 바로 시작
echo "[3/4] 서비스 시작..."
sudo systemctl start ${SERVICE_NAME}.service

# 5. 상태 확인
echo "[4/4] 서비스 상태 확인..."
sleep 3
sudo systemctl status ${SERVICE_NAME}.service --no-pager || true

echo ""
echo "=========================================="
echo " 완료. 다음 명령어로 관리하세요:"
echo "=========================================="
echo "  sudo systemctl status   ${SERVICE_NAME}    # 상태 확인"
echo "  sudo systemctl stop     ${SERVICE_NAME}    # 잠시 멈춤"
echo "  sudo systemctl start    ${SERVICE_NAME}    # 다시 시작"
echo "  sudo systemctl restart  ${SERVICE_NAME}    # 재시작"
echo "  sudo systemctl disable  ${SERVICE_NAME}    # 자동 시작 해제"
echo "  journalctl -u ${SERVICE_NAME} -f               # 로그 실시간 보기"
echo ""
echo " 로그 위치: ~/hado-smart-court-iot/data/hado.log"
echo " 영상 위치: ~/hado-smart-court-iot/data/auto_recordings/"
