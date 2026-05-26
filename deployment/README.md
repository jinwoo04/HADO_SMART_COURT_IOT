# 자동 실행 셋업 (Pi 부팅 시 자동 시작)

> Pi 전원 켜기 → HDMI 모니터에 HADO 시스템 자동 표시

## 한 번만 실행

```bash
cd ~/hado-smart-court-iot
chmod +x deployment/install.sh
./deployment/install.sh
```

끝. 이후 Pi를 재부팅(`sudo reboot`) 해보면 자동으로 시스템이 켜집니다.

## 관리 명령어

```bash
# 상태 확인 (running인지 dead인지)
sudo systemctl status hado-smart-court

# 잠시 멈추기 (개발할 때)
sudo systemctl stop hado-smart-court

# 다시 시작
sudo systemctl start hado-smart-court

# 코드 수정 후 재시작
sudo systemctl restart hado-smart-court

# 부팅 시 자동 시작 해제 (개발 모드로 전환)
sudo systemctl disable hado-smart-court

# 다시 등록
sudo systemctl enable hado-smart-court

# 로그 실시간 보기 (Ctrl+C로 종료)
journalctl -u hado-smart-court -f

# 또는 파일 로그
tail -f ~/hado-smart-court-iot/data/hado.log
```

## 시나리오별 셋팅 변경

### A. 모니터 연결됐을 때만 GUI 모드로

`hado-smart-court.service` 파일의 `ExecStart`에서 `--headless` 빼면 됩니다:

```ini
ExecStart=/home/pi/hado_venv/bin/python -m src.main --level 2 --voice
```

수정 후:
```bash
sudo systemctl daemon-reload
sudo systemctl restart hado-smart-court
```

### B. 매일 오후 6시에만 자동 시작

systemd service 대신 cron 사용 (둘 다 동시 사용 X):

```bash
# 자동 시작 해제 먼저
sudo systemctl disable hado-smart-court

# crontab 편집
crontab -e

# 다음 줄 추가 (매일 18:00 시작)
0 18 * * * /home/pi/hado_venv/bin/python -m /home/pi/hado-smart-court-iot/src/main --level 2 --headless --record /home/pi/hado-smart-court-iot/data/auto_recordings/$(date +\%Y\%m\%d_\%H\%M).mp4
```

### C. 게임 시작 버튼만 만들기 (자동 실행 X)

```bash
# 자동 시작 완전히 끄기
sudo systemctl disable hado-smart-court
sudo systemctl stop hado-smart-court

# 데스크탑 바탕화면에 .desktop 단축아이콘 만들기
cat > ~/Desktop/HADO-Start.desktop << 'EOF'
[Desktop Entry]
Name=HADO Smart Court
Comment=Start tracking system
Exec=/home/pi/hado-smart-court-iot/run.sh main --level 2 --voice
Icon=video-display
Terminal=true
Type=Application
EOF

chmod +x ~/Desktop/HADO-Start.desktop
```

이러면 바탕화면 아이콘 더블클릭으로 시작.

## 트러블슈팅

| 증상 | 해결 |
|------|------|
| `Active: failed (Result: exit-code)` | `journalctl -u hado-smart-court -n 50` 으로 마지막 로그 확인 |
| 카메라 못 찾음 | `ExecStartPre=/bin/sleep 5` 시간을 10초로 늘리기 |
| 권한 오류 | `.service` 파일의 `User=pi` 확인 |
| 음성 안 나옴 | systemd 환경엔 audio device가 없을 수 있음 → `--voice` 빼거나 PulseAudio user 설정 |
| 메모리 부족 종료 | `htop`으로 RAM 확인, imgsz=256으로 낮추기 |
