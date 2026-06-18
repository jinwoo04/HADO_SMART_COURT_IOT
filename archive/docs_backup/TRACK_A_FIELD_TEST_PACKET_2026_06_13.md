# Track A Field Test Packet — 2026-06-13

이 문서는 내일 실제 코트에서 **Track A 발표용 시스템**을 바로 테스트하기 위한 실행 문서다.
Track A는 발표에서 말한 내용만 다룬다. Roboflow, AR effect occlusion, player-only relabel/retrain 루프는 Track B이므로 여기서는 제외한다.

## 1. 내일 테스트 목표

내일 목표는 "완벽한 제품 시연"이 아니라, 발표에서 말한 Track A 주장을 실제 수치와 영상으로 검증하는 것이다.

성공 기준:

| 항목 | 최소 성공 기준 | 발표에서 연결되는 문장 |
|---|---:|---|
| 카메라 입력 | Pi4 또는 Mac에서 라이브 프레임 수신 | fixed single camera |
| 모델 실행 | YOLOv8n-pose bbox + keypoints 출력 | bbox + 17 keypoints |
| 위치 변환 | homography 후 bird-eye 좌표 표시 | pixel to court meters |
| 전술 엔진 | Level 2 rule label / arrow / voice 중 하나 이상 출력 | tactical guidance |
| 동작 인식 | skeleton overlay + 7-action classifier 동작 | action recognition demo |
| 실측 기록 | FPS/RAM/temp/calibration/error 중 가능한 값 저장 | validation numbers |

## 2. 발표 내용과 구현 일치 확인

| 발표에서 말한 내용 | 현재 repo 구현/실행 경로 | 내일 확인할 증거 |
|---|---|---|
| 단일 카메라로 HADO 코트를 본다 | `src/camera.py`, `src/main.py`, `src/demo_pose.py` | 라이브 화면 또는 녹화 파일 |
| Raspberry Pi 4에서 edge inference를 목표로 한다 | `run.sh preflight`, `src/pi_preflight.py`, NCNN 자동 선택 | `./run.sh preflight` 결과 |
| YOLOv8n-pose로 bbox와 17 keypoints를 얻는다 | `src/detector.py`, `src/action_demo.py`, `src/demo_pose.py` | skeleton overlay 화면 |
| IoU tracker로 player ID를 유지한다 | `src/tracker.py`, `src/main.py` | bird-eye dot ID 유지 |
| homography로 pixel 좌표를 court meter로 바꾼다 | `src/homography.py`, `./run.sh calibrate` | calibration error, 좌표 CSV |
| rule-based tactical engine을 사용한다 | `src/tactic_engine.py`, `src/guide.py` | rule ID, urgency, voice guide |
| 7개 HADO action을 분류한다 | `src/pose.py`, `src/action_demo.py` | action panel 녹화 |
| 오프라인 IoT 시스템이다 | local model + local TTS + local recording | 네트워크 없이 실행 가능 여부 |
| 257 tests passing을 발표한다 | test suite 기준 문서화 완료 | 발표 전 `pytest` 가능하면 재확인 |

정리하면, 현재 만드는 내용은 발표와 일치한다. Track B에서 하던 AR effect가 겹친 영상의 Roboflow 재학습은 발표 범위 밖이다.

## 3. 테스트 전 준비물

하드웨어:

- Raspberry Pi 4
- Pi 전원 어댑터 또는 안정적인 전원
- USB camera 또는 GoPro/캡처 입력
- MacBook
- 코트 마커 4개 이상, 가능하면 위치 오차용 10점 마커
- Pi 냉각 팬

파일/모델:

- repo 최신 버전
- `yolov8n-pose_ncnn_model/` 또는 `yolov8n-pose.onnx`
- `requirements.txt` 설치된 Python 환경
- `config/court_config.yaml`
- 기존 calibration이 있더라도 현장에서 다시 `./run.sh calibrate` 권장

## 4. 내일 실행 순서

### Step 0 — 도착 직후 사전 점검

```bash
./run.sh preflight
```

확인할 것:

- Python dependencies OK
- model path OK
- camera read OK
- RAM/temp 확인
- TTS는 실패해도 치명적이지 않음. 음성 없이 시각화로 시연 가능하다.

### Step 1 — 코트 캘리브레이션

```bash
./run.sh calibrate
```

진행:

1. 코트 네 모서리 또는 기준점에 마커를 둔다.
2. 화면에서 좌상, 우상, 우하, 좌하 순서로 클릭한다.
3. `MEAN ROUND-TRIP ERROR`를 기록한다.

목표:

- 50 mm 이하: 그대로 진행
- 50 mm 초과: 카메라 고정/마커 위치/클릭 순서 확인 후 재실행

### Step 2 — Level 1/2 라이브 확인

```bash
./run.sh main --level 2 --voice
```

현장에서 볼 것:

- player bbox가 잡히는가
- bird-eye view에 위치가 표시되는가
- rule label 또는 arrow가 출력되는가
- `r` 키로 CSV 기록 시작/중지 가능
- `t` 키로 team assignment 초기화 가능

TTS가 불안정하면 아래처럼 음성 없이 진행한다.

```bash
./run.sh main --level 2
```

### Step 3 — 발표용 Level 2 녹화

```bash
./run.sh main --level 2 --voice --headless --record data/field_demo.mp4 --max-frames 2700
```

목표:

- 약 90초 녹화
- 최소 한 번 이상 rule output이 보이거나 기록된다
- 실패하면 `--voice`를 빼고 재시도

### Step 4 — skeleton + bird-eye demo 확인

```bash
./run.sh pose_live --out data/field_pose.mp4
```

목표:

- raw camera view 위에 skeleton이 표시된다
- bird-eye 위치가 함께 표시된다
- 이 영상은 발표 Slide 9 데모와 직접 연결된다

### Step 5 — action recognition demo 녹화

```bash
./run.sh action --threaded --record data/action_demo_field.mp4 --max-frames 1800
```

테스트할 7동작:

- ready
- charge
- shoot
- shield
- dodge left
- dodge right
- crouch

정확도가 완벽하지 않아도 괜찮다. 발표에서는 "prototype demo"로 말하고, 실패 케이스는 Q&A에서 single-camera pose ambiguity로 설명하면 된다.

### Step 6 — W5 성능 실측

```bash
./run.sh w5_measure
```

또는 안정적인 녹화 클립이 있으면:

```bash
./run.sh w5 --video data/field_demo.mp4
```

생성 파일:

```text
data/w5_measurements.md
```

이 파일의 값을 발표 대본과 Q&A의 TODO에 채운다.

### Step 7 — 위치 오차 RMS 측정

GUI가 가능하면:

```bash
./run.sh main --level 1
```

진행:

1. `r` 키로 CSV 기록 시작
2. 코트 위 10개 기준점에 차례대로 3~5초씩 선다
3. `r` 키로 기록 종료
4. 저장된 CSV 경로를 확인한다

예시:

```bash
./run.sh measure-error --csv data/position_logs/positions_XXXXXX.csv \
  --ref 0.0,0.0 5.0,0.0 10.0,0.0 0.0,3.0 5.0,3.0 10.0,3.0 \
        0.0,6.0 5.0,6.0 10.0,6.0 5.0,1.5
```

목표:

- RMS < 10 cm이면 발표 목표 달성
- 초과해도 실제 값으로 발표하고, calibration/camera angle 개선을 next step으로 말한다

## 5. 현장 기록표

| 항목 | 측정값 | 목표 | 파일/메모 |
|---|---:|---:|---|
| Calibration mean error | | < 50 mm | |
| NCNN average FPS | | >= 15 fps | `data/w5_measurements.md` |
| ONNX average FPS | | >= 6 fps | `data/w5_measurements.md` |
| Peak RAM | | < 1.5 GB | `data/w5_measurements.md` |
| CPU temp after run | | < 70 C | `data/w5_measurements.md` |
| TTS latency | | < 500 ms | optional |
| Position RMS error | | < 10 cm | `measure-error` output |
| Level 2 recording | yes/no | yes | `data/field_demo.mp4` |
| Pose recording | yes/no | yes | `data/field_pose.mp4` |
| Action recording | yes/no | yes | `data/action_demo_field.mp4` |

## 6. 문제가 생겼을 때 바로 바꿀 것

| 문제 | 즉시 대응 |
|---|---|
| 카메라가 안 열린다 | `ls /dev/video*` 확인 후 `--source 1`, `--source 2` 시도 |
| GUI가 안 뜬다 | `--headless --record ...`로 녹화 중심 진행 |
| TTS가 안 나온다 | `--voice` 제거. 발표에서는 offline TTS optional로 설명 |
| FPS가 낮다 | NCNN 모델 존재 확인, 가능하면 `imgsz 320` 유지, 배경 앱 종료 |
| 캘리브레이션 오차가 크다 | 카메라 고정 후 4점 다시 클릭 |
| tracker ID가 바뀐다 | 발표에서는 simple IoU tracker 한계로 설명, 핵심은 위치/전술 흐름 |
| action이 틀린다 | 한 명씩 천천히 동작. prototype classifier로 설명 |

## 7. 테스트 후 바로 업데이트할 문서

테스트가 끝나면 아래만 업데이트하면 발표와 문서가 정리된다.

1. `docs/FINAL_PRESENTATION_SCRIPT_EN.md`
   - Slide 4: actual Pi4 FPS
   - Slide 8: calibration/position RMS
   - Slide 10: TODO 숫자 제거
2. `docs/QA_PREP.md`
   - Q5: FPS/RAM
   - Q6: position accuracy
   - Q9: CPU temperature
3. `docs/TECH_REPORT_2026_06_09.md`
   - W5 실측 결과 표
4. 녹화 파일 보관:
   - `data/field_demo.mp4`
   - `data/field_pose.mp4`
   - `data/action_demo_field.mp4`
   - `data/w5_measurements.md`

## 8. 발표에서 그대로 말해도 되는 정리

영어 발표용 핵심 문장:

> "This Track A system is a presentation-level IoT prototype. It uses one fixed camera, YOLOv8n-pose, IoU tracking, homography, and a rule-based tactical engine to provide real-time HADO player tracking and tactical guidance. The on-court test verifies whether the Raspberry Pi 4 pipeline reaches the target FPS and whether court-coordinate error stays within tactical relevance."

한국어 이해용:

> Track A는 HADO 경기를 단일 카메라로 보고, 선수 위치와 스켈레톤을 잡아서, 버드아이뷰와 전술 추천을 실시간으로 보여주는 발표용 IoT 프로토타입이다. 내일 테스트는 이 설명을 실제 영상과 수치로 증명하는 과정이다.
