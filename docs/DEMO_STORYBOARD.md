# 데모 영상 스토리보드

> 6/22 제출용 short demo video (총 90~120초)
> 평가 기준: "Include a short demo video of your project (important)"

---

## 영상 전체 구조 (총 110초)

```
[0:00 – 0:10]  Intro 타이틀 + HADO 코트 소개
[0:10 – 0:25]  Hardware 셋업 (실물 카메라/Pi)
[0:25 – 0:40]  Calibration 4점 클릭 → JSON 저장
[0:40 – 1:10]  Level 1 — 2명 선수 추적 + Bird-eye view
[1:10 – 1:35]  Level 2 — 화살표 가이드 + 음성 안내
[1:35 – 1:50]  Outro — 시스템 사양 요약 + 끝
```

---

## SHOT-BY-SHOT 가이드

### S1 — Intro Title (0:00 – 0:10) · 10초

**화면**: 검정 배경에 큰 타이틀 텍스트 페이드인.

> **HADO Smart Court IoT**
> Real-time Player Tracking + Tactical Guidance
> Jinu Park · HUFS · Spring 2026

**자막 / 보이스오버**: 없음 (텍스트만, 8초간 정지 + 페이드)

**음악**: 잔잔한 일렉트로닉 (HADO 헤드셋의 신스 사운드 느낌 — Bensound free track 권장)

**촬영 팁**: After Effects나 Canva로 만들거나, PowerPoint에서 슬라이드 → mp4 변환.

---

### S2 — HADO 코트 소개 (0:00 – 0:10 중첩) · 인서트 컷

**화면**: HADO 코트 전경 (멀리서 와이드샷, 6m × 2.66m 라인이 보이는 각도)

**촬영**:
- 카메라: 스마트폰 와이드 렌즈, 1080p 60fps
- 위치: 코트 측면 상단 (스탠드)
- 시간: 조명이 균일한 낮 또는 형광등 켜진 시간

**자막** (영문):
> "HADO — AR-based physical e-sport, 6.0m × 2.66m court"

---

### S3 — Hardware 셋업 (0:10 – 0:25) · 15초

**화면 컷 시퀀스**:
1. **Pi 4 클로즈업** (3초): 가운데 LED 깜빡임 보이게
2. **Pi Camera 클로즈업** (3초): 렌즈 정면
3. **카메라 거치된 모습** (4초): 코트 측면 상단, 케이블이 깔끔하게 정리된 상태
4. **모니터에 시작 화면** (5초): 터미널 + `./run.sh main --level 2` 입력

**자막**:
> "Raspberry Pi 4 + Pi Camera v2 + Hailo AI Kit"
> "Total cost: under $200"

**촬영 팁**:
- 케이블이 어수선하면 점수 깎임. 케이블 타이로 정리.
- Pi 본체에 발열 케이스 + 팬 보이는 게 좋음 (전문성 어필).
- 모니터 화면 녹화는 OBS Studio로 별도 녹화 후 편집 시 합성.

---

### S4 — Calibration 4점 클릭 (0:25 – 0:40) · 15초

**화면**: 모니터 화면 녹화 (OBS or `ffmpeg`)

**시나리오**:
1. (3초) `python -m src.calibrate` 실행 → 라이브 카메라 뷰
2. (2초) SPACE 누르면 프레임 freeze
3. (8초) 좌상 → 우상 → 우하 → 좌하 순서로 마우스 클릭 (가속 편집)
4. (2초) ENTER → "MEAN ROUND-TRIP ERROR: X.X mm" 메시지

**자막**:
> "Step 1 — Mark 4 corners on the court"
> "Step 2 — Click 4 corners in clockwise order"
> "Step 3 — Calibration error reported in mm"

**촬영 팁**:
- 클릭하는 마우스 커서가 잘 보이도록 OBS에서 cursor highlight 설정.
- 결과 mm 숫자가 잘 보이도록 마지막 프레임 1초간 정지.

---

### S5 — Level 1 추적 (0:40 – 1:10) · 30초

**화면**: 풀스크린 분할 — 카메라 뷰(좌) + bird-eye view(우)

**시나리오**:
1. (10초) 본인 혼자 코트에 들어가 다양한 위치로 이동
   - 좌상 → 우하로 대각선 걷기
   - bird-eye view에서 궤적 fade-out 효과 강조
2. (10초) 둘째 선수 입장 (팀 동료 또는 가족)
   - 두 명이 동시에 움직이는 동안 ID #1, #2 안정적으로 유지
3. (10초) 일부러 카메라 앞에서 잠깐 가렸다가 (occlusion) 다시 나타남
   - 같은 ID로 복귀하는 모습 강조

**자막**:
> "Level 1 — YOLOv8n + IoU tracker → 1280×720 court overlay"
> "Persistent IDs across occlusion"
> "Running on Raspberry Pi 4 at [TODO: X] fps"

**촬영 팁**:
- HADO 헤드셋 착용 모습 보이면 가산점 (실제 사용 시나리오)
- HUD에 FPS 숫자 보이게
- 마지막에 `r` 키 눌러 CSV 녹화 시작하는 모습도 살짝 포함

---

### S6 — Level 2 전술 가이드 (1:10 – 1:35) · 25초

**화면**: 같은 분할 뷰, Level 2 활성화 상태

**시나리오**:
1. (8초) **R1 Spacing**: 두 명이 의도적으로 가깝게 붙음
   - 즉시 화살표 등장, 양 옆으로 분산하라는 가이드
   - 음성 안내 "1번 거리 확보" 들림
2. (8초) **R3 Counter**: 한 명이 다른 한 명 정면 가까이 접근
   - 빨간색(HIGH) 화살표 측면으로
   - 음성 "2번 측면 회피"
3. (9초) **R4 Gap Attack**: 두 팀 간 갭이 벌어진 순간 캡처
   - 노란 화살표가 갭 쪽으로 전진 지시
   - 음성 "3번 공격 전진"

**자막**:
> "Level 2 — Rule-based tactical engine"
> "6 rules: Spacing · Coverage · Lane Cover · Counter · Gap Attack · Backline"
> "Visual arrows + Korean TTS voice prompts"

**촬영 팁**:
- 음성이 잘 들리도록 환경음 적은 시간대 촬영
- 또는 음성을 별도로 녹음 후 편집 시 합성 (마이크 품질 차이 줄임)
- 화살표가 너무 짧게 사라지면 슬로우 모션(50%) 처리

---

### S7 — Outro (1:35 – 1:50) · 15초

**화면**: 검정 배경 + 결과 텍스트

> **Summary**
> · End-to-end fps on Pi 4: **[TODO: X]**
> · Mean position error: **[TODO: X] cm**
> · Tactical recommendation concordance: **[TODO: X]%**
> · Total deployment cost: **under $200**

**마지막 카드** (2초):
> "Thank you."
> Jinu Park · piaojinu@hufs.ac.kr

---

## 편집 가이드

### 추천 도구 (무료)

| 용도 | 도구 |
|------|------|
| 영상 편집 | DaVinci Resolve (Mac/Win/Linux) — 무료, 프로 기능 |
| 자막 | 위와 동일 |
| 화면 녹화 | OBS Studio (실시간 + 카메라 합성 가능) |
| 음성 합성 | 직접 녹음 또는 ElevenLabs/Naver Clova Voice 무료 plan |
| 음악 | bensound.com — 출처 표기 시 무료 |

### 편집 팁

1. **컷 사이 트랜지션**: cross-fade 300ms 정도, 화려한 트랜지션 금지
2. **자막 글꼴**: 영문 Calibri/Helvetica, 한글 Pretendard, **두꺼운 굵기**
3. **자막 위치**: 화면 하단 1/4 지점, 검정 반투명 배경
4. **속도 조정**:
   - 정적인 컷 (Hardware) → 평속
   - Calibration 클릭 시퀀스 → 1.5배속
   - Level 1/2 동작 → 평속 또는 0.75배속 (가이드 명확히)
5. **마지막 1초 freeze**: 시청자가 마지막 텍스트 읽을 시간 확보

### 출력 사양

- **해상도**: 1920×1080 (1080p)
- **프레임률**: 30fps (60fps는 파일 크기만 커짐)
- **코덱**: H.264, AAC 오디오
- **비트레이트**: 5~8 Mbps (총 파일 ~80MB 내외)
- **파일명**: `IoT_박진우_StudentID_Demo.mp4`

---

## 백업 영상 (집에서 사전 녹화)

> ⚠ **반드시 만들기**: 제출일 발표 중 하드웨어 실패 시 증빙용

**다른 점**: 풀버전 5분 영상으로 길게, 컷 없이 한 번에:
- 시작 시 시계 + 본인 얼굴 잠깐 노출 (편집 안 했다는 증빙)
- 전체 과정을 끊김 없이 한 번에 녹화
- 파일명: `IoT_박진우_StudentID_BackupRecording.mp4`

---

## 촬영 체크리스트 (6/17 ~ 6/19 사이 실행)

- [ ] 코트 예약 — 다른 선수 없는 빈 시간 확보
- [ ] 카메라 + Pi + 모니터 전원 시험 완료
- [ ] 4점 마커 부착 (절연 테이프 권장)
- [ ] 캘리브레이션 사전 실행 → 오차 < 10mm 확인
- [ ] 헤드셋 fine-tuned 모델 적용 상태 확인 (`models/yolov8n_hado.pt`)
- [ ] OBS Studio 설치 + 화면 녹화 테스트
- [ ] 외장 마이크 또는 조용한 환경
- [ ] 풀 백업 영상 1차 녹화 (5분, 컷 없이)
- [ ] 90초 편집본 1차 작업
- [ ] 두 영상 모두 USB에 사본 + 클라우드 백업

---

## 시간 분배 권장

| 작업 | 소요 시간 |
|------|----------|
| 촬영 셋업 + 리허설 | 1시간 |
| 본 촬영 (NG 포함) | 2시간 |
| 편집 1차 | 3시간 |
| 자막 + 음향 | 2시간 |
| 백업 영상 풀 녹화 | 30분 |
| **합계** | **약 8.5시간** |

— 끝 —
