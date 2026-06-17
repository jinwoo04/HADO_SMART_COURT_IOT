# W5 실측 결과

**최종 업데이트**: 2026-06-17  
**목표**: FPS ≥15, RAM <1.5 GB, CPU <70°C, TTS <500 ms, 위치오차 <10 cm

---

## A. Mac 참고 측정 (2026-06-17 — 사전 검증)

> ⚠️ 아래 수치는 Mac (Apple Silicon)에서 외부 카메라(1280×720 @30fps)로 측정한 참고값.  
> 최종 보고서/QA에는 **B섹션 Pi4 실측값**을 기입할 것.

### 추론 FPS / RAM (카메라 소스 1, 200 프레임, imgsz=320)

| 모델 | 평균 FPS | Wall FPS | P95 지연(ms) | 피크 RAM(MB) |
|------|---------|---------|------------|------------|
| NCNN | 98.5 | 29.3 | 19.0 | 396 |
| ONNX | 72.9 | 29.1 | 20.8 | 586 |
| PT   | 97.8 | 30.0 | 11.2 | 567 |

> Wall FPS ≈ 30 = 카메라 캡처 병목. 추론 FPS는 모델 순수 처리 속도.  
> RAM 모두 <1.5 GB 목표 ✅

### TTS 지연 (Mac — 참고 불가)

| 지표 | 값 | 비고 |
|------|-----|------|
| pyttsx3 중앙값 지연 | 1 ms | Mac NSSpeechSynthesizer 비동기 반환 — 실제 발화 완료 시간 아님 |

---

## B. Pi4 실측 [TODO] (현장 측정 필요)

> 실행: `./run.sh w5_measure` (Pi4에서)  
> 또는: `python -m src.measure_w5 --frames 200` → 결과를 아래에 기입

### 추론 FPS / RAM / CPU 온도 (Pi4)

| 모델 | 평균 FPS | Wall FPS | P95 지연(ms) | 피크 RAM(MB) | 최고 온도(°C) |
|------|---------|---------|------------|------------|------------|
| NCNN | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| ONNX | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

> Pi4에서 PT 모델은 너무 느릴 수 있음 — NCNN 우선, ONNX fallback.  
> 목표: FPS ≥15 ✅, RAM <1.5 GB ✅, CPU <70°C ✅

### TTS 지연 (Pi4 / espeak-ng)

| 지표 | 값 | 목표 |
|------|-----|------|
| 한국어 TTS 중앙값 지연 | [TODO] ms | <500 ms |

> Pi4 설치 확인: `sudo apt install espeak-ng`  
> 측정 명령: `python -m src.measure_w5 --source 0 --frames 50`

---

## C. 위치 오차 (수동 측정 필요)

> 기준점 P1–P10을 코트 위에 표시 후 `./run.sh measure-error` 실행

| 기준점 번호 | 측정 오차(cm) |
|------------|--------------|
| P1 | [TODO] |
| P2 | [TODO] |
| P3 | [TODO] |
| P4 | [TODO] |
| P5 | [TODO] |
| P6 | [TODO] |
| P7 | [TODO] |
| P8 | [TODO] |
| P9 | [TODO] |
| P10 | [TODO] |

**RMS 오차**: `[계산값 입력]` cm  (목표 <10 cm)

---

## D. QA_PREP.md / TECH_REPORT §5 기입 참조

Pi4 실측 완료 후 아래 슬롯 채울 것:

```
Q5  — NCNN FPS    : [TODO] fps  (Pi4 실측)
Q5  — ONNX FPS    : [TODO] fps  (Pi4 실측)
Q6  — 피크 RAM    : [TODO] MB
Q6  — CPU 온도    : [TODO] °C
Q9  — TTS 지연    : [TODO] ms
Q9  — 위치오차    : [TODO] cm  RMS
```

---

*Mac 사전 검증: `python -m src.measure_w5 --source 1 --frames 200` (2026-06-17)*  
*Pi4 실측: `./run.sh w5_measure` (Pi4 현장에서 실행)*
