# HADO 헤드셋 YOLOv8 Fine-Tuning 가이드

> AR 헤드셋을 쓴 HADO 선수를 더 정확히 감지하기 위해 COCO 사전학습 YOLOv8n을 in-domain 데이터로 미세 조정하는 절차.

## 왜 필요한가

기본 YOLOv8n은 COCO 데이터셋(일반인 사진)으로 학습돼서:

- **AR 헤드셋이 얼굴을 가리면** confidence가 0.3 ~ 0.5까지 떨어짐 (정상은 0.7+)
- **빠른 동작 + 모션 블러**에 약함
- **위에서 비스듬히 내려다보는 카메라 각도**에 대한 학습 부족 — 일반 COCO는 눈높이 사진이 많음

300장 정도만 fine-tune해도 confidence가 평균 0.4 → 0.85까지 올라가는 게 일반적입니다. 작업 시간 총 2~3시간.

---

## 전체 흐름

```
[1] 데이터 캡처 (30분)
        ↓
[2] Roboflow 업로드 + 라벨링 (60~90분)
        ↓
[3] 데이터셋 export (5분)
        ↓
[4] Colab T4 GPU에서 학습 (30분)
        ↓
[5] Pi로 모델 복사 + 적용 (10분)
        ↓
[6] before/after 정확도 측정 (10분)
```

---

## STEP 1 — 데이터 캡처 (30분)

목표: **다양한 조건의 HADO 헤드셋 착용 이미지 300장**.

### 캡처 스크립트

프로젝트의 `src/camera.py`를 활용한 간단한 캡처 도구:

```bash
cd hado-smart-court-iot
source hado_venv/bin/activate
python -m src.camera --source 0 --width 1280 --height 720
```

키 단축키가 없으니 별도 스크립트가 편합니다. 다음 파일을 `scripts/capture_dataset.py`로 저장:

```python
"""HADO fine-tuning 데이터셋 캡처."""
import cv2
import time
from pathlib import Path
from src.camera import Camera

OUT_DIR = Path("data/raw_capture")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with Camera(source=0, width=1280, height=720) as cam:
    counter = len(list(OUT_DIR.glob("*.jpg")))
    print(f"시작 인덱스: {counter}. SPACE=캡처, ESC=종료")
    last_t = 0
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        cv2.putText(frame, f"saved: {counter}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32 and time.time() - last_t > 0.3:
            path = OUT_DIR / f"hado_{counter:04d}.jpg"
            cv2.imwrite(str(path), frame)
            counter += 1
            last_t = time.time()
            print(f"saved {path}")
cv2.destroyAllWindows()
```

### 캡처 다양성 체크리스트

균형 잡힌 데이터셋을 만들기 위해 다음을 골고루 포함:

- [ ] **선수 수**: 1명(50장) / 2명(100장) / 3-4명(150장)
- [ ] **위치**: 코너 / 중앙 / 라인 근처
- [ ] **자세**: 서있음 / 슈팅 동작 / 실드 / 회피 / 점프
- [ ] **조명**: 천장 형광등 / 사이드 조명 / 어두운 환경
- [ ] **헤드셋 종류**: 검정 본체 / 빛 반사가 다른 시간대
- [ ] **카메라 각도**: 측면 상단 (실제 설치 각도)
- [ ] **가림(occlusion)**: 두 선수 일부 겹침 30장 정도

캡처 후 폴더에 약 **300장**이 들어있어야 합니다.

---

## STEP 2 — Roboflow에서 라벨링 (60~90분)

### 2-1. 프로젝트 생성

1. [roboflow.com](https://roboflow.com) 가입 (무료 — Public workspace)
2. **Create New Project**
3. 설정:
   - Project Name: `hado-player-detection`
   - Project Type: **Object Detection**
   - License: CC BY 4.0 (편리)
   - Annotation Group: `player`

### 2-2. 이미지 업로드

`Upload` 탭 → 캡처한 300장 드래그앤드롭 → **Continue**.

`Train/Valid/Test` 분할 → **70/20/10**으로 자동 분배.

### 2-3. 라벨링

`Annotate` 탭에서 한 장씩 진행:

1. 키보드 단축키 **B**를 누르면 bounding box 모드
2. 선수의 **머리 정수리부터 발끝까지** 박스 그리기
3. 클래스 선택: `player` (사전에 정의)
4. **Spacebar** = 저장 + 다음 이미지

> ⚡ **속도 팁**: 한 장당 5~10초 목표. 300장이면 약 60분.

### 2-4. 라벨링 가이드라인

- **헤드셋 포함**: 박스 위쪽 경계는 머리 정수리(헤드셋 포함)까지
- **발 정확하게**: 시스템이 발 위치를 핵심으로 쓰므로 박스 아래 경계는 신발 끝과 정확히 일치
- **부분 가림**: 다른 선수에 가려 50% 이상 보이지 않는 경우는 라벨링 skip
- **모션 블러 심함**: 사람이라고 확신 못하는 정도면 skip
- **헬멧/모자만 보임**: 라벨링 skip (false positive 학습 방지)

### 2-5. 데이터셋 버전 생성

`Generate` 탭:

- **Preprocessing**: Auto-Orient ✓ / Resize: 640×640
- **Augmentations** (선택 권장):
  - Brightness: ±15%
  - Blur: 1.5px
  - Mosaic: 3x — 학습 다양성 증가
  - **No Flip** — 코트 방향성 있으면 좌우 반전은 의미 다름

**Create**.

---

## STEP 3 — 데이터셋 Export (5분)

Generate된 버전 페이지에서:

1. **Export Dataset** 클릭
2. Format: **YOLOv8** 선택
3. **Show download code** → Jupyter Notebook용 스니펫 복사

이런 식으로 나옵니다:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("your-workspace").project("hado-player-detection")
version = project.version(1)
dataset = version.download("yolov8")
```

이 스니펫을 다음 단계 Colab에 그대로 붙여넣을 겁니다.

---

## STEP 4 — Colab에서 학습 (30분)

### 4-1. Colab 노트북 열기

[colab.research.google.com](https://colab.research.google.com/) → New Notebook.

**Runtime → Change runtime type → T4 GPU** 선택.

### 4-2. 다음 셀들을 순서대로 실행

**셀 1 — 환경 설치**
```python
!pip install ultralytics roboflow -q
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

**셀 2 — Roboflow 데이터셋 다운로드** (Step 3에서 복사한 코드)
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("your-workspace").project("hado-player-detection")
version = project.version(1)
dataset = version.download("yolov8")
print("Downloaded to:", dataset.location)
```

**셀 3 — 학습 시작**
```python
from ultralytics import YOLO
import os

# 사전학습 yolov8n 로드
model = YOLO("yolov8n.pt")

# 학습
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,                 # 데이터 300장 기준 충분
    imgsz=640,
    batch=16,
    name="hado_v1",
    patience=10,               # early stopping
    optimizer="AdamW",
    lr0=0.001,
    cos_lr=True,
    augment=True,
    save_period=10,
)

# 결과 위치 출력
print(f"Best model: runs/detect/hado_v1/weights/best.pt")
```

T4 GPU 기준 약 **20~30분** 소요.

**셀 4 — 학습 결과 확인**
```python
import matplotlib.pyplot as plt
from PIL import Image

# Loss / mAP 그래프
Image.open("runs/detect/hado_v1/results.png")
```

기대 결과:
- `mAP50` (IoU 0.5 기준 정확도): **0.85 이상**
- `mAP50-95`: 0.55 이상

낮으면 데이터 추가 또는 epochs 늘리기.

**셀 5 — 추론 테스트**
```python
trained = YOLO("runs/detect/hado_v1/weights/best.pt")

# Roboflow validation set의 첫 이미지로 테스트
import os
val_dir = f"{dataset.location}/valid/images"
test_img = os.path.join(val_dir, os.listdir(val_dir)[0])

results = trained(test_img, conf=0.35)
results[0].show()
```

**셀 6 — 모델 다운로드**
```python
from google.colab import files
files.download("runs/detect/hado_v1/weights/best.pt")
```

다운로드된 `best.pt`를 노트북에 저장.

---

## STEP 5 — Pi에 적용 (10분)

### 5-1. Pi로 모델 복사

노트북에서 Pi로 SCP:

```bash
scp best.pt pi@hado-pi.local:~/hado-smart-court-iot/models/yolov8n_hado.pt
```

### 5-2. 설정 변경

`config/court_config.yaml` 편집:

```yaml
detector:
  model_path: "models/yolov8n_hado.pt"   # 변경
  imgsz: 320
  conf_threshold: 0.45                    # fine-tuned 모델은 더 높은 threshold OK
  iou_threshold: 0.5
  target_class: 0
```

> ⚠ Roboflow에서 export한 yolov8 형식은 첫 클래스 ID가 0(player)입니다. 다른 클래스가 있다면 `target_class`를 조정.

### 5-3. 동작 확인

```bash
python -m src.main --level 1
```

이전과 동일하게 동작하되, confidence 값이 평균 0.85+ 가 떠야 합니다.

---

## STEP 6 — Before / After 비교 측정 (10분)

발표 자료(보고서)용 정량 비교.

```bash
# Before — COCO 사전학습 모델
python -m src.benchmark --model yolov8n.pt --frames 200 --imgsz 320

# After — fine-tuned 모델
python -m src.benchmark --model models/yolov8n_hado.pt --frames 200 --imgsz 320
```

수동 정확도 비교 (테스트셋 50장):

```python
# scripts/compare_models.py
from ultralytics import YOLO
from pathlib import Path
import cv2

test_imgs = list(Path("data/test_set").glob("*.jpg"))
ground_truth_count = {p.name: int(p.stem.split("_")[-1]) for p in test_imgs}
# 파일명을 hado_NNNN_2.jpg 같이 끝에 사람 수를 적어두는 방식

for model_path in ["yolov8n.pt", "models/yolov8n_hado.pt"]:
    model = YOLO(model_path)
    tp, fp, fn = 0, 0, 0
    for img in test_imgs:
        res = model(str(img), classes=[0], conf=0.35, verbose=False)
        detected = len(res[0].boxes)
        gt = ground_truth_count[img.name]
        if detected == gt:
            tp += gt
        elif detected > gt:
            fp += detected - gt
            tp += gt
        else:
            tp += detected
            fn += gt - detected
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    print(f"{model_path}: precision={precision:.3f}, recall={recall:.3f}")
```

기대 개선폭:
- Precision: 0.78 → 0.94 정도
- Recall: 0.71 → 0.91 정도

---

## 트러블슈팅

| 증상 | 원인 / 해결 |
|------|-----------|
| Colab에서 학습 중 OOM | `batch=8` 로 낮추기 |
| mAP 0.5 미만 | 데이터 부족 — 100장 더 캡처 + epoch 80까지 |
| Pi에서 추론 느려짐 | fine-tuned도 yolov8n 기반이라 속도 영향 거의 없음. 만약 느리면 imgsz=256 |
| 모델이 빈 박스 감지 | conf_threshold를 0.5로 더 높이기 |
| 헤드셋 본체만 감지함 | 라벨링 시 머리부터 발끝까지로 다시 라벨링 |

---

## 보고서 활용 포인트

이 작업을 보고서 **Section 5.3 (Position Estimation Accuracy)** 또는 **Section 5.6 (Domain Adaptation)**으로 추가하면 평가 만점 요인입니다:

> "Recognizing that COCO-pretrained YOLOv8n suffers from reduced confidence on HADO 
> athletes wearing AR headsets (mean confidence dropped from 0.78 to 0.43 in our pilot 
> tests), we collected 300 in-domain images and fine-tuned the network for 50 epochs 
> on a Colab T4 GPU. The fine-tuned model achieved mAP50 of 0.92 (up from 0.71) and 
> improved end-to-end recall by 19 percentage points."

(실제 숫자로 채워 넣기)

---

## 시간/비용 정리

| 단계 | 시간 | 비용 |
|------|------|------|
| 캡처 | 30분 | 0원 |
| 라벨링 | 60~90분 | 0원 (Roboflow Free) |
| Colab 학습 | 20~30분 | 0원 (Free T4) |
| Pi 배포 | 10분 | 0원 |
| **합계** | **약 3시간** | **0원** |

— 끝 —
