"""yolov8n으로 frames_for_labeling 이미지를 자동 라벨링.

COCO 학습된 yolov8n.pt로 person(class 0)을 감지해
YOLO 포맷 .txt 라벨 파일을 생성한다.
생성된 이미지+라벨을 Roboflow에 "Import Annotations"로 올리면 된다.

실행:
    python -m src.auto_label
    python -m src.auto_label --conf 0.30 --img-dir data/frames_for_labeling
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def run(
    img_dir: str = "data/frames_for_labeling",
    model_path: str = "yolov8n.pt",
    conf: float = 0.30,
    imgsz: int = 640,
) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("pip install ultralytics")

    imgs = sorted(Path(img_dir).glob("*.jpg")) + sorted(Path(img_dir).glob("*.png"))
    if not imgs:
        print(f"[AutoLabel] 이미지 없음: {img_dir}")
        return

    # 이미 .txt가 있는 파일은 skip
    todo = [p for p in imgs if not p.with_suffix(".txt").exists()]
    already = len(imgs) - len(todo)
    print(f"[AutoLabel] 총 {len(imgs)}장 | 기존 라벨 {already}장 | 처리 대상 {len(todo)}장")
    if not todo:
        print("[AutoLabel] 모두 완료됨.")
        return

    model = YOLO(model_path)
    stats = {"labeled": 0, "empty": 0, "persons": 0}

    for i, img_path in enumerate(todo, 1):
        results = model(
            str(img_path),
            imgsz=imgsz,
            conf=conf,
            classes=[0],        # person only
            verbose=False,
        )
        boxes = results[0].boxes
        h, w = results[0].orig_shape

        lines = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # YOLO 포맷: class cx cy w h (정규화 0~1)
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            stats["persons"] += len(lines)
            stats["labeled"] += 1
        else:
            stats["empty"] += 1

        # .txt 저장 (감지 없으면 빈 파일)
        txt_path = img_path.with_suffix(".txt")
        txt_path.write_text("\n".join(lines))

        if i % 100 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}  labeled={stats['labeled']}  "
                  f"empty={stats['empty']}  persons={stats['persons']}")

    print(f"\n[AutoLabel] 완료")
    print(f"  라벨 있음: {stats['labeled']}장  ({stats['persons']}개 bbox)")
    print(f"  감지 없음: {stats['empty']}장  (heavy 프레임 위주 — Roboflow에서 수동 보완)")
    print(f"\n다음 단계:")
    print(f"  1. Roboflow 프로젝트 → Upload → 'Images and Annotations' 선택")
    print(f"  2. {img_dir}/ 폴더 전체 드래그앤드롭 (jpg + txt 같이)")
    print(f"  3. Format: YOLOv8 선택 → Upload")
    print(f"  4. 감지 안 된 heavy 프레임만 수동으로 보완")


def preview(
    img_dir: str = "data/frames_for_labeling",
    n: int = 5,
) -> None:
    """생성된 라벨을 이미지 위에 시각화해서 품질 확인."""
    labeled = [p for p in Path(img_dir).glob("*.jpg") if p.with_suffix(".txt").exists()]
    if not labeled:
        print("라벨 파일 없음")
        return

    import random
    samples = random.sample(labeled, min(n, len(labeled)))
    for img_path in samples:
        frame = cv2.imread(str(img_path))
        h, w = frame.shape[:2]
        txt = img_path.with_suffix(".txt").read_text().strip()
        n_boxes = 0
        for line in txt.splitlines():
            if not line.strip():
                continue
            _, cx, cy, bw, bh = map(float, line.split())
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            n_boxes += 1
        out = Path("data") / f"preview_{img_path.stem}.jpg"
        cv2.imwrite(str(out), frame)
        print(f"  {img_path.name}: {n_boxes}명 → {out.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="yolov8n 자동 라벨링")
    parser.add_argument("--img-dir", default="data/frames_for_labeling")
    parser.add_argument("--model",   default="yolov8n.pt")
    parser.add_argument("--conf",    type=float, default=0.30)
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--preview", action="store_true", help="라벨 품질 미리보기")
    args = parser.parse_args()

    if args.preview:
        preview(args.img_dir)
    else:
        run(args.img_dir, args.model, args.conf, args.imgsz)


if __name__ == "__main__":
    main()
