"""라벨링용 프레임 추출 스크립트.

영상에서 다양한 상황(클린/부분가림/강한가림)의 프레임을 샘플링해
data/frames_for_labeling/ 에 저장한다.

실행:
    python -m src.extract_frames
    python -m src.extract_frames --out data/frames_for_labeling --total 400
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np


def _frame_blur_score(frame: np.ndarray) -> float:
    """라플라시안 분산 — 낮을수록 흐릿한 프레임."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _ar_effect_ratio(frame: np.ndarray) -> float:
    """화면 내 AR 이펙트(강한 채도·밝기 픽셀) 비율 추정."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 채도 180+ AND 밝기 200+ → 파티클·이펙트 픽셀 특성
    mask = cv2.inRange(hsv, (0, 180, 200), (180, 255, 255))
    return float(mask.sum() / 255) / (frame.shape[0] * frame.shape[1])


def extract_from_video(
    video_path: Path,
    out_dir: Path,
    every_n: int,
    min_blur: float = 80.0,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [skip] 열기 실패: {video_path.name}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stem = video_path.stem.replace(" ", "_")
    saved = 0

    for fi in range(0, total_frames, every_n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue

        # 너무 흐린 프레임 skip
        if _frame_blur_score(frame) < min_blur:
            continue

        ar_ratio = _ar_effect_ratio(frame)
        # 카테고리: clean / partial / heavy
        if ar_ratio < 0.005:
            tag = "clean"
        elif ar_ratio < 0.03:
            tag = "partial"
        else:
            tag = "heavy"

        fname = out_dir / f"{stem}_{fi:05d}_{tag}.jpg"
        cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved += 1

    cap.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="라벨링용 프레임 추출")
    parser.add_argument("--out",   default="data/frames_for_labeling")
    parser.add_argument("--total", type=int, default=350,
                        help="목표 총 프레임 수 (자동으로 every_n 계산)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(Path("data").glob("*.mp4"))
    videos = [v for v in videos
              if not any(x in v.name for x in ["demo", "out", "preview"])]

    # 각 영상의 총 프레임 합산 → every_n 결정
    total_src = 0
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        total_src += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    every_n = max(1, total_src // args.total)
    print(f"영상 {len(videos)}개, 총 {total_src}프레임 → every_n={every_n} (목표 ~{args.total}장)")
    print(f"출력 디렉토리: {out_dir}\n")

    grand_total = 0
    counts: dict[str, int] = {"clean": 0, "partial": 0, "heavy": 0}

    for v in videos:
        n = extract_from_video(v, out_dir, every_n)
        grand_total += n
        print(f"  {v.name:45s}  {n}장")

    # 카테고리 집계
    for f in out_dir.iterdir():
        for tag in counts:
            if f"_{tag}.jpg" in f.name:
                counts[tag] += 1

    print(f"\n총 {grand_total}장 저장 → {out_dir}")
    print(f"  clean={counts['clean']}  partial={counts['partial']}  heavy={counts['heavy']}")
    print("\n다음 단계: Roboflow에 업로드 후 라벨링")


if __name__ == "__main__":
    main()
