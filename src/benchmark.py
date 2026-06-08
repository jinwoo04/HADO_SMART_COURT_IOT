"""FPS 벤치마크.

목적: Pi(또는 노트북)에서 YOLOv8 추론 속도를 측정.
판단 기준: 평균 fps가 15 이상이면 Level 1 사용 가능. 미만이면 최적화 필요.

사용법
------
    python -m src.benchmark --frames 200 --imgsz 320
    python -m src.benchmark --source test_video.mp4 --imgsz 416
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from statistics import mean, median, stdev

import cv2
import numpy as np

from src.camera import Camera
from src.detector import PersonDetector

try:
    import psutil as _psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


def _ram_mb() -> float:
    """현재 프로세스 RSS 메모리 (MB). psutil 없으면 0 반환."""
    if _PSUTIL:
        return _psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    return 0.0


def _cpu_temp_c() -> float:
    """Pi4 CPU 온도(°C). /sys/class/thermal 없으면 0 반환."""
    try:
        temp = Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()
        return int(temp) / 1000.0
    except (OSError, ValueError):
        return 0.0


def benchmark(
    source,
    n_frames: int = 200,
    imgsz: int = 320,
    model_path: str = "yolov8n.pt",
    warmup: int = 10,
) -> dict:
    print("=" * 60)
    print(f" YOLOv8 FPS 벤치마크")
    print(f"  - source: {source}")
    print(f"  - n_frames: {n_frames}")
    print(f"  - imgsz: {imgsz}")
    print(f"  - model: {model_path}")
    print("=" * 60)

    detector = PersonDetector(model_path=model_path, imgsz=imgsz)

    # 카메라 또는 비디오 파일
    is_camera = isinstance(source, int)
    if is_camera:
        cam = Camera(source=source)
        cam.open()
        read_fn = cam.read
    else:
        cap = cv2.VideoCapture(source)
        read_fn = cap.read

    latencies = []
    person_counts = []
    ram_samples: list[float] = []
    temp_samples: list[float] = []

    try:
        # 워밍업
        print(f"[Bench] 워밍업 {warmup} 프레임...")
        for _ in range(warmup):
            ok, frame = read_fn()
            if not ok:
                break
            _ = detector.detect(frame)

        # 본 측정
        print(f"[Bench] 측정 시작 ({n_frames} 프레임)...")
        t_start = time.time()
        for i in range(n_frames):
            ok, frame = read_fn()
            if not ok:
                print(f"  프레임 읽기 실패 @ {i}")
                break

            t0 = time.perf_counter()
            dets = detector.detect(frame)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000)  # ms
            person_counts.append(len(dets))
            ram_samples.append(_ram_mb())
            temp_samples.append(_cpu_temp_c())

            if (i + 1) % 50 == 0:
                avg = mean(latencies[-50:])
                ram_cur = ram_samples[-1]
                temp_cur = temp_samples[-1]
                temp_str = f", CPU {temp_cur:.1f}°C" if temp_cur > 0 else ""
                print(f"  [{i+1}/{n_frames}] {avg:.1f} ms ({1000/avg:.1f} fps)"
                      f"  RAM {ram_cur:.0f} MB{temp_str}")

        t_total = time.time() - t_start
    finally:
        if is_camera:
            cam.close()
        else:
            cap.release()

    # 결과 집계
    avg_lat = mean(latencies)
    med_lat = median(latencies)
    sd_lat = stdev(latencies) if len(latencies) > 1 else 0.0
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    overall_fps = len(latencies) / t_total

    print("\n" + "=" * 60)
    print(" 결과 요약")
    print("=" * 60)
    print(f"  프레임 수      : {len(latencies)}")
    print(f"  총 소요 시간   : {t_total:.2f} s")
    print(f"  평균 지연      : {avg_lat:.2f} ms")
    print(f"  중앙값 지연    : {med_lat:.2f} ms")
    print(f"  표준편차       : {sd_lat:.2f} ms")
    print(f"  P95 지연       : {p95:.2f} ms")
    print(f"  추론 처리 FPS  : {1000/avg_lat:.2f}")
    print(f"  Wall-clock FPS : {overall_fps:.2f}")
    print(f"  평균 감지 인원 : {mean(person_counts):.2f}")

    # RAM
    if ram_samples and max(ram_samples) > 0:
        print(f"  피크 RAM       : {max(ram_samples):.0f} MB")
        print(f"  평균 RAM       : {mean(ram_samples):.0f} MB")
    # CPU 온도 (Pi4)
    if temp_samples and max(temp_samples) > 0:
        print(f"  최고 CPU 온도  : {max(temp_samples):.1f} °C")
        print(f"  평균 CPU 온도  : {mean(temp_samples):.1f} °C")

    # 판정
    print("\n" + "-" * 60)
    if 1000 / avg_lat >= 15:
        print("  ✅ Level 1 사용 가능 (15+ fps)")
    elif 1000 / avg_lat >= 10:
        print("  ⚠ 경계선 — imgsz 낮추거나 AI Kit 추가 고려")
    else:
        print("  ❌ 너무 느림 — imgsz 256으로 낮추거나 모델 교체 필요")
    print("=" * 60)

    return {
        "n_frames": len(latencies),
        "avg_latency_ms": avg_lat,
        "median_latency_ms": med_lat,
        "p95_latency_ms": p95,
        "inference_fps": 1000 / avg_lat,
        "wall_fps": overall_fps,
        "peak_ram_mb": max(ram_samples) if ram_samples else 0.0,
        "peak_temp_c": max(temp_samples) if temp_samples else 0.0,
        "imgsz": imgsz,
        "model": model_path,
    }


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 FPS 벤치마크")
    parser.add_argument("--source", default="0", help="0 (카메라) 또는 비디오 파일")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    benchmark(source, n_frames=args.frames, imgsz=args.imgsz,
              model_path=args.model, warmup=args.warmup)


if __name__ == "__main__":
    main()
