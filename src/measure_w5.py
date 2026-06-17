"""W5 Pi4 실측 헬퍼.

Pi4에서 실행하면 NCNN / ONNX / TTS 측정값을 자동 수집하고
data/w5_measurements.md 에 저장한다.

사용법
------
    python -m src.measure_w5                  # 라이브 카메라 200프레임
    python -m src.measure_w5 --video clip.mp4 # 비디오 파일
    python -m src.measure_w5 --frames 100     # 빠른 테스트
    python -m src.measure_w5 --skip-tts       # TTS 제외
"""
from __future__ import annotations

import argparse
import platform
import time
from datetime import datetime
from pathlib import Path


def _is_pi4() -> bool:
    """Raspberry Pi 4 여부 확인."""
    try:
        model = Path("/proc/device-tree/model").read_text()
        return "Raspberry Pi 4" in model
    except OSError:
        return False

# PROJECT_ROOT = src 상위 디렉터리
PROJECT_ROOT = Path(__file__).parent.parent


# ─────────────────────────────────────────────────────────
# 모델 우선순위 자동 탐지
# ─────────────────────────────────────────────────────────

def _find_models() -> dict[str, str | None]:
    """사용 가능한 pose 모델 경로 반환."""
    ncnn = PROJECT_ROOT / "yolov8n-pose_ncnn_model"
    onnx = PROJECT_ROOT / "yolov8n-pose.onnx"
    pt   = PROJECT_ROOT / "yolov8n-pose.pt"
    return {
        "NCNN": str(ncnn) if ncnn.exists() else None,
        "ONNX": str(onnx) if onnx.exists() else None,
        "PT":   str(pt)   if pt.exists()   else None,
    }


# ─────────────────────────────────────────────────────────
# TTS 지연 측정
# ─────────────────────────────────────────────────────────

def _measure_tts_latency(n: int = 5) -> float:
    """pyttsx3 한국어 TTS 발화 지연(ms) 중앙값. 실패 시 -1."""
    try:
        import pyttsx3  # type: ignore
    except ImportError:
        print("[W5] pyttsx3 없음 — TTS 측정 건너뜀")
        return -1.0

    if not _is_pi4() and platform.system() == "Darwin":
        print("[W5] ⚠ Mac 감지 — macOS NSSpeech는 runAndWait()가 즉시 반환하므로 TTS 지연 측정 건너뜀.")
        print("[W5]    Pi4(espeak-ng)에서 실측 필요.")
        return -1.0

    try:
        engine = pyttsx3.init()
        # 한국어 보이스 선택
        for v in engine.getProperty("voices"):
            if "ko" in v.id.lower():
                engine.setProperty("voice", v.id)
                break

        samples: list[float] = []
        for _ in range(n):
            t0 = time.perf_counter()
            engine.say("1번 측면 회피")
            engine.runAndWait()
            samples.append((time.perf_counter() - t0) * 1000)

        engine.stop()
        samples.sort()
        return samples[len(samples) // 2]  # 중앙값
    except Exception as e:
        print(f"[W5] TTS 측정 실패: {e}")
        return -1.0


# ─────────────────────────────────────────────────────────
# 마크다운 보고서 생성
# ─────────────────────────────────────────────────────────

def _format_report(results: list[dict], tts_ms: float, measured_at: str) -> str:
    is_pi = _is_pi4()
    platform_label = "Pi4 실측" if is_pi else f"참고값 ({platform.node()}, {platform.system()})"

    lines: list[str] = []
    lines.append(f"# W5 실측 결과 — {platform_label}\n")
    lines.append(f"**측정일**: {measured_at}  \n")
    lines.append(f"**목표**: FPS ≥15, RAM <1.5 GB, CPU <70°C, 위치오차 <10 cm\n")
    if not is_pi:
        lines.append(f"> ⚠️ Pi4가 아닌 환경에서 측정된 참고값입니다. 최종 보고서에는 Pi4 실측값을 사용하세요.\n")
    lines.append("")
    lines.append("## 추론 FPS / RAM / CPU 온도\n")
    lines.append("| 모델 | 평균 FPS | Wall FPS | P95 지연(ms) | 피크 RAM(MB) | 최고 온도(°C) |")
    lines.append("|------|---------|---------|------------|------------|------------|")

    for r in results:
        fps     = f"{r['inference_fps']:.1f}"
        wfps    = f"{r['wall_fps']:.1f}"
        p95     = f"{r['p95_latency_ms']:.1f}"
        ram     = f"{r['peak_ram_mb']:.0f}" if r['peak_ram_mb'] > 0 else "N/A"
        temp    = f"{r['peak_temp_c']:.1f}" if r['peak_temp_c'] > 0 else "N/A"
        flag    = " ✅" if r['inference_fps'] >= 15 else (" ⚠" if r['inference_fps'] >= 10 else " ❌")
        lines.append(f"| {r['model_label']} | {fps}{flag} | {wfps} | {p95} | {ram} | {temp} |")

    lines.append("")
    lines.append("## TTS 지연\n")
    if tts_ms > 0:
        flag = "✅" if tts_ms < 500 else "⚠"
        lines.append(f"| 지표 | 값 | 목표 |")
        lines.append(f"|------|-----|------|")
        lines.append(f"| 한국어 TTS 중앙값 지연 | {tts_ms:.0f} ms | <500 ms {flag} |")
    elif not is_pi:
        lines.append("*Mac 환경: TTS 측정 건너뜀 (Pi4/espeak-ng에서 측정 필요)*\n")
    else:
        lines.append("*pyttsx3 없음 또는 측정 실패*\n")

    lines.append("")
    lines.append("## QA_PREP.md 채우기 참조\n")
    lines.append("```")
    for r in results:
        lines.append(f"# Q5 — {r['model_label']}: {r['inference_fps']:.1f} fps")
    if tts_ms > 0:
        lines.append(f"# TTS 지연: {tts_ms:.0f} ms")
    lines.append("```")
    lines.append("")
    lines.append("## 위치 오차 (수동 측정)\n")
    lines.append("| 기준점 번호 | 측정 오차(cm) |")
    lines.append("|------------|--------------|")
    for i in range(1, 11):
        lines.append(f"| P{i} | [측정값 입력] |")
    lines.append("")
    lines.append("**RMS 오차**: `[계산값 입력]` cm  (목표 <10 cm)\n")
    lines.append("")
    lines.append("---")
    lines.append("*이 파일은 `python -m src.measure_w5` 에 의해 자동 생성되었습니다.*")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="W5 Pi4 실측 헬퍼")
    parser.add_argument("--video",     help="비디오 파일 (없으면 카메라 0)")
    parser.add_argument("--source",    default="0", help="카메라 인덱스 (--video 미지정 시)")
    parser.add_argument("--frames",    type=int, default=200, help="프레임 수 (기본 200)")
    parser.add_argument("--imgsz",     type=int, default=320)
    parser.add_argument("--warmup",    type=int, default=10)
    parser.add_argument("--skip-tts",  action="store_true", help="TTS 측정 건너뜀")
    parser.add_argument("--out",       default="data/w5_measurements.md")
    args = parser.parse_args()

    models = _find_models()
    available = {k: v for k, v in models.items() if v is not None}

    if not available:
        print("[W5] 경고: pose 모델 없음. 'yolov8n-pose.pt' 를 기본으로 사용합니다.")
        available = {"PT": "yolov8n-pose.pt"}

    is_pi = _is_pi4()
    env_label = "Raspberry Pi 4" if is_pi else f"{platform.system()} ({platform.node()})"
    print(f"\n{'='*60}")
    print(f" W5 실측 — {args.frames} 프레임")
    print(f" 실행 환경: {env_label}")
    if not is_pi:
        print(f" ⚠ Pi4 아님 — FPS/온도는 참고값. TTS는 Pi4에서 재측정 필요.")
    print(f" 발견된 모델: {list(available.keys())}")
    print(f"{'='*60}\n")

    # 소스 결정
    if args.video:
        source = args.video
    else:
        try:
            source = int(args.source)
        except ValueError:
            source = args.source

    # benchmark import
    from src.benchmark import benchmark

    results: list[dict] = []
    for label, model_path in available.items():
        print(f"\n[W5] ▶ {label} 벤치마크 시작: {model_path}")
        r = benchmark(
            source=source,
            n_frames=args.frames,
            imgsz=args.imgsz,
            model_path=model_path,
            warmup=args.warmup,
        )
        r["model_label"] = label
        results.append(r)

    # TTS 측정
    tts_ms = -1.0
    if not args.skip_tts:
        print(f"\n[W5] ▶ TTS 지연 측정 (5회 중앙값)...")
        tts_ms = _measure_tts_latency(n=5)
        if tts_ms > 0:
            flag = "✅" if tts_ms < 500 else "⚠"
            print(f"[W5] TTS 중앙값: {tts_ms:.0f} ms {flag}")

    # 보고서 생성
    measured_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    report = _format_report(results, tts_ms, measured_at)

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(f"\n{'='*60}")
    print(f" 보고서 저장: {out_path}")
    print(f"{'='*60}")
    print(report)


if __name__ == "__main__":
    main()
