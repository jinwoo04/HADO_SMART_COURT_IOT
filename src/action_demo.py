"""HADO 실시간 동작 인식 데모.

카메라(Pi USB 카메라 / 웹캠)에서 YOLOv8n-pose 키포인트를 추출하고
HADO 6동작을 실시간 분류해 화면에 표시한다.

동작 클래스:
  ⚡ 공격 발사   — 한 팔 이상 어깨 위로 들어올림
  🛡 쉴드 방어   — 양팔 넓게 벌려 올림
  ←  회피 좌     — 상체가 왼쪽으로 기울어짐
  →  회피 우     — 상체가 오른쪽으로 기울어짐
  ↓  슬라이딩   — 웅크린 자세
  ●  준비 자세   — 기본 자세

실행:
    python -m src.action_demo                        # 웹캠 (Mac 개발)
    python -m src.action_demo --source 0             # 카메라 인덱스 0
    python -m src.action_demo --record demo_act.mp4  # 저장
    python -m src.action_demo --headless --record demo_act.mp4  # Pi headless
"""
from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import cv2
import numpy as np

from src.camera import Camera
from src.detector import PersonDetector
from src.pose import (
    ACTION_COLOR, ACTION_EMOJI, ACTION_KO, ActionResult,
    classify_hado_action, draw_skeleton,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── 상수 ──────────────────────────────────────────────────────────────────────
_SMOOTH_N   = 6       # 최근 N 프레임 중 최다 동작을 확정 레이블로 사용
_FONT_SCALE = 1.8     # 대형 동작 레이블 폰트 크기
_BAR_H      = 8       # 점수 막대 높이
_HISTORY_N  = 30      # 히스토리 패널에 표시할 프레임 수


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────
def _put_kr(img: np.ndarray, text: str, xy: tuple[int, int],
            size: int, color: tuple[int, int, int]) -> None:
    try:
        from src.annotate import put_text_kr
        put_text_kr(img, text, xy, size, color)
    except Exception:
        cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                    size / 30, color, 2, cv2.LINE_AA)


def _largest_det(dets):
    """가장 큰 바운딩 박스(면적 기준) — 데모 1인 기준."""
    if not dets:
        return None
    return max(dets, key=lambda d: d.area)


# ── 오버레이 드로잉 ────────────────────────────────────────────────────────────
def _draw_action_panel(
    img: np.ndarray,
    result: ActionResult,
    smoothed: str,
    fps: float,
) -> None:
    """하단 동작 레이블 패널 (전체 너비)."""
    h, w = img.shape[:2]
    panel_h = 90
    y0 = h - panel_h

    # 패널 배경
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    color = ACTION_COLOR.get(smoothed, (180, 180, 180))
    emoji = ACTION_EMOJI.get(smoothed, "?")
    label = ACTION_KO.get(smoothed, smoothed)
    conf  = result.confidence

    # 동작 레이블 (대형)
    _put_kr(img, f"{emoji}  {label}", (20, y0 + 14), 28, color)

    # 신뢰도 바
    bar_w = int((w - 200) * min(1.0, conf))
    bar_y = y0 + 56
    cv2.rectangle(img, (20, bar_y), (w - 180, bar_y + _BAR_H), (50, 50, 50), -1)
    cv2.rectangle(img, (20, bar_y), (20 + bar_w, bar_y + _BAR_H), color, -1)
    cv2.putText(img, f"{conf:.0%}", (w - 170, bar_y + _BAR_H),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # FPS
    cv2.putText(img, f"FPS {fps:.1f}", (w - 110, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1, cv2.LINE_AA)


def _draw_score_bars(
    img: np.ndarray,
    scores: dict[str, float],
) -> None:
    """우측 점수 막대 — 각 동작별 현재 점수."""
    h, w = img.shape[:2]
    bar_max_w = 90
    x0 = w - bar_max_w - 10
    y0 = 10
    line_h = 22

    overlay = img.copy()
    cv2.rectangle(overlay, (x0 - 6, y0 - 4),
                  (w - 4, y0 + len(scores) * line_h + 4), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.70, img, 0.30, 0, img)

    for i, (action, score) in enumerate(scores.items()):
        if action == "ready":
            continue
        color = ACTION_COLOR.get(action, (140, 140, 140))
        label = ACTION_KO.get(action, action)
        y = y0 + i * line_h
        bw = int(bar_max_w * min(1.0, score))
        cv2.rectangle(img, (x0, y + 4), (x0 + bar_max_w, y + 14), (40, 40, 40), -1)
        if bw > 0:
            cv2.rectangle(img, (x0, y + 4), (x0 + bw, y + 14), color, -1)
        _put_kr(img, label, (x0 - 68, y + 2), 11, color)


def _draw_history(
    img: np.ndarray,
    history: "collections.deque[str]",
) -> None:
    """좌측 상단: 최근 동작 히스토리 타임라인."""
    if not history:
        return
    cell_w = max(4, min(20, (img.shape[1] // 4) // max(1, len(history))))
    x0, y0, bar_h = 10, 10, 16
    for i, act in enumerate(history):
        color = ACTION_COLOR.get(act, (80, 80, 80))
        cv2.rectangle(img, (x0 + i * cell_w, y0),
                      (x0 + (i + 1) * cell_w - 1, y0 + bar_h), color, -1)


# ── 메인 루프 ─────────────────────────────────────────────────────────────────
def run(args) -> int:
    # 모델 우선순위: NCNN(Pi4 최적) > ONNX > .pt
    ncnn_path = PROJECT_ROOT / "yolov8n-pose_ncnn_model"
    onnx_path = PROJECT_ROOT / "yolov8n-pose.onnx"
    if ncnn_path.exists() and not args.pt and not args.onnx:
        model_path = str(ncnn_path)
        print(f"[ActionDemo] NCNN 모델 사용 (Pi4 최적화): {model_path}")
    elif onnx_path.exists() and not args.pt:
        model_path = str(onnx_path)
        print(f"[ActionDemo] ONNX 모델 사용: {model_path}")
    else:
        model_path = "yolov8n-pose.pt"
        print(f"[ActionDemo] PyTorch 모델 사용: {model_path}")

    detector = PersonDetector(
        model_path=model_path,
        imgsz=args.imgsz,
        conf_threshold=args.conf,
        device=args.device,
    )

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    cam = Camera(
        source=source,
        width=args.width,
        height=args.height,
        fps=args.fps,
        threaded=args.threaded,
    )
    cam.open()

    writer = None
    if args.record:
        rec_path = Path(args.record)
        rec_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.headless:
        cv2.namedWindow("HADO Action Demo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("HADO Action Demo", args.width, args.height + 90)

    smooth_buf: collections.deque[str] = collections.deque(maxlen=_SMOOTH_N)
    history:    collections.deque[str] = collections.deque(maxlen=_HISTORY_N)
    smoothed    = "ready"
    fps         = 0.0
    fps_alpha   = 0.9
    last_t      = time.time()
    frame_idx   = 0
    last_result: ActionResult | None = None

    print("[ActionDemo] 실행 — 카메라 앞에서 동작을 취하세요. ESC로 종료.")

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("[ActionDemo] 프레임 읽기 실패")
                break

            dets   = detector.detect(frame)
            target = _largest_det(dets)

            if target is not None:
                draw_skeleton(frame, target)
                result = classify_hado_action(target, frame_center_x=args.width / 2)
                if result is not None:
                    last_result = result
                    smooth_buf.append(result.action)
                    # 최근 N프레임 최다 동작 확정
                    smoothed = max(set(smooth_buf), key=list(smooth_buf).count)
                    history.append(smoothed)

            # ── HUD 렌더링 ────────────────────────────────────────
            _draw_history(frame, history)
            if last_result is not None:
                _draw_score_bars(frame, last_result.scores)
                _draw_action_panel(frame, last_result, smoothed, fps)
            else:
                # 아무도 감지 안 된 경우
                h, w = frame.shape[:2]
                _put_kr(frame, "카메라 앞에 서주세요",
                        (w // 2 - 100, h // 2), 20, (180, 180, 180))

            # 감지된 인원 수
            cv2.putText(frame, f"감지: {len(dets)}명",
                        (10, frame.shape[0] - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1)

            # 녹화
            if writer is None and args.record:
                h_out, w_out = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.record, fourcc, 20.0, (w_out, h_out))
                print(f"[ActionDemo] 저장: {args.record}")
            if writer:
                writer.write(frame)

            if not args.headless:
                cv2.imshow("HADO Action Demo", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:   # ESC
                    break

            # FPS
            now   = time.time()
            inst  = 1.0 / max(0.001, now - last_t)
            fps   = fps_alpha * fps + (1 - fps_alpha) * inst if fps > 0 else inst
            last_t = now
            frame_idx += 1

            if args.max_frames and frame_idx >= args.max_frames:
                break

    except KeyboardInterrupt:
        print("\n[ActionDemo] 중단")
    finally:
        if writer:
            writer.release()
            print(f"[ActionDemo] 저장 완료: {args.record}")
        cam.close()
        cv2.destroyAllWindows()

    print(f"[ActionDemo] 종료 — {frame_idx}프레임, 평균 FPS {fps:.1f}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="HADO 실시간 동작 인식")
    parser.add_argument("--source",     default="0",   help="카메라 인덱스 또는 비디오 경로")
    parser.add_argument("--model",      default="",    help="사용 안 함 (자동 선택)")
    parser.add_argument("--pt",         action="store_true", help="ONNX 대신 .pt 강제 사용")
    parser.add_argument("--imgsz",      type=int,   default=320)
    parser.add_argument("--conf",       type=float, default=0.40)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--width",      type=int,   default=640)
    parser.add_argument("--height",     type=int,   default=480)
    parser.add_argument("--fps",        type=int,   default=30)
    parser.add_argument("--record",     default="",    help="출력 mp4 경로")
    parser.add_argument("--max-frames", type=int,   default=0)
    parser.add_argument("--headless",   action="store_true")
    parser.add_argument("--threaded",   action="store_true", help="스레드 캡처 (Pi4 FPS 향상)")
    parser.add_argument("--onnx",       action="store_true", help="NCNN 대신 ONNX 강제 사용")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
