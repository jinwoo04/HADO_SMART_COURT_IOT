"""카메라 추상화.

Pi Camera (picamera2)와 일반 USB 웹캠(cv2.VideoCapture)을 동일한 인터페이스로 사용.
노트북에서도 동작하도록 자동 fallback.
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np


class Camera:
    """카메라 통합 인터페이스.

    사용 예시
    ---------
    >>> cam = Camera(width=1280, height=720, fps=30)
    >>> cam.open()
    >>> while True:
    ...     ok, frame = cam.read()
    ...     if not ok: break
    >>> cam.close()
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        prefer_picamera: bool = True,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.prefer_picamera = prefer_picamera

        self._backend: str = "none"
        self._cap: Optional[cv2.VideoCapture] = None
        self._picam = None  # picamera2 인스턴스

    # ---------- 라이프사이클 ----------
    def open(self) -> str:
        """카메라 오픈. picamera2 → cv2.VideoCapture 순서로 시도.

        Returns
        -------
        str : "picamera2" | "opencv" — 실제 사용된 백엔드
        """
        if self.prefer_picamera and self.source == 0:
            try:
                from picamera2 import Picamera2  # type: ignore
                self._picam = Picamera2()
                config = self._picam.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"}
                )
                self._picam.configure(config)
                self._picam.start()
                time.sleep(0.5)  # 노출 안정화
                self._backend = "picamera2"
                print(f"[Camera] picamera2 백엔드 사용 ({self.width}x{self.height} @{self.fps}fps)")
                return self._backend
            except (ImportError, Exception) as e:
                print(f"[Camera] picamera2 사용 불가 ({e.__class__.__name__}), OpenCV로 fallback")

        # OpenCV fallback (USB 웹캠, 노트북 내장)
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"카메라 열기 실패: source={self.source}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        print(f"[Camera] OpenCV 백엔드 사용 (요청 {self.width}x{self.height}, "
              f"실제 {actual_w}x{actual_h} @{actual_fps:.0f}fps)")
        self._backend = "opencv"
        return self._backend

    def close(self):
        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception:
                pass
            self._picam = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._backend = "none"

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ---------- 프레임 읽기 ----------
    def read(self) -> Tuple[bool, np.ndarray]:
        """단일 프레임 캡처. BGR ndarray 반환."""
        if self._backend == "picamera2":
            frame = self._picam.capture_array()
            # picamera2가 RGB로 반환 → OpenCV용 BGR로 변환
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame_bgr
        elif self._backend == "opencv":
            return self._cap.read()
        else:
            raise RuntimeError("카메라가 열려있지 않습니다. open()을 먼저 호출하세요.")

    @property
    def backend(self) -> str:
        return self._backend


def main():
    """단독 실행: 카메라 미리보기 + fps 측정."""
    import argparse
    parser = argparse.ArgumentParser(description="카메라 동작 확인")
    parser.add_argument("--source", default=0, help="카메라 소스 (정수 또는 비디오 파일 경로)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-display", action="store_true", help="헤드리스 모드 (fps만 측정)")
    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    with Camera(source=source, width=args.width, height=args.height) as cam:
        frame_count = 0
        t_start = time.time()
        t_last_print = t_start

        while True:
            ok, frame = cam.read()
            if not ok:
                print("[Camera] 프레임 읽기 실패")
                break
            frame_count += 1

            now = time.time()
            if now - t_last_print >= 2.0:
                fps = frame_count / (now - t_start)
                print(f"[Camera] {frame_count} frames, {fps:.1f} fps "
                      f"(shape={frame.shape}, backend={cam.backend})")
                t_last_print = now

            if not args.no_display:
                cv2.putText(frame, f"backend: {cam.backend}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Camera Test (ESC to exit)", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
