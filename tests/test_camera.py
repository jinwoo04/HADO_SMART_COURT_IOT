"""Camera 클래스 단위 테스트.

pytest 또는 단독 실행: python -m tests.test_camera

실제 카메라 불필요. 합성 비디오 파일로 opencv 백엔드만 검증.
picamera2 경로는 Pi에서만 동작하므로 제외.
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.camera import Camera


# ── 합성 비디오 픽스처 ────────────────────────────────────────────────────────

def _make_video(n_frames: int = 10, w: int = 320, h: int = 240,
                fps: float = 30.0) -> str:
    """테스트용 임시 .avi 비디오 파일 생성 후 경로 반환."""
    path = tempfile.mktemp(suffix=".avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), i * 25 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


# ── 초기 상태 ─────────────────────────────────────────────────────────────────

def test_initial_backend_is_none():
    cam = Camera(source=0)
    assert cam.backend == "none"
    print("  ✓ 초기 backend == 'none'")


def test_initial_attributes_stored():
    cam = Camera(source=0, width=640, height=480, fps=15)
    assert cam.width == 640
    assert cam.height == 480
    assert cam.fps == 15
    print("  ✓ 생성자 파라미터 저장 확인")


def test_read_before_open_raises():
    cam = Camera(source=0)
    raised = False
    try:
        cam.read()
    except RuntimeError:
        raised = True
    assert raised, "open() 전 read() → RuntimeError 기대"
    print("  ✓ open() 전 read() → RuntimeError")


def test_close_before_open_no_crash():
    cam = Camera(source=0)
    cam.close()   # 열지 않아도 close()는 안전해야 함
    assert cam.backend == "none"
    print("  ✓ open() 전 close() → 크래시 없음")


# ── 파일 소스(opencv 백엔드) ──────────────────────────────────────────────────

def test_open_video_file_returns_opencv():
    path = _make_video(n_frames=5)
    cam = Camera(source=path, prefer_picamera=False)
    backend = cam.open()
    assert backend == "opencv"
    assert cam.backend == "opencv"
    cam.close()
    print("  ✓ 비디오 파일 → opencv 백엔드")


def test_read_returns_bgr_frame():
    path = _make_video(n_frames=5, w=320, h=240)
    cam = Camera(source=path, prefer_picamera=False)
    cam.open()
    ok, frame = cam.read()
    cam.close()
    assert ok is True
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert frame.shape[2] == 3   # BGR 3채널
    print(f"  ✓ read() → BGR 프레임 {frame.shape}")


def test_read_frame_shape_reasonable():
    path = _make_video(n_frames=5, w=320, h=240)
    cam = Camera(source=path, prefer_picamera=False)
    cam.open()
    ok, frame = cam.read()
    cam.close()
    h, w = frame.shape[:2]
    assert h > 0 and w > 0
    print(f"  ✓ 프레임 크기 양수: {w}x{h}")


def test_read_all_frames_until_eof():
    """영상 끝까지 읽으면 ok=False 반환."""
    n = 8
    path = _make_video(n_frames=n)
    cam = Camera(source=path, prefer_picamera=False)
    cam.open()
    count = 0
    while True:
        ok, _ = cam.read()
        if not ok:
            break
        count += 1
    cam.close()
    assert count == n
    print(f"  ✓ EOF까지 {count}프레임 읽기")


def test_close_resets_backend():
    path = _make_video()
    cam = Camera(source=path, prefer_picamera=False)
    cam.open()
    assert cam.backend == "opencv"
    cam.close()
    assert cam.backend == "none"
    print("  ✓ close() 후 backend == 'none'")


def test_close_idempotent():
    """close()를 여러 번 호출해도 안전."""
    path = _make_video()
    cam = Camera(source=path, prefer_picamera=False)
    cam.open()
    cam.close()
    cam.close()
    assert cam.backend == "none"
    print("  ✓ close() 멱등성")


# ── 컨텍스트 매니저 ───────────────────────────────────────────────────────────

def test_context_manager_opens_and_closes():
    path = _make_video(n_frames=3)
    with Camera(source=path, prefer_picamera=False) as cam:
        assert cam.backend == "opencv"
        ok, frame = cam.read()
        assert ok and frame is not None
    assert cam.backend == "none"
    print("  ✓ with Camera(...): 자동 open/close")


def test_context_manager_closes_on_exception():
    """예외 발생 시에도 close() 호출 보장."""
    path = _make_video(n_frames=3)
    cam_ref = None
    try:
        with Camera(source=path, prefer_picamera=False) as cam:
            cam_ref = cam
            raise ValueError("테스트 예외")
    except ValueError:
        pass
    assert cam_ref is not None
    assert cam_ref.backend == "none"
    print("  ✓ 예외 시 컨텍스트 매니저 close() 보장")


def test_open_nonexistent_file_raises():
    cam = Camera(source="/tmp/nonexistent_file_12345.mp4", prefer_picamera=False)
    raised = False
    try:
        cam.open()
    except RuntimeError:
        raised = True
    finally:
        cam.close()
    assert raised, "존재하지 않는 파일 → RuntimeError 기대"
    print("  ✓ 존재하지 않는 파일 → RuntimeError")


# ── 진입점 ────────────────────────────────────────────────────────────────────

def run_all():
    print("=" * 50)
    print(" Camera 테스트")
    print("=" * 50)
    test_initial_backend_is_none()
    test_initial_attributes_stored()
    test_read_before_open_raises()
    test_close_before_open_no_crash()
    test_open_video_file_returns_opencv()
    test_read_returns_bgr_frame()
    test_read_frame_shape_reasonable()
    test_read_all_frames_until_eof()
    test_close_resets_backend()
    test_close_idempotent()
    test_context_manager_opens_and_closes()
    test_context_manager_closes_on_exception()
    test_open_nonexistent_file_raises()
    print("\n  모든 테스트 통과 ✓")


if __name__ == "__main__":
    run_all()
