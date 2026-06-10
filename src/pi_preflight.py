"""Pi4 현장 테스트 사전 점검 스크립트.

실행:
    python -m src.pi_preflight
    ./run.sh preflight

점검 항목:
  1. 의존성 — ultralytics, cv2, numpy, pyttsx3
  2. 모델 — NCNN/ONNX/PT 존재 여부
  3. espeak-ng — 한국어 TTS 가용성
  4. 캘리브레이션 — config/calibration.json 존재 + 파싱
  5. 카메라 — /dev/video* 접근 가능 여부
  6. 시스템 — RAM 여유 / CPU 온도 / DISPLAY 환경변수
  7. data/ 디렉터리 — 쓰기 권한
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── ANSI 색상 ──────────────────────────────────────────────────────────────────
_OK   = "\033[32m✅\033[0m"
_WARN = "\033[33m⚠ \033[0m"
_FAIL = "\033[31m❌\033[0m"


def _ok(msg: str)   -> None: print(f"  {_OK}  {msg}")
def _warn(msg: str) -> None: print(f"  {_WARN} {msg}")
def _fail(msg: str) -> None: print(f"  {_FAIL} {msg}")


# ── 점검 함수들 ────────────────────────────────────────────────────────────────

def check_python_deps() -> bool:
    print("\n[1] Python 의존성")
    ok = True
    for pkg, import_name in [
        ("ultralytics", "ultralytics"),
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("pyttsx3", "pyttsx3"),
        ("yaml", "yaml"),
    ]:
        try:
            __import__(import_name)
            _ok(pkg)
        except ImportError:
            _fail(f"{pkg} 없음 → pip install {pkg}")
            ok = False
    return ok


def check_models() -> bool:
    print("\n[2] 모델 파일")
    ncnn = PROJECT_ROOT / "yolov8n-pose_ncnn_model"
    onnx = PROJECT_ROOT / "yolov8n-pose.onnx"
    pt   = PROJECT_ROOT / "yolov8n-pose.pt"

    if ncnn.exists():
        _ok(f"NCNN 모델 (Pi4 최적): {ncnn.name}/")
        return True
    _warn("NCNN 없음 — Mac에서 export 후 scp 권장")

    if onnx.exists():
        _ok(f"ONNX 모델: {onnx.name}")
        return True
    _warn("ONNX 없음")

    if pt.exists():
        _warn(f"PT 모델만 있음 ({pt.name}) — Pi4에서 약 3 fps (목표 15 fps 미달)")
        return True

    _fail("모델 없음 — yolov8n-pose_ncnn_model/ 또는 yolov8n-pose.onnx 필요")
    return False


def check_tts() -> bool:
    print("\n[3] TTS (espeak-ng + 한국어 음성)")
    is_linux = platform.system() == "Linux"

    # espeak-ng 바이너리
    if is_linux:
        rc = subprocess.call(["which", "espeak-ng"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if rc != 0:
            _fail("espeak-ng 없음 → sudo apt install espeak-ng")
            return False
        _ok("espeak-ng 설치됨")

    # pyttsx3 + 한국어 보이스
    try:
        import pyttsx3  # type: ignore
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        ko_voices = [
            v for v in voices
            if "ko" in (v.id or "").lower() or "korean" in (v.name or "").lower()
        ]
        engine.stop()
        if ko_voices:
            _ok(f"한국어 음성 {len(ko_voices)}개: {ko_voices[0].id.split('.')[-1]}")
            return True
        else:
            _fail("한국어 음성 없음 — sudo apt install espeak-ng 후 재시도")
            return False
    except Exception as e:
        _warn(f"pyttsx3 초기화 실패 ({e}) — --voice 없이 실행 가능")
        return False


def check_calibration() -> bool:
    print("\n[4] 캘리브레이션")
    calib_path = PROJECT_ROOT / "config" / "calibration.json"
    if not calib_path.exists():
        _fail(f"calibration.json 없음 → ./run.sh calibrate 실행")
        return False
    try:
        from src.homography import Calibration
        calib = Calibration.from_json(calib_path)
        _ok(f"calibration.json — {calib.court_width_m}m × {calib.court_height_m}m")
        # 코트 크기 상식 검증
        if not (8.0 < calib.court_width_m < 12.0):
            _warn(f"court_width_m={calib.court_width_m} — 10m가 아님, 재보정 필요?")
        if not (4.0 < calib.court_height_m < 8.0):
            _warn(f"court_height_m={calib.court_height_m} — 6m가 아님, 재보정 필요?")
        return True
    except Exception as e:
        _fail(f"calibration.json 파싱 실패: {e}")
        return False


def check_camera() -> bool:
    print("\n[5] 카메라")
    import cv2
    found = []

    # Linux: /dev/video* 열거
    if platform.system() == "Linux":
        for i in range(4):
            dev = Path(f"/dev/video{i}")
            if dev.exists():
                found.append(i)
        if not found:
            _fail("/dev/video* 없음 — USB 카메라 연결 확인")
            return False

    # OpenCV로 실제 열기 테스트
    for idx in (found or [0]):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ok_flag, _ = cap.read()
            cap.release()
            if ok_flag:
                _ok(f"/dev/video{idx} — 프레임 읽기 정상")
                return True
            else:
                _warn(f"/dev/video{idx} 열렸으나 프레임 읽기 실패")
        else:
            _warn(f"/dev/video{idx} 열기 실패")

    _fail("사용 가능한 카메라 없음")
    return False


def check_system() -> bool:
    print("\n[6] 시스템 리소스")
    ok = True

    # RAM
    try:
        import psutil  # type: ignore
        ram = psutil.virtual_memory()
        avail_mb = ram.available / 1024 / 1024
        total_mb = ram.total / 1024 / 1024
        if avail_mb < 512:
            _fail(f"가용 RAM {avail_mb:.0f} MB / {total_mb:.0f} MB — 500 MB 이상 필요")
            ok = False
        else:
            _ok(f"RAM 가용: {avail_mb:.0f} MB / {total_mb:.0f} MB")
    except ImportError:
        _warn("psutil 없음 — RAM 확인 불가 (pip install psutil)")

    # CPU 온도
    temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
    if temp_path.exists():
        try:
            temp_c = int(temp_path.read_text()) / 1000
            if temp_c >= 70:
                _fail(f"CPU 온도 {temp_c:.1f}°C — 70°C 이상 (쿨링 필요)")
                ok = False
            elif temp_c >= 60:
                _warn(f"CPU 온도 {temp_c:.1f}°C — 60°C 이상 (주의)")
            else:
                _ok(f"CPU 온도 {temp_c:.1f}°C")
        except Exception:
            _warn("CPU 온도 읽기 실패")
    else:
        _ok("CPU 온도 — 비 Pi 환경 (건너뜀)")

    # DISPLAY 환경변수
    if platform.system() == "Linux":
        if os.environ.get("DISPLAY"):
            _ok(f"DISPLAY={os.environ['DISPLAY']}")
        else:
            _warn("DISPLAY 없음 → main 실행 시 headless 자동 전환됨 (정상)")

    return ok


def check_data_dirs() -> bool:
    print("\n[7] data/ 디렉터리 쓰기 권한")
    ok = True
    for rel in ("data/position_logs", "data/snapshots", "data"):
        d = PROJECT_ROOT / rel
        d.mkdir(parents=True, exist_ok=True)
        test_file = d / ".write_test"
        try:
            test_file.write_text("ok")
            test_file.unlink()
            _ok(f"{rel}/ — 쓰기 가능")
        except OSError as e:
            _fail(f"{rel}/ — 쓰기 실패: {e}")
            ok = False
    return ok


# ── 종합 요약 ──────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 55)
    print(" HADO Smart Court — Pi4 현장 테스트 사전 점검")
    print("=" * 55)

    results = {
        "Python 의존성":   check_python_deps(),
        "모델 파일":       check_models(),
        "TTS":             check_tts(),
        "캘리브레이션":    check_calibration(),
        "카메라":          check_camera(),
        "시스템 리소스":   check_system(),
        "data/ 쓰기권한":  check_data_dirs(),
    }

    print("\n" + "=" * 55)
    print(" 점검 결과 요약")
    print("=" * 55)
    failed = []
    warned = []
    for name, passed in results.items():
        if passed:
            print(f"  {_OK}  {name}")
        else:
            print(f"  {_FAIL} {name}")
            failed.append(name)

    if not failed:
        print("\n  ✅ 모든 항목 통과 — 현장 테스트 준비 완료!")
        print("  다음 실행:")
        print("    ./run.sh calibrate            # 캘리브레이션 (아직 안 했다면)")
        print("    ./run.sh w5_measure           # W5 실측 (FPS/RAM/TTS)")
        print("    ./run.sh main --level 2 --voice  # Level 2 라이브 데모")
        return 0
    else:
        print(f"\n  ❌ {len(failed)}개 항목 실패: {', '.join(failed)}")
        print("  위 메시지의 수정 방법을 따른 후 재실행하세요.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
