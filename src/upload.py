"""구글 드라이브 업로드 모듈 (rclone 래퍼).

사전 설정 (Pi에서 1회):
    sudo apt install rclone
    rclone config          # 브라우저로 구글 계정 로그인 → remote 이름: gdrive

사용:
    from src.upload import upload_match
    upload_match(match_dir)                       # 기본 remote
    upload_match(match_dir, remote="gdrive:HADO") # 커스텀 경로
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


DEFAULT_REMOTE = "gdrive:HADO/matches"


def is_rclone_available() -> bool:
    return shutil.which("rclone") is not None


def upload_match(
    match_dir: Path,
    remote: str = DEFAULT_REMOTE,
    dry_run: bool = False,
) -> bool:
    """match_dir를 구글 드라이브에 업로드.

    Parameters
    ----------
    match_dir : 업로드할 경기 폴더
    remote    : rclone remote 경로 (기본: gdrive:HADO/matches)
    dry_run   : True면 실제 업로드 안 하고 시뮬레이션만

    Returns
    -------
    bool : 성공 여부
    """
    if not is_rclone_available():
        print("[Upload] rclone 미설치 — 업로드 건너뜀")
        print("         설치: sudo apt install rclone && rclone config")
        return False

    if not match_dir.exists():
        print(f"[Upload] 폴더 없음: {match_dir}")
        return False

    dest = f"{remote}/{match_dir.name}"
    cmd = ["rclone", "copy", str(match_dir), dest, "--progress", "--stats-one-line"]
    if dry_run:
        cmd.append("--dry-run")

    print(f"[Upload] ▲ 업로드 시작: {match_dir.name} → {dest}")
    try:
        result = subprocess.run(cmd, timeout=300)
        if result.returncode == 0:
            print(f"[Upload] ✅ 완료: {dest}")
            return True
        print(f"[Upload] ❌ 실패 (code={result.returncode})")
        return False
    except subprocess.TimeoutExpired:
        print("[Upload] ❌ 타임아웃 (5분)")
        return False
    except Exception as e:
        print(f"[Upload] ❌ 오류: {e}")
        return False


def setup_guide() -> None:
    """rclone 설정 안내 출력."""
    print("""
[Upload] 구글 드라이브 연동 설정 방법:

  1. rclone 설치:
     sudo apt install rclone

  2. 구글 드라이브 remote 설정:
     rclone config
     → New remote → 이름: gdrive → Google Drive 선택
     → 브라우저 로그인 → Shared drive: No → 완료

  3. 연결 확인:
     rclone ls gdrive:

  4. HADO 폴더 생성 (선택):
     rclone mkdir gdrive:HADO/matches
""")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="경기 폴더 구글 드라이브 업로드")
    parser.add_argument("--match-dir", required=True, help="업로드할 match 폴더")
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--setup", action="store_true", help="설정 방법 안내")
    args = parser.parse_args()

    if args.setup:
        setup_guide()
        return

    upload_match(Path(args.match_dir), remote=args.remote, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
