"""경기 단위 녹화 모듈.

MP4 영상 + positions.csv를 동시에 기록하고 종료 시 match_dir 경로를 반환한다.
analyzer.py가 이 폴더를 읽어 통계/히트맵을 생성한다.

실행:
    python -m src.main --match      # --match 플래그로 자동 활성화
"""
from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from src.homography import Calibration, pixel_to_court


class MatchRecorder:
    """경기 단위 녹화 관리자.

    사용 예시
    ---------
    recorder = MatchRecorder(base_dir, calib)
    recorder.start(frame.shape)
    for frame in ...:
        recorder.write(combined_frame, tracks, advices)
    match_dir = recorder.stop()
    """

    def __init__(self, base_dir: Path, calib: Calibration, fps: float = 20.0):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.match_dir = base_dir / ts
        self.match_dir.mkdir(parents=True, exist_ok=True)
        self.calib = calib
        self.fps = fps

        self._video_writer: Optional[cv2.VideoWriter] = None
        self._csv_file = None
        self._csv_writer = None
        self._frame_count: int = 0
        self._start_time: float = 0.0
        self._last_birdeye: Optional[np.ndarray] = None

    def start(self, frame_shape: tuple) -> None:
        """녹화 시작. frame_shape = combined 프레임의 (h, w) 또는 (h, w, c)."""
        h, w = frame_shape[:2]
        video_path = self.match_dir / "raw_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (w, h))

        csv_path = self.match_dir / "positions.csv"
        self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "timestamp", "frame_idx", "track_id",
            "x_m", "y_m",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "confidence",
        ])
        self._start_time = time.time()
        print(f"[Recorder] ● 녹화 시작 → {self.match_dir}")

    def write(
        self,
        combined_frame: np.ndarray,
        tracks: list,
        birdeye: Optional[np.ndarray] = None,
    ) -> None:
        """프레임 + 위치 데이터 기록."""
        if self._video_writer is not None:
            self._video_writer.write(combined_frame)

        if birdeye is not None:
            self._last_birdeye = birdeye.copy()

        if self._csv_writer is not None and tracks:
            ts = time.time() - self._start_time
            foot_px = np.array([t.foot_point for t in tracks], dtype=np.float32)
            foot_m = pixel_to_court(foot_px, self.calib)
            for t, m in zip(tracks, foot_m):
                self._csv_writer.writerow([
                    f"{ts:.3f}", self._frame_count, t.track_id,
                    f"{m[0]:.3f}", f"{m[1]:.3f}",
                    f"{t.bbox[0]:.1f}", f"{t.bbox[1]:.1f}",
                    f"{t.bbox[2]:.1f}", f"{t.bbox[3]:.1f}",
                    f"{t.confidence:.3f}",
                ])

        self._frame_count += 1

    def stop(self) -> Path:
        """녹화 종료. 마지막 버드아이뷰 저장 후 match_dir 반환."""
        if self._video_writer:
            self._video_writer.release()
        if self._csv_file:
            self._csv_file.close()

        if self._last_birdeye is not None:
            snap_path = self.match_dir / "birdeye_final.png"
            cv2.imwrite(str(snap_path), self._last_birdeye)

        duration = time.time() - self._start_time if self._start_time else 0.0
        print(f"[Recorder] ■ 종료 — {self._frame_count}프레임, {duration:.1f}초 → {self.match_dir}")
        return self.match_dir
