"""IoU 기반 다중 객체 추적기.

ByteTrack/DeepSORT 같은 외부 의존성 없이 동작하는 가벼운 트래커.
HADO처럼 선수가 4명 이하이고 코트 안에서만 움직이는 환경에 적합.

알고리즘
--------
1. 매 프레임 새 감지(detections) ↔ 활성 트랙(tracks)의 IoU 행렬 계산
2. Greedy matching (가장 큰 IoU부터 매칭)
3. 매칭 안 된 감지 → 새 트랙 생성
4. 매칭 안 된 트랙 → "lost" 카운트 증가, threshold 넘으면 삭제
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from src.detector import Detection


@dataclass
class Track:
    """단일 추적 객체. 동일 선수의 시간적 연속성을 유지."""
    track_id: int
    bbox: np.ndarray              # [x1, y1, x2, y2]
    confidence: float
    age: int = 0                  # 살아있는 프레임 수
    lost_frames: int = 0          # 연속 감지 실패 프레임 수
    history: list = field(default_factory=list)  # 발 위치 궤적 (픽셀)

    @property
    def foot_point(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, self.bbox[3])


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """두 bbox의 IoU."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


class IoUTracker:
    """Greedy IoU matching 기반 트래커."""

    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 15,
                 max_history: int = 60):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.max_history = max_history
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: List[Detection]) -> List[Track]:
        """프레임 단위 업데이트. 활성 트랙 리스트 반환.

        반환되는 Track 객체는 매 프레임 새로 갱신된 bbox/history를 가집니다.
        """
        if not self.tracks:
            # 첫 프레임 — 모든 감지를 새 트랙으로
            for d in detections:
                self._spawn_track(d)
            return list(self.tracks)

        if not detections:
            # 감지 없음 — 모든 트랙 lost 증가
            for t in self.tracks:
                t.lost_frames += 1
            self._purge_lost()
            return [t for t in self.tracks if t.lost_frames == 0]

        # IoU 행렬 (tracks × detections)
        n_tracks = len(self.tracks)
        n_dets = len(detections)
        iou_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)
        for i, t in enumerate(self.tracks):
            for j, d in enumerate(detections):
                iou_matrix[i, j] = _iou(t.bbox, d.bbox)

        # Greedy 매칭
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        matches: list[tuple[int, int]] = []  # (track_idx, det_idx)

        # IoU 큰 것부터 정렬
        flat = [(iou_matrix[i, j], i, j) for i in range(n_tracks) for j in range(n_dets)]
        flat.sort(key=lambda x: -x[0])

        for iou_val, i, j in flat:
            if iou_val < self.iou_threshold:
                break
            if i in matched_tracks or j in matched_dets:
                continue
            matches.append((i, j))
            matched_tracks.add(i)
            matched_dets.add(j)

        # 매칭된 트랙 갱신
        for t_idx, d_idx in matches:
            t = self.tracks[t_idx]
            d = detections[d_idx]
            t.bbox = d.bbox
            t.confidence = d.confidence
            t.age += 1
            t.lost_frames = 0
            t.history.append(t.foot_point)
            if len(t.history) > self.max_history:
                t.history.pop(0)

        # 매칭 안 된 트랙 — lost
        for i in range(n_tracks):
            if i not in matched_tracks:
                self.tracks[i].lost_frames += 1

        # 매칭 안 된 감지 — 새 트랙
        for j in range(n_dets):
            if j not in matched_dets:
                self._spawn_track(detections[j])

        self._purge_lost()
        return [t for t in self.tracks if t.lost_frames == 0]

    def _spawn_track(self, d: Detection):
        track = Track(
            track_id=self._next_id,
            bbox=d.bbox,
            confidence=d.confidence,
            age=1,
            lost_frames=0,
            history=[d.foot_point],
        )
        self.tracks.append(track)
        self._next_id += 1

    def _purge_lost(self):
        self.tracks = [t for t in self.tracks if t.lost_frames <= self.max_lost_frames]

    def reset(self):
        self.tracks.clear()
        self._next_id = 1


def main():
    """단독 테스트: 더미 감지 sequence로 ID 유지 확인."""
    tracker = IoUTracker(iou_threshold=0.3, max_lost_frames=5)

    # 프레임 1: 두 선수 등장
    f1 = [
        Detection(100, 100, 200, 300, 0.9),
        Detection(400, 100, 500, 300, 0.85),
    ]
    tracks = tracker.update(f1)
    print(f"frame 1: {[(t.track_id, t.bbox.tolist()) for t in tracks]}")

    # 프레임 2: 두 선수가 살짝 움직임
    f2 = [
        Detection(110, 110, 210, 310, 0.92),
        Detection(395, 105, 495, 305, 0.88),
    ]
    tracks = tracker.update(f2)
    print(f"frame 2: {[(t.track_id, t.bbox.tolist()) for t in tracks]}")
    assert tracks[0].track_id == 1, "ID 유지 실패!"
    assert tracks[1].track_id == 2, "ID 유지 실패!"

    # 프레임 3: 한 명만 감지됨
    f3 = [Detection(120, 120, 220, 320, 0.91)]
    tracks = tracker.update(f3)
    print(f"frame 3: {[(t.track_id, t.bbox.tolist()) for t in tracks]}")

    # 프레임 4-9: 두번째 선수 lost 누적
    for _ in range(6):
        tracker.update(f3)
    print(f"전체 트랙 수: {len(tracker.tracks)}")

    print("[Tracker] 모든 단독 테스트 통과")


if __name__ == "__main__":
    main()
