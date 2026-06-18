"""IoUTracker 단위 테스트.

pytest 또는 단독 실행: python -m tests.test_tracker

커버리지:
  - _iou 계산 (완전 일치, 비겹침, 부분 겹침, 단일 픽셀)
  - Track.foot_point 프로퍼티
  - IoUTracker: 초기화, ID 부여, ID 유지, lost 처리,
    history 누적/max_history 제한, reset()
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detector import Detection
from src.tracker import IoUTracker, Track, _iou


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def _det(x1, y1, x2, y2, conf=0.9) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)


def _tracker(**kw) -> IoUTracker:
    defaults = dict(iou_threshold=0.3, max_lost_frames=3, max_history=10)
    defaults.update(kw)
    return IoUTracker(**defaults)


# ── _iou ──────────────────────────────────────────────────────────────────────

def test_iou_identical_boxes():
    box = np.array([0, 0, 100, 100], dtype=np.float32)
    assert abs(_iou(box, box) - 1.0) < 1e-6
    print("  ✓ _iou: 동일 박스 = 1.0")


def test_iou_no_overlap():
    a = np.array([0, 0, 50, 50], dtype=np.float32)
    b = np.array([60, 60, 110, 110], dtype=np.float32)
    assert _iou(a, b) == 0.0
    print("  ✓ _iou: 비겹침 = 0.0")


def test_iou_partial_overlap():
    """두 100×100 박스가 50×50만큼 겹치면 IoU = 2500/(10000+10000-2500) = 1/7."""
    a = np.array([0, 0, 100, 100], dtype=np.float32)
    b = np.array([50, 50, 150, 150], dtype=np.float32)
    expected = 2500 / (10000 + 10000 - 2500)
    assert abs(_iou(a, b) - expected) < 1e-6
    print(f"  ✓ _iou: 부분 겹침 ≈ {expected:.4f}")


def test_iou_contained_box():
    """작은 박스가 큰 박스 안에 완전히 포함 → IoU = small/big."""
    big = np.array([0, 0, 100, 100], dtype=np.float32)
    small = np.array([25, 25, 75, 75], dtype=np.float32)
    expected = 2500 / 10000
    assert abs(_iou(big, small) - expected) < 1e-6
    print(f"  ✓ _iou: 내포 박스 = {expected:.4f}")


def test_iou_symmetric():
    a = np.array([0, 0, 80, 80], dtype=np.float32)
    b = np.array([40, 40, 120, 120], dtype=np.float32)
    assert abs(_iou(a, b) - _iou(b, a)) < 1e-9
    print("  ✓ _iou: 대칭성")


def test_iou_touching_edge():
    """모서리만 닿는 박스 → 겹침 면적 0."""
    a = np.array([0, 0, 50, 50], dtype=np.float32)
    b = np.array([50, 50, 100, 100], dtype=np.float32)
    assert _iou(a, b) == 0.0
    print("  ✓ _iou: 모서리 접촉 = 0.0")


def test_iou_zero_area_box():
    """면적 0인 박스(점) → IoU 0."""
    point = np.array([50, 50, 50, 50], dtype=np.float32)
    box = np.array([0, 0, 100, 100], dtype=np.float32)
    assert _iou(point, box) == 0.0
    print("  ✓ _iou: 점(면적 0) 박스 = 0.0")


# ── Track 프로퍼티 ────────────────────────────────────────────────────────────

def test_track_foot_point_bottom_center():
    t = Track(track_id=1, bbox=np.array([100, 200, 200, 500], dtype=np.float32),
              confidence=0.9, age=1)
    fx, fy = t.foot_point
    assert fx == 150.0
    assert fy == 500.0
    print("  ✓ Track.foot_point: 하단 중앙 정확")


def test_track_foot_point_matches_detection():
    """Track.foot_point == Detection.foot_point (같은 bbox)."""
    d = _det(x1=200, y1=100, x2=400, y2=600)
    t = Track(track_id=1, bbox=d.bbox, confidence=d.confidence, age=1)
    assert t.foot_point == d.foot_point
    print("  ✓ Track.foot_point == Detection.foot_point")


# ── IoUTracker 초기화 ──────────────────────────────────────────────────────────

def test_tracker_initial_state():
    tr = _tracker()
    assert tr.tracks == []
    assert tr._next_id == 1
    print("  ✓ IoUTracker: 초기 상태 빈 트랙")


def test_tracker_update_empty_returns_empty():
    tr = _tracker()
    result = tr.update([])
    assert result == []
    print("  ✓ IoUTracker.update([]): 빈 리스트 반환")


# ── ID 부여 ───────────────────────────────────────────────────────────────────

def test_tracker_first_frame_assigns_ids():
    tr = _tracker()
    dets = [_det(0, 0, 50, 100), _det(200, 0, 250, 100)]
    tracks = tr.update(dets)
    assert len(tracks) == 2
    ids = {t.track_id for t in tracks}
    assert ids == {1, 2}
    print("  ✓ 첫 프레임 ID 1, 2 부여")


def test_tracker_ids_increment():
    tr = _tracker()
    tr.update([_det(0, 0, 50, 100)])
    tr.update([_det(5, 0, 55, 100)])      # 매칭됨
    tr.update([_det(5, 0, 55, 100), _det(200, 0, 250, 100)])  # 새 감지
    tracks = tr.update([_det(5, 0, 55, 100), _det(200, 0, 250, 100)])
    ids = {t.track_id for t in tracks}
    assert 1 in ids and 2 in ids
    print("  ✓ ID 순차 증가 확인")


# ── ID 유지 ───────────────────────────────────────────────────────────────────

def test_tracker_id_preserved_across_frames():
    """같은 위치 박스는 같은 ID를 유지해야 한다."""
    tr = _tracker()
    tr.update([_det(0, 0, 100, 200), _det(300, 0, 400, 200)])
    tracks = tr.update([_det(5, 5, 105, 205), _det(305, 5, 405, 205)])
    ids = [t.track_id for t in sorted(tracks, key=lambda t: t.bbox[0])]
    assert ids == [1, 2]
    print("  ✓ ID 유지: 살짝 이동해도 동일 ID")


def test_tracker_no_id_swap_on_close_approach():
    """두 선수가 교차하지 않는 한 ID 스왑 없음."""
    tr = _tracker()
    tr.update([_det(0, 0, 50, 100), _det(400, 0, 450, 100)])
    # 조금씩 이동
    for dx in range(1, 5):
        tracks = tr.update([_det(dx * 10, 0, dx * 10 + 50, 100),
                            _det(400 + dx * 5, 0, 450 + dx * 5, 100)])
    ids = sorted(t.track_id for t in tracks)
    assert ids == [1, 2]
    print("  ✓ ID 스왑 없음: 독립 이동")


# ── lost 처리 ─────────────────────────────────────────────────────────────────

def test_tracker_lost_frame_counter():
    """감지 사라지면 lost_frames 증가."""
    tr = _tracker(max_lost_frames=5)
    tr.update([_det(0, 0, 50, 100)])
    tr.update([])   # 감지 없음 → lost_frames=1
    assert tr.tracks[0].lost_frames == 1
    print("  ✓ lost_frames: 감지 없을 때 증가")


def test_tracker_lost_track_not_in_return():
    """lost 상태 트랙은 반환 리스트에서 제외."""
    tr = _tracker(max_lost_frames=5)
    tr.update([_det(0, 0, 50, 100)])
    result = tr.update([])
    assert result == []
    print("  ✓ lost 트랙 반환 제외")


def test_tracker_track_purged_after_max_lost():
    """max_lost_frames 초과 → 트랙 완전 삭제."""
    tr = _tracker(max_lost_frames=3)
    tr.update([_det(0, 0, 50, 100)])
    for _ in range(4):   # 4프레임 연속 미감지
        tr.update([])
    assert len(tr.tracks) == 0
    print("  ✓ max_lost_frames 초과 → 트랙 삭제")


def test_tracker_track_recovers_after_reappear():
    """lost 중에 다시 나타나면 새 ID로 재등록."""
    tr = _tracker(max_lost_frames=2)
    tr.update([_det(0, 0, 50, 100)])          # ID=1 등록
    for _ in range(3): tr.update([])           # ID=1 삭제
    tracks = tr.update([_det(0, 0, 50, 100)]) # 재등장 → ID=2
    assert len(tracks) == 1
    assert tracks[0].track_id == 2
    print("  ✓ 재등장 → 새 ID 부여")


# ── history 관리 ──────────────────────────────────────────────────────────────

def test_tracker_history_grows_with_frames():
    tr = _tracker(max_history=20)
    tr.update([_det(0, 0, 50, 100)])
    for i in range(4):
        tr.update([_det(i * 5, 0, i * 5 + 50, 100)])
    assert len(tr.tracks[0].history) == 5   # 첫 프레임 포함
    print("  ✓ history: 프레임마다 foot_point 누적")


def test_tracker_history_capped_at_max():
    """history는 max_history를 넘지 않는다."""
    tr = _tracker(max_history=5)
    tr.update([_det(0, 0, 50, 100)])
    for i in range(10):
        tr.update([_det(i, 0, i + 50, 100)])
    assert len(tr.tracks[0].history) <= 5
    print("  ✓ history: max_history 초과 않음")


def test_tracker_history_contains_foot_points():
    """history 원소가 Detection.foot_point와 일치."""
    tr = _tracker()
    d = _det(100, 200, 200, 500)
    tr.update([d])
    assert tr.tracks[0].history[0] == d.foot_point
    print("  ✓ history[0] == Detection.foot_point")


# ── reset ─────────────────────────────────────────────────────────────────────

def test_tracker_reset_clears_all():
    tr = _tracker()
    tr.update([_det(0, 0, 50, 100), _det(300, 0, 350, 100)])
    tr.reset()
    assert tr.tracks == []
    assert tr._next_id == 1
    print("  ✓ reset(): 트랙 초기화 + ID 리셋")


def test_tracker_after_reset_starts_from_id_1():
    tr = _tracker()
    tr.update([_det(0, 0, 50, 100)])
    tr.reset()
    tracks = tr.update([_det(0, 0, 50, 100)])
    assert tracks[0].track_id == 1
    print("  ✓ reset 후 재시작 ID=1")


# ── age 관리 ──────────────────────────────────────────────────────────────────

def test_track_age_increments():
    tr = _tracker()
    tr.update([_det(0, 0, 50, 100)])
    for i in range(1, 4):
        tr.update([_det(i, 0, i + 50, 100)])
    # 첫 등록(age=1) + 3번 매칭 = age=4
    assert tr.tracks[0].age == 4
    print("  ✓ age: 매칭 프레임마다 증가")


def test_track_confidence_updated():
    tr = _tracker()
    tr.update([_det(0, 0, 50, 100, conf=0.7)])
    tr.update([_det(5, 5, 55, 105, conf=0.95)])
    assert abs(tr.tracks[0].confidence - 0.95) < 1e-6
    print("  ✓ confidence: 매칭 시 최신값 갱신")


# ── 진입점 ────────────────────────────────────────────────────────────────────

def run_all():
    print("=" * 50)
    print(" Tracker 테스트")
    print("=" * 50)
    test_iou_identical_boxes()
    test_iou_no_overlap()
    test_iou_partial_overlap()
    test_iou_contained_box()
    test_iou_symmetric()
    test_iou_touching_edge()
    test_iou_zero_area_box()
    test_track_foot_point_bottom_center()
    test_track_foot_point_matches_detection()
    test_tracker_initial_state()
    test_tracker_update_empty_returns_empty()
    test_tracker_first_frame_assigns_ids()
    test_tracker_ids_increment()
    test_tracker_id_preserved_across_frames()
    test_tracker_no_id_swap_on_close_approach()
    test_tracker_lost_frame_counter()
    test_tracker_lost_track_not_in_return()
    test_tracker_track_purged_after_max_lost()
    test_tracker_track_recovers_after_reappear()
    test_tracker_history_grows_with_frames()
    test_tracker_history_capped_at_max()
    test_tracker_history_contains_foot_points()
    test_tracker_reset_clears_all()
    test_tracker_after_reset_starts_from_id_1()
    test_track_age_increments()
    test_track_confidence_updated()
    print("\n  모든 테스트 통과 ✓")


if __name__ == "__main__":
    run_all()
