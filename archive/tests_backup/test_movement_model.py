"""MovementModel 단위 테스트."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from src.movement_model import MovementModel, MovementPrediction, _clip, _dist


# ---------- 픽스처 ----------

def _write_csv(path: Path, rows: list[dict]) -> None:
    header = ["pattern_id","step","player_id","team","role",
              "from_x","from_y","to_x","to_y","context","timestamp"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


@pytest.fixture()
def simple_csv(tmp_path):
    """팀A technician/attack 패턴 3개 + 팀B technician/defend 패턴 1개."""
    rows = [
        # 패턴1: 팀A technician attack — (0,0)→(2,1)→(4,0)
        dict(pattern_id=1,step=1,player_id=1,team="A",role="technician",
             from_x=0.0,from_y=0.0,to_x=2.0,to_y=1.0,context="attack",timestamp="t"),
        dict(pattern_id=1,step=2,player_id=1,team="A",role="technician",
             from_x=2.0,from_y=1.0,to_x=4.0,to_y=0.0,context="attack",timestamp="t"),
        # 패턴2: 팀A technician attack — (0,3)→(3,3)
        dict(pattern_id=2,step=1,player_id=1,team="A",role="technician",
             from_x=0.0,from_y=3.0,to_x=3.0,to_y=3.0,context="attack",timestamp="t"),
        # 패턴3: 팀A technician attack — (1,1)→(2,0)
        dict(pattern_id=3,step=1,player_id=1,team="A",role="technician",
             from_x=1.0,from_y=1.0,to_x=2.0,to_y=0.0,context="attack",timestamp="t"),
        # 패턴4: 팀B technician defend — (9,1)→(8,2)
        dict(pattern_id=4,step=1,player_id=4,team="B",role="technician",
             from_x=9.0,from_y=1.0,to_x=8.0,to_y=2.0,context="defend",timestamp="t"),
    ]
    p = tmp_path / "mv.csv"
    _write_csv(p, rows)
    return p


# ---------- 기본 동작 ----------

def test_load_pattern_count(simple_csv):
    m = MovementModel(simple_csv, k=3)
    assert m.pattern_count == 4


def test_predict_returns_prediction(simple_csv):
    m = MovementModel(simple_csv, k=3)
    pred = m.predict((0.5, 0.5), "technician", "attack", "A")
    assert isinstance(pred, MovementPrediction)


def test_predict_no_match_returns_none(simple_csv):
    m = MovementModel(simple_csv, k=3)
    pred = m.predict((0.0, 0.0), "defender", "attack", "A")
    assert pred is None


def test_predict_target_in_court(simple_csv):
    m = MovementModel(simple_csv, k=3)
    pred = m.predict((0.5, 0.5), "technician", "attack", "A")
    assert pred is not None
    x, y = pred.target_pos
    assert 0.0 <= x < 5.0, "팀A 목표는 x<5.0"
    assert 0.0 <= y <= 6.0


def test_predict_team_b_target_clipped(simple_csv):
    m = MovementModel(simple_csv, k=1)
    pred = m.predict((9.0, 1.0), "technician", "defend", "B")
    assert pred is not None
    x, _ = pred.target_pos
    assert x > 5.0, "팀B 목표는 x>5.0"


def test_confidence_high_when_close(simple_csv):
    """현재 위치가 패턴 시작점과 정확히 일치하면 신뢰도 높아야 함."""
    m = MovementModel(simple_csv, k=1)
    pred = m.predict((0.0, 0.0), "technician", "attack", "A")
    assert pred is not None
    assert pred.confidence > 0.9


def test_confidence_low_when_far(simple_csv):
    """현재 위치가 패턴들과 멀면 신뢰도 낮아야 함."""
    m = MovementModel(simple_csv, k=1)
    # 모든 패턴이 x<5에 있는데 x=4.9에서 예측
    pred = m.predict((4.9, 0.0), "technician", "attack", "A")
    assert pred is not None
    assert pred.confidence < 0.5


def test_full_path_not_empty(simple_csv):
    m = MovementModel(simple_csv, k=1)
    pred = m.predict((0.0, 0.0), "technician", "attack", "A")
    assert pred is not None
    assert len(pred.full_path) >= 1


def test_matched_pattern_id_valid(simple_csv):
    m = MovementModel(simple_csv, k=1)
    pred = m.predict((0.0, 0.0), "technician", "attack", "A")
    assert pred is not None
    assert pred.matched_pattern_id in {1, 2, 3}


# ---------- k 조정 ----------

def test_k_capped_to_available(simple_csv):
    """k=10으로 설정해도 후보가 3개면 오류 없이 동작."""
    m = MovementModel(simple_csv, k=10)
    pred = m.predict((0.0, 0.0), "technician", "attack", "A")
    assert pred is not None


# ---------- 헬퍼 ----------

def test_dist_zero():
    assert _dist((1.0, 2.0), (1.0, 2.0)) == pytest.approx(0.0)


def test_dist_basic():
    assert _dist((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)


def test_clip_team_a():
    x, y = _clip((6.0, 3.0), "A")
    assert x < 5.0


def test_clip_team_b():
    x, y = _clip((4.0, 3.0), "B")
    assert x > 5.0


def test_clip_y_bounds():
    _, y = _clip((2.0, -1.0), "A")
    assert y == pytest.approx(0.0)
    _, y2 = _clip((2.0, 7.0), "A")
    assert y2 == pytest.approx(6.0)


# ---------- 빈 CSV ----------

def test_empty_csv(tmp_path):
    p = tmp_path / "empty.csv"
    _write_csv(p, [])
    m = MovementModel(p)
    assert m.pattern_count == 0
    assert m.predict((1.0, 1.0), "technician", "attack", "A") is None


# ---------- coverage() 메서드 ----------

def test_coverage_returns_dict(simple_csv):
    """coverage()는 (role, context) → 패턴 수 dict를 반환한다."""
    m = MovementModel(simple_csv)
    cov = m.coverage()
    assert isinstance(cov, dict)
    assert cov[("technician", "attack")] == 3   # 패턴 1~3
    assert cov[("technician", "defend")] == 1   # 패턴 4


# ---------- intent 필터 ----------

def test_intent_filter_uses_matching(tmp_path):
    """intent가 일치하면 해당 패턴만 사용한다."""
    header = ["pattern_id","step","player_id","team","role",
              "from_x","from_y","to_x","to_y","context","timestamp","intent"]
    rows = [
        dict(pattern_id=1,step=1,player_id=1,team="A",role="technician",
             from_x=0.0,from_y=0.0,to_x=4.0,to_y=0.0,context="attack",timestamp="t",intent="gap_exploit"),
        dict(pattern_id=2,step=1,player_id=2,team="A",role="technician",
             from_x=0.0,from_y=0.0,to_x=0.5,to_y=0.0,context="attack",timestamp="t",intent="lure_attention"),
    ]
    p = tmp_path / "intent.csv"
    import csv as _csv
    with open(p, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    m = MovementModel(p)
    pred = m.predict((0.0, 0.0), "technician", "attack", "A", intent="gap_exploit")
    assert pred is not None
    assert pred.intent == "gap_exploit"


def test_intent_filter_fallback(tmp_path):
    """intent가 없는 패턴만 있으면 전체로 fallback해 None이 아니어야 한다."""
    header = ["pattern_id","step","player_id","team","role",
              "from_x","from_y","to_x","to_y","context","timestamp","intent"]
    rows = [
        dict(pattern_id=1,step=1,player_id=1,team="A",role="technician",
             from_x=0.0,from_y=0.0,to_x=3.0,to_y=0.0,context="attack",timestamp="t",intent="lure_attention"),
    ]
    p = tmp_path / "fallback.csv"
    import csv as _csv
    with open(p, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    m = MovementModel(p)
    # 존재하지 않는 intent → fallback → 전체 후보 사용
    pred = m.predict((0.0, 0.0), "technician", "attack", "A", intent="nonexistent_intent")
    assert pred is not None, "intent fallback 시 None이면 안 됨"


# ---------- 실제 CSV 연동 ----------

def test_real_csv_loads():
    """실제 data/movement_data.csv로 로드 및 예측 동작 확인."""
    real_csv = Path(__file__).resolve().parent.parent / "data" / "movement_data.csv"
    if not real_csv.exists():
        pytest.skip("movement_data.csv 없음")
    m = MovementModel(real_csv, k=3)
    assert m.pattern_count > 0
    pred = m.predict((1.0, 1.0), "technician", "attack", "A")
    assert pred is not None
    assert 0.0 <= pred.confidence <= 1.0
