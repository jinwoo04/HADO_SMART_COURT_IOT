"""카메라/캘리브레이션 없이 버드아이뷰 파이프라인을 시연하는 데모.

calibration.json, YOLOv8, 실제 카메라 없이도 동작.
패턴 데이터(movement_data.csv)를 기반으로 선수가 실제 수집된 경로로 움직임.

실행:
    python -m src.demo                  # 창 표시 + data/demo.mp4 저장
    python -m src.demo --headless       # 창 없이 video만 저장
    python -m src.demo --frames 600     # 20초 미리보기
    ./run.sh demo
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.detector import Detection
from src.guide import draw_guide_on_birdeye
from src.homography import compute_homography, court_to_pixel
from src.movement_model import MovementModel
from src.tactic_engine import PlayerState, TacticEngine
from src.tracker import IoUTracker
from src.visualizer import (
    combine_views,
    draw_detections_on_frame,
    draw_hud,
    draw_players_on_birdeye,
    render_court_birdeye,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_COURT_W = 10.0
_COURT_H = 6.0
_FRAME_W, _FRAME_H = 1280, 720
_CSV_PATH = PROJECT_ROOT / "data" / "movement_data.csv"
_CONFIG_PATH = PROJECT_ROOT / "config" / "court_config.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------- 의도별 표시 상수 ----------
_INTENT_KO = {
    "direct_attack":         "직접공격",
    "feint_attack":          "페인트공격",
    "cross_court":           "코트가로지르기",
    "gap_exploit":           "공간공략",
    "lure_attention":        "시선유도",
    "create_space":          "공간창출",
    "bait_inward":           "안쪽유도",
    "support_fire":          "공격지원",
    "shield_protect":        "수비쉴드",
    "shield_attack_support": "공격지원쉴드",
    "shield_feint":          "쉴드페인트",
    "counter_shield":        "맞쉴드",
}
_ROLE_KO = {
    "main_attacker": "어태커",
    "technician":    "테크니션",
    "defender":      "디펜더",
}
# intent별 스텝 소요 프레임 (빠른 공격 ↔ 신중한 수비)
_FRAMES_PER_INTENT: dict[str, int] = {
    "direct_attack":         32,
    "feint_attack":          38,
    "cross_court":           55,
    "gap_exploit":           36,
    "lure_attention":        48,
    "create_space":          50,
    "bait_inward":           45,
    "support_fire":          42,
    "shield_protect":        52,
    "shield_attack_support": 44,
    "shield_feint":          40,
    "counter_shield":        38,
}
_DEFAULT_FPT = 44  # frames per step (intent 없을 때)

_PLAYER_ROLES = {
    1: "technician",    2: "defender",      3: "main_attacker",
    4: "main_attacker", 5: "defender",      6: "technician",
}


# ---------- 패턴 라이브러리 로드 ----------
def _load_pattern_library() -> dict[str, list[list[dict]]]:
    """CSV → role별 패턴 딕셔너리 {role: [[step_dict, ...], ...]}."""
    if not _CSV_PATH.exists():
        return {}
    rows = list(csv.DictReader(_CSV_PATH.open(encoding="utf-8")))
    by_pat: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_pat[int(row["pattern_id"])].append(row)
    for steps in by_pat.values():
        steps.sort(key=lambda r: int(r.get("step", 0)))
    by_role: dict[str, list[list[dict]]] = defaultdict(list)
    for steps in by_pat.values():
        role = steps[0].get("role", "")
        if role:
            by_role[role].append(steps)
    return dict(by_role)


# ---------- PatternPlayer ----------
def _smooth(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def _sway(frame: int, pid: int) -> tuple[float, float]:
    """선수별 미세 체중이동 (정지 중에도 자연스럽게 흔들림)."""
    ph = pid * 1.73
    return (0.06 * math.sin(frame * 0.12 + ph),
            0.05 * math.cos(frame * 0.09 + ph * 1.6))


class PatternPlayer:
    """CSV 패턴 데이터로 단일 선수 움직임을 재생하는 시뮬레이터.

    각 스텝의 이동을 '상대 델타'로 처리해 패턴 간 이음새가 매끄럽도록 한다.
    Team B는 x축을 코트 중앙(5.0m)에 대해 미러링한다.
    """

    def __init__(
        self,
        patterns: list[list[dict]],
        team: str,
        start: tuple[float, float],
        seed: int = 0,
    ):
        self._team = team
        self._pos: list[float] = list(start)
        self._pat_idx = 0
        self._step_idx = 0
        self._frame_in_step = 0
        self._step_start: list[float] = list(start)
        self._step_end: list[float] = list(start)
        self._fpt = _DEFAULT_FPT       # frames per step (이번 스텝)
        self._intent = ""
        self._role = ""
        self._frame_total = 0          # 전체 경과 프레임 (sway용)

        # 패턴이 없으면 제자리 대기용 더미 생성
        if not patterns:
            dummy = [{"from_x": str(start[0]), "from_y": str(start[1]),
                      "to_x":   str(start[0]), "to_y":   str(start[1]),
                      "intent": "", "role": ""}]
            self._pats = [dummy]
        else:
            self._pats = patterns[:]
            random.Random(seed).shuffle(self._pats)

        self._load_step()

    # ── 공개 프로퍼티 ────────────────────────────────────────────────
    @property
    def intent(self) -> str:
        return self._intent

    @property
    def role(self) -> str:
        return self._role

    @property
    def pos(self) -> tuple[float, float]:
        return (self._pos[0], self._pos[1])

    # ── 내부 ─────────────────────────────────────────────────────────
    def _current_step(self) -> dict:
        pat = self._pats[self._pat_idx % len(self._pats)]
        return pat[self._step_idx % len(pat)]

    def _load_step(self) -> None:
        """현재 스텝의 목표 위치를 현재 위치 기준 상대 델타로 계산."""
        step = self._current_step()
        self._intent = step.get("intent", "")
        self._role   = step.get("role", "")

        dx = float(step["to_x"]) - float(step["from_x"])
        dy = float(step["to_y"]) - float(step["from_y"])
        if self._team == "B":
            dx = -dx   # x축 미러

        tx = self._pos[0] + dx
        ty = self._pos[1] + dy

        # 진영 클리핑
        if self._team == "A":
            tx = max(0.1, min(4.9, tx))
        else:
            tx = max(5.1, min(9.9, tx))
        ty = max(0.1, min(5.9, ty))

        self._step_start = self._pos[:]
        self._step_end   = [tx, ty]
        self._fpt        = _FRAMES_PER_INTENT.get(self._intent, _DEFAULT_FPT)
        self._frame_in_step = 0

    def update(self) -> tuple[float, float]:
        """프레임 한 칸 전진 후 현재 위치 반환."""
        t = _smooth(self._frame_in_step / max(1, self._fpt - 1))
        sx, sy = self._sway_now()
        self._pos = [
            self._step_start[0] + (self._step_end[0] - self._step_start[0]) * t + sx,
            self._step_start[1] + (self._step_end[1] - self._step_start[1]) * t + sy,
        ]
        self._frame_in_step += 1
        self._frame_total   += 1

        if self._frame_in_step >= self._fpt:
            # 스텝 완료 → 목표 위치 확정 후 다음 스텝으로
            self._pos = self._step_end[:]
            self._step_idx += 1
            pat = self._pats[self._pat_idx % len(self._pats)]
            if self._step_idx >= len(pat):
                self._step_idx = 0
                self._pat_idx += 1
            self._load_step()

        return (self._pos[0], self._pos[1])

    def _sway_now(self) -> tuple[float, float]:
        return _sway(self._frame_total, id(self) % 13)


# ---------- 버드아이뷰 상태 바 ----------
def _draw_status_bar(img: np.ndarray, sim: dict[int, PatternPlayer]) -> None:
    """하단 바: 팀별 현재 의도 표시 (한글 PIL 렌더링)."""
    from src.annotate import put_text_kr

    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 34), (w, h), (12, 12, 12), -1)

    def _label(pid: int) -> str:
        p = sim[pid]
        rk = _ROLE_KO.get(p.role, "?")
        ik = _INTENT_KO.get(p.intent, p.intent or "-")
        return f"{rk}:{ik}"

    a_txt = "A팀  " + "  |  ".join(_label(pid) for pid in [1, 2, 3])
    b_txt = "B팀  " + "  |  ".join(_label(pid) for pid in [4, 5, 6])
    put_text_kr(img, a_txt, (6, h - 32), 13, (100, 255, 150))
    put_text_kr(img, b_txt, (6, h - 17), 13, (100, 150, 255))


# ---------- 카메라 배경 / bbox ----------
def _make_calib():
    """1280×720 가상 카메라: 코트를 위에서 비스듬히 내려다보는 원근감."""
    return compute_homography(
        corners_pixel=np.array([
            [180, 140],   # 좌상
            [1100, 140],  # 우상
            [1160, 590],  # 우하
            [120, 590],   # 좌하
        ], dtype=np.float32),
        court_width_m=_COURT_W,
        court_height_m=_COURT_H,
        image_size=(_FRAME_W, _FRAME_H),
    )


def _build_camera_bg(calib) -> np.ndarray:
    bg = np.full((_FRAME_H, _FRAME_W, 3), (30, 30, 30), dtype=np.uint8)
    pts = calib.corners_pixel.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(bg, [pts], (45, 55, 40))
    cv2.polylines(bg, [pts], True, (200, 200, 200), 2)
    mid_top = court_to_pixel(np.array([[5.0, 0.0]]), calib)[0].astype(int)
    mid_bot = court_to_pixel(np.array([[5.0, _COURT_H]]), calib)[0].astype(int)
    cv2.line(bg, tuple(mid_top), tuple(mid_bot), (150, 150, 60), 1, cv2.LINE_AA)
    for xi in range(1, int(_COURT_W)):
        p0 = court_to_pixel(np.array([[float(xi), 0.0]]), calib)[0].astype(int)
        p1 = court_to_pixel(np.array([[float(xi), _COURT_H]]), calib)[0].astype(int)
        cv2.line(bg, tuple(p0), tuple(p1), (55, 65, 50), 1, cv2.LINE_AA)
    for yi in range(1, int(_COURT_H) + 1):
        p0 = court_to_pixel(np.array([[0.0, float(yi)]]), calib)[0].astype(int)
        p1 = court_to_pixel(np.array([[_COURT_W, float(yi)]]), calib)[0].astype(int)
        cv2.line(bg, tuple(p0), tuple(p1), (55, 65, 50), 1, cv2.LINE_AA)
    return bg


def _court_to_bbox(xm: float, ym: float, calib) -> np.ndarray:
    foot_px = court_to_pixel(np.array([[xm, ym]]), calib)[0]
    fx, fy = float(foot_px[0]), float(foot_px[1])
    cam_top = float(calib.corners_pixel[:, 1].min())
    cam_bot = float(calib.corners_pixel[:, 1].max())
    depth = max(0.0, min(1.0, (fy - cam_top) / max(1.0, cam_bot - cam_top)))
    h_box = int(70 + 110 * depth)
    w_box = int(h_box * 0.42)
    return np.array([fx - w_box / 2, fy - h_box, fx + w_box / 2, fy], dtype=np.float32)


# ---------- 메인 루프 ----------
def run(args) -> int:
    config = _load_config()
    calib  = _make_calib()
    bg     = _build_camera_bg(calib)
    px_per_m   = 100
    court_tmpl = render_court_birdeye(calib, px_per_m=px_per_m)
    tracker    = IoUTracker(iou_threshold=0.3, max_lost_frames=10)

    # TacticEngine + MovementModel
    mv_csv = PROJECT_ROOT / "data" / "movement_data.csv"
    movement_model = MovementModel(mv_csv) if mv_csv.exists() else None
    tactic_engine  = TacticEngine.from_config(config, movement_model=movement_model)
    if movement_model:
        print(f"[Demo] MovementModel: {movement_model.pattern_count}패턴 로드")
    for tid in range(1, 4):
        tactic_engine._team_assignment[tid] = "A"
    for tid in range(4, 7):
        tactic_engine._team_assignment[tid] = "B"

    # PatternPlayer 초기화
    pat_lib = _load_pattern_library()
    n_roles = {r: len(v) for r, v in pat_lib.items()}
    print(f"[Demo] 패턴 라이브러리: { {_ROLE_KO.get(r,r): n for r,n in n_roles.items()} }")

    # 초기 위치: 양팀 3레인 분산 진형
    sim: dict[int, PatternPlayer] = {
        1: PatternPlayer(pat_lib.get("technician",    []), "A", (1.8, 1.5), seed=1),
        2: PatternPlayer(pat_lib.get("defender",      []), "A", (1.0, 4.5), seed=2),
        3: PatternPlayer(pat_lib.get("main_attacker", []), "A", (2.5, 3.0), seed=3),
        4: PatternPlayer(pat_lib.get("main_attacker", []), "B", (7.5, 3.0), seed=4),
        5: PatternPlayer(pat_lib.get("defender",      []), "B", (9.0, 1.5), seed=5),
        6: PatternPlayer(pat_lib.get("technician",    []), "B", (8.2, 4.5), seed=6),
    }

    out_dir  = PROJECT_ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    vid_path = out_dir / "demo.mp4"

    _sample = combine_views(np.zeros((_FRAME_H, _FRAME_W, 3), dtype=np.uint8), court_tmpl)
    out_h, out_w = _sample.shape[:2]
    writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             30.0, (out_w, out_h))
    print(f"[Demo] 저장: {vid_path}  ({out_w}×{out_h} @30fps, {args.frames}프레임)")

    if not args.headless:
        cv2.namedWindow("HADO Demo", cv2.WINDOW_NORMAL)

    preview_saved = False
    for fi in range(args.frames):
        # ── 선수 위치 업데이트 (PatternPlayer 1회 호출) ──────────
        pos = {pid: p.update() for pid, p in sim.items()}

        dets = [
            Detection(
                x1=float(b[0]), y1=float(b[1]),
                x2=float(b[2]), y2=float(b[3]),
                confidence=0.90,
            )
            for pid in range(1, 7)
            for b in [_court_to_bbox(*pos[pid], calib)]
        ]
        tracks = tracker.update(dets)

        player_states = [
            PlayerState(
                track_id=t.track_id,
                court_x=pos[t.track_id][0],
                court_y=pos[t.track_id][1],
                role=_PLAYER_ROLES.get(t.track_id, ""),
            )
            for t in tracks
        ]
        advices = tactic_engine.analyze(player_states)

        cam_view = bg.copy()
        draw_detections_on_frame(cam_view, tracks)

        birdeye = draw_players_on_birdeye(
            court_tmpl, tracks, calib,
            px_per_m=px_per_m,
            show_trajectory=True,
            trajectory_length=60,
        )
        birdeye = draw_guide_on_birdeye(birdeye, advices, px_per_m=px_per_m)
        _draw_status_bar(birdeye, sim)

        combined = combine_views(cam_view, birdeye)
        draw_hud(combined, fps=30.0, n_players=len(tracks))
        writer.write(combined)

        if not preview_saved and fi >= args.frames // 2:
            preview_path = out_dir / "demo_preview.png"
            cv2.imwrite(str(preview_path), combined)
            print(f"[Demo] 프리뷰: {preview_path}")
            preview_saved = True

        if not args.headless:
            cv2.imshow("HADO Demo", combined)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        if fi % 90 == 0:
            print(f"[Demo] {fi}/{args.frames} 프레임...")

    writer.release()
    if not args.headless:
        cv2.destroyAllWindows()
    print(f"[Demo] 완료 — {vid_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="HADO 버드아이뷰 데모 (카메라 불필요)")
    parser.add_argument("--headless", action="store_true", help="창 없이 실행")
    parser.add_argument("--frames", type=int, default=600,
                        help="총 프레임 수 (600=20초 @30fps)")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
