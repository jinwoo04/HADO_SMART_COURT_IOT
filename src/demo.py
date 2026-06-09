"""카메라/캘리브레이션 없이 버드아이뷰 파이프라인을 시연하는 데모.

calibration.json, YOLOv8, 실제 카메라 없이도 동작.
패턴 데이터(movement_data.csv)를 기반으로 선수가 실제 수집된 경로로 움직임.

실행:
    python -m src.demo                  # 창 표시 + data/demo.mp4 저장
    python -m src.demo --headless       # 창 없이 video만 저장
    python -m src.demo --frames 600     # 20초 미리보기
    python -m src.demo --max-frames 600 # 위와 동일한 별칭
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
from src.guide import draw_guide_on_birdeye, draw_player_overlays
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
_CSV_PATH_ANNOTATED = PROJECT_ROOT / "data" / "movement_data_annotated.csv"
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
# 팀 배정 (1-3: 팀A, 4-6: 팀B)
_PLAYER_TEAMS = {1: "A", 2: "A", 3: "A", 4: "B", 5: "B", 6: "B"}


# ---------- 패턴 라이브러리 로드 ----------
def _load_pattern_library(
    team: str = "A",
    csv_path: Path | None = None,
    player_id: int | None = None,
) -> dict[str, dict[str, list[list[dict]]]]:
    """CSV → {role: {context: [[step_dict, ...], ...]}} 구조로 로드.

    team 필터: team A 패턴(x=0-5m)만 사용.
    player_id 필터: 특정 선수의 패턴만 로드할 때 사용.
    """
    path = csv_path or _CSV_PATH
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows = [r for r in rows if r.get("team", "A") == team]
    if player_id is not None:
        rows = [r for r in rows if int(r.get("player_id", 0)) == player_id]

    by_pat: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_pat[int(row["pattern_id"])].append(row)
    for steps in by_pat.values():
        steps.sort(key=lambda r: int(r.get("step", 0)))

    # role → context → [patterns]
    result: dict[str, dict[str, list[list[dict]]]] = {}
    for steps in by_pat.values():
        role = steps[0].get("role", "")
        ctx  = steps[0].get("context", "attack")
        if not role:
            continue
        result.setdefault(role, {}).setdefault(ctx, []).append(steps)
    return result


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

    - 절대 좌표(to_x, to_y)로 목표 위치를 결정해 drift 방지
    - context(attack/defend/transition)에 맞는 패턴만 선택
    - Team B는 x축을 5.0m 기준으로 미러링
    - role_name으로 역할별 구역 제한(defender: x≤2.2m, attacker: x≥2.5m)
    """

    # 역할별 구역 제한 (컨텍스트별)
    # defender: 수비 시 x≤1.8m, 공격 시에도 최대 x≤3.8m (완전 전방은 안 감)
    # main_attacker: 공격 시 x≥2.5m, 수비 시에도 최소 x≥0.5m
    _ZONE_CLAMP: dict[str, dict[str, tuple[float, float]]] = {
        "defender": {
            "defend":     (0.1, 1.8),
            "attack":     (0.3, 3.8),
            "transition": (0.2, 2.5),
        },
        "main_attacker": {
            "attack":     (2.5, 4.9),
            "defend":     (0.5, 3.5),
            "transition": (1.5, 4.0),
        },
    }

    def __init__(
        self,
        patterns_by_ctx: dict[str, list[list[dict]]],
        team: str,
        start: tuple[float, float],
        seed: int = 0,
        role_name: str = "",
    ):
        self._team = team
        self._role_name = role_name
        self._pos: list[float] = list(start)
        self._context = "attack"
        self._pending_context: str | None = None
        self._rng = random.Random(seed)
        self._patterns_by_ctx = patterns_by_ctx
        self._intent = ""
        self._role = ""
        self._frame_total = 0

        self._step_start: list[float] = list(start)
        self._step_end: list[float] = list(start)
        self._fpt = _DEFAULT_FPT
        self._frame_in_step = 0
        self._cur_pattern: list[dict] = []
        self._step_idx = 0

        self._pick_new_pattern()

    # ── 공개 인터페이스 ──────────────────────────────────────────────
    @property
    def velocity_dir(self) -> tuple[float, float]:
        """현재 스텝의 이동 방향 단위벡터. 정지 중이면 (0,0)."""
        dx = self._step_end[0] - self._step_start[0]
        dy = self._step_end[1] - self._step_start[1]
        norm = math.hypot(dx, dy)
        if norm < 1e-4:
            return (0.0, 0.0)
        return (dx / norm, dy / norm)

    def set_context(self, ctx: str) -> None:
        """컨텍스트 전환 요청 — 현재 스텝 완료 후 반영(끊김 방지)."""
        if ctx != self._context and ctx != self._pending_context:
            self._pending_context = ctx

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
    def _pick_new_pattern(self) -> None:
        candidates = self._patterns_by_ctx.get(self._context, [])
        if not candidates:
            # context 매칭 없으면 전체 fallback
            candidates = [p for pats in self._patterns_by_ctx.values() for p in pats]
        if not candidates:
            self._cur_pattern = [{"from_x": str(self._pos[0]), "from_y": str(self._pos[1]),
                                   "to_x":  str(self._pos[0]), "to_y":  str(self._pos[1]),
                                   "intent": "", "role": ""}]
        else:
            self._cur_pattern = self._rng.choice(candidates)
        self._step_idx = 0
        self._load_step()

    def _load_step(self) -> None:
        """현재 스텝의 절대 목표 좌표를 계산. Team B는 x 미러링."""
        step = self._cur_pattern[self._step_idx % len(self._cur_pattern)]
        self._intent = step.get("intent", "")
        self._role   = step.get("role", "")

        # 절대 좌표 사용 (델타 누적 X)
        tx = float(step["to_x"])
        ty = float(step["to_y"])
        if self._team == "B":
            tx = _COURT_W - tx   # x축 미러

        if self._team == "A":
            tx = max(0.1, min(4.9, tx))
        else:
            tx = max(5.1, min(9.9, tx))
        ty = max(0.1, min(5.9, ty))

        # 역할별 구역 제한 — defender는 수비 시 x≤2.2m, attacker는 공격 시 x≥2.5m
        zone = self._ZONE_CLAMP.get(self._role_name, {}).get(self._context)
        if zone:
            tx = max(zone[0], min(zone[1], tx))

        self._step_start    = self._pos[:]
        self._step_end      = [tx, ty]
        self._fpt           = _FRAMES_PER_INTENT.get(self._intent, _DEFAULT_FPT)
        self._frame_in_step = 0

    def update(self) -> tuple[float, float]:
        t = _smooth(self._frame_in_step / max(1, self._fpt - 1))
        sx, sy = _sway(self._frame_total, id(self) % 13)
        self._pos = [
            self._step_start[0] + (self._step_end[0] - self._step_start[0]) * t + sx,
            self._step_start[1] + (self._step_end[1] - self._step_start[1]) * t + sy,
        ]
        self._frame_in_step += 1
        self._frame_total   += 1

        if self._frame_in_step >= self._fpt:
            self._pos = self._step_end[:]
            # 대기 중인 컨텍스트 전환이 있으면 스텝 경계에서 처리
            if self._pending_context is not None:
                self._context = self._pending_context
                self._pending_context = None
                self._pick_new_pattern()
            else:
                self._step_idx += 1
                if self._step_idx >= len(self._cur_pattern):
                    self._pick_new_pattern()
                else:
                    self._load_step()

        return (self._pos[0], self._pos[1])


# ---------- 버드아이뷰 상태 바 ----------
def _draw_status_bar(img: np.ndarray, sim: dict[int, PatternPlayer]) -> None:
    """하단 2줄 바: 팀A/B 선수별 역할 + 현재 의도 표시."""
    from src.annotate import put_text_kr

    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 38), (w, h), (12, 12, 12), -1)

    # 팀A (pid 1-3)
    parts_a = []
    for pid in sorted(p for p in sim if p <= 3):
        p = sim[pid]
        role_ko   = _ROLE_KO.get(p.role or _PLAYER_ROLES.get(pid, ""), "?")
        intent_ko = _INTENT_KO.get(p.intent, "")
        parts_a.append(f"#{pid}{role_ko[:2]}" + (f"[{intent_ko[:4]}]" if intent_ko else ""))
    put_text_kr(img, "A:  " + "  ".join(parts_a), (8, h - 34), 11, (100, 255, 150))

    # 팀B (pid 4-6)
    parts_b = []
    for pid in sorted(p for p in sim if p >= 4):
        p = sim[pid]
        role_ko   = _ROLE_KO.get(p.role or _PLAYER_ROLES.get(pid, ""), "?")
        intent_ko = _INTENT_KO.get(p.intent, "")
        parts_b.append(f"#{pid}{role_ko[:2]}" + (f"[{intent_ko[:4]}]" if intent_ko else ""))
    put_text_kr(img, "B:  " + "  ".join(parts_b), (8, h - 20), 11, (100, 150, 255))


def _draw_zone_hud(img: np.ndarray, pos: dict[int, tuple[float, float]]) -> None:
    """우상단 미니 HUD: 현재 선수별 구역(1·2·3선) 위치 표시.

    Zone 1: x < 1.5m (깊은 수비)
    Zone 2: 1.5 ≤ x < 3.0m (미들)
    Zone 3: x ≥ 3.0m (전방/공격)
    """
    zones = {1: [], 2: [], 3: []}
    for pid, (x, _) in pos.items():
        if x < 1.5:
            zones[1].append(pid)
        elif x < 3.0:
            zones[2].append(pid)
        else:
            zones[3].append(pid)

    h, w = img.shape[:2]
    panel_w, panel_h = 110, 58
    x0 = w - panel_w - 6
    y0 = 4

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (60, 60, 60), 1)

    cv2.putText(img, "Zone", (x0 + 4, y0 + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

    zone_labels = ["1  (back)", "2  (mid)", "3  (front)"]
    zone_colors = [(255, 160, 60), (60, 200, 255), (60, 80, 255)]
    for i, (lbl, clr) in enumerate(zip(zone_labels, zone_colors)):
        pids = zones[i + 1]
        count_dot = "●" * len(pids) if pids else "○"
        iy = y0 + 26 + i * 13
        cv2.putText(img, lbl, (x0 + 4, iy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (130, 130, 130), 1, cv2.LINE_AA)
        cv2.putText(img, str(len(pids)), (x0 + panel_w - 18, iy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, clr, 1, cv2.LINE_AA)
        # 작은 원으로 각 선수 표시
        for j in range(len(pids)):
            cx = x0 + panel_w - 28 - j * 9
            cv2.circle(img, (cx, iy - 3), 3, clr, -1, cv2.LINE_AA)


# ---------- 규칙 발동 카운터 HUD ----------
def _draw_rule_counter(img: np.ndarray, counts: dict[str, int], frame: int) -> None:
    """좌하단 규칙 발동 누적 카운터 (R1-R6 × 2열 그리드).

    frame이 작을 때는 표시 안 함(워밍업 구간).
    """
    if frame < 50:
        return
    h, w = img.shape[:2]
    rules = [("R1", "분산"), ("R2", "커버"),
             ("R3", "회피"), ("R4", "갭공격"),
             ("R5", "후퇴"), ("R6", "레인")]
    colors = {
        "R1": (0, 220, 255), "R2": (0, 200, 160),
        "R3": (0, 80,  255), "R4": (0, 200, 255),
        "R5": (0, 60,  200), "R6": (120, 255, 80),
    }

    cols, rows = 3, 2
    cell_w, cell_h = 60, 20
    pad = 5
    panel_w = cols * cell_w + 2 * pad
    panel_h = rows * cell_h + 2 * pad + 12  # +12 for title
    x0 = pad
    y0 = h - panel_h - 44  # status bar 위

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.70, img, 0.30, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (60, 60, 60), 1)
    cv2.putText(img, "Rules", (x0 + pad, y0 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 160, 160), 1)

    for i, (rule, short) in enumerate(rules):
        col, row = i % cols, i // cols
        cx = x0 + pad + col * cell_w
        cy = y0 + 14 + row * cell_h
        cnt = counts.get(rule, 0)
        clr = colors.get(rule, (180, 180, 180))
        # 발동된 규칙은 밝게
        if cnt > 0:
            cv2.rectangle(img, (cx, cy), (cx + cell_w - 2, cy + cell_h - 2),
                          tuple(max(0, c // 4) for c in clr), -1)
        label = f"{rule}:{cnt // max(1, frame // 30):02d}/s" if cnt > 0 else f"{rule}:--"
        cv2.putText(img, label, (cx + 2, cy + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                    clr if cnt > 0 else (80, 80, 80), 1, cv2.LINE_AA)


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
    bg = np.full((_FRAME_H, _FRAME_W, 3), (25, 25, 25), dtype=np.uint8)
    pts = calib.corners_pixel.astype(np.int32).reshape(-1, 1, 2)

    # 팀A 진영(x=0-5) 녹색, 팀B 진영(x=5-10) 파랑 — 반투명 팀 구역 표시
    mid_top = court_to_pixel(np.array([[5.0, 0.0]]), calib)[0].astype(int)
    mid_bot = court_to_pixel(np.array([[5.0, _COURT_H]]), calib)[0].astype(int)
    left_pts  = np.array([calib.corners_pixel[0], mid_top,
                          mid_bot, calib.corners_pixel[3]], dtype=np.int32)
    right_pts = np.array([mid_top, calib.corners_pixel[1],
                          calib.corners_pixel[2], mid_bot], dtype=np.int32)
    overlay = bg.copy()
    cv2.fillPoly(overlay, [left_pts],  (45, 65, 40))   # 팀A: 초록
    cv2.fillPoly(overlay, [right_pts], (40, 45, 65))   # 팀B: 파랑
    cv2.addWeighted(overlay, 0.85, bg, 0.15, 0, bg)

    # 코트 외곽
    cv2.polylines(bg, [pts], True, (200, 200, 200), 2, cv2.LINE_AA)

    # 1m 그리드 (옅게)
    for xi in range(1, int(_COURT_W)):
        p0 = court_to_pixel(np.array([[float(xi), 0.0]]), calib)[0].astype(int)
        p1 = court_to_pixel(np.array([[float(xi), _COURT_H]]), calib)[0].astype(int)
        cv2.line(bg, tuple(p0), tuple(p1), (50, 60, 48), 1, cv2.LINE_AA)
    for yi in range(1, int(_COURT_H) + 1):
        p0 = court_to_pixel(np.array([[0.0, float(yi)]]), calib)[0].astype(int)
        p1 = court_to_pixel(np.array([[_COURT_W, float(yi)]]), calib)[0].astype(int)
        cv2.line(bg, tuple(p0), tuple(p1), (50, 60, 48), 1, cv2.LINE_AA)

    # 중앙선 (굵게)
    cv2.line(bg, tuple(mid_top), tuple(mid_bot), (160, 160, 60), 2, cv2.LINE_AA)

    # HADO 전술 구역선 (x=1.5, 3.0m — 팀A 쪽; 7.0, 8.5m — 팀B 쪽)
    for xz, clr in [(1.5, (60, 160, 80)), (3.0, (60, 200, 80)),
                    (7.0, (80, 100, 200)), (8.5, (60, 80, 200))]:
        p0 = court_to_pixel(np.array([[xz, 0.0]]), calib)[0].astype(int)
        p1 = court_to_pixel(np.array([[xz, _COURT_H]]), calib)[0].astype(int)
        cv2.line(bg, tuple(p0), tuple(p1), clr, 1, cv2.LINE_AA)

    # 레인 경계 (y=2.0, 4.0m)
    for yz in [2.0, 4.0]:
        p0 = court_to_pixel(np.array([[0.0, yz]]), calib)[0].astype(int)
        p1 = court_to_pixel(np.array([[_COURT_W, yz]]), calib)[0].astype(int)
        cv2.line(bg, tuple(p0), tuple(p1), (60, 130, 80), 1, cv2.LINE_AA)

    # 팀 라벨 (원근 투영 좌표계 내)
    team_a_mid = court_to_pixel(np.array([[2.5, 3.0]]), calib)[0].astype(int)
    team_b_mid = court_to_pixel(np.array([[7.5, 3.0]]), calib)[0].astype(int)
    cv2.putText(bg, "TEAM A", tuple(team_a_mid - [30, 0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 200, 100), 1, cv2.LINE_AA)
    cv2.putText(bg, "TEAM B", tuple(team_b_mid - [30, 0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 120, 220), 1, cv2.LINE_AA)

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


# ---------- Role별 히트맵 ----------
_ROLE_CMAPS = {
    "main_attacker": cv2.COLORMAP_HOT,
    "technician":    cv2.COLORMAP_OCEAN,
    "defender":      cv2.COLORMAP_WINTER,
}
_ROLE_LABEL_KO = {
    "main_attacker": "Attacker",
    "technician":    "Technician",
    "defender":      "Defender",
}
_TEAM_COLOR = {
    "A": (100, 255, 150),
    "B": (100, 150, 255),
}


def _generate_demo_heatmap(
    pos_log: list[dict],
    calib,
    px_per_m: int,
    out_dir: Path,
) -> None:
    """팀A 선수들의 포지션 로그로 role별 히트맵 생성 (x=0–5m 반쪽 확대)."""
    # 팀A 반쪽만 표시: x=0–5m → W_half 폭
    W_half = int(5.0 * px_per_m)
    H = int(_COURT_H * px_per_m)
    BLUR = 41  # 넓은 blur로 zone이 자연스럽게 표시

    from src.visualizer import render_court_birdeye as _rcb

    def _one(rows: list[dict], cmap: int, label: str) -> np.ndarray:
        hm = np.zeros((H, W_half), dtype=np.float32)
        for r in rows:
            px_i = int(np.clip(r["x"] * px_per_m, 0, W_half - 1))
            py_i = int(np.clip(r["y"] * px_per_m, 0, H - 1))
            hm[py_i, px_i] += 1.0
        if hm.max() > 0:
            hm = cv2.GaussianBlur(hm, (BLUR, BLUR), 0)
            hm /= hm.max()
        colored = cv2.applyColorMap((hm * 255).astype(np.uint8), cmap)

        # 코트 라인 오버레이 (팀A 반쪽 crop)
        court_full = _rcb(calib, px_per_m=px_per_m)
        court_crop = court_full[:H, :W_half]
        gray = cv2.cvtColor(court_crop, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        colored[mask > 0] = (200, 200, 200)

        # 역할 구역선: 1선(x=1.5), 2선(x=3.0), 레인(y=2,4)
        for xz in [1.5, 3.0]:
            cv2.line(colored, (int(xz*px_per_m), 0), (int(xz*px_per_m), H),
                     (200, 200, 80), 1, cv2.LINE_AA)
        for yz in [2.0, 4.0]:
            cv2.line(colored, (0, int(yz*px_per_m)), (W_half, int(yz*px_per_m)),
                     (80, 200, 80), 1, cv2.LINE_AA)

        # 역할별 기본 구역 강조 (반투명 사각형)
        zone_map = {
            "Technician": (0.0, 5.0),    # 전체 폭 (side-to-side)
            "Attacker":   (3.0, 5.0),    # 전방 2m
            "Defender":   (0.0, 2.0),    # 후방 2m
        }
        if label in zone_map:
            zx0, zx1 = zone_map[label]
            px0, px1 = int(zx0*px_per_m), min(W_half-1, int(zx1*px_per_m))
            overlay = colored.copy()
            cv2.rectangle(overlay, (px0, 0), (px1, H), (255, 255, 100), 2)
            cv2.addWeighted(overlay, 0.3, colored, 0.7, 0, colored)
            cv2.rectangle(colored, (px0, 0), (px1, H), (255, 255, 100), 2)

        # 구역 점유율 계산 — 수비/공격 페이즈 기준 통계
        if rows:
            if label == "Attacker":
                atk_rows = [r for r in rows if r.get("phase") == "attack"]
                if atk_rows:
                    in_zone = sum(1 for r in atk_rows if r["x"] >= 3.0)
                    pct_txt = f"Atk Zone(atk phase): {in_zone/len(atk_rows)*100:.0f}%"
                else:
                    in_zone = sum(1 for r in rows if r["x"] >= 3.0)
                    pct_txt = f"Atk Zone: {in_zone/len(rows)*100:.0f}%"
            elif label == "Defender":
                def_rows = [r for r in rows if r.get("phase") == "defend"]
                if def_rows:
                    in_zone = sum(1 for r in def_rows if r["x"] <= 2.0)
                    pct_txt = f"Def Zone(def phase): {in_zone/len(def_rows)*100:.0f}%"
                else:
                    in_zone = sum(1 for r in rows if r["x"] <= 2.0)
                    pct_txt = f"Def Zone: {in_zone/len(rows)*100:.0f}%"
            else:
                xs = [r["x"] for r in rows]
                travel = sum(abs(xs[i]-xs[i-1]) for i in range(1, len(xs)))
                pct_txt = f"Travel: {travel:.0f}m"
            cv2.putText(colored, pct_txt, (6, H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 200), 1, cv2.LINE_AA)

        # 라벨
        cv2.putText(colored, label, (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(colored, label, (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        # 좌표 힌트
        cv2.putText(colored, "0m", (4, H - 22 if rows else H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        cv2.putText(colored, "5m(center)", (W_half - 72, H - 22 if rows else H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        return colored

    by_role: dict[str, list[dict]] = {}
    for r in pos_log:
        by_role.setdefault(r["role"], []).append(r)

    roles_order = ["technician", "main_attacker", "defender"]
    sep = np.full((H, 6, 3), 60, dtype=np.uint8)
    panels = []
    for role in roles_order:
        rows = by_role.get(role, [])
        cmap  = _ROLE_CMAPS.get(role, cv2.COLORMAP_JET)
        label = _ROLE_LABEL_KO.get(role, role)
        panels.append(_one(rows, cmap, label))
        panels.append(sep.copy())

    if panels:
        combined = np.hstack(panels[:-1])
        out_path = out_dir / "heatmap_role.png"
        cv2.imwrite(str(out_path), combined)
        print(f"[Demo] 포지션 히트맵: {out_path}  ({len(pos_log)}개 로그포인트)")


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

    # PatternPlayer 초기화 — 팀A 3명 + 팀B 3명 (6 vs 6)
    csv_override = Path(args.csv) if getattr(args, "csv", None) else None

    def _pat(pid: int, role: str, team: str = "A") -> dict:
        if csv_override:
            return _load_pattern_library(csv_path=csv_override, player_id=pid)
        lib = _load_pattern_library(team="A")   # 항상 팀A 패턴 사용 (팀B는 미러링)
        return {role: lib.get(role, {})}

    pat1 = _pat(1, "technician")
    pat2 = _pat(2, "technician")
    pat3 = _pat(3, "main_attacker")

    # 팀B는 팀A 패턴을 미러링 — PatternPlayer(team="B")가 x축 반전 처리
    pat4 = _pat(4, "main_attacker")
    pat5 = _pat(5, "defender")
    pat6 = _pat(6, "technician")

    for pid, lib in [(1, pat1), (2, pat2), (3, pat3)]:
        n = sum(len(v) for ctxs in lib.values() for v in ctxs.values())
        print(f"[Demo] P{pid}(A) 패턴: {n}개")
    for pid, lib in [(4, pat4), (5, pat5), (6, pat6)]:
        role = _PLAYER_ROLES.get(pid, "")
        n = sum(len(v) for ctxs in lib.values() for v in ctxs.values())
        print(f"[Demo] P{pid}(B/{role[:4]}) 패턴: {n}개 (미러)")

    sim: dict[int, PatternPlayer] = {
        # 팀A: x=0-5m 진영
        1: PatternPlayer(pat1.get("technician",    {}), "A", (1.5, 1.0), seed=1, role_name="technician"),
        2: PatternPlayer(pat2.get("technician",    {}), "A", (4.0, 3.0), seed=2, role_name="technician"),
        3: PatternPlayer(pat3.get("main_attacker", {}), "A", (2.8, 1.0), seed=3, role_name="main_attacker"),
        # 팀B: x=5-10m 진영 (미러링)
        4: PatternPlayer(pat4.get("main_attacker", {}), "B", (7.2, 5.0), seed=14, role_name="main_attacker"),
        5: PatternPlayer(pat5.get("defender",      {}), "B", (9.0, 4.5), seed=15, role_name="defender"),
        6: PatternPlayer(pat6.get("technician",    {}), "B", (6.5, 1.5), seed=16, role_name="technician"),
    }

    # 게임 페이즈 사이클: attack(9s) → transition(2s) → defend(8s) → transition(2s)
    _PHASE_CYCLE = [
        ("attack",     270),
        ("transition", 60),
        ("defend",     240),
        ("transition", 60),
    ]
    _phase_idx   = 0
    _phase_frame = 0

    out_dir  = PROJECT_ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    vid_path = out_dir / "demo.mp4"

    # x=0-6m 확장 뷰 (팀A 전체 + 팀B 진입 1m) — 팀B가 화면에 보임
    _ext_w = int(6.0 * px_per_m)   # 600px
    _court_ext = court_tmpl[:, :_ext_w]
    _sample = combine_views(np.zeros((_FRAME_H, _FRAME_W, 3), dtype=np.uint8), _court_ext)
    out_h, out_w = _sample.shape[:2]
    writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             30.0, (out_w, out_h))
    print(f"[Demo] 저장: {vid_path}  ({out_w}×{out_h} @30fps, {args.frames}프레임)")

    # 규칙 발동 누적 카운터 (발표 중 어떤 규칙이 얼마나 활성화됐는지)
    _rule_counts: dict[str, int] = {r: 0 for r in ("R1", "R2", "R3", "R4", "R5", "R6")}

    if not args.headless:
        cv2.namedWindow("HADO Demo", cv2.WINDOW_NORMAL)

    # 포지션 로그 (role별 히트맵용) — 팀A만, 워밍업 300프레임 제외
    _pos_log: list[dict] = []
    _TEAM_A_PIDS = {1, 2, 3}
    _WARMUP_FRAMES = 300

    preview_saved = False
    for fi in range(args.frames):
        # ── 게임 페이즈 전환 ──────────────────────────────────────
        cur_phase, phase_dur = _PHASE_CYCLE[_phase_idx]
        _phase_frame += 1
        if _phase_frame >= phase_dur:
            _phase_frame = 0
            _phase_idx   = (_phase_idx + 1) % len(_PHASE_CYCLE)
            cur_phase, _ = _PHASE_CYCLE[_phase_idx]
            for p in sim.values():
                p.set_context(cur_phase)

        # ── 선수 위치 업데이트 (팀A + 팀B 모두) ─────────────────
        pos = {pid: p.update() for pid, p in sim.items()}
        if fi >= _WARMUP_FRAMES:
            for pid, (x, y) in pos.items():
                if pid in _TEAM_A_PIDS:
                    _pos_log.append({
                        "pid":   pid,
                        "role":  sim[pid].role or _PLAYER_ROLES.get(pid, ""),
                        "phase": cur_phase,
                        "x": x, "y": y,
                    })

        # 팀A + 팀B 6명 모두 Detection 생성
        dets = [
            Detection(
                x1=float(b[0]), y1=float(b[1]),
                x2=float(b[2]), y2=float(b[3]),
                confidence=0.90,
            )
            for pid in [1, 2, 3, 4, 5, 6]
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
            if t.track_id in pos
        ]
        advices = tactic_engine.analyze(player_states)

        # 규칙 발동 카운터 업데이트
        for adv in advices:
            if adv.rule in _rule_counts:
                _rule_counts[adv.rule] += 1

        cam_view = bg.copy()
        draw_detections_on_frame(cam_view, tracks)

        birdeye = draw_players_on_birdeye(
            court_tmpl, tracks, calib,
            px_per_m=px_per_m,
            show_trajectory=True,
            trajectory_length=60,
        )
        # 전술 분석 패널 (텍스트 + HIGH urgency 링)
        birdeye = draw_guide_on_birdeye(birdeye, advices, px_per_m=px_per_m)

        # 선수별 이동방향 화살표 + 의도 뱃지 (팀A + 팀B)
        p_overlays = [
            {
                "pid":     pid,
                "x":       x,
                "y":       y,
                "vx":      sim[pid].velocity_dir[0],
                "vy":      sim[pid].velocity_dir[1],
                "intent":  sim[pid].intent,
                "role_ko": _ROLE_KO.get(sim[pid].role or _PLAYER_ROLES.get(pid, ""), ""),
            }
            for pid, (x, y) in pos.items()
        ]
        birdeye = draw_player_overlays(birdeye, p_overlays, px_per_m=px_per_m)

        # x=0-6m 확장 크롭 (팀A 전체 + 팀B 진입 구역 1m 노출)
        ext_w = int(6.0 * px_per_m)
        birdeye_ext = birdeye[:, :ext_w].copy()

        # 센터라인 + 팀B 진입 경계 레이블
        bh = birdeye_ext.shape[0]
        center_px = int(5.0 * px_per_m)
        cv2.line(birdeye_ext, (center_px, 0), (center_px, bh), (180, 180, 60), 2)
        cv2.putText(birdeye_ext, "CENTER",
                    (center_px - 34, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (180, 180, 60), 1, cv2.LINE_AA)

        # 팀B 진입 구역 (x=5-6m) 반투명 하이라이트
        overlay_b = birdeye_ext.copy()
        cv2.rectangle(overlay_b, (center_px, 0), (ext_w - 1, bh - 1), (60, 60, 160), -1)
        cv2.addWeighted(overlay_b, 0.12, birdeye_ext, 0.88, 0, birdeye_ext)
        cv2.putText(birdeye_ext, "TEAM B",
                    (center_px + 6, bh // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (100, 100, 200), 1, cv2.LINE_AA)

        # 페이즈 레이블 (좌상단)
        _PHASE_KO = {"attack": "ATTACK", "defend": "DEFEND", "transition": "TRANSITION"}
        _PHASE_CLR = {"attack": (60, 80, 255), "defend": (255, 120, 60),
                      "transition": (60, 200, 255)}
        ph_txt = _PHASE_KO.get(cur_phase, cur_phase.upper())
        ph_clr = _PHASE_CLR.get(cur_phase, (200, 200, 200))
        cv2.rectangle(birdeye_ext, (6, 4), (6 + len(ph_txt) * 10 + 8, 24), (0, 0, 0), -1)
        cv2.putText(birdeye_ext, ph_txt, (10, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, ph_clr, 2, cv2.LINE_AA)

        # 실시간 구역 점유 HUD (우상단)
        _draw_zone_hud(birdeye_ext, {p: pos[p] for p in pos if p <= 3})

        # 규칙 발동 카운터 HUD (좌하단)
        _draw_rule_counter(birdeye_ext, _rule_counts, fi)

        _draw_status_bar(birdeye_ext, sim)

        combined = combine_views(cam_view, birdeye_ext)
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

    _generate_demo_heatmap(_pos_log, calib, px_per_m, out_dir)
    return 0


def main():
    parser = argparse.ArgumentParser(description="HADO 버드아이뷰 데모 (카메라 불필요)")
    parser.add_argument("--headless", action="store_true", help="창 없이 실행")
    parser.add_argument("--frames", "--max-frames", type=int, default=600, dest="frames",
                        help="총 프레임 수 (600=20초 @30fps)")
    parser.add_argument("--csv", type=str, default=None,
                        help="패턴 CSV 경로 (기본: data/movement_data.csv)")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
