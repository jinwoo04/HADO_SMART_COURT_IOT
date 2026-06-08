"""수동 어노테이션 도구 — 경기 영상에서 선수 동선을 movement_data.csv 포맷으로 기록.

사용법:
    python -m src.annotate_tool                       # 윈터컵 결승 2경기 (기본)
    python -m src.annotate_tool --video data/다른영상.mp4 --team A

[캘리브레이션] — 첫 실행 시
  코트 모서리를 순서대로 클릭:
    (1) 좌하단 — near-left  (x=0m, y=6m, 카메라에 가까운 왼쪽)
    (2) 우하단 — near-right (x=10m, y=6m)
    (3) 우상단 — far-right  (x=10m, y=0m, 멀리 있는 오른쪽)
    (4) 좌상단 — far-left   (x=0m, y=0m)
  → data/annotate_corners.json에 저장 (다음 실행 시 자동 로드)
  → [D] 키로 기본값 적용 (윈터컵 결승 2경기 전용)

[어노테이션 모드]
  프레임:  Space=+30프레임  . =+5  , =-5  F=+1  B=-1
  선수선택: 1 / 2 / 3  (팀에 따라 해당 팀 선수 1·2·3번)
  역할:    T=테크니션  D=디펜더  M=어태커
  컨텍스트: A=공격(attack)  V=수비(defend)  R=전환(transition)
  의도:    I=다음의도  U=이전의도
  마킹:    좌클릭 = 카메라뷰에서 선수 발끝 클릭 → 코트 좌표 변환
           Z = 마지막 클릭 취소
  저장:    Enter = 현재 선수 패턴 저장 (2개 이상 점 필요)
           C = 현재 선수 흔적 초기화
           S = CSV 파일에 즉시 쓰기
           Q = 저장 후 종료

[출력]
  data/movement_data_annotated.csv  (movement_data.csv와 동일 포맷)
  어노테이션 완료 후 다음 명령으로 병합:
    tail -n +2 data/movement_data_annotated.csv >> data/movement_data.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ─── 경로 ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_VIDEO = PROJECT_ROOT / "data" / "윈터컵 결승 2경기 RGB vs MAJOR 블루코트.mp4"
_OUT_CSV       = PROJECT_ROOT / "data" / "movement_data_annotated.csv"
_CORNER_JSON   = PROJECT_ROOT / "data" / "annotate_corners.json"
_CSV_FIELDS    = [
    "pattern_id", "step", "player_id", "team", "role",
    "from_x", "from_y", "to_x", "to_y",
    "context", "intent", "timestamp",
]

# 윈터컵 결승 2경기 기본 코너 (픽셀) — 필요시 캘리브레이션으로 정밀 조정
_DEFAULT_CORNERS_WINTER = np.array([
    [ 85.0, 500.0],  # (1) 좌하단 near-left  → (x=0m,  y=6m)
    [1195.0, 500.0], # (2) 우하단 near-right → (x=10m, y=6m)
    [1185.0, 148.0], # (3) 우상단 far-right  → (x=10m, y=0m)
    [  90.0, 148.0], # (4) 좌상단 far-left   → (x=0m,  y=0m)
], dtype=np.float32)

# 풀코트 (0-10m × 0-6m)
_COURT_FULL_M = np.array([
    [ 0.0, 6.0], [10.0, 6.0], [10.0, 0.0], [ 0.0, 0.0]
], dtype=np.float32)

# 반코트 (팀A 0-5m / 팀B 5-10m)
_COURT_HALF_M: dict[str, np.ndarray] = {
    "A": np.array([[0.0,6.0],[5.0,6.0],[5.0,0.0],[0.0,0.0]], dtype=np.float32),
    "B": np.array([[5.0,6.0],[10.0,6.0],[10.0,0.0],[5.0,0.0]], dtype=np.float32),
}

_ROLES    = ["main_attacker", "technician", "defender"]
_CONTEXTS = ["attack", "defend", "transition"]
_INTENTS  = [
    "direct_attack", "feint_attack", "cross_court", "gap_exploit",
    "lure_attention", "create_space", "bait_inward", "support_fire",
    "shield_protect", "shield_attack_support", "shield_feint", "counter_shield",
]
_INTENT_KO = {
    "direct_attack": "직접공격",  "feint_attack": "페인트공격",
    "cross_court":   "코트횡단",  "gap_exploit":  "공간공략",
    "lure_attention":"시선유도",  "create_space": "공간창출",
    "bait_inward":   "안쪽유도",  "support_fire": "공격지원",
    "shield_protect":"수비쉴드",  "shield_attack_support":"쉴드지원",
    "shield_feint":  "쉴드페인트","counter_shield":"맞쉴드",
}
_ROLE_KO = {"main_attacker": "어태커", "technician": "테크니션", "defender": "디펜더"}
_CTX_KO  = {"attack": "공격", "defend": "수비", "transition": "전환"}

# 선수별 색상 (BGR)
_P_COLOR = {1: (0, 230, 100), 2: (80, 130, 255), 3: (0, 190, 255)}
_CAM_W, _CAM_H = 1280, 720   # 원본 해상도
_DISP_W, _DISP_H = 960, 540  # 화면 표시 해상도 (3/4 축소)
_PX_M = 55                    # bird-eye px/m


# ─── DataClass ───────────────────────────────────────────────────────
@dataclass
class PlayerTrace:
    player_id: int
    team: str
    role: str = "main_attacker"
    context: str = "attack"
    intent: str = "direct_attack"
    pts: list[tuple[float, float]] = field(default_factory=list)  # court (m)

    def to_rows(self, pattern_id: int) -> list[dict]:
        rows = []
        for i in range(1, len(self.pts)):
            fx, fy = self.pts[i - 1]
            tx, ty = self.pts[i]
            rows.append({
                "pattern_id": pattern_id, "step": i,
                "player_id":  self.player_id, "team": self.team,
                "role": self.role,
                "from_x": f"{fx:.3f}", "from_y": f"{fy:.3f}",
                "to_x":   f"{tx:.3f}", "to_y":   f"{ty:.3f}",
                "context": self.context, "intent": self.intent,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
        return rows


# ─── Annotation Tool ─────────────────────────────────────────────────
class AnnotateTool:
    def __init__(self, video: Path, team: str):
        self.video_path = video
        self.team = team  # "A" or "B"

        self.cap = cv2.VideoCapture(str(video))
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps   = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_idx = 0
        self.frame: Optional[np.ndarray] = None

        # Homography pixel→court (풀코트 기준, 반코트 옵션)
        self.corners_px: list[tuple[float, float]] = []
        self.H: Optional[np.ndarray] = None
        self.calibrated = False

        # Annotation state
        self.active_pid = 1
        self.traces: dict[int, PlayerTrace] = {
            p: PlayerTrace(p, team) for p in (1, 2, 3)
        }
        self.intent_idx: dict[int, int] = {p: 0 for p in (1, 2, 3)}
        self.saved: list[dict] = []
        self.base_pid = self._next_pattern_id()
        self.pattern_id = self.base_pid

        self._try_load_corners()
        self._seek(0)

    # ── CSV ──────────────────────────────────────────────────────────
    def _next_pattern_id(self) -> int:
        max_id = 0
        for p in (_OUT_CSV, PROJECT_ROOT / "data" / "movement_data.csv"):
            if p.exists():
                for r in csv.DictReader(p.open(encoding="utf-8")):
                    max_id = max(max_id, int(r.get("pattern_id", 0)))
        return max_id + 1

    def _try_load_corners(self):
        if _CORNER_JSON.exists():
            data = json.loads(_CORNER_JSON.read_text())
            key = self.video_path.name
            if key in data:
                self.corners_px = [tuple(c) for c in data[key]]
                self._build_H()
                print(f"[Annotate] 캘리브레이션 로드: {key}")

    def _save_corners(self):
        data: dict = {}
        if _CORNER_JSON.exists():
            data = json.loads(_CORNER_JSON.read_text())
        data[self.video_path.name] = [[c[0], c[1]] for c in self.corners_px]
        _CORNER_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"[Annotate] 코너 저장: {_CORNER_JSON}")

    def _build_H(self):
        src = np.array(self.corners_px, dtype=np.float32)
        # 코트 범위 판단 (풀코트 vs 반코트) — 4점 모두 입력된 경우만
        if len(src) == 4:
            self.H, _ = cv2.findHomography(src, _COURT_FULL_M)
            self.calibrated = self.H is not None
            if self.calibrated:
                print("[Annotate] 캘리브레이션 완료 (풀코트 기준)")

    def _apply_default_corners(self):
        self.corners_px = [tuple(r) for r in _DEFAULT_CORNERS_WINTER.tolist()]
        self._build_H()
        self._save_corners()
        print("[Annotate] 기본 코너 적용 완료")

    # ── Homography ───────────────────────────────────────────────────
    def px_to_court(self, x: float, y: float) -> tuple[float, float]:
        if self.H is None:
            return (0.0, 0.0)
        pt = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), self.H)[0][0]
        # 팀별 범위 클램핑
        xlo, xhi = (0.0, 5.0) if self.team == "A" else (5.0, 10.0)
        cx = float(np.clip(pt[0], xlo + 0.05, xhi - 0.05))
        cy = float(np.clip(pt[1], 0.05, 5.95))
        return cx, cy

    def court_to_px_display(self, mx: float, my: float) -> tuple[int, int]:
        """코트 좌표 → 화면(960×540) 픽셀 (역투영)."""
        if self.H is None:
            return (0, 0)
        H_inv = np.linalg.inv(self.H)
        pt = cv2.perspectiveTransform(np.array([[[mx, my]]], np.float32), H_inv)[0][0]
        sx = int(pt[0] * _DISP_W / _CAM_W)
        sy = int(pt[1] * _DISP_H / _CAM_H)
        return sx, sy

    # ── Frame nav ────────────────────────────────────────────────────
    def _seek(self, idx: int):
        idx = max(0, min(self.total - 1, idx))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frm = self.cap.read()
        if ret:
            self.frame = frm.copy()
            self.frame_idx = idx

    def _advance(self, delta: int):
        self._seek(self.frame_idx + delta)

    # ── Annotation ops ───────────────────────────────────────────────
    def _add_point_from_display(self, disp_x: int, disp_y: int):
        """화면(960×540) 클릭 → 원본 픽셀 → 코트 좌표 → trace 추가."""
        orig_x = disp_x * _CAM_W / _DISP_W
        orig_y = disp_y * _CAM_H / _DISP_H
        mx, my = self.px_to_court(orig_x, orig_y)
        self.traces[self.active_pid].pts.append((mx, my))
        n = len(self.traces[self.active_pid].pts)
        print(f"[P{self.active_pid}] 점{n}: ({mx:.2f}m, {my:.2f}m) "
              f"← 픽셀({int(orig_x)},{int(orig_y)})")

    def _save_trace(self) -> bool:
        tr = self.traces[self.active_pid]
        if len(tr.pts) < 2:
            print(f"[P{self.active_pid}] 최소 2개 점 필요 (현재 {len(tr.pts)}개)")
            return False
        rows = tr.to_rows(self.pattern_id)
        self.saved.extend(rows)
        print(f"[P{self.active_pid}] 패턴 #{self.pattern_id} 저장 "
              f"({len(rows)}스텝, {tr.role}/{tr.context}/{tr.intent})")
        self.pattern_id += 1
        tr.pts.clear()
        return True

    def _clear_trace(self):
        self.traces[self.active_pid].pts.clear()
        print(f"[P{self.active_pid}] 흔적 초기화")

    def write_csv(self):
        if not self.saved:
            print("[Annotate] 저장할 데이터 없음")
            return
        is_new = not _OUT_CSV.exists()
        with _OUT_CSV.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            if is_new:
                w.writeheader()
            w.writerows(self.saved)
        n = len(self.saved)
        self.saved.clear()
        print(f"[Annotate] {n}행 저장 → {_OUT_CSV}")

    # ── Rendering ────────────────────────────────────────────────────
    def _render_calib(self) -> np.ndarray:
        disp = cv2.resize(self.frame, (_DISP_W, _DISP_H))
        labels = [
            "(1) 좌하단  near-left  (x=0m,  y=6m)",
            "(2) 우하단  near-right (x=10m, y=6m)",
            "(3) 우상단  far-right  (x=10m, y=0m)",
            "(4) 좌상단  far-left   (x=0m,  y=0m)",
        ]
        # 이미 찍은 코너 표시
        for i, (px, py) in enumerate(self.corners_px):
            sx = int(px * _DISP_W / _CAM_W)
            sy = int(py * _DISP_H / _CAM_H)
            cv2.circle(disp, (sx, sy), 8, (0, 255, 0), -1)
            cv2.putText(disp, str(i + 1), (sx + 10, sy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # 다음 클릭 안내
        n_done = len(self.corners_px)
        if n_done < 4:
            cv2.putText(disp, f">>> {labels[n_done]}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        else:
            cv2.putText(disp, "캘리브레이션 완료! 아무 키나 누르세요",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
            if self.calibrated:
                self._draw_grid_overlay(disp)
        cv2.putText(disp, "[D]=기본값 적용  [클릭]=코너 지정",
                    (20, _DISP_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (180, 180, 180), 1)
        return disp

    def _draw_grid_overlay(self, disp: np.ndarray):
        """역투영으로 코트 그리드를 카메라뷰에 표시."""
        if self.H is None:
            return
        H_inv = np.linalg.inv(self.H)
        for xm in np.arange(0, 10.5, 1.0):
            pts_m = np.array([[[xm, ym]] for ym in np.linspace(0, 6, 30)], np.float32)
            pts_px = cv2.perspectiveTransform(pts_m, H_inv)
            for j in range(len(pts_px) - 1):
                p0 = (int(pts_px[j][0][0] * _DISP_W / _CAM_W),
                      int(pts_px[j][0][1] * _DISP_H / _CAM_H))
                p1 = (int(pts_px[j+1][0][0] * _DISP_W / _CAM_W),
                      int(pts_px[j+1][0][1] * _DISP_H / _CAM_H))
                clr = (100, 255, 100) if xm == 5.0 else (50, 150, 50)
                thick = 2 if xm == 5.0 else 1
                cv2.line(disp, p0, p1, clr, thick)
        for ym in np.arange(0, 6.5, 1.0):
            pts_m = np.array([[[xm, ym]] for xm in np.linspace(0, 10, 30)], np.float32)
            pts_px = cv2.perspectiveTransform(pts_m, H_inv)
            for j in range(len(pts_px) - 1):
                p0 = (int(pts_px[j][0][0] * _DISP_W / _CAM_W),
                      int(pts_px[j][0][1] * _DISP_H / _CAM_H))
                p1 = (int(pts_px[j+1][0][0] * _DISP_W / _CAM_W),
                      int(pts_px[j+1][0][1] * _DISP_H / _CAM_H))
                cv2.line(disp, p0, p1, (50, 150, 50), 1)

    def _render_annotate(self) -> np.ndarray:
        """카메라뷰(960×540) + 버드아이뷰 + HUD."""
        cam = cv2.resize(self.frame, (_DISP_W, _DISP_H))
        if self.calibrated:
            self._draw_grid_overlay(cam)
        self._draw_traces_on_cam(cam)
        self._draw_active_player_label(cam)

        bird = self._render_birdeye()
        hud  = self._render_hud()

        # 우측 패널: bird-eye + HUD
        right_w = max(bird.shape[1], hud.shape[1])
        if bird.shape[1] < right_w:
            bird = np.hstack([bird, np.zeros((bird.shape[0], right_w - bird.shape[1], 3), np.uint8)])
        if hud.shape[1] < right_w:
            hud = np.hstack([hud, np.zeros((hud.shape[0], right_w - hud.shape[1], 3), np.uint8)])
        right = np.vstack([bird, hud])

        # 높이 맞추기
        if right.shape[0] < _DISP_H:
            pad = np.zeros((_DISP_H - right.shape[0], right.shape[1], 3), np.uint8)
            right = np.vstack([right, pad])
        elif right.shape[0] > _DISP_H:
            right = right[:_DISP_H]

        return np.hstack([cam, right])

    def _draw_traces_on_cam(self, disp: np.ndarray):
        """클릭 위치를 역투영해서 카메라뷰에 표시."""
        if self.H is None:
            return
        for pid, tr in self.traces.items():
            if not tr.pts:
                continue
            clr = _P_COLOR[pid]
            prev = None
            for i, (mx, my) in enumerate(tr.pts):
                sx, sy = self.court_to_px_display(mx, my)
                cv2.circle(disp, (sx, sy), 5, clr, -1)
                if prev:
                    cv2.arrowedLine(disp, prev, (sx, sy), clr, 2,
                                    tipLength=0.25, line_type=cv2.LINE_AA)
                prev = (sx, sy)
                # 점 번호
                cv2.putText(disp, str(i + 1), (sx + 6, sy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, clr, 1)
            # 마지막 점 하이라이트
            if pid == self.active_pid and tr.pts:
                sx, sy = self.court_to_px_display(*tr.pts[-1])
                cv2.circle(disp, (sx, sy), 10, clr, 2)

    def _draw_active_player_label(self, disp: np.ndarray):
        tr = self.traces[self.active_pid]
        clr = _P_COLOR[self.active_pid]
        txt = (f"P{self.active_pid}  {_ROLE_KO.get(tr.role,'?')} | "
               f"{_CTX_KO.get(tr.context,'?')} | "
               f"{_INTENT_KO.get(tr.intent,'?')}  ({len(tr.pts)}pts)")
        cv2.rectangle(disp, (0, 0), (len(txt) * 9 + 12, 26), (0, 0, 0), -1)
        cv2.putText(disp, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, clr, 1)
        # 프레임 정보
        secs = self.frame_idx / max(1, self.fps)
        ft = f"Frame {self.frame_idx}/{self.total}  ({secs:.1f}s / {secs/60:.1f}min)"
        cv2.putText(disp, ft, (6, _DISP_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    def _render_birdeye(self) -> np.ndarray:
        W = int(10.0 * _PX_M)   # 550px
        H = int(6.0  * _PX_M)   # 330px
        img = np.full((H, W, 3), (35, 45, 30), np.uint8)
        cv2.rectangle(img, (1, 1), (W - 2, H - 2), (200, 200, 200), 2)

        def m2px(mx: float, my: float) -> tuple[int, int]:
            return int(mx * _PX_M), int(my * _PX_M)

        # 팀 구분선 (x=5m)
        cv2.line(img, m2px(5.0, 0), m2px(5.0, 6.0), (100, 255, 100), 2)
        cv2.putText(img, "A", (int(2.5*_PX_M)-8, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 230, 100), 1)
        cv2.putText(img, "B", (int(7.5*_PX_M)-8, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 150, 255), 1)

        # 구역선 A팀: 1.5, 3.0 / B팀: 6.5, 8.5
        for xz in [1.5, 3.0, 6.5, 8.5]:
            cv2.line(img, m2px(xz, 0), m2px(xz, 6.0), (180, 180, 60), 1)
        for yz in [2.0, 4.0]:
            cv2.line(img, m2px(0, yz), m2px(10.0, yz), (60, 180, 60), 1)

        # x/y 숫자 눈금
        for xm in range(0, 11, 2):
            px = int(xm * _PX_M)
            cv2.putText(img, str(xm), (px + 2, H - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

        # 선수 흔적
        for pid, tr in self.traces.items():
            if not tr.pts:
                continue
            clr = _P_COLOR[pid]
            prev = None
            for i, (mx, my) in enumerate(tr.pts):
                px = m2px(mx, my)
                cv2.circle(img, px, 5 if pid != self.active_pid else 7, clr,
                           -1 if pid == self.active_pid else 1)
                if prev:
                    cv2.arrowedLine(img, prev, px, clr, 2,
                                    tipLength=0.25, line_type=cv2.LINE_AA)
                prev = px
            # 선수 번호 + 역할
            if tr.pts:
                lx, ly = m2px(*tr.pts[-1])
                cv2.putText(img, f"P{pid}", (lx + 6, ly - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, clr, 1)

        # 테두리 레이블
        cv2.putText(img, "Bird-eye  (m)", (4, H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)
        return img

    def _render_hud(self) -> np.ndarray:
        W = int(10.0 * _PX_M)
        lines = [
            f"팀: {self.team}  패턴: {self.pattern_id - self.base_pid}저장  미기록: {len(self.saved)}행",
        ]
        for pid in (1, 2, 3):
            tr = self.traces[pid]
            marker = "▶" if pid == self.active_pid else " "
            lines.append(
                f"{marker}P{pid} [{_ROLE_KO.get(tr.role,'?')}/"
                f"{_CTX_KO.get(tr.context,'?')}/"
                f"{_INTENT_KO.get(tr.intent,'?')}]  {len(tr.pts)}pts"
            )
        lines += [
            "─" * 35,
            "Space+30  . +5  , -5  F+1  B-1",
            "1/2/3=선수  T/D/M=역할  A/V/R=컨텍스트",
            "I/U=의도  클릭=마킹  Z=취소",
            "Enter=저장  C=초기화  S=CSV  Q=종료",
        ]
        hud_h = len(lines) * 20 + 10
        hud = np.full((hud_h, W, 3), (25, 25, 25), np.uint8)
        for i, line in enumerate(lines):
            cv2.putText(hud, line, (6, 18 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (0, 230, 100) if i == 0 else (180, 200, 180), 1)
        return hud

    # ── Mouse callback ───────────────────────────────────────────────
    def _mouse_cb(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # 카메라 뷰 영역(0..DISP_W)만 처리
        if x >= _DISP_W:
            return
        if not self.calibrated:
            # 캘리브레이션 클릭
            orig_x = x * _CAM_W / _DISP_W
            orig_y = y * _CAM_H / _DISP_H
            if len(self.corners_px) < 4:
                self.corners_px.append((orig_x, orig_y))
                print(f"[Calib] 코너 {len(self.corners_px)}: ({orig_x:.0f}, {orig_y:.0f})")
                if len(self.corners_px) == 4:
                    self._build_H()
                    self._save_corners()
        else:
            self._add_point_from_display(x, y)

    # ── Key handler ──────────────────────────────────────────────────
    def _handle_key(self, key: int) -> bool:
        if key == -1:
            return True
        if key == ord('q') or key == 27:
            if self.saved:
                self.write_csv()
            return False
        # 프레임 이동
        elif key == ord(' '):
            self._advance(30)
        elif key == ord('.'):
            self._advance(5)
        elif key == ord(','):
            self._advance(-5)
        elif key in (ord('f'), ord('F')):
            self._advance(1)
        elif key in (ord('b'), ord('B')):
            self._advance(-1)
        # 기본 코너 적용 (캘리브레이션 중)
        elif key == ord('d') and not self.calibrated:
            self._apply_default_corners()
        # 선수 선택
        elif key == ord('1'):
            self.active_pid = 1; print("[Annotate] P1 선택")
        elif key == ord('2'):
            self.active_pid = 2; print("[Annotate] P2 선택")
        elif key == ord('3'):
            self.active_pid = 3; print("[Annotate] P3 선택")
        # 역할
        elif key == ord('t'):
            self.traces[self.active_pid].role = "technician"
            print(f"[P{self.active_pid}] 역할 → 테크니션")
        elif key == ord('d') and self.calibrated:
            self.traces[self.active_pid].role = "defender"
            print(f"[P{self.active_pid}] 역할 → 디펜더")
        elif key == ord('m'):
            self.traces[self.active_pid].role = "main_attacker"
            print(f"[P{self.active_pid}] 역할 → 어태커")
        # 컨텍스트
        elif key == ord('a'):
            self.traces[self.active_pid].context = "attack"
            print(f"[P{self.active_pid}] 컨텍스트 → 공격")
        elif key == ord('v'):
            self.traces[self.active_pid].context = "defend"
            print(f"[P{self.active_pid}] 컨텍스트 → 수비")
        elif key == ord('r'):
            self.traces[self.active_pid].context = "transition"
            print(f"[P{self.active_pid}] 컨텍스트 → 전환")
        # 의도 순환
        elif key == ord('i'):
            self.intent_idx[self.active_pid] = (
                self.intent_idx[self.active_pid] + 1) % len(_INTENTS)
            self.traces[self.active_pid].intent = _INTENTS[self.intent_idx[self.active_pid]]
            print(f"[P{self.active_pid}] 의도 → {_INTENT_KO[self.traces[self.active_pid].intent]}")
        elif key == ord('u'):
            self.intent_idx[self.active_pid] = (
                self.intent_idx[self.active_pid] - 1) % len(_INTENTS)
            self.traces[self.active_pid].intent = _INTENTS[self.intent_idx[self.active_pid]]
            print(f"[P{self.active_pid}] 의도 → {_INTENT_KO[self.traces[self.active_pid].intent]}")
        # 마킹 / 저장
        elif key == ord('z'):
            if self.traces[self.active_pid].pts:
                self.traces[self.active_pid].pts.pop()
                print(f"[P{self.active_pid}] 마지막 점 취소")
        elif key == 13:  # Enter
            self._save_trace()
        elif key == ord('c'):
            self._clear_trace()
        elif key == ord('s'):
            self.write_csv()
        return True

    # ── Main loop ────────────────────────────────────────────────────
    def run(self):
        cv2.namedWindow("HADO Annotator", cv2.WINDOW_NORMAL)
        # 창 크기: 카메라(960) + 버드아이(550) = 1510 × 540+HUD
        cv2.resizeWindow("HADO Annotator", 1510, 600)
        cv2.setMouseCallback("HADO Annotator", self._mouse_cb)

        print("=" * 60)
        print("HADO 수동 어노테이션 도구")
        print(f"영상: {self.video_path.name}  ({self.total}프레임, {self.total/self.fps:.0f}초)")
        print(f"팀: {self.team}  |  패턴 시작 ID: {self.pattern_id}")
        print("=" * 60)
        if not self.calibrated:
            print(">>> 먼저 코트 모서리 4곳을 클릭하거나 [D]로 기본값 적용")

        while True:
            if not self.calibrated:
                display = self._render_calib()
            else:
                display = self._render_annotate()
            cv2.imshow("HADO Annotator", display)
            key = cv2.waitKey(20) & 0xFF
            if not self._handle_key(key):
                break

        cv2.destroyAllWindows()
        self.cap.release()

        if self.saved:
            print(f"\n[!] 미저장 {len(self.saved)}행이 있습니다. S로 저장하거나:")
            print(f"    툴 재실행 후 Q 누르면 자동 저장됩니다.")

        print(f"\n[완료] 출력: {_OUT_CSV}")
        print("\n── movement_data.csv 병합 방법 ──")
        print(f"  tail -n +2 {_OUT_CSV} >> data/movement_data.csv")
        print("  (헤더 1줄 제외 후 원본 파일에 append)")


def main():
    parser = argparse.ArgumentParser(description="HADO 수동 어노테이션 도구")
    parser.add_argument("--video", default=str(_DEFAULT_VIDEO),
                        help="어노테이션할 영상 경로")
    parser.add_argument("--team", default="A", choices=["A", "B"],
                        help="A=RGB(왼쪽, x=0-5m)  B=MAJOR(오른쪽, x=5-10m)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path
    if not video_path.exists():
        print(f"[오류] 영상 없음: {video_path}")
        return

    AnnotateTool(video_path, args.team).run()


if __name__ == "__main__":
    main()
