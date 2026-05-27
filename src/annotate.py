"""움직임 어노테이션 도구.

코트 버드아이 뷰에서 클릭으로 화살표를 그려 이상적인 이동 경로를 정의.
저장된 데이터는 data/movement_data.csv에 누적되며 MovementModel 학습에 사용.

조작법
------
좌클릭 1회  : 출발점 지정 (초록 원)
좌클릭 2회  : 도착점 지정 → 화살표 저장
우클릭      : 현재 출발점 취소
u           : 마지막 화살표 실행 취소
1~6         : 선수 번호 선택 (1-3=팀A, 4-6=팀B)
q / w / e   : 포지션 (q=메인공격수 / w=테크니션 / e=디펜더)
a / d / t   : 컨텍스트  (a=공격 / d=수비 / t=전환)
s           : CSV 저장 (자동 누적)
ESC         : 저장 후 종료
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── 한글 폰트 (macOS / Linux 자동 선택) ──────────────────────
def _find_korean_font() -> Optional[str]:
    candidates = [
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",   # macOS
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",       # Ubuntu
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None

_KOREAN_FONT_PATH = _find_korean_font()
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}

def _kr_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        if _KOREAN_FONT_PATH:
            _font_cache[size] = ImageFont.truetype(_KOREAN_FONT_PATH, size)
        else:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]

def put_text_kr(img: np.ndarray, text: str, xy: Tuple[int,int],
                size: int, color: Tuple[int,int,int]) -> None:
    """한글을 포함한 텍스트를 img에 in-place로 렌더링 (BGR)."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(xy, text, font=_kr_font(size),
                             fill=(color[2], color[1], color[0]))
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ── 코트 상수 ──────────────────────────────────────────────
COURT_W_M = 6.0
COURT_H_M = 2.66
PX_PER_M  = 160          # 960 × 426 px

CW = int(COURT_W_M * PX_PER_M)   # 960
CH = int(COURT_H_M * PX_PER_M)   # 426
STATUS_H = 130
WIN_W, WIN_H = CW, CH + STATUS_H

# ── 색상 (BGR) ─────────────────────────────────────────────
BG_C      = (30,  40,  25)
LINE_C    = (220, 220, 220)
GRID_C    = (55,  65,  50)
CENTER_C  = (160, 160,  50)
FROM_C    = (60,  255,  60)   # 출발점 — 밝은 초록
PENDING_C = (80,  200, 255)   # 확인 대기 중
STATUS_BG = (20,  28,  18)

# 선수별 색상 (1-3=팀A 주황 계열, 4-6=팀B 파랑 계열)
PLAYER_COLORS: dict[int, tuple[int, int, int]] = {
    1: (0,  110, 255),
    2: (0,  170, 220),
    3: (0,  200, 160),
    4: (255, 100,  0),
    5: (220, 150,  0),
    6: (180, 200,  0),
}

CONTEXT_MAP = {"a": "attack", "d": "defend", "t": "transition", "n": ""}
ROLE_MAP    = {"q": "main_attacker", "w": "technician", "e": "defender", "r": ""}
ROLE_KO     = {"main_attacker": "메인공격수", "technician": "테크니션",
               "defender": "디펜더", "": "미지정"}
OUTPUT_CSV = Path(__file__).resolve().parent.parent / "data" / "movement_data.csv"
CSV_HEADER = ["pattern_id", "step", "player_id", "team", "role",
              "from_x", "from_y", "to_x", "to_y", "context", "timestamp"]


# ── 데이터 ─────────────────────────────────────────────────
@dataclass
class Arrow:
    player_id: int
    from_m:   Tuple[float, float]
    to_m:     Tuple[float, float]
    role:     str = ""
    context:  str = ""

    @property
    def team(self) -> str:
        return "A" if self.player_id <= 3 else "B"


# ── 좌표 변환 ───────────────────────────────────────────────
def px_to_m(px: int, py: int) -> Tuple[float, float]:
    return round(px / PX_PER_M, 3), round(py / PX_PER_M, 3)


def m_to_px(mx: float, my: float) -> Tuple[int, int]:
    return int(mx * PX_PER_M), int(my * PX_PER_M)


# ── 코트 이미지 생성 ────────────────────────────────────────
def _make_court() -> np.ndarray:
    img = np.full((CH, CW, 3), BG_C, dtype=np.uint8)

    # 1m 그리드
    for xm in range(1, int(COURT_W_M)):
        xp = int(xm * PX_PER_M)
        cv2.line(img, (xp, 0), (xp, CH), GRID_C, 1, cv2.LINE_AA)
    for ym in range(1, int(COURT_H_M) + 1):
        yp = int(ym * PX_PER_M)
        if yp < CH:
            cv2.line(img, (0, yp), (CW, yp), GRID_C, 1, cv2.LINE_AA)

    # 중앙선
    cv2.line(img, (CW // 2, 0), (CW // 2, CH), CENTER_C, 2, cv2.LINE_AA)

    # 외곽선
    cv2.rectangle(img, (2, 2), (CW - 3, CH - 3), LINE_C, 2)

    # 좌표 레이블 (0, 1, 2, ..., 6 m)
    for xm in range(0, int(COURT_W_M) + 1):
        xp = int(xm * PX_PER_M)
        cv2.putText(img, f"{xm}m", (max(2, xp - 10), 14),
                    cv2.FONT_HERSHEY_PLAIN, 0.75, GRID_C, 1, cv2.LINE_AA)
    for ym in range(0, int(COURT_H_M) + 1):
        yp = int(ym * PX_PER_M)
        if yp < CH:
            cv2.putText(img, f"{ym}m", (3, min(CH - 4, yp + 12)),
                        cv2.FONT_HERSHEY_PLAIN, 0.75, GRID_C, 1, cv2.LINE_AA)

    # 팀 구역 레이블
    cv2.putText(img, "Team A", (20, CH - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 160, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "Team B", (CW - 100, CH - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 1, cv2.LINE_AA)
    return img


def _draw_arrow(img: np.ndarray, arrow: Arrow):
    fx, fy = m_to_px(*arrow.from_m)
    tx, ty = m_to_px(*arrow.to_m)
    color = PLAYER_COLORS.get(arrow.player_id, (200, 200, 200))
    cv2.arrowedLine(img, (fx, fy), (tx, ty), color, 2, cv2.LINE_AA, tipLength=0.25)
    cv2.circle(img, (fx, fy), 6, color, -1, cv2.LINE_AA)
    role_abbr = {"main_attacker": "M", "technician": "T", "defender": "D"}.get(arrow.role, "?")
    label = f"{arrow.player_id}{role_abbr}"
    cv2.putText(img, label, (fx + 7, fy - 5),
                cv2.FONT_HERSHEY_PLAIN, 1.0, color, 1, cv2.LINE_AA)


# ── 상태 패널 ───────────────────────────────────────────────
def _draw_status(canvas: np.ndarray, state: dict):
    y0 = CH
    canvas[y0:, :] = STATUS_BG

    player_id = state["player_id"]
    context   = state["context"] or "없음"
    arrows    = state["arrows"]
    hover_m   = state["hover_m"]
    pt_from   = state["pt_from"]
    mode      = state["mode"]

    color   = PLAYER_COLORS.get(player_id, (200, 200, 200))
    team    = "A" if player_id <= 3 else "B"
    role_ko = ROLE_KO.get(state.get("role", ""), "미지정")

    state_str = ("출발점 지정 중" if mode == "from"
                 else f"도착점 지정 중 (출발: {pt_from})")
    lines = [
        (f"선수: {player_id}  팀: {team}  포지션: {role_ko}  컨텍스트: {context}  저장: {len(arrows)}개", color),
        (f"마우스: ({hover_m[0]:.2f}, {hover_m[1]:.2f}) m", LINE_C),
        (f"상태: {state_str}", LINE_C),
        ("키: [1-6]선수  [q]메인공격수 [w]테크니션 [e]디펜더  [a/d/t]컨텍스트  [u]취소  [s]저장  [ESC]종료", LINE_C),
    ]
    for i, (line, clr) in enumerate(lines):
        put_text_kr(canvas, line, (10, y0 + 8 + i * 28), 16, clr)


# ── 메인 루프 ────────────────────────────────────────────────
def run():
    court_base = _make_court()
    arrows: List[Arrow] = []

    state = {
        "player_id":   1,
        "role":        "",
        "context":     "",
        "mode":        "from",      # "from" | "to"
        "pt_from":     None,        # (x_m, y_m)
        "hover_m":     (0.0, 0.0),
        "arrows":      arrows,
        "saved_count": 0,           # 마지막 저장 시점의 화살표 수
    }

    def on_mouse(event, x, y, flags, _param):
        # 마우스가 코트 영역 안에 있을 때만 좌표 갱신
        if 0 <= y < CH:
            state["hover_m"] = px_to_m(x, y)

        if event == cv2.EVENT_LBUTTONDOWN and 0 <= y < CH:
            mx, my = px_to_m(x, y)
            team = "A" if state["player_id"] <= 3 else "B"
            # 진영 불가침: 팀A는 x<3.0, 팀B는 x>3.0
            if (team == "A" and mx >= 3.0) or (team == "B" and mx <= 3.0):
                print(f"[Annotate] 진영 침범 불가 — 팀{team} 선수는 {'x<3.0' if team=='A' else 'x>3.0'} 영역만 가능")
                return
            if state["mode"] == "from":
                state["pt_from"] = (mx, my)
                state["mode"]    = "to"
            else:
                arrows.append(Arrow(
                    player_id=state["player_id"],
                    from_m=state["pt_from"],
                    to_m=(mx, my),
                    role=state["role"],
                    context=state["context"],
                ))
                state["pt_from"] = None
                state["mode"]    = "from"

        elif event == cv2.EVENT_RBUTTONDOWN:
            # 우클릭 → 현재 출발점 취소
            state["pt_from"] = None
            state["mode"]    = "from"

    win_name = "HADO 움직임 어노테이션"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, on_mouse)

    print("\n[Annotate] 시작 — 코트 창에서 클릭하여 화살표를 그리세요.")
    print(f"[Annotate] 저장 경로: {OUTPUT_CSV}\n")

    while True:
        canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        court  = court_base.copy()

        # 저장된 화살표 모두 그리기
        for arr in arrows:
            _draw_arrow(court, arr)

        # 출발점 대기 중인 점
        if state["pt_from"]:
            fx, fy = m_to_px(*state["pt_from"])
            cv2.circle(court, (fx, fy), 8, FROM_C, -1, cv2.LINE_AA)
            cv2.circle(court, (fx, fy), 8, LINE_C,  1, cv2.LINE_AA)

            # 출발점 → 마우스까지 미리보기 선
            mx, my = m_to_px(*state["hover_m"])
            cv2.line(court, (fx, fy), (mx, my), PENDING_C, 1, cv2.LINE_AA)

        # 마우스 십자선
        hx, hy = m_to_px(*state["hover_m"])
        if 0 <= hx < CW and 0 <= hy < CH:
            cv2.line(court, (hx, 0), (hx, CH), (80, 80, 80), 1)
            cv2.line(court, (0, hy), (CW, hy), (80, 80, 80), 1)

        canvas[:CH, :CW] = court
        _draw_status(canvas, state)
        cv2.imshow(win_name, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            _save_csv(arrows, state)
            break
        elif key == ord('s'):
            _save_csv(arrows, state)
        elif key == ord('u'):
            if state["mode"] == "to":
                state["pt_from"] = None
                state["mode"]    = "from"
            elif arrows:
                arrows.pop()
                print(f"[Annotate] 마지막 화살표 취소 (남은: {len(arrows)}개)")
        elif chr(key) in "123456":
            state["player_id"] = int(chr(key))
            print(f"[Annotate] 선수 {state['player_id']} 선택")
        elif chr(key) in ROLE_MAP:
            state["role"] = ROLE_MAP[chr(key)]
            print(f"[Annotate] 포지션: '{ROLE_KO[state['role']]}'")
        elif chr(key) in CONTEXT_MAP:
            state["context"] = CONTEXT_MAP[chr(key)]
            print(f"[Annotate] 컨텍스트: '{state['context'] or '없음'}'")

    cv2.destroyAllWindows()


def _next_pattern_id() -> int:
    """CSV에 저장된 마지막 pattern_id + 1을 반환."""
    if not OUTPUT_CSV.exists():
        return 1
    max_id = 0
    with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                max_id = max(max_id, int(row["pattern_id"]))
            except (KeyError, ValueError):
                pass
    return max_id + 1


def _save_csv(arrows: List[Arrow], state: dict):
    new_arrows = arrows[state["saved_count"]:]
    if not new_arrows:
        print("[Annotate] 새로 저장할 데이터 없음")
        return

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = OUTPUT_CSV.exists()
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    # 선수별로 묶어 각각 하나의 패턴으로 저장
    from collections import defaultdict
    by_player: dict[int, list[Arrow]] = defaultdict(list)
    for arr in new_arrows:
        by_player[arr.player_id].append(arr)

    next_id = _next_pattern_id()
    rows_written = 0
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(CSV_HEADER)
        for pid in sorted(by_player.keys()):
            pattern_arrows = by_player[pid]
            for step, arr in enumerate(pattern_arrows, start=1):
                writer.writerow([
                    next_id, step,
                    arr.player_id, arr.team, arr.role,
                    arr.from_m[0], arr.from_m[1],
                    arr.to_m[0],   arr.to_m[1],
                    arr.context,   ts,
                ])
            print(f"[Annotate] 패턴{next_id} | 선수{pid}({arr.role}) | {len(pattern_arrows)}개 동작")
            next_id += 1
            rows_written += len(pattern_arrows)

    state["saved_count"] = len(arrows)
    print(f"[Annotate] 총 {rows_written}개 행 저장 → {OUTPUT_CSV}")


def main():
    run()


if __name__ == "__main__":
    main()
