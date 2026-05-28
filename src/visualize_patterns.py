"""movement_data.csv → 패턴 시각화 PNG 생성.

포지션×컨텍스트 9개 조합별로 코트 위에 화살표를 겹쳐 그린 이미지를 저장.
추가로 패턴 ID별 개별 이미지도 저장.

실행:
    python -m src.visualize_patterns
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── 경로 설정 ──────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
INPUT_CSV  = ROOT / "data" / "movement_data.csv"
OUT_DIR    = ROOT / "data" / "pattern_snapshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 코트 상수 ─────────────────────────────────────────────
COURT_W_M = 10.0
COURT_H_M = 6.0
PX_PER_M  = 80
CW = int(COURT_W_M * PX_PER_M)   # 800
CH = int(COURT_H_M * PX_PER_M)   # 480

# ── 색상 (BGR) ────────────────────────────────────────────
BG_C     = (30,  40,  25)
LINE_C   = (220, 220, 220)
GRID_C   = (55,  65,  50)
CENTER_C = (160, 160,  50)
ZONE_C   = (100, 180, 100)
LANE_C   = (80,  80,  160)

PLAYER_COLORS = {
    1: (0,  110, 255),
    2: (0,  170, 220),
    3: (0,  200, 160),
    4: (255, 100,  0),
    5: (220, 150,  0),
    6: (180, 200,  0),
}

ROLE_EN  = {"technician": "Technician", "defender": "Defender",
            "main_attacker": "MainAttacker"}
CTX_EN   = {"attack": "Attack", "defend": "Defend", "transition": "Transition"}


# ── 한글 폰트 ─────────────────────────────────────────────
def _find_korean_font() -> str | None:
    for p in [
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]:
        if Path(p).exists():
            return p
    return None

_FONT_PATH = _find_korean_font()

def _kr_font(size: int):
    if _FONT_PATH:
        return ImageFont.truetype(_FONT_PATH, size)
    return ImageFont.load_default()

def put_text_kr(img: np.ndarray, text: str, xy: Tuple[int, int],
                size: int, color: Tuple[int, int, int]) -> None:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(xy, text, font=_kr_font(size),
                             fill=(color[2], color[1], color[0]))
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ── 코트 베이스 이미지 ────────────────────────────────────
def _make_court() -> np.ndarray:
    img = np.full((CH, CW, 3), BG_C, dtype=np.uint8)

    for xm in range(1, int(COURT_W_M)):
        xp = int(xm * PX_PER_M)
        cv2.line(img, (xp, 0), (xp, CH), GRID_C, 1, cv2.LINE_AA)
    for ym in range(1, int(COURT_H_M) + 1):
        yp = int(ym * PX_PER_M)
        if yp < CH:
            cv2.line(img, (0, yp), (CW, yp), GRID_C, 1, cv2.LINE_AA)

    cv2.line(img, (CW // 2, 0), (CW // 2, CH), CENTER_C, 2, cv2.LINE_AA)
    cv2.rectangle(img, (2, 2), (CW - 3, CH - 3), LINE_C, 2)

    for ym in [2.0, 4.0]:
        yp = int(ym * PX_PER_M)
        cv2.line(img, (0, yp), (CW, yp), LANE_C, 1, cv2.LINE_AA)
    for lane, ym_center in enumerate([1.0, 3.0, 5.0], start=1):
        yp = int(ym_center * PX_PER_M)
        cv2.putText(img, f"L{lane}", (CW - 28, yp + 5),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, LANE_C, 1, cv2.LINE_AA)

    for xm in [1.5, 3.0, 7.0, 8.5]:
        xp = int(xm * PX_PER_M)
        cv2.line(img, (xp, 0), (xp, CH), ZONE_C, 1, cv2.LINE_AA)
    for xm, label in [(0.75, "3"), (2.25, "2"), (4.0, "1"),
                      (6.0, "1"), (7.75, "2"), (9.25, "3")]:
        xp = int(xm * PX_PER_M) - 6
        cv2.putText(img, label, (xp, 22), cv2.FONT_HERSHEY_PLAIN,
                    1.2, ZONE_C, 1, cv2.LINE_AA)

    cv2.putText(img, "Team A", (10, CH - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 160, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "Team B", (CW - 80, CH - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 1, cv2.LINE_AA)
    return img


def _draw_arrow(img: np.ndarray, row: dict, alpha: float = 1.0):
    fx = int(float(row["from_x"]) * PX_PER_M)
    fy = int(float(row["from_y"]) * PX_PER_M)
    tx = int(float(row["to_x"]) * PX_PER_M)
    ty = int(float(row["to_y"]) * PX_PER_M)
    pid = int(row["player_id"])
    color = PLAYER_COLORS.get(pid, (200, 200, 200))
    cv2.arrowedLine(img, (fx, fy), (tx, ty), color, 2, cv2.LINE_AA, tipLength=0.25)
    cv2.circle(img, (fx, fy), 5, color, -1, cv2.LINE_AA)
    role_abbr = {"main_attacker": "M", "technician": "T", "defender": "D"}.get(row["role"], "?")
    cv2.putText(img, f"{pid}{role_abbr}", (fx + 7, fy - 4),
                cv2.FONT_HERSHEY_PLAIN, 0.9, color, 1, cv2.LINE_AA)


def _add_title(img: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    """상단 타이틀 바 추가."""
    bar_h = 44
    bar = np.full((bar_h, CW, 3), (20, 20, 30), dtype=np.uint8)
    put_text_kr(bar, title, (10, 6), 20, (220, 220, 220))
    if subtitle:
        put_text_kr(bar, subtitle, (10, 26), 14, (140, 140, 140))
    return np.vstack([bar, img])


def _load_csv() -> List[dict]:
    if not INPUT_CSV.exists():
        return []
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def generate_role_context_maps(rows: List[dict]) -> int:
    """포지션×컨텍스트 9개 조합 이미지 생성. 반환값: 생성된 파일 수."""
    grouped: dict[tuple, list] = defaultdict(list)
    for r in rows:
        grouped[(r["role"], r["context"])].append(r)

    count = 0
    for (role, ctx), group in sorted(grouped.items()):
        pattern_ids = sorted(set(r["pattern_id"] for r in group))
        img = _make_court()
        for r in group:
            _draw_arrow(img, r)

        role_ko = {"main_attacker": "메인공격수", "technician": "테크니션",
                   "defender": "디펜더"}.get(role, role)
        ctx_ko  = {"attack": "공격", "defend": "수비",
                   "transition": "전환"}.get(ctx, ctx)
        title    = f"{role_ko} — {ctx_ko}"
        subtitle = f"패턴 {len(pattern_ids)}개 | ID: {', '.join(pattern_ids[:10])}{'...' if len(pattern_ids)>10 else ''}"
        img = _add_title(img, title, subtitle)

        fname = f"{ROLE_EN.get(role, role)}_{CTX_EN.get(ctx, ctx)}.png"
        cv2.imwrite(str(OUT_DIR / fname), img)
        print(f"  저장: {fname}  ({len(pattern_ids)}패턴)")
        count += 1
    return count


def generate_per_pattern(rows: List[dict]) -> int:
    """패턴 ID별 개별 이미지 생성."""
    by_pattern: dict[str, list] = defaultdict(list)
    for r in rows:
        by_pattern[r["pattern_id"]].append(r)

    count = 0
    for pid, group in sorted(by_pattern.items(), key=lambda x: int(x[0])):
        img = _make_court()
        for r in group:
            _draw_arrow(img, r)
        role = group[0]["role"]
        ctx  = group[0]["context"]
        player = group[0]["player_id"]
        team   = group[0]["team"]
        role_ko = {"main_attacker": "메인공격수", "technician": "테크니션",
                   "defender": "디펜더"}.get(role, role)
        ctx_ko  = {"attack": "공격", "defend": "수비",
                   "transition": "전환"}.get(ctx, ctx)
        title    = f"패턴 {pid} | 선수{player}(팀{team}) — {role_ko} / {ctx_ko}"
        subtitle = f"{len(group)}개 동작"
        img = _add_title(img, title, subtitle)

        fname = f"pattern_{int(pid):03d}.png"
        cv2.imwrite(str(OUT_DIR / fname), img)
        count += 1
    return count


def main():
    rows = _load_csv()
    if not rows:
        print(f"[visualize] CSV 없음: {INPUT_CSV}")
        return

    print(f"[visualize] {len(rows)}행 로드 — 패턴 {len(set(r['pattern_id'] for r in rows))}개")
    print(f"[visualize] 저장 경로: {OUT_DIR}\n")

    print("[1/2] 포지션×컨텍스트 조합 이미지 생성...")
    n1 = generate_role_context_maps(rows)

    print(f"\n[2/2] 패턴별 개별 이미지 생성...")
    n2 = generate_per_pattern(rows)

    print(f"\n완료: 조합 {n1}개 + 개별 {n2}개 = 총 {n1+n2}개 PNG → {OUT_DIR}")


if __name__ == "__main__":
    main()
