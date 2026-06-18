"""영상 분석 기반 패턴 추출 및 시각화.

각 패턴을 개별 PNG로 저장:
  data/proposed_patterns/<intent>/<번호>_<설명>.png
  - 코트 다이어그램 (화살표 + 스텝 번호)
  - 우측 하단: 참고 게임 프레임 썸네일

실행:
    python -m src.extract_patterns
"""
from __future__ import annotations

import csv
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── 프로젝트 경로 ──
ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = ROOT / "data" / "analysis_frames"
OUT_DIR    = ROOT / "data" / "proposed_patterns"
CSV_PATH   = ROOT / "data" / "movement_data.csv"

PX = 80
COURT_W, COURT_H = 10.0, 6.0
CW, CH = int(COURT_W * PX), int(COURT_H * PX)
HEADER_H = 48
THUMB_W, THUMB_H = 280, 158

# ── 색상 (BGR) ──
INTENT_COLORS = {
    "direct_attack":         (60,  200, 255),
    "feint_attack":          (0,   180, 255),
    "cross_court":           (50,  255, 200),
    "gap_exploit":           (180, 255,  50),
    "lure_attention":        (255, 200,  50),
    "create_space":          (255, 160,  30),
    "bait_inward":           (30,  140, 255),
    "support_fire":          (180, 100, 255),
    "shield_protect":        (0,   255, 120),
    "shield_attack_support": (0,   220, 180),
    "shield_feint":          (80,  255, 160),
    "counter_shield":        (50,  160, 255),
}
INTENT_KO = {
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
ROLE_KO = {
    "main_attacker": "어태커",
    "technician":    "테크니션",
    "defender":      "디펜더",
}
TEAM_CLR = {"A": (80, 120, 255), "B": (255, 140, 60)}

# ── 영상 분석 기반 패턴 정의 ──────────────────────────────────
# (video, ref_frame, player_id, role, team, context, intent, desc, steps)
PATTERNS = [
    # ── VIDEO 1: 1번 어태커·3번 어태커 직접공격, 4번 어태커 공간공략 ──
    (1,"f0130.jpg",1,"main_attacker","A","attack","direct_attack",
     "1번 어태커: 1선 전진 후 대각 퇴각",
     [((1.0,1.5),(3.0,1.5)),((3.0,1.5),(1.5,2.8))]),

    (1,"f0130.jpg",3,"main_attacker","A","attack","direct_attack",
     "3번 어태커: 2선 전진 공격",
     [((1.0,4.8),(2.5,4.5)),((2.5,4.5),(3.2,3.8))]),

    (1,"f0200.jpg",4,"main_attacker","B","attack","gap_exploit",
     "4번 어태커: 중앙 빈 공간 전진 후 복귀",
     [((8.5,2.5),(7.2,2.0)),((7.2,2.0),(6.5,3.2)),((6.5,3.2),(8.0,3.0))]),

    # ── VIDEO 2: 디펜더 수비→공격쉴드 전환 ──
    (2,"f0150.jpg",5,"defender","B","attack","shield_attack_support",
     "5번 디펜더: 전방 전진해 공격용 쉴드 전개",
     [((8.5,2.8),(7.5,2.5)),((7.5,2.5),(6.8,1.5))]),

    (2,"f0100.jpg",4,"technician","B","attack","create_space",
     "4번 테크니션: 바깥 레인 이동 공간창출",
     [((8.0,3.5),(7.5,1.2)),((7.5,1.2),(8.5,0.8))]),

    (2,"f0200.jpg",6,"main_attacker","B","attack","direct_attack",
     "6번 어태커: 공격쉴드 뒤에서 직접공격",
     [((9.0,3.0),(7.5,2.8)),((7.5,2.8),(6.5,2.0))]),

    # ── VIDEO 3: 쉴드 깨질 때 전진 공격 ──
    (3,"f0080.jpg",1,"main_attacker","A","attack","direct_attack",
     "1번 어태커: 상대 쉴드 깨지는 순간 전진",
     [((1.2,1.5),(2.8,1.5)),((2.8,1.5),(3.5,2.2)),((3.5,2.2),(1.8,2.5))]),

    (3,"f0170.jpg",3,"main_attacker","A","attack","gap_exploit",
     "3번 어태커: 2선까지 전진 공격 지원",
     [((1.0,4.5),(2.5,4.0)),((2.5,4.0),(3.0,3.5))]),

    (3,"f0200.jpg",5,"defender","B","attack","shield_attack_support",
     "5번 디펜더: 추가 공격쉴드로 6번 지원",
     [((7.5,4.0),(7.0,3.5)),((7.0,3.5),(6.5,2.8))]),

    (3,"f0250.jpg",6,"main_attacker","B","attack","direct_attack",
     "6번 어태커: 공격쉴드 뒤 직접공격 득점",
     [((8.8,2.5),(7.8,2.5)),((7.8,2.5),(6.8,2.0))]),

    # ── VIDEO 4: 1번 어태커 코트 가로지르기 ──
    (4,"f0220.jpg",1,"main_attacker","A","attack","cross_court",
     "1번 어태커: 쉴드 방향 따라 반대 레인으로 가로지르기",
     [((1.5,1.0),(3.0,1.5)),((3.0,1.5),(3.5,3.0)),((3.5,3.0),(2.0,4.8))]),

    (4,"f0220.jpg",2,"technician","A","attack","create_space",
     "2번 테크니션: 어태커 가로지르기 위해 비켜주는 공간창출",
     [((1.5,3.0),(1.2,1.5))]),

    (4,"f0280.jpg",3,"main_attacker","A","attack","direct_attack",
     "3번 어태커: 자신의 쉴드 사용하며 직접공격",
     [((1.0,4.8),(2.5,4.5)),((2.5,4.5),(3.5,4.0)),((3.5,4.0),(2.0,3.0))]),

    (4,"f0300.jpg",5,"defender","B","attack","shield_attack_support",
     "5번 디펜더: 1번 어태커 이동 지원 추가 쉴드",
     [((8.5,1.5),(7.8,2.5)),((7.8,2.5),(7.0,3.5))]),

    # ── VIDEO 5: 마지막 득점 압박 ──
    (5,"f0320.jpg",1,"main_attacker","A","attack","direct_attack",
     "1번 어태커: 마지막 전방 압박 직접공격",
     [((1.2,1.5),(3.5,1.8)),((3.5,1.8),(4.2,2.5)),((4.2,2.5),(2.5,2.0))]),

    (5,"f0360.jpg",2,"technician","A","attack","lure_attention",
     "2번 테크니션: 마지막 상황 지그재그 시선유도",
     [((1.5,3.0),(2.5,2.0)),((2.5,2.0),(1.8,4.0)),((1.8,4.0),(2.8,3.0))]),

    (5,"f0320.jpg",3,"main_attacker","A","attack","gap_exploit",
     "3번 어태커: 마지막 공간 활용 전진",
     [((1.0,5.0),(2.8,4.5)),((2.8,4.5),(3.8,3.8))]),

    # ── VIDEO 6: JINU(4) 슬라이딩 직접공격 ──
    (6,"f0210.jpg",4,"main_attacker","B","attack","direct_attack",
     "JINU(4): 슬라이딩 전진 직접공격 후 후퇴",
     [((8.5,2.5),(7.0,2.2)),((7.0,2.2),(6.2,2.8)),((6.2,2.8),(8.0,2.0))]),

    (6,"f0300.jpg",2,"defender","A","defend","shield_protect",
     "URI(2) 디펜더: 어태커 지원 수비 포지션",
     [((1.8,3.0),(1.5,1.5)),((1.5,1.5),(2.0,0.8))]),

    (6,"f0420.jpg",1,"main_attacker","A","attack","feint_attack",
     "BAXTER(1): 페인트 후 실제 공격 각도 전환",
     [((1.0,1.5),(2.5,1.5)),((2.5,1.5),(2.0,3.0)),((2.0,3.0),(3.5,2.5))]),

    # ── VIDEO 7: 팀 연계 플레이 ──
    (7,"f0100.jpg",4,"main_attacker","B","attack","gap_exploit",
     "JINU(4): 공간창출 이동으로 수비 시선 분산",
     [((8.0,3.0),(7.0,1.5)),((7.0,1.5),(6.5,0.8)),((6.5,0.8),(7.5,2.0))]),

    (7,"f0200.jpg",2,"defender","A","attack","shield_attack_support",
     "URI(2) 디펜더: 어태커 앞에 공격쉴드 전개",
     [((1.5,4.0),(2.5,3.5)),((2.5,3.5),(3.5,3.0))]),

    (7,"f0350.jpg",1,"main_attacker","A","attack","direct_attack",
     "BAXTER(1): 공격쉴드 뒤에서 직접공격 득점",
     [((1.2,3.5),(2.5,3.2)),((2.5,3.2),(3.8,2.8)),((3.8,2.8),(2.0,2.5))]),

    (7,"f0250.jpg",3,"technician","A","attack","support_fire",
     "TAEOH(3) 테크니션: 어태커 공격 시 앞 지원",
     [((1.0,1.5),(2.2,2.0)),((2.2,2.0),(3.0,1.8))]),

    (7,"f0300.jpg",2,"defender","A","attack","shield_attack_support",
     "URI(2) 디펜더: 1번 어태커 앞 추가 쉴드",
     [((2.5,2.5),(3.2,2.0)),((3.2,2.0),(4.0,1.8))]),

    # ── VIDEO 8: 이중 쉴드 + 1선 좌우 공격 ──
    (8,"f0080.jpg",2,"defender","A","attack","shield_attack_support",
     "URI(2) 디펜더: 어태커 앞 이중 공격쉴드",
     [((1.5,2.0),(2.5,1.5)),((2.5,1.5),(3.5,1.2))]),

    (8,"f0120.jpg",4,"main_attacker","B","attack","shield_attack_support",
     "JINU(4): 빠른 전진 공격쉴드 전개",
     [((9.0,3.5),(7.5,3.0)),((7.5,3.0),(6.5,2.5))]),

    (8,"f0150.jpg",1,"main_attacker","A","attack","direct_attack",
     "BAXTER(1): 1선에서 좌우 이동하며 직접공격",
     [((1.2,2.0),(3.2,1.5)),((3.2,1.5),(3.5,3.0)),
      ((3.5,3.0),(3.2,4.5)),((3.2,4.5),(1.5,3.0))]),

    (8,"f0140.jpg",6,"main_attacker","B","attack","gap_exploit",
     "MIN(6): 중앙 공격쉴드로 공간공략",
     [((9.2,4.5),(8.0,4.0)),((8.0,4.0),(6.8,3.5))]),

    # ── VIDEO 9: 주공격수·공격지원 ──
    (9,"f0080.jpg",6,"main_attacker","B","attack","gap_exploit",
     "MIN(6): 공간활용 전진 공격지원",
     [((9.0,4.5),(7.5,4.0)),((7.5,4.0),(6.8,3.2))]),

    (9,"f0120.jpg",4,"main_attacker","B","attack","direct_attack",
     "JINU(4): 1선 좌우 이동 직접공격(주공격수)",
     [((8.5,2.0),(7.0,1.8)),((7.0,1.8),(6.8,3.5)),
      ((6.8,3.5),(7.5,4.5)),((7.5,4.5),(8.5,3.5))]),

    (9,"f0140.jpg",5,"technician","B","defend","lure_attention",
     "JAY(5) 테크니션: 상대 시선 유도 좌우 이동",
     [((8.2,1.5),(7.5,3.0)),((7.5,3.0),(8.5,4.5))]),

    # ── VIDEO 10: 종합 팀플레이 (JINU 주도) ──
    (10,"f0080.jpg",4,"main_attacker","B","attack","direct_attack",
     "JINU(4): 직접공격 시도",
     [((8.5,2.0),(7.0,1.8)),((7.0,1.8),(6.5,2.5)),((6.5,2.5),(8.0,2.2))]),

    (10,"f0240.jpg",4,"main_attacker","B","attack","cross_court",
     "JINU(4): 쉴드에 막혀 반대 레인으로 코트 가로지르기",
     [((7.5,1.5),(8.0,3.0)),((8.0,3.0),(7.0,4.5)),((7.0,4.5),(6.5,5.0))]),

    (10,"f0400.jpg",4,"main_attacker","B","attack","bait_inward",
     "JINU(4): 팀원 공격 지원 앞뒤 시선유도",
     [((6.8,4.5),(5.5,4.0)),((5.5,4.0),(7.0,4.8)),((7.0,4.8),(5.8,4.2))]),

    (10,"f0480.jpg",3,"technician","A","attack","support_fire",
     "TAEOH(3): 지그재그로 전진하며 공격지원",
     [((1.0,1.5),(2.0,2.5)),((2.0,2.5),(2.8,1.5)),
      ((2.8,1.5),(3.5,2.5)),((3.5,2.5),(4.2,1.8))]),

    (10,"f0240.jpg",2,"defender","A","defend","counter_shield",
     "URI(2) 디펜더: 상대 쉴드 전개 위치 앞 맞쉴드",
     [((1.5,3.5),(2.8,4.0)),((2.8,4.0),(4.0,4.5))]),

    (10,"f0350.jpg",1,"main_attacker","A","attack","direct_attack",
     "BAXTER(1): 1선에서 쉴드 사용하며 직접공격",
     [((1.2,2.5),(3.0,2.5)),((3.0,2.5),(3.5,1.5)),((3.5,1.5),(2.0,2.0))]),

    (10,"f0480.jpg",3,"technician","A","attack","bait_inward",
     "TAEOH(3): 안쪽 유도 후 외곽 탈출",
     [((2.0,1.2),(3.2,2.8)),((3.2,2.8),(2.5,4.5))]),

    (10,"f0560.jpg",2,"defender","A","attack","shield_attack_support",
     "URI(2) 디펜더: 1번 어태커 앞 공격쉴드 추가",
     [((2.0,2.5),(3.0,2.0)),((3.0,2.0),(4.0,1.8))]),
]


# ── 헬퍼 ─────────────────────────────────────────────────────

def _make_court() -> np.ndarray:
    img = np.full((CH, CW, 3), (30, 40, 25), dtype=np.uint8)
    for xm in range(1, int(COURT_W)):
        cv2.line(img, (xm*PX, 0), (xm*PX, CH), (55,65,50), 1)
    for ym in range(1, int(COURT_H)+1):
        yp = ym*PX
        if yp < CH:
            cv2.line(img, (0, yp), (CW, yp), (55,65,50), 1)
    # 중앙선
    cv2.line(img, (CW//2, 0), (CW//2, CH), (160,160,50), 2)
    # 레인선
    for ym in [2.0, 4.0]:
        cv2.line(img, (0, int(ym*PX)), (CW, int(ym*PX)), (80,80,160), 1)
    # 구역선
    for xm in [1.5, 3.0, 7.0, 8.5]:
        cv2.line(img, (int(xm*PX), 0), (int(xm*PX), CH), (80,130,80), 1)
    cv2.rectangle(img, (2,2), (CW-3, CH-3), (220,220,220), 2)
    return img


def _kr_font(size: int):
    paths = [
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def put_kr(img: np.ndarray, text: str, xy, size: int, color):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(xy, text, font=_kr_font(size),
                             fill=(color[2], color[1], color[0]))
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _draw_pattern(court: np.ndarray, steps, color, pid: int):
    for i, (fm, tm) in enumerate(steps):
        fx, fy = int(fm[0]*PX), int(fm[1]*PX)
        tx, ty = int(tm[0]*PX), int(tm[1]*PX)
        cv2.arrowedLine(court, (fx,fy),(tx,ty), color, 3,
                        cv2.LINE_AA, tipLength=0.25)
        cv2.circle(court, (fx,fy), 7, color, -1, cv2.LINE_AA)
        cv2.circle(court, (fx,fy), 7, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(court, str(i+1), (fx-4, fy+5),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, (0,0,0), 1)
    # 마지막 도착점
    lx, ly = int(steps[-1][1][0]*PX), int(steps[-1][1][1]*PX)
    cv2.drawMarker(court, (lx,ly), color,
                   cv2.MARKER_TILTED_CROSS, 16, 2)


def _load_ref(video: int, frame: str) -> np.ndarray | None:
    p = FRAMES_DIR / str(video) / frame
    if not p.exists():
        # 가장 가까운 프레임 찾기
        frames = sorted((FRAMES_DIR / str(video)).glob("f*.jpg"))
        if frames:
            p = frames[len(frames)//2]
        else:
            return None
    img = cv2.imread(str(p))
    if img is None:
        return None
    return cv2.resize(img, (THUMB_W, THUMB_H))


def _next_pattern_id() -> int:
    if not CSV_PATH.exists():
        return 1
    max_id = 0
    for row in csv.DictReader(CSV_PATH.open(encoding="utf-8")):
        try:
            max_id = max(max_id, int(row["pattern_id"]))
        except (KeyError, ValueError):
            pass
    return max_id + 1


# ── 메인 ─────────────────────────────────────────────────────

def generate():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 의도별 서브폴더
    intents = set(p[6] for p in PATTERNS)
    for it in intents:
        (OUT_DIR / it).mkdir(exist_ok=True)

    court_base = _make_court()
    total = len(PATTERNS)

    # 의도별 카운터
    cnt: dict[str, int] = defaultdict(int)

    for i, pat in enumerate(PATTERNS):
        vid, ref, pid, role, team, ctx, intent, desc, steps = pat
        cnt[intent] += 1
        idx = cnt[intent]
        color = INTENT_COLORS.get(intent, (200,200,200))

        # ── 캔버스 구성 ──
        canvas_h = CH + HEADER_H
        canvas = np.zeros((canvas_h, CW, 3), dtype=np.uint8)

        # 헤더 바
        canvas[:HEADER_H, :] = (20, 28, 18)
        team_clr = TEAM_CLR[team]
        role_ko  = ROLE_KO.get(role, role)
        intent_ko = INTENT_KO.get(intent, intent)
        title = f"영상{vid}  선수{pid}({team}팀 {role_ko})  [{intent_ko}]  {desc}"
        put_kr(canvas, title, (8, 10), 15, (220, 220, 220))

        # 코트
        court = court_base.copy()

        # 팀 표시선 (반투명 오버레이)
        overlay = court.copy()
        if team == "A":
            cv2.rectangle(overlay, (0,0),(CW//2, CH),(40,60,100),-1)
        else:
            cv2.rectangle(overlay, (CW//2,0),(CW,CH),(80,50,30),-1)
        cv2.addWeighted(overlay, 0.12, court, 0.88, 0, court)

        _draw_pattern(court, steps, color, pid)

        # 시작 위치 레이블
        sx, sy = int(steps[0][0][0]*PX), int(steps[0][0][1]*PX)
        label = f"P{pid}"
        cv2.putText(court, label, (sx+8, sy-8),
                    cv2.FONT_HERSHEY_PLAIN, 1.1, color, 1, cv2.LINE_AA)

        # 참고 프레임 썸네일 (우측 하단)
        thumb = _load_ref(vid, ref)
        if thumb is not None:
            ty0 = CH - THUMB_H
            tx0 = CW - THUMB_W
            # 반투명 테두리
            cv2.rectangle(court, (tx0-2, ty0-2),
                          (CW-1, CH-1), color, 2)
            court[ty0:CH, tx0:CW] = thumb
            cv2.putText(court, f"Vid{vid} ref",
                        (tx0+4, ty0+14), cv2.FONT_HERSHEY_PLAIN,
                        0.85, (200,200,200), 1)

        canvas[HEADER_H:, :] = court

        # 의도 색상 바 (헤더 왼쪽)
        cv2.rectangle(canvas, (0,0),(5,HEADER_H), color, -1)

        # ── 저장 ──
        safe_desc = desc[:30].replace(" ","_").replace(":","").replace("(","").replace(")","")
        fname = f"{idx:02d}_v{vid}_p{pid}_{safe_desc}.png"
        fpath = OUT_DIR / intent / fname
        cv2.imwrite(str(fpath), canvas)

        print(f"[{i+1:2}/{total}] {intent}/{fname}")

    print(f"\n완료 — {total}개 PNG → {OUT_DIR}")
    _print_summary()


def _print_summary():
    from collections import Counter
    cnt = Counter(p[6] for p in PATTERNS)
    print("\n=== 의도별 제안 패턴 수 ===")
    for intent, n in sorted(cnt.items(), key=lambda x: -x[1]):
        bar = "█" * n
        ko  = INTENT_KO.get(intent, intent)
        print(f"  {ko:12} ({intent:25}) {n:2}개  {bar}")
    print(f"\n총 {sum(cnt.values())}개 패턴 제안")


if __name__ == "__main__":
    generate()
