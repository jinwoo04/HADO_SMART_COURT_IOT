"""Level 2 가이드 출력 — 시각 화살표 + 텍스트 + (선택) 음성.

- Bird-eye view 위에 현재→목표 화살표 (긴급도별 색)
- 카메라 프레임에 텍스트 상태 메시지
- pyttsx3로 한국어 음성 안내 (3초 이내 동일 메시지 무시)
"""
from __future__ import annotations

import threading
import time
from queue import Queue, Empty
from typing import List, Optional

import cv2
import numpy as np

from src.tactic_engine import TacticAdvice


# 긴급도별 색상 (BGR)
URGENCY_COLOR = {
    "LOW": (180, 220, 100),    # 연두
    "MID": (0, 220, 255),      # 노랑
    "HIGH": (0, 80, 255),      # 빨강
}

_RULE_KO = {
    "R1": "팀원 분산",
    "R2": "코트 커버",
    "R3": "측면 회피",
    "R4": "갭 공격",
    "R5": "수비 후퇴",
    "R6": "레인 커버",
    "BASE": "",
}


def _put_kr(img: np.ndarray, text: str, xy: tuple[int, int],
            size: int, color: tuple[int, int, int]) -> None:
    """한글 포함 텍스트 렌더링 (PIL 경유)."""
    try:
        from src.annotate import put_text_kr
        put_text_kr(img, text, xy, size, color)
    except Exception:
        cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                    size / 28, color, 1, cv2.LINE_AA)


def draw_guide_on_birdeye(
    court_img: np.ndarray,
    advices: List[TacticAdvice],
    px_per_m: int = 100,
) -> np.ndarray:
    """전술 분석 결과를 화살표 + 텍스트 패널로 표시.

    - MID urgency: 노란색 화살표 (current → target)
    - HIGH urgency: 빨간색 굵은 화살표 + 경고 링
    - 전술 규칙 내용은 우하단 텍스트 패널
    """
    out = court_img.copy()
    h_img, w_img = out.shape[:2]

    active = [a for a in advices if a.distance_m >= 0.20]
    if not active:
        return out

    # ── 전술 방향 화살표 (current_pos → target_pos) ──────────────
    for a in active:
        if a.urgency == "LOW":
            continue
        color = URGENCY_COLOR.get(a.urgency, (180, 180, 180))
        thick = 3 if a.urgency == "HIGH" else 2

        cx = int(a.current_pos[0] * px_per_m)
        cy = int(a.current_pos[1] * px_per_m)
        tx = int(a.target_pos[0] * px_per_m)
        ty = int(a.target_pos[1] * px_per_m)

        # 클램핑
        cx = max(2, min(w_img - 2, cx))
        cy = max(2, min(h_img - 2, cy))
        tx = max(2, min(w_img - 2, tx))
        ty = max(2, min(h_img - 2, ty))

        if (cx, cy) != (tx, ty):
            cv2.arrowedLine(out, (cx, cy), (tx, ty), color,
                            thickness=thick, tipLength=0.30, line_type=cv2.LINE_AA)
            # 목표 위치 작은 원
            cv2.circle(out, (tx, ty), 4, color, -1, cv2.LINE_AA)
            # 화살표 중간 지점에 규칙 레이블 (설명 가능성)
            if a.rule and a.rule != "BASE":
                mid_x = (cx + tx) // 2 + 4
                mid_y = (cy + ty) // 2 - 4
                mid_x = max(2, min(w_img - 20, mid_x))
                mid_y = max(8, min(h_img - 2, mid_y))
                cv2.putText(out, a.rule, (mid_x, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

        # HIGH urgency: 선수 위치에 경고 링 추가
        if a.urgency == "HIGH":
            cv2.circle(out, (cx, cy), 22, color, 2, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 16, color, 1, cv2.LINE_AA)

    _draw_tactic_panel(out, active)
    return out


def _draw_tactic_panel(img: np.ndarray, advices: List[TacticAdvice]) -> None:
    """우하단 전술 분석 패널 — 텍스트 전용."""
    h, w = img.shape[:2]

    # HIGH/MID 우선, 최대 4줄
    sorted_adv = sorted(
        advices,
        key=lambda a: ({"HIGH": 0, "MID": 1, "LOW": 2}.get(a.urgency, 3), -a.distance_m),
    )[:4]

    line_h = 17
    pad = 6
    panel_h = pad + len(sorted_adv) * line_h + pad
    panel_w = 190
    x0 = w - panel_w - 6
    y0 = h - panel_h - 6

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (70, 70, 70), 1)

    _put_kr(img, "전술 분석", (x0 + pad, y0 + pad - 2), 11, (180, 180, 180))
    cv2.line(img, (x0 + pad, y0 + pad + 10), (x0 + panel_w - pad, y0 + pad + 10),
             (50, 50, 50), 1)

    for i, a in enumerate(sorted_adv):
        color = URGENCY_COLOR.get(a.urgency, (200, 200, 200))
        rule_text = _RULE_KO.get(a.rule, a.rule)
        if not rule_text:
            continue
        iy = y0 + pad + 14 + i * line_h
        # ● 색점 (HIGH = 더 큰 원)
        r = 5 if a.urgency == "HIGH" else 3
        cv2.circle(img, (x0 + pad + 4, iy + 3), r, color, -1, cv2.LINE_AA)
        # 팀 + 규칙 + 이동거리
        dist_txt = f"{a.distance_m:.1f}m" if a.distance_m < 9.9 else ""
        txt = f"{'A' if a.team == 'A' else 'B'}팀 #{a.track_id}  {rule_text}"
        if dist_txt:
            txt += f"  {dist_txt}"
        _put_kr(img, txt, (x0 + pad + 14, iy - 1), 11, color)


_INTENT_KO = {
    "direct_attack":         "직접공격",
    "feint_attack":          "페인트",
    "cross_court":           "횡단",
    "gap_exploit":           "공간공략",
    "lure_attention":        "시선유도",
    "create_space":          "공간창출",
    "bait_inward":           "안쪽유도",
    "support_fire":          "공격지원",
    "shield_protect":        "수비쉴드",
    "shield_attack_support": "쉴드지원",
    "shield_feint":          "쉴드페인트",
    "counter_shield":        "맞쉴드",
}

_INTENT_COLOR = {
    "direct_attack":  (60, 80, 255),
    "feint_attack":   (30, 140, 255),
    "cross_court":    (80, 200, 255),
    "gap_exploit":    (0, 200, 160),
    "lure_attention": (180, 255, 80),
    "create_space":   (120, 255, 120),
    "bait_inward":    (200, 255, 100),
    "support_fire":   (255, 200, 60),
    "shield_protect": (255, 120, 60),
    "shield_attack_support": (255, 160, 80),
    "shield_feint":   (200, 100, 255),
    "counter_shield": (160, 80, 255),
}

_TEAM_COLOR = {
    "A": (100, 255, 150),
    "B": (100, 150, 255),
}

# 선수별 고정 색 (BGR)
_PLAYER_COLORS = [
    (0, 100, 255), (255, 100, 0), (0, 200, 100),
    (200, 0, 255), (0, 255, 255), (255, 0, 100),
]


def draw_player_overlays(
    court_img: np.ndarray,
    player_info: list[dict],
    px_per_m: int = 100,
) -> np.ndarray:
    """선수별 이동방향 화살표 + 의도 뱃지를 bird-eye view 위에 그린다.

    Parameters
    ----------
    player_info : list of dict, 각 원소:
        pid         int    선수 고유 ID (1–6)
        x, y        float  코트 좌표 (m)
        vx, vy      float  이동방향 단위벡터
        intent      str    현재 의도 키 (e.g. "direct_attack")
        role_ko     str    역할 한글 (e.g. "어태커")
    """
    out = court_img.copy()
    ARROW_LEN = int(0.6 * px_per_m)   # 화살표 길이 (0.6m 상당)

    for p in player_info:
        cx = int(p["x"] * px_per_m)
        cy = int(p["y"] * px_per_m)
        pid = p["pid"]
        p_color = _PLAYER_COLORS[(pid - 1) % len(_PLAYER_COLORS)]
        intent  = p.get("intent", "")
        vx, vy  = p.get("vx", 0.0), p.get("vy", 0.0)

        # 1) 이동방향 화살표 (선수 원에서 뻗는 굵은 화살표)
        moving = abs(vx) > 0.05 or abs(vy) > 0.05
        if moving:
            ex = int(cx + vx * ARROW_LEN)
            ey = int(cy + vy * ARROW_LEN)
            h_img, w_img = out.shape[:2]
            ex = max(2, min(w_img - 2, ex))
            ey = max(2, min(h_img - 2, ey))
            cv2.arrowedLine(out, (cx, cy), (ex, ey), p_color,
                            thickness=3, tipLength=0.35, line_type=cv2.LINE_AA)

        # 2) 의도 뱃지 — 선수 오른쪽 위
        intent_ko  = _INTENT_KO.get(intent, "")
        badge_color = _INTENT_COLOR.get(intent, (180, 180, 180))
        if intent_ko:
            bx = cx + 14
            by = cy - 14
            _put_kr(out, intent_ko, (bx, by), 12, badge_color)

    return out


def draw_guide_text_on_frame(
    frame: np.ndarray,
    advices: List[TacticAdvice],
    max_lines: int = 4,
) -> np.ndarray:
    """카메라 프레임 좌하단에 가이드 메시지 패널."""
    if not advices:
        return frame

    # 긴급도/거리 큰 순으로 정렬
    sorted_adv = sorted(
        advices,
        key=lambda a: (
            {"HIGH": 0, "MID": 1, "LOW": 2}.get(a.urgency, 3),
            -a.distance_m,
        ),
    )[:max_lines]

    h, w = frame.shape[:2]
    panel_h = 28 * len(sorted_adv) + 10
    panel_top = h - panel_h - 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, panel_top), (480, h - 8), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, a in enumerate(sorted_adv):
        y = panel_top + 22 + i * 28
        color = URGENCY_COLOR.get(a.urgency, (200, 200, 200))
        # 다국어 한글이 OpenCV 기본 폰트로는 안 나오므로 영문 매핑 사용
        text = f"#{a.track_id} [{a.team}] {a.urgency}: {a.rule}"
        cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 1, cv2.LINE_AA)

    return frame


# ---------- 음성 안내 (백그라운드 스레드) ----------
class VoiceGuide:
    """pyttsx3 기반 비동기 TTS.

    - 동일 메시지가 cooldown_sec 안에 또 들어오면 무시
    - 백그라운드 스레드로 동작하므로 메인 루프 블로킹 없음
    - pyttsx3가 없으면 자동으로 no-op (개발 환경 호환)
    """

    def __init__(self, cooldown_sec: float = 3.0, rate: int = 200, enabled: bool = True):
        self.cooldown_sec = cooldown_sec
        self.enabled = enabled
        self._engine = None
        self._queue: Queue[str] = Queue(maxsize=8)
        self._last_spoken: dict[str, float] = {}
        self._stop = False
        self._thread: Optional[threading.Thread] = None

        if not self.enabled:
            return
        try:
            import pyttsx3  # type: ignore
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", rate)
            self._select_korean_voice()
        except ImportError:
            print("[VoiceGuide] pyttsx3 없음 — 음성 비활성화")
            self.enabled = False
            return
        except Exception as e:
            print(f"[VoiceGuide] 엔진 초기화 실패: {e} — 음성 비활성화")
            self.enabled = False
            return

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print("[VoiceGuide] 음성 가이드 준비 완료")

    def _select_korean_voice(self) -> None:
        """한국어 음성 선택 — Yuna(Mac) > ko-KR compact > ko 포함 > 기본 유지."""
        voices = self._engine.getProperty("voices")
        selected: Optional[str] = None
        for v in voices:
            vid = (v.id or "").lower()
            vname = (v.name or "").lower()
            if "yuna" in vid:           # Mac 최고품질 한국어 compact
                selected = v.id
                break
            if selected is None and "ko-kr" in vid and "compact" in vid:
                selected = v.id
            elif selected is None and ("ko" in vid or "korean" in vname):
                selected = v.id
        if selected:
            self._engine.setProperty("voice", selected)
            print(f"[VoiceGuide] 한국어 음성: {selected.split('.')[-1]}")

    def speak(self, text: str):
        """음성 안내 큐에 추가. cooldown 내 동일 메시지는 무시."""
        if not self.enabled or not text:
            return
        now = time.time()
        last = self._last_spoken.get(text, 0)
        if now - last < self.cooldown_sec:
            return
        self._last_spoken[text] = now
        try:
            self._queue.put_nowait(text)
        except Exception:
            pass

    def speak_advices(self, advices: List[TacticAdvice]):
        """가장 긴급한 1개만 음성으로 출력 (음성 폭주 방지)."""
        candidates = [a for a in advices if a.voice_message and a.urgency in ("HIGH", "MID")]
        if not candidates:
            return
        # 우선순위 1: HIGH 먼저, 2: 이동거리 큼
        best = sorted(
            candidates,
            key=lambda a: (
                {"HIGH": 0, "MID": 1}.get(a.urgency, 2),
                -a.distance_m,
            ),
        )[0]
        self.speak(f"{best.track_id}번 {best.voice_message}")

    def _worker(self):
        while not self._stop:
            try:
                text = self._queue.get(timeout=0.5)
            except Empty:
                continue
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as e:
                print(f"[VoiceGuide] 출력 실패: {e}")

    def close(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=2.0)


def main():
    """단독 테스트: 더미 advice로 화살표 시각화."""
    from src.homography import compute_homography
    from src.tactic_engine import PlayerState, TacticEngine
    from src.visualizer import render_court_birdeye

    calib = compute_homography(
        corners_pixel=np.array([[100, 100], [1180, 100], [1180, 600], [100, 600]], dtype=np.float32),
        court_width_m=10.0, court_height_m=6.0, image_size=(1280, 720),
    )
    court = render_court_birdeye(calib, px_per_m=100)

    engine = TacticEngine()
    players = [
        PlayerState(track_id=1, court_x=1.5, court_y=1.0),
        PlayerState(track_id=2, court_x=1.6, court_y=1.1),   # 너무 가까움
        PlayerState(track_id=3, court_x=4.5, court_y=1.3),
        PlayerState(track_id=4, court_x=4.6, court_y=1.4),   # 너무 가까움
    ]
    advices = engine.analyze(players)

    rendered = draw_guide_on_birdeye(court, advices)
    out_path = "/tmp/guide_test.png"
    cv2.imwrite(out_path, rendered)
    print(f"가이드 시각화 저장: {out_path}")


if __name__ == "__main__":
    main()
