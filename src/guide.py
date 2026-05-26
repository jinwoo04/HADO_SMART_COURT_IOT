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
    "LOW": (180, 220, 100),    # 옅은 청록
    "MID": (0, 220, 255),      # 노랑
    "HIGH": (0, 80, 255),      # 빨강
}


def draw_guide_on_birdeye(
    court_img: np.ndarray,
    advices: List[TacticAdvice],
    px_per_m: int = 100,
) -> np.ndarray:
    """Bird-eye view 위에 추천 위치 + 이동 화살표 표시."""
    out = court_img.copy()

    for a in advices:
        if a.distance_m < 0.15:
            continue  # 이동 권장 없음

        color = URGENCY_COLOR.get(a.urgency, (200, 200, 200))

        cx = int(a.current_pos[0] * px_per_m)
        cy = int(a.current_pos[1] * px_per_m)
        tx = int(a.target_pos[0] * px_per_m)
        ty = int(a.target_pos[1] * px_per_m)

        h, w = out.shape[:2]
        # 코트 안쪽으로 클램프
        tx = max(4, min(w - 4, tx))
        ty = max(4, min(h - 4, ty))

        # 추천 위치 X 마크
        cv2.drawMarker(out, (tx, ty), color, markerType=cv2.MARKER_TILTED_CROSS,
                       markerSize=18, thickness=2)
        # 점선 효과의 두꺼운 화살표
        cv2.arrowedLine(out, (cx, cy), (tx, ty), color,
                        thickness=3, tipLength=0.25, line_type=cv2.LINE_AA)

        # 긴급도 HIGH면 추천 위치에 펄스 원
        if a.urgency == "HIGH":
            cv2.circle(out, (tx, ty), 22, color, 2, cv2.LINE_AA)

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

    def __init__(self, cooldown_sec: float = 3.0, rate: int = 180, enabled: bool = True):
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
            # 한국어 음성이 시스템에 있으면 사용 시도
            for v in self._engine.getProperty("voices"):
                if "ko" in (v.id or "").lower() or "korean" in (v.name or "").lower():
                    self._engine.setProperty("voice", v.id)
                    break
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
        court_width_m=6.0, court_height_m=2.66, image_size=(1280, 720),
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
