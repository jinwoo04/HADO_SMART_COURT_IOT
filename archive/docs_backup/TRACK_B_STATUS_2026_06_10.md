# Track B 현황 보고 — 2026-06-10

병렬 세션 확인 기준 상태 스냅샷.

---

## 1. 모델 리더보드

| Run | Score | Precision | Recall | mAP50 | 이미지 수 |
|-----|------:|----------:|-------:|------:|--------:|
| **trackb_v4_player_only** ← 현재 최고 | **53.34** | 0.676 | 0.648 | **0.653** | 87 |
| trackb_v3_player_only | 50.77 | 0.712 | 0.660 | 0.622 | 87 |
| trackb_v2_player_only | 36.30 | 0.822 | 0.432 | 0.518 | 41 |
| trackb_v1_player_only | 37.09 | 0.015 | 0.838 | 0.495 | 36 |

**v4 checkpoint**: `outputs/track_b_retrain_runs/trackb_v4_player_only/train/yolov8n_player_only/weights/best.pt`

---

## 2. batch02 평가 결과

- 영상: 10개 (`data/drive_imports/batch02_raw/match01~10.mp4`)
- 처리 프레임: 994개
- Hard frames: **130개** 발굴

### 매치별 심각도

| 매치 | Severity | Hard | Over-det | Under-det | 주 원인 |
|------|--------:|-----:|---------:|----------:|--------|
| match08 | **53** | 18 | 17 | 0 | AR 이펙트 중첩 (최대 13명 검출) |
| match02 | **43** | 16 | 13 | 2 | 고밀도 + miss 혼재 |
| match03 | 42 | 14 | 14 | 0 | AR 이펙트 overlap |
| match04 | 42 | 14 | 14 | 0 | AR 이펙트 overlap |
| match09 | 42 | 14 | 14 | 0 | AR 이펙트 overlap |
| match10 | 41 | 14 | 13 | 0 | AR 이펙트 overlap |
| match01 | 33 | 12 |  9 | 1 | 혼재 |
| match05 | 33 | 11 | 11 | 0 | AR 이펙트 overlap |
| match06 | 30 | 10 | 10 | 0 | AR 이펙트 overlap |
| match07 | **19** |  7 |  5 | 0 | 상대적으로 양호 |

**결론**: 문제의 97%는 over-detection. AR shield/effect가 player bbox로 잡히는 것이 핵심 원인.

---

## 3. Roboflow 업로드 번들

**v2 bundle (권장)**: 40장, 매치당 4장씩 균등 선발, 전부 `label_priority`

```
data/track_b_batch_review/batch02_v4_eval/roboflow_upload_bundle_v2_playable_bias.zip
```

| 검출 수 | 프레임 수 | 의미 |
|--------|--------:|------|
| 10개 | 24 | AR 이펙트 1~2개 overlap |
| 11개 | 9 | AR 이펙트 2~3개 overlap |
| 12개 | 5 | 심한 overlap |
| 13-14개 | 2 | 최악 케이스 |

**라벨링 지침**:
- 선수(player) bbox만 — effect/shield 절대 박스 금지
- 박스는 body에 타이트하게 (AR glow 포함 금지)
- 화면이 intro/scoreboard면 skip
- 겹치는 선수는 각각 개별 박스

---

## 4. 다음 루프: v5 준비

### 사람이 해야 할 것 (Roboflow)

1. `roboflow_upload_bundle_v2_playable_bias.zip` → Roboflow 업로드
2. `player-only` 프로젝트에서 relabeling (위 지침 준수)
3. YOLOv8 format으로 export 다운로드

### export 다운로드 후 즉시 실행

```bash
# Downloads 폴더의 새 export 이름으로 교체
./hado_venv/bin/python tools/retrain_track_b_export.py \
  --export-src "/Users/jinu/Downloads/hado-track-b-player-only.v5-trackb_v5_batch02_40.yolov8" \
  --run-name "trackb_v5_player_only" \
  --epochs 35 \
  --imgsz 416 \
  --batch 16
```

예상 결과:
- 이미지 수: ~127장 (v4 87 + 신규 40)
- 목표 mAP50: **0.68+** (v4: 0.653 대비 +0.03 기대)

---

## 5. Track A 현황

현재 브랜치 `presentation/demo-finalization` — 미push 커밋 5개:

| 커밋 | 내용 |
|------|------|
| `793a768` | docs: 테스트 수 253→257 갱신 |
| `a1d246d` | test: TTS skip 처리 (espeak 없는 환경) |
| `aa6ee99` | fix: demo.py 스모크가 demo.mp4 덮어쓰기 방지 |
| `3e97913` | fix: action_demo headless 크래시 + 한글 깨짐 수정 |
| `2fb5fe9` | perf: put_text_kr ROI 최적화 (Pi4 FPS 직결) |

**테스트**: 257개 전부 통과 ✅

---

## 6. 잔여 Track A 작업

| 항목 | 상태 | 방법 |
|------|------|------|
| W5 Pi4 실측 (FPS/RAM/TTS) | **미완** | `./run.sh w5_measure` on Pi4 |
| 위치오차 측정 | **미완** | `./run.sh measure-error` on Pi4 |
| TECH_REPORT §5 [TODO] 기입 | W5 실측 후 | 수동 |
| Q&A 영어 리허설 ×5 | **미완** | 직접 연습 |
| demo.mp4 실제 카메라 교체 | **미완** | Pi4에서 `./run.sh main --level 2 --record data/demo.mp4` |
