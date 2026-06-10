# Track A / Track B OpenAI + GitHub Reference — 2026-06-10

이 문서는 현재 HADO Smart Court 프로젝트를 진행할 때 참고할 만한
공식 OpenAI 문서와 주요 GitHub 리소스를 Track A / Track B / 자동화 기준으로 정리한 참고 문서다.

목적은 단순하다.

- Track A에서는 발표 범위 안에서 바로 구현에 도움이 되는 자료만 본다.
- Track B에서는 실제 경기영상, occlusion, relabeling, 재학습 루프에 직접 도움이 되는 자료를 본다.
- 자동화와 협업은 OpenAI Codex + GitHub 흐름으로 정리한다.

중요:

- 아래 링크는 가능하면 공식 문서 또는 프로젝트 공식 저장소만 넣었다.
- Track A와 Track B를 다시 섞지 않도록 용도별로 분리해 적었다.

## 1. 가장 먼저 보는 기준

### Track A에 더 가까운 자료

- single-camera demo
- pose / keypoint inference
- edge device optimization
- 발표용 실행 흐름
- GitHub 협업과 PR 리뷰 자동화

### Track B에 더 가까운 자료

- hard frame review
- vision-based batch QA
- relabeling workflow
- retraining / run comparison
- tracking after overlap / occlusion
- pose verification after detector stabilization

## 2. OpenAI 공식 문서

### 2.1 Images and Vision

링크:

- [OpenAI Images and vision guide](https://platform.openai.com/docs/guides/images-vision)

이 문서가 유용한 이유:

- 여러 이미지를 한 요청에 함께 넣어 비교할 수 있다.
- 이미지 URL, base64, file input 흐름을 모두 지원한다.
- Track B에서 hard frame 묶음을 검토하거나, Track A에서 발표용 샘플 프레임 설명을 자동 정리할 때 쓸 수 있다.

추천 용도:

- Track B:
  - `hard frame 20~100장` 묶음 QA
  - “선수 수가 몇 명인가”
  - “가림 정도가 clean / partial / heavy 중 무엇인가”
  - “현재 bbox 라벨이 이상한지” 같은 반자동 검토
- Track A:
  - 발표용 데모 프레임 설명 초안 만들기
  - 시연 중 어떤 장면이 잘 보이는지 선별

### 2.2 Structured Outputs

링크:

- [OpenAI Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs)

이 문서가 유용한 이유:

- JSON schema를 강제해 응답 형식을 안정적으로 고정할 수 있다.

추천 용도:

- Track B frame QA 결과를 아래처럼 통일된 JSON으로 저장:

```json
{
  "frame_name": "match09_frame_001500.jpg",
  "player_count_visible": 3,
  "occlusion_level": "heavy",
  "label_issue": true,
  "notes": "shield overlap around center player"
}
```

- 이후 `CSV`, `priority list`, `relabel queue`로 자동 변환 가능

### 2.3 Batch API

링크:

- [OpenAI Batch API guide](https://platform.openai.com/docs/guides/batch)

이 문서가 유용한 이유:

- 대량 요청을 비동기로 처리할 수 있다.
- 문서 기준으로 비용 절감형 오프라인 처리에 맞다.

추천 용도:

- Track B nightly QA
- batch02 / batch03 frame set 자동 판독
- 경기영상에서 추출한 프레임 수백 장을 한 번에 검토

실전 아이디어:

1. `prepare_track_b_batch_review.py`로 프레임 추출
2. OpenAI Batch API로 frame QA 요청 생성
3. Structured Outputs로 결과 수집
4. `relabel_priority.csv` 자동 생성

### 2.4 Codex GitHub Integration

링크:

- [Codex GitHub integration](https://developers.openai.com/codex/integrations/github)

이 문서가 유용한 이유:

- PR 단위 리뷰
- 코멘트 기반 재실행
- GitHub 안에서 follow-up iteration

추천 용도:

- Track A:
  - 발표 직전 demo polish PR 리뷰
- Track B:
  - batch review script, retrain script, report update PR 검토

### 2.5 Codex Automations

링크:

- [Codex automations](https://developers.openai.com/codex/app/automations)

이 문서가 유용한 이유:

- 반복 작업을 스레드 기반 자동화로 실행할 수 있다.

추천 용도:

- Track B 새벽 batch review 자동 점검
- leaderboard 업데이트 후 요약 알림
- 문서/리포트 재생성 체크

## 3. GitHub / 오픈소스 참고 자료

### 3.1 Ultralytics YOLO

링크:

- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

가장 관련이 큰 이유:

- 현재 프로젝트의 detection / pose / export 흐름과 가장 직접적으로 맞닿아 있다.

추천 용도:

- Track A:
  - `YOLOv8n-pose` 데모 안정화
  - NCNN / ONNX / PT 흐름 이해
- Track B:
  - player-only detector retraining
  - 실제 영상 재평가 파이프라인 정리

### 3.2 Ultralytics Tracking Docs

링크:

- [Ultralytics tracking mode docs](https://github.com/ultralytics/ultralytics/blob/main/docs/en/modes/track.md)

추천 용도:

- Track B에서 occlusion 이후 ID 유지 개선 아이디어 참고
- 현재 IoU tracker만으로 부족할 때 대안 검토

### 3.3 ByteTrack

링크:

- [FoundationVision/ByteTrack](https://github.com/FoundationVision/ByteTrack)

추천 이유:

- 낮은 confidence detection까지 association에 활용하는 방식이라
  effect overlap, partial occlusion, short miss recovery에 참고 가치가 높다.

Track B relevance:

- 선수 3명이 보여야 하는 장면에서 2명만 잡히는 문제
- 겹침 후 track recovery
- high-count / low-count hard frame 분석 이후 tracker 교체 검토

### 3.4 MMPose

링크:

- [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose)

추천 이유:

- detector stabilization 이후 skeleton verification 정확도를 더 보고 싶을 때 참고 가능

Track B relevance:

- `tools/export_track_b_pose_review.py` 이후 단계
- action / pose consistency review
- heavy occlusion 장면에서 keypoint robustness 비교

### 3.5 MediaPipe

링크:

- [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe)

추천 이유:

- 가벼운 skeleton demo나 edge-device orientation에서 참고하기 좋다.

Track A relevance:

- 발표용 보조 프로토타입
- 경량 skeleton visualization 참고

주의:

- 현재 저장소는 YOLOv8 기반이므로, Track A 주력 파이프라인을 갑자기 MediaPipe로 바꾸는 용도는 아니다.

### 3.6 Roboflow Supervision

링크:

- [roboflow/supervision](https://github.com/roboflow/supervision)

추천 이유:

- detection visualization, video annotation, result inspection이 편하다.

Track B relevance:

- batch review contact sheet 개선
- detection overlay mp4 재생성
- frame-level review artifact 품질 향상

## 4. 트랙별 추천 조합

### Track A 추천 조합

가장 추천하는 조합:

1. Ultralytics YOLO
2. Codex GitHub integration
3. Codex Automations
4. MediaPipe는 참고만

해석:

- Track A는 발표용 안정성이 핵심이므로
- 새로운 연구보다 현재 demo path를 더 안정화하는 쪽이 맞다.

### Track B 추천 조합

가장 추천하는 조합:

1. Ultralytics YOLO
2. ByteTrack
3. MMPose
4. Roboflow Supervision
5. OpenAI Vision
6. OpenAI Structured Outputs
7. OpenAI Batch API

해석:

- Track B는 “모델 재학습”과 “실제 실패 장면 QA 자동화”가 같이 가야 한다.
- OpenAI는 detector 자체를 대신 학습하는 역할이 아니라,
  frame triage / relabel priority / report generation 쪽에서 훨씬 강하다.

## 5. 지금 저장소에 바로 연결할 수 있는 아이디어

### 아이디어 A — Track B frame QA 자동화

현재 도구:

- `tools/prepare_track_b_batch_review.py`
- `tools/export_track_b_pose_review.py`

다음 확장:

1. hard frame를 추출한다.
2. OpenAI Vision에 묶음 요청을 보낸다.
3. Structured Outputs로 결과를 받는다.
4. `batch_priority.csv`를 다시 정렬한다.

예상 효과:

- 사람이 100장을 전부 직접 열어보는 시간을 줄일 수 있다.

### 아이디어 B — GitHub PR review 루프

대상:

- `src/demo.py`
- `src/demo_pose.py`
- `src/action_demo.py`
- Track B tools / docs

흐름:

1. 작은 단위로 commit
2. GitHub PR 생성
3. Codex GitHub integration으로 리뷰
4. 리뷰 반영 후 재실행

예상 효과:

- Track A/B가 같은 저장소에 있어도 섞이는 위험을 조금 줄일 수 있다.

### 아이디어 C — 발표용 Track A QA 자동화

대상:

- demo output mp4
- preview png
- presentation HUD screenshot

흐름:

1. headless smoke output 생성
2. 주요 프레임 이미지 저장
3. OpenAI Vision으로 “텍스트 가독성 / 사람 수 / 오버레이 겹침” 점검

이건 모델 성능 향상보다는 발표 화면 quality gate에 가깝다.

## 6. 당장 참고 우선순위

시간이 없으면 아래 순서만 봐도 충분하다.

### Track A

1. [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
2. [Codex GitHub integration](https://developers.openai.com/codex/integrations/github)
3. [Codex automations](https://developers.openai.com/codex/app/automations)

### Track B

1. [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
2. [ByteTrack](https://github.com/FoundationVision/ByteTrack)
3. [OpenAI Images and vision guide](https://platform.openai.com/docs/guides/images-vision)
4. [OpenAI Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs)
5. [OpenAI Batch API guide](https://platform.openai.com/docs/guides/batch)

## 7. 결론

정리하면:

- Track A는 “현재 데모 파이프라인을 안정화하는 자료”를 봐야 한다.
- Track B는 “실패 장면을 더 잘 추려서 다시 학습시키는 루프”를 강화하는 자료를 봐야 한다.
- OpenAI는 Track B에서 detector 자체보다 `QA / triage / 자동 리포트`에 더 잘 맞는다.
- GitHub + Codex 조합은 두 트랙을 분리한 채 협업하는 데 가장 실용적이다.
