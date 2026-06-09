# Track B Collaboration Report — 2026-06-09

## 1. 이 문서의 목적

이 문서는 Track B 진행 상황을 다른 사람에게 빠르게 설명하고, 협업할 때 어떤 상태인지 바로 공유할 수 있도록 만든 요약 보고서다.

대상:

- 팀원
- 외부 협업자
- Claude Code 같은 다른 에이전트
- 이후 작업을 이어받을 사람

## 2. 지금 무엇을 하고 있는가

Track B는 HADO 경기영상에서 가상 이펙트가 선수와 겹칠 때도 선수를 잘 찾는 detector를 만드는 개인 장기 프로젝트다.

이번 단계에서는 모델을 새로 갈아엎기보다, 실제 경기영상에서 실패하는 장면을 모아서 다시 학습에 넣을 수 있는 구조를 만드는 데 집중했다.

## 3. 현재까지 한 일

### 완료

- Roboflow V4 export를 player-only로 정규화
- YOLOv8n baseline 재학습
- 실제 경기영상 10개 intake 완료
- hard frame 자동 추출 파이프라인 구축
- relabeling 우선순위 선정 완료
- Roboflow 업로드용 ZIP 생성 완료

### 현재 baseline

- Precision `0.619`
- Recall `0.686`
- mAP50-95 `0.430`

## 4. 왜 추가 작업이 필요한가

Roboflow 내부 test set에서는 성능이 좋아졌지만, 실제 경기영상에서는 여전히 다음 문제가 남아 있다.

- AR shield와 projectile이 선수 몸을 가림
- detector confidence가 낮아짐
- 어떤 장면에서는 중복 검출이 나옴
- intro / scoreboard 같은 비플레이 장면이 섞임

즉, 지금 문제는 “모델이 전혀 안 되는 것”이 아니라 “실제 경기 분포를 더 잘 반영한 데이터 보강이 필요하다”는 것이다.

## 5. 지금 바로 협업자가 할 수 있는 일

가장 중요한 다음 액션은 하나다.

- `data/track_b_batch_review/roboflow_upload_bundle_batch01_phase1.zip`
  - 36장
  - player-only relabeling 진행

그 다음:

- Roboflow에서 `V5` export 생성
- export 결과를 다시 전달

## 6. 협업 시 주의사항

- 이번 라운드는 `player-only` 유지
- effect/object 클래스를 섣불리 추가하지 않기
- playable gameplay frame 우선
- intro / roster / scoreboard는 positive 학습용으로 무리하게 쓰지 않기
- shield overlap에서도 player box는 최대한 일관되게 그리기

## 7. 협업자에게 전달하면 좋은 파일

- 실무 인계용:
  - `docs/TRACK_B_HANDOFF_REPORT_2026_06_09.md`
- 기술 상세:
  - `docs/TRACK_B_TECHNICAL_REPORT_2026_06_09.md`
- 업로드용 ZIP:
  - `data/track_b_batch_review/roboflow_upload_bundle_batch01_phase1.zip`
- 업로드 설명 CSV:
  - `data/track_b_batch_review/roboflow_upload_bundle_batch01_phase1_manifest.csv`

## 8. 현재 상태 한 줄 요약

Track B는 지금 “실제 경기 실패 장면을 다시 학습 루프로 연결할 수 있는 상태”까지 왔고, 다음 단계는 phase1 36장 relabeling 후 V5 재학습이다.
