# datasets/ — 수집 데이터 (LeRobot 포맷)

이 폴더는 시뮬레이션에서 수집한 에피소드 데이터를 LeRobot 포맷으로 저장한다.

## 데이터 스키마 (Phase 5에서 구현)

LeRobot 포맷을 따르며, 각 에피소드는 다음 필드를 포함한다:

```
episode_NNNNNN/
├── observation/
│   ├── joint_positions    # [6] float32 — 각 조인트 위치 [rad]
│   ├── joint_velocities   # [6] float32 — 각 조인트 속도 [rad/s]
│   └── images/            # RGB 이미지 (카메라별)
├── action/
│   └── joint_targets      # [6] float32 — 목표 조인트 위치 [rad]
├── timestamp              # float64 — 에피소드 시작 기준 [s]
└── (Phase 7) instruction  # str — VLA용 언어 지시
```

## 저장 규칙

- 대용량 데이터는 `datasets/data/`에 저장한다 (`.gitignore`에 의해 git 제외).
- 스키마 정의와 메타데이터만 git에 포함한다.
- 현재 (Phase 0): 비어 있음 — Phase 5에서 채워진다.
