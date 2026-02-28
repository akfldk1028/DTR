# Architecture — 데이터 흐름도

## 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│                        soarm_stack                              │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ robot_       │    │ scripts/     │    │ assets/      │      │
│  │ description/ │───>│ import_urdf  │───>│ usd/         │      │
│  │              │    │ _to_isaac.py │    │ scenes/      │      │
│  │ - urdf/      │    └──────────────┘    └──────┬───────┘      │
│  │ - xacro/     │                               │              │
│  │ - meshes/    │    ┌──────────────┐            │              │
│  └──────────────┘    │ params/      │            │              │
│                      │ physics.yaml │────────────┤              │
│                      │ control.yaml │            │              │
│                      └──────────────┘            │              │
│                                                  v              │
│                                         ┌──────────────┐       │
│                                         │ Isaac        │       │
│                                         │ Sim / Lab    │       │
│                                         └──────┬───────┘       │
│                                                │               │
│                                                v               │
│                                         ┌──────────────┐       │
│                                         │ datasets/    │       │
│                                         │ (LeRobot)    │       │
│                                         └──────┬───────┘       │
│                                                │               │
│                                                v               │
│                                         ┌──────────────┐       │
│                                         │ training/    │       │
│                                         │ il/ rl/ vla/ │       │
│                                         └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

## 단계별 흐름

### 1. 로봇 정의 (SSOT-A)
- `robot_description/urdf/` — 로봇의 물리적 구조 정의
- `robot_description/xacro/` — 파라미터화된 URDF
- `robot_description/meshes/` — 3D 메쉬 파일 (시각/충돌)
- `params/` — 물리/제어 파라미터 (YAML)

### 2. USD 변환
- `scripts/import_urdf_to_isaac.py` — URDF를 USD로 변환
- 결과물은 `assets/usd/`에 저장

### 3. 시뮬레이션
- `assets/usd/` + `assets/scenes/` → Isaac Sim/Lab에서 로드
- `params/physics.yaml` → PhysX 물리 파라미터 적용
- `params/control.yaml` → 제어 게인/주기 적용

### 4. 데이터 수집
- Isaac Sim/Lab에서 에피소드 실행 → `datasets/`에 LeRobot 포맷으로 저장
- 관측(observation), 액션(action), 이미지, 타임스탬프 포함

### 5. 학습
- `training/il/` — Imitation Learning (데모 데이터 기반)
- `training/rl/` — Reinforcement Learning (보상 함수 기반)
- `training/vla/` — Vision-Language-Action (확장)
