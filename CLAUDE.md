# CLAUDE.md — AI Agent Instructions for soarm_stack

> 이 파일은 Claude Code(또는 다른 AI 에이전트)가 이 프로젝트에서 작업할 때 가장 먼저 읽는 지침서이다.

---

## 프로젝트 개요

SO-ARM101 로봇의 디지털 트윈 파이프라인을 구축한다.
Isaac Sim/Lab에서 시뮬레이션 → 데이터 수집 → 학습(IL/RL) → VLA 확장까지의 전체 흐름을 다룬다.

---

## 폴더 맵

```
soarm_stack/
├── CLAUDE.md                 ← 이 파일 (AI 에이전트 지침서)
├── README.md                 ← 프로젝트 전체 개요 + Quick Start
├── .gitignore                ← 대형 바이너리 제외 규칙
├── .gitattributes            ← Git LFS 추적 규칙
├── VERSION                   ← 현재 버전 (SemVer)
├── CHANGELOG.md              ← Phase별 변경 기록
│
├── docs/                     ← 프로젝트 문서
│   ├── architecture.md       ← 데이터 흐름도
│   ├── naming-conventions.md ← 네이밍 규칙
│   ├── phase-checklist.md    ← Phase별 DoD 체크리스트
│   └── references.md         ← 외부 링크 중앙 관리
│
├── third_party/              ← 외부 레포 (clone/submodule)
│
├── robot_description/        ← 로봇 정의 SSOT-A
│   ├── urdf/                 ← URDF 파일
│   ├── xacro/                ← Xacro 파일
│   └── meshes/               ← 3D 메쉬 (STL/DAE)
│
├── params/                   ← 물리/제어 파라미터
│   ├── physics.yaml          ← 마찰, 댐핑, 질량, 접촉
│   └── control.yaml          ← 제어 주기, 게인, 지연
│
├── assets/                   ← 시뮬 자산 SSOT-B
│   ├── usd/                  ← USD 파일 (Isaac Sim/Lab)
│   └── scenes/               ← 씬 파일 (환경 구성)
│
├── scripts/                  ← 유틸리티 스크립트
│   ├── download_assets.sh    ← USD 자산 다운로드
│   ├── sanity_checks.py      ← 조인트/스케일/리밋 검증
│   └── import_urdf_to_isaac.py ← URDF→USD 변환
│
├── datasets/                 ← 수집 데이터 (LeRobot 포맷)
│
└── training/                 ← 학습 코드/설정
    ├── il/                   ← Imitation Learning
    ├── rl/                   ← Reinforcement Learning
    └── vla/                  ← Vision-Language-Action
```

---

## SSOT 원칙 (Single Source of Truth)

| 구분 | 내용 | 위치 |
|------|------|------|
| SSOT-A | 로봇 정의 (URDF/Xacro + meshes + params) | `robot_description/` + `params/` |
| SSOT-B | 시뮬 자산 (USD, PhysX/Drive 튜닝 포함) | `assets/` |

- 파라미터 변경 시 반드시 `params/*.yaml`을 먼저 수정하고, 거기서 시뮬/스크립트가 읽도록 한다.
- USD 파일을 직접 수정하지 않는다. URDF → USD 변환 스크립트를 통해 재생성한다.

---

## 네이밍 규칙

| 대상 | 규칙 | 예시 |
|------|------|------|
| 폴더 | `snake_case` | `robot_description/` |
| 파일 | `snake_case` | `physics.yaml`, `sanity_checks.py` |
| Python 변수/함수 | `snake_case` | `joint_limits` |
| Python 클래스 | `PascalCase` | `RobotConfig` |
| YAML 키 | `snake_case` | `friction_coefficient` |
| 상수 | `UPPER_SNAKE_CASE` | `MAX_JOINT_VELOCITY` |

---

## 현재 Phase 상태

- [x] **Phase 0** — Repo Skeleton (뼈대)
- [x] **Phase 1** — 자산 확보 (USD 다운로드, SHA256 검증 완료)
- [x] **Phase 2** — 최소 구동 (6 joints OK, 0.632kg, controller 0.00° error)
- [x] **Phase 3** — SSOT 정착 (URDF/Xacro + params 실제값, 단위/범위 문서화)
- [x] **Phase 4** — URDF→USD 자동화 (import_urdf_to_isaac.py, 1129줄, 정적검증 PASS)
- [x] **Phase 5** — 데이터 파이프라인 (collect_data.py + validate_dataset.py, LeRobot v2)
- [x] **Phase 6** — 학습 루프 (ACT IL + PPO RL + 통합 평가)
- [x] **Phase 7** — VLA 확장 (DummyVLA + SmolVLA wrapper, dry-run 검증 PASS)

### Isaac Sim 5.1.0 API 주의사항
- `set_joint_position_targets()` 대신 `apply_action(ArticulationAction(joint_positions=...))` 사용
- `print()` 대신 `logging` 모듈 사용 (Isaac Sim이 stdout 캡처함)
- conda 활성화: `eval "$(conda shell.bash hook 2>/dev/null)" && conda activate soarm`

### Headless-First 원칙 (CLI/AI 자동화 필수)

> **모든 스크립트는 GUI 없이 CLI로 실행 가능해야 한다.** AI 에이전트가 24/7 자동 조정하므로 GUI 의존 금지.

- **`--headless` 플래그**: 모든 Isaac Sim/Lab 스크립트에 `--headless` 지원 필수
- **AppLauncher 패턴**: `AppLauncher(headless=True)` 또는 `SimulationApp({"headless": True})`
- **카메라/영상 녹화**: `--enable_cameras` + `--headless`로 off-screen 렌더링
- **파라미터 오버라이드**: Hydra CLI (`env.key=value`) 또는 `params/*.yaml`에서 읽기
- **URDF→USD 변환**: `./isaaclab.sh -p scripts/tools/convert_urdf.py robot.urdf out.usd --headless`
- **학습 실행**: `./isaaclab.sh -p train.py --task Task --headless --num_envs N`
- **평가**: `./isaaclab.sh -p play.py --headless --video --enable_cameras`
- **환경변수**: `HEADLESS=1`, `ENABLE_CAMERAS=1` (X11 DISPLAY 불필요)
- **원격 모니터링**: `LIVESTREAM=2` (WebRTC) 또는 TensorBoard (`--logdir=logs`)

---

## AI 에이전트 작업 규칙

1. **새 파일을 만들기 전에** 기존 파일/폴더 구조를 확인한다.
2. **외부 링크**는 `docs/references.md`에 중앙 관리한다. 코드 안에 URL을 하드코딩하지 않는다.
3. **대형 파일** (USD, STL, PT 등)은 git에 커밋하지 않는다. `scripts/download_assets.sh`로 다운로드한다.
4. **파라미터**는 `params/*.yaml`에서 관리한다. 스크립트에 매직넘버를 넣지 않는다.
5. **모든 스크립트**에는 docstring으로 목적과 사용법을 명시한다.
6. **변경 사항**은 `CHANGELOG.md`에 기록한다.
7. 문서는 **한국어**로 작성한다 (코드 주석은 영어 가능).
