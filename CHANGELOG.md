# Changelog

이 파일은 Phase별 변경 사항을 기록한다.
형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.0.0/)를 따른다.

---

## [0.7.0] - 2026-03-01

### Phase 7 — VLA 확장: instruction→action 경로 구현

#### Added
- `training/vla/inference.py` — VLAInference ABC + DummyVLA baseline + SmolVLAWrapper
- `training/vla/eval_in_sim.py` — Isaac Sim closed-loop VLA 평가 루프 (--dry-run 지원)
- `training/vla/__init__.py` — VLA 모듈 패키지 초기화 (__all__ exports)
- `training/vla/README.md` — VLA 추론 인터페이스 및 평가 문서
- `params/vla_eval.yaml` — VLA 평가 파라미터 (모델, 성공기준, 메트릭, Isaac Sim 설정)
- `datasets/README.md` — VLA task instruction 필드 문서화 추가

#### Changed
- 모든 Python 스크립트에 `--headless` (BooleanOptionalAction, default=True) 추가
- `training/vla/inference.py` — PROJECT_ROOT 패턴 추가

---

## [0.6.0] - 2026-03-01

### Phase 6 — 학습 루프: ACT IL baseline + Isaac Lab RL task

#### Added
- `training/il/config.yaml` — ACT 학습 하이퍼파라미터 설정 (chunk_size=100, dim_model=512, AdamW lr=1e-5)
- `training/il/train_act.py` — LeRobot ACT training wrapper (Phase 5 데이터셋 연동)
- `training/rl/config.yaml` — RL 학습 설정 (PPO, 1024 envs, domain randomization)
- `training/rl/so101_env.py` — Isaac Lab ArticulationCfg + SO-ARM101 Reach task 환경
- `training/rl/train_rl.py` — skrl PPO training wrapper (SharedActorCritic MLP [256,128,64])
- `training/eval/__init__.py` — 평가 모듈 패키지 초기화
- `training/eval/evaluate_policy.py` — IL/RL 통합 정책 평가 스크립트 (success_rate, trajectory_error 등)
- `training/__init__.py`, `training/il/__init__.py`, `training/rl/__init__.py` — 패키지 초기화 파일

#### Changed
- `training/README.md` — Phase 6 구현 상태 및 파일 목록 반영

---

## [0.5.0] - 2026-03-01

### Phase 5 — 데이터 파이프라인

#### Added
- `scripts/collect_data.py` — Isaac Sim teleop 데이터 수집 → LeRobot v2 포맷 저장
- `scripts/validate_dataset.py` — LeRobot v2 데이터셋 검증 및 시뮬 리플레이
- `params/data_pipeline.yaml` — 데이터 파이프라인 파라미터 (fps, 카메라 해상도, 에피소드 설정)
- `datasets/README.md` — LeRobot v2 데이터셋 포맷 문서화

#### Changed
- `.gitignore` — LeRobot v2 데이터셋 아티팩트 (parquet, video) 제외 규칙 추가
- `docs/references.md` — LeRobot v2 Dataset Format 문서 및 API 레퍼런스 추가
- `scripts/README.md` — Phase 5 스크립트 목록 추가
- `params/README.md` — `data_pipeline.yaml` 항목 추가

---

## [0.4.0] - 2026-03-01

### Phase 4 — URDF→USD 자동화

#### Added
- `scripts/import_urdf_to_isaac.py` — URDF→USD 자동 변환 (1129줄, SimulationApp headless 패턴)
- 정적 검증 108개 테스트 PASS (syntax, patterns, units, URDF cross-validation)

---

## [0.3.0] - 2026-03-01

### Phase 3 — SSOT 정착

#### Changed
- `params/physics.yaml` — placeholder → 실제 물리 파라미터 (마찰, 댐핑, 질량, 접촉)
- `params/control.yaml` — placeholder → 실제 제어 파라미터 (drive gains, joint limits)
- `robot_description/README.md` — URDF 출처, 조인트/링크 테이블, 메쉬 목록
- `params/README.md` — 파라미터 스키마 설명 업데이트

---

## [0.2.0] - 2026-03-01

### Phase 1-2 — 자산 확보 + 최소 구동

#### Added
- `assets/usd/so101_follower.usd` — LeIsaac 릴리즈 USD (SHA256 검증)
- `assets/scenes/kitchen_with_orange/` — 씬 자산 (zip 해제)
- `robot_description/urdf/so101_new_calib.urdf` — SO-ARM101 URDF
- `robot_description/meshes/` — 13개 STL 메쉬 파일
- `scripts/sanity_checks.py` — 조인트/스케일/리밋 검증 (6 joints, 0.632kg)
- `scripts/min_controller.py` — 최소 목표각 제어 (0.00° error)
- `assets/README.md` — 출처, 버전, SHA256 체크섬

---

## [0.1.0] - 2026-02-28

### Phase 0 — Repo Skeleton

#### Added
- 프로젝트 폴더 구조 전체 생성 (12개 디렉토리)
- `CLAUDE.md` — AI 에이전트 지침서
- `README.md` — 프로젝트 개요 + Quick Start
- `.gitignore` — 대형 바이너리 제외 규칙
- `.gitattributes` — Git LFS 추적 규칙
- `VERSION` — 0.1.0
- `docs/` — architecture, naming-conventions, phase-checklist, references
- 모든 하위 폴더에 `README.md` 배치 (11개)
- `params/physics.yaml`, `params/control.yaml` — 주석 포함 스키마 (placeholder)
- `scripts/download_assets.sh` — Phase 1용 스텁
- `scripts/sanity_checks.py` — Phase 2용 스텁
- `scripts/import_urdf_to_isaac.py` — Phase 4용 스텁
- `.gitkeep` 파일로 빈 디렉토리 보존
