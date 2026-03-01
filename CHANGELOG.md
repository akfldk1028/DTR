# Changelog

이 파일은 Phase별 변경 사항을 기록한다.
형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.0.0/)를 따른다.

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
