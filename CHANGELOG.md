# Changelog

이 파일은 Phase별 변경 사항을 기록한다.
형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.0.0/)를 따른다.

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
