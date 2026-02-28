# Phase Checklist — DoD (Definition of Done)

각 Phase의 완료 조건(DoD)을 체크리스트로 관리한다.

---

## Phase 0 — Repo Skeleton (뼈대)

- [x] 폴더 구조 생성 (12개 디렉토리)
- [x] `CLAUDE.md` 작성 (폴더맵, 네이밍 규칙, SSOT 원칙, Phase 상태)
- [x] `README.md`에 Quick Start 섹션 포함
- [x] `.gitignore` — 대형 바이너리 제외
- [x] `.gitattributes` — Git LFS 규칙
- [x] `VERSION` — 0.1.0
- [x] `CHANGELOG.md` — Phase 0 기록
- [x] `docs/` — architecture, naming-conventions, phase-checklist, references
- [x] 모든 하위 폴더에 README.md
- [x] `params/*.yaml` — 주석 포함 스키마 (placeholder)
- [x] `scripts/` — 3개 스텁 (download_assets.sh, sanity_checks.py, import_urdf_to_isaac.py)
- [x] 빈 디렉토리 없음 (.gitkeep 배치)

---

## Phase 1 — 자산 확보 (USD 우선) + 재현성 고정

- [ ] `scripts/download_assets.sh` 구현 (LeIsaac USD 다운로드)
- [ ] `assets/usd/so101_follower.usd` 다운로드 완료
- [ ] `assets/scenes/kitchen_with_orange/` 다운로드 및 압축 해제
- [ ] `assets/README.md`에 출처, 버전, 체크섬 기록
- [ ] Isaac Sim/Lab에서 USD 로드 성공 (에러 없음)

---

## Phase 2 — 최소 구동 (Sanity Control)

- [ ] `scripts/sanity_checks.py` 구현 (조인트 리스트/리밋/스케일 출력)
- [ ] `scripts/min_controller.py` 구현 (목표각 추종 제어)
- [ ] 관절이 떨지 않고 목표각을 추종
- [ ] 간단 접촉(박스/테이블)에서 물리 폭발 없음
- [ ] 재실행해도 결과가 동일 (재현성)

---

## Phase 3 — SSOT 정착 (URDF/Xacro + params 표준화)

- [ ] `robot_description/`에 기준 URDF/Xacro + meshes 정리
- [ ] URDF 출처/버전 기록
- [ ] `params/physics.yaml` — 실제 물리 파라미터 값 채움
- [ ] `params/control.yaml` — 실제 제어 파라미터 값 채움
- [ ] params에 단위/범위 문서화 완료
- [ ] README에 "기준 URDF 링크 + 기준 USD 파일" 명시

---

## Phase 4 — URDF→USD 자동화

- [ ] `scripts/import_urdf_to_isaac.py` 구현
- [ ] 입력 URDF 경로 → 출력 USD 경로 자동 생성
- [ ] 수동 단계가 있다면 문서화 완료
- [ ] 변환된 USD로 Isaac Sim/Lab 로드 성공

---

## Phase 5 — 데이터 파이프라인 (LeRobot)

- [ ] `datasets/` 구조/스키마 문서화
- [ ] 최소 1 에피소드 생성 스크립트
- [ ] 에피소드 리플레이 스크립트
- [ ] 데이터 생성/저장/리플레이 성공
- [ ] 같은 데이터로 재학습/재평가 가능

---

## Phase 6 — 학습 루프 (IL/RL)

- [ ] `training/il/` baseline 1개 구현
- [ ] (옵션) `training/rl/` Isaac Lab 태스크 1개
- [ ] 학습 1회 이상 돌아감
- [ ] 체크포인트/로그 생성 확인

---

## Phase 7 — VLA 확장

- [ ] 데이터 스키마에 `instruction` 필드 포함
- [ ] `training/vla/`에 inference I/O 계약 문서화
- [ ] (더미 모델이라도) instruction → action 경로가 시뮬 평가로 연결
