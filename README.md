# soarm_stack — SO-ARM101 Digital Twin Pipeline

> "이 repo만 받으면 누구나 재현 가능"한 상태를 목표로 한다.

## 목적

SO-ARM101 로봇을 대상으로:
1. **디지털 트윈(Isaac Sim/Lab)** 에서 먼저 연동/안정화
2. **3D 프린트/조립(실물)** 로 내려가며
3. **데이터 수집 → 학습(IL/RL) → 평가** 를 반복 가능한 형태로 만들고
4. 추후 **VLA(vision-language-action)** 로 확장 가능한 구조 확보

## Quick Start

```bash
# 1. 레포 클론
git clone <repo-url>
cd soarm_stack/

# 2. 구조 파악
cat CLAUDE.md                    # AI 에이전트 지침서 (폴더맵, 규칙)
cat docs/phase-checklist.md      # Phase별 진행 상황

# 3. (Phase 1) 자산 다운로드
bash scripts/download_assets.sh

# 4. (Phase 2) 조인트 검증 — Isaac Sim 환경 필요
python scripts/sanity_checks.py --usd assets/usd/so101_follower.usd
```

## 폴더 구조

```
soarm_stack/
├── CLAUDE.md                        ← AI 에이전트 지침서
├── README.md                        ← 이 파일
├── .gitignore                       ← 대형 바이너리 제외
├── .gitattributes                   ← Git LFS 규칙
├── VERSION                          ← 현재 버전 (0.1.0)
├── CHANGELOG.md                     ← Phase별 변경 기록
│
├── docs/                            ← 프로젝트 문서
│   ├── architecture.md              ← 데이터 흐름도
│   ├── naming-conventions.md        ← 네이밍 규칙
│   ├── phase-checklist.md           ← Phase별 DoD 체크리스트
│   └── references.md               ← 외부 링크 중앙 관리
│
├── third_party/                     ← 외부 레포 (clone / submodule)
│
├── robot_description/               ← 로봇 정의 (SSOT-A)
│   ├── urdf/                        ← URDF 파일
│   ├── xacro/                       ← Xacro 파일 (파라미터화된 URDF)
│   └── meshes/                      ← 3D 메쉬 파일 (STL/DAE)
│
├── params/                          ← 물리/제어 파라미터 (YAML)
│   ├── physics.yaml                 ← 마찰, 댐핑, 질량, 접촉 범위
│   └── control.yaml                 ← 제어 주기, 지연, drive 게인 범위
│
├── assets/                          ← 시뮬 자산 (SSOT-B)
│   ├── usd/                         ← USD 파일 (Isaac Sim/Lab용)
│   └── scenes/                      ← 씬 파일 (환경 구성)
│
├── scripts/                         ← 유틸리티 스크립트
│   ├── download_assets.sh           ← 자산 다운로드 (Phase 1)
│   ├── sanity_checks.py             ← 조인트/스케일/리밋 검증 (Phase 2)
│   └── import_urdf_to_isaac.py      ← URDF→USD 변환 (Phase 4)
│
├── datasets/                        ← 수집 데이터 (LeRobot 포맷)
│
└── training/                        ← 학습 코드/설정
    ├── il/                          ← Imitation Learning
    ├── rl/                          ← Reinforcement Learning
    └── vla/                         ← Vision-Language-Action (확장)
```

## SSOT (Single Source of Truth) 원칙

| 구분 | 내용 | 위치 |
|------|------|------|
| SSOT-A | 로봇 정의 (URDF/Xacro + meshes + params) | `robot_description/` + `params/` |
| SSOT-B | 시뮬 자산 (USD, PhysX/Drive 튜닝 포함) | `assets/` |

## Phase 진행 현황

- [x] Phase 0 — Repo Skeleton (뼈대)
- [ ] Phase 1 — 자산 확보 (USD 우선) + 재현성 고정
- [ ] Phase 2 — 최소 구동 (Sanity Control)
- [ ] Phase 3 — SSOT 정착 (URDF/Xacro + params 표준화)
- [ ] Phase 4 — URDF→USD 자동화
- [ ] Phase 5 — 데이터 파이프라인 (LeRobot) 연결
- [ ] Phase 6 — 학습 루프 (IL/RL)
- [ ] Phase 7 — VLA 확장

## 레퍼런스

모든 외부 링크는 [docs/references.md](docs/references.md)에서 중앙 관리한다.

## 핵심 철학

- 양산이 아닌 **유연함 + 빠른 반복 실험** 우선
- "정답 물리 파라미터"가 아닌 **재현 가능한 자산 + 범위 기반(랜덤화)**
- 하드웨어는 **시뮬 연동 성공 후** 구매
