# Naming Conventions — 네이밍 규칙

## 파일 / 폴더

| 대상 | 규칙 | 예시 | 비고 |
|------|------|------|------|
| 폴더 | `snake_case` | `robot_description/` | 소문자, 밑줄 구분 |
| Python 파일 | `snake_case.py` | `sanity_checks.py` | |
| YAML 파일 | `snake_case.yaml` | `physics.yaml` | `.yml` 사용하지 않음 |
| Shell 스크립트 | `snake_case.sh` | `download_assets.sh` | |
| URDF 파일 | `snake_case.urdf` | `so_arm101.urdf` | |
| USD 파일 | `snake_case.usd` | `so101_follower.usd` | 원본 이름 유지 가능 |
| 문서 | `kebab-case.md` | `naming-conventions.md` | docs/ 내 문서만 |

## Python 코드

| 대상 | 규칙 | 예시 |
|------|------|------|
| 변수 | `snake_case` | `joint_limits` |
| 함수 | `snake_case` | `load_urdf()` |
| 클래스 | `PascalCase` | `RobotConfig` |
| 상수 | `UPPER_SNAKE_CASE` | `MAX_JOINT_VELOCITY` |
| 모듈 | `snake_case` | `import sanity_checks` |

## YAML 키

| 대상 | 규칙 | 예시 |
|------|------|------|
| 키 | `snake_case` | `friction_coefficient` |
| 단위 표기 | 주석으로 명시 | `# [N/m]` |
| 범위 표기 | `_min` / `_max` 접미사 | `stiffness_min`, `stiffness_max` |

## 조인트 이름

SO-ARM101의 조인트 이름은 원본 URDF의 네이밍을 따른다.
수정이 필요한 경우 `robot_description/README.md`에 매핑 테이블을 기록한다.

## 버전 관리

- SemVer 형식: `MAJOR.MINOR.PATCH`
- `VERSION` 파일에 현재 버전 기록
- `CHANGELOG.md`에 Phase별 변경 사항 기록
