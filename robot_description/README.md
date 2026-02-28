# robot_description/ — 로봇 정의 (SSOT-A)

이 폴더는 SO-ARM101 로봇의 물리적 정의를 담는 **Single Source of Truth (SSOT-A)** 이다.

## 구조

```
robot_description/
├── urdf/       ← URDF 파일 (로봇 구조 정의)
├── xacro/      ← Xacro 파일 (파라미터화된 URDF)
├── meshes/     ← 3D 메쉬 파일 (STL/DAE, 시각/충돌)
└── README.md   ← 이 파일
```

## 역할

| 하위 폴더 | 내용 | Phase |
|----------|------|-------|
| `urdf/` | 로봇의 링크, 조인트, 관성 정의 | Phase 3 |
| `xacro/` | URDF의 파라미터화 버전 (반복 방지) | Phase 3 |
| `meshes/` | 시각용/충돌용 3D 메쉬 | Phase 3 |

## SSOT 원칙

- 로봇 구조를 변경하려면 이 폴더의 파일을 수정한다.
- 물리/제어 파라미터는 `params/`에서 관리한다 (이 폴더가 아님).
- USD 파일은 이 폴더의 URDF에서 `scripts/import_urdf_to_isaac.py`로 변환한다.

## 출처

- Phase 3에서 기준 URDF 출처, 버전, 수정 이력을 여기에 기록한다.
- 현재 (Phase 0): 비어 있음 — Phase 3에서 채워진다.
