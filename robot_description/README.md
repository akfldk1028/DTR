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

- **원본 레포**: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- **생성 도구**: onshape-to-robot (Onshape CAD → URDF 자동 변환)
- **브랜치**: `main`
- **기준 URDF 파일**: `urdf/so101_new_calib.urdf`
- **외부 링크 관리**: `docs/references.md` 참조

## 조인트 요약

6개 조인트, 모두 revolute 타입이다.

| 조인트 이름 | 타입 | 리밋 (도) | 리밋 (라디안) |
|------------|------|----------|--------------|
| `shoulder_pan` | revolute | ±110° | ±1.9199 rad |
| `shoulder_lift` | revolute | ±100° | ±1.7453 rad |
| `elbow_flex` | revolute | ±90° | ±1.5708 rad |
| `wrist_flex` | revolute | ±95° | ±1.6581 rad |
| `wrist_roll` | revolute | ±160° | ±2.7925 rad |
| `gripper` | revolute | 10°~100° | 0.1745~1.7453 rad |

> 정확한 수치는 `params/control.yaml`의 `joint_limits` 참조.

## 링크/질량 요약

| 링크 이름 | 질량 (kg) |
|----------|----------|
| `base` | 0.147 |
| `shoulder` | 0.100 |
| `upper_arm` | 0.103 |
| `lower_arm` | 0.104 |
| `wrist` | 0.079 |
| `gripper` | 0.087 |
| `jaw` | 0.012 |
| **합계** | **0.632** |

> 정확한 수치는 `params/physics.yaml`의 `per_link_masses` 참조.

## 메쉬 파일

`meshes/` 디렉토리에 아래 8개의 STL 파일이 필요하다.

| # | 파일명 | 대응 링크 |
|---|--------|----------|
| 1 | `base.stl` | base |
| 2 | `shoulder.stl` | shoulder |
| 3 | `upper_arm.stl` | upper_arm |
| 4 | `lower_arm.stl` | lower_arm |
| 5 | `wrist.stl` | wrist |
| 6 | `wrist_roll.stl` | wrist_roll |
| 7 | `gripper.stl` | gripper |
| 8 | `jaw.stl` | jaw |

> STL 파일은 대형 바이너리이므로 Git LFS로 관리한다 (`.gitattributes` 참조).
> 최초 다운로드는 `scripts/download_assets.sh`를 통해 수행한다.
