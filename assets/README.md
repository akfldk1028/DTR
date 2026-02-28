# assets/ — 시뮬레이션 자산 (SSOT-B)

이 폴더는 Isaac Sim/Lab에서 사용하는 시뮬레이션 자산을 담는 **Single Source of Truth (SSOT-B)** 이다.

## 구조

```
assets/
├── usd/       ← USD 파일 (로봇, 오브젝트)
├── scenes/    ← 씬 파일 (환경 구성)
└── README.md  ← 이 파일
```

## 자산 다운로드 방법

대형 바이너리는 git에 포함하지 않는다. 다운로드 스크립트를 사용한다:

```bash
cd soarm_stack/
bash scripts/download_assets.sh
```

### 다운로드되는 파일 (Phase 1)

| 파일 | 출처 | 버전 |
|------|------|------|
| `usd/so101_follower.usd` | LeIsaac 릴리즈 | v0.1.0 |
| `scenes/kitchen_with_orange/` | LeIsaac 릴리즈 | v0.1.0 |

출처 URL은 `docs/references.md`에서 관리한다.

## 규칙

- USD 파일을 직접 편집하지 않는다.
- 커스텀 USD가 필요하면 `scripts/import_urdf_to_isaac.py`로 URDF에서 변환한다.
- 다운로드한 파일의 체크섬을 기록한다 (Phase 1에서 추가).
- 현재 (Phase 0): 비어 있음 — Phase 1에서 채워진다.
