# assets/ — 시뮬레이션 자산 (SSOT-B)

이 폴더는 Isaac Sim/Lab에서 사용하는 시뮬레이션 자산을 담는 **Single Source of Truth (SSOT-B)** 이다.

## 구조

```
assets/
├── usd/
│   └── so101_follower.usd       ← SO-ARM101 로봇 USD (23MB)
├── scenes/
│   └── kitchen_with_orange/     ← 주방+오렌지 씬 (104MB)
│       └── kitchen_with_orange/
│           ├── scene.usd
│           └── objects/         ← Orange, Plate 등
└── README.md
```

## 자산 다운로드 방법

대형 바이너리는 git에 포함하지 않는다. 다운로드 스크립트를 사용한다:

```bash
cd soarm_stack/
bash scripts/download_assets.sh
```

## 자산 목록

| 파일 | 출처 | 버전 | 크기 | SHA256 |
|------|------|------|------|--------|
| `usd/so101_follower.usd` | LeIsaac 릴리즈 | v0.1.0 | 23MB | `64a877c3b82cdc4a48ab8a1f321a2dd3ef7c55d4b10bce222b58c530d978ae58` |
| `scenes/kitchen_with_orange/` | LeIsaac 릴리즈 (zip) | v0.1.0 | 104MB | `d314c54b63a17e91402bfaddf26e21ff614adf2430fa092b78897f15b8adea34` |

출처 URL은 `docs/references.md`에서 관리한다.

## Isaac Sim/Lab 호환 정보

- Isaac Sim 5.1.0+ 에서 검증 (LeIsaac v0.1.0 기준)
- PhysX 5.x 기반 물리 시뮬레이션
- Drive 타입: position / velocity

## 규칙

- USD 파일을 직접 편집하지 않는다.
- 커스텀 USD가 필요하면 `scripts/import_urdf_to_isaac.py`로 URDF에서 변환한다.
- 다운로드한 파일의 체크섬을 위 표에서 관리한다.
