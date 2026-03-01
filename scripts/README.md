# scripts/ — 유틸리티 스크립트

이 폴더는 파이프라인의 각 단계에서 사용하는 스크립트를 모아 놓은 곳이다.

## 스크립트 목록

| 파일 | 목적 | Phase | 상태 |
|------|------|-------|------|
| `download_assets.sh` | LeIsaac USD/씬 자산 다운로드 | Phase 1 | 스텁 |
| `sanity_checks.py` | 조인트/리밋/스케일 검증 | Phase 2 | 스텁 |
| `import_urdf_to_isaac.py` | URDF→USD 변환 자동화 | Phase 4 | 스텁 |
| `collect_data.py` | Isaac Sim teleop 데이터 수집 → LeRobot v2 저장 | Phase 5 | 구현 |
| `validate_dataset.py` | LeRobot v2 데이터셋 검증 및 리플레이 | Phase 5 | 구현 |

## 사용 방법

```bash
# 자산 다운로드
bash scripts/download_assets.sh

# 조인트 검증 (Isaac Sim 환경 필요)
python scripts/sanity_checks.py --usd assets/usd/so101_follower.usd

# URDF→USD 변환 (Isaac Sim 환경 필요)
python scripts/import_urdf_to_isaac.py --input robot_description/urdf/so_arm101.urdf --output assets/usd/

# 데이터 수집 (Isaac Sim + LeRobot 환경 필요)
python scripts/collect_data.py --task LeIsaac-SO101-PickOrange-v0 --num_episodes 1

# 데이터셋 검증
python scripts/validate_dataset.py --repo_id local/so101_teleop

# 데이터셋 검증 + 시뮬 리플레이
python scripts/validate_dataset.py --repo_id local/so101_teleop --replay --task_name LeIsaac-SO101-PickOrange-v0
```

## 규칙

- 모든 스크립트에 docstring으로 목적과 사용법을 명시한다.
- 파라미터는 `params/*.yaml`에서 읽는다 (매직넘버 금지).
- 새 스크립트 추가 시 이 README도 업데이트한다.
