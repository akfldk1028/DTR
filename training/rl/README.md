# training/rl/ — Reinforcement Learning (강화 학습)

## 개요

Isaac Lab 환경에서 보상 함수를 정의하고 RL 알고리즘으로 정책을 학습한다.

## 파이프라인

```
assets/ (USD) + params/ (YAML)
  → Isaac Lab 환경 정의 (태스크)
  → RL 학습 (PPO / SAC)
  → 체크포인트 저장
  → 평가 (성공률, 보상 곡선)
```

## 후보 알고리즘

| 알고리즘 | 설명 | 프레임워크 |
|---------|------|-----------|
| PPO | Proximal Policy Optimization | Isaac Lab (RSL-RL) |
| SAC | Soft Actor-Critic | Isaac Lab (RSL-RL) |

## 참고

- 커뮤니티 태스크 예시: `third_party/isaac_so_arm101/` (Phase 3 이후 추가)

## 현재 상태

Phase 0: 폴더 구조만 생성됨. Phase 6에서 구현 시작 (IL 이후, 옵션).
