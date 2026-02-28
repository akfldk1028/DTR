# training/ — 학습 코드 및 설정

이 폴더는 SO-ARM101의 정책 학습 코드와 설정을 관리한다.

## 구조

```
training/
├── il/    ← Imitation Learning (모방 학습)
├── rl/    ← Reinforcement Learning (강화 학습)
└── vla/   ← Vision-Language-Action (확장)
```

## 학습 방법별 개요

| 방법 | 데이터 | 프레임워크 | Phase |
|------|--------|-----------|-------|
| IL | 데모 에피소드 (datasets/) | LeRobot / PyTorch | Phase 6 |
| RL | 보상 함수 (Isaac Lab) | Isaac Lab RL | Phase 6 |
| VLA | 데모 + language instruction | 외부 VLA 모델 | Phase 7 |

## 파이프라인

1. `datasets/`에서 데이터 로드
2. 각 `il/`, `rl/`, `vla/` 폴더의 학습 코드 실행
3. 체크포인트는 각 폴더 하위에 저장 (git 제외)
4. 평가는 Isaac Sim/Lab 환경에서 수행

## 현재 상태

Phase 0: 폴더 구조만 생성됨. Phase 6에서 IL/RL 코드 구현 시작.
