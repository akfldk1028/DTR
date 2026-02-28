# training/il/ — Imitation Learning (모방 학습)

## 개요

데모 에피소드 데이터를 기반으로 행동 복제(Behavior Cloning) 정책을 학습한다.

## 파이프라인

```
datasets/ (LeRobot 포맷)
  → 데이터 로드
  → 정책 학습 (BC / ACT / Diffusion Policy)
  → 체크포인트 저장
  → Isaac Sim/Lab에서 평가
```

## 후보 알고리즘

| 알고리즘 | 설명 | 레퍼런스 |
|---------|------|---------|
| ACT | Action Chunking with Transformers | LeRobot 내장 |
| Diffusion Policy | 확산 모델 기반 정책 | LeRobot 내장 |

## 현재 상태

Phase 0: 폴더 구조만 생성됨. Phase 6에서 구현 시작.
