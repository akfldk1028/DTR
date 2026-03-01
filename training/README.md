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

Phase 6: IL/RL 학습 코드 및 평가 스크립트 구현 완료.

### Phase 6 구현 파일

#### Imitation Learning (training/il/)
- `config.yaml` — ACT 학습 하이퍼파라미터 설정 (chunk_size=100, dim_model=512, n_heads=8)
- `train_act.py` — LeRobot ACT training wrapper (argparse CLI, config 검증, params/control.yaml 연동)

#### Reinforcement Learning (training/rl/)
- `config.yaml` — RL 학습 설정 (PPO, 1024 envs, domain randomization)
- `so101_env.py` — Isaac Lab ArticulationCfg + Reach task 환경 (SO-ARM101 6 DOF)
- `train_rl.py` — skrl PPO training wrapper (SharedActorCritic MLP [256,128,64])

#### 평가 (training/eval/)
- `evaluate_policy.py` — IL/RL 통합 정책 평가 (success_rate, trajectory_error, episode_reward)

### 주요 설정값
- 관절: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper (6 DOF)
- 구동 게인: stiffness=40.0, damping=4.0 (params/control.yaml 참조)
- Domain randomization: 질량 ±20%, 마찰 ±30%, actuator 게인 ±15%
- 시뮬레이션: sim_dt=0.005s, decimation=4 (params/physics.yaml 참조)
