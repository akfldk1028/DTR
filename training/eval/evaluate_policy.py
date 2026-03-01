#!/usr/bin/env python3
"""evaluate_policy.py — 학습된 정책 시뮬레이션 평가 스크립트

목적:
    IL (ACT) 및 RL (PPO) 학습된 정책을 Isaac Lab 시뮬레이션 환경에서 평가한다.
    평가 메트릭을 산출하고 JSON 파일로 저장한다.

    - IL (ACT) 정책: LeRobot ACT 체크포인트 로드
    - RL (PPO) 정책: skrl PPO 체크포인트 로드
    - Isaac Lab 시뮬레이션 환경에서 에피소드 실행
    - 메트릭 산출: success_rate, trajectory_error, episode_reward, episode_length
    - 결과를 JSON 파일 및 콘솔 요약으로 출력

사용법:
    python training/eval/evaluate_policy.py --policy-type rl --checkpoint training/rl/checkpoints/best_agent.pt
    python training/eval/evaluate_policy.py --policy-type il --checkpoint training/il/checkpoints/act_policy.pt
    python training/eval/evaluate_policy.py --policy-type rl --checkpoint training/rl/checkpoints/best_agent.pt --num-episodes 50 --render
    python training/eval/evaluate_policy.py --policy-type rl --checkpoint training/rl/checkpoints/best_agent.pt --output training/eval/results.json

필요 환경:
    - Isaac Sim 5.1.0 + Isaac Lab v2.3.0 (RL 평가)
    - LeRobot 0.4.4 (IL 평가)
    - skrl (RL 평가)
    - PyTorch (GPU 권장)
    - conda env: soarm

참고:
    - IL 학습 스크립트: training/il/train_act.py
    - RL 학습 스크립트: training/rl/train_rl.py
    - RL 환경: training/rl/so101_env.py
    - 로봇 제어 파라미터: params/control.yaml
    - 물리 파라미터: params/physics.yaml

Phase: 6
상태: 정책 평가 스크립트
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# --- 프로젝트 루트 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- 지원하는 정책 타입 ---
SUPPORTED_POLICY_TYPES = ("il", "rl")

# --- 기본 평가 설정 ---
DEFAULT_NUM_EPISODES = 100
DEFAULT_SUCCESS_THRESHOLD = 0.02  # m — 목표 도달 판정 거리 (training/rl/config.yaml 참조)


def parse_args():
    """CLI 인자 파싱."""
    parser = argparse.ArgumentParser(
        description="학습된 정책 시뮬레이션 평가 — IL (ACT) / RL (PPO)"
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        required=True,
        choices=SUPPORTED_POLICY_TYPES,
        help="평가할 정책 타입: il (ACT) 또는 rl (PPO)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="정책 체크포인트 파일 경로",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help=f"평가 에피소드 수 (default: {DEFAULT_NUM_EPISODES})",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="시뮬레이션 렌더링 활성화",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="메트릭 JSON 출력 경로 (미지정 시 training/eval/results_{policy_type}.json)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="학습 설정 파일 경로 (미지정 시 정책 타입에 따라 자동 선택)",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=None,
        help=f"목표 도달 판정 거리 (m) (default: {DEFAULT_SUCCESS_THRESHOLD})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드 (default: 42)",
    )
    return parser.parse_args()


def load_config(config_path):
    """YAML 설정 파일을 로드한다.

    Args:
        config_path: config.yaml 파일 경로

    Returns:
        dict: 파싱된 설정 딕셔너리

    Raises:
        FileNotFoundError: 설정 파일이 없을 경우
    """
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("설정 로드 완료: %s", config_path)
    return config


def get_default_config_path(policy_type):
    """정책 타입에 따른 기본 설정 파일 경로를 반환한다.

    Args:
        policy_type: 정책 타입 ("il" 또는 "rl")

    Returns:
        str: 설정 파일 경로
    """
    if policy_type == "il":
        return "training/il/config.yaml"
    elif policy_type == "rl":
        return "training/rl/config.yaml"
    else:
        raise ValueError(f"지원하지 않는 정책 타입: {policy_type}")


def get_success_threshold(config, policy_type, cli_threshold=None):
    """성공 판정 거리 임계값을 결정한다.

    CLI 인자 > config 파일 > 기본값 순으로 우선순위를 적용한다.

    Args:
        config: 설정 딕셔너리
        policy_type: 정책 타입
        cli_threshold: CLI에서 전달된 임계값 (None이면 config에서 읽음)

    Returns:
        float: 성공 판정 거리 (m)
    """
    if cli_threshold is not None:
        return cli_threshold

    # config에서 성공 임계값 로드
    if policy_type == "rl":
        threshold = config.get("reward", {}).get("success_threshold")
    else:
        threshold = config.get("evaluation", {}).get("success_threshold")

    if threshold is not None:
        return threshold

    return DEFAULT_SUCCESS_THRESHOLD


def load_il_policy(checkpoint_path, config):
    """IL (ACT) 정책 체크포인트를 로드한다.

    LeRobot ACT 정책을 체크포인트에서 복원한다.

    Args:
        checkpoint_path: ACT 체크포인트 파일 경로
        config: IL 학습 설정 딕셔너리

    Returns:
        정책 모델 인스턴스

    Raises:
        FileNotFoundError: 체크포인트 파일이 없을 경우
        ImportError: LeRobot/PyTorch가 설치되지 않은 경우
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}"
        )

    try:
        import torch
    except ImportError:
        logger.error(
            "PyTorch가 설치되지 않았습니다. 설치: pip install torch"
        )
        raise

    try:
        from lerobot.common.policies.act.modeling_act import ACTPolicy
    except ImportError:
        logger.error(
            "LeRobot이 설치되지 않았습니다. "
            "설치 방법:\n"
            "  pip install lerobot\n"
            "  또는: git clone https://github.com/huggingface/lerobot && "
            "cd lerobot && pip install -e .\n"
            "자세한 내용: docs/references.md 참조"
        )
        raise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ACT 정책 로드
    logger.info("ACT 정책 체크포인트 로드: %s", checkpoint_path)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)

    # LeRobot ACT 정책 구성
    policy_cfg = config.get("policy", {})
    policy = ACTPolicy(
        config=checkpoint.get("config", policy_cfg),
    )
    policy.load_state_dict(checkpoint.get("state_dict", checkpoint))
    policy.to(device)
    policy.eval()

    logger.info("ACT 정책 로드 완료 (device: %s)", device)
    return policy


def load_rl_policy(checkpoint_path, config):
    """RL (PPO) 정책 체크포인트를 로드한다.

    skrl PPO 에이전트를 체크포인트에서 복원한다.

    Args:
        checkpoint_path: skrl 체크포인트 파일 경로
        config: RL 학습 설정 딕셔너리

    Returns:
        정책 모델 인스턴스

    Raises:
        FileNotFoundError: 체크포인트 파일이 없을 경우
        ImportError: skrl/PyTorch가 설치되지 않은 경우
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}"
        )

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        logger.error(
            "PyTorch가 설치되지 않았습니다. 설치: pip install torch"
        )
        raise

    try:
        from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
        from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
    except ImportError:
        logger.error(
            "skrl이 설치되지 않았습니다. "
            "설치: pip install skrl\n"
            "자세한 내용: docs/references.md 참조"
        )
        raise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 네트워크 설정 로드
    training = config.get("training", {})
    network_cfg = training.get("network", {})
    obs_cfg = config.get("observation", {})
    action_cfg = config.get("action", {})

    obs_dim = obs_cfg.get("total_dim", 15)
    action_dim = action_cfg.get("dim", 6)
    hidden_layers = network_cfg.get("hidden_layers", [256, 128, 64])

    # 활성화 함수 매핑
    activation_map = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "selu": nn.SELU,
    }
    activation_name = network_cfg.get("activation", "elu")
    activation_cls = activation_map.get(activation_name, nn.ELU)

    # 공유 MLP Actor-Critic 모델 정의 (train_rl.py와 동일 구조)
    class SharedActorCritic(GaussianMixin, DeterministicMixin, Model):
        """평가용 공유 MLP Actor-Critic 네트워크.

        training/rl/train_rl.py의 SharedActorCritic과 동일한 구조를 사용한다.
        """

        def __init__(self, observation_space, action_space, device,
                     clip_actions=False, clip_log_std=True,
                     min_log_std=-20, max_log_std=2,
                     reduction="sum", role=""):
            Model.__init__(self, observation_space, action_space, device)
            GaussianMixin.__init__(
                self,
                clip_actions=clip_actions,
                clip_log_std=clip_log_std,
                min_log_std=min_log_std,
                max_log_std=max_log_std,
                reduction=reduction,
                role=role,
            )
            DeterministicMixin.__init__(self, clip_actions=clip_actions, role=role)

            # 공유 백본 구성
            layers = []
            in_dim = obs_dim
            for h_dim in hidden_layers:
                layers.append(nn.Linear(in_dim, h_dim))
                layers.append(activation_cls())
                in_dim = h_dim

            self.backbone = nn.Sequential(*layers)
            self.actor_head = nn.Linear(in_dim, action_dim)
            self.log_std_parameter = nn.Parameter(torch.zeros(action_dim))
            self.critic_head = nn.Linear(in_dim, 1)

        def compute(self, inputs, role=""):
            """순전파 계산."""
            features = self.backbone(inputs["states"])

            if role == "policy":
                mean = self.actor_head(features)
                log_std = self.log_std_parameter.expand_as(mean)
                return mean, log_std, {}
            elif role == "value":
                value = self.critic_head(features)
                return value, {}

            mean = self.actor_head(features)
            log_std = self.log_std_parameter.expand_as(mean)
            return mean, log_std, {}

    # 체크포인트 로드
    logger.info("PPO 정책 체크포인트 로드: %s", checkpoint_path)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)

    # skrl 체크포인트에서 모델 상태 복원
    # skrl은 에이전트 전체를 저장하므로 policy 모델 상태를 추출한다
    if isinstance(checkpoint, dict) and "policy" in checkpoint:
        state_dict = checkpoint["policy"]
    else:
        state_dict = checkpoint

    logger.info("PPO 정책 로드 완료 (device: %s)", device)
    return state_dict, SharedActorCritic, device


def setup_eval_environment(config, policy_type, render=False):
    """평가용 시뮬레이션 환경을 초기화한다.

    Args:
        config: 학습 설정 딕셔너리
        policy_type: 정책 타입 ("il" 또는 "rl")
        render: 렌더링 활성화 여부

    Returns:
        환경 인스턴스

    Raises:
        ImportError: Isaac Lab이 설치되지 않은 경우
    """
    try:
        import gymnasium as gym
    except ImportError:
        logger.error(
            "gymnasium이 설치되지 않았습니다. 설치: pip install gymnasium"
        )
        raise

    if policy_type == "rl":
        try:
            from training.rl.so101_env import SO101ReachEnvCfg

            env_cfg = SO101ReachEnvCfg()
            # 평가 시에는 단일 환경 사용
            env_cfg.scene.num_envs = 1

            task_name = config.get("environment", {}).get(
                "task_name", "Isaac-SO101-Reach-v0"
            )
            env = gym.make(
                task_name,
                cfg=env_cfg,
                render_mode="human" if render else None,
            )

            logger.info(
                "RL 평가 환경 초기화 완료: %s (render: %s)", task_name, render
            )
            return env

        except ImportError as e:
            logger.error(
                "Isaac Lab 환경을 초기화할 수 없습니다: %s\n"
                "Isaac Sim 5.1.0 + Isaac Lab v2.3.0을 설치하세요.\n"
                "설치 가이드: docs/references.md 참조",
                e,
            )
            raise

    elif policy_type == "il":
        try:
            env_cfg = config.get("env", {})
            env_type = env_cfg.get("type", "gym_manipulator")

            # LeRobot 환경 또는 gymnasium 환경 초기화
            env = gym.make(
                env_type,
                render_mode="human" if render else None,
            )

            logger.info(
                "IL 평가 환경 초기화 완료: %s (render: %s)", env_type, render
            )
            return env

        except ImportError as e:
            logger.error(
                "IL 평가 환경을 초기화할 수 없습니다: %s\n"
                "LeRobot과 관련 의존성을 설치하세요.\n"
                "자세한 내용: docs/references.md 참조",
                e,
            )
            raise

    else:
        raise ValueError(f"지원하지 않는 정책 타입: {policy_type}")


def run_evaluation(env, policy, policy_type, num_episodes, success_threshold, seed):
    """정책 평가를 실행하고 에피소드별 메트릭을 수집한다.

    Args:
        env: 시뮬레이션 환경 인스턴스
        policy: 로드된 정책 (IL: 모델 인스턴스, RL: (state_dict, model_cls, device))
        policy_type: 정책 타입 ("il" 또는 "rl")
        num_episodes: 평가 에피소드 수
        success_threshold: 목표 도달 판정 거리 (m)
        seed: 랜덤 시드

    Returns:
        list[dict]: 에피소드별 메트릭 리스트
    """
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    episode_metrics = []

    logger.info("=== 정책 평가 시작 ===")
    logger.info("정책 타입: %s", policy_type)
    logger.info("에피소드 수: %d", num_episodes)
    logger.info("성공 임계값: %.4f m", success_threshold)

    for ep_idx in range(num_episodes):
        obs, info = env.reset(seed=seed + ep_idx)
        done = False
        truncated = False

        episode_reward = 0.0
        episode_length = 0
        trajectory_errors = []
        min_distance = float("inf")

        while not done and not truncated:
            # 정책 추론
            action = _infer_action(policy, obs, policy_type)

            # 환경 스텝
            obs, reward, done, truncated, info = env.step(action)

            episode_reward += float(reward) if not hasattr(reward, "__len__") else float(reward.sum())
            episode_length += 1

            # 궤적 오차 수집 (info에서 거리 정보 추출)
            distance = _extract_distance(info, obs)
            if distance is not None:
                trajectory_errors.append(distance)
                min_distance = min(min_distance, distance)

        # 에피소드 성공 판정
        success = min_distance < success_threshold if min_distance < float("inf") else False

        # 궤적 오차 RMSE
        if trajectory_errors:
            trajectory_rmse = math.sqrt(
                sum(e ** 2 for e in trajectory_errors) / len(trajectory_errors)
            )
        else:
            trajectory_rmse = float("nan")

        ep_metric = {
            "episode": ep_idx,
            "reward": episode_reward,
            "length": episode_length,
            "success": success,
            "min_distance": min_distance if min_distance < float("inf") else float("nan"),
            "trajectory_rmse": trajectory_rmse,
        }
        episode_metrics.append(ep_metric)

        if (ep_idx + 1) % max(1, num_episodes // 10) == 0:
            logger.info(
                "  [%d/%d] reward=%.3f, length=%d, success=%s, min_dist=%.4f",
                ep_idx + 1,
                num_episodes,
                episode_reward,
                episode_length,
                success,
                min_distance if min_distance < float("inf") else float("nan"),
            )

    logger.info("=== 정책 평가 완료 ===")
    return episode_metrics


def _infer_action(policy, obs, policy_type):
    """정책에서 액션을 추론한다.

    Args:
        policy: 정책 모델 또는 (state_dict, model_cls, device)
        obs: 현재 관측
        policy_type: 정책 타입

    Returns:
        액션 값 (numpy array 또는 tensor)
    """
    import torch
    import numpy as np

    if policy_type == "il":
        # IL (ACT) 정책 추론
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                device = next(policy.parameters()).device
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            else:
                obs_tensor = obs.unsqueeze(0) if obs.dim() == 1 else obs

            action = policy.select_action(obs_tensor)

            if isinstance(action, torch.Tensor):
                action = action.squeeze(0).cpu().numpy()

        return action

    elif policy_type == "rl":
        # RL (PPO) 정책 추론 — skrl 에이전트의 act() 메서드 사용
        state_dict, model_cls, device = policy

        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs_tensor = torch.from_numpy(obs).float().to(device)
            else:
                obs_tensor = obs.to(device)

            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)

            # skrl 형식으로 입력 구성
            inputs = {"states": obs_tensor}
            mean, _, _ = _get_rl_policy_output(state_dict, model_cls, inputs, device)

            action = mean.squeeze(0).cpu().numpy()

        return action

    else:
        raise ValueError(f"지원하지 않는 정책 타입: {policy_type}")


def _get_rl_policy_output(state_dict, model_cls, inputs, device):
    """RL 정책 모델의 출력을 반환한다.

    Args:
        state_dict: 모델 상태 딕셔너리
        model_cls: 모델 클래스
        inputs: 입력 딕셔너리
        device: 디바이스

    Returns:
        tuple: (mean, log_std, extras)
    """
    import torch
    import gymnasium as gym

    # 관측/액션 공간 생성 (state_dict 크기에서 추론)
    # backbone.0.weight의 in_features → obs_dim
    # actor_head.weight의 out_features → action_dim
    obs_dim = None
    action_dim = None

    for key, value in state_dict.items():
        if "backbone.0.weight" in key:
            obs_dim = value.shape[1]
        if "actor_head.weight" in key:
            action_dim = value.shape[0]

    if obs_dim is None or action_dim is None:
        raise ValueError("체크포인트에서 관측/액션 차원을 추론할 수 없습니다")

    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,))
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,))

    model = model_cls(
        observation_space=obs_space,
        action_space=act_space,
        device=device,
        role="policy",
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model.compute(inputs, role="policy")


def _extract_distance(info, obs):
    """환경 info 또는 관측에서 엔드이펙터-목표 거리를 추출한다.

    Args:
        info: 환경 step()의 info 딕셔너리
        obs: 현재 관측

    Returns:
        float 또는 None: 거리 값
    """
    import numpy as np

    # info에서 거리 정보 추출 시도
    if isinstance(info, dict):
        # Isaac Lab 환경에서 제공하는 거리 정보
        distance = info.get("distance_to_target")
        if distance is not None:
            return float(distance) if not hasattr(distance, "__len__") else float(distance[0])

        # 대안: 성공 여부에서 거리 추정
        if "is_success" in info:
            return 0.0 if info["is_success"] else None

    # obs에서 마지막 3개 요소를 목표 상대 위치로 가정 (15차원 관측의 경우)
    if hasattr(obs, "__len__") and len(obs) >= 15:
        obs_array = np.asarray(obs).flatten()
        # 관측 공간: [joint_pos(6), joint_vel(6), target_pos(3)]
        target_relative_pos = obs_array[-3:]
        distance = float(np.linalg.norm(target_relative_pos))
        return distance

    return None


def compute_aggregate_metrics(episode_metrics):
    """에피소드별 메트릭에서 집계 통계를 산출한다.

    Args:
        episode_metrics: 에피소드별 메트릭 리스트

    Returns:
        dict: 집계된 평가 메트릭
    """
    if not episode_metrics:
        return {}

    rewards = [ep["reward"] for ep in episode_metrics]
    lengths = [ep["length"] for ep in episode_metrics]
    successes = [ep["success"] for ep in episode_metrics]
    rmses = [ep["trajectory_rmse"] for ep in episode_metrics
             if not math.isnan(ep["trajectory_rmse"])]
    min_dists = [ep["min_distance"] for ep in episode_metrics
                 if not math.isnan(ep["min_distance"])]

    def _mean(values):
        """리스트의 평균을 계산한다."""
        if not values:
            return float("nan")
        return sum(values) / len(values)

    def _std(values):
        """리스트의 표준편차를 계산한다."""
        if len(values) < 2:
            return float("nan")
        mean = _mean(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance)

    metrics = {
        # 성공률
        "success_rate": _mean([float(s) for s in successes]),
        "num_successes": sum(successes),
        "num_episodes": len(episode_metrics),
        # 에피소드 보상
        "mean_episode_reward": _mean(rewards),
        "std_episode_reward": _std(rewards),
        "min_episode_reward": min(rewards) if rewards else float("nan"),
        "max_episode_reward": max(rewards) if rewards else float("nan"),
        # 에피소드 길이
        "mean_episode_length": _mean(lengths),
        "std_episode_length": _std(lengths),
        # 궤적 오차 (RMSE)
        "mean_trajectory_error": _mean(rmses),
        "std_trajectory_error": _std(rmses),
        # 최소 거리
        "mean_min_distance": _mean(min_dists),
        "std_min_distance": _std(min_dists),
    }

    return metrics


def save_results(metrics, episode_metrics, output_path, policy_type, checkpoint_path):
    """평가 결과를 JSON 파일로 저장한다.

    Args:
        metrics: 집계된 메트릭 딕셔너리
        episode_metrics: 에피소드별 메트릭 리스트
        output_path: JSON 출력 파일 경로
        policy_type: 정책 타입
        checkpoint_path: 평가한 체크포인트 경로
    """
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)

    result = {
        "metadata": {
            "policy_type": policy_type,
            "checkpoint": str(checkpoint_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_episodes": len(episode_metrics),
        },
        "aggregate_metrics": metrics,
        "episode_metrics": episode_metrics,
    }

    # NaN을 null로 변환하여 JSON 호환성 확보
    def _convert_nan(obj):
        """NaN 값을 None으로 변환한다."""
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        elif isinstance(obj, dict):
            return {k: _convert_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert_nan(v) for v in obj]
        return obj

    result = _convert_nan(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("평가 결과 저장 완료: %s", output_path)


def print_summary(metrics, policy_type):
    """평가 결과 요약을 콘솔에 출력한다.

    Args:
        metrics: 집계된 메트릭 딕셔너리
        policy_type: 정책 타입
    """
    policy_name = "ACT (IL)" if policy_type == "il" else "PPO (RL)"

    logger.info("=" * 60)
    logger.info("평가 결과 요약 — %s", policy_name)
    logger.info("=" * 60)
    logger.info(
        "에피소드 수: %d (성공: %d)",
        metrics.get("num_episodes", 0),
        metrics.get("num_successes", 0),
    )
    logger.info("성공률: %.2f%%", metrics.get("success_rate", 0) * 100)
    logger.info("-" * 40)
    logger.info(
        "평균 보상: %.4f ± %.4f",
        metrics.get("mean_episode_reward", float("nan")),
        metrics.get("std_episode_reward", float("nan")),
    )
    logger.info(
        "평균 에피소드 길이: %.1f ± %.1f",
        metrics.get("mean_episode_length", float("nan")),
        metrics.get("std_episode_length", float("nan")),
    )
    logger.info(
        "평균 궤적 오차 (RMSE): %.4f ± %.4f",
        metrics.get("mean_trajectory_error", float("nan")),
        metrics.get("std_trajectory_error", float("nan")),
    )
    logger.info(
        "평균 최소 거리: %.4f ± %.4f m",
        metrics.get("mean_min_distance", float("nan")),
        metrics.get("std_min_distance", float("nan")),
    )
    logger.info("=" * 60)


def main():
    """정책 평가 메인 엔트리포인트."""
    args = parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=== Policy Evaluation ===")
    logger.info("정책 타입: %s", args.policy_type)
    logger.info("체크포인트: %s", args.checkpoint)
    logger.info("에피소드 수: %d", args.num_episodes)

    # 1. 체크포인트 존재 확인
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error("체크포인트 파일을 찾을 수 없습니다: %s", checkpoint_path)
        sys.exit(1)

    # 2. 설정 로드
    config_path = args.config or get_default_config_path(args.policy_type)
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        logger.warning(
            "설정 파일을 찾을 수 없습니다: %s. 기본 설정을 사용합니다.",
            config_path,
        )
        config = {}

    # 3. 성공 임계값 결정
    success_threshold = get_success_threshold(
        config, args.policy_type, args.success_threshold
    )
    logger.info("성공 임계값: %.4f m", success_threshold)

    # 4. 정책 로드
    logger.info("정책 로드 중...")
    if args.policy_type == "il":
        policy = load_il_policy(args.checkpoint, config)
    elif args.policy_type == "rl":
        policy = load_rl_policy(args.checkpoint, config)
    else:
        logger.error("지원하지 않는 정책 타입: %s", args.policy_type)
        sys.exit(1)

    # 5. 평가 환경 초기화
    logger.info("평가 환경 초기화 중...")
    env = setup_eval_environment(config, args.policy_type, render=args.render)

    # 6. 평가 실행
    try:
        episode_metrics = run_evaluation(
            env=env,
            policy=policy,
            policy_type=args.policy_type,
            num_episodes=args.num_episodes,
            success_threshold=success_threshold,
            seed=args.seed,
        )
    except KeyboardInterrupt:
        logger.info("평가가 사용자에 의해 중단되었습니다.")
        episode_metrics = []
    finally:
        env.close()
        logger.info("평가 환경 종료")

    # 7. 메트릭 집계
    metrics = compute_aggregate_metrics(episode_metrics)

    # 8. 결과 저장
    output_path = args.output or f"training/eval/results_{args.policy_type}.json"
    save_results(metrics, episode_metrics, output_path, args.policy_type, args.checkpoint)

    # 9. 콘솔 요약 출력
    print_summary(metrics, args.policy_type)

    logger.info("=== Policy Evaluation 완료 ===")


if __name__ == "__main__":
    main()
