#!/usr/bin/env python3
"""evaluate_policy.py — 학습된 정책 평가 스크립트

목적:
    학습된 IL (ACT) 또는 RL (PPO) 정책을 Isaac Sim/Lab 환경에서 평가한다.
    - 학습된 체크포인트 로드 (IL: ACT, RL: skrl PPO)
    - Isaac Sim/Lab 평가 환경 초기화
    - N개 에피소드 실행 및 메트릭 수집
    - 성공률, 평균 궤적 오차, 총 보상 등 메트릭 로깅
    - ValidationReport 패턴으로 결과 요약 출력

사용법:
    # IL 정책 평가
    python training/eval/evaluate_policy.py \\
        --policy_type il \\
        --checkpoint_path training/il/checkpoints/checkpoint_0100000.pt \\
        --task LeIsaac-SO101-PickOrange-v0 \\
        --num_episodes 10 \\
        --headless

    # RL 정책 평가
    python training/eval/evaluate_policy.py \\
        --policy_type rl \\
        --checkpoint_path training/rl/checkpoints/final_checkpoint/agent.pt \\
        --task Isaac-SO101-Reach-v0 \\
        --num_episodes 50 \\
        --headless

    # 도움말
    python training/eval/evaluate_policy.py --help

필요 환경:
    - Isaac Sim 5.1.0 + Isaac Lab v2.3.0 (RL 평가 시)
    - LeRobot 0.4.4 (IL 평가 시)
    - skrl (RL 평가 시)
    - PyTorch 2.x
    - conda env: soarm
    - GPU: NVIDIA RTX 4090 Laptop (권장)

Phase: 6
상태: 구현 완료 — 학습된 정책 시뮬 평가
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Heavy dependency imports are guarded so the script can be
# syntax-checked and --help can run without these dependencies.
# ---------------------------------------------------------------------------
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import numpy as np
except ImportError:
    np = None

# LeRobot ACT policy
try:
    from lerobot.common.policies.act.configuration_act import ACTConfig
    from lerobot.common.policies.act.modeling_act import ACTPolicy

    LEROBOT_ACT_AVAILABLE = True
except ImportError:
    ACTConfig = None
    ACTPolicy = None
    LEROBOT_ACT_AVAILABLE = False

# Isaac Lab — try both namespace variants
try:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

    ISAAC_LAB_AVAILABLE = True
except ImportError:
    try:
        from isaaclab.envs import ManagerBasedRLEnv

        ISAAC_LAB_AVAILABLE = True
    except ImportError:
        ISAAC_LAB_AVAILABLE = False

# Isaac Sim (ManagerBasedEnv for IL evaluation)
try:
    from isaaclab.envs import ManagerBasedEnv

    ISAAC_SIM_AVAILABLE = True
except ImportError:
    ISAAC_SIM_AVAILABLE = False

# skrl — RL agent loading
try:
    from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
    from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
    from skrl.envs.wrappers.torch import wrap_env

    SKRL_AVAILABLE = True
except ImportError:
    SKRL_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- Parameter file paths ---
_SCRIPT_DIR = Path(__file__).resolve().parent
_IL_CONFIG_YAML = _SCRIPT_DIR.parent / "il" / "config.yaml"
_RL_CONFIG_YAML = _SCRIPT_DIR.parent / "rl" / "config.yaml"
_PARAMS_DIR = _SCRIPT_DIR.parent.parent / "params"
_CONTROL_YAML = _PARAMS_DIR / "control.yaml"


def _load_yaml(path: Path) -> dict:
    """Load a YAML parameter file and return its contents."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _get_value(param):
    """Extract the 'value' field from a parameter dict.

    Handles both top-level {value: ...} and plain values.
    """
    if isinstance(param, dict) and "value" in param:
        return param["value"]
    return param


# ---------------------------------------------------------------------------
# Evaluation report (follows scripts/validate_dataset.py ValidationReport)
# ---------------------------------------------------------------------------


class EvaluationReport:
    """Collects PASS/FAIL results and metrics for each evaluation check."""

    def __init__(self):
        self.results = []
        self.metrics = {}

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        """Record a single check result."""
        self.results.append(
            {"name": name, "passed": passed, "detail": detail}
        )

    def set_metric(self, name: str, value: float) -> None:
        """Record a numeric metric."""
        self.metrics[name] = value

    @property
    def all_passed(self) -> bool:
        return all(r["passed"] for r in self.results)

    def log_report(self) -> None:
        """Log the full evaluation report."""
        logger.info("=== Evaluation Report ===")
        for r in self.results:
            status = "PASS" if r["passed"] else "FAIL"
            detail = f" — {r['detail']}" if r["detail"] else ""
            logger.info("  [%s] %s%s", status, r["name"], detail)

        if self.metrics:
            logger.info("--- Metrics ---")
            for name, value in self.metrics.items():
                logger.info("  %s: %.6f", name, value)

        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed
        logger.info("---")
        logger.info(
            "Total: %d | Passed: %d | Failed: %d", total, passed, failed
        )


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------


def load_il_policy(checkpoint_path: str, device: "torch.device"):
    """Load a trained IL (ACT) policy from a checkpoint.

    Args:
        checkpoint_path: Path to the ACT checkpoint file (.pt).
        device: Torch device to load the model onto.

    Returns:
        Loaded ACT policy in eval mode.

    Raises:
        RuntimeError: If LeRobot ACT or PyTorch is not available.
        FileNotFoundError: If checkpoint file does not exist.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not available. "
            "Install with: pip install torch"
        )
    if not LEROBOT_ACT_AVAILABLE:
        raise RuntimeError(
            "LeRobot ACT is not available. "
            "Install with: pip install lerobot"
        )

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"IL checkpoint not found: {checkpoint_file}"
        )

    logger.info("Loading IL (ACT) checkpoint: %s", checkpoint_file)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_file, map_location=device)

    # Load IL config for policy dimensions
    il_cfg = _load_yaml(_IL_CONFIG_YAML)
    policy_cfg = il_cfg["policy"]

    # Build ACT config from saved hyperparameters
    act_config = ACTConfig(
        chunk_size=_get_value(policy_cfg["chunk_size"]),
        n_obs_steps=_get_value(policy_cfg["n_obs_steps"]),
        dim_model=_get_value(policy_cfg["dim_model"]),
        n_heads=_get_value(policy_cfg["n_heads"]),
        n_encoder_layers=_get_value(policy_cfg["n_encoder_layers"]),
        n_decoder_layers=_get_value(policy_cfg["n_decoder_layers"]),
    )

    # Create policy and load weights
    policy = ACTPolicy(config=act_config)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy = policy.to(device)
    policy.eval()

    logger.info("IL policy loaded successfully (step=%d).", checkpoint["step"])
    return policy


def load_rl_policy(checkpoint_path: str, env, device: "torch.device"):
    """Load a trained RL (PPO) agent from a checkpoint.

    Args:
        checkpoint_path: Path to the skrl agent checkpoint file.
        env: Wrapped skrl-compatible environment (for spaces).
        device: Torch device to load the model onto.

    Returns:
        Loaded skrl PPO agent in eval mode.

    Raises:
        RuntimeError: If skrl or PyTorch is not available.
        FileNotFoundError: If checkpoint file does not exist.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not available. "
            "Install with: pip install torch"
        )
    if not SKRL_AVAILABLE:
        raise RuntimeError(
            "skrl is not available. "
            "Install with: pip install skrl"
        )

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"RL checkpoint not found: {checkpoint_file}"
        )

    logger.info("Loading RL (PPO) checkpoint: %s", checkpoint_file)

    # Load RL config for network architecture
    rl_cfg = _load_yaml(_RL_CONFIG_YAML)
    net_cfg = rl_cfg["network"]
    policy_hidden_dims = _get_value(net_cfg["policy_hidden_dims"])
    value_hidden_dims = _get_value(net_cfg["value_hidden_dims"])
    activation = _get_value(net_cfg["activation"])

    # Import network definitions from train_rl module
    from training.rl.train_rl import PolicyNetwork, ValueNetwork

    # Create models
    models = {}
    models["policy"] = PolicyNetwork(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        hidden_dims=policy_hidden_dims,
        activation=activation,
    )
    models["value"] = ValueNetwork(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        hidden_dims=value_hidden_dims,
        activation=activation,
    )

    # Configure a minimal PPO agent for evaluation
    cfg = PPO_DEFAULT_CONFIG.copy()
    agent = PPO(
        models=models,
        memory=None,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # Load checkpoint weights
    agent.load(str(checkpoint_file))

    logger.info("RL agent loaded successfully.")
    return agent


# ---------------------------------------------------------------------------
# Environment initialization
# ---------------------------------------------------------------------------


def initialize_eval_env(
    task: str, policy_type: str, num_envs: int, headless: bool
):
    """Initialize the evaluation environment.

    Args:
        task: Isaac Sim/Lab task name.
        policy_type: Policy type ('il' or 'rl').
        num_envs: Number of parallel environments.
        headless: Whether to run without GUI.

    Returns:
        Environment instance (wrapped for skrl if RL).

    Raises:
        RuntimeError: If Isaac Sim/Lab is not available.
    """
    if policy_type == "rl":
        if not ISAAC_LAB_AVAILABLE:
            raise RuntimeError(
                "Isaac Lab is not available. "
                "Run this script inside the Isaac Sim/Lab Python environment."
            )

        from training.rl.so101_env import SO101ReachEnvCfg

        logger.info(
            "Initializing RL evaluation environment: task=%s, "
            "num_envs=%d, headless=%s",
            task,
            num_envs,
            headless,
        )

        env_cfg = SO101ReachEnvCfg()
        env_cfg.scene.num_envs = num_envs
        env = ManagerBasedRLEnv(cfg=env_cfg)

        if SKRL_AVAILABLE:
            env = wrap_env(env, wrapper="isaaclab")

        return env

    else:  # IL
        if not ISAAC_SIM_AVAILABLE:
            raise RuntimeError(
                "Isaac Sim is not available. "
                "Run this script inside the Isaac Sim Python environment."
            )

        logger.info(
            "Initializing IL evaluation environment: task=%s, "
            "num_envs=%d, headless=%s",
            task,
            num_envs,
            headless,
        )

        env = ManagerBasedEnv(task=task, num_envs=num_envs)
        return env


# ---------------------------------------------------------------------------
# Episode evaluation
# ---------------------------------------------------------------------------


def evaluate_il_episode(
    env, policy, episode_idx: int, max_steps: int, device: "torch.device"
) -> dict:
    """Run a single IL evaluation episode and collect metrics.

    Args:
        env: Isaac Sim environment instance.
        policy: ACT policy in eval mode.
        episode_idx: Current episode index (for logging).
        max_steps: Maximum steps per episode.
        device: Torch device.

    Returns:
        dict with episode metrics: total_reward, trajectory_error,
        num_steps, success.
    """
    obs = env.reset()
    total_reward = 0.0
    trajectory_errors = []
    num_steps = 0
    success = False

    for step_idx in range(max_steps):
        # Prepare observation for policy
        state = obs["joint_pos"]
        if hasattr(state, "cpu"):
            state_tensor = state.float().to(device)
        else:
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=device
            )

        # Get action from policy
        with torch.no_grad():
            obs_dict = {"observation.state": state_tensor.unsqueeze(0)}
            action = policy.select_action(obs_dict)
            if hasattr(action, "squeeze"):
                action = action.squeeze(0)

        # Convert action for environment
        if hasattr(action, "cpu"):
            action_np = action.cpu().numpy()
        else:
            action_np = np.asarray(action)

        obs, reward, terminated, truncated, info = env.step(action_np)

        # Accumulate metrics
        if hasattr(reward, "item"):
            total_reward += reward.sum().item()
        else:
            total_reward += float(reward)

        num_steps += 1

        # Track trajectory error (distance between current and target)
        if "target_pos" in obs and "ee_pos" in obs:
            target = obs["target_pos"]
            ee = obs["ee_pos"]
            if hasattr(target, "cpu"):
                target = target.cpu().numpy()
                ee = ee.cpu().numpy()
            error = np.linalg.norm(
                np.asarray(target) - np.asarray(ee)
            )
            trajectory_errors.append(error)

        # Check success from info
        if isinstance(info, dict) and info.get("success", False):
            success = True

        # End episode on termination or truncation
        if hasattr(terminated, "any"):
            if terminated.any() or truncated.any():
                break
        elif terminated or truncated:
            break

    mean_traj_error = (
        float(np.mean(trajectory_errors)) if trajectory_errors else 0.0
    )

    return {
        "total_reward": total_reward,
        "trajectory_error": mean_traj_error,
        "num_steps": num_steps,
        "success": success,
    }


def evaluate_rl_episode(
    env, agent, episode_idx: int, max_steps: int, device: "torch.device"
) -> dict:
    """Run a single RL evaluation episode and collect metrics.

    Args:
        env: Wrapped skrl-compatible environment.
        agent: skrl PPO agent.
        episode_idx: Current episode index (for logging).
        max_steps: Maximum steps per episode.
        device: Torch device.

    Returns:
        dict with episode metrics: total_reward, trajectory_error,
        num_steps, success.
    """
    obs, info = env.reset()
    total_reward = 0.0
    trajectory_errors = []
    num_steps = 0
    success = False

    for step_idx in range(max_steps):
        # Get action from agent
        with torch.no_grad():
            action = agent.act(obs, timestep=step_idx, timesteps=max_steps)
            if isinstance(action, tuple):
                action = action[0]

        obs, reward, terminated, truncated, info = env.step(action)

        # Accumulate metrics
        if hasattr(reward, "item"):
            total_reward += reward.sum().item()
        elif hasattr(reward, "sum"):
            total_reward += float(reward.sum())
        else:
            total_reward += float(reward)

        num_steps += 1

        # Track trajectory error from info or observations
        if isinstance(info, dict) and "trajectory_error" in info:
            err = info["trajectory_error"]
            if hasattr(err, "item"):
                trajectory_errors.append(err.item())
            else:
                trajectory_errors.append(float(err))

        # Check success from info
        if isinstance(info, dict) and info.get("success", False):
            success = True

        # End episode on termination
        if hasattr(terminated, "any"):
            if terminated.any() or truncated.any():
                break
        elif terminated or truncated:
            break

    mean_traj_error = (
        float(np.mean(trajectory_errors)) if trajectory_errors else 0.0
    )

    return {
        "total_reward": total_reward,
        "trajectory_error": mean_traj_error,
        "num_steps": num_steps,
        "success": success,
    }


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------


def run_evaluation(args: argparse.Namespace) -> bool:
    """Run the full policy evaluation pipeline.

    Args:
        args: Parsed CLI arguments.

    Returns:
        True if evaluation completed successfully, False otherwise.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not available. "
            "Install with: pip install torch"
        )
    if np is None:
        raise RuntimeError(
            "NumPy is not available. "
            "Install with: pip install numpy"
        )

    # Resolve parameters
    policy_type = args.policy_type
    checkpoint_path = args.checkpoint_path
    task = args.task
    num_episodes = args.num_episodes
    max_steps = args.max_steps
    headless = args.headless
    success_threshold = args.success_threshold

    logger.info("=== SO-ARM101 Policy Evaluation ===")
    logger.info("Policy type:        %s", policy_type)
    logger.info("Checkpoint:         %s", checkpoint_path)
    logger.info("Task:               %s", task)
    logger.info("Num episodes:       %d", num_episodes)
    logger.info("Max steps/episode:  %d", max_steps)
    logger.info("Headless:           %s", headless)
    logger.info("Success threshold:  %.4f", success_threshold)

    report = EvaluationReport()

    # --- Check checkpoint exists ---
    checkpoint_file = Path(checkpoint_path)
    report.add(
        "체크포인트 파일 존재",
        checkpoint_file.exists(),
        str(checkpoint_file),
    )
    if not checkpoint_file.exists():
        report.log_report()
        return report.all_passed

    # --- Select device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Initialize evaluation environment ---
    logger.info("Initializing evaluation environment...")
    env = initialize_eval_env(
        task=task,
        policy_type=policy_type,
        num_envs=1,
        headless=headless,
    )
    report.add("평가 환경 초기화", True, f"task={task}")

    # --- Load policy ---
    logger.info("Loading trained policy...")
    if policy_type == "il":
        policy = load_il_policy(checkpoint_path, device)
        report.add("IL 정책 로드", True, str(checkpoint_file))
    else:
        policy = load_rl_policy(checkpoint_path, env, device)
        report.add("RL 에이전트 로드", True, str(checkpoint_file))

    # --- Run evaluation episodes ---
    logger.info("Running %d evaluation episodes...", num_episodes)
    episode_results = []

    for ep_idx in range(num_episodes):
        logger.info(
            "Episode %d/%d ...", ep_idx + 1, num_episodes
        )
        start_time = time.time()

        if policy_type == "il":
            result = evaluate_il_episode(
                env=env,
                policy=policy,
                episode_idx=ep_idx,
                max_steps=max_steps,
                device=device,
            )
        else:
            result = evaluate_rl_episode(
                env=env,
                agent=policy,
                episode_idx=ep_idx,
                max_steps=max_steps,
                device=device,
            )

        elapsed = time.time() - start_time
        result["elapsed_time"] = elapsed
        episode_results.append(result)

        logger.info(
            "  Episode %d: reward=%.4f, traj_error=%.6f, "
            "steps=%d, success=%s, time=%.2fs",
            ep_idx + 1,
            result["total_reward"],
            result["trajectory_error"],
            result["num_steps"],
            result["success"],
            elapsed,
        )

    # --- Compute aggregate metrics ---
    total_rewards = [r["total_reward"] for r in episode_results]
    traj_errors = [r["trajectory_error"] for r in episode_results]
    successes = [r["success"] for r in episode_results]
    steps = [r["num_steps"] for r in episode_results]

    mean_reward = float(np.mean(total_rewards))
    std_reward = float(np.std(total_rewards))
    mean_traj_error = float(np.mean(traj_errors))
    success_rate = sum(1 for s in successes if s) / len(successes)
    mean_steps = float(np.mean(steps))

    # Record metrics
    report.set_metric("mean_reward", mean_reward)
    report.set_metric("std_reward", std_reward)
    report.set_metric("mean_trajectory_error", mean_traj_error)
    report.set_metric("success_rate", success_rate)
    report.set_metric("mean_episode_steps", mean_steps)

    # --- Validation checks ---
    report.add(
        "에피소드 실행 완료",
        len(episode_results) == num_episodes,
        f"{len(episode_results)}/{num_episodes} 에피소드 완료",
    )

    report.add(
        "평균 보상",
        mean_reward > 0,
        f"mean={mean_reward:.4f}, std={std_reward:.4f}",
    )

    report.add(
        "성공률",
        True,
        f"{success_rate * 100:.1f}% ({sum(successes)}/{len(successes)})",
    )

    report.add(
        "평균 궤적 오차",
        True,
        f"{mean_traj_error:.6f}",
    )

    # --- Print report ---
    report.log_report()

    # Cleanup
    env.close()
    logger.info("Environment closed. Evaluation complete.")

    return report.all_passed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for policy evaluation.

    Supports both IL (ACT) and RL (PPO) policy evaluation.
    """
    parser = argparse.ArgumentParser(
        description="SO-ARM101 학습된 정책 평가 (IL/RL → Isaac Sim/Lab)"
    )
    parser.add_argument(
        "--policy_type",
        type=str,
        required=True,
        choices=["il", "rl"],
        help="정책 타입: 'il' (ACT Imitation Learning) 또는 'rl' (PPO RL)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="학습된 정책 체크포인트 파일 경로 "
        "(IL: .pt, RL: agent.pt)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-SO101-Reach-v0",
        help="Isaac Sim/Lab 평가 환경 태스크 이름 "
        "(기본값: Isaac-SO101-Reach-v0)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="평가 에피소드 수 (기본값: 10)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="에피소드당 최대 스텝 수 (기본값: 200)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="헤드리스 모드 (GUI 없이 평가)",
    )
    parser.add_argument(
        "--success_threshold",
        type=float,
        default=0.02,
        help="목표 도달 성공 임계값 (m, 기본값: 0.02)",
    )
    parser.add_argument(
        "--il_config",
        type=str,
        default=str(_IL_CONFIG_YAML),
        help="IL 학습 설정 YAML 파일 경로",
    )
    parser.add_argument(
        "--rl_config",
        type=str,
        default=str(_RL_CONFIG_YAML),
        help="RL 학습 설정 YAML 파일 경로",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for policy evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    # Allow overriding YAML paths via CLI
    global _IL_CONFIG_YAML, _RL_CONFIG_YAML
    _IL_CONFIG_YAML = Path(args.il_config)
    _RL_CONFIG_YAML = Path(args.rl_config)

    try:
        all_passed = run_evaluation(args)
    except RuntimeError as exc:
        logger.error("Evaluation failed: %s", exc)
        sys.exit(1)
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Unexpected error during evaluation: %s", exc)
        sys.exit(1)

    if not all_passed:
        logger.error("평가 실패: 일부 항목이 FAIL입니다.")
        sys.exit(1)

    logger.info("평가 완료: 모든 항목 PASS.")


if __name__ == "__main__":
    main()
