#!/usr/bin/env python3
"""train_rl.py — skrl PPO 강화학습 학습 스크립트

목적:
    Isaac Lab 환경에서 SO-ARM101 Reach task 강화학습을 실행한다.
    - training/rl/config.yaml에서 PPO 하이퍼파라미터 로드
    - params/control.yaml에서 드라이브 게인/조인트 설정 로드
    - skrl PPO 에이전트 구성 및 학습 실행
    - 체크포인트를 training/rl/checkpoints/에 저장

사용법:
    # 기본 설정으로 학습 실행
    python training/rl/train_rl.py

    # CLI 인자로 오버라이드
    python training/rl/train_rl.py \\
        --task Isaac-SO101-Reach-v0 \\
        --num_envs 1024 \\
        --headless

    # 짧은 테스트 학습
    python training/rl/train_rl.py --num_envs 4 --max_iterations 100

필요 환경:
    - Isaac Sim 5.1.0 + Isaac Lab v2.3.0
    - skrl (RL training framework)
    - conda env: soarm
    - GPU: NVIDIA RTX 4090 Laptop (권장)

Phase: 6
상태: 구현 완료 — skrl PPO RL 학습 파이프라인
"""

import argparse
import logging
import sys
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

# skrl — RL training framework
try:
    from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
    from skrl.memories.torch import RandomMemory
    from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
    from skrl.trainers.torch import SequentialTrainer
    from skrl.envs.wrappers.torch import wrap_env

    SKRL_AVAILABLE = True
except ImportError:
    SKRL_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- Parameter file paths ---
_SCRIPT_DIR = Path(__file__).resolve().parent
_CONFIG_YAML = _SCRIPT_DIR / "config.yaml"
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


def load_rl_config() -> dict:
    """Load training/rl/config.yaml and return raw dict."""
    raw = _load_yaml(_CONFIG_YAML)
    return raw


def load_control_params() -> dict:
    """Load params/control.yaml and return raw dict."""
    raw = _load_yaml(_CONTROL_YAML)
    return raw


# ---------------------------------------------------------------------------
# skrl model definitions — Policy and Value networks
# ---------------------------------------------------------------------------
if SKRL_AVAILABLE and TORCH_AVAILABLE:

    def _get_activation(name: str):
        """Return a PyTorch activation module by name.

        Args:
            name: Activation function name (elu, relu, tanh, selu).

        Returns:
            torch.nn activation module instance.
        """
        activations = {
            "elu": torch.nn.ELU(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
            "selu": torch.nn.SELU(),
        }
        if name.lower() not in activations:
            logger.warning(
                "Unknown activation '%s', falling back to ELU", name
            )
            return torch.nn.ELU()
        return activations[name.lower()]

    class PolicyNetwork(GaussianMixin, Model):
        """Gaussian policy network for PPO (actor).

        Outputs mean and log_std for a diagonal Gaussian distribution
        over joint position actions.
        """

        def __init__(
            self,
            observation_space,
            action_space,
            device,
            hidden_dims=None,
            activation="elu",
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-20.0,
            max_log_std=2.0,
            initial_log_std=0.0,
        ):
            Model.__init__(
                self, observation_space, action_space, device
            )
            GaussianMixin.__init__(
                self,
                clip_actions=clip_actions,
                clip_log_std=clip_log_std,
                min_log_std=min_log_std,
                max_log_std=max_log_std,
                initial_log_std=initial_log_std,
            )

            if hidden_dims is None:
                hidden_dims = [256, 128, 64]

            activation_fn = _get_activation(activation)

            # Build MLP layers
            layers = []
            in_dim = self.num_observations
            for h_dim in hidden_dims:
                layers.append(torch.nn.Linear(in_dim, h_dim))
                layers.append(activation_fn)
                in_dim = h_dim
            layers.append(torch.nn.Linear(in_dim, self.num_actions))

            self.net = torch.nn.Sequential(*layers)

        def compute(self, inputs, role):
            """Forward pass — compute action mean from observations."""
            return self.net(inputs["states"]), {}

    class ValueNetwork(DeterministicMixin, Model):
        """Deterministic value network for PPO (critic).

        Outputs a scalar state value estimate.
        """

        def __init__(
            self,
            observation_space,
            action_space,
            device,
            hidden_dims=None,
            activation="elu",
            clip_actions=False,
        ):
            Model.__init__(
                self, observation_space, action_space, device
            )
            DeterministicMixin.__init__(
                self, clip_actions=clip_actions
            )

            if hidden_dims is None:
                hidden_dims = [256, 128, 64]

            activation_fn = _get_activation(activation)

            # Build MLP layers
            layers = []
            in_dim = self.num_observations
            for h_dim in hidden_dims:
                layers.append(torch.nn.Linear(in_dim, h_dim))
                layers.append(activation_fn)
                in_dim = h_dim
            layers.append(torch.nn.Linear(in_dim, 1))

            self.net = torch.nn.Sequential(*layers)

        def compute(self, inputs, role):
            """Forward pass — compute state value from observations."""
            return self.net(inputs["states"]), {}


def create_environment(task: str, num_envs: int, headless: bool):
    """Create the Isaac Lab RL environment.

    Args:
        task: Isaac Lab task name (e.g. 'Isaac-SO101-Reach-v0').
        num_envs: Number of parallel environments.
        headless: Whether to run without GUI.

    Returns:
        Wrapped skrl-compatible environment.

    Raises:
        RuntimeError: If Isaac Lab is not available.
    """
    if not ISAAC_LAB_AVAILABLE:
        raise RuntimeError(
            "Isaac Lab is not available. "
            "Run this script inside the Isaac Sim/Lab Python environment."
        )
    if not SKRL_AVAILABLE:
        raise RuntimeError(
            "skrl is not available. "
            "Install with: pip install skrl"
        )

    # Import environment configuration from so101_env.py
    from training.rl.so101_env import SO101ReachEnvCfg

    logger.info(
        "Creating Isaac Lab environment: task=%s, num_envs=%d, headless=%s",
        task,
        num_envs,
        headless,
    )

    # Configure environment
    env_cfg = SO101ReachEnvCfg()
    env_cfg.scene.num_envs = num_envs

    # Create the RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Wrap for skrl compatibility
    env = wrap_env(env, wrapper="isaaclab")

    logger.info("Environment created and wrapped for skrl.")
    return env


def configure_ppo_agent(
    env,
    rl_cfg: dict,
    device: "torch.device",
) -> "PPO":
    """Configure a skrl PPO agent from YAML configuration.

    Args:
        env: Wrapped skrl-compatible environment.
        rl_cfg: Loaded training/rl/config.yaml dict.
        device: Torch device (cuda or cpu).

    Returns:
        Configured PPO agent instance.
    """
    algo_cfg = rl_cfg["algorithm"]
    net_cfg = rl_cfg["network"]
    train_cfg = rl_cfg["training"]

    # Extract hyperparameters
    lr = _get_value(algo_cfg["learning_rate"])
    gamma = _get_value(algo_cfg["gamma"])
    lam = _get_value(algo_cfg["lam"])
    clip_range = _get_value(algo_cfg["clip_range"])
    entropy_coef = _get_value(algo_cfg["entropy_coef"])
    value_loss_coef = _get_value(algo_cfg["value_loss_coef"])
    max_grad_norm = _get_value(algo_cfg["max_grad_norm"])
    n_epochs = _get_value(algo_cfg["n_epochs"])
    n_minibatches = _get_value(algo_cfg["n_minibatches"])
    horizon_length = _get_value(algo_cfg["horizon_length"])

    # Network architecture
    policy_hidden_dims = _get_value(net_cfg["policy_hidden_dims"])
    value_hidden_dims = _get_value(net_cfg["value_hidden_dims"])
    activation = _get_value(net_cfg["activation"])

    logger.info("Configuring PPO agent:")
    logger.info("  learning_rate:    %e", lr)
    logger.info("  gamma:            %.4f", gamma)
    logger.info("  lambda:           %.4f", lam)
    logger.info("  clip_range:       %.2f", clip_range)
    logger.info("  entropy_coef:     %.4f", entropy_coef)
    logger.info("  value_loss_coef:  %.2f", value_loss_coef)
    logger.info("  max_grad_norm:    %.1f", max_grad_norm)
    logger.info("  n_epochs:         %d", n_epochs)
    logger.info("  n_minibatches:    %d", n_minibatches)
    logger.info("  horizon_length:   %d", horizon_length)
    logger.info("  policy_dims:      %s", policy_hidden_dims)
    logger.info("  value_dims:       %s", value_hidden_dims)
    logger.info("  activation:       %s", activation)

    # Create policy and value networks
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

    # Create replay memory
    num_envs = env.num_envs
    memory = RandomMemory(
        memory_size=horizon_length,
        num_envs=num_envs,
        device=device,
    )

    # Configure PPO hyperparameters
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = horizon_length
    cfg["learning_epochs"] = n_epochs
    cfg["mini_batches"] = n_minibatches
    cfg["discount_factor"] = gamma
    cfg["lambda"] = lam
    cfg["learning_rate"] = lr
    cfg["grad_norm_clip"] = max_grad_norm
    cfg["ratio_clip"] = clip_range
    cfg["value_clip"] = clip_range
    cfg["entropy_loss_scale"] = entropy_coef
    cfg["value_loss_scale"] = value_loss_coef
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0

    # Logging and checkpointing
    seed = _get_value(train_cfg["seed"])
    cfg["seed"] = seed

    # Create PPO agent
    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    logger.info("PPO agent configured (seed=%d).", seed)
    return agent


def run_training(args: argparse.Namespace) -> None:
    """Run the RL training pipeline.

    Args:
        args: Parsed CLI arguments.
    """
    # Load configuration files
    rl_cfg = load_rl_config()
    control_params = load_control_params()

    # Resolve effective values (CLI overrides > YAML defaults)
    task = (
        args.task if args.task is not None
        else _get_value(rl_cfg["environment"]["task"])
    )
    num_envs = (
        args.num_envs if args.num_envs is not None
        else _get_value(rl_cfg["environment"]["num_envs"])
    )
    headless = args.headless if args.headless is not None else _get_value(
        rl_cfg["training"]["headless"]
    )
    max_iterations = (
        args.max_iterations if args.max_iterations is not None
        else _get_value(rl_cfg["training"]["max_iterations"])
    )
    checkpoint_freq = (
        args.checkpoint_freq if args.checkpoint_freq is not None
        else _get_value(rl_cfg["training"]["checkpoint_freq"])
    )
    output_dir = Path(
        args.output_dir if args.output_dir is not None
        else _get_value(rl_cfg["training"]["output_dir"])
    )
    seed = args.seed if args.seed is not None else _get_value(
        rl_cfg["training"]["seed"]
    )

    logger.info("=== SO-ARM101 RL Training (skrl PPO) ===")
    logger.info("Task:              %s", task)
    logger.info("Num envs:          %d", num_envs)
    logger.info("Headless:          %s", headless)
    logger.info("Max iterations:    %d", max_iterations)
    logger.info("Checkpoint freq:   %d", checkpoint_freq)
    logger.info("Output dir:        %s", output_dir)
    logger.info("Seed:              %d", seed)
    logger.info("")
    logger.info("--- Drive Gains (params/control.yaml) ---")
    logger.info(
        "  stiffness: %.1f, damping: %.1f",
        _get_value(control_params["drive"]["stiffness"]),
        _get_value(control_params["drive"]["damping"]),
    )

    # Check dependencies
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not available. "
            "Install with: pip install torch"
        )
    if not ISAAC_LAB_AVAILABLE:
        raise RuntimeError(
            "Isaac Lab is not available. "
            "Run this script inside the Isaac Sim/Lab Python environment."
        )
    if not SKRL_AVAILABLE:
        raise RuntimeError(
            "skrl is not available. "
            "Install with: pip install skrl"
        )

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = create_environment(
        task=task,
        num_envs=num_envs,
        headless=headless,
    )

    # Configure PPO agent
    agent = configure_ppo_agent(
        env=env,
        rl_cfg=rl_cfg,
        device=device,
    )

    # Create trainer
    trainer_cfg = {
        "timesteps": max_iterations * _get_value(
            rl_cfg["algorithm"]["horizon_length"]
        ),
        "headless": headless,
    }

    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        cfg=trainer_cfg,
    )

    # Run training
    logger.info("Starting RL training...")
    trainer.train()

    # Save final checkpoint
    checkpoint_path = output_dir / "final_checkpoint"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    agent.save(str(checkpoint_path / "agent.pt"))
    logger.info("Final checkpoint saved: %s", checkpoint_path)

    # Training summary
    logger.info("=== RL Training Summary ===")
    logger.info("Task:              %s", task)
    logger.info("Num envs:          %d", num_envs)
    logger.info("Max iterations:    %d", max_iterations)
    logger.info("Checkpoints:       %s", output_dir)
    logger.info("Config:            %s", _CONFIG_YAML)
    logger.info("Control params:    %s", _CONTROL_YAML)
    logger.info("Training complete.")

    # Cleanup
    env.close()
    logger.info("Environment closed.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for RL training.

    CLI arguments override default values from training/rl/config.yaml.
    """
    parser = argparse.ArgumentParser(
        description="SO-ARM101 RL 학습 (skrl PPO + Isaac Lab)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Isaac Lab 환경 태스크 이름 "
        "(기본값: training/rl/config.yaml에서 로드)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="병렬 환경 수 (기본값: config.yaml에서 로드)",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="헤드리스 모드 (기본: True, --no-headless로 GUI 활성화)",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="최대 학습 반복 횟수 (기본값: config.yaml에서 로드)",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=None,
        help="체크포인트 저장 주기 (기본값: config.yaml에서 로드)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="체크포인트 저장 경로 (기본값: training/rl/checkpoints)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="난수 시드 (기본값: config.yaml에서 로드)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_CONFIG_YAML),
        help="RL 학습 설정 YAML 파일 경로",
    )
    parser.add_argument(
        "--control_yaml",
        type=str,
        default=str(_CONTROL_YAML),
        help="제어 파라미터 YAML 파일 경로",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for RL training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    # Allow overriding YAML paths via CLI
    global _CONFIG_YAML, _CONTROL_YAML
    _CONFIG_YAML = Path(args.config)
    _CONTROL_YAML = Path(args.control_yaml)

    try:
        run_training(args)
    except RuntimeError as exc:
        logger.error("RL training failed: %s", exc)
        sys.exit(1)
    except FileNotFoundError as exc:
        logger.error("Configuration file not found: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
