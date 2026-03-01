#!/usr/bin/env python3
"""train_act.py — LeRobot ACT (Action Chunking with Transformers) 학습 스크립트

목적:
    Phase 5에서 수집한 데이터로 ACT 정책을 학습한다.
    - training/il/config.yaml에서 ACT 하이퍼파라미터 로드
    - params/data_pipeline.yaml에서 데이터셋 설정 로드
    - LeRobot ACT 정책 구성 및 학습 실행
    - 체크포인트를 training/il/checkpoints/에 저장

사용법:
    # 기본 설정으로 학습 실행
    python training/il/train_act.py

    # CLI 인자로 오버라이드
    python training/il/train_act.py \\
        --repo_id local/so101_teleop \\
        --batch_size 8 \\
        --steps 100000 \\
        --output_dir training/il/checkpoints

    # 짧은 테스트 학습
    python training/il/train_act.py --steps 1000 --save_freq 500

필요 환경:
    - LeRobot 0.4.4 (lerobot 패키지)
    - PyTorch 2.x (torch 패키지)
    - conda env: soarm
    - GPU: NVIDIA RTX 4090 Laptop (권장)

Phase: 6
상태: 구현 완료 — ACT IL 학습 파이프라인
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# LeRobot / torch imports are guarded so the script can be
# syntax-checked and --help can run without these heavy dependencies.
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    LeRobotDataset = None

try:
    from lerobot.common.policies.act.configuration_act import ACTConfig
    from lerobot.common.policies.act.modeling_act import ACTPolicy

    LEROBOT_ACT_AVAILABLE = True
except ImportError:
    ACTConfig = None
    ACTPolicy = None
    LEROBOT_ACT_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- Parameter file paths ---
_SCRIPT_DIR = Path(__file__).resolve().parent
_CONFIG_YAML = _SCRIPT_DIR / "config.yaml"
_PARAMS_DIR = _SCRIPT_DIR.parent.parent / "params"
_DATA_PIPELINE_YAML = _PARAMS_DIR / "data_pipeline.yaml"


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


def load_training_config() -> dict:
    """Load training/il/config.yaml and return raw dict."""
    raw = _load_yaml(_CONFIG_YAML)
    return raw


def load_pipeline_params() -> dict:
    """Load params/data_pipeline.yaml and return raw dict."""
    raw = _load_yaml(_DATA_PIPELINE_YAML)
    return raw


def build_act_config(training_cfg: dict, pipeline_params: dict) -> "ACTConfig":
    """Build ACT policy configuration from YAML hyperparameters.

    Args:
        training_cfg: Loaded training/il/config.yaml dict.
        pipeline_params: Loaded params/data_pipeline.yaml dict.

    Returns:
        ACTConfig instance configured with the specified hyperparameters.

    Raises:
        RuntimeError: If LeRobot ACT is not installed.
    """
    if not LEROBOT_ACT_AVAILABLE:
        raise RuntimeError(
            "LeRobot ACT is not installed. "
            "Install with: pip install lerobot"
        )

    policy_cfg = training_cfg["policy"]

    # Extract policy hyperparameters
    chunk_size = _get_value(policy_cfg["chunk_size"])
    n_obs_steps = _get_value(policy_cfg["n_obs_steps"])
    dim_model = _get_value(policy_cfg["dim_model"])
    n_heads = _get_value(policy_cfg["n_heads"])
    n_encoder_layers = _get_value(policy_cfg["n_encoder_layers"])
    n_decoder_layers = _get_value(policy_cfg["n_decoder_layers"])

    # Extract environment dimensions
    env_cfg = training_cfg["env"]
    state_dim = _get_value(env_cfg["state_dim"])
    action_dim = _get_value(env_cfg["action_dim"])

    logger.info("Building ACT config:")
    logger.info("  chunk_size:       %d", chunk_size)
    logger.info("  n_obs_steps:      %d", n_obs_steps)
    logger.info("  dim_model:        %d", dim_model)
    logger.info("  n_heads:          %d", n_heads)
    logger.info("  n_encoder_layers: %d", n_encoder_layers)
    logger.info("  n_decoder_layers: %d", n_decoder_layers)
    logger.info("  state_dim:        %d", state_dim)
    logger.info("  action_dim:       %d", action_dim)

    act_config = ACTConfig(
        chunk_size=chunk_size,
        n_obs_steps=n_obs_steps,
        dim_model=dim_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
    )

    return act_config


def load_dataset(repo_id: str) -> "LeRobotDataset":
    """Load a LeRobot dataset for training.

    Args:
        repo_id: Dataset repository identifier (e.g. 'local/so101_teleop').

    Returns:
        LeRobotDataset instance.

    Raises:
        RuntimeError: If LeRobot is not installed.
    """
    if LeRobotDataset is None:
        raise RuntimeError(
            "LeRobot is not installed. "
            "Install with: pip install lerobot"
        )

    logger.info("Loading dataset: repo_id=%s", repo_id)
    dataset = LeRobotDataset(repo_id=repo_id)
    logger.info(
        "Dataset loaded: %d episodes, %d total frames",
        dataset.meta.total_episodes,
        dataset.meta.total_frames,
    )
    return dataset


def create_optimizer(policy, training_cfg: dict):
    """Create optimizer from training configuration.

    Args:
        policy: ACT policy model.
        training_cfg: Loaded training/il/config.yaml dict.

    Returns:
        torch optimizer instance.

    Raises:
        RuntimeError: If PyTorch is not available.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not available. "
            "Install with: pip install torch"
        )

    opt_cfg = training_cfg["training"]["optimizer"]
    lr = _get_value(opt_cfg["lr"])
    weight_decay = _get_value(opt_cfg["weight_decay"])
    opt_type = _get_value(opt_cfg["type"])

    logger.info(
        "Creating optimizer: type=%s, lr=%e, weight_decay=%e",
        opt_type,
        lr,
        weight_decay,
    )

    if opt_type == "AdamW":
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif opt_type == "Adam":
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")

    return optimizer


def save_checkpoint(
    policy, optimizer, step: int, output_dir: Path
) -> Path:
    """Save a training checkpoint.

    Args:
        policy: ACT policy model.
        optimizer: Optimizer state.
        step: Current training step.
        output_dir: Directory to save checkpoints.

    Returns:
        Path to the saved checkpoint file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"checkpoint_{step:07d}.pt"

    checkpoint = {
        "step": step,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info("Checkpoint saved: %s (step=%d)", checkpoint_path, step)
    return checkpoint_path


def run_training(args: argparse.Namespace) -> None:
    """Run the ACT training pipeline.

    Args:
        args: Parsed CLI arguments.
    """
    # Load configuration files
    training_cfg = load_training_config()
    pipeline_params = load_pipeline_params()

    # Resolve effective values (CLI overrides > YAML defaults)
    repo_id = args.repo_id or _get_value(
        training_cfg["dataset"]["repo_id"]
    )
    batch_size = args.batch_size or _get_value(
        training_cfg["training"]["batch_size"]
    )
    total_steps = args.steps or _get_value(
        training_cfg["training"]["steps"]
    )
    save_freq = args.save_freq or _get_value(
        training_cfg["training"]["save_freq"]
    )
    output_dir = Path(
        args.output_dir or _get_value(
            training_cfg["training"]["output_dir"]
        )
    )

    logger.info("=== SO-ARM101 ACT IL Training ===")
    logger.info("Dataset:       %s", repo_id)
    logger.info("Batch size:    %d", batch_size)
    logger.info("Total steps:   %d", total_steps)
    logger.info("Save freq:     %d", save_freq)
    logger.info("Output dir:    %s", output_dir)

    # Check dependencies
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

    # Build ACT policy config
    act_config = build_act_config(training_cfg, pipeline_params)

    # Load dataset
    dataset = load_dataset(repo_id)

    # Create policy
    logger.info("Creating ACT policy...")
    policy = ACTPolicy(config=act_config, dataset_stats=dataset.stats)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    logger.info("Device: %s", device)

    # Create optimizer
    optimizer = create_optimizer(policy, training_cfg)

    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Training loop
    logger.info("Starting training...")
    policy.train()
    step = 0
    epoch = 0

    while step < total_steps:
        epoch += 1
        logger.info("Epoch %d (step %d / %d)", epoch, step, total_steps)

        for batch in dataloader:
            if step >= total_steps:
                break

            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            loss_dict = policy.forward(batch)
            loss = loss_dict["loss"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            # Logging
            if step % 100 == 0:
                logger.info(
                    "Step %d/%d — loss: %.6f",
                    step,
                    total_steps,
                    loss.item(),
                )

            # Save checkpoint
            if step % save_freq == 0:
                save_checkpoint(policy, optimizer, step, output_dir)

    # Save final checkpoint
    save_checkpoint(policy, optimizer, step, output_dir)

    # Training summary
    logger.info("=== Training Summary ===")
    logger.info("Total steps completed: %d", step)
    logger.info("Total epochs:          %d", epoch)
    logger.info("Checkpoints saved to:  %s", output_dir)
    logger.info("Training config:       %s", _CONFIG_YAML)
    logger.info("Pipeline params:       %s", _DATA_PIPELINE_YAML)
    logger.info("Training complete.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for ACT training.

    CLI arguments override default values from training/il/config.yaml.
    """
    parser = argparse.ArgumentParser(
        description="SO-ARM101 ACT IL 학습 (LeRobot ACT Policy)"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="LeRobot 데이터셋 리포지토리 ID "
        "(기본값: training/il/config.yaml에서 로드)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="미니배치 크기 (기본값: config.yaml에서 로드)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="총 학습 스텝 수 (기본값: config.yaml에서 로드)",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=None,
        help="체크포인트 저장 주기 (기본값: config.yaml에서 로드)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="체크포인트 저장 경로 (기본값: training/il/checkpoints)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="학습률 (기본값: config.yaml에서 로드)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_CONFIG_YAML),
        help="학습 설정 YAML 파일 경로",
    )
    parser.add_argument(
        "--data_pipeline_yaml",
        type=str,
        default=str(_DATA_PIPELINE_YAML),
        help="데이터 파이프라인 파라미터 파일 경로",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for ACT training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    # Allow overriding YAML paths via CLI
    global _CONFIG_YAML, _DATA_PIPELINE_YAML
    _CONFIG_YAML = Path(args.config)
    _DATA_PIPELINE_YAML = Path(args.data_pipeline_yaml)

    try:
        run_training(args)
    except RuntimeError as exc:
        logger.error("Training failed: %s", exc)
        sys.exit(1)
    except FileNotFoundError as exc:
        logger.error("Configuration file not found: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
