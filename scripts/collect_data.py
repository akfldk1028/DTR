#!/usr/bin/env python3
"""collect_data.py — Isaac Sim 텔레오퍼레이션 데이터 수집 스크립트

목적:
    Isaac Sim에서 텔레오퍼레이션으로 데이터를 수집하고,
    LeRobot v2 포맷으로 저장한다.
    - 카메라 이미지 + 조인트 위치를 프레임 단위로 기록
    - LeRobotDataset.create() → add_frame() → save_episode()
    - 에피소드 완료 후 요약 로그 출력

사용법:
    python scripts/collect_data.py \\
        --task LeIsaac-SO101-PickOrange-v0 \\
        --repo_id local/so101_teleop \\
        --num_episodes 1 \\
        --fps 30 \\
        --num_envs 1

    # 파라미터 파일에서 기본값 로드 (CLI 인자로 오버라이드 가능)
    python scripts/collect_data.py

필요 환경:
    - Isaac Sim 5.1.0 (isaaclab 패키지)
    - LeRobot 0.4.4 (lerobot 패키지)
    - LeIsaac 0.3.0 (선택)
    - conda env: soarm

Phase: 5
상태: 구현 완료 — 텔레오퍼레이션 데이터 수집 파이프라인
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Isaac Sim / LeRobot imports are guarded so the script can be
# syntax-checked and --help can run without these heavy dependencies.
try:
    import numpy as np
except ImportError:
    np = None

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    LeRobotDataset = None

try:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.mdp import teleop_se3_agent  # noqa: F401

    ISAAC_SIM_AVAILABLE = True
except ImportError:
    ISAAC_SIM_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- Parameter file paths ---
_PARAMS_DIR = Path(__file__).resolve().parent.parent / "params"
_DATA_PIPELINE_YAML = _PARAMS_DIR / "data_pipeline.yaml"
_CONTROL_YAML = _PARAMS_DIR / "control.yaml"


def _load_yaml(path: Path) -> dict:
    """Load a YAML parameter file and return its contents."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _get_value(param):
    """Extract the 'value' field from a parameter dict."""
    if isinstance(param, dict) and "value" in param:
        return param["value"]
    return param


def load_pipeline_params() -> dict:
    """Load params/data_pipeline.yaml and return flattened values."""
    raw = _load_yaml(_DATA_PIPELINE_YAML)
    return raw


def load_control_params() -> dict:
    """Load params/control.yaml and return raw dict."""
    raw = _load_yaml(_CONTROL_YAML)
    return raw


def build_features(pipeline_params: dict) -> dict:
    """Build the LeRobot v2 feature schema from pipeline parameters.

    Returns:
        dict: features dict suitable for LeRobotDataset.create().
    """
    feat_cfg = pipeline_params["features"]
    camera_cfg = pipeline_params["camera"]
    joint_names = _get_value(pipeline_params["joint_names"])

    img_shape = tuple(_get_value(feat_cfg["observation_images"]["shape"]))
    state_shape = tuple(_get_value(feat_cfg["observation_state"]["shape"]))
    action_shape = tuple(_get_value(feat_cfg["action"]["shape"]))

    features = {
        "observation.images.camera": {
            "dtype": _get_value(feat_cfg["observation_images"]["dtype"]),
            "shape": img_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": _get_value(feat_cfg["observation_state"]["dtype"]),
            "shape": state_shape,
            "names": list(joint_names),
        },
        "action": {
            "dtype": _get_value(feat_cfg["action"]["dtype"]),
            "shape": action_shape,
            "names": list(joint_names),
        },
    }
    return features


def create_dataset(repo_id: str, fps: int, features: dict) -> "LeRobotDataset":
    """Create a new LeRobot v2 dataset.

    Args:
        repo_id: Dataset repository identifier (e.g. 'local/so101_teleop').
        fps: Frames per second for the dataset.
        features: Feature schema dict.

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

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
    )
    logger.info("LeRobot dataset created: repo_id=%s, fps=%d", repo_id, fps)
    return dataset


def initialize_env(task_name: str, num_envs: int):
    """Initialize the Isaac Sim teleop environment.

    Args:
        task_name: Isaac Sim task identifier.
        num_envs: Number of parallel environments.

    Returns:
        Isaac Sim environment instance.

    Raises:
        RuntimeError: If Isaac Sim is not available.
    """
    if not ISAAC_SIM_AVAILABLE:
        raise RuntimeError(
            "Isaac Sim is not available. "
            "Run this script inside the Isaac Sim Python environment."
        )

    logger.info(
        "Initializing Isaac Sim environment: task=%s, num_envs=%d",
        task_name,
        num_envs,
    )
    env = ManagerBasedEnv(task=task_name, num_envs=num_envs)
    return env


def collect_episode(
    env,
    dataset: "LeRobotDataset",
    episode_idx: int,
    task_description: str,
    max_steps: int,
    camera_name: str,
) -> int:
    """Collect a single episode of teleop data.

    Runs the teleop loop: captures camera images and joint positions,
    adds each frame to the dataset, and saves the episode.

    Args:
        env: Isaac Sim environment instance.
        dataset: LeRobotDataset to record into.
        episode_idx: Current episode index (for logging).
        task_description: Task string saved with the episode.
        max_steps: Maximum number of steps per episode.
        camera_name: Name of the camera sensor in the environment.

    Returns:
        Number of frames recorded in this episode.
    """
    if np is None:
        raise RuntimeError("NumPy is required but not installed.")

    logger.info("Starting episode %d (max_steps=%d)", episode_idx, max_steps)

    obs = env.reset()
    frame_count = 0

    for step_idx in range(max_steps):
        # Get observation data from the environment
        image = env.unwrapped.scene[camera_name].data.output["rgb"]
        joint_positions = obs["joint_pos"]

        # Compute action via teleop agent
        action = env.unwrapped.action_manager.action
        obs, _, terminated, truncated, info = env.step(action)

        # Record frame to dataset
        dataset.add_frame(
            {
                "observation.images.camera": image,
                "observation.state": joint_positions,
                "action": action,
            }
        )
        frame_count += 1

        # End episode on termination or truncation
        if terminated.any() or truncated.any():
            logger.info(
                "Episode %d terminated at step %d", episode_idx, step_idx
            )
            break

    # Save the completed episode
    dataset.save_episode(task=task_description)
    logger.info(
        "Episode %d saved: %d frames, task='%s'",
        episode_idx,
        frame_count,
        task_description,
    )
    return frame_count


def run_collection(args: argparse.Namespace) -> None:
    """Run the full data collection pipeline.

    Args:
        args: Parsed CLI arguments.
    """
    # Load pipeline parameters
    pipeline_params = load_pipeline_params()
    control_params = load_control_params()

    # Resolve effective values (CLI overrides > YAML defaults)
    task_name = args.task or _get_value(
        pipeline_params["isaac_sim"]["task"]
    )
    repo_id = args.repo_id or _get_value(
        pipeline_params["dataset"]["repo_id"]
    )
    fps = args.fps or _get_value(pipeline_params["fps"])
    num_episodes = args.num_episodes or _get_value(
        pipeline_params["episode"]["num_episodes"]
    )
    num_envs = args.num_envs or _get_value(
        pipeline_params["isaac_sim"]["num_envs"]
    )
    task_description = args.task_description or _get_value(
        pipeline_params["dataset"]["task_name"]
    )
    max_steps = _get_value(pipeline_params["episode"]["max_steps"])
    camera_name = _get_value(pipeline_params["camera"]["camera_name"])

    logger.info("=== SO-ARM101 Data Collection ===")
    logger.info("Task:          %s", task_name)
    logger.info("Repo ID:       %s", repo_id)
    logger.info("FPS:           %d", fps)
    logger.info("Num episodes:  %d", num_episodes)
    logger.info("Num envs:      %d", num_envs)
    logger.info("Max steps:     %d", max_steps)
    logger.info("Task desc:     %s", task_description)

    # Build feature schema from parameters
    features = build_features(pipeline_params)

    # Create LeRobot dataset
    dataset = create_dataset(repo_id=repo_id, fps=fps, features=features)

    # Initialize Isaac Sim environment
    env = initialize_env(task_name=task_name, num_envs=num_envs)

    # Collect episodes
    total_frames = 0
    for ep_idx in range(num_episodes):
        frames = collect_episode(
            env=env,
            dataset=dataset,
            episode_idx=ep_idx,
            task_description=task_description,
            max_steps=max_steps,
            camera_name=camera_name,
        )
        total_frames += frames

    # Log collection summary
    logger.info("=== Collection Summary ===")
    logger.info("Episodes collected: %d", num_episodes)
    logger.info("Total frames:       %d", total_frames)
    logger.info("Output dataset:     %s", repo_id)
    logger.info("Data pipeline parameters: %s", _DATA_PIPELINE_YAML)
    logger.info("Control parameters:       %s", _CONTROL_YAML)

    # Cleanup
    env.close()
    logger.info("Environment closed. Data collection complete.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for data collection.

    CLI arguments override default values from params/data_pipeline.yaml.
    """
    parser = argparse.ArgumentParser(
        description="SO-ARM101 텔레오퍼레이션 데이터 수집 (Isaac Sim → LeRobot v2)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Isaac Sim 환경 태스크 이름 (기본값: params/data_pipeline.yaml에서 로드)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="LeRobot 데이터셋 리포지토리 ID (예: local/so101_teleop)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="수집할 에피소드 수 (기본값: params/data_pipeline.yaml에서 로드)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="데이터 수집 프레임 레이트 (기본값: params/data_pipeline.yaml에서 로드)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="병렬 환경 수 (기본값: params/data_pipeline.yaml에서 로드)",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default=None,
        help="에피소드 태스크 설명 (기본값: params/data_pipeline.yaml에서 로드)",
    )
    parser.add_argument(
        "--data_pipeline_yaml",
        type=str,
        default=str(_DATA_PIPELINE_YAML),
        help="데이터 파이프라인 파라미터 파일 경로",
    )
    parser.add_argument(
        "--control_yaml",
        type=str,
        default=str(_CONTROL_YAML),
        help="제어 파라미터 파일 경로",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Headless 모드 실행 (기본값: True, --no-headless로 GUI 활성화)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for data collection."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    # Allow overriding YAML paths via CLI
    global _DATA_PIPELINE_YAML, _CONTROL_YAML
    _DATA_PIPELINE_YAML = Path(args.data_pipeline_yaml)
    _CONTROL_YAML = Path(args.control_yaml)

    try:
        run_collection(args)
    except RuntimeError as exc:
        logger.error("Data collection failed: %s", exc)
        sys.exit(1)
    except FileNotFoundError as exc:
        logger.error("Parameter file not found: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
