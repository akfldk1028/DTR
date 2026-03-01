#!/usr/bin/env python3
"""eval_in_sim.py — Isaac Sim VLA 모델 시뮬레이션 평가 스크립트

목적:
    Isaac Sim에서 VLA(Vision-Language-Action) 모델의 closed-loop 평가를 수행한다.
    - instruction → model.predict() → apply_action() → sim step → metric logging
    - 에피소드 단위 성공률, 궤적 오차, 에피소드 길이 측정
    - --dry-run 모드: Isaac Sim 없이 mock 관측으로 파이프라인 검증

사용법:
    # Isaac Sim 환경에서 실제 평가
    python training/vla/eval_in_sim.py \\
        --model dummy \\
        --instruction "pick up the orange from the table" \\
        --num-episodes 10

    # 파이프라인 검증 (Isaac Sim 없이)
    python training/vla/eval_in_sim.py \\
        --model dummy \\
        --dry-run

    # 파라미터 파일에서 기본값 로드
    python training/vla/eval_in_sim.py

필요 환경:
    - Isaac Sim 5.1.0 (isaaclab 패키지) — 실제 평가 시
    - NumPy
    - conda env: soarm

Phase: 7
상태: 구현 완료 — VLA 시뮬레이션 평가 루프
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Add project root to sys.path so the script can be run directly
# (e.g. python training/vla/eval_in_sim.py --dry-run)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Isaac Sim imports are guarded so the script can be
# syntax-checked and --help can run without these heavy dependencies.
try:
    import numpy as np
except ImportError:
    np = None

try:
    from isaaclab.envs import ManagerBasedEnv

    ISAAC_SIM_AVAILABLE = True
except ImportError:
    ISAAC_SIM_AVAILABLE = False

from training.vla.inference import DummyVLA, SmolVLAWrapper, VLAInference

logger = logging.getLogger(__name__)

# --- Parameter file paths ---
_PARAMS_DIR = Path(__file__).resolve().parent.parent.parent / "params"
_VLA_EVAL_YAML = _PARAMS_DIR / "vla_eval.yaml"


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


def load_eval_params() -> dict:
    """Load params/vla_eval.yaml and return raw dict.

    Returns:
        dict: 평가 파라미터 딕셔너리.

    Raises:
        FileNotFoundError: 파라미터 파일이 존재하지 않는 경우.
    """
    return _load_yaml(_VLA_EVAL_YAML)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

_MODEL_REGISTRY = {
    "dummy": "DummyVLA",
    "smolvla": "SmolVLAWrapper",
}


def model_factory(model_type: str, **kwargs) -> VLAInference:
    """Instantiate a VLA model by name.

    Args:
        model_type: 모델 타입 문자열 ("dummy", "smolvla").
        **kwargs: 모델 생성자에 전달할 추가 인자
            (예: checkpoint_path for SmolVLAWrapper).

    Returns:
        VLAInference 인스턴스.

    Raises:
        ValueError: 지원하지 않는 모델 타입인 경우.
    """
    model_type = model_type.lower().strip()

    if model_type == "dummy":
        return DummyVLA()
    elif model_type == "smolvla":
        checkpoint_path = kwargs.get("checkpoint_path", "")
        if not checkpoint_path:
            raise ValueError(
                "SmolVLAWrapper requires 'checkpoint_path'. "
                "Provide --checkpoint or set model.checkpoint_path in "
                "params/vla_eval.yaml."
            )
        return SmolVLAWrapper(checkpoint_path=checkpoint_path)
    else:
        supported = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Supported types: {supported}"
        )


# ---------------------------------------------------------------------------
# Episode metrics
# ---------------------------------------------------------------------------


class EpisodeMetrics:
    """Single episode evaluation metrics.

    Attributes:
        episode_idx: 에피소드 인덱스.
        episode_length: 에피소드 길이 (스텝 수).
        success: 성공 여부.
        trajectory_error: 궤적 오차 (조인트 공간 L2 노름 평균).
        terminated: 환경 종료 여부.
        truncated: 환경 절단(max steps) 여부.
    """

    def __init__(self, episode_idx: int) -> None:
        self.episode_idx = episode_idx
        self.episode_length = 0
        self.success = False
        self.trajectory_error = 0.0
        self.terminated = False
        self.truncated = False


# ---------------------------------------------------------------------------
# EvalRunner
# ---------------------------------------------------------------------------


class EvalRunner:
    """Closed-loop VLA evaluation runner for Isaac Sim.

    Isaac Sim 환경에서 VLA 모델의 closed-loop 평가를 수행한다.
    관측값(이미지 + 상태) → model.predict() → apply_action() → sim step
    루프를 반복하고, 에피소드 단위로 성공률, 궤적 오차, 에피소드 길이를
    기록한다.

    --dry-run 모드에서는 Isaac Sim 없이 mock 관측을 사용하여
    파이프라인을 검증할 수 있다.

    Args:
        model: VLAInference 인스턴스.
        eval_params: params/vla_eval.yaml에서 로드한 파라미터 딕셔너리.
        dry_run: True이면 Isaac Sim 없이 mock 관측으로 파이프라인 검증.
    """

    def __init__(
        self,
        model: VLAInference,
        eval_params: dict,
        dry_run: bool = False,
    ) -> None:
        if np is None:
            raise RuntimeError("NumPy is required but not installed.")

        self.model = model
        self.params = eval_params
        self.dry_run = dry_run

        # Extract evaluation parameters
        eval_cfg = eval_params.get("evaluation", {})
        self.max_steps = _get_value(
            eval_cfg.get("max_steps_per_episode", {"value": 500})
        )
        self.instruction = _get_value(
            eval_cfg.get(
                "instruction",
                {"value": "pick up the orange from the table"},
            )
        )

        # Extract observation parameters
        obs_cfg = eval_params.get("observation", {})
        self.image_height = _get_value(
            obs_cfg.get("image_height", {"value": 480})
        )
        self.image_width = _get_value(
            obs_cfg.get("image_width", {"value": 640})
        )
        self.image_channels = _get_value(
            obs_cfg.get("image_channels", {"value": 3})
        )
        self.state_dim = _get_value(
            obs_cfg.get("state_dim", {"value": 6})
        )
        self.action_dim = _get_value(
            obs_cfg.get("action_dim", {"value": 6})
        )

        # Extract success criteria
        criteria_cfg = eval_params.get("success_criteria", {})
        self.trajectory_error_tolerance = _get_value(
            criteria_cfg.get("trajectory_error_tolerance", {"value": 0.05})
        )
        self.min_episode_length = _get_value(
            criteria_cfg.get("min_episode_length", {"value": 10})
        )

        # Extract Isaac Sim parameters
        sim_cfg = eval_params.get("isaac_sim", {})
        self.task_name = _get_value(
            sim_cfg.get("task", {"value": "LeIsaac-SO101-PickOrange-v0"})
        )
        self.num_envs = _get_value(
            sim_cfg.get("num_envs", {"value": 1})
        )
        self.render = _get_value(
            sim_cfg.get("render", {"value": False})
        )

        # Environment (initialized lazily)
        self._env = None

        logger.info(
            "EvalRunner initialized: model=%s, dry_run=%s, max_steps=%d",
            type(model).__name__,
            dry_run,
            self.max_steps,
        )

    def _initialize_env(self):
        """Initialize the Isaac Sim evaluation environment.

        Raises:
            RuntimeError: Isaac Sim이 사용 불가능한 경우.
        """
        if self.dry_run:
            logger.info("Dry-run mode: skipping Isaac Sim initialization")
            return

        if not ISAAC_SIM_AVAILABLE:
            raise RuntimeError(
                "Isaac Sim is not available. "
                "Run this script inside the Isaac Sim Python environment, "
                "or use --dry-run for pipeline verification."
            )

        logger.info(
            "Initializing Isaac Sim environment: task=%s, num_envs=%d",
            self.task_name,
            self.num_envs,
        )
        self._env = ManagerBasedEnv(
            task=self.task_name, num_envs=self.num_envs
        )

    def _get_observation(self, obs_dict):
        """Extract image and state from environment observation.

        Args:
            obs_dict: 환경 관측 딕셔너리 (또는 dry-run 시 None).

        Returns:
            Tuple of (image, state) numpy arrays.
        """
        if self.dry_run:
            # Mock observation for pipeline verification
            image = np.zeros(
                (self.image_height, self.image_width, self.image_channels),
                dtype=np.uint8,
            )
            state = np.zeros(self.state_dim, dtype=np.float32)
            return image, state

        # Extract from Isaac Sim observation
        image = self._env.unwrapped.scene["camera"].data.output["rgb"]
        state = obs_dict["joint_pos"]

        # Ensure correct numpy types
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)

        return image, state

    def _compute_step_error(
        self, action: "np.ndarray", state: "np.ndarray"
    ) -> float:
        """Compute trajectory error for a single step.

        Args:
            action: 예측된 액션 (target joint positions).
            state: 현재 관절 상태.

        Returns:
            L2 norm of (action - state) in joint space.
        """
        return float(np.linalg.norm(action - state))

    def run_episode(
        self, instruction: str, episode_idx: int = 0
    ) -> EpisodeMetrics:
        """Run a single evaluation episode.

        Closed-loop: observation → predict → apply_action → step → metrics.

        Args:
            instruction: VLA 모델에 전달할 자연어 지시문.
            episode_idx: 에피소드 인덱스 (로깅용).

        Returns:
            EpisodeMetrics with episode results.
        """
        metrics = EpisodeMetrics(episode_idx=episode_idx)
        step_errors = []

        logger.info(
            "Starting episode %d: instruction='%s', max_steps=%d",
            episode_idx,
            instruction,
            self.max_steps,
        )

        # Reset environment
        obs_dict = None
        if not self.dry_run:
            obs_dict = self._env.reset()

        for step_idx in range(self.max_steps):
            # 1. Get observation
            image, state = self._get_observation(obs_dict)

            # 2. Predict action
            action = self.model.predict(instruction, image, state)

            # 3. Compute step error
            step_error = self._compute_step_error(action, state)
            step_errors.append(step_error)

            # 4. Apply action and step environment
            if self.dry_run:
                # In dry-run, simulate for a fixed number of steps
                terminated = step_idx >= (self.max_steps - 1)
                truncated = False
            else:
                obs_dict, _, terminated, truncated, info = self._env.step(
                    action
                )
                terminated = bool(
                    terminated.any() if hasattr(terminated, "any")
                    else terminated
                )
                truncated = bool(
                    truncated.any() if hasattr(truncated, "any")
                    else truncated
                )

            metrics.episode_length = step_idx + 1

            if terminated or truncated:
                metrics.terminated = terminated
                metrics.truncated = truncated
                logger.info(
                    "Episode %d ended at step %d "
                    "(terminated=%s, truncated=%s)",
                    episode_idx,
                    step_idx,
                    terminated,
                    truncated,
                )
                break

        # Compute aggregate trajectory error
        if step_errors:
            metrics.trajectory_error = float(np.mean(step_errors))

        # Determine success: episode long enough and trajectory error
        # within tolerance
        metrics.success = (
            metrics.episode_length >= self.min_episode_length
            and metrics.trajectory_error <= self.trajectory_error_tolerance
        )

        logger.info(
            "Episode %d complete: length=%d, trajectory_error=%.4f, "
            "success=%s",
            episode_idx,
            metrics.episode_length,
            metrics.trajectory_error,
            metrics.success,
        )

        return metrics

    def run_evaluation(
        self,
        instruction: str,
        num_episodes: int,
    ) -> dict:
        """Run multi-episode evaluation and aggregate metrics.

        Args:
            instruction: VLA 모델에 전달할 자연어 지시문.
            num_episodes: 평가 에피소드 수.

        Returns:
            dict with aggregated metrics:
                - success_rate (float): 성공률 (0.0 ~ 1.0).
                - mean_trajectory_error (float): 평균 궤적 오차.
                - mean_episode_length (float): 평균 에피소드 길이.
                - episodes (list): 각 에피소드의 EpisodeMetrics.
        """
        if np is None:
            raise RuntimeError("NumPy is required but not installed.")

        # Initialize environment if needed
        self._initialize_env()

        logger.info(
            "=== VLA Evaluation Start ===\n"
            "  Model:        %s\n"
            "  Instruction:  %s\n"
            "  Episodes:     %d\n"
            "  Max steps:    %d\n"
            "  Dry-run:      %s",
            type(self.model).__name__,
            instruction,
            num_episodes,
            self.max_steps,
            self.dry_run,
        )

        episode_results = []

        for ep_idx in range(num_episodes):
            metrics = self.run_episode(
                instruction=instruction,
                episode_idx=ep_idx,
            )
            episode_results.append(metrics)

        # Aggregate metrics
        successes = sum(1 for m in episode_results if m.success)
        success_rate = successes / num_episodes if num_episodes > 0 else 0.0

        mean_traj_error = float(
            np.mean([m.trajectory_error for m in episode_results])
        )
        mean_ep_length = float(
            np.mean([m.episode_length for m in episode_results])
        )

        results = {
            "success_rate": success_rate,
            "mean_trajectory_error": mean_traj_error,
            "mean_episode_length": mean_ep_length,
            "num_episodes": num_episodes,
            "episodes": episode_results,
        }

        # Log summary
        logger.info("=== Evaluation Summary ===")
        logger.info("  Success rate:           %.2f%%", success_rate * 100)
        logger.info("  Mean trajectory error:  %.4f", mean_traj_error)
        logger.info("  Mean episode length:    %.1f", mean_ep_length)
        logger.info(
            "  Episodes: %d total, %d success, %d fail",
            num_episodes,
            successes,
            num_episodes - successes,
        )

        # Cleanup environment
        if self._env is not None:
            self._env.close()
            logger.info("Environment closed.")

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for VLA evaluation.

    CLI arguments override default values from params/vla_eval.yaml.
    """
    parser = argparse.ArgumentParser(
        description="SO-ARM101 VLA 모델 시뮬레이션 평가 (Isaac Sim closed-loop)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "VLA 모델 타입 (dummy, smolvla). "
            "기본값: params/vla_eval.yaml에서 로드"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="모델 체크포인트 경로 (smolvla 등 학습된 모델 사용 시)",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help=(
            "VLA 모델에 전달할 태스크 지시문 "
            "(기본값: params/vla_eval.yaml에서 로드)"
        ),
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="평가 에피소드 수 (기본값: params/vla_eval.yaml에서 로드)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Isaac Sim 없이 mock 관측으로 파이프라인 검증",
    )
    parser.add_argument(
        "--vla-eval-yaml",
        type=str,
        default=str(_VLA_EVAL_YAML),
        help="VLA 평가 파라미터 파일 경로",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Headless 모드 실행 (기본값: True, --no-headless로 GUI 활성화)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for VLA evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    # Allow overriding YAML path via CLI
    global _VLA_EVAL_YAML
    _VLA_EVAL_YAML = Path(args.vla_eval_yaml)

    try:
        # Load evaluation parameters
        eval_params = load_eval_params()

        # Resolve effective values (CLI overrides > YAML defaults)
        model_cfg = eval_params.get("model", {})
        eval_cfg = eval_params.get("evaluation", {})

        model_type = args.model or _get_value(
            model_cfg.get("type", {"value": "dummy"})
        )
        checkpoint_path = args.checkpoint or _get_value(
            model_cfg.get("checkpoint_path", {"value": ""})
        )
        instruction = args.instruction or _get_value(
            eval_cfg.get(
                "instruction",
                {"value": "pick up the orange from the table"},
            )
        )
        num_episodes = args.num_episodes or _get_value(
            eval_cfg.get("num_episodes", {"value": 10})
        )
        dry_run = args.dry_run or _get_value(
            eval_cfg.get("dry_run", {"value": False})
        )

        # Instantiate model
        model = model_factory(
            model_type, checkpoint_path=checkpoint_path
        )

        # Create runner and execute evaluation
        runner = EvalRunner(
            model=model,
            eval_params=eval_params,
            dry_run=dry_run,
        )

        results = runner.run_evaluation(
            instruction=instruction,
            num_episodes=num_episodes,
        )

        # Exit with success/failure based on results
        logger.info(
            "Evaluation complete. Success rate: %.2f%%",
            results["success_rate"] * 100,
        )

    except RuntimeError as exc:
        logger.error("Evaluation failed: %s", exc)
        sys.exit(1)
    except FileNotFoundError as exc:
        logger.error("Parameter file not found: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
