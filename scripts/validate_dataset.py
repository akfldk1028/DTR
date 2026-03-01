#!/usr/bin/env python3
"""validate_dataset.py — LeRobot v2 데이터셋 검증 스크립트

목적:
    수집된 LeRobot v2 데이터셋의 구조와 무결성을 검증한다.
    - 에피소드 수 / 프레임 수 확인
    - Feature shape 및 dtype 검증 (observation.state, action, observation.images.camera)
    - 메타데이터 파일 존재 여부 확인 (info.json, episodes.jsonl, tasks.jsonl)
    - Parquet / 비디오 파일 존재 여부 확인
    - (선택) 저장된 action 시퀀스의 시뮬레이션 리플레이

사용법:
    # 데이터셋 구조 검증만 실행
    python scripts/validate_dataset.py --repo_id local/so101_teleop

    # 경로 지정 검증
    python scripts/validate_dataset.py --dataset_path datasets/so101_teleop

    # 시뮬 리플레이 포함 검증 (Isaac Sim 환경 필요)
    python scripts/validate_dataset.py \\
        --repo_id local/so101_teleop \\
        --replay \\
        --task_name LeIsaac-SO101-PickOrange-v0

필요 환경:
    - LeRobot 0.4.4 (lerobot 패키지)
    - Isaac Sim 5.1.0 (--replay 모드 시에만 필요)
    - conda env: soarm

Phase: 5
상태: 구현 완료 — 데이터셋 검증 및 리플레이
"""

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import yaml

# LeRobot / Isaac Sim imports are guarded so the script can be
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

    ISAAC_SIM_AVAILABLE = True
except ImportError:
    ISAAC_SIM_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- Parameter file paths ---
_PARAMS_DIR = Path(__file__).resolve().parent.parent / "params"
_DATA_PIPELINE_YAML = _PARAMS_DIR / "data_pipeline.yaml"


def _load_yaml(path: Path) -> dict:
    """Load a YAML parameter file and return its contents."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _get_value(param):
    """Extract the 'value' field from a parameter dict."""
    if isinstance(param, dict) and "value" in param:
        return param["value"]
    return param


def load_expected_features(pipeline_params: dict) -> dict:
    """Build the expected feature schema from pipeline parameters.

    Returns:
        dict mapping feature name to (expected_shape, expected_dtype).
    """
    feat_cfg = pipeline_params["features"]
    return {
        "observation.state": {
            "shape": tuple(_get_value(feat_cfg["observation_state"]["shape"])),
            "dtype": _get_value(feat_cfg["observation_state"]["dtype"]),
        },
        "action": {
            "shape": tuple(_get_value(feat_cfg["action"]["shape"])),
            "dtype": _get_value(feat_cfg["action"]["dtype"]),
        },
        "observation.images.camera": {
            "shape": tuple(
                _get_value(feat_cfg["observation_images"]["shape"])
            ),
            "dtype": _get_value(feat_cfg["observation_images"]["dtype"]),
        },
    }


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


class ValidationReport:
    """Collects PASS/FAIL results for each validation check."""

    def __init__(self):
        self.results = []

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        """Record a single check result."""
        self.results.append(
            {"name": name, "passed": passed, "detail": detail}
        )

    @property
    def all_passed(self) -> bool:
        return all(r["passed"] for r in self.results)

    def log_report(self) -> None:
        """Log the full validation report."""
        logger.info("=== Validation Report ===")
        for r in self.results:
            status = "PASS" if r["passed"] else "FAIL"
            detail = f" — {r['detail']}" if r["detail"] else ""
            logger.info("  [%s] %s%s", status, r["name"], detail)

        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed
        logger.info("---")
        logger.info(
            "Total: %d | Passed: %d | Failed: %d", total, passed, failed
        )


def check_episode_count(dataset, report: ValidationReport) -> None:
    """Check that the dataset contains at least 1 episode."""
    num_episodes = dataset.meta.total_episodes
    passed = num_episodes >= 1
    report.add(
        "에피소드 수 >= 1",
        passed,
        f"총 {num_episodes}개 에피소드",
    )


def check_frame_counts(dataset, report: ValidationReport) -> None:
    """Check that each episode has at least 1 frame."""
    num_episodes = dataset.meta.total_episodes
    all_ok = True
    details = []

    for ep_idx in range(num_episodes):
        ep_length = dataset.meta.episodes[ep_idx]["length"]
        if ep_length < 1:
            all_ok = False
            details.append(f"episode {ep_idx}: {ep_length} frames (< 1)")
        else:
            details.append(f"episode {ep_idx}: {ep_length} frames")

    report.add(
        "에피소드별 프레임 수",
        all_ok,
        "; ".join(details),
    )


def check_feature_shapes(
    dataset, expected_features: dict, report: ValidationReport
) -> None:
    """Verify feature shapes match expected dimensions."""
    dataset_features = dataset.meta.features

    for feat_name, expected in expected_features.items():
        expected_shape = expected["shape"]

        if feat_name not in dataset_features:
            report.add(
                f"Feature shape: {feat_name}",
                False,
                f"feature가 데이터셋에 존재하지 않음",
            )
            continue

        actual_shape = tuple(dataset_features[feat_name]["shape"])
        passed = actual_shape == expected_shape
        report.add(
            f"Feature shape: {feat_name}",
            passed,
            f"expected={expected_shape}, actual={actual_shape}",
        )


def check_feature_dtypes(
    dataset, expected_features: dict, report: ValidationReport
) -> None:
    """Verify feature data types match expected types."""
    dataset_features = dataset.meta.features

    for feat_name, expected in expected_features.items():
        expected_dtype = expected["dtype"]

        if feat_name not in dataset_features:
            report.add(
                f"Feature dtype: {feat_name}",
                False,
                f"feature가 데이터셋에 존재하지 않음",
            )
            continue

        actual_dtype = dataset_features[feat_name]["dtype"]
        passed = actual_dtype == expected_dtype
        report.add(
            f"Feature dtype: {feat_name}",
            passed,
            f"expected={expected_dtype}, actual={actual_dtype}",
        )


def check_meta_files(dataset_path: Path, report: ValidationReport) -> None:
    """Check that required metadata files exist."""
    required_meta = ["info.json", "episodes.jsonl", "tasks.jsonl"]

    for filename in required_meta:
        filepath = dataset_path / "meta" / filename
        exists = filepath.exists()
        report.add(
            f"메타 파일: meta/{filename}",
            exists,
            str(filepath) if not exists else "존재함",
        )


def check_parquet_files(
    dataset_path: Path, report: ValidationReport
) -> None:
    """Check that parquet data files exist in data/chunk-*/ directories."""
    parquet_pattern = str(dataset_path / "data" / "chunk-*" / "*.parquet")
    parquet_files = glob.glob(parquet_pattern)
    passed = len(parquet_files) > 0
    report.add(
        "Parquet 파일 존재 (data/chunk-*/)",
        passed,
        f"{len(parquet_files)}개 parquet 파일 발견",
    )


def check_video_files(dataset_path: Path, report: ValidationReport) -> None:
    """Check that video files exist in videos/chunk-*/ directories."""
    video_pattern = str(dataset_path / "videos" / "chunk-*" / "*.mp4")
    video_files = glob.glob(video_pattern)
    passed = len(video_files) > 0
    report.add(
        "비디오 파일 존재 (videos/chunk-*/)",
        passed,
        f"{len(video_files)}개 비디오 파일 발견",
    )


# ---------------------------------------------------------------------------
# Replay validation
# ---------------------------------------------------------------------------


def run_replay(
    dataset,
    task_name: str,
    num_envs: int,
    tolerance: float,
    report: ValidationReport,
) -> None:
    """Replay stored actions in Isaac Sim and compare resulting states.

    Args:
        dataset: Loaded LeRobotDataset.
        task_name: Isaac Sim environment task identifier.
        num_envs: Number of parallel environments.
        tolerance: Maximum allowed state deviation (radians).
        report: ValidationReport to record results into.
    """
    if not ISAAC_SIM_AVAILABLE:
        report.add(
            "리플레이 검증",
            False,
            "Isaac Sim을 사용할 수 없습니다. Isaac Sim Python 환경에서 실행하세요.",
        )
        return

    if np is None:
        report.add(
            "리플레이 검증",
            False,
            "NumPy가 설치되어 있지 않습니다.",
        )
        return

    logger.info("Initializing Isaac Sim for replay: task=%s", task_name)
    env = ManagerBasedEnv(task=task_name, num_envs=num_envs)

    num_episodes = dataset.meta.total_episodes
    replay_all_ok = True
    replay_details = []

    for ep_idx in range(num_episodes):
        logger.info("Replaying episode %d ...", ep_idx)

        # Get episode data slice
        ep_start = dataset.episode_data_index["from"][ep_idx].item()
        ep_end = dataset.episode_data_index["to"][ep_idx].item()

        obs = env.reset()
        max_deviation = 0.0

        for frame_idx in range(ep_start, ep_end):
            frame = dataset[frame_idx]
            action = frame["action"]

            # Convert to numpy if needed
            if hasattr(action, "numpy"):
                action_np = action.numpy()
            else:
                action_np = np.asarray(action)

            obs, _, terminated, truncated, _ = env.step(action_np)

            # Compare resulting state with recorded state
            if frame_idx + 1 < ep_end:
                next_frame = dataset[frame_idx + 1]
                expected_state = next_frame["observation.state"]
                if hasattr(expected_state, "numpy"):
                    expected_state = expected_state.numpy()
                else:
                    expected_state = np.asarray(expected_state)

                actual_state = obs["joint_pos"]
                if hasattr(actual_state, "numpy"):
                    actual_state = actual_state.numpy()
                else:
                    actual_state = np.asarray(actual_state)

                deviation = np.max(np.abs(expected_state - actual_state))
                max_deviation = max(max_deviation, deviation)

            if terminated.any() or truncated.any():
                break

        ep_passed = max_deviation <= tolerance
        if not ep_passed:
            replay_all_ok = False
        replay_details.append(
            f"episode {ep_idx}: max_deviation={max_deviation:.6f} rad"
        )
        logger.info(
            "Episode %d replay: max_deviation=%.6f rad (%s)",
            ep_idx,
            max_deviation,
            "PASS" if ep_passed else "FAIL",
        )

    env.close()

    report.add(
        "리플레이 검증",
        replay_all_ok,
        f"tolerance={tolerance} rad; " + "; ".join(replay_details),
    )


# ---------------------------------------------------------------------------
# Main validation pipeline
# ---------------------------------------------------------------------------


def resolve_dataset_path(repo_id: str, output_dir: str) -> Path:
    """Resolve the local filesystem path for a LeRobot dataset.

    Args:
        repo_id: Dataset repository ID (e.g. 'local/so101_teleop').
        output_dir: Root output directory (e.g. 'datasets').

    Returns:
        Path to the dataset directory.
    """
    # LeRobot local datasets are stored as <output_dir>/<dataset_name>
    dataset_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    return Path(output_dir) / dataset_name


def run_validation(args: argparse.Namespace) -> bool:
    """Run the full validation pipeline.

    Args:
        args: Parsed CLI arguments.

    Returns:
        True if all checks passed, False otherwise.
    """
    # Load pipeline parameters for expected feature schema
    pipeline_params = _load_yaml(_DATA_PIPELINE_YAML)
    expected_features = load_expected_features(pipeline_params)

    # Resolve dataset location
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        output_dir = _get_value(pipeline_params["dataset"]["output_dir"])
        repo_id = args.repo_id or _get_value(
            pipeline_params["dataset"]["repo_id"]
        )
        dataset_path = resolve_dataset_path(repo_id, output_dir)

    repo_id = args.repo_id or _get_value(
        pipeline_params["dataset"]["repo_id"]
    )

    logger.info("=== SO-ARM101 Dataset Validation ===")
    logger.info("Dataset path: %s", dataset_path)
    logger.info("Repo ID:      %s", repo_id)
    logger.info("Replay mode:  %s", args.replay)

    report = ValidationReport()

    # --- File-system level checks (no LeRobot needed) ---
    check_meta_files(dataset_path, report)
    check_parquet_files(dataset_path, report)
    check_video_files(dataset_path, report)

    # --- Dataset-level checks (requires LeRobot) ---
    if LeRobotDataset is None:
        logger.error(
            "LeRobot is not installed. "
            "Install with: pip install lerobot"
        )
        report.add(
            "LeRobot 데이터셋 로드",
            False,
            "LeRobot이 설치되어 있지 않습니다.",
        )
        report.log_report()
        return report.all_passed

    try:
        dataset = LeRobotDataset(repo_id=repo_id)
        report.add("LeRobot 데이터셋 로드", True, f"repo_id={repo_id}")
    except Exception as exc:
        report.add(
            "LeRobot 데이터셋 로드",
            False,
            f"로드 실패: {exc}",
        )
        report.log_report()
        return report.all_passed

    check_episode_count(dataset, report)
    check_frame_counts(dataset, report)
    check_feature_shapes(dataset, expected_features, report)
    check_feature_dtypes(dataset, expected_features, report)

    # --- Replay validation (optional) ---
    if args.replay:
        task_name = args.task_name or _get_value(
            pipeline_params["isaac_sim"]["task"]
        )
        num_envs = _get_value(pipeline_params["isaac_sim"]["num_envs"])
        tolerance = args.tolerance

        logger.info("Running replay validation: task=%s", task_name)
        run_replay(
            dataset=dataset,
            task_name=task_name,
            num_envs=num_envs,
            tolerance=tolerance,
            report=report,
        )

    # --- Print report ---
    report.log_report()
    return report.all_passed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset validation.

    Supports both repo_id and direct dataset_path specification.
    """
    parser = argparse.ArgumentParser(
        description="SO-ARM101 LeRobot v2 데이터셋 검증"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="LeRobot 데이터셋 리포지토리 ID (예: local/so101_teleop)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="데이터셋 로컬 디렉토리 경로 (--repo_id 대신 사용 가능)",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        default=False,
        help="시뮬레이션 리플레이 검증 활성화 (Isaac Sim 환경 필요)",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="리플레이 시 사용할 Isaac Sim 태스크 이름 "
        "(기본값: params/data_pipeline.yaml에서 로드)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="리플레이 상태 비교 허용 오차 (radian, 기본값: 0.01)",
    )
    parser.add_argument(
        "--data_pipeline_yaml",
        type=str,
        default=str(_DATA_PIPELINE_YAML),
        help="데이터 파이프라인 파라미터 파일 경로",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Headless 모드 실행 (기본값: True, --no-headless로 GUI 활성화)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for dataset validation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    # Allow overriding YAML path via CLI
    global _DATA_PIPELINE_YAML
    _DATA_PIPELINE_YAML = Path(args.data_pipeline_yaml)

    try:
        all_passed = run_validation(args)
    except FileNotFoundError as exc:
        logger.error("파라미터 파일을 찾을 수 없습니다: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("검증 중 예기치 못한 오류 발생: %s", exc)
        sys.exit(1)

    if not all_passed:
        logger.error("검증 실패: 일부 항목이 FAIL입니다.")
        sys.exit(1)

    logger.info("검증 완료: 모든 항목 PASS.")


if __name__ == "__main__":
    main()
