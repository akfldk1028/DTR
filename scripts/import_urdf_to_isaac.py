#!/usr/bin/env python3
"""import_urdf_to_isaac.py — URDF→USD 변환 자동화 스크립트

목적:
    URDF 파일을 Isaac Sim의 URDF Importer를 통해 USD로 변환한다.
    robot_description/urdf/의 URDF를 입력받아 assets/usd/에 USD를 생성한다.

사용법:
    python scripts/import_urdf_to_isaac.py \\
        --input robot_description/urdf/so_arm101.urdf \\
        --output assets/usd/so_arm101.usd

    python scripts/import_urdf_to_isaac.py \\
        --input robot_description/urdf/so_arm101.urdf \\
        --output assets/usd/ \\
        --skip-verify

필요 환경:
    - Isaac Sim 5.1.0 Python 환경
    - URDF Importer extension 활성화

참고:
    - Isaac Sim URDF Importer 공식 문서: docs/references.md 참조
    - Isaac Lab 자산 Import 가이드: docs/references.md 참조

Phase: 4
상태: 구현 중
"""

# ---------------------------------------------------------------------------
# SimulationApp MUST be initialized BEFORE any omni.* imports.
# This is an Isaac Sim requirement — do not reorder these imports.
# ---------------------------------------------------------------------------
import argparse
import logging
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_NUM_JOINTS = 6
EXPECTED_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
NUM_STABILITY_STEPS = 200
LOG_FILE = "assets/urdf_import_results.log"
DEFAULT_PARAMS_CONTROL = "params/control.yaml"
DEFAULT_PARAMS_PHYSICS = "params/physics.yaml"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def _setup_logging(log_file: str = LOG_FILE) -> logging.Logger:
    """Configure logging with FileHandler and StreamHandler(stderr).

    Isaac Sim captures stdout, so all log output goes to stderr
    and a persistent log file for post-run inspection.

    Args:
        log_file: Path to the log file (default: assets/urdf_import_results.log).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("urdf_to_usd")
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # FileHandler — persistent log
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # StreamHandler — stderr (Isaac Sim captures stdout)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


# ---------------------------------------------------------------------------
# YAML parameter loading
# ---------------------------------------------------------------------------
def load_params(yaml_path: str) -> dict:
    """Load parameters from a YAML file.

    The YAML files follow the project schema:
        key:
          value: <actual value>
          unit: <SI unit string>
          range: [min, max]
          description: <string>

    This function returns the raw dict; callers extract ``value`` fields
    as needed.

    Args:
        yaml_path: Path to the YAML parameter file.

    Returns:
        Parsed YAML dict.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Parameter file not found: {yaml_path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    return data


def get_param_value(params: dict, *keys):
    """Extract a nested ``value`` field from loaded YAML params.

    Example::

        control = load_params("params/control.yaml")
        stiffness = get_param_value(control, "drive", "stiffness")
        # returns 40.0

    Args:
        params: Dict returned by :func:`load_params`.
        *keys: Sequence of nested keys leading to the parameter block.

    Returns:
        The ``value`` field of the target parameter block.

    Raises:
        KeyError: If a key is missing in the hierarchy.
    """
    node = params
    for key in keys:
        node = node[key]
    if isinstance(node, dict) and "value" in node:
        return node["value"]
    return node


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    """Parse command-line arguments for URDF→USD conversion.

    Preserved CLI interface:
        --input   (required) URDF file path
        --output  (optional) USD output path (file or directory)

    Added flags:
        --headless         Run SimulationApp without GUI (default: True)
        --skip-verify      Skip post-conversion verification steps
        --params-control   Path to control.yaml
        --params-physics   Path to physics.yaml
    """
    parser = argparse.ArgumentParser(
        description="URDF→USD 변환 (Isaac Sim URDF Importer)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="입력 URDF 파일 경로",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="assets/usd/",
        help="출력 USD 저장 경로 (파일 또는 디렉토리)",
    )
    parser.add_argument(
        "--fix-base",
        action="store_true",
        default=True,
        help="베이스 링크 고정 여부",
    )
    parser.add_argument(
        "--merge-fixed-joints",
        action="store_true",
        default=False,
        help="고정 조인트 병합 여부",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="GUI 없이 headless 모드로 실행 (default: True)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        default=False,
        help="변환 후 검증 단계 건너뛰기",
    )
    parser.add_argument(
        "--params-control",
        type=str,
        default=DEFAULT_PARAMS_CONTROL,
        help="제어 파라미터 YAML 경로 (default: params/control.yaml)",
    )
    parser.add_argument(
        "--params-physics",
        type=str,
        default=DEFAULT_PARAMS_PHYSICS,
        help="물리 파라미터 YAML 경로 (default: params/physics.yaml)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# SimulationApp initialization
# ---------------------------------------------------------------------------
def init_simulation_app(headless: bool = True):
    """Initialize Isaac Sim SimulationApp in headless mode.

    SimulationApp MUST be created before any ``omni.*`` imports.
    This function handles the initialization and returns the app instance.

    Args:
        headless: If True, run without GUI rendering.

    Returns:
        SimulationApp instance.
    """
    from isaacsim import SimulationApp

    config = {
        "headless": headless,
        "width": 1280,
        "height": 720,
    }
    app = SimulationApp(config)
    return app


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    logger = _setup_logging()

    logger.info("=== URDF → USD Converter ===")
    logger.info("Input URDF: %s", args.input)
    logger.info("Output path: %s", args.output)
    logger.info("Fix base: %s", args.fix_base)
    logger.info("Merge fixed joints: %s", args.merge_fixed_joints)
    logger.info("Headless: %s", args.headless)
    logger.info("Skip verify: %s", args.skip_verify)

    # Validate input URDF exists
    urdf_path = Path(args.input)
    if not urdf_path.exists():
        logger.error("URDF file not found: %s", args.input)
        sys.exit(1)

    # Load parameters from YAML
    try:
        control_params = load_params(args.params_control)
        logger.info("Loaded control params from: %s", args.params_control)
        logger.debug(
            "  drive.stiffness = %s",
            get_param_value(control_params, "drive", "stiffness"),
        )
        logger.debug(
            "  drive.damping = %s",
            get_param_value(control_params, "drive", "damping"),
        )
        logger.debug(
            "  drive.max_effort = %s",
            get_param_value(control_params, "drive", "max_effort"),
        )
    except (FileNotFoundError, yaml.YAMLError) as exc:
        logger.error("Failed to load control params: %s", exc)
        sys.exit(1)

    try:
        physics_params = load_params(args.params_physics)
        logger.info("Loaded physics params from: %s", args.params_physics)
        logger.debug(
            "  simulation.timestep = %s",
            get_param_value(physics_params, "simulation", "timestep"),
        )
        logger.debug(
            "  simulation.gravity = %s",
            get_param_value(physics_params, "simulation", "gravity"),
        )
    except (FileNotFoundError, yaml.YAMLError) as exc:
        logger.error("Failed to load physics params: %s", exc)
        sys.exit(1)

    # Resolve output path
    output_path = Path(args.output)
    if output_path.is_dir() or str(args.output).endswith("/"):
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / urdf_path.with_suffix(".usd").name
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Resolved output USD: %s", output_path)

    # Initialize SimulationApp (must happen before omni imports)
    logger.info("Initializing SimulationApp (headless=%s)...", args.headless)
    app = None
    try:
        app = init_simulation_app(headless=args.headless)
        logger.info("SimulationApp initialized successfully")

        # --- Phase 2: URDF conversion pipeline will be added here ---
        # --- Phase 3: Verification & stability test will be added here ---

    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)
    finally:
        if app is not None:
            logger.info("Closing SimulationApp...")
            app.close()
            logger.info("SimulationApp closed")


if __name__ == "__main__":
    main()
