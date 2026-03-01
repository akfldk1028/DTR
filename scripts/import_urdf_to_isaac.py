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
# URDF Import Configuration
# ---------------------------------------------------------------------------
def create_import_config(
    fix_base: bool = True,
    merge_fixed_joints: bool = False,
) -> object:
    """Create and configure URDF import settings via Isaac Sim API.

    Uses ``omni.kit.commands.execute("URDFCreateImportConfig")`` to obtain
    a config object, then sets all fields for SO-ARM101 import.

    Args:
        fix_base: Whether to fix the robot base link.
        merge_fixed_joints: Whether to merge fixed joints.

    Returns:
        Configured ImportConfig object.
    """
    import omni.kit.commands

    logger = logging.getLogger("urdf_to_usd")

    result, import_config = omni.kit.commands.execute("URDFCreateImportConfig")

    import_config.merge_fixed_joints = merge_fixed_joints
    import_config.convex_decomp = True
    import_config.import_inertia_tensor = True
    import_config.fix_base = fix_base
    import_config.collision_from_visuals = True
    import_config.distance_scale = 1.0

    logger.info("ImportConfig created:")
    logger.info("  merge_fixed_joints = %s", import_config.merge_fixed_joints)
    logger.info("  convex_decomp = %s", import_config.convex_decomp)
    logger.info("  import_inertia_tensor = %s", import_config.import_inertia_tensor)
    logger.info("  fix_base = %s", import_config.fix_base)
    logger.info("  collision_from_visuals = %s", import_config.collision_from_visuals)
    logger.info("  distance_scale = %s", import_config.distance_scale)

    return import_config


# ---------------------------------------------------------------------------
# Physics Scene Setup
# ---------------------------------------------------------------------------
def setup_physics_scene(physics_params: dict) -> None:
    """Set up the physics scene with gravity and ground plane.

    Reads gravity vector from physics.yaml and configures the USD
    physics scene accordingly. Adds a ground plane at z=0.

    Args:
        physics_params: Dict loaded from params/physics.yaml.
    """
    import omni.usd
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    logger = logging.getLogger("urdf_to_usd")

    gravity = get_param_value(physics_params, "simulation", "gravity")
    logger.info("Setting up physics scene with gravity = %s", gravity)

    stage = omni.usd.get_context().get_stage()

    # Create /World scope if not present
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim.IsValid():
        UsdGeom.Xform.Define(stage, "/World")
        logger.debug("Created /World Xform")

    # Configure physics scene
    scene_path = "/World/PhysicsScene"
    scene = UsdPhysics.Scene.Define(stage, scene_path)
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(abs(gravity[2]))
    logger.info("Physics scene created at %s (gravity_mag=%.2f)", scene_path, abs(gravity[2]))

    # Add ground plane
    ground_path = "/World/GroundPlane"
    ground_prim = stage.GetPrimAtPath(ground_path)
    if not ground_prim.IsValid():
        from omni.isaac.core.objects import GroundPlane

        GroundPlane(prim_path=ground_path, size=10.0)
        logger.info("Ground plane added at %s", ground_path)


# ---------------------------------------------------------------------------
# URDF Parsing and Robot Import
# ---------------------------------------------------------------------------
ROBOT_PRIM_PATH = "/World/SO101"


def import_urdf(urdf_path: str, import_config: object) -> str:
    """Parse URDF and import robot into the USD stage.

    Uses the low-level URDF interface:
      1. ``_urdf.acquire_urdf_interface()``
      2. ``urdf_interface.parse_urdf(urdf_path, import_config)``
      3. ``urdf_interface.import_robot(...)`` to ``/World/SO101``

    Args:
        urdf_path: Absolute path to the URDF file.
        import_config: Configured ImportConfig from :func:`create_import_config`.

    Returns:
        The USD prim path where the robot was imported.

    Raises:
        RuntimeError: If URDF parsing or import fails.
    """
    import omni.usd
    from omni.importer.urdf import _urdf

    logger = logging.getLogger("urdf_to_usd")

    # Resolve to absolute path for the URDF interface
    abs_urdf_path = str(Path(urdf_path).resolve())
    logger.info("Parsing URDF: %s", abs_urdf_path)

    urdf_interface = _urdf.acquire_urdf_interface()

    # Parse URDF
    parse_result = urdf_interface.parse_urdf(abs_urdf_path, import_config)
    if parse_result is None:
        raise RuntimeError(f"Failed to parse URDF: {abs_urdf_path}")
    logger.info("URDF parsed successfully")

    # Import robot into stage
    stage = omni.usd.get_context().get_stage()
    prim_path = urdf_interface.import_robot(
        dest_path=ROBOT_PRIM_PATH,
        urdf_robot=parse_result,
        import_config=import_config,
        stage=stage,
    )

    if not prim_path:
        raise RuntimeError(f"Failed to import robot to {ROBOT_PRIM_PATH}")

    logger.info("Robot imported at: %s", prim_path)
    return prim_path


# ---------------------------------------------------------------------------
# Drive Parameter Application
# ---------------------------------------------------------------------------
def apply_drive_params(prim_path: str, control_params: dict) -> int:
    """Apply drive stiffness, damping, and max effort to all joints.

    Traverses the USD stage under the robot prim path to find all
    ``UsdPhysics.RevoluteJoint`` prims and applies the drive parameters
    from control.yaml.

    Args:
        prim_path: USD prim path of the imported robot.
        control_params: Dict loaded from params/control.yaml.

    Returns:
        Number of joints with drives applied.
    """
    import omni.usd
    from pxr import Sdf, UsdPhysics

    logger = logging.getLogger("urdf_to_usd")

    stiffness = get_param_value(control_params, "drive", "stiffness")
    damping = get_param_value(control_params, "drive", "damping")
    max_effort = get_param_value(control_params, "drive", "max_effort")

    logger.info(
        "Applying drive params: stiffness=%.1f, damping=%.1f, max_effort=%.1f",
        stiffness,
        damping,
        max_effort,
    )

    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath(prim_path)

    if not robot_prim.IsValid():
        raise RuntimeError(f"Robot prim not found at: {prim_path}")

    joint_count = 0

    # Traverse all descendant prims to find joints
    for prim in stage.Traverse():
        # Only process prims under the robot path
        if not str(prim.GetPath()).startswith(prim_path):
            continue

        # Check if this prim is a joint (RevoluteJoint or PrismaticJoint)
        joint = UsdPhysics.RevoluteJoint(prim)
        if not joint:
            continue

        # Check if the prim actually has the RevoluteJoint schema applied
        if not prim.HasAPI(UsdPhysics.DriveAPI):
            # Apply drive API to angular axis
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        else:
            drive = UsdPhysics.DriveAPI(prim, "angular")

        # Set drive parameters
        drive.CreateStiffnessAttr(stiffness)
        drive.CreateDampingAttr(damping)
        drive.CreateMaxForceAttr(max_effort)

        joint_count += 1
        logger.debug(
            "  Joint %s: stiffness=%.1f, damping=%.1f, max_effort=%.1f",
            prim.GetName(),
            stiffness,
            damping,
            max_effort,
        )

    logger.info("Drive params applied to %d joints", joint_count)
    return joint_count


# ---------------------------------------------------------------------------
# USD Export
# ---------------------------------------------------------------------------
def export_usd(output_path: Path) -> None:
    """Export the current USD stage to a file.

    Args:
        output_path: Target file path for the USD export.

    Raises:
        RuntimeError: If the export fails.
    """
    import omni.usd

    logger = logging.getLogger("urdf_to_usd")

    stage = omni.usd.get_context().get_stage()
    abs_output = str(output_path.resolve())

    logger.info("Exporting USD to: %s", abs_output)
    result = stage.Export(abs_output)

    if not result:
        raise RuntimeError(f"Failed to export USD to: {abs_output}")

    logger.info("USD exported successfully: %s", abs_output)


# ---------------------------------------------------------------------------
# Conversion Summary Logging
# ---------------------------------------------------------------------------
def log_conversion_summary(
    urdf_path: Path,
    output_path: Path,
    prim_path: str,
    joint_count: int,
) -> None:
    """Log a summary of the URDF→USD conversion.

    Reports joint count, link count, and output file size.

    Args:
        urdf_path: Input URDF file path.
        output_path: Output USD file path.
        prim_path: USD prim path where the robot was imported.
        joint_count: Number of joints with drives applied.
    """
    import omni.usd

    logger = logging.getLogger("urdf_to_usd")

    # Count links by traversing prims under the robot
    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath(prim_path)
    link_count = 0
    if robot_prim.IsValid():
        for prim in stage.Traverse():
            if not str(prim.GetPath()).startswith(prim_path):
                continue
            # Links are typically Xform prims with rigid body API
            if prim.GetTypeName() in ("Xform", "Mesh") or prim.HasAPI(
                __import__("pxr").UsdPhysics.RigidBodyAPI
            ):
                link_count += 1

    # Get output file size
    file_size_bytes = 0
    if output_path.exists():
        file_size_bytes = output_path.stat().st_size

    file_size_kb = file_size_bytes / 1024.0

    logger.info("=== Conversion Summary ===")
    logger.info("  Input URDF:   %s", urdf_path)
    logger.info("  Output USD:   %s", output_path)
    logger.info("  Robot prim:   %s", prim_path)
    logger.info("  Joints:       %d", joint_count)
    logger.info("  Links/prims:  %d", link_count)
    logger.info("  File size:    %.1f KB (%d bytes)", file_size_kb, file_size_bytes)
    logger.info("==========================")


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

        # ---------------------------------------------------------------
        # Phase 2: URDF Conversion Pipeline
        # ---------------------------------------------------------------

        # Step 1: Create URDF import configuration
        import_config = create_import_config(
            fix_base=args.fix_base,
            merge_fixed_joints=args.merge_fixed_joints,
        )

        # Step 2: Set up physics scene (gravity, ground plane)
        setup_physics_scene(physics_params)

        # Step 3: Parse URDF and import robot
        prim_path = import_urdf(str(urdf_path), import_config)

        # Step 4: Apply drive parameters from control.yaml
        joint_count = apply_drive_params(prim_path, control_params)

        # Step 5: Export USD to output path
        export_usd(output_path)

        # Step 6: Log conversion summary
        log_conversion_summary(urdf_path, output_path, prim_path, joint_count)

        logger.info("URDF → USD conversion completed successfully")

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
