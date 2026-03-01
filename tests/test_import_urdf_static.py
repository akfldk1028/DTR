#!/usr/bin/env python3
"""test_import_urdf_static.py — Static verification for import_urdf_to_isaac.py

Verifies the URDF→USD conversion script through static analysis:
  1. Python syntax check via py_compile
  2. argparse --input/--output CLI support
  3. YAML parameter loading (load_params, get_param_value)
  4. Logging setup (_setup_logging) handler configuration
  5. URDF XML joint structure validation

These tests do NOT import omni.* or isaacsim — they test only
pure-Python functions and static code properties.
"""

import logging
import os
import py_compile
import sys
import tempfile
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# Path setup — add scripts/ to sys.path for importing pure-Python functions
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

# Import pure-Python functions from import_urdf_to_isaac.py
# Guard: only import functions that do not depend on omni/isaacsim
from import_urdf_to_isaac import (
    DEFAULT_PARAMS_CONTROL,
    DEFAULT_PARAMS_PHYSICS,
    EXPECTED_JOINT_NAMES,
    EXPECTED_NUM_JOINTS,
    get_param_value,
    load_params,
    _setup_logging,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_PATH = os.path.join(SCRIPTS_DIR, "import_urdf_to_isaac.py")
URDF_PATH = os.path.join(PROJECT_ROOT, "robot_description", "urdf", "so101_new_calib.urdf")
CONTROL_YAML = os.path.join(PROJECT_ROOT, DEFAULT_PARAMS_CONTROL)
PHYSICS_YAML = os.path.join(PROJECT_ROOT, DEFAULT_PARAMS_PHYSICS)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
class StaticVerificationRunner:
    """Runs all static verification tests and tracks results."""

    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0

    def run_test(self, name, test_fn):
        """Run a single test function and record the result."""
        try:
            test_fn()
            self.results.append((name, "PASS", ""))
            self.passed += 1
        except AssertionError as exc:
            self.results.append((name, "FAIL", str(exc)))
            self.failed += 1
        except Exception as exc:
            self.results.append((name, "ERROR", str(exc)))
            self.failed += 1

    def print_summary(self):
        """Print test results summary."""
        sys.stderr.write("\n=== Static Verification Results ===\n")
        for name, status, msg in self.results:
            marker = "[PASS]" if status == "PASS" else f"[{status}]"
            line = f"  {marker} {name}"
            if msg:
                line += f" — {msg}"
            sys.stderr.write(line + "\n")
        sys.stderr.write(f"\nTotal: {self.passed + self.failed} | ")
        sys.stderr.write(f"Passed: {self.passed} | Failed: {self.failed}\n")
        sys.stderr.write("=" * 40 + "\n")


# ---------------------------------------------------------------------------
# Test 1: Python syntax check via py_compile
# ---------------------------------------------------------------------------
def test_python_syntax():
    """Verify import_urdf_to_isaac.py compiles without syntax errors."""
    assert os.path.exists(SCRIPT_PATH), (
        f"Script not found: {SCRIPT_PATH}"
    )
    # py_compile.compile raises py_compile.PyCompileError on syntax error
    py_compile.compile(SCRIPT_PATH, doraise=True)


# ---------------------------------------------------------------------------
# Test 2: argparse --input/--output support
# ---------------------------------------------------------------------------
def test_argparse_input_output():
    """Verify parse_args() accepts --input and --output arguments."""
    from import_urdf_to_isaac import parse_args

    # Monkey-patch sys.argv to simulate CLI invocation
    original_argv = sys.argv
    try:
        sys.argv = [
            "import_urdf_to_isaac.py",
            "--input", "robot_description/urdf/so101_new_calib.urdf",
            "--output", "assets/usd/test_output.usd",
        ]
        args = parse_args()

        assert args.input == "robot_description/urdf/so101_new_calib.urdf", (
            f"Expected --input value, got: {args.input}"
        )
        assert args.output == "assets/usd/test_output.usd", (
            f"Expected --output value, got: {args.output}"
        )
    finally:
        sys.argv = original_argv


def test_argparse_default_output():
    """Verify parse_args() defaults --output to 'assets/usd/'."""
    from import_urdf_to_isaac import parse_args

    original_argv = sys.argv
    try:
        sys.argv = [
            "import_urdf_to_isaac.py",
            "--input", "robot_description/urdf/so101_new_calib.urdf",
        ]
        args = parse_args()

        assert args.output == "assets/usd/", (
            f"Expected default output 'assets/usd/', got: {args.output}"
        )
    finally:
        sys.argv = original_argv


def test_argparse_headless_default():
    """Verify --headless defaults to True."""
    from import_urdf_to_isaac import parse_args

    original_argv = sys.argv
    try:
        sys.argv = [
            "import_urdf_to_isaac.py",
            "--input", "test.urdf",
        ]
        args = parse_args()
        assert args.headless is True, (
            f"Expected --headless default True, got: {args.headless}"
        )
    finally:
        sys.argv = original_argv


# ---------------------------------------------------------------------------
# Test 3: YAML parameter loading
# ---------------------------------------------------------------------------
def test_load_params_control():
    """Verify load_params() reads params/control.yaml successfully."""
    assert os.path.exists(CONTROL_YAML), (
        f"Control YAML not found: {CONTROL_YAML}"
    )
    params = load_params(CONTROL_YAML)
    assert isinstance(params, dict), "load_params should return a dict"
    assert "drive" in params, "control.yaml must contain 'drive' key"
    assert "joint_limits" in params, "control.yaml must contain 'joint_limits' key"
    assert "joint_names" in params, "control.yaml must contain 'joint_names' key"


def test_load_params_physics():
    """Verify load_params() reads params/physics.yaml successfully."""
    assert os.path.exists(PHYSICS_YAML), (
        f"Physics YAML not found: {PHYSICS_YAML}"
    )
    params = load_params(PHYSICS_YAML)
    assert isinstance(params, dict), "load_params should return a dict"
    assert "simulation" in params, "physics.yaml must contain 'simulation' key"
    assert "friction" in params, "physics.yaml must contain 'friction' key"
    assert "mass" in params, "physics.yaml must contain 'mass' key"


def test_load_params_missing_file():
    """Verify load_params() raises FileNotFoundError for missing files."""
    try:
        load_params("/nonexistent/path/missing.yaml")
        raise AssertionError("Expected FileNotFoundError was not raised")
    except FileNotFoundError:
        pass  # expected


def test_get_param_value_drive_stiffness():
    """Verify get_param_value() extracts drive.stiffness from control.yaml."""
    params = load_params(CONTROL_YAML)
    stiffness = get_param_value(params, "drive", "stiffness")
    assert stiffness == 40.0, (
        f"Expected drive.stiffness=40.0, got: {stiffness}"
    )


def test_get_param_value_drive_damping():
    """Verify get_param_value() extracts drive.damping from control.yaml."""
    params = load_params(CONTROL_YAML)
    damping = get_param_value(params, "drive", "damping")
    assert damping == 4.0, (
        f"Expected drive.damping=4.0, got: {damping}"
    )


def test_get_param_value_joint_names():
    """Verify get_param_value() extracts joint_names from control.yaml."""
    params = load_params(CONTROL_YAML)
    joint_names = get_param_value(params, "joint_names")
    assert isinstance(joint_names, list), "joint_names should be a list"
    assert len(joint_names) == 6, (
        f"Expected 6 joint names, got: {len(joint_names)}"
    )
    for name in EXPECTED_JOINT_NAMES:
        assert name in joint_names, (
            f"Expected joint name '{name}' in joint_names list"
        )


def test_get_param_value_physics_gravity():
    """Verify get_param_value() extracts simulation.gravity from physics.yaml."""
    params = load_params(PHYSICS_YAML)
    gravity = get_param_value(params, "simulation", "gravity")
    assert isinstance(gravity, list), "gravity should be a list"
    assert len(gravity) == 3, f"gravity should have 3 components, got: {len(gravity)}"
    assert gravity[2] == -9.81, f"Expected gravity[2]=-9.81, got: {gravity[2]}"


def test_get_param_value_physics_timestep():
    """Verify get_param_value() extracts simulation.timestep from physics.yaml."""
    params = load_params(PHYSICS_YAML)
    timestep = get_param_value(params, "simulation", "timestep")
    assert timestep == 0.005, f"Expected timestep=0.005, got: {timestep}"


def test_get_param_value_missing_key():
    """Verify get_param_value() raises KeyError for missing keys."""
    params = load_params(CONTROL_YAML)
    try:
        get_param_value(params, "nonexistent", "key")
        raise AssertionError("Expected KeyError was not raised")
    except KeyError:
        pass  # expected


def test_get_param_value_joint_limits():
    """Verify get_param_value() extracts joint_limits.position_min/max."""
    params = load_params(CONTROL_YAML)

    position_min = get_param_value(params, "joint_limits", "position_min")
    assert isinstance(position_min, list), "position_min should be a list"
    assert len(position_min) == 6, (
        f"Expected 6 position_min values, got: {len(position_min)}"
    )

    position_max = get_param_value(params, "joint_limits", "position_max")
    assert isinstance(position_max, list), "position_max should be a list"
    assert len(position_max) == 6, (
        f"Expected 6 position_max values, got: {len(position_max)}"
    )

    # Verify min < max for each joint
    for i in range(6):
        assert position_min[i] < position_max[i], (
            f"Joint {i}: position_min ({position_min[i]}) >= "
            f"position_max ({position_max[i]})"
        )


# ---------------------------------------------------------------------------
# Test 4: Logging setup — _setup_logging produces FileHandler + StreamHandler
# ---------------------------------------------------------------------------
def test_setup_logging_handlers():
    """Verify _setup_logging() creates FileHandler and StreamHandler."""
    # Use a temp file to avoid polluting the project directory
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test_log.log")

        # Clear any existing handlers from previous test runs
        logger_name = "urdf_to_usd"
        existing_logger = logging.getLogger(logger_name)
        existing_logger.handlers.clear()

        logger = _setup_logging(log_file=log_file)

        assert logger.name == "urdf_to_usd", (
            f"Expected logger name 'urdf_to_usd', got: {logger.name}"
        )
        assert logger.level == logging.DEBUG, (
            f"Expected log level DEBUG, got: {logger.level}"
        )

        # Check handler types
        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "FileHandler" in handler_types, (
            f"Expected FileHandler in handlers, got: {handler_types}"
        )
        assert "StreamHandler" in handler_types, (
            f"Expected StreamHandler in handlers, got: {handler_types}"
        )

        # Verify exactly 2 handlers
        assert len(logger.handlers) == 2, (
            f"Expected 2 handlers, got: {len(logger.handlers)}"
        )

        # Verify log file was created
        assert os.path.exists(log_file), (
            f"Log file was not created: {log_file}"
        )

        # Write a test message and verify it appears in the file
        logger.info("Static verification test message")
        # Flush handlers
        for handler in logger.handlers:
            handler.flush()

        with open(log_file, "r", encoding="utf-8") as fh:
            log_content = fh.read()
        assert "Static verification test message" in log_content, (
            "Test log message not found in log file"
        )

        # Clean up handlers to avoid interference with other tests
        logger.handlers.clear()


def test_setup_logging_no_duplicate_handlers():
    """Verify _setup_logging() does not create duplicate handlers on repeated calls."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test_dup.log")

        # Clear existing handlers
        logger_name = "urdf_to_usd"
        existing_logger = logging.getLogger(logger_name)
        existing_logger.handlers.clear()

        logger1 = _setup_logging(log_file=log_file)
        handler_count_1 = len(logger1.handlers)

        logger2 = _setup_logging(log_file=log_file)
        handler_count_2 = len(logger2.handlers)

        assert handler_count_1 == handler_count_2, (
            f"Duplicate handler creation: {handler_count_1} -> {handler_count_2}"
        )
        assert logger1 is logger2, (
            "Repeated calls should return the same logger instance"
        )

        # Clean up
        logger1.handlers.clear()


def test_setup_logging_stderr_stream():
    """Verify StreamHandler outputs to stderr (not stdout)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test_stderr.log")

        logger_name = "urdf_to_usd"
        existing_logger = logging.getLogger(logger_name)
        existing_logger.handlers.clear()

        logger = _setup_logging(log_file=log_file)

        stream_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 1, (
            f"Expected 1 StreamHandler, got: {len(stream_handlers)}"
        )
        assert stream_handlers[0].stream is sys.stderr, (
            "StreamHandler should output to stderr (Isaac Sim captures stdout)"
        )

        # Clean up
        logger.handlers.clear()


# ---------------------------------------------------------------------------
# Test 5: URDF XML has exactly 6 revolute joints with expected names
# ---------------------------------------------------------------------------
def test_urdf_exists():
    """Verify the URDF file exists at the expected path."""
    assert os.path.exists(URDF_PATH), (
        f"URDF file not found: {URDF_PATH}"
    )


def test_urdf_valid_xml():
    """Verify the URDF file is valid XML."""
    tree = ET.parse(URDF_PATH)
    root = tree.getroot()
    assert root.tag == "robot", (
        f"Expected root tag 'robot', got: {root.tag}"
    )


def test_urdf_revolute_joint_count():
    """Verify the URDF has exactly 6 revolute joints."""
    tree = ET.parse(URDF_PATH)
    root = tree.getroot()

    revolute_joints = [
        joint for joint in root.findall("joint")
        if joint.get("type") == "revolute"
    ]

    assert len(revolute_joints) == EXPECTED_NUM_JOINTS, (
        f"Expected {EXPECTED_NUM_JOINTS} revolute joints, "
        f"got: {len(revolute_joints)}"
    )


def test_urdf_joint_names():
    """Verify all 6 expected revolute joint names exist in the URDF."""
    tree = ET.parse(URDF_PATH)
    root = tree.getroot()

    revolute_joints = [
        joint for joint in root.findall("joint")
        if joint.get("type") == "revolute"
    ]
    actual_names = [joint.get("name") for joint in revolute_joints]

    for expected_name in EXPECTED_JOINT_NAMES:
        assert expected_name in actual_names, (
            f"Expected joint name '{expected_name}' not found in URDF. "
            f"Actual names: {actual_names}"
        )


def test_urdf_joint_limits_present():
    """Verify all revolute joints have limit elements with lower/upper."""
    tree = ET.parse(URDF_PATH)
    root = tree.getroot()

    revolute_joints = [
        joint for joint in root.findall("joint")
        if joint.get("type") == "revolute"
    ]

    for joint in revolute_joints:
        name = joint.get("name")
        limit_elem = joint.find("limit")
        assert limit_elem is not None, (
            f"Joint '{name}' is missing <limit> element"
        )
        lower = limit_elem.get("lower")
        upper = limit_elem.get("upper")
        assert lower is not None, (
            f"Joint '{name}' <limit> missing 'lower' attribute"
        )
        assert upper is not None, (
            f"Joint '{name}' <limit> missing 'upper' attribute"
        )
        # Verify lower < upper
        lower_val = float(lower)
        upper_val = float(upper)
        assert lower_val < upper_val, (
            f"Joint '{name}': lower ({lower_val}) >= upper ({upper_val})"
        )


def test_urdf_joint_names_match_constants():
    """Verify EXPECTED_JOINT_NAMES constant matches actual URDF joints."""
    tree = ET.parse(URDF_PATH)
    root = tree.getroot()

    revolute_joints = [
        joint for joint in root.findall("joint")
        if joint.get("type") == "revolute"
    ]
    actual_names = sorted([joint.get("name") for joint in revolute_joints])
    expected_names = sorted(EXPECTED_JOINT_NAMES)

    assert actual_names == expected_names, (
        f"Joint name mismatch:\n"
        f"  URDF:     {actual_names}\n"
        f"  Expected: {expected_names}"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    runner = StaticVerificationRunner()

    # Test 1: Python syntax
    runner.run_test("Python syntax (py_compile)", test_python_syntax)

    # Test 2: argparse --input/--output
    runner.run_test("argparse --input/--output", test_argparse_input_output)
    runner.run_test("argparse default --output", test_argparse_default_output)
    runner.run_test("argparse --headless default", test_argparse_headless_default)

    # Test 3: YAML param loading
    runner.run_test("load_params(control.yaml)", test_load_params_control)
    runner.run_test("load_params(physics.yaml)", test_load_params_physics)
    runner.run_test("load_params(missing file)", test_load_params_missing_file)
    runner.run_test("get_param_value(drive.stiffness)", test_get_param_value_drive_stiffness)
    runner.run_test("get_param_value(drive.damping)", test_get_param_value_drive_damping)
    runner.run_test("get_param_value(joint_names)", test_get_param_value_joint_names)
    runner.run_test("get_param_value(gravity)", test_get_param_value_physics_gravity)
    runner.run_test("get_param_value(timestep)", test_get_param_value_physics_timestep)
    runner.run_test("get_param_value(missing key)", test_get_param_value_missing_key)
    runner.run_test("get_param_value(joint_limits)", test_get_param_value_joint_limits)

    # Test 4: Logging setup
    runner.run_test("_setup_logging handlers", test_setup_logging_handlers)
    runner.run_test("_setup_logging no duplicates", test_setup_logging_no_duplicate_handlers)
    runner.run_test("_setup_logging stderr stream", test_setup_logging_stderr_stream)

    # Test 5: URDF XML validation
    runner.run_test("URDF file exists", test_urdf_exists)
    runner.run_test("URDF valid XML", test_urdf_valid_xml)
    runner.run_test("URDF 6 revolute joints", test_urdf_revolute_joint_count)
    runner.run_test("URDF expected joint names", test_urdf_joint_names)
    runner.run_test("URDF joint limits present", test_urdf_joint_limits_present)
    runner.run_test("URDF joint names match constants", test_urdf_joint_names_match_constants)

    # Summary
    runner.print_summary()

    if runner.failed == 0:
        sys.stderr.write("All static verification tests passed\n")
        return 0
    else:
        sys.stderr.write(
            f"FAILED: {runner.failed} of {runner.passed + runner.failed} tests failed\n"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
