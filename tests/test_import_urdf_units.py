#!/usr/bin/env python3
"""test_import_urdf_units.py — Unit tests for import_urdf_to_isaac.py pure-Python functions

Tests functions that do NOT require Isaac Sim or omni.* imports:
  1. parse_args() — various --input/--output combinations, default values
  2. load_params() — valid YAML, missing file error, malformed YAML
  3. get_param_value() — nested key extraction with real control.yaml schema
  4. _setup_logging() — handler configuration, log file creation
  5. Output path resolution — directory auto-names from URDF stem, file preserves name

These tests use tempfile for isolation and sys.argv monkey-patching
for parse_args testing.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Path setup — add scripts/ to sys.path for importing pure-Python functions
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

# Import pure-Python functions from import_urdf_to_isaac.py
from import_urdf_to_isaac import (
    DEFAULT_PARAMS_CONTROL,
    DEFAULT_PARAMS_PHYSICS,
    EXPECTED_JOINT_NAMES,
    EXPECTED_NUM_JOINTS,
    LOG_FILE,
    get_param_value,
    load_params,
    parse_args,
    _setup_logging,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONTROL_YAML = os.path.join(PROJECT_ROOT, DEFAULT_PARAMS_CONTROL)
PHYSICS_YAML = os.path.join(PROJECT_ROOT, DEFAULT_PARAMS_PHYSICS)


# ---------------------------------------------------------------------------
# Test runner (same pattern as test_import_urdf_static.py)
# ---------------------------------------------------------------------------
class UnitTestRunner:
    """Runs all unit tests and tracks results."""

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
        sys.stderr.write("\n=== Unit Test Results ===\n")
        for name, status, msg in self.results:
            marker = "[PASS]" if status == "PASS" else f"[{status}]"
            line = f"  {marker} {name}"
            if msg:
                line += f" — {msg}"
            sys.stderr.write(line + "\n")
        sys.stderr.write(f"\nTotal: {self.passed + self.failed} | ")
        sys.stderr.write(f"Passed: {self.passed} | Failed: {self.failed}\n")
        sys.stderr.write("=" * 40 + "\n")


# ===========================================================================
# Test 1: parse_args() — various --input/--output combinations
# ===========================================================================

def test_parse_args_input_output():
    """parse_args() accepts explicit --input and --output arguments."""
    original_argv = sys.argv
    try:
        sys.argv = [
            "import_urdf_to_isaac.py",
            "--input", "robot_description/urdf/so101.urdf",
            "--output", "assets/usd/so101.usd",
        ]
        args = parse_args()
        assert args.input == "robot_description/urdf/so101.urdf", (
            f"--input mismatch: {args.input}"
        )
        assert args.output == "assets/usd/so101.usd", (
            f"--output mismatch: {args.output}"
        )
    finally:
        sys.argv = original_argv


def test_parse_args_default_output():
    """parse_args() defaults --output to 'assets/usd/' when not specified."""
    original_argv = sys.argv
    try:
        sys.argv = [
            "import_urdf_to_isaac.py",
            "--input", "test.urdf",
        ]
        args = parse_args()
        assert args.output == "assets/usd/", (
            f"Expected default output 'assets/usd/', got: {args.output}"
        )
    finally:
        sys.argv = original_argv


def test_parse_args_headless_default_true():
    """parse_args() defaults --headless to True."""
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


def test_parse_args_no_headless():
    """parse_args() supports --no-headless to disable headless mode."""
    original_argv = sys.argv
    try:
        sys.argv = [
            "import_urdf_to_isaac.py",
            "--input", "test.urdf",
            "--no-headless",
        ]
        args = parse_args()
        assert args.headless is False, (
            f"Expected --no-headless to set False, got: {args.headless}"
        )
    finally:
        sys.argv = original_argv


def test_parse_args_skip_verify():
    """parse_args() supports --skip-verify flag."""
    original_argv = sys.argv
    try:
        sys.argv = [
            "import_urdf_to_isaac.py",
            "--input", "test.urdf",
            "--skip-verify",
        ]
        args = parse_args()
        assert args.skip_verify is True, (
            f"Expected --skip-verify True, got: {args.skip_verify}"
        )
    finally:
        sys.argv = original_argv


def test_parse_args_skip_verify_default_false():
    """parse_args() defaults --skip-verify to False."""
    original_argv = sys.argv
    try:
        sys.argv = [
            "import_urdf_to_isaac.py",
            "--input", "test.urdf",
        ]
        args = parse_args()
        assert args.skip_verify is False, (
            f"Expected --skip-verify default False, got: {args.skip_verify}"
        )
    finally:
        sys.argv = original_argv


def test_parse_args_fix_base_default():
    """parse_args() defaults --fix-base to True."""
    original_argv = sys.argv
    try:
        sys.argv = [
            "import_urdf_to_isaac.py",
            "--input", "test.urdf",
        ]
        args = parse_args()
        assert args.fix_base is True, (
            f"Expected --fix-base default True, got: {args.fix_base}"
        )
    finally:
        sys.argv = original_argv


def test_parse_args_no_fix_base():
    """parse_args() supports --no-fix-base to disable base fixing."""
    original_argv = sys.argv
    try:
        sys.argv = [
            "import_urdf_to_isaac.py",
            "--input", "test.urdf",
            "--no-fix-base",
        ]
        args = parse_args()
        assert args.fix_base is False, (
            f"Expected --no-fix-base to set False, got: {args.fix_base}"
        )
    finally:
        sys.argv = original_argv


def test_parse_args_params_control_default():
    """parse_args() defaults --params-control to DEFAULT_PARAMS_CONTROL."""
    original_argv = sys.argv
    try:
        sys.argv = [
            "import_urdf_to_isaac.py",
            "--input", "test.urdf",
        ]
        args = parse_args()
        assert args.params_control == DEFAULT_PARAMS_CONTROL, (
            f"Expected default params-control '{DEFAULT_PARAMS_CONTROL}', "
            f"got: {args.params_control}"
        )
    finally:
        sys.argv = original_argv


def test_parse_args_params_physics_default():
    """parse_args() defaults --params-physics to DEFAULT_PARAMS_PHYSICS."""
    original_argv = sys.argv
    try:
        sys.argv = [
            "import_urdf_to_isaac.py",
            "--input", "test.urdf",
        ]
        args = parse_args()
        assert args.params_physics == DEFAULT_PARAMS_PHYSICS, (
            f"Expected default params-physics '{DEFAULT_PARAMS_PHYSICS}', "
            f"got: {args.params_physics}"
        )
    finally:
        sys.argv = original_argv


def test_parse_args_custom_params_paths():
    """parse_args() accepts custom --params-control and --params-physics paths."""
    original_argv = sys.argv
    try:
        sys.argv = [
            "import_urdf_to_isaac.py",
            "--input", "test.urdf",
            "--params-control", "custom/control.yaml",
            "--params-physics", "custom/physics.yaml",
        ]
        args = parse_args()
        assert args.params_control == "custom/control.yaml", (
            f"--params-control mismatch: {args.params_control}"
        )
        assert args.params_physics == "custom/physics.yaml", (
            f"--params-physics mismatch: {args.params_physics}"
        )
    finally:
        sys.argv = original_argv


def test_parse_args_merge_fixed_joints():
    """parse_args() supports --merge-fixed-joints flag."""
    original_argv = sys.argv
    try:
        sys.argv = [
            "import_urdf_to_isaac.py",
            "--input", "test.urdf",
            "--merge-fixed-joints",
        ]
        args = parse_args()
        assert args.merge_fixed_joints is True, (
            f"Expected --merge-fixed-joints True, got: {args.merge_fixed_joints}"
        )
    finally:
        sys.argv = original_argv


def test_parse_args_merge_fixed_joints_default():
    """parse_args() defaults --merge-fixed-joints to False."""
    original_argv = sys.argv
    try:
        sys.argv = [
            "import_urdf_to_isaac.py",
            "--input", "test.urdf",
        ]
        args = parse_args()
        assert args.merge_fixed_joints is False, (
            f"Expected --merge-fixed-joints default False, got: {args.merge_fixed_joints}"
        )
    finally:
        sys.argv = original_argv


# ===========================================================================
# Test 2: load_params() — valid YAML, missing file, malformed YAML
# ===========================================================================

def test_load_params_valid_yaml():
    """load_params() correctly loads a valid YAML file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as fh:
        yaml.dump({"key": {"value": 42, "unit": "m"}}, fh)
        tmp_path = fh.name

    try:
        params = load_params(tmp_path)
        assert isinstance(params, dict), "load_params should return a dict"
        assert "key" in params, "Expected 'key' in params"
        assert params["key"]["value"] == 42, (
            f"Expected value 42, got: {params['key']['value']}"
        )
    finally:
        os.unlink(tmp_path)


def test_load_params_missing_file():
    """load_params() raises FileNotFoundError for a non-existent file."""
    try:
        load_params("/nonexistent/path/to/file.yaml")
        raise AssertionError("Expected FileNotFoundError was not raised")
    except FileNotFoundError:
        pass  # expected


def test_load_params_real_control_yaml():
    """load_params() loads the real params/control.yaml and returns expected keys."""
    params = load_params(CONTROL_YAML)
    assert isinstance(params, dict), "load_params should return a dict"

    # Verify key top-level sections exist
    required_keys = ["drive", "joint_limits", "joint_names", "control_frequency"]
    for key in required_keys:
        assert key in params, f"Missing required key '{key}' in control.yaml"


def test_load_params_real_physics_yaml():
    """load_params() loads the real params/physics.yaml and returns expected keys."""
    params = load_params(PHYSICS_YAML)
    assert isinstance(params, dict), "load_params should return a dict"

    required_keys = ["simulation", "friction", "damping", "mass", "contact"]
    for key in required_keys:
        assert key in params, f"Missing required key '{key}' in physics.yaml"


def test_load_params_empty_yaml():
    """load_params() handles a YAML file that parses to None gracefully."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as fh:
        fh.write("# empty YAML file\n")
        tmp_path = fh.name

    try:
        # yaml.safe_load returns None for empty/comment-only files
        params = load_params(tmp_path)
        # The function returns whatever yaml.safe_load returns
        assert params is None, (
            f"Expected None for empty YAML, got: {type(params)}"
        )
    finally:
        os.unlink(tmp_path)


def test_load_params_nested_structure():
    """load_params() correctly preserves nested YAML structure."""
    nested_data = {
        "level1": {
            "level2": {
                "value": 3.14,
                "unit": "rad",
            }
        }
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as fh:
        yaml.dump(nested_data, fh)
        tmp_path = fh.name

    try:
        params = load_params(tmp_path)
        assert params["level1"]["level2"]["value"] == 3.14, (
            "Nested value not preserved correctly"
        )
        assert params["level1"]["level2"]["unit"] == "rad", (
            "Nested unit not preserved correctly"
        )
    finally:
        os.unlink(tmp_path)


# ===========================================================================
# Test 3: get_param_value() — nested key extraction
# ===========================================================================

def test_get_param_value_drive_stiffness():
    """get_param_value() extracts drive.stiffness = 40.0 from control.yaml."""
    params = load_params(CONTROL_YAML)
    stiffness = get_param_value(params, "drive", "stiffness")
    assert stiffness == 40.0, (
        f"Expected drive.stiffness=40.0, got: {stiffness}"
    )


def test_get_param_value_drive_damping():
    """get_param_value() extracts drive.damping = 4.0 from control.yaml."""
    params = load_params(CONTROL_YAML)
    damping = get_param_value(params, "drive", "damping")
    assert damping == 4.0, (
        f"Expected drive.damping=4.0, got: {damping}"
    )


def test_get_param_value_drive_max_effort():
    """get_param_value() extracts drive.max_effort = 5.0 from control.yaml."""
    params = load_params(CONTROL_YAML)
    max_effort = get_param_value(params, "drive", "max_effort")
    assert max_effort == 5.0, (
        f"Expected drive.max_effort=5.0, got: {max_effort}"
    )


def test_get_param_value_joint_limits_position_min():
    """get_param_value() extracts joint_limits.position_min as a 6-element list."""
    params = load_params(CONTROL_YAML)
    position_min = get_param_value(params, "joint_limits", "position_min")
    assert isinstance(position_min, list), (
        f"Expected list, got: {type(position_min)}"
    )
    assert len(position_min) == EXPECTED_NUM_JOINTS, (
        f"Expected {EXPECTED_NUM_JOINTS} values, got: {len(position_min)}"
    )
    # Verify first value matches control.yaml
    assert abs(position_min[0] - (-1.9199)) < 1e-4, (
        f"Expected position_min[0]=-1.9199, got: {position_min[0]}"
    )


def test_get_param_value_joint_limits_position_max():
    """get_param_value() extracts joint_limits.position_max as a 6-element list."""
    params = load_params(CONTROL_YAML)
    position_max = get_param_value(params, "joint_limits", "position_max")
    assert isinstance(position_max, list), (
        f"Expected list, got: {type(position_max)}"
    )
    assert len(position_max) == EXPECTED_NUM_JOINTS, (
        f"Expected {EXPECTED_NUM_JOINTS} values, got: {len(position_max)}"
    )
    # Verify first value matches control.yaml
    assert abs(position_max[0] - 1.9199) < 1e-4, (
        f"Expected position_max[0]=1.9199, got: {position_max[0]}"
    )


def test_get_param_value_joint_limits_velocity_max():
    """get_param_value() extracts joint_limits.velocity_max as a 6-element list."""
    params = load_params(CONTROL_YAML)
    velocity_max = get_param_value(params, "joint_limits", "velocity_max")
    assert isinstance(velocity_max, list), (
        f"Expected list, got: {type(velocity_max)}"
    )
    assert len(velocity_max) == EXPECTED_NUM_JOINTS, (
        f"Expected {EXPECTED_NUM_JOINTS} values, got: {len(velocity_max)}"
    )
    # All joints have 6.28 rad/s max velocity
    for i, v in enumerate(velocity_max):
        assert abs(v - 6.28) < 1e-4, (
            f"Expected velocity_max[{i}]=6.28, got: {v}"
        )


def test_get_param_value_joint_names():
    """get_param_value() extracts joint_names matching EXPECTED_JOINT_NAMES."""
    params = load_params(CONTROL_YAML)
    joint_names = get_param_value(params, "joint_names")
    assert isinstance(joint_names, list), (
        f"Expected list, got: {type(joint_names)}"
    )
    assert len(joint_names) == EXPECTED_NUM_JOINTS, (
        f"Expected {EXPECTED_NUM_JOINTS} names, got: {len(joint_names)}"
    )
    for name in EXPECTED_JOINT_NAMES:
        assert name in joint_names, (
            f"Missing joint name '{name}' in: {joint_names}"
        )


def test_get_param_value_physics_gravity():
    """get_param_value() extracts simulation.gravity = [0,0,-9.81] from physics.yaml."""
    params = load_params(PHYSICS_YAML)
    gravity = get_param_value(params, "simulation", "gravity")
    assert isinstance(gravity, list), f"Expected list, got: {type(gravity)}"
    assert len(gravity) == 3, f"Expected 3 components, got: {len(gravity)}"
    assert gravity[0] == 0.0, f"gravity[0] should be 0.0, got: {gravity[0]}"
    assert gravity[1] == 0.0, f"gravity[1] should be 0.0, got: {gravity[1]}"
    assert gravity[2] == -9.81, f"gravity[2] should be -9.81, got: {gravity[2]}"


def test_get_param_value_physics_timestep():
    """get_param_value() extracts simulation.timestep = 0.005 from physics.yaml."""
    params = load_params(PHYSICS_YAML)
    timestep = get_param_value(params, "simulation", "timestep")
    assert timestep == 0.005, f"Expected 0.005, got: {timestep}"


def test_get_param_value_physics_total_mass():
    """get_param_value() extracts mass.total_mass = 0.632 from physics.yaml."""
    params = load_params(PHYSICS_YAML)
    total_mass = get_param_value(params, "mass", "total_mass")
    assert abs(total_mass - 0.632) < 1e-4, (
        f"Expected 0.632, got: {total_mass}"
    )


def test_get_param_value_missing_key():
    """get_param_value() raises KeyError for a missing key in the hierarchy."""
    params = load_params(CONTROL_YAML)
    try:
        get_param_value(params, "nonexistent_key")
        raise AssertionError("Expected KeyError was not raised")
    except KeyError:
        pass  # expected


def test_get_param_value_missing_nested_key():
    """get_param_value() raises KeyError for a missing nested key."""
    params = load_params(CONTROL_YAML)
    try:
        get_param_value(params, "drive", "nonexistent_subkey")
        raise AssertionError("Expected KeyError was not raised")
    except KeyError:
        pass  # expected


def test_get_param_value_returns_raw_when_no_value_key():
    """get_param_value() returns the raw node when 'value' key is absent."""
    # per_joint_overrides.shoulder_pan has no 'value' key — it's a flat dict
    params = load_params(CONTROL_YAML)
    overrides = get_param_value(params, "per_joint_overrides", "shoulder_pan")
    assert isinstance(overrides, dict), (
        f"Expected dict, got: {type(overrides)}"
    )
    assert "stiffness" in overrides, "Missing 'stiffness' in overrides"
    assert overrides["stiffness"] == 40.0, (
        f"Expected stiffness 40.0, got: {overrides['stiffness']}"
    )


def test_get_param_value_with_synthetic_data():
    """get_param_value() works correctly with synthetic nested YAML data."""
    params = {
        "level1": {
            "level2": {
                "value": 99.9,
                "unit": "Hz",
            }
        }
    }
    result = get_param_value(params, "level1", "level2")
    assert result == 99.9, f"Expected 99.9, got: {result}"


def test_get_param_value_single_key():
    """get_param_value() extracts value with a single key."""
    params = load_params(CONTROL_YAML)
    control_freq = get_param_value(params, "control_frequency")
    assert control_freq == 20.0, (
        f"Expected control_frequency=20.0, got: {control_freq}"
    )


# ===========================================================================
# Test 4: _setup_logging() — handler configuration, log file creation
# ===========================================================================

def _clear_logger():
    """Helper to clear the urdf_to_usd logger handlers between tests."""
    logger = logging.getLogger("urdf_to_usd")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def test_setup_logging_creates_file_handler():
    """_setup_logging() creates a FileHandler writing to the specified log file."""
    _clear_logger()
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test_fh.log")
        logger = _setup_logging(log_file=log_file)

        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1, (
            f"Expected 1 FileHandler, got: {len(file_handlers)}"
        )
        assert os.path.exists(log_file), "Log file was not created"
        _clear_logger()


def test_setup_logging_creates_stream_handler():
    """_setup_logging() creates a StreamHandler writing to stderr."""
    _clear_logger()
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test_sh.log")
        logger = _setup_logging(log_file=log_file)

        stream_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 1, (
            f"Expected 1 StreamHandler, got: {len(stream_handlers)}"
        )
        assert stream_handlers[0].stream is sys.stderr, (
            "StreamHandler should write to stderr"
        )
        _clear_logger()


def test_setup_logging_handler_count():
    """_setup_logging() creates exactly 2 handlers (FileHandler + StreamHandler)."""
    _clear_logger()
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test_count.log")
        logger = _setup_logging(log_file=log_file)

        assert len(logger.handlers) == 2, (
            f"Expected 2 handlers, got: {len(logger.handlers)}"
        )
        _clear_logger()


def test_setup_logging_logger_level():
    """_setup_logging() sets logger level to DEBUG."""
    _clear_logger()
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test_level.log")
        logger = _setup_logging(log_file=log_file)

        assert logger.level == logging.DEBUG, (
            f"Expected DEBUG level, got: {logger.level}"
        )
        _clear_logger()


def test_setup_logging_logger_name():
    """_setup_logging() uses 'urdf_to_usd' as the logger name."""
    _clear_logger()
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test_name.log")
        logger = _setup_logging(log_file=log_file)

        assert logger.name == "urdf_to_usd", (
            f"Expected logger name 'urdf_to_usd', got: {logger.name}"
        )
        _clear_logger()


def test_setup_logging_writes_to_file():
    """_setup_logging() actually writes log messages to the file."""
    _clear_logger()
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test_write.log")
        logger = _setup_logging(log_file=log_file)

        test_msg = "unit_test_log_message_12345"
        logger.info(test_msg)

        # Flush all handlers
        for handler in logger.handlers:
            handler.flush()

        with open(log_file, "r", encoding="utf-8") as fh:
            content = fh.read()

        assert test_msg in content, (
            f"Test message not found in log file. Content: {content[:200]}"
        )
        _clear_logger()


def test_setup_logging_no_duplicate_handlers():
    """Repeated _setup_logging() calls do not create duplicate handlers."""
    _clear_logger()
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test_dup.log")
        logger1 = _setup_logging(log_file=log_file)
        count1 = len(logger1.handlers)

        logger2 = _setup_logging(log_file=log_file)
        count2 = len(logger2.handlers)

        assert count1 == count2, (
            f"Handler count changed: {count1} -> {count2}"
        )
        assert logger1 is logger2, (
            "Repeated calls should return the same logger"
        )
        _clear_logger()


def test_setup_logging_creates_parent_directories():
    """_setup_logging() creates parent directories for the log file."""
    _clear_logger()
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_log = os.path.join(tmpdir, "sub", "dir", "test.log")
        logger = _setup_logging(log_file=nested_log)

        assert os.path.exists(nested_log), (
            f"Log file not created at nested path: {nested_log}"
        )
        _clear_logger()


def test_setup_logging_file_handler_level():
    """FileHandler level is set to DEBUG."""
    _clear_logger()
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test_fh_level.log")
        logger = _setup_logging(log_file=log_file)

        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert file_handlers[0].level == logging.DEBUG, (
            f"Expected FileHandler level DEBUG, got: {file_handlers[0].level}"
        )
        _clear_logger()


def test_setup_logging_stream_handler_level():
    """StreamHandler level is set to INFO."""
    _clear_logger()
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test_sh_level.log")
        logger = _setup_logging(log_file=log_file)

        stream_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert stream_handlers[0].level == logging.INFO, (
            f"Expected StreamHandler level INFO, got: {stream_handlers[0].level}"
        )
        _clear_logger()


# ===========================================================================
# Test 5: Output path resolution logic
# ===========================================================================
# The output path resolution logic from main():
#   output_path = Path(args.output)
#   if output_path.is_dir() or str(args.output).endswith("/"):
#       output_path.mkdir(parents=True, exist_ok=True)
#       output_path = output_path / urdf_path.with_suffix(".usd").name
#   else:
#       output_path.parent.mkdir(parents=True, exist_ok=True)

def test_output_path_directory_auto_names():
    """When output is a directory, the USD filename is derived from URDF stem."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "usd_output"
        output_dir.mkdir()

        urdf_path = Path("robot_description/urdf/so101_new_calib.urdf")

        # Simulate the resolution logic from main()
        output_path = output_dir
        if output_path.is_dir() or str(output_dir).endswith("/"):
            output_path = output_path / urdf_path.with_suffix(".usd").name

        expected = output_dir / "so101_new_calib.usd"
        assert output_path == expected, (
            f"Expected {expected}, got: {output_path}"
        )


def test_output_path_trailing_slash_auto_names():
    """When output ends with '/', the USD filename is derived from URDF stem."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_str = os.path.join(tmpdir, "output") + "/"
        urdf_path = Path("robot_description/urdf/my_robot.urdf")

        output_path = Path(output_str)
        if output_path.is_dir() or output_str.endswith("/"):
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / urdf_path.with_suffix(".usd").name

        expected_name = "my_robot.usd"
        assert output_path.name == expected_name, (
            f"Expected filename '{expected_name}', got: '{output_path.name}'"
        )


def test_output_path_file_preserves_name():
    """When output is a file path (not dir), the name is preserved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_str = os.path.join(tmpdir, "custom_name.usd")
        urdf_path = Path("robot_description/urdf/so101.urdf")

        output_path = Path(output_str)
        # File path — does not end with "/" and is_dir() is False
        if output_path.is_dir() or output_str.endswith("/"):
            output_path = output_path / urdf_path.with_suffix(".usd").name
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        assert output_path.name == "custom_name.usd", (
            f"Expected 'custom_name.usd', got: '{output_path.name}'"
        )


def test_output_path_file_creates_parent():
    """File output path creates parent directories if needed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_output = os.path.join(tmpdir, "nested", "dir", "robot.usd")
        urdf_path = Path("test.urdf")

        output_path = Path(nested_output)
        if output_path.is_dir() or nested_output.endswith("/"):
            output_path = output_path / urdf_path.with_suffix(".usd").name
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        assert output_path.parent.exists(), (
            f"Parent directory was not created: {output_path.parent}"
        )
        assert output_path.name == "robot.usd", (
            f"Expected 'robot.usd', got: '{output_path.name}'"
        )


def test_output_path_default_assets_usd():
    """Default output 'assets/usd/' triggers auto-naming from URDF stem."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate default output path ending with "/"
        output_str = os.path.join(tmpdir, "assets", "usd") + "/"
        urdf_path = Path("robot_description/urdf/so101_new_calib.urdf")

        output_path = Path(output_str)
        if output_path.is_dir() or output_str.endswith("/"):
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / urdf_path.with_suffix(".usd").name

        assert output_path.name == "so101_new_calib.usd", (
            f"Expected 'so101_new_calib.usd', got: '{output_path.name}'"
        )
        assert "assets" in str(output_path) and "usd" in str(output_path), (
            f"Path should contain 'assets/usd': {output_path}"
        )


def test_output_path_urdf_suffix_replaced():
    """URDF stem's .urdf suffix is replaced with .usd in auto-naming."""
    urdf_path = Path("path/to/robot_v2.urdf")
    auto_name = urdf_path.with_suffix(".usd").name
    assert auto_name == "robot_v2.usd", (
        f"Expected 'robot_v2.usd', got: '{auto_name}'"
    )


# ===========================================================================
# Main entry point
# ===========================================================================
def main():
    runner = UnitTestRunner()

    # --- Test 1: parse_args() ---
    runner.run_test("parse_args --input/--output", test_parse_args_input_output)
    runner.run_test("parse_args default --output", test_parse_args_default_output)
    runner.run_test("parse_args --headless default", test_parse_args_headless_default_true)
    runner.run_test("parse_args --no-headless", test_parse_args_no_headless)
    runner.run_test("parse_args --skip-verify", test_parse_args_skip_verify)
    runner.run_test("parse_args --skip-verify default", test_parse_args_skip_verify_default_false)
    runner.run_test("parse_args --fix-base default", test_parse_args_fix_base_default)
    runner.run_test("parse_args --no-fix-base", test_parse_args_no_fix_base)
    runner.run_test("parse_args --params-control default", test_parse_args_params_control_default)
    runner.run_test("parse_args --params-physics default", test_parse_args_params_physics_default)
    runner.run_test("parse_args custom params paths", test_parse_args_custom_params_paths)
    runner.run_test("parse_args --merge-fixed-joints", test_parse_args_merge_fixed_joints)
    runner.run_test("parse_args --merge-fixed-joints default", test_parse_args_merge_fixed_joints_default)

    # --- Test 2: load_params() ---
    runner.run_test("load_params valid YAML", test_load_params_valid_yaml)
    runner.run_test("load_params missing file", test_load_params_missing_file)
    runner.run_test("load_params real control.yaml", test_load_params_real_control_yaml)
    runner.run_test("load_params real physics.yaml", test_load_params_real_physics_yaml)
    runner.run_test("load_params empty YAML", test_load_params_empty_yaml)
    runner.run_test("load_params nested structure", test_load_params_nested_structure)

    # --- Test 3: get_param_value() ---
    runner.run_test("get_param_value drive.stiffness", test_get_param_value_drive_stiffness)
    runner.run_test("get_param_value drive.damping", test_get_param_value_drive_damping)
    runner.run_test("get_param_value drive.max_effort", test_get_param_value_drive_max_effort)
    runner.run_test("get_param_value joint_limits.position_min", test_get_param_value_joint_limits_position_min)
    runner.run_test("get_param_value joint_limits.position_max", test_get_param_value_joint_limits_position_max)
    runner.run_test("get_param_value joint_limits.velocity_max", test_get_param_value_joint_limits_velocity_max)
    runner.run_test("get_param_value joint_names", test_get_param_value_joint_names)
    runner.run_test("get_param_value physics gravity", test_get_param_value_physics_gravity)
    runner.run_test("get_param_value physics timestep", test_get_param_value_physics_timestep)
    runner.run_test("get_param_value physics total_mass", test_get_param_value_physics_total_mass)
    runner.run_test("get_param_value missing key", test_get_param_value_missing_key)
    runner.run_test("get_param_value missing nested key", test_get_param_value_missing_nested_key)
    runner.run_test("get_param_value raw when no value key", test_get_param_value_returns_raw_when_no_value_key)
    runner.run_test("get_param_value synthetic data", test_get_param_value_with_synthetic_data)
    runner.run_test("get_param_value single key", test_get_param_value_single_key)

    # --- Test 4: _setup_logging() ---
    runner.run_test("_setup_logging FileHandler", test_setup_logging_creates_file_handler)
    runner.run_test("_setup_logging StreamHandler", test_setup_logging_creates_stream_handler)
    runner.run_test("_setup_logging handler count", test_setup_logging_handler_count)
    runner.run_test("_setup_logging logger level", test_setup_logging_logger_level)
    runner.run_test("_setup_logging logger name", test_setup_logging_logger_name)
    runner.run_test("_setup_logging writes to file", test_setup_logging_writes_to_file)
    runner.run_test("_setup_logging no duplicates", test_setup_logging_no_duplicate_handlers)
    runner.run_test("_setup_logging creates parent dirs", test_setup_logging_creates_parent_directories)
    runner.run_test("_setup_logging FileHandler level", test_setup_logging_file_handler_level)
    runner.run_test("_setup_logging StreamHandler level", test_setup_logging_stream_handler_level)

    # --- Test 5: Output path resolution ---
    runner.run_test("output path dir auto-names", test_output_path_directory_auto_names)
    runner.run_test("output path trailing slash", test_output_path_trailing_slash_auto_names)
    runner.run_test("output path file preserves name", test_output_path_file_preserves_name)
    runner.run_test("output path file creates parent", test_output_path_file_creates_parent)
    runner.run_test("output path default assets/usd/", test_output_path_default_assets_usd)
    runner.run_test("output path URDF suffix replaced", test_output_path_urdf_suffix_replaced)

    # Summary
    runner.print_summary()

    if runner.failed == 0:
        sys.stderr.write("All unit tests passed\n")
        return 0
    else:
        sys.stderr.write(
            f"FAILED: {runner.failed} of {runner.passed + runner.failed} tests failed\n"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
