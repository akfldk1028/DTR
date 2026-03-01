#!/usr/bin/env python3
"""test_vla_units.py — Unit tests for VLA inference interface

Tests the VLA (Vision-Language-Action) inference module through runtime checks:
  1. DummyVLA.predict() returns np.ndarray shape (6,)
  2. DummyVLA.predict() returns all zeros
  3. DummyVLA accepts instruction str + image ndarray (480,640,3) + state ndarray (6,)
  4. VLAInference is abstract — raises NotImplementedError on predict()
  5. eval_in_sim module can be imported without Isaac Sim (graceful degradation)
  6. Inference I/O contract matches spec (instruction + image + state → 6-DOF action)

These tests import the VLA module and exercise the DummyVLA class.
Heavy dependencies (Isaac Sim, LeRobot, PyTorch) are NOT required.
"""

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — add training/vla/ to sys.path for importing VLA classes
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VLA_DIR = os.path.join(PROJECT_ROOT, "training", "vla")
sys.path.insert(0, VLA_DIR)


# ---------------------------------------------------------------------------
# Test runner (same pattern as test_import_urdf_units.py)
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
        sys.stderr.write("\n=== VLA Unit Test Results ===\n")
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
# Lazy import helper — import VLA classes once, cache the result
# ---------------------------------------------------------------------------
_vla_imports = {}


def _import_vla():
    """Import VLAInference and DummyVLA from inference.py.

    Returns a dict with keys 'VLAInference', 'DummyVLA', or raises ImportError.
    """
    if _vla_imports:
        return _vla_imports

    try:
        from inference import VLAInference, DummyVLA
        _vla_imports["VLAInference"] = VLAInference
        _vla_imports["DummyVLA"] = DummyVLA
        return _vla_imports
    except ImportError as exc:
        raise ImportError(
            f"Cannot import VLA classes from inference.py: {exc}. "
            f"Ensure training/vla/inference.py exists and defines "
            f"VLAInference and DummyVLA."
        ) from exc


# ---------------------------------------------------------------------------
# Standard test inputs matching the VLA inference contract
# ---------------------------------------------------------------------------
SPEC_IMAGE_SHAPE = (480, 640, 3)
SPEC_STATE_SHAPE = (6,)
SPEC_ACTION_SHAPE = (6,)
SPEC_INSTRUCTION = "pick up the orange from the table"


def _make_test_image():
    """Create a synthetic camera image matching the spec (480×640×3, uint8)."""
    return np.random.randint(0, 256, size=SPEC_IMAGE_SHAPE, dtype=np.uint8)


def _make_test_state():
    """Create a synthetic joint state matching the spec (6,), float32."""
    return np.random.randn(*SPEC_STATE_SHAPE).astype(np.float32)


# ===========================================================================
# Test 1: DummyVLA.predict() returns np.ndarray shape (6,)
# ===========================================================================

def test_dummyvla_predict_returns_ndarray():
    """DummyVLA.predict() returns a numpy ndarray."""
    modules = _import_vla()
    DummyVLA = modules["DummyVLA"]

    model = DummyVLA()
    image = _make_test_image()
    state = _make_test_state()

    result = model.predict(SPEC_INSTRUCTION, image, state)

    assert isinstance(result, np.ndarray), (
        f"DummyVLA.predict() must return np.ndarray, got: {type(result)}"
    )


def test_dummyvla_predict_shape_6():
    """DummyVLA.predict() returns array with shape (6,)."""
    modules = _import_vla()
    DummyVLA = modules["DummyVLA"]

    model = DummyVLA()
    image = _make_test_image()
    state = _make_test_state()

    result = model.predict(SPEC_INSTRUCTION, image, state)

    assert result.shape == SPEC_ACTION_SHAPE, (
        f"DummyVLA.predict() must return shape {SPEC_ACTION_SHAPE}, "
        f"got: {result.shape}"
    )


def test_dummyvla_predict_dtype_float():
    """DummyVLA.predict() returns float32 array (joint position targets)."""
    modules = _import_vla()
    DummyVLA = modules["DummyVLA"]

    model = DummyVLA()
    image = _make_test_image()
    state = _make_test_state()

    result = model.predict(SPEC_INSTRUCTION, image, state)

    assert result.dtype == np.float32, (
        f"DummyVLA.predict() must return float32 dtype, got: {result.dtype}"
    )


# ===========================================================================
# Test 2: DummyVLA.predict() returns all zeros
# ===========================================================================

def test_dummyvla_predict_returns_zeros():
    """DummyVLA.predict() returns all-zero action (zero-action baseline)."""
    modules = _import_vla()
    DummyVLA = modules["DummyVLA"]

    model = DummyVLA()
    image = _make_test_image()
    state = _make_test_state()

    result = model.predict(SPEC_INSTRUCTION, image, state)

    assert np.all(result == 0.0), (
        f"DummyVLA.predict() must return all zeros, got: {result}"
    )


def test_dummyvla_predict_zeros_allclose():
    """DummyVLA.predict() output is numerically close to np.zeros(6)."""
    modules = _import_vla()
    DummyVLA = modules["DummyVLA"]

    model = DummyVLA()
    image = _make_test_image()
    state = _make_test_state()

    result = model.predict(SPEC_INSTRUCTION, image, state)
    expected = np.zeros(6, dtype=np.float32)

    assert np.allclose(result, expected, atol=1e-7), (
        f"DummyVLA.predict() output not close to zeros: {result}"
    )


# ===========================================================================
# Test 3: DummyVLA accepts spec-compliant inputs
# ===========================================================================

def test_dummyvla_accepts_str_instruction():
    """DummyVLA.predict() accepts a string instruction as first argument."""
    modules = _import_vla()
    DummyVLA = modules["DummyVLA"]

    model = DummyVLA()
    image = _make_test_image()
    state = _make_test_state()

    # Various instruction strings
    instructions = [
        "pick up the orange",
        "move to the left",
        "",  # empty instruction should also work
        "한국어 지시도 동작해야 함",  # Korean instruction
    ]

    for instruction in instructions:
        result = model.predict(instruction, image, state)
        assert isinstance(result, np.ndarray), (
            f"DummyVLA.predict() failed with instruction: {instruction!r}"
        )


def test_dummyvla_accepts_image_480x640x3():
    """DummyVLA.predict() accepts an image ndarray of shape (480,640,3)."""
    modules = _import_vla()
    DummyVLA = modules["DummyVLA"]

    model = DummyVLA()
    image = np.zeros(SPEC_IMAGE_SHAPE, dtype=np.uint8)
    state = _make_test_state()

    result = model.predict(SPEC_INSTRUCTION, image, state)

    assert result.shape == SPEC_ACTION_SHAPE, (
        f"DummyVLA.predict() failed with image shape {SPEC_IMAGE_SHAPE}: "
        f"got result shape {result.shape}"
    )


def test_dummyvla_accepts_state_6():
    """DummyVLA.predict() accepts a state ndarray of shape (6,)."""
    modules = _import_vla()
    DummyVLA = modules["DummyVLA"]

    model = DummyVLA()
    image = _make_test_image()
    state = np.zeros(SPEC_STATE_SHAPE, dtype=np.float32)

    result = model.predict(SPEC_INSTRUCTION, image, state)

    assert result.shape == SPEC_ACTION_SHAPE, (
        f"DummyVLA.predict() failed with state shape {SPEC_STATE_SHAPE}: "
        f"got result shape {result.shape}"
    )


def test_dummyvla_accepts_all_spec_inputs():
    """DummyVLA.predict() accepts all spec inputs: str + (480,640,3) + (6,)."""
    modules = _import_vla()
    DummyVLA = modules["DummyVLA"]

    model = DummyVLA()
    instruction = "pick up the orange from the table"
    image = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
    state = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=np.float32)

    result = model.predict(instruction, image, state)

    assert isinstance(result, np.ndarray), (
        f"Expected np.ndarray, got: {type(result)}"
    )
    assert result.shape == (6,), (
        f"Expected shape (6,), got: {result.shape}"
    )
    assert result.dtype == np.float32, (
        f"Expected float32, got: {result.dtype}"
    )


# ===========================================================================
# Test 4: VLAInference is abstract — raises NotImplementedError
# ===========================================================================

def test_vlainference_predict_raises_not_implemented():
    """VLAInference.predict() raises NotImplementedError."""
    modules = _import_vla()
    VLAInference = modules["VLAInference"]

    model = VLAInference()
    image = _make_test_image()
    state = _make_test_state()

    try:
        model.predict(SPEC_INSTRUCTION, image, state)
        raise AssertionError(
            "VLAInference.predict() should raise NotImplementedError "
            "but returned without error"
        )
    except NotImplementedError:
        pass  # expected


def test_vlainference_is_base_class():
    """VLAInference can be instantiated but predict() is abstract."""
    modules = _import_vla()
    VLAInference = modules["VLAInference"]
    DummyVLA = modules["DummyVLA"]

    # VLAInference should be instantiable (not using abc.ABC enforcement)
    # or if using ABC, it should raise TypeError
    try:
        base = VLAInference()
        # If instantiation succeeds, predict() must raise NotImplementedError
        try:
            base.predict(SPEC_INSTRUCTION, _make_test_image(), _make_test_state())
            raise AssertionError(
                "VLAInference.predict() must raise NotImplementedError"
            )
        except NotImplementedError:
            pass  # expected
    except TypeError:
        # If using abc.ABC with @abstractmethod, instantiation itself fails
        pass  # also acceptable


def test_dummyvla_is_subclass_of_vlainference():
    """DummyVLA is a subclass of VLAInference."""
    modules = _import_vla()
    VLAInference = modules["VLAInference"]
    DummyVLA = modules["DummyVLA"]

    assert issubclass(DummyVLA, VLAInference), (
        f"DummyVLA must be a subclass of VLAInference. "
        f"MRO: {DummyVLA.__mro__}"
    )


def test_dummyvla_isinstance_of_vlainference():
    """DummyVLA instance is an instance of VLAInference."""
    modules = _import_vla()
    VLAInference = modules["VLAInference"]
    DummyVLA = modules["DummyVLA"]

    model = DummyVLA()
    assert isinstance(model, VLAInference), (
        f"DummyVLA() must be an instance of VLAInference"
    )


# ===========================================================================
# Test 5: eval_in_sim module can be imported without Isaac Sim
# ===========================================================================

def test_eval_in_sim_importable():
    """eval_in_sim module can be imported without Isaac Sim installed.

    The module should use try/except guards on heavy dependencies (omni, isaacsim)
    to allow importing for testing/linting without the full simulator stack.
    """
    eval_in_sim_path = os.path.join(VLA_DIR, "eval_in_sim.py")
    assert os.path.isfile(eval_in_sim_path), (
        f"eval_in_sim.py not found at {eval_in_sim_path}"
    )

    try:
        import importlib
        spec = importlib.util.spec_from_file_location("eval_in_sim", eval_in_sim_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise AssertionError(
            f"eval_in_sim.py should be importable without Isaac Sim. "
            f"Heavy imports must be guarded with try/except. Error: {exc}"
        )


def test_eval_in_sim_has_module_attributes():
    """eval_in_sim module has expected attributes after import.

    Even without Isaac Sim, the module should define its structure
    (function signatures, constants, etc.) at the module level.
    """
    eval_in_sim_path = os.path.join(VLA_DIR, "eval_in_sim.py")
    assert os.path.isfile(eval_in_sim_path), (
        f"eval_in_sim.py not found at {eval_in_sim_path}"
    )

    import importlib
    spec = importlib.util.spec_from_file_location("eval_in_sim", eval_in_sim_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Module should have a docstring
    assert module.__doc__ is not None, (
        "eval_in_sim module should have a module-level docstring"
    )


# ===========================================================================
# Test 6: Inference I/O contract matches spec
# ===========================================================================

def test_io_contract_instruction_to_action():
    """End-to-end inference I/O: instruction + image + state → 6-DOF action.

    Verifies the complete inference contract:
      Input:  instruction (str) + image (480×640×3) + state (6,)
      Output: action (6,) — 6-DOF joint position targets
    """
    modules = _import_vla()
    DummyVLA = modules["DummyVLA"]

    model = DummyVLA()

    # Spec-compliant inputs
    instruction = "pick up the orange from the table"
    image = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Execute inference
    action = model.predict(instruction, image, state)

    # Verify output contract
    assert isinstance(action, np.ndarray), (
        f"Action must be np.ndarray, got: {type(action)}"
    )
    assert action.shape == (6,), (
        f"Action must have shape (6,) for 6-DOF, got: {action.shape}"
    )
    assert np.issubdtype(action.dtype, np.floating), (
        f"Action must be floating-point dtype, got: {action.dtype}"
    )


def test_io_contract_action_is_finite():
    """Inference output contains only finite values (no NaN or Inf)."""
    modules = _import_vla()
    DummyVLA = modules["DummyVLA"]

    model = DummyVLA()
    action = model.predict(
        SPEC_INSTRUCTION,
        _make_test_image(),
        _make_test_state(),
    )

    assert np.all(np.isfinite(action)), (
        f"Action contains non-finite values: {action}"
    )


def test_io_contract_multiple_predictions_consistent():
    """DummyVLA.predict() returns consistent results across calls.

    For the dummy baseline, every prediction should be identical (all zeros).
    """
    modules = _import_vla()
    DummyVLA = modules["DummyVLA"]

    model = DummyVLA()
    image = _make_test_image()
    state = _make_test_state()

    action1 = model.predict(SPEC_INSTRUCTION, image, state)
    action2 = model.predict(SPEC_INSTRUCTION, image, state)
    action3 = model.predict("different instruction", image, state)

    assert np.array_equal(action1, action2), (
        f"DummyVLA predictions not consistent: {action1} vs {action2}"
    )
    assert np.array_equal(action1, action3), (
        f"DummyVLA predictions should be instruction-independent: "
        f"{action1} vs {action3}"
    )


def test_io_contract_6dof_joint_positions():
    """Output action represents 6-DOF joint position targets.

    Each element corresponds to one joint (6 joints total for SO-101).
    All values must be finite float32 in the action space.
    """
    modules = _import_vla()
    DummyVLA = modules["DummyVLA"]

    model = DummyVLA()
    action = model.predict(
        SPEC_INSTRUCTION,
        _make_test_image(),
        _make_test_state(),
    )

    # 6-DOF: one value per joint
    assert len(action) == 6, (
        f"Expected 6 joint position targets, got: {len(action)}"
    )

    # Each element must be individually accessible as a float
    for i in range(6):
        val = float(action[i])
        assert np.isfinite(val), (
            f"Joint {i} position target is not finite: {val}"
        )


# ===========================================================================
# Main entry point
# ===========================================================================
def main():
    runner = UnitTestRunner()

    # --- Test 1: DummyVLA.predict() returns np.ndarray shape (6,) ---
    runner.run_test(
        "DummyVLA.predict() returns ndarray",
        test_dummyvla_predict_returns_ndarray,
    )
    runner.run_test(
        "DummyVLA.predict() shape (6,)",
        test_dummyvla_predict_shape_6,
    )
    runner.run_test(
        "DummyVLA.predict() dtype float32",
        test_dummyvla_predict_dtype_float,
    )

    # --- Test 2: DummyVLA.predict() returns all zeros ---
    runner.run_test(
        "DummyVLA.predict() returns zeros",
        test_dummyvla_predict_returns_zeros,
    )
    runner.run_test(
        "DummyVLA.predict() zeros allclose",
        test_dummyvla_predict_zeros_allclose,
    )

    # --- Test 3: DummyVLA accepts spec-compliant inputs ---
    runner.run_test(
        "DummyVLA accepts str instruction",
        test_dummyvla_accepts_str_instruction,
    )
    runner.run_test(
        "DummyVLA accepts image (480,640,3)",
        test_dummyvla_accepts_image_480x640x3,
    )
    runner.run_test(
        "DummyVLA accepts state (6,)",
        test_dummyvla_accepts_state_6,
    )
    runner.run_test(
        "DummyVLA accepts all spec inputs",
        test_dummyvla_accepts_all_spec_inputs,
    )

    # --- Test 4: VLAInference abstract / NotImplementedError ---
    runner.run_test(
        "VLAInference.predict() raises NotImplementedError",
        test_vlainference_predict_raises_not_implemented,
    )
    runner.run_test(
        "VLAInference base class behavior",
        test_vlainference_is_base_class,
    )
    runner.run_test(
        "DummyVLA is subclass of VLAInference",
        test_dummyvla_is_subclass_of_vlainference,
    )
    runner.run_test(
        "DummyVLA isinstance of VLAInference",
        test_dummyvla_isinstance_of_vlainference,
    )

    # --- Test 5: eval_in_sim graceful degradation ---
    runner.run_test(
        "eval_in_sim importable without Isaac Sim",
        test_eval_in_sim_importable,
    )
    runner.run_test(
        "eval_in_sim has module attributes",
        test_eval_in_sim_has_module_attributes,
    )

    # --- Test 6: Inference I/O contract ---
    runner.run_test(
        "I/O contract: instruction → 6-DOF action",
        test_io_contract_instruction_to_action,
    )
    runner.run_test(
        "I/O contract: action is finite",
        test_io_contract_action_is_finite,
    )
    runner.run_test(
        "I/O contract: consistent predictions",
        test_io_contract_multiple_predictions_consistent,
    )
    runner.run_test(
        "I/O contract: 6-DOF joint positions",
        test_io_contract_6dof_joint_positions,
    )

    # Summary
    runner.print_summary()

    if runner.failed == 0:
        sys.stderr.write("All VLA unit tests passed\n")
        return 0
    else:
        sys.stderr.write(
            f"FAILED: {runner.failed} of {runner.passed + runner.failed} "
            f"tests failed\n"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
