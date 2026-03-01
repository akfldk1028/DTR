#!/usr/bin/env python3
"""test_vla_static.py — Static verification for Phase 7 VLA extension

Verifies the VLA (Vision-Language-Action) extension through static analysis:
  1. File existence for all expected VLA module files
  2. Python syntax check via py_compile on each .py file
  3. training/vla/README.md contains I/O contract documentation
  4. datasets/README.md contains task instruction field documentation
  5. training/vla/__init__.py exists for proper package structure

These tests do NOT import heavy dependencies (Isaac Sim, LeRobot, PyTorch) —
they test only file existence, syntax, and documentation content.
"""

import os
import py_compile
import sys

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Expected file paths
# ---------------------------------------------------------------------------
VLA_DIR = os.path.join(PROJECT_ROOT, "training", "vla")

EXPECTED_PYTHON_FILES = [
    os.path.join(VLA_DIR, "inference.py"),
    os.path.join(VLA_DIR, "eval_in_sim.py"),
    os.path.join(VLA_DIR, "__init__.py"),
]

VLA_README = os.path.join(VLA_DIR, "README.md")
DATASETS_README = os.path.join(PROJECT_ROOT, "datasets", "README.md")

# Content markers expected in documentation files
VLA_README_MARKERS = [
    # I/O contract keywords
    "instruction",
    "image",
    "action",
    "I/O",
]

DATASETS_README_MARKERS = [
    # task instruction field keywords
    "task",
    "save_episode",
]


# ---------------------------------------------------------------------------
# Test runner (follows test_import_urdf_static.py pattern)
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
        sys.stderr.write("\n=== VLA Static Verification Results ===\n")
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
# Test 1: VLA directory exists
# ---------------------------------------------------------------------------
def test_vla_directory_exists():
    """Verify training/vla/ directory exists."""
    assert os.path.isdir(VLA_DIR), (
        f"VLA directory not found: {VLA_DIR}"
    )


# ---------------------------------------------------------------------------
# Test 2: File existence checks
# ---------------------------------------------------------------------------
def test_inference_py_exists():
    """Verify training/vla/inference.py exists."""
    path = os.path.join(VLA_DIR, "inference.py")
    assert os.path.isfile(path), (
        f"File not found: {path}"
    )


def test_eval_in_sim_py_exists():
    """Verify training/vla/eval_in_sim.py exists."""
    path = os.path.join(VLA_DIR, "eval_in_sim.py")
    assert os.path.isfile(path), (
        f"File not found: {path}"
    )


def test_init_py_exists():
    """Verify training/vla/__init__.py exists."""
    path = os.path.join(VLA_DIR, "__init__.py")
    assert os.path.isfile(path), (
        f"File not found: {path}"
    )


def test_vla_readme_exists():
    """Verify training/vla/README.md exists."""
    assert os.path.isfile(VLA_README), (
        f"VLA README not found: {VLA_README}"
    )


def test_datasets_readme_exists():
    """Verify datasets/README.md exists."""
    assert os.path.isfile(DATASETS_README), (
        f"Datasets README not found: {DATASETS_README}"
    )


# ---------------------------------------------------------------------------
# Test 3: Python syntax checks via py_compile
# ---------------------------------------------------------------------------
def _make_syntax_test(filepath):
    """Factory: create a syntax-check test function for a given file."""
    def test_fn():
        basename = os.path.basename(filepath)
        assert os.path.isfile(filepath), (
            f"Cannot syntax-check missing file: {basename}"
        )
        # py_compile.compile raises py_compile.PyCompileError on syntax error
        py_compile.compile(filepath, doraise=True)
    test_fn.__doc__ = f"Verify {os.path.basename(filepath)} compiles without syntax errors."
    return test_fn


def test_syntax_inference_py():
    """Verify training/vla/inference.py has valid Python syntax."""
    path = os.path.join(VLA_DIR, "inference.py")
    assert os.path.isfile(path), (
        f"Cannot syntax-check missing file: inference.py"
    )
    py_compile.compile(path, doraise=True)


def test_syntax_eval_in_sim_py():
    """Verify training/vla/eval_in_sim.py has valid Python syntax."""
    path = os.path.join(VLA_DIR, "eval_in_sim.py")
    assert os.path.isfile(path), (
        f"Cannot syntax-check missing file: eval_in_sim.py"
    )
    py_compile.compile(path, doraise=True)


def test_syntax_init_py():
    """Verify training/vla/__init__.py has valid Python syntax."""
    path = os.path.join(VLA_DIR, "__init__.py")
    assert os.path.isfile(path), (
        f"Cannot syntax-check missing file: __init__.py"
    )
    py_compile.compile(path, doraise=True)


# ---------------------------------------------------------------------------
# Test 4: VLA README content verification
# ---------------------------------------------------------------------------
def test_vla_readme_io_contract():
    """Verify training/vla/README.md documents the I/O contract."""
    assert os.path.isfile(VLA_README), (
        f"VLA README not found: {VLA_README}"
    )
    with open(VLA_README, "r", encoding="utf-8") as fh:
        content = fh.read().lower()

    # Must mention instruction input
    assert "instruction" in content, (
        "VLA README must document 'instruction' as an input"
    )
    # Must mention image input
    assert "image" in content, (
        "VLA README must document 'image' as an input"
    )
    # Must mention action output
    assert "action" in content, (
        "VLA README must document 'action' as an output"
    )
    # Must document I/O contract section
    assert "i/o" in content or "계약" in content or "contract" in content, (
        "VLA README must contain I/O contract section (I/O, 계약, or contract)"
    )


def test_vla_readme_model_info():
    """Verify training/vla/README.md mentions VLA model context."""
    assert os.path.isfile(VLA_README), (
        f"VLA README not found: {VLA_README}"
    )
    with open(VLA_README, "r", encoding="utf-8") as fh:
        content = fh.read().lower()

    # Must reference VLA concept
    assert "vla" in content, (
        "VLA README must mention 'VLA'"
    )


def test_vla_readme_action_dimensions():
    """Verify training/vla/README.md specifies action dimensions."""
    assert os.path.isfile(VLA_README), (
        f"VLA README not found: {VLA_README}"
    )
    with open(VLA_README, "r", encoding="utf-8") as fh:
        content = fh.read()

    # Must specify 6-DOF or [6] or (6,) action shape
    has_6dof = "6" in content
    assert has_6dof, (
        "VLA README must specify action dimensionality (6-DOF)"
    )


# ---------------------------------------------------------------------------
# Test 5: datasets/README.md task instruction content
# ---------------------------------------------------------------------------
def test_datasets_readme_task_field():
    """Verify datasets/README.md documents the task instruction field."""
    assert os.path.isfile(DATASETS_README), (
        f"Datasets README not found: {DATASETS_README}"
    )
    with open(DATASETS_README, "r", encoding="utf-8") as fh:
        content = fh.read().lower()

    # Must mention task field
    assert "task" in content, (
        "datasets/README.md must document the 'task' field"
    )


def test_datasets_readme_save_episode():
    """Verify datasets/README.md documents save_episode with task parameter."""
    assert os.path.isfile(DATASETS_README), (
        f"Datasets README not found: {DATASETS_README}"
    )
    with open(DATASETS_README, "r", encoding="utf-8") as fh:
        content = fh.read()

    # Must mention save_episode with task
    assert "save_episode" in content, (
        "datasets/README.md must reference save_episode() API"
    )
    assert "task=" in content or 'task="' in content, (
        "datasets/README.md must show task= parameter usage in save_episode()"
    )


def test_datasets_readme_instruction_context():
    """Verify datasets/README.md provides instruction context for VLA."""
    assert os.path.isfile(DATASETS_README), (
        f"Datasets README not found: {DATASETS_README}"
    )
    with open(DATASETS_README, "r", encoding="utf-8") as fh:
        content = fh.read().lower()

    # Must connect task field to VLA or Phase 7
    has_vla_context = "vla" in content or "phase 7" in content or "instruction" in content
    assert has_vla_context, (
        "datasets/README.md must connect task field to VLA/Phase 7 context"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    runner = StaticVerificationRunner()

    # Test 1: VLA directory exists
    runner.run_test("VLA directory exists", test_vla_directory_exists)

    # Test 2: File existence
    runner.run_test("inference.py exists", test_inference_py_exists)
    runner.run_test("eval_in_sim.py exists", test_eval_in_sim_py_exists)
    runner.run_test("__init__.py exists", test_init_py_exists)
    runner.run_test("VLA README.md exists", test_vla_readme_exists)
    runner.run_test("datasets/README.md exists", test_datasets_readme_exists)

    # Test 3: Python syntax (py_compile)
    runner.run_test("inference.py syntax", test_syntax_inference_py)
    runner.run_test("eval_in_sim.py syntax", test_syntax_eval_in_sim_py)
    runner.run_test("__init__.py syntax", test_syntax_init_py)

    # Test 4: VLA README content
    runner.run_test("VLA README I/O contract", test_vla_readme_io_contract)
    runner.run_test("VLA README model info", test_vla_readme_model_info)
    runner.run_test("VLA README action dimensions", test_vla_readme_action_dimensions)

    # Test 5: datasets/README.md content
    runner.run_test("datasets/README.md task field", test_datasets_readme_task_field)
    runner.run_test("datasets/README.md save_episode", test_datasets_readme_save_episode)
    runner.run_test("datasets/README.md instruction context", test_datasets_readme_instruction_context)

    # Summary
    runner.print_summary()

    if runner.failed == 0:
        sys.stderr.write("All VLA static verification tests passed\n")
        return 0
    else:
        sys.stderr.write(
            f"FAILED: {runner.failed} of {runner.passed + runner.failed} tests failed\n"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
