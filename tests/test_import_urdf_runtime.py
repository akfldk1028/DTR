#!/usr/bin/env python3
"""test_import_urdf_runtime.py — Runtime URDF→USD conversion test

Attempts the actual URDF→USD conversion using Isaac Sim 5.1.0:
  1. Checks if Isaac Sim is available (try import isaacsim)
  2. If available: runs import_urdf_to_isaac.py via subprocess
  3. Verifies output USD file exists in assets/usd/
  4. Verifies assets/urdf_import_results.log was created with expected fields
  5. If Isaac Sim unavailable: logs skip reason and exits with code 0

This test gracefully skips when Isaac Sim is not installed,
making it safe for CI environments without GPU/Isaac Sim.
"""

import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Script under test
IMPORT_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "import_urdf_to_isaac.py")

# Input URDF
URDF_PATH = os.path.join(
    PROJECT_ROOT, "robot_description", "urdf", "so101_new_calib.urdf"
)

# Expected output directory and file
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "assets", "usd")
# import_urdf_to_isaac.py auto-names the output from the URDF stem
EXPECTED_USD = os.path.join(OUTPUT_DIR, "so101_new_calib.usd")

# Log file written by the script
LOG_FILE = os.path.join(PROJECT_ROOT, "assets", "urdf_import_results.log")

# Expected report fields in the log
EXPECTED_LOG_FIELDS = [
    "Joint count:",
    "Stability result:",
    "File size:",
]


# ---------------------------------------------------------------------------
# Test runner (same pattern as other test files)
# ---------------------------------------------------------------------------
class RuntimeTestRunner:
    """Runs all runtime tests and tracks results."""

    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def run_test(self, name, test_fn):
        """Run a single test function and record the result."""
        try:
            test_fn()
            self.results.append((name, "PASS", ""))
            self.passed += 1
        except SkipTest as exc:
            self.results.append((name, "SKIP", str(exc)))
            self.skipped += 1
        except AssertionError as exc:
            self.results.append((name, "FAIL", str(exc)))
            self.failed += 1
        except Exception as exc:
            self.results.append((name, "ERROR", str(exc)))
            self.failed += 1

    def print_summary(self):
        """Print test results summary."""
        sys.stderr.write("\n=== Runtime Test Results ===\n")
        for name, status, msg in self.results:
            marker = f"[{status}]"
            line = f"  {marker} {name}"
            if msg:
                line += f" — {msg}"
            sys.stderr.write(line + "\n")
        sys.stderr.write(
            f"\nTotal: {self.passed + self.failed + self.skipped} | "
            f"Passed: {self.passed} | Failed: {self.failed} | "
            f"Skipped: {self.skipped}\n"
        )
        sys.stderr.write("=" * 40 + "\n")


class SkipTest(Exception):
    """Raised when a test should be skipped (e.g., Isaac Sim unavailable)."""


# ---------------------------------------------------------------------------
# Isaac Sim availability check
# ---------------------------------------------------------------------------
_isaac_sim_available = None


def check_isaac_sim_available():
    """Check whether Isaac Sim (isaacsim) can be imported.

    Caches the result to avoid repeated import attempts.

    Returns:
        True if isaacsim is importable, False otherwise.
    """
    global _isaac_sim_available
    if _isaac_sim_available is not None:
        return _isaac_sim_available

    try:
        import importlib

        importlib.import_module("isaacsim")
        _isaac_sim_available = True
    except (ImportError, ModuleNotFoundError):
        _isaac_sim_available = False

    return _isaac_sim_available


def require_isaac_sim():
    """Raise SkipTest if Isaac Sim is not available."""
    if not check_isaac_sim_available():
        raise SkipTest("Isaac Sim (isaacsim) not available — skipping runtime test")


# ---------------------------------------------------------------------------
# Test 1: Isaac Sim availability detection
# ---------------------------------------------------------------------------
def test_isaac_sim_detection():
    """Verify that Isaac Sim availability can be detected without error."""
    result = check_isaac_sim_available()
    assert isinstance(result, bool), (
        f"check_isaac_sim_available() should return bool, got: {type(result)}"
    )
    # Log the detection result (informational, not a pass/fail criterion)
    status = "AVAILABLE" if result else "NOT AVAILABLE"
    sys.stderr.write(f"  Isaac Sim detection: {status}\n")


# ---------------------------------------------------------------------------
# Test 2: Prerequisites check
# ---------------------------------------------------------------------------
def test_prerequisites():
    """Verify that the import script and URDF file exist."""
    assert os.path.exists(IMPORT_SCRIPT), (
        f"Import script not found: {IMPORT_SCRIPT}"
    )
    assert os.path.exists(URDF_PATH), (
        f"URDF file not found: {URDF_PATH}"
    )


# ---------------------------------------------------------------------------
# Test 3: Run URDF→USD conversion via subprocess
# ---------------------------------------------------------------------------
def test_run_conversion():
    """Run import_urdf_to_isaac.py and verify it exits successfully.

    Skipped if Isaac Sim is not available.
    """
    require_isaac_sim()

    cmd = [
        sys.executable,
        IMPORT_SCRIPT,
        "--input", URDF_PATH,
        "--output", OUTPUT_DIR + "/",
        "--headless",
    ]

    sys.stderr.write(f"  Running: {' '.join(cmd)}\n")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,  # 5-minute timeout for Isaac Sim startup + conversion
        cwd=PROJECT_ROOT,
    )

    # Log stderr output for debugging
    if result.stderr:
        # Limit output to last 50 lines to avoid flooding
        stderr_lines = result.stderr.strip().split("\n")
        if len(stderr_lines) > 50:
            sys.stderr.write(f"  (showing last 50 of {len(stderr_lines)} stderr lines)\n")
            stderr_lines = stderr_lines[-50:]
        for line in stderr_lines:
            sys.stderr.write(f"    | {line}\n")

    assert result.returncode == 0, (
        f"import_urdf_to_isaac.py exited with code {result.returncode}.\n"
        f"stderr (last 20 lines):\n"
        + "\n".join(result.stderr.strip().split("\n")[-20:])
    )


# ---------------------------------------------------------------------------
# Test 4: Verify output USD file exists
# ---------------------------------------------------------------------------
def test_usd_output_exists():
    """Verify the output USD file was created in assets/usd/.

    Skipped if Isaac Sim is not available.
    """
    require_isaac_sim()

    assert os.path.exists(EXPECTED_USD), (
        f"Expected USD output not found: {EXPECTED_USD}"
    )

    file_size = os.path.getsize(EXPECTED_USD)
    assert file_size > 0, (
        f"USD file exists but is empty (0 bytes): {EXPECTED_USD}"
    )

    sys.stderr.write(
        f"  USD file found: {EXPECTED_USD} ({file_size / 1024:.1f} KB)\n"
    )


# ---------------------------------------------------------------------------
# Test 5: Verify log file exists with expected report fields
# ---------------------------------------------------------------------------
def test_log_file_exists():
    """Verify assets/urdf_import_results.log was created.

    Skipped if Isaac Sim is not available.
    """
    require_isaac_sim()

    assert os.path.exists(LOG_FILE), (
        f"Log file not found: {LOG_FILE}"
    )

    file_size = os.path.getsize(LOG_FILE)
    assert file_size > 0, (
        f"Log file exists but is empty (0 bytes): {LOG_FILE}"
    )


def test_log_file_fields():
    """Verify the log file contains expected report fields.

    The write_final_report() function in import_urdf_to_isaac.py writes
    a structured report block with fields like 'Joint count:',
    'Stability result:', and 'File size:'.

    Skipped if Isaac Sim is not available.
    """
    require_isaac_sim()

    assert os.path.exists(LOG_FILE), (
        f"Log file not found: {LOG_FILE}"
    )

    with open(LOG_FILE, "r", encoding="utf-8") as fh:
        log_content = fh.read()

    missing_fields = []
    for field in EXPECTED_LOG_FIELDS:
        if field not in log_content:
            missing_fields.append(field)

    assert not missing_fields, (
        f"Log file missing expected report fields: {missing_fields}\n"
        f"Log file content (last 30 lines):\n"
        + "\n".join(log_content.strip().split("\n")[-30:])
    )

    sys.stderr.write(f"  All expected log fields found: {EXPECTED_LOG_FIELDS}\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    runner = RuntimeTestRunner()

    # Test 1: Detection (always runs)
    runner.run_test("Isaac Sim detection", test_isaac_sim_detection)

    # Test 2: Prerequisites (always runs)
    runner.run_test("Prerequisites check", test_prerequisites)

    # Test 3-6: Runtime tests (skipped if Isaac Sim unavailable)
    runner.run_test("URDF→USD conversion subprocess", test_run_conversion)
    runner.run_test("USD output file exists", test_usd_output_exists)
    runner.run_test("Log file exists", test_log_file_exists)
    runner.run_test("Log file report fields", test_log_file_fields)

    # Summary
    runner.print_summary()

    if runner.failed > 0:
        sys.stderr.write(
            f"FAILED: {runner.failed} of "
            f"{runner.passed + runner.failed + runner.skipped} tests failed\n"
        )
        return 1

    # All tests passed or skipped — both are acceptable outcomes
    if runner.skipped > 0:
        sys.stderr.write(
            f"Runtime verification complete (SKIPPED: {runner.skipped} tests — "
            f"Isaac Sim not available)\n"
        )
    else:
        sys.stderr.write("Runtime verification complete\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
