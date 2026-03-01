#!/usr/bin/env python3
"""run_verification.py — Phase 4 URDF→USD verification runner

Aggregates results from all verification test phases into a final
summary report with acceptance criteria checklist:

  1. Runs all test scripts in sequence:
     - test_import_urdf_static.py   (static code verification)
     - test_import_urdf_patterns.py (pattern compliance)
     - test_import_urdf_units.py    (unit tests)
     - test_urdf_crosscheck.py      (URDF cross-validation)
     - test_import_urdf_runtime.py  (runtime URDF→USD conversion)

  2. Collects pass/fail/skip status for each

  3. Writes a final summary to assets/verification_summary.txt
     with acceptance criteria checklist mapped to test results

  4. Exits 0 if all critical tests pass (runtime test skip is
     acceptable), exits 1 otherwise

Phase: 5 (Verification Summary Report)
"""

import os
import re
import subprocess
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Output report path
SUMMARY_PATH = os.path.join(PROJECT_ROOT, "assets", "verification_summary.txt")

# ---------------------------------------------------------------------------
# Test suite definitions
# ---------------------------------------------------------------------------
# Each entry: (script_name, display_name, is_critical)
# is_critical=False means SKIP results are acceptable (e.g., Isaac Sim tests)
TEST_SUITES = [
    (
        "test_import_urdf_static.py",
        "Static Code Verification",
        True,
    ),
    (
        "test_import_urdf_patterns.py",
        "Pattern Compliance",
        True,
    ),
    (
        "test_import_urdf_units.py",
        "Unit Tests",
        True,
    ),
    (
        "test_urdf_crosscheck.py",
        "URDF Cross-Validation",
        True,
    ),
    (
        "test_import_urdf_runtime.py",
        "Runtime URDF\u2192USD Conversion",
        False,
    ),
]

# ---------------------------------------------------------------------------
# Acceptance criteria mapping
# ---------------------------------------------------------------------------
# Maps each acceptance criterion to the test suites that verify it.
# Criteria verified by runtime tests (non-critical) are marked accordingly.
ACCEPTANCE_CRITERIA = [
    {
        "criterion": "import_urdf_to_isaac.py passes Python syntax check (py_compile)",
        "verified_by": ["test_import_urdf_static.py"],
        "requires_runtime": False,
    },
    {
        "criterion": "All public functions have docstrings",
        "verified_by": ["test_import_urdf_patterns.py"],
        "requires_runtime": False,
    },
    {
        "criterion": "No print() calls -- logging module used throughout",
        "verified_by": ["test_import_urdf_patterns.py"],
        "requires_runtime": False,
    },
    {
        "criterion": "No hardcoded magic numbers for drive params -- params from YAML",
        "verified_by": ["test_import_urdf_patterns.py"],
        "requires_runtime": False,
    },
    {
        "criterion": "SimulationApp initialized before omni.* imports in code flow",
        "verified_by": ["test_import_urdf_patterns.py"],
        "requires_runtime": False,
    },
    {
        "criterion": "--input and --output CLI arguments supported",
        "verified_by": ["test_import_urdf_static.py", "test_import_urdf_units.py"],
        "requires_runtime": False,
    },
    {
        "criterion": "load_params() correctly reads params/control.yaml and params/physics.yaml",
        "verified_by": ["test_import_urdf_static.py", "test_import_urdf_units.py"],
        "requires_runtime": False,
    },
    {
        "criterion": "get_param_value() extracts nested value fields from YAML schema",
        "verified_by": ["test_import_urdf_static.py", "test_import_urdf_units.py"],
        "requires_runtime": False,
    },
    {
        "criterion": (
            "URDF has 6 revolute joints: shoulder_pan, shoulder_lift, "
            "elbow_flex, wrist_flex, wrist_roll, gripper"
        ),
        "verified_by": ["test_import_urdf_static.py", "test_urdf_crosscheck.py"],
        "requires_runtime": False,
    },
    {
        "criterion": "URDF mesh references point to existing files",
        "verified_by": ["test_urdf_crosscheck.py"],
        "requires_runtime": False,
    },
    {
        "criterion": "Joint limit cross-check discrepancies are documented",
        "verified_by": ["test_urdf_crosscheck.py"],
        "requires_runtime": False,
    },
    {
        "criterion": "Runtime URDF->USD conversion produces output file (if Isaac Sim available)",
        "verified_by": ["test_import_urdf_runtime.py"],
        "requires_runtime": True,
    },
    {
        "criterion": "Conversion log written to assets/urdf_import_results.log (if Isaac Sim available)",
        "verified_by": ["test_import_urdf_runtime.py"],
        "requires_runtime": True,
    },
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
class SuiteResult:
    """Stores the result of running a single test suite."""

    def __init__(self, script_name, display_name, is_critical):
        self.script_name = script_name
        self.display_name = display_name
        self.is_critical = is_critical
        self.returncode = None
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.warnings = 0
        self.total = 0
        self.stderr_output = ""
        self.status = "NOT_RUN"

    @property
    def all_passed(self):
        """True if no failures occurred (skips and warnings are acceptable)."""
        return self.failed == 0 and self.returncode == 0


# ---------------------------------------------------------------------------
# Parse test output
# ---------------------------------------------------------------------------
def parse_test_output(stderr_text):
    """Extract pass/fail/skip/warning counts from test stderr output.

    Scans for the summary line pattern used by all test runners:
        Total: N | Passed: P | Failed: F
        Total: N | Passed: P | Failed: F | Skipped: S
        Total: N | Passed: P | Warnings: W | Failed: F

    Args:
        stderr_text: The stderr output from a test subprocess.

    Returns:
        Dict with keys: passed, failed, skipped, warnings, total.
    """
    result = {"passed": 0, "failed": 0, "skipped": 0, "warnings": 0, "total": 0}

    # Match the summary line (last occurrence takes precedence)
    pattern = re.compile(
        r"Total:\s*(\d+)\s*\|"
        r"\s*Passed:\s*(\d+)"
        r"(?:\s*\|\s*Warnings:\s*(\d+))?"
        r"\s*\|\s*Failed:\s*(\d+)"
        r"(?:\s*\|\s*Skipped:\s*(\d+))?",
    )

    matches = list(pattern.finditer(stderr_text))
    if matches:
        m = matches[-1]  # Use the last match (summary line)
        result["total"] = int(m.group(1))
        result["passed"] = int(m.group(2))
        result["warnings"] = int(m.group(3)) if m.group(3) else 0
        result["failed"] = int(m.group(4))
        result["skipped"] = int(m.group(5)) if m.group(5) else 0

    return result


# ---------------------------------------------------------------------------
# Run a test suite
# ---------------------------------------------------------------------------
def run_test_suite(script_name, display_name, is_critical):
    """Run a single test script as a subprocess and collect results.

    Args:
        script_name: Filename of the test script in tests/.
        display_name: Human-readable name for reporting.
        is_critical: Whether failures in this suite should cause overall failure.

    Returns:
        SuiteResult with populated fields.
    """
    suite = SuiteResult(script_name, display_name, is_critical)
    script_path = os.path.join(SCRIPT_DIR, script_name)

    if not os.path.exists(script_path):
        suite.status = "MISSING"
        suite.failed = 1
        suite.total = 1
        suite.stderr_output = f"Test script not found: {script_path}"
        return suite

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=PROJECT_ROOT,
        )

        suite.returncode = result.returncode
        suite.stderr_output = result.stderr

        # Parse counts from stderr
        counts = parse_test_output(result.stderr)
        suite.passed = counts["passed"]
        suite.failed = counts["failed"]
        suite.skipped = counts["skipped"]
        suite.warnings = counts["warnings"]
        suite.total = counts["total"]

        # Determine status
        if result.returncode == 0:
            if suite.skipped > 0 and suite.passed == 0:
                suite.status = "SKIP"
            elif suite.skipped > 0:
                suite.status = "PARTIAL"
            else:
                suite.status = "PASS"
        else:
            suite.status = "FAIL"

    except subprocess.TimeoutExpired:
        suite.status = "TIMEOUT"
        suite.failed = 1
        suite.total = 1
        suite.stderr_output = f"Test script timed out after 300 seconds"

    except Exception as exc:
        suite.status = "ERROR"
        suite.failed = 1
        suite.total = 1
        suite.stderr_output = f"Error running test script: {exc}"

    return suite


# ---------------------------------------------------------------------------
# Determine acceptance criterion status
# ---------------------------------------------------------------------------
def get_criterion_status(criterion_info, suite_results_map):
    """Determine whether an acceptance criterion is met.

    Args:
        criterion_info: Dict with criterion, verified_by, requires_runtime.
        suite_results_map: Dict mapping script_name -> SuiteResult.

    Returns:
        Tuple of (status_str, detail_str).
        status_str is one of: "PASS", "FAIL", "SKIP".
    """
    verified_by = criterion_info["verified_by"]
    requires_runtime = criterion_info["requires_runtime"]

    all_pass = True
    any_skip = False
    any_fail = False

    for script_name in verified_by:
        suite = suite_results_map.get(script_name)
        if suite is None:
            any_fail = True
            continue

        if suite.status == "FAIL" or suite.status == "ERROR" or suite.status == "TIMEOUT":
            any_fail = True
        elif suite.status == "SKIP" or suite.status == "PARTIAL":
            any_skip = True
        elif suite.status == "PASS":
            pass  # good
        elif suite.status == "MISSING":
            any_fail = True
        else:
            any_fail = True

    if any_fail:
        return "FAIL", "Test suite(s) failed"
    elif any_skip and requires_runtime:
        return "SKIP", "Isaac Sim not available (acceptable)"
    elif any_skip:
        return "PASS", "Partial skip (non-critical tests)"
    else:
        return "PASS", "Verified by: " + ", ".join(verified_by)


# ---------------------------------------------------------------------------
# Write verification summary report
# ---------------------------------------------------------------------------
def write_summary(suite_results, report_path):
    """Write the final verification summary report.

    Args:
        suite_results: List of SuiteResult objects.
        report_path: Path to write the summary file.
    """
    report_dir = os.path.dirname(report_path)
    os.makedirs(report_dir, exist_ok=True)

    suite_results_map = {s.script_name: s for s in suite_results}

    lines = []
    sep = "=" * 72

    # Header
    lines.append(sep)
    lines.append("Phase 4 URDF->USD Verification Summary Report")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Project: SO-ARM101 URDF->USD Automation (import_urdf_to_isaac.py)")
    lines.append(sep)

    # --- Section 1: Test Suite Results ---
    lines.append("")
    lines.append("--- Test Suite Results ---")
    lines.append("")
    lines.append(
        f"{'Suite':<35} {'Status':>8} {'Passed':>8} {'Failed':>8} "
        f"{'Skipped':>8} {'Warn':>8} {'Total':>8}"
    )
    lines.append("-" * 95)

    total_passed = 0
    total_failed = 0
    total_skipped = 0
    total_warnings = 0
    total_tests = 0
    critical_failures = 0

    for suite in suite_results:
        critical_tag = "" if suite.is_critical else " *"
        lines.append(
            f"{suite.display_name + critical_tag:<35} {suite.status:>8} "
            f"{suite.passed:>8} {suite.failed:>8} "
            f"{suite.skipped:>8} {suite.warnings:>8} {suite.total:>8}"
        )
        total_passed += suite.passed
        total_failed += suite.failed
        total_skipped += suite.skipped
        total_warnings += suite.warnings
        total_tests += suite.total

        if suite.is_critical and not suite.all_passed:
            critical_failures += 1

    lines.append("-" * 95)
    lines.append(
        f"{'TOTAL':<35} {'':>8} {total_passed:>8} {total_failed:>8} "
        f"{total_skipped:>8} {total_warnings:>8} {total_tests:>8}"
    )
    lines.append("")
    lines.append("  * = non-critical (SKIP is acceptable)")

    # --- Section 2: Acceptance Criteria Checklist ---
    lines.append("")
    lines.append("")
    lines.append("--- Acceptance Criteria Checklist ---")
    lines.append("")

    criteria_pass = 0
    criteria_fail = 0
    criteria_skip = 0

    for idx, criterion_info in enumerate(ACCEPTANCE_CRITERIA, 1):
        status, detail = get_criterion_status(criterion_info, suite_results_map)

        if status == "PASS":
            marker = "[x]"
            criteria_pass += 1
        elif status == "SKIP":
            marker = "[-]"
            criteria_skip += 1
        else:
            marker = "[ ]"
            criteria_fail += 1

        runtime_tag = " (runtime)" if criterion_info["requires_runtime"] else ""
        lines.append(
            f"  {idx:>2}. {marker} {criterion_info['criterion']}{runtime_tag}"
        )
        lines.append(f"       {detail}")

    lines.append("")
    lines.append(
        f"Criteria: {criteria_pass} passed, {criteria_fail} failed, "
        f"{criteria_skip} skipped (runtime)"
    )

    # --- Section 3: Detailed Test Output ---
    lines.append("")
    lines.append("")
    lines.append("--- Detailed Test Output ---")

    for suite in suite_results:
        lines.append("")
        lines.append(f">> {suite.display_name} ({suite.script_name})")
        lines.append(f"   Status: {suite.status}")

        if suite.stderr_output:
            # Include individual test results from stderr
            stderr_lines = suite.stderr_output.strip().split("\n")
            # Extract test result lines ([PASS], [FAIL], [SKIP], [WARN])
            result_lines = [
                line.strip() for line in stderr_lines
                if any(tag in line for tag in ["[PASS]", "[FAIL]", "[SKIP]", "[WARN]", "[ERROR]"])
            ]
            if result_lines:
                for line in result_lines:
                    lines.append(f"   {line}")
            else:
                # If no structured results found, show last few lines
                for line in stderr_lines[-5:]:
                    lines.append(f"   {line.strip()}")

    # --- Section 4: Final Verdict ---
    lines.append("")
    lines.append("")
    lines.append(sep)

    overall_pass = (
        critical_failures == 0
        and total_failed == 0
        and criteria_fail == 0
    )

    if overall_pass:
        lines.append("VERDICT: PASS")
        lines.append("")
        lines.append(
            "All critical verification tests passed. "
            "The import_urdf_to_isaac.py script meets all testable acceptance criteria."
        )
        if total_skipped > 0:
            lines.append(
                f"Note: {total_skipped} runtime test(s) skipped "
                f"(Isaac Sim not available). These tests are non-critical."
            )
    else:
        lines.append("VERDICT: FAIL")
        lines.append("")
        lines.append(
            f"Verification failed: {critical_failures} critical suite(s) failed, "
            f"{total_failed} total test(s) failed, "
            f"{criteria_fail} acceptance criteria not met."
        )

    lines.append(sep)

    # Write to file
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    sys.stderr.write("\n" + "=" * 60 + "\n")
    sys.stderr.write("Phase 4 URDF->USD Verification Runner\n")
    sys.stderr.write("=" * 60 + "\n\n")

    # Run all test suites in sequence
    suite_results = []

    for script_name, display_name, is_critical in TEST_SUITES:
        sys.stderr.write(f"Running: {display_name} ({script_name})...\n")

        result = run_test_suite(script_name, display_name, is_critical)
        suite_results.append(result)

        # Print immediate status
        status_line = (
            f"  -> {result.status} "
            f"(passed={result.passed}, failed={result.failed}, "
            f"skipped={result.skipped}, warnings={result.warnings})\n"
        )
        sys.stderr.write(status_line)

    # Write summary report
    write_summary(suite_results, SUMMARY_PATH)
    sys.stderr.write(f"\nSummary report written to: {SUMMARY_PATH}\n")

    # Determine overall result
    critical_failures = sum(
        1 for s in suite_results if s.is_critical and not s.all_passed
    )
    total_failed = sum(s.failed for s in suite_results)

    sys.stderr.write("\n" + "=" * 60 + "\n")

    if critical_failures == 0 and total_failed == 0:
        sys.stderr.write("Verification PASSED\n")
        sys.stderr.write("=" * 60 + "\n")
        return 0
    else:
        sys.stderr.write(
            f"Verification FAILED "
            f"({critical_failures} critical suite(s), "
            f"{total_failed} test(s) failed)\n"
        )
        sys.stderr.write("=" * 60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
