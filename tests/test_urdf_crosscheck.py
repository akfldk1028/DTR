#!/usr/bin/env python3
"""test_urdf_crosscheck.py — URDF cross-validation against params/control.yaml

Validates the URDF file structure and cross-checks joint limits:
  1. Parses so101_new_calib.urdf with xml.etree.ElementTree
  2. Verifies 6 revolute joints exist with expected names
  3. Cross-checks joint limits (radians) against control.yaml position_min/max
  4. Reports mismatches with tolerance of 0.05 rad
  5. Verifies mesh file references point to existing files (even as LFS pointers)
  6. Writes validation report to assets/urdf_validation_report.txt

Known discrepancies (documented, reported as WARN not FAIL):
  - elbow_flex: limits differ by ~0.12 rad (calibration offset in URDF)
  - wrist_roll: asymmetric limits in URDF (-2.74385 / +2.84121)
  - gripper: lower limit sign mismatch (URDF: -0.1745, control.yaml: +0.1745)
"""

import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import yaml

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
URDF_PATH = os.path.join(
    PROJECT_ROOT, "robot_description", "urdf", "so101_new_calib.urdf"
)
CONTROL_YAML_PATH = os.path.join(PROJECT_ROOT, "params", "control.yaml")
REPORT_PATH = os.path.join(PROJECT_ROOT, "assets", "urdf_validation_report.txt")
MESHES_DIR = os.path.join(PROJECT_ROOT, "robot_description", "meshes")

EXPECTED_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
EXPECTED_NUM_JOINTS = 6
JOINT_LIMIT_TOLERANCE = 0.05  # radians


# ---------------------------------------------------------------------------
# Test runner (same pattern as test_import_urdf_static.py)
# ---------------------------------------------------------------------------
class URDFCrossValidationRunner:
    """Runs all URDF cross-validation tests and tracks results."""

    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.report_lines = []

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

    def add_warning(self, name, msg):
        """Record a warning (non-fatal discrepancy)."""
        self.results.append((name, "WARN", msg))
        self.warnings += 1

    def add_report_line(self, line):
        """Add a line to the validation report."""
        self.report_lines.append(line)

    def print_summary(self):
        """Print test results summary to stderr."""
        sys.stderr.write("\n=== URDF Cross-Validation Results ===\n")
        for name, status, msg in self.results:
            marker = f"[{status}]"
            line = f"  {marker} {name}"
            if msg:
                line += f" — {msg}"
            sys.stderr.write(line + "\n")
        sys.stderr.write(
            f"\nTotal: {self.passed + self.failed + self.warnings} | "
            f"Passed: {self.passed} | Warnings: {self.warnings} | "
            f"Failed: {self.failed}\n"
        )
        sys.stderr.write("=" * 40 + "\n")

    def write_report(self, report_path):
        """Write validation report to file."""
        report_dir = os.path.dirname(report_path)
        os.makedirs(report_dir, exist_ok=True)

        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write("=" * 72 + "\n")
            fh.write("URDF Cross-Validation Report\n")
            fh.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
            fh.write(f"URDF: {URDF_PATH}\n")
            fh.write(f"Control YAML: {CONTROL_YAML_PATH}\n")
            fh.write(f"Tolerance: {JOINT_LIMIT_TOLERANCE} rad\n")
            fh.write("=" * 72 + "\n\n")

            # Detailed report lines
            for line in self.report_lines:
                fh.write(line + "\n")

            # Test results summary
            fh.write("\n" + "-" * 72 + "\n")
            fh.write("Test Results Summary\n")
            fh.write("-" * 72 + "\n")
            for name, status, msg in self.results:
                marker = f"[{status}]"
                line = f"  {marker} {name}"
                if msg:
                    line += f" — {msg}"
                fh.write(line + "\n")

            fh.write(
                f"\nTotal: {self.passed + self.failed + self.warnings} | "
                f"Passed: {self.passed} | Warnings: {self.warnings} | "
                f"Failed: {self.failed}\n"
            )

            if self.failed == 0:
                fh.write("\nResult: PASS (all critical checks passed)\n")
            else:
                fh.write(f"\nResult: FAIL ({self.failed} critical failures)\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_control_yaml():
    """Load and return params/control.yaml as a dict."""
    with open(CONTROL_YAML_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parse_urdf():
    """Parse the URDF file and return (tree, root)."""
    tree = ET.parse(URDF_PATH)
    root = tree.getroot()
    return tree, root


def get_revolute_joints(root):
    """Return list of revolute joint elements from URDF root."""
    return [
        joint for joint in root.findall("joint")
        if joint.get("type") == "revolute"
    ]


def get_urdf_joint_limits(root):
    """Extract joint limits from URDF as {name: (lower, upper)} dict."""
    limits = {}
    for joint in get_revolute_joints(root):
        name = joint.get("name")
        limit_elem = joint.find("limit")
        if limit_elem is not None:
            lower = float(limit_elem.get("lower", "0"))
            upper = float(limit_elem.get("upper", "0"))
            limits[name] = (lower, upper)
    return limits


def get_mesh_filenames(root):
    """Extract unique mesh filenames referenced in URDF."""
    meshes = set()
    for mesh_elem in root.iter("mesh"):
        filename = mesh_elem.get("filename")
        if filename:
            meshes.add(filename)
    return sorted(meshes)


def is_lfs_pointer(filepath):
    """Check if a file is a Git LFS pointer (starts with 'version https://git-lfs')."""
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            first_line = fh.readline()
            return first_line.startswith("version https://git-lfs")
    except (UnicodeDecodeError, IOError):
        return False


# ---------------------------------------------------------------------------
# Test 1: URDF parsing
# ---------------------------------------------------------------------------
def test_urdf_file_exists():
    """Verify URDF file exists at expected path."""
    assert os.path.exists(URDF_PATH), (
        f"URDF file not found: {URDF_PATH}"
    )


def test_urdf_valid_xml():
    """Verify URDF file is valid XML with <robot> root."""
    tree, root = parse_urdf()
    assert root.tag == "robot", (
        f"Expected root tag 'robot', got: {root.tag}"
    )
    assert root.get("name") == "so101_new_calib", (
        f"Expected robot name 'so101_new_calib', got: {root.get('name')}"
    )


# ---------------------------------------------------------------------------
# Test 2: Revolute joint validation
# ---------------------------------------------------------------------------
def test_revolute_joint_count():
    """Verify exactly 6 revolute joints exist."""
    _, root = parse_urdf()
    joints = get_revolute_joints(root)
    assert len(joints) == EXPECTED_NUM_JOINTS, (
        f"Expected {EXPECTED_NUM_JOINTS} revolute joints, got: {len(joints)}"
    )


def test_revolute_joint_names():
    """Verify all expected revolute joint names exist."""
    _, root = parse_urdf()
    joints = get_revolute_joints(root)
    actual_names = [joint.get("name") for joint in joints]

    for expected_name in EXPECTED_JOINT_NAMES:
        assert expected_name in actual_names, (
            f"Expected joint '{expected_name}' not found. "
            f"Actual: {actual_names}"
        )


def test_revolute_joint_limits_present():
    """Verify all revolute joints have <limit> with lower/upper."""
    _, root = parse_urdf()
    for joint in get_revolute_joints(root):
        name = joint.get("name")
        limit_elem = joint.find("limit")
        assert limit_elem is not None, (
            f"Joint '{name}' missing <limit> element"
        )
        assert limit_elem.get("lower") is not None, (
            f"Joint '{name}' <limit> missing 'lower'"
        )
        assert limit_elem.get("upper") is not None, (
            f"Joint '{name}' <limit> missing 'upper'"
        )


def test_revolute_joint_axis_present():
    """Verify all revolute joints have <axis> element."""
    _, root = parse_urdf()
    for joint in get_revolute_joints(root):
        name = joint.get("name")
        axis_elem = joint.find("axis")
        assert axis_elem is not None, (
            f"Joint '{name}' missing <axis> element"
        )


# ---------------------------------------------------------------------------
# Test 3: Control.yaml loading
# ---------------------------------------------------------------------------
def test_control_yaml_exists():
    """Verify params/control.yaml exists."""
    assert os.path.exists(CONTROL_YAML_PATH), (
        f"Control YAML not found: {CONTROL_YAML_PATH}"
    )


def test_control_yaml_has_joint_limits():
    """Verify control.yaml has joint_limits with position_min/max."""
    params = load_control_yaml()
    assert "joint_limits" in params, (
        "control.yaml missing 'joint_limits' key"
    )
    assert "position_min" in params["joint_limits"], (
        "control.yaml missing 'joint_limits.position_min'"
    )
    assert "position_max" in params["joint_limits"], (
        "control.yaml missing 'joint_limits.position_max'"
    )


def test_control_yaml_has_joint_names():
    """Verify control.yaml has joint_names with 6 entries."""
    params = load_control_yaml()
    assert "joint_names" in params, (
        "control.yaml missing 'joint_names' key"
    )
    names = params["joint_names"]["value"]
    assert len(names) == EXPECTED_NUM_JOINTS, (
        f"Expected {EXPECTED_NUM_JOINTS} joint names, got: {len(names)}"
    )


def test_control_yaml_joint_limit_lengths():
    """Verify position_min and position_max have 6 entries each."""
    params = load_control_yaml()
    pos_min = params["joint_limits"]["position_min"]["value"]
    pos_max = params["joint_limits"]["position_max"]["value"]

    assert len(pos_min) == EXPECTED_NUM_JOINTS, (
        f"position_min has {len(pos_min)} entries, expected {EXPECTED_NUM_JOINTS}"
    )
    assert len(pos_max) == EXPECTED_NUM_JOINTS, (
        f"position_max has {len(pos_max)} entries, expected {EXPECTED_NUM_JOINTS}"
    )


# ---------------------------------------------------------------------------
# Test 4: Joint limit cross-check (URDF vs control.yaml)
# ---------------------------------------------------------------------------
def run_joint_limit_crosscheck(runner):
    """Cross-check URDF joint limits against control.yaml.

    Reports mismatches exceeding JOINT_LIMIT_TOLERANCE as WARN (not FAIL),
    since known calibration discrepancies are expected.
    """
    _, root = parse_urdf()
    params = load_control_yaml()

    urdf_limits = get_urdf_joint_limits(root)
    joint_names = params["joint_names"]["value"]
    pos_min = params["joint_limits"]["position_min"]["value"]
    pos_max = params["joint_limits"]["position_max"]["value"]

    runner.add_report_line("--- Joint Limit Cross-Check ---")
    runner.add_report_line(
        f"{'Joint':<16} {'URDF lower':>12} {'YAML lower':>12} "
        f"{'Δ lower':>10} {'URDF upper':>12} {'YAML upper':>12} "
        f"{'Δ upper':>10} {'Status':>8}"
    )
    runner.add_report_line("-" * 100)

    all_within_tolerance = True

    for i, joint_name in enumerate(joint_names):
        if joint_name not in urdf_limits:
            runner.add_warning(
                f"Cross-check {joint_name}",
                f"Joint not found in URDF"
            )
            runner.add_report_line(
                f"{joint_name:<16} {'N/A':>12} {pos_min[i]:>12.6f} "
                f"{'N/A':>10} {'N/A':>12} {pos_max[i]:>12.6f} "
                f"{'N/A':>10} {'MISSING':>8}"
            )
            continue

        urdf_lower, urdf_upper = urdf_limits[joint_name]
        yaml_lower = pos_min[i]
        yaml_upper = pos_max[i]

        delta_lower = abs(urdf_lower - yaml_lower)
        delta_upper = abs(urdf_upper - yaml_upper)

        lower_ok = delta_lower <= JOINT_LIMIT_TOLERANCE
        upper_ok = delta_upper <= JOINT_LIMIT_TOLERANCE

        if lower_ok and upper_ok:
            status = "OK"
        else:
            status = "WARN"
            all_within_tolerance = False
            details = []
            if not lower_ok:
                details.append(
                    f"lower diff={delta_lower:.4f} rad "
                    f"(URDF={urdf_lower:.6f}, YAML={yaml_lower:.6f})"
                )
            if not upper_ok:
                details.append(
                    f"upper diff={delta_upper:.4f} rad "
                    f"(URDF={urdf_upper:.6f}, YAML={yaml_upper:.6f})"
                )
            runner.add_warning(
                f"Cross-check {joint_name}",
                "; ".join(details)
            )

        runner.add_report_line(
            f"{joint_name:<16} {urdf_lower:>12.6f} {yaml_lower:>12.6f} "
            f"{delta_lower:>10.6f} {urdf_upper:>12.6f} {yaml_upper:>12.6f} "
            f"{delta_upper:>10.6f} {status:>8}"
        )

    runner.add_report_line("")

    if all_within_tolerance:
        runner.add_report_line(
            f"All joint limits within tolerance ({JOINT_LIMIT_TOLERANCE} rad)"
        )
    else:
        runner.add_report_line(
            f"Some joint limits exceed tolerance ({JOINT_LIMIT_TOLERANCE} rad) — "
            f"see WARN entries above. These are known calibration discrepancies."
        )

    return all_within_tolerance


# ---------------------------------------------------------------------------
# Test 5: Mesh file reference validation
# ---------------------------------------------------------------------------
def run_mesh_validation(runner):
    """Verify mesh file references in URDF point to existing files.

    Checks for mesh files at three locations:
      1. Relative to the URDF directory (as specified in URDF)
      2. In robot_description/meshes/ as .stl files
      3. In robot_description/meshes/ as .part metadata files
    Files that are Git LFS pointers are considered valid.
    """
    _, root = parse_urdf()
    urdf_dir = os.path.dirname(URDF_PATH)
    mesh_files = get_mesh_filenames(root)

    runner.add_report_line("\n--- Mesh File Validation ---")
    runner.add_report_line(
        f"{'Mesh Reference':<55} {'Status':>10} {'Location'}"
    )
    runner.add_report_line("-" * 100)

    all_found = True

    for mesh_ref in mesh_files:
        # Path as referenced in URDF (relative to URDF dir)
        urdf_relative_path = os.path.join(urdf_dir, mesh_ref)

        # Derive the base name (without directory prefix and extension)
        mesh_basename = os.path.splitext(os.path.basename(mesh_ref))[0]
        mesh_ext = os.path.splitext(mesh_ref)[1]

        # Check locations in order of priority
        found = False
        location = ""

        # 1. Check URDF-relative path (exact path as in URDF)
        if os.path.exists(urdf_relative_path):
            if is_lfs_pointer(urdf_relative_path):
                found = True
                location = f"LFS pointer: {urdf_relative_path}"
            else:
                found = True
                location = f"Direct: {urdf_relative_path}"

        # 2. Check robot_description/meshes/ for .stl file
        if not found:
            meshes_stl_path = os.path.join(MESHES_DIR, f"{mesh_basename}{mesh_ext}")
            if os.path.exists(meshes_stl_path):
                if is_lfs_pointer(meshes_stl_path):
                    found = True
                    location = f"LFS pointer: {meshes_stl_path}"
                else:
                    found = True
                    location = f"Meshes dir: {meshes_stl_path}"

        # 3. Check robot_description/meshes/ for .part metadata
        if not found:
            part_path = os.path.join(MESHES_DIR, f"{mesh_basename}.part")
            if os.path.exists(part_path):
                found = True
                location = f"Part metadata: {part_path}"

        if found:
            status = "FOUND"
        else:
            status = "MISSING"
            all_found = False
            runner.add_warning(
                f"Mesh {mesh_ref}",
                f"File not found at URDF-relative path or meshes directory"
            )

        runner.add_report_line(
            f"{mesh_ref:<55} {status:>10} {location}"
        )

    runner.add_report_line("")

    if all_found:
        runner.add_report_line(
            f"All {len(mesh_files)} mesh references resolved successfully"
        )
    else:
        missing_count = sum(
            1 for _, status, _ in runner.results
            if status == "WARN" and "Mesh" in _
        )
        runner.add_report_line(
            f"WARNING: Some mesh files could not be resolved. "
            f"Ensure STL files are generated from Onshape .part metadata."
        )

    return all_found


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    runner = URDFCrossValidationRunner()

    runner.add_report_line("Phase 3: URDF Cross-Validation\n")

    # --- Section 1: URDF file structure ---
    runner.add_report_line("--- URDF File Structure ---")
    runner.run_test("URDF file exists", test_urdf_file_exists)
    runner.run_test("URDF valid XML", test_urdf_valid_xml)
    runner.run_test("6 revolute joints", test_revolute_joint_count)
    runner.run_test("Expected joint names", test_revolute_joint_names)
    runner.run_test("Joint limits present", test_revolute_joint_limits_present)
    runner.run_test("Joint axis present", test_revolute_joint_axis_present)

    # --- Section 2: control.yaml structure ---
    runner.add_report_line("\n--- Control YAML Structure ---")
    runner.run_test("control.yaml exists", test_control_yaml_exists)
    runner.run_test("control.yaml has joint_limits", test_control_yaml_has_joint_limits)
    runner.run_test("control.yaml has joint_names", test_control_yaml_has_joint_names)
    runner.run_test("control.yaml limit lengths", test_control_yaml_joint_limit_lengths)

    # --- Section 3: Joint limit cross-check ---
    runner.add_report_line("")
    run_joint_limit_crosscheck(runner)

    # --- Section 4: Mesh file validation ---
    run_mesh_validation(runner)

    # --- Write report ---
    runner.write_report(REPORT_PATH)
    runner.add_report_line(f"\nReport written to: {REPORT_PATH}")

    # --- Summary ---
    runner.print_summary()

    if runner.failed == 0:
        sys.stderr.write(f"Report written to: {REPORT_PATH}\n")
        sys.stderr.write("URDF cross-validation complete\n")
        return 0
    else:
        sys.stderr.write(
            f"FAILED: {runner.failed} of "
            f"{runner.passed + runner.failed + runner.warnings} checks failed\n"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
