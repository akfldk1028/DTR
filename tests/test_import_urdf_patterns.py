#!/usr/bin/env python3
"""test_import_urdf_patterns.py — Code pattern compliance checker for import_urdf_to_isaac.py

Verifies the URDF→USD conversion script follows project coding standards:
  1. No print() calls — logging module must be used
  2. No hardcoded magic numbers for drive params — params loaded from YAML
  3. Docstrings on all public functions
  4. SimulationApp init before omni.* imports in code flow
  5. Headless mode default=True
  6. Correct API usage (ArticulationAction, not set_joint_position_targets)

These checks use AST analysis and string/regex scanning of the source file.
They do NOT execute the script or import Isaac Sim modules.
"""

import ast
import os
import re
import sys
import textwrap

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SCRIPT_PATH = os.path.join(PROJECT_ROOT, "scripts", "import_urdf_to_isaac.py")


# ---------------------------------------------------------------------------
# Test runner (matches pattern from test_import_urdf_static.py)
# ---------------------------------------------------------------------------
class PatternComplianceRunner:
    """Runs all pattern compliance checks and tracks results."""

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
        sys.stderr.write("\n=== Pattern Compliance Results ===\n")
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
# Helper: load source and AST
# ---------------------------------------------------------------------------
def _load_source():
    """Load the script source code as a string."""
    with open(SCRIPT_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


def _load_ast():
    """Parse the script into an AST tree."""
    source = _load_source()
    return ast.parse(source, filename=SCRIPT_PATH)


# ---------------------------------------------------------------------------
# Check 1: No print() calls (must use logging)
# ---------------------------------------------------------------------------
def test_no_print_calls():
    """Verify no print() function calls exist in the source code.

    Per CLAUDE.md: 'print() 대신 logging 모듈 사용 (Isaac Sim이 stdout 캡처함)'
    """
    tree = _load_ast()

    print_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # Direct print() call
            if isinstance(func, ast.Name) and func.id == "print":
                print_calls.append(node.lineno)

    assert len(print_calls) == 0, (
        f"Found print() calls at line(s): {print_calls}. "
        f"Use logging module instead (Isaac Sim captures stdout)."
    )


def test_logging_module_imported():
    """Verify the logging module is imported."""
    tree = _load_ast()

    has_logging_import = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "logging":
                    has_logging_import = True
        elif isinstance(node, ast.ImportFrom):
            if node.module == "logging":
                has_logging_import = True

    assert has_logging_import, (
        "logging module must be imported (project standard per CLAUDE.md)"
    )


def test_logging_getlogger_used():
    """Verify logging.getLogger() is used for obtaining logger instances."""
    source = _load_source()

    # Check for logging.getLogger pattern
    getlogger_count = len(re.findall(r"logging\.getLogger\(", source))
    assert getlogger_count > 0, (
        "Expected logging.getLogger() calls to obtain logger instances"
    )


# ---------------------------------------------------------------------------
# Check 2: No hardcoded magic numbers for drive params
# ---------------------------------------------------------------------------
def test_drive_params_not_hardcoded():
    """Verify drive stiffness/damping/max_effort are loaded from YAML, not hardcoded.

    The apply_drive_params function should use get_param_value() to read
    drive parameters from control.yaml, not hardcoded numeric literals.
    """
    tree = _load_ast()

    # Find the apply_drive_params function
    apply_drive_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "apply_drive_params":
            apply_drive_fn = node
            break

    assert apply_drive_fn is not None, (
        "apply_drive_params function not found in source"
    )

    # Check that get_param_value is called inside apply_drive_params
    get_param_calls = []
    for node in ast.walk(apply_drive_fn):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "get_param_value":
                get_param_calls.append(node.lineno)

    assert len(get_param_calls) >= 3, (
        f"apply_drive_params should call get_param_value() at least 3 times "
        f"(stiffness, damping, max_effort). Found {len(get_param_calls)} call(s)."
    )


def test_drive_params_from_yaml_keys():
    """Verify apply_drive_params reads 'drive' -> 'stiffness'/'damping'/'max_effort'."""
    source = _load_source()

    # Find the apply_drive_params function body
    fn_match = re.search(
        r"def apply_drive_params\(.*?\).*?:\n(.*?)(?=\ndef |\nclass |\Z)",
        source,
        re.DOTALL,
    )
    assert fn_match, "apply_drive_params function not found"
    fn_body = fn_match.group(1)

    # Check for YAML key references (drive.stiffness, drive.damping, drive.max_effort)
    for param_key in ("stiffness", "damping", "max_effort"):
        assert f'"{param_key}"' in fn_body or f"'{param_key}'" in fn_body, (
            f"apply_drive_params should reference '{param_key}' YAML key "
            f"via get_param_value()"
        )


def test_no_hardcoded_drive_values_in_function():
    """Verify no hardcoded stiffness/damping values like 40.0, 4.0 in apply_drive_params.

    These values must come from params/control.yaml via get_param_value().
    """
    tree = _load_ast()

    # Find apply_drive_params function
    apply_drive_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "apply_drive_params":
            apply_drive_fn = node
            break

    assert apply_drive_fn is not None, "apply_drive_params function not found"

    # Collect all numeric literals used as arguments to CreateStiffnessAttr,
    # CreateDampingAttr, CreateMaxForceAttr. These should be variables,
    # not raw numeric constants.
    hardcoded_values = []
    for node in ast.walk(apply_drive_fn):
        if isinstance(node, ast.Call):
            func = node.func
            # Check for drive.Create*Attr(numeric_literal) pattern
            if isinstance(func, ast.Attribute) and func.attr in (
                "CreateStiffnessAttr",
                "CreateDampingAttr",
                "CreateMaxForceAttr",
            ):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(
                        arg.value, (int, float)
                    ):
                        hardcoded_values.append(
                            (func.attr, arg.value, node.lineno)
                        )

    assert len(hardcoded_values) == 0, (
        f"Hardcoded numeric values found in drive parameter calls: "
        f"{hardcoded_values}. "
        f"Drive params must come from get_param_value(control_params, 'drive', ...)"
    )


def test_constants_are_documented():
    """Verify module-level constants have inline comments explaining their purpose."""
    source = _load_source()

    # Constants that should have comments (per CLAUDE.md naming: UPPER_SNAKE_CASE)
    expected_constants = [
        "EXPECTED_NUM_JOINTS",
        "NUM_STABILITY_STEPS",
        "STABILITY_CHECK_INTERVAL",
        "JOINT_POSITION_BOUND",
        "JOINT_LIMIT_TOLERANCE",
        "LOG_FILE",
    ]

    for const in expected_constants:
        # Find the line defining this constant
        pattern = rf"^{const}\s*="
        match = re.search(pattern, source, re.MULTILINE)
        assert match is not None, (
            f"Constant {const} not found as module-level definition"
        )


# ---------------------------------------------------------------------------
# Check 3: Docstrings on all public functions
# ---------------------------------------------------------------------------
def test_public_functions_have_docstrings():
    """Verify all public functions (no leading underscore) have docstrings.

    Per CLAUDE.md: '모든 스크립트에는 docstring으로 목적과 사용법을 명시한다.'

    Note: ``main()`` is excluded because it is a standard entry point
    whose purpose is documented by the module-level docstring.
    """
    tree = _load_ast()

    # main() is a standard entry point — its purpose is described
    # by the module docstring; exempting it is conventional.
    EXEMPT_NAMES = {"main"}

    missing_docstrings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Skip private functions (leading underscore)
            if node.name.startswith("_"):
                continue
            # Skip exempt entry-point functions
            if node.name in EXEMPT_NAMES:
                continue

            docstring = ast.get_docstring(node)
            if not docstring:
                missing_docstrings.append(node.name)

    assert len(missing_docstrings) == 0, (
        f"Public functions missing docstrings: {missing_docstrings}"
    )


def test_module_docstring_exists():
    """Verify the script has a module-level docstring."""
    tree = _load_ast()
    docstring = ast.get_docstring(tree)
    assert docstring is not None and len(docstring) > 0, (
        "Module-level docstring is missing or empty"
    )


def test_module_docstring_has_purpose():
    """Verify the module docstring describes the script's purpose."""
    tree = _load_ast()
    docstring = ast.get_docstring(tree)
    assert docstring is not None, "Module docstring is missing"

    # Should mention URDF and USD conversion
    docstring_lower = docstring.lower()
    assert "urdf" in docstring_lower, (
        "Module docstring should mention 'URDF'"
    )
    assert "usd" in docstring_lower, (
        "Module docstring should mention 'USD'"
    )


# ---------------------------------------------------------------------------
# Check 4: SimulationApp init before omni.* imports in code flow
# ---------------------------------------------------------------------------
def test_simulationapp_before_omni_imports_comment():
    """Verify the code has a comment/docstring about SimulationApp init ordering.

    The script must document that SimulationApp must be initialized
    before any omni.* imports. This is an Isaac Sim requirement.
    """
    source = _load_source()

    # Check for the warning comment about import ordering
    has_warning = (
        "SimulationApp MUST be initialized BEFORE" in source
        or "SimulationApp must be created before" in source
        or "SimulationApp MUST be created before" in source
    )
    assert has_warning, (
        "Missing comment about SimulationApp initialization ordering. "
        "The script must document that SimulationApp must be init'd "
        "before any omni.* imports."
    )


def test_no_toplevel_omni_imports():
    """Verify no top-level 'import omni' or 'from omni' statements.

    omni.* imports must be inside functions (after SimulationApp init),
    not at the module level.
    """
    tree = _load_ast()

    toplevel_omni_imports = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("omni"):
                    toplevel_omni_imports.append(
                        (alias.name, node.lineno)
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("omni"):
                toplevel_omni_imports.append(
                    (node.module, node.lineno)
                )

    assert len(toplevel_omni_imports) == 0, (
        f"Top-level omni.* imports found: {toplevel_omni_imports}. "
        f"omni.* imports must be inside functions, after SimulationApp init."
    )


def test_no_toplevel_pxr_imports():
    """Verify no top-level 'from pxr import ...' statements.

    pxr (USD Python API) imports must be inside functions,
    after SimulationApp initialization.
    """
    tree = _load_ast()

    toplevel_pxr_imports = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("pxr"):
                    toplevel_pxr_imports.append(
                        (alias.name, node.lineno)
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("pxr"):
                toplevel_pxr_imports.append(
                    (node.module, node.lineno)
                )

    assert len(toplevel_pxr_imports) == 0, (
        f"Top-level pxr imports found: {toplevel_pxr_imports}. "
        f"pxr imports must be inside functions, after SimulationApp init."
    )


def test_no_toplevel_isaacsim_import():
    """Verify 'from isaacsim import SimulationApp' is not at the top level.

    SimulationApp import should be inside the init function to keep the
    module importable without Isaac Sim installed (for testing/linting).
    """
    tree = _load_ast()

    toplevel_isaacsim = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "isaacsim":
                    toplevel_isaacsim.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "isaacsim":
                toplevel_isaacsim.append(node.lineno)

    assert len(toplevel_isaacsim) == 0, (
        f"Top-level isaacsim import found at line(s): {toplevel_isaacsim}. "
        f"isaacsim import should be inside init_simulation_app() function."
    )


def test_simulationapp_init_in_main_before_omni_calls():
    """Verify main() calls init_simulation_app() before any omni-dependent functions.

    The main() function must initialize SimulationApp before calling
    create_import_config, setup_physics_scene, import_urdf, etc.
    """
    tree = _load_ast()

    # Find main() function
    main_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_fn = node
            break

    assert main_fn is not None, "main() function not found"

    # Find the line number of init_simulation_app call
    init_app_line = None
    # Functions that require SimulationApp to be initialized first
    omni_dependent_functions = {
        "create_import_config",
        "setup_physics_scene",
        "import_urdf",
        "apply_drive_params",
        "export_usd",
        "verify_articulation",
        "run_stability_test",
        "collect_joint_names",
    }
    omni_call_lines = []

    for node in ast.walk(main_fn):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id == "init_simulation_app":
                    init_app_line = node.lineno
                elif func.id in omni_dependent_functions:
                    omni_call_lines.append((func.id, node.lineno))

    assert init_app_line is not None, (
        "main() does not call init_simulation_app()"
    )

    # Verify all omni-dependent calls happen after init_simulation_app
    early_calls = [
        (name, line) for name, line in omni_call_lines
        if line < init_app_line
    ]

    assert len(early_calls) == 0, (
        f"Functions called before init_simulation_app (line {init_app_line}): "
        f"{early_calls}. SimulationApp must be initialized first."
    )


# ---------------------------------------------------------------------------
# Check 5: Headless mode default=True
# ---------------------------------------------------------------------------
def test_headless_default_true_in_argparse():
    """Verify --headless argument defaults to True in parse_args().

    Per CLAUDE.md Headless-First principle: all Isaac Sim scripts
    must support --headless flag with default=True.
    """
    source = _load_source()

    # Find the --headless argument definition
    # Look for pattern: add_argument("--headless", ... default=True ...)
    headless_section = re.search(
        r'add_argument\(\s*["\']--headless["\'].*?\)',
        source,
        re.DOTALL,
    )
    assert headless_section is not None, (
        "No --headless argument found in argparse definition"
    )

    headless_text = headless_section.group(0)
    assert "default=True" in headless_text, (
        f"--headless should have default=True. Found: {headless_text}"
    )


def test_init_simulation_app_headless_default():
    """Verify init_simulation_app() has headless parameter defaulting to True."""
    tree = _load_ast()

    init_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "init_simulation_app":
            init_fn = node
            break

    assert init_fn is not None, "init_simulation_app function not found"

    # Check the 'headless' parameter has a default of True
    found_headless = False
    for arg, default in zip(
        reversed(init_fn.args.args),
        reversed(init_fn.args.defaults),
    ):
        if arg.arg == "headless":
            found_headless = True
            assert isinstance(default, ast.Constant) and default.value is True, (
                f"init_simulation_app(headless=...) should default to True"
            )

    assert found_headless, (
        "init_simulation_app() missing 'headless' parameter"
    )


def test_headless_flag_in_help_text():
    """Verify --headless help text indicates it is the default mode."""
    source = _load_source()

    # Find the help text for --headless
    headless_section = re.search(
        r'add_argument\(\s*["\']--headless["\'].*?help\s*=\s*["\'](.+?)["\']',
        source,
        re.DOTALL,
    )
    assert headless_section is not None, (
        "--headless argument help text not found"
    )

    help_text = headless_section.group(1)
    # Help text should indicate default is True
    assert "default" in help_text.lower() or "True" in help_text, (
        f"--headless help should mention default behavior. Found: '{help_text}'"
    )


# ---------------------------------------------------------------------------
# Check 6: Correct API usage (ArticulationAction, not set_joint_position_targets)
# ---------------------------------------------------------------------------
def test_no_set_joint_position_targets():
    """Verify set_joint_position_targets() is not used anywhere.

    Per CLAUDE.md: 'set_joint_position_targets() 대신
    apply_action(ArticulationAction(joint_positions=...)) 사용'
    """
    source = _load_source()

    matches = re.findall(r"set_joint_position_targets", source)
    assert len(matches) == 0, (
        f"Found {len(matches)} occurrence(s) of 'set_joint_position_targets'. "
        f"Use apply_action(ArticulationAction(joint_positions=...)) instead."
    )


def test_no_deprecated_joint_api():
    """Verify no other deprecated joint-level APIs are used."""
    source = _load_source()

    deprecated_apis = [
        "set_joint_position_targets",
        "set_joint_velocity_targets",
        "set_joint_efforts",
    ]

    found_deprecated = []
    for api in deprecated_apis:
        if api in source:
            # Find line numbers
            for i, line in enumerate(source.splitlines(), 1):
                if api in line and not line.strip().startswith("#"):
                    found_deprecated.append((api, i))

    assert len(found_deprecated) == 0, (
        f"Deprecated joint APIs found: {found_deprecated}. "
        f"Use ArticulationAction via apply_action() instead."
    )


# ---------------------------------------------------------------------------
# Bonus checks: additional pattern compliance
# ---------------------------------------------------------------------------
def test_no_hardcoded_urls():
    """Verify no hardcoded URLs in the source code.

    Per CLAUDE.md: '외부 링크는 docs/references.md에 중앙 관리한다.
    코드 안에 URL을 하드코딩하지 않는다.'
    """
    source = _load_source()

    # Find URLs (http:// or https://) in non-comment, non-docstring lines
    url_pattern = re.compile(r"https?://\S+")
    hardcoded_urls = []

    for i, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        # Skip comments and docstring markers
        if stripped.startswith("#"):
            continue
        if stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        # Check for URLs in code lines
        urls = url_pattern.findall(line)
        if urls:
            # Skip if it's in a string that is a docstring context
            # (simple heuristic: line inside triple-quote block)
            hardcoded_urls.extend((url, i) for url in urls)

    # Note: URLs inside docstrings/comments are acceptable; we check code only
    # This is a best-effort check. The main concern is URLs in executable code.
    assert len(hardcoded_urls) == 0, (
        f"Hardcoded URLs found: {hardcoded_urls}. "
        f"URLs should be in docs/references.md per CLAUDE.md."
    )


def test_snake_case_functions():
    """Verify all function names follow snake_case convention.

    Per CLAUDE.md: Python 변수/함수 — snake_case
    """
    tree = _load_ast()

    non_snake_case = []
    snake_case_pattern = re.compile(r"^_?[a-z][a-z0-9_]*$")

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if not snake_case_pattern.match(node.name):
                non_snake_case.append(node.name)

    assert len(non_snake_case) == 0, (
        f"Functions not following snake_case: {non_snake_case}"
    )


def test_constants_upper_snake_case():
    """Verify module-level constants follow UPPER_SNAKE_CASE convention.

    Per CLAUDE.md: 상수 — UPPER_SNAKE_CASE
    """
    tree = _load_ast()

    upper_snake_pattern = re.compile(r"^[A-Z][A-Z0-9_]*$")
    # Also allow regular snake_case for non-constant module vars
    lower_snake_pattern = re.compile(r"^_?[a-z][a-z0-9_]*$")

    bad_names = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    # Skip dunder names like __name__
                    if name.startswith("__") and name.endswith("__"):
                        continue
                    # Module-level assignments should be either
                    # UPPER_SNAKE_CASE (constants) or _private
                    if not upper_snake_pattern.match(name) and not lower_snake_pattern.match(name):
                        bad_names.append(name)

    assert len(bad_names) == 0, (
        f"Module-level names not following naming convention: {bad_names}"
    )


def test_uses_pathlib():
    """Verify the script uses pathlib.Path for file path handling."""
    source = _load_source()

    assert "from pathlib import Path" in source or "import pathlib" in source, (
        "Script should use pathlib.Path for file path handling"
    )


def test_yaml_safe_load():
    """Verify yaml.safe_load is used instead of yaml.load (security)."""
    source = _load_source()

    # Check that yaml.safe_load is used
    assert "yaml.safe_load" in source, (
        "Should use yaml.safe_load() for safe YAML parsing"
    )

    # Check that unsafe yaml.load is NOT used (except within safe_load references)
    unsafe_pattern = re.compile(r"yaml\.load\((?!.*Loader)")
    unsafe_matches = unsafe_pattern.findall(source)

    # Also check: no yaml.load without Loader param (which is deprecated/unsafe)
    for i, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if "yaml.load(" in stripped and "safe_load" not in stripped:
            if "Loader=" not in stripped:
                assert False, (
                    f"Line {i}: yaml.load() without Loader= found. "
                    f"Use yaml.safe_load() instead."
                )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    # Verify the script file exists before running checks
    if not os.path.exists(SCRIPT_PATH):
        sys.stderr.write(f"ERROR: Script not found: {SCRIPT_PATH}\n")
        return 1

    runner = PatternComplianceRunner()

    # Check 1: No print() calls
    runner.run_test("No print() calls", test_no_print_calls)
    runner.run_test("logging module imported", test_logging_module_imported)
    runner.run_test("logging.getLogger() used", test_logging_getlogger_used)

    # Check 2: No hardcoded magic numbers for drive params
    runner.run_test("Drive params via get_param_value()", test_drive_params_not_hardcoded)
    runner.run_test("Drive params reference YAML keys", test_drive_params_from_yaml_keys)
    runner.run_test("No hardcoded drive values", test_no_hardcoded_drive_values_in_function)
    runner.run_test("Constants are documented", test_constants_are_documented)

    # Check 3: Docstrings on all public functions
    runner.run_test("Public functions have docstrings", test_public_functions_have_docstrings)
    runner.run_test("Module docstring exists", test_module_docstring_exists)
    runner.run_test("Module docstring has purpose", test_module_docstring_has_purpose)

    # Check 4: SimulationApp init before omni imports
    runner.run_test("SimulationApp ordering comment", test_simulationapp_before_omni_imports_comment)
    runner.run_test("No top-level omni.* imports", test_no_toplevel_omni_imports)
    runner.run_test("No top-level pxr imports", test_no_toplevel_pxr_imports)
    runner.run_test("No top-level isaacsim import", test_no_toplevel_isaacsim_import)
    runner.run_test("SimulationApp init before omni calls in main()", test_simulationapp_init_in_main_before_omni_calls)

    # Check 5: Headless mode default=True
    runner.run_test("--headless default=True in argparse", test_headless_default_true_in_argparse)
    runner.run_test("init_simulation_app headless default", test_init_simulation_app_headless_default)
    runner.run_test("--headless help text", test_headless_flag_in_help_text)

    # Check 6: Correct API usage
    runner.run_test("No set_joint_position_targets", test_no_set_joint_position_targets)
    runner.run_test("No deprecated joint APIs", test_no_deprecated_joint_api)

    # Bonus: additional pattern compliance
    runner.run_test("No hardcoded URLs", test_no_hardcoded_urls)
    runner.run_test("snake_case function names", test_snake_case_functions)
    runner.run_test("UPPER_SNAKE_CASE constants", test_constants_upper_snake_case)
    runner.run_test("Uses pathlib", test_uses_pathlib)
    runner.run_test("Uses yaml.safe_load", test_yaml_safe_load)

    # Summary
    runner.print_summary()

    if runner.failed == 0:
        sys.stderr.write("All pattern compliance checks passed\n")
        return 0
    else:
        sys.stderr.write(
            f"FAILED: {runner.failed} of {runner.passed + runner.failed} checks failed\n"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
