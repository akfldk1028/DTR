#!/usr/bin/env python3
"""test_vla_patterns.py — Code pattern compliance checker for VLA extension

Verifies the VLA (Vision-Language-Action) module follows project coding standards:
  1. No print() calls — logging module must be used
  2. logging module imported and getLogger() used
  3. Docstrings on all public functions and classes
  4. PROJECT_ROOT pattern used for path setup
  5. Heavy imports (Isaac Sim, LeRobot, PyTorch) guarded with try/except
  6. VLAInference abstract class exists with predict() method
  7. DummyVLA inherits from VLAInference
  8. SmolVLAWrapper class exists

These checks use AST analysis and string/regex scanning of the source files.
They do NOT execute the scripts or import heavy dependencies.
"""

import ast
import os
import re
import sys

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

VLA_DIR = os.path.join(PROJECT_ROOT, "training", "vla")
INFERENCE_PATH = os.path.join(VLA_DIR, "inference.py")
EVAL_IN_SIM_PATH = os.path.join(VLA_DIR, "eval_in_sim.py")

# All VLA Python source files to check
VLA_PYTHON_FILES = [INFERENCE_PATH, EVAL_IN_SIM_PATH]


# ---------------------------------------------------------------------------
# Test runner (matches pattern from test_import_urdf_patterns.py)
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
        sys.stderr.write("\n=== VLA Pattern Compliance Results ===\n")
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
# Helper: load source and AST for a given file
# ---------------------------------------------------------------------------
def _load_source(filepath):
    """Load a source file as a string."""
    with open(filepath, "r", encoding="utf-8") as fh:
        return fh.read()


def _load_ast(filepath):
    """Parse a source file into an AST tree."""
    source = _load_source(filepath)
    return ast.parse(source, filename=filepath)


def _require_file(filepath):
    """Assert that a file exists; raise AssertionError if not."""
    basename = os.path.basename(filepath)
    assert os.path.isfile(filepath), (
        f"File not found: {basename} (expected at {filepath})"
    )


# ---------------------------------------------------------------------------
# Check 1: No print() calls (must use logging)
# ---------------------------------------------------------------------------
def test_no_print_calls_inference():
    """Verify no print() calls in inference.py.

    Per CLAUDE.md: 'print() 대신 logging 모듈 사용 (Isaac Sim이 stdout 캡처함)'
    """
    _require_file(INFERENCE_PATH)
    tree = _load_ast(INFERENCE_PATH)

    print_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "print":
                print_calls.append(node.lineno)

    assert len(print_calls) == 0, (
        f"Found print() calls in inference.py at line(s): {print_calls}. "
        f"Use logging module instead."
    )


def test_no_print_calls_eval_in_sim():
    """Verify no print() calls in eval_in_sim.py.

    Per CLAUDE.md: 'print() 대신 logging 모듈 사용 (Isaac Sim이 stdout 캡처함)'
    """
    _require_file(EVAL_IN_SIM_PATH)
    tree = _load_ast(EVAL_IN_SIM_PATH)

    print_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "print":
                print_calls.append(node.lineno)

    assert len(print_calls) == 0, (
        f"Found print() calls in eval_in_sim.py at line(s): {print_calls}. "
        f"Use logging module instead."
    )


# ---------------------------------------------------------------------------
# Check 2: logging module usage
# ---------------------------------------------------------------------------
def test_logging_imported_inference():
    """Verify the logging module is imported in inference.py."""
    _require_file(INFERENCE_PATH)
    tree = _load_ast(INFERENCE_PATH)

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
        "inference.py: logging module must be imported (project standard per CLAUDE.md)"
    )


def test_logging_imported_eval_in_sim():
    """Verify the logging module is imported in eval_in_sim.py."""
    _require_file(EVAL_IN_SIM_PATH)
    tree = _load_ast(EVAL_IN_SIM_PATH)

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
        "eval_in_sim.py: logging module must be imported (project standard per CLAUDE.md)"
    )


def test_logging_getlogger_inference():
    """Verify logging.getLogger() is used in inference.py."""
    _require_file(INFERENCE_PATH)
    source = _load_source(INFERENCE_PATH)

    getlogger_count = len(re.findall(r"logging\.getLogger\(", source))
    assert getlogger_count > 0, (
        "inference.py: Expected logging.getLogger() call to obtain logger instance"
    )


def test_logging_getlogger_eval_in_sim():
    """Verify logging.getLogger() is used in eval_in_sim.py."""
    _require_file(EVAL_IN_SIM_PATH)
    source = _load_source(EVAL_IN_SIM_PATH)

    getlogger_count = len(re.findall(r"logging\.getLogger\(", source))
    assert getlogger_count > 0, (
        "eval_in_sim.py: Expected logging.getLogger() call to obtain logger instance"
    )


# ---------------------------------------------------------------------------
# Check 3: Docstrings on all public functions and classes
# ---------------------------------------------------------------------------
def test_public_functions_have_docstrings_inference():
    """Verify all public functions in inference.py have docstrings.

    Per CLAUDE.md: '모든 스크립트에는 docstring으로 목적과 사용법을 명시한다.'
    """
    _require_file(INFERENCE_PATH)
    tree = _load_ast(INFERENCE_PATH)

    EXEMPT_NAMES = {"main"}

    missing_docstrings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("_"):
                continue
            if node.name in EXEMPT_NAMES:
                continue
            docstring = ast.get_docstring(node)
            if not docstring:
                missing_docstrings.append(node.name)

    assert len(missing_docstrings) == 0, (
        f"inference.py: Public functions missing docstrings: {missing_docstrings}"
    )


def test_public_functions_have_docstrings_eval_in_sim():
    """Verify all public functions in eval_in_sim.py have docstrings.

    Per CLAUDE.md: '모든 스크립트에는 docstring으로 목적과 사용법을 명시한다.'
    """
    _require_file(EVAL_IN_SIM_PATH)
    tree = _load_ast(EVAL_IN_SIM_PATH)

    EXEMPT_NAMES = {"main"}

    missing_docstrings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("_"):
                continue
            if node.name in EXEMPT_NAMES:
                continue
            docstring = ast.get_docstring(node)
            if not docstring:
                missing_docstrings.append(node.name)

    assert len(missing_docstrings) == 0, (
        f"eval_in_sim.py: Public functions missing docstrings: {missing_docstrings}"
    )


def test_classes_have_docstrings_inference():
    """Verify all classes in inference.py have docstrings."""
    _require_file(INFERENCE_PATH)
    tree = _load_ast(INFERENCE_PATH)

    missing_docstrings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            docstring = ast.get_docstring(node)
            if not docstring:
                missing_docstrings.append(node.name)

    assert len(missing_docstrings) == 0, (
        f"inference.py: Classes missing docstrings: {missing_docstrings}"
    )


def test_module_docstring_inference():
    """Verify inference.py has a module-level docstring."""
    _require_file(INFERENCE_PATH)
    tree = _load_ast(INFERENCE_PATH)
    docstring = ast.get_docstring(tree)
    assert docstring is not None and len(docstring) > 0, (
        "inference.py: Module-level docstring is missing or empty"
    )


def test_module_docstring_eval_in_sim():
    """Verify eval_in_sim.py has a module-level docstring."""
    _require_file(EVAL_IN_SIM_PATH)
    tree = _load_ast(EVAL_IN_SIM_PATH)
    docstring = ast.get_docstring(tree)
    assert docstring is not None and len(docstring) > 0, (
        "eval_in_sim.py: Module-level docstring is missing or empty"
    )


# ---------------------------------------------------------------------------
# Check 4: PROJECT_ROOT pattern used
# ---------------------------------------------------------------------------
def test_project_root_pattern_inference():
    """Verify inference.py uses the PROJECT_ROOT path setup pattern.

    Project convention: scripts derive PROJECT_ROOT from __file__ for
    portable path resolution.
    """
    _require_file(INFERENCE_PATH)
    source = _load_source(INFERENCE_PATH)

    # Check for either PROJECT_ROOT assignment or os.path.dirname(__file__) pattern
    has_project_root = (
        "PROJECT_ROOT" in source
        or "os.path.dirname" in source
        or "Path(__file__)" in source
        or "pathlib" in source
    )
    assert has_project_root, (
        "inference.py: Expected PROJECT_ROOT path setup or "
        "os.path.dirname(__file__) pattern for portable path resolution"
    )


def test_project_root_pattern_eval_in_sim():
    """Verify eval_in_sim.py uses the PROJECT_ROOT path setup pattern.

    Project convention: scripts derive PROJECT_ROOT from __file__ for
    portable path resolution.
    """
    _require_file(EVAL_IN_SIM_PATH)
    source = _load_source(EVAL_IN_SIM_PATH)

    has_project_root = (
        "PROJECT_ROOT" in source
        or "os.path.dirname" in source
        or "Path(__file__)" in source
        or "pathlib" in source
    )
    assert has_project_root, (
        "eval_in_sim.py: Expected PROJECT_ROOT path setup or "
        "os.path.dirname(__file__) pattern for portable path resolution"
    )


# ---------------------------------------------------------------------------
# Check 5: Heavy imports guarded with try/except
# ---------------------------------------------------------------------------
def test_heavy_imports_guarded_inference():
    """Verify heavy imports in inference.py are guarded with try/except.

    Heavy dependencies (Isaac Sim, LeRobot, PyTorch) must be wrapped
    in try/except blocks so the module can be imported for testing/linting
    without these packages installed.
    """
    _require_file(INFERENCE_PATH)
    tree = _load_ast(INFERENCE_PATH)
    source = _load_source(INFERENCE_PATH)

    # Heavy module prefixes to check for
    heavy_modules = ["lerobot", "torch", "isaacsim", "omni", "pxr", "transformers"]

    # Find all imports of heavy modules
    heavy_imports_found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for heavy in heavy_modules:
                    if alias.name.startswith(heavy):
                        heavy_imports_found.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for heavy in heavy_modules:
                    if node.module.startswith(heavy):
                        heavy_imports_found.append((node.module, node.lineno))

    if not heavy_imports_found:
        # No heavy imports found — this is fine (they might be deferred)
        return

    # Check that heavy imports are inside try/except blocks (not at top level)
    unguarded = []
    for module_name, lineno in heavy_imports_found:
        # Check if the import is inside a Try node
        is_guarded = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for child in ast.walk(node):
                    if isinstance(child, (ast.Import, ast.ImportFrom)):
                        if hasattr(child, "lineno") and child.lineno == lineno:
                            is_guarded = True
                            break
                if is_guarded:
                    break
        if not is_guarded:
            unguarded.append((module_name, lineno))

    assert len(unguarded) == 0, (
        f"inference.py: Heavy imports not guarded with try/except: {unguarded}. "
        f"Heavy dependencies (Isaac Sim, LeRobot, PyTorch) must be "
        f"wrapped in try/except for safe import."
    )


def test_heavy_imports_guarded_eval_in_sim():
    """Verify heavy imports in eval_in_sim.py are guarded with try/except.

    Heavy dependencies (Isaac Sim, LeRobot, PyTorch) must be wrapped
    in try/except blocks so the module can be imported for testing/linting
    without these packages installed.
    """
    _require_file(EVAL_IN_SIM_PATH)
    tree = _load_ast(EVAL_IN_SIM_PATH)
    source = _load_source(EVAL_IN_SIM_PATH)

    heavy_modules = ["lerobot", "torch", "isaacsim", "omni", "pxr", "transformers"]

    heavy_imports_found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for heavy in heavy_modules:
                    if alias.name.startswith(heavy):
                        heavy_imports_found.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for heavy in heavy_modules:
                    if node.module.startswith(heavy):
                        heavy_imports_found.append((node.module, node.lineno))

    if not heavy_imports_found:
        return

    unguarded = []
    for module_name, lineno in heavy_imports_found:
        is_guarded = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for child in ast.walk(node):
                    if isinstance(child, (ast.Import, ast.ImportFrom)):
                        if hasattr(child, "lineno") and child.lineno == lineno:
                            is_guarded = True
                            break
                if is_guarded:
                    break
        if not is_guarded:
            unguarded.append((module_name, lineno))

    assert len(unguarded) == 0, (
        f"eval_in_sim.py: Heavy imports not guarded with try/except: {unguarded}. "
        f"Heavy dependencies (Isaac Sim, LeRobot, PyTorch) must be "
        f"wrapped in try/except for safe import."
    )


# ---------------------------------------------------------------------------
# Check 6: VLAInference abstract class with predict() method
# ---------------------------------------------------------------------------
def test_vlainference_class_exists():
    """Verify VLAInference abstract class exists in inference.py."""
    _require_file(INFERENCE_PATH)
    tree = _load_ast(INFERENCE_PATH)

    vla_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "VLAInference":
            vla_class = node
            break

    assert vla_class is not None, (
        "inference.py: VLAInference class not found. "
        "Expected abstract base class for VLA inference interface."
    )


def test_vlainference_has_predict_method():
    """Verify VLAInference class has a predict() method."""
    _require_file(INFERENCE_PATH)
    tree = _load_ast(INFERENCE_PATH)

    vla_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "VLAInference":
            vla_class = node
            break

    assert vla_class is not None, (
        "inference.py: VLAInference class not found"
    )

    # Find predict method in VLAInference
    has_predict = False
    for node in ast.iter_child_nodes(vla_class):
        if isinstance(node, ast.FunctionDef) and node.name == "predict":
            has_predict = True
            break

    assert has_predict, (
        "inference.py: VLAInference class must have a predict() method. "
        "This is the core inference contract: predict(instruction, image, state) -> action"
    )


def test_vlainference_predict_raises_not_implemented():
    """Verify VLAInference.predict() raises NotImplementedError (abstract pattern)."""
    _require_file(INFERENCE_PATH)
    tree = _load_ast(INFERENCE_PATH)

    vla_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "VLAInference":
            vla_class = node
            break

    assert vla_class is not None, "VLAInference class not found"

    predict_fn = None
    for node in ast.iter_child_nodes(vla_class):
        if isinstance(node, ast.FunctionDef) and node.name == "predict":
            predict_fn = node
            break

    assert predict_fn is not None, "VLAInference.predict() not found"

    # Check that the function body raises NotImplementedError
    source = _load_source(INFERENCE_PATH)
    has_not_implemented = False
    for child in ast.walk(predict_fn):
        if isinstance(child, ast.Raise):
            if child.exc is not None:
                # Check for raise NotImplementedError or raise NotImplementedError(...)
                if isinstance(child.exc, ast.Name) and child.exc.id == "NotImplementedError":
                    has_not_implemented = True
                elif isinstance(child.exc, ast.Call):
                    func = child.exc.func
                    if isinstance(func, ast.Name) and func.id == "NotImplementedError":
                        has_not_implemented = True

    assert has_not_implemented, (
        "inference.py: VLAInference.predict() must raise NotImplementedError. "
        "This enforces the abstract interface contract."
    )


def test_vlainference_predict_signature():
    """Verify VLAInference.predict() has the correct parameter signature.

    Expected: predict(self, instruction, image, state) -> action
    """
    _require_file(INFERENCE_PATH)
    tree = _load_ast(INFERENCE_PATH)

    vla_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "VLAInference":
            vla_class = node
            break

    assert vla_class is not None, "VLAInference class not found"

    predict_fn = None
    for node in ast.iter_child_nodes(vla_class):
        if isinstance(node, ast.FunctionDef) and node.name == "predict":
            predict_fn = node
            break

    assert predict_fn is not None, "VLAInference.predict() not found"

    # Check parameters: self, instruction, image, state
    param_names = [arg.arg for arg in predict_fn.args.args]

    assert "self" in param_names, (
        "VLAInference.predict() must be an instance method (has 'self')"
    )

    expected_params = ["instruction", "image", "state"]
    for param in expected_params:
        assert param in param_names, (
            f"VLAInference.predict() missing parameter '{param}'. "
            f"Expected signature: predict(self, instruction, image, state). "
            f"Found parameters: {param_names}"
        )


# ---------------------------------------------------------------------------
# Check 7: DummyVLA inherits from VLAInference
# ---------------------------------------------------------------------------
def test_dummyvla_class_exists():
    """Verify DummyVLA class exists in inference.py."""
    _require_file(INFERENCE_PATH)
    tree = _load_ast(INFERENCE_PATH)

    dummy_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DummyVLA":
            dummy_class = node
            break

    assert dummy_class is not None, (
        "inference.py: DummyVLA class not found. "
        "Expected zero-action baseline for pipeline verification."
    )


def test_dummyvla_inherits_vlainference():
    """Verify DummyVLA inherits from VLAInference."""
    _require_file(INFERENCE_PATH)
    tree = _load_ast(INFERENCE_PATH)

    dummy_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DummyVLA":
            dummy_class = node
            break

    assert dummy_class is not None, "DummyVLA class not found"

    # Check base classes
    base_names = []
    for base in dummy_class.bases:
        if isinstance(base, ast.Name):
            base_names.append(base.id)
        elif isinstance(base, ast.Attribute):
            base_names.append(base.attr)

    assert "VLAInference" in base_names, (
        f"inference.py: DummyVLA must inherit from VLAInference. "
        f"Found base classes: {base_names}"
    )


def test_dummyvla_has_predict():
    """Verify DummyVLA implements predict() method."""
    _require_file(INFERENCE_PATH)
    tree = _load_ast(INFERENCE_PATH)

    dummy_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DummyVLA":
            dummy_class = node
            break

    assert dummy_class is not None, "DummyVLA class not found"

    has_predict = False
    for node in ast.iter_child_nodes(dummy_class):
        if isinstance(node, ast.FunctionDef) and node.name == "predict":
            has_predict = True
            break

    assert has_predict, (
        "inference.py: DummyVLA must implement predict() method. "
        "This is the zero-action baseline for pipeline verification."
    )


# ---------------------------------------------------------------------------
# Check 8: SmolVLAWrapper class exists
# ---------------------------------------------------------------------------
def test_smolvlawrapper_class_exists():
    """Verify SmolVLAWrapper class exists in inference.py."""
    _require_file(INFERENCE_PATH)
    tree = _load_ast(INFERENCE_PATH)

    wrapper_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SmolVLAWrapper":
            wrapper_class = node
            break

    assert wrapper_class is not None, (
        "inference.py: SmolVLAWrapper class not found. "
        "Expected LeRobot SmolVLA policy wrapper."
    )


def test_smolvlawrapper_inherits_vlainference():
    """Verify SmolVLAWrapper inherits from VLAInference."""
    _require_file(INFERENCE_PATH)
    tree = _load_ast(INFERENCE_PATH)

    wrapper_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SmolVLAWrapper":
            wrapper_class = node
            break

    assert wrapper_class is not None, "SmolVLAWrapper class not found"

    base_names = []
    for base in wrapper_class.bases:
        if isinstance(base, ast.Name):
            base_names.append(base.id)
        elif isinstance(base, ast.Attribute):
            base_names.append(base.attr)

    assert "VLAInference" in base_names, (
        f"inference.py: SmolVLAWrapper must inherit from VLAInference. "
        f"Found base classes: {base_names}"
    )


def test_smolvlawrapper_has_predict():
    """Verify SmolVLAWrapper implements predict() method."""
    _require_file(INFERENCE_PATH)
    tree = _load_ast(INFERENCE_PATH)

    wrapper_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SmolVLAWrapper":
            wrapper_class = node
            break

    assert wrapper_class is not None, "SmolVLAWrapper class not found"

    has_predict = False
    for node in ast.iter_child_nodes(wrapper_class):
        if isinstance(node, ast.FunctionDef) and node.name == "predict":
            has_predict = True
            break

    assert has_predict, (
        "inference.py: SmolVLAWrapper must implement predict() method."
    )


# ---------------------------------------------------------------------------
# Bonus checks: additional pattern compliance
# ---------------------------------------------------------------------------
def test_snake_case_functions_inference():
    """Verify all function names in inference.py follow snake_case convention.

    Per CLAUDE.md: Python 변수/함수 — snake_case
    """
    _require_file(INFERENCE_PATH)
    tree = _load_ast(INFERENCE_PATH)

    non_snake_case = []
    snake_case_pattern = re.compile(r"^_?[a-z][a-z0-9_]*$")

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if not snake_case_pattern.match(node.name):
                # Allow __init__, __str__, etc.
                if not (node.name.startswith("__") and node.name.endswith("__")):
                    non_snake_case.append(node.name)

    assert len(non_snake_case) == 0, (
        f"inference.py: Functions not following snake_case: {non_snake_case}"
    )


def test_snake_case_functions_eval_in_sim():
    """Verify all function names in eval_in_sim.py follow snake_case convention.

    Per CLAUDE.md: Python 변수/함수 — snake_case
    """
    _require_file(EVAL_IN_SIM_PATH)
    tree = _load_ast(EVAL_IN_SIM_PATH)

    non_snake_case = []
    snake_case_pattern = re.compile(r"^_?[a-z][a-z0-9_]*$")

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if not snake_case_pattern.match(node.name):
                if not (node.name.startswith("__") and node.name.endswith("__")):
                    non_snake_case.append(node.name)

    assert len(non_snake_case) == 0, (
        f"eval_in_sim.py: Functions not following snake_case: {non_snake_case}"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    runner = PatternComplianceRunner()

    # Check 1: No print() calls
    runner.run_test("No print() in inference.py", test_no_print_calls_inference)
    runner.run_test("No print() in eval_in_sim.py", test_no_print_calls_eval_in_sim)

    # Check 2: logging module usage
    runner.run_test("logging imported in inference.py", test_logging_imported_inference)
    runner.run_test("logging imported in eval_in_sim.py", test_logging_imported_eval_in_sim)
    runner.run_test("logging.getLogger() in inference.py", test_logging_getlogger_inference)
    runner.run_test("logging.getLogger() in eval_in_sim.py", test_logging_getlogger_eval_in_sim)

    # Check 3: Docstrings
    runner.run_test("Public functions have docstrings (inference.py)", test_public_functions_have_docstrings_inference)
    runner.run_test("Public functions have docstrings (eval_in_sim.py)", test_public_functions_have_docstrings_eval_in_sim)
    runner.run_test("Classes have docstrings (inference.py)", test_classes_have_docstrings_inference)
    runner.run_test("Module docstring (inference.py)", test_module_docstring_inference)
    runner.run_test("Module docstring (eval_in_sim.py)", test_module_docstring_eval_in_sim)

    # Check 4: PROJECT_ROOT pattern
    runner.run_test("PROJECT_ROOT pattern (inference.py)", test_project_root_pattern_inference)
    runner.run_test("PROJECT_ROOT pattern (eval_in_sim.py)", test_project_root_pattern_eval_in_sim)

    # Check 5: Heavy imports guarded
    runner.run_test("Heavy imports guarded (inference.py)", test_heavy_imports_guarded_inference)
    runner.run_test("Heavy imports guarded (eval_in_sim.py)", test_heavy_imports_guarded_eval_in_sim)

    # Check 6: VLAInference abstract class
    runner.run_test("VLAInference class exists", test_vlainference_class_exists)
    runner.run_test("VLAInference has predict()", test_vlainference_has_predict_method)
    runner.run_test("VLAInference.predict() raises NotImplementedError", test_vlainference_predict_raises_not_implemented)
    runner.run_test("VLAInference.predict() signature", test_vlainference_predict_signature)

    # Check 7: DummyVLA class
    runner.run_test("DummyVLA class exists", test_dummyvla_class_exists)
    runner.run_test("DummyVLA inherits VLAInference", test_dummyvla_inherits_vlainference)
    runner.run_test("DummyVLA has predict()", test_dummyvla_has_predict)

    # Check 8: SmolVLAWrapper class
    runner.run_test("SmolVLAWrapper class exists", test_smolvlawrapper_class_exists)
    runner.run_test("SmolVLAWrapper inherits VLAInference", test_smolvlawrapper_inherits_vlainference)
    runner.run_test("SmolVLAWrapper has predict()", test_smolvlawrapper_has_predict)

    # Bonus: naming conventions
    runner.run_test("snake_case functions (inference.py)", test_snake_case_functions_inference)
    runner.run_test("snake_case functions (eval_in_sim.py)", test_snake_case_functions_eval_in_sim)

    # Summary
    runner.print_summary()

    if runner.failed == 0:
        sys.stderr.write("All VLA pattern compliance checks passed\n")
        return 0
    else:
        sys.stderr.write(
            f"FAILED: {runner.failed} of {runner.passed + runner.failed} checks failed\n"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
