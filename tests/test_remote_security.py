import ast
from pathlib import Path


REMOTE_HELPER = Path(__file__).parents[1] / "scripts" / "_kgb_remote.py"
FULLSCALE_RUNNER = Path(__file__).parents[1] / "scripts" / "run_fullscale_diff_test.py"
PROJECT_ROOT = Path(__file__).parents[1]


def test_remote_helper_contains_no_literal_credentials():
    tree = ast.parse(REMOTE_HELPER.read_text(encoding="utf-8"))
    sensitive_names = {
        "PASS",
        "PASSWORD",
        "TOKEN",
        "SECRET",
        "API_KEY",
        "ACCESS_TOKEN",
    }

    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id.upper() in sensitive_names:
                if value.value:
                    violations.append((target.id, node.lineno))

    assert violations == []


def test_remote_helper_requires_host_key_verification():
    source = REMOTE_HELPER.read_text(encoding="utf-8")
    assert "RejectPolicy" in source
    assert "AutoAddPolicy" not in source


def test_gpu_reset_requires_explicit_operator_opt_in():
    tree = ast.parse(FULLSCALE_RUNNER.read_text(encoding="utf-8"))
    reset_functions = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        function_source = ast.get_source_segment(
            FULLSCALE_RUNNER.read_text(encoding="utf-8"), node
        ) or ""
        if '"--gpu-reset"' in function_source:
            reset_functions.append((node.name, function_source))

    assert len(reset_functions) == 1
    name, source = reset_functions[0]
    assert name == "_maybe_reset_gpu"
    assert 'os.environ.get("MUTAKERNEL_ALLOW_GPU_RESET") != "1"' in source


def test_maintained_python_has_no_literal_secret_assignment():
    paths = [PROJECT_ROOT / "config.py"]
    paths.extend((PROJECT_ROOT / "src").rglob("*.py"))
    paths.extend((PROJECT_ROOT / "scripts").glob("*.py"))

    def sensitive(name: str) -> bool:
        normalized = name.upper()
        return (
            "PASSWORD" in normalized
            or "SECRET" in normalized
            or normalized in {
                "PASSWORD",
                "TOKEN",
                "API_KEY",
                "ACCESS_TOKEN",
                "AUTH_TOKEN",
                "PRIVATE_KEY",
            }
            or normalized.endswith("_API_KEY")
            or normalized.endswith("_ACCESS_TOKEN")
            or normalized.endswith("_AUTH_TOKEN")
        )

    violations = []
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value:
                    for target in targets:
                        if isinstance(target, ast.Name) and sensitive(target.id):
                            violations.append((path.relative_to(PROJECT_ROOT).as_posix(), node.lineno, target.id))
            elif isinstance(node, ast.keyword) and node.arg and sensitive(node.arg):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) and node.value.value:
                    violations.append((path.relative_to(PROJECT_ROOT).as_posix(), node.lineno, node.arg))

    assert violations == []
