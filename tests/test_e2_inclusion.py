"""CPU-only, torch-free tests for the E2 inclusion rules and the C4 static
contract-draft extraction (both must be reproducible pure functions)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.extract_contracts import (
    collect_frame_task_keys,
    extract_triton_tensor_hints,
)
from scripts.reconcile_corpora import (
    build_task_subject_frame,
    detect_language,
    inclusion_check,
    stable_id,
    task_resolvable,
)


# ---------------------------------------------------------------------------
# Language evidence (rule R4)
# ---------------------------------------------------------------------------

def test_detect_language_cuda_markers():
    assert detect_language("src = '''__global__ void k() {}'''") == "cuda"
    assert detect_language("from torch.utils.cpp_extension import load_inline") == "cuda"
    assert detect_language("mod = load_inline(name='x', cuda_sources=[s])") == "cuda"


def test_detect_language_triton_beats_cuda():
    source = "import triton\n@triton.jit\ndef k(): pass\nx.cuda()\n"
    assert detect_language(source) == "triton"


def test_detect_language_python_only():
    source = (
        "import torch\nclass ModelNew(torch.nn.Module):\n"
        "    def forward(self, x):\n        return x.cuda().half()\n"
    )
    assert detect_language(source) == "python_only"


# ---------------------------------------------------------------------------
# Task resolution (rule R2)
# ---------------------------------------------------------------------------

def test_task_resolvable_kernelbench_bounds():
    assert task_resolvable("C2", "KB_L1_P1")
    assert task_resolvable("C3", "KB_L1_P100")
    assert task_resolvable("C5", "KB_L3_P50")
    assert not task_resolvable("C2", "KB_L1_P101")
    assert not task_resolvable("C5", "KB_L3_P51")
    assert not task_resolvable("C2", "KB_L4_P1")
    assert not task_resolvable("C2", "garbage")


def test_task_resolvable_tritonbench():
    assert task_resolvable("C4", "TBG_softmax_kernel")
    assert not task_resolvable("C4", "TBG_")
    assert not task_resolvable("C4", "KB_L1_P1")


# ---------------------------------------------------------------------------
# Rule precedence + accounting
# ---------------------------------------------------------------------------

CUDA_SRC = "s = '__global__ void k() {}'"


def test_inclusion_check_passes_clean_cuda_row():
    included, reason, detected = inclusion_check("C2", CUDA_SRC, "KB_L1_P1", False)
    assert included and reason is None and detected == "cuda"


def test_inclusion_check_first_failing_rule_wins():
    # duplicate (R3) is checked before language (R4)
    _, reason, _ = inclusion_check("C2", "print(1)", "KB_L1_P1", True)
    assert reason == "duplicate"
    _, reason, _ = inclusion_check("C2", "print(1)", "KB_L1_P1", False)
    assert reason == "language_mismatch"
    _, reason, _ = inclusion_check("C2", CUDA_SRC, "KB_L9_P1", False)
    assert reason == "task_unresolvable"


# ---------------------------------------------------------------------------
# C3 task-level subject frame (representative selection)
# ---------------------------------------------------------------------------

def _frame_row(task, source, accepted=True, included=True, index=0):
    return {
        "task_key": task,
        "stable_id": stable_id(source),
        "frame_id": f"C3-{index:05d}",
        "origin": f"test:{index}",
        "accepted": accepted,
        "included": included,
    }


def test_subject_frame_picks_min_stable_id_and_reports_exclusions(tmp_path):
    rows = [
        _frame_row("KB_L1_P1", "a = 1\n", index=0),
        _frame_row("KB_L1_P1", "b = 2\n", index=1),
        # task with candidates but none both accepted and included:
        _frame_row("KB_L1_P2", "c = 3\n", accepted=False, index=2),
        _frame_row("KB_L1_P2", "d = 4\n", accepted=True, included=False, index=3),
    ]
    summary = build_task_subject_frame("C3", rows, tmp_path)
    assert summary["tasks_collected"] == 2
    assert summary["tasks_included"] == 1
    assert summary["excluded_tasks"] == ["KB_L1_P2"]

    subject_rows = [
        json.loads(line)
        for line in (tmp_path / "C3_subject_frame.jsonl").read_text(
            encoding="utf-8").splitlines()
    ]
    assert len(subject_rows) == 1
    expected = min(stable_id("a = 1\n"), stable_id("b = 2\n"))
    assert subject_rows[0]["stable_id"] == expected
    assert subject_rows[0]["candidates_eligible"] == 2


# ---------------------------------------------------------------------------
# Contract extraction plumbing for C2-C5
# ---------------------------------------------------------------------------

def test_collect_frame_task_keys_respects_included_flag(tmp_path):
    frame = tmp_path / "frame.jsonl"
    rows = [
        {"corpus": "C2", "task_key": "KB_L1_P1", "included": True},
        {"corpus": "C2", "task_key": "KB_L1_P2", "included": False},
        {"corpus": "C5", "task_key": "KB_L1_P1", "included": True},
        {"corpus": "C4", "task_key": "TBG_softmax", "included": True},  # not KB
    ]
    frame.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    tasks = collect_frame_task_keys([frame])
    assert tasks == {"KB_L1_P1": ["C2", "C5"]}
    tasks_all = collect_frame_task_keys([frame], only_included=False)
    assert set(tasks_all) == {"KB_L1_P1", "KB_L1_P2"}


TRITON_TEST_FILE = '''
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x_ptr, out_ptr, n: tl.constexpr):
    pass

def softmax(x):
    return x

def test_softmax():
    results = {}
    x = torch.randn(64, 128, device="cuda")
    results["case_1"] = softmax(x)
    y = torch.rand((2, 3), dtype=torch.float16, device="cuda")
    results["case_2"] = softmax(y)
    return results

test_results = test_softmax()
'''


def test_extract_triton_tensor_hints_prefers_test_functions():
    hints, test_functions = extract_triton_tensor_hints(TRITON_TEST_FILE)
    assert test_functions == ["test_softmax"]
    constructors = {(h["constructor"], tuple(h["shape"])) for h in hints}
    assert ("randn", (64, 128)) in constructors
    assert ("rand", (2, 3)) in constructors
    float16 = [h for h in hints if h.get("dtype") == "float16"]
    assert len(float16) == 1


def test_extract_triton_tensor_hints_module_fallback():
    source = "import torch\nx = torch.zeros(4, 4)\n"
    hints, test_functions = extract_triton_tensor_hints(source)
    assert test_functions == []
    assert hints == [{"constructor": "zeros", "shape": [4, 4]}]
