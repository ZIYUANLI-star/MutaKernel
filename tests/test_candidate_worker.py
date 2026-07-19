"""CPU tests for the mutation-independent FSE candidate worker."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


WORKER_PATH = Path(__file__).parents[1] / "scripts" / "_candidate_worker.py"


def _load_worker() -> Any:
    spec = importlib.util.spec_from_file_location("candidate_worker_test", WORKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _config(reference: Path, candidate: Path, **case_updates: Any) -> dict[str, Any]:
    case = {
        "test_id": "1" * 64,
        "policy": "iid",
        "seed": 17,
        "scope": "in_contract",
        "mode": "eval",
        "parameters": {},
    }
    case.update(case_updates)
    return {
        "subject_id": "cpu-test",
        "reference_path": str(reference),
        "candidate_path": str(candidate),
        "case": case,
        "device": "cpu",
        "atol": 0.0,
        "rtol": 0.0,
    }


LINEAR_REFERENCE = """
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, value):
        return self.linear(value)

def get_inputs():
    return [torch.arange(6, dtype=torch.float32).reshape(2, 3)]

def get_init_inputs():
    return []
"""


LINEAR_CANDIDATE = """
import torch
from torch import nn

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, value):
        return self.linear(value)
"""


def test_direct_candidate_passes_after_strict_state_sync(tmp_path: Path) -> None:
    worker = _load_worker()
    reference = _write(tmp_path / "reference.py", LINEAR_REFERENCE)
    candidate = _write(tmp_path / "candidate.py", LINEAR_CANDIDATE)

    result = worker.execute(_config(reference, candidate))

    assert result["validation_status"] == "pass"
    assert result["candidate_runs"] == 1
    assert result["reference_runs"] == 1


def test_scalar_init_input_is_not_silently_dropped(tmp_path: Path) -> None:
    worker = _load_worker()
    reference = _write(
        tmp_path / "reference.py",
        """
import torch
from torch import nn
class Model(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, value):
        return value * self.scale
def get_inputs():
    return [torch.ones(3)]
def get_init_inputs():
    return 4.0
""",
    )
    candidate = _write(
        tmp_path / "candidate.py",
        """
from torch import nn
class ModelNew(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, value):
        return value * self.scale
""",
    )

    result = worker.execute(_config(reference, candidate))

    assert result["validation_status"] == "pass"


def test_candidate_dtype_change_is_a_failure_not_cast_away(tmp_path: Path) -> None:
    worker = _load_worker()
    reference = _write(tmp_path / "reference.py", LINEAR_REFERENCE)
    candidate = _write(
        tmp_path / "candidate.py",
        LINEAR_CANDIDATE.replace(
            "return self.linear(value)", "return self.linear(value).to(torch.float64)"
        ),
    )

    result = worker.execute(_config(reference, candidate))

    assert result["validation_status"] == "fail"
    assert result["verdict"]["oracle"]["mismatches"][0]["kind"] == "dtype"


def test_each_side_receives_isolated_inputs(tmp_path: Path) -> None:
    worker = _load_worker()
    reference = _write(
        tmp_path / "reference.py",
        """
import torch
from torch import nn
class Model(nn.Module):
    def forward(self, value):
        return value.add_(1)
def get_inputs():
    return [torch.zeros(4)]
def get_init_inputs():
    return []
""",
    )
    candidate = _write(
        tmp_path / "candidate.py",
        """
from torch import nn
class ModelNew(nn.Module):
    def forward(self, value):
        return value.add_(1)
""",
    )

    result = worker.execute(_config(reference, candidate))

    assert result["validation_status"] == "pass"


def test_reference_crash_is_inconclusive(tmp_path: Path) -> None:
    worker = _load_worker()
    reference = _write(
        tmp_path / "reference.py",
        """
import torch
from torch import nn
class Model(nn.Module):
    def forward(self, value):
        raise RuntimeError('reference crash')
def get_inputs():
    return [torch.zeros(1)]
def get_init_inputs():
    return []
""",
    )
    candidate = _write(
        tmp_path / "candidate.py",
        """
from torch import nn
class ModelNew(nn.Module):
    def forward(self, value):
        return value
""",
    )

    result = worker.execute(_config(reference, candidate))

    assert result["validation_status"] == "inconclusive"
    assert "reference execution failed" in result["reason"]


def test_candidate_crash_with_valid_reference_is_a_failure(tmp_path: Path) -> None:
    worker = _load_worker()
    reference = _write(tmp_path / "reference.py", LINEAR_REFERENCE)
    candidate = _write(
        tmp_path / "candidate.py",
        LINEAR_CANDIDATE.replace(
            "return self.linear(value)", "raise RuntimeError('candidate crash')"
        ),
    )

    result = worker.execute(_config(reference, candidate))

    assert result["validation_status"] == "fail"
    assert result["errors"][0]["phase"] == "candidate"


def test_candidate_system_exit_during_import_is_an_attributed_failure(tmp_path: Path) -> None:
    worker = _load_worker()
    reference = _write(tmp_path / "reference.py", LINEAR_REFERENCE)
    candidate = _write(tmp_path / "candidate.py", "raise SystemExit(7)\n")

    result = worker.execute(_config(reference, candidate))

    assert result["validation_status"] == "fail"
    assert result["phase"] == "candidate_import"
    assert result["errors"][0]["exception_type"] == "SystemExit"


def test_candidate_state_schema_guessing_is_refused(tmp_path: Path) -> None:
    worker = _load_worker()
    reference = _write(tmp_path / "reference.py", LINEAR_REFERENCE)
    candidate = _write(
        tmp_path / "candidate.py",
        LINEAR_CANDIDATE.replace("self.linear", "self.renamed"),
    )

    result = worker.execute(_config(reference, candidate))

    assert result["validation_status"] == "inconclusive"
    assert "state_dict keys differ" in result["errors"][0]["message"]


def test_batch_resize_requires_explicit_argument_contract(tmp_path: Path) -> None:
    worker = _load_worker()
    reference = _write(tmp_path / "reference.py", LINEAR_REFERENCE)
    candidate = _write(tmp_path / "candidate.py", LINEAR_CANDIDATE)

    result = worker.execute(
        _config(
            reference,
            candidate,
            mode="config",
            parameters={"batch_size": 1},
        )
    )

    assert result["validation_status"] == "inconclusive"
    assert result["phase"] == "input_generation"
    assert "batch_size requires" in result["errors"][0]["message"]


def test_forward_backward_mode_compares_input_gradients(tmp_path: Path) -> None:
    worker = _load_worker()
    reference = _write(tmp_path / "reference.py", LINEAR_REFERENCE)
    candidate = _write(
        tmp_path / "candidate.py",
        LINEAR_CANDIDATE.replace(
            "return self.linear(value)", "return self.linear(value.detach())"
        ),
    )

    result = worker.execute(
        _config(
            reference,
            candidate,
            mode="train",
            parameters={"requires_backward": True},
        )
    )

    assert result["validation_status"] == "fail"
    paths = [item["path"] for item in result["verdict"]["oracle"]["mismatches"]]
    assert any("input_gradients" in path for path in paths)


def test_backward_uses_nonuniform_vjps_not_only_output_sum(tmp_path: Path) -> None:
    worker = _load_worker()
    reference = _write(
        tmp_path / "reference.py",
        """
import torch
from torch import nn
class Model(nn.Module):
    def forward(self, value):
        return value
def get_inputs():
    return [torch.arange(6, dtype=torch.float32).reshape(2, 3)]
def get_init_inputs():
    return []
""",
    )
    candidate = _write(
        tmp_path / "candidate.py",
        """
import torch
from torch import nn
class WrongBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()
    @staticmethod
    def backward(ctx, gradient):
        return gradient.mean().expand_as(gradient)
class ModelNew(nn.Module):
    def forward(self, value):
        return WrongBackward.apply(value)
""",
    )

    result = worker.execute(
        _config(
            reference,
            candidate,
            mode="train",
            parameters={"requires_backward": True},
        )
    )

    assert result["validation_status"] == "fail"
    paths = [item["path"] for item in result["verdict"]["oracle"]["mismatches"]]
    assert any("vjp_gradients" in path and "input_gradients" in path for path in paths)


def test_explicit_batch_adapter_executes_config_case(tmp_path: Path) -> None:
    worker = _load_worker()
    reference = _write(tmp_path / "reference.py", LINEAR_REFERENCE)
    candidate = _write(tmp_path / "candidate.py", LINEAR_CANDIDATE)

    result = worker.execute(
        _config(
            reference,
            candidate,
            mode="config",
            parameters={
                "batch_size": 1,
                "batch_arg_indices": [0],
                "batch_dimension": 0,
            },
        )
    )

    assert result["validation_status"] == "pass"


def test_explicit_layout_adapter_produces_noncontiguous_input(tmp_path: Path) -> None:
    worker = _load_worker()
    reference = _write(
        tmp_path / "reference.py",
        """
import torch
from torch import nn
class Model(nn.Module):
    def forward(self, value):
        assert not value.is_contiguous()
        return value * 2
def get_inputs():
    return [torch.arange(12, dtype=torch.float32).reshape(3, 4)]
def get_init_inputs():
    return []
""",
    )
    candidate = _write(
        tmp_path / "candidate.py",
        """
from torch import nn
class ModelNew(nn.Module):
    def forward(self, value):
        assert not value.is_contiguous()
        return value * 2
""",
    )

    result = worker.execute(
        _config(
            reference,
            candidate,
            parameters={"layout": "noncontiguous", "layout_arg_indices": [0]},
        )
    )

    assert result["validation_status"] == "pass"


def test_repeated_mode_accounts_for_every_candidate_execution(tmp_path: Path) -> None:
    worker = _load_worker()
    reference = _write(tmp_path / "reference.py", LINEAR_REFERENCE)
    candidate = _write(tmp_path / "candidate.py", LINEAR_CANDIDATE)

    result = worker.execute(
        _config(
            reference,
            candidate,
            mode="repeated",
            parameters={"repeat_count": 2},
        )
    )

    assert result["validation_status"] == "pass"
    assert result["candidate_runs"] == 2
    assert result["reference_runs"] == 2


def test_repeated_mode_detects_a_candidate_that_only_passes_once(tmp_path: Path) -> None:
    worker = _load_worker()
    reference = _write(
        tmp_path / "reference.py",
        """
import torch
from torch import nn
class Model(nn.Module):
    def forward(self, value):
        return value
def get_inputs():
    return [torch.ones(4)]
def get_init_inputs():
    return []
""",
    )
    candidate = _write(
        tmp_path / "candidate.py",
        """
from torch import nn
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
    def forward(self, value):
        self.calls += 1
        return value if self.calls == 1 else value + 1
""",
    )

    single = worker.execute(_config(reference, candidate))
    repeated = worker.execute(
        _config(
            reference,
            candidate,
            mode="repeated",
            parameters={"repeat_count": 2},
        )
    )

    assert single["validation_status"] == "pass"
    assert repeated["validation_status"] == "fail"
    assert repeated["candidate_runs"] == 2


def test_repeated_mode_preserves_buffer_sequence_between_calls(tmp_path: Path) -> None:
    worker = _load_worker()
    reference = _write(
        tmp_path / "reference.py",
        """
import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('counter', torch.tensor(0.0))
    def forward(self, value):
        self.counter.add_(1)
        return value + self.counter
def get_inputs():
    return [torch.zeros(2)]
def get_init_inputs():
    return []
""",
    )
    candidate = _write(
        tmp_path / "candidate.py",
        """
import torch
from torch import nn
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('counter', torch.tensor(0.0))
    def forward(self, value):
        increment = 1 if self.counter.item() == 0 else 2
        self.counter.add_(increment)
        return value + self.counter
""",
    )

    result = worker.execute(
        _config(
            reference,
            candidate,
            mode="repeated",
            parameters={"repeat_count": 2},
        )
    )

    assert result["validation_status"] == "fail"
    assert result["candidate_runs"] == 2
    assert result["verdict"]["sequence_semantics"] == "same_instances_without_state_reset"
