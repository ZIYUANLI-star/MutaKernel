"""CPU soundness regressions for ``scripts/_mutant_worker.py`` equiv mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from scripts import _mutant_worker


def _run_equiv(
    tmp_path: Path,
    reference_source: str,
    candidate_source: str,
    **overrides: Any,
) -> Dict[str, Any]:
    problem_file = tmp_path / "problem.py"
    problem_file.write_text(reference_source, encoding="utf-8")
    cfg = {
        "mode": "equiv",
        "problem_file": str(problem_file),
        "kernel_code": "",
        "mutant_id": overrides.pop("mutant_id", "cpu-soundness"),
        "mutated_code": candidate_source,
        "operator_name": "test",
        "device": "cpu",
        "equiv_runs": 2,
        "base_seed": 101,
        "stress_policies": [],
        "atol": 0.0,
        "rtol": 0.0,
        **overrides,
    }
    return _mutant_worker._equiv_mode(cfg)


def test_random_parameters_and_forward_rng_are_replayed(tmp_path: Path) -> None:
    reference = """
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, value):
        return self.linear(value) + torch.rand(value.shape[0], 2)

def get_inputs():
    return [torch.arange(6, dtype=torch.float32).reshape(2, 3)]

def get_init_inputs():
    return []
"""
    candidate = reference.replace("class Model(nn.Module):", "class ModelNew(nn.Module):")

    result = _run_equiv(tmp_path, reference, candidate)

    assert result["validation_status"] == "pass"
    assert result["is_equivalent"] is True
    assert result["valid_rounds"] == 2
    assert not result["errors"]


def test_output_dtype_mismatch_is_a_concrete_non_equivalence(tmp_path: Path) -> None:
    reference = """
import torch
from torch import nn

class Model(nn.Module):
    def forward(self, value):
        return value.clone()

def get_inputs():
    return [torch.ones(4, dtype=torch.float32)]
"""
    candidate = """
from torch import nn

class ModelNew(nn.Module):
    def forward(self, value):
        return value.double()
"""

    result = _run_equiv(tmp_path, reference, candidate)

    assert result["validation_status"] == "fail"
    assert result["is_equivalent"] is False
    assert result["divergence"]["detail"] == "output_diverged"
    mismatches = result["divergence"]["oracle"]["mismatches"]
    assert any(mismatch["kind"] == "dtype" for mismatch in mismatches)


def test_in_place_mutation_does_not_contaminate_the_other_side(tmp_path: Path) -> None:
    reference = """
import torch
from torch import nn

class Model(nn.Module):
    def forward(self, value):
        value.add_(1)
        return value.clone()

def get_inputs():
    return [torch.zeros(3)]
"""
    candidate = reference.replace("class Model(nn.Module):", "class ModelNew(nn.Module):")

    result = _run_equiv(tmp_path, reference, candidate)

    assert result["validation_status"] == "pass"
    assert result["is_equivalent"] is True


def test_reference_crash_is_inconclusive_not_non_equivalent(tmp_path: Path) -> None:
    reference = """
import torch
from torch import nn

class Model(nn.Module):
    def forward(self, value):
        raise RuntimeError("reference rejected input")

def get_inputs():
    return [torch.ones(2)]
"""
    candidate = """
from torch import nn

class ModelNew(nn.Module):
    def forward(self, value):
        return value
"""

    result = _run_equiv(tmp_path, reference, candidate)

    assert result["validation_status"] == "inconclusive"
    assert result["is_equivalent"] is None
    assert any(error["phase"] == "reference" for error in result["errors"])
    assert "unknown" in result["reason"] or "inconclusive" in result["reason"]


def test_input_generation_failure_is_inconclusive(tmp_path: Path) -> None:
    reference = """
from torch import nn

class Model(nn.Module):
    def forward(self, value):
        return value

def get_inputs():
    raise RuntimeError("cannot generate a valid input")
"""
    candidate = """
from torch import nn

class ModelNew(nn.Module):
    def forward(self, value):
        return value
"""

    result = _run_equiv(tmp_path, reference, candidate)

    assert result["validation_status"] == "inconclusive"
    assert result["is_equivalent"] is None
    assert all(trial["status"] == "inconclusive" for trial in result["trials"])
    assert any(error["phase"] == "input_generation" for error in result["errors"])


def test_candidate_resource_failure_is_inconclusive(tmp_path: Path) -> None:
    reference = """
import torch
from torch import nn

class Model(nn.Module):
    def forward(self, value):
        return value

def get_inputs():
    return [torch.ones(2)]
"""
    candidate = """
from torch import nn

class ModelNew(nn.Module):
    def forward(self, value):
        raise MemoryError("simulated resource exhaustion")
"""

    result = _run_equiv(tmp_path, reference, candidate)

    assert result["validation_status"] == "inconclusive"
    assert result["is_equivalent"] is None
    assert any(error["phase"] == "candidate" for error in result["errors"])


def test_zero_completed_rounds_cannot_claim_equivalence(tmp_path: Path) -> None:
    reference = """
from torch import nn

class Model(nn.Module):
    def forward(self, value):
        return value

def get_inputs():
    return [1]
"""
    candidate = """
from torch import nn

class ModelNew(nn.Module):
    def forward(self, value):
        return value
"""

    result = _run_equiv(
        tmp_path,
        reference,
        candidate,
        equiv_runs=0,
        stress_policies=[],
    )

    assert result["validation_status"] == "inconclusive"
    assert result["is_equivalent"] is None
    assert result["valid_rounds"] == 0
