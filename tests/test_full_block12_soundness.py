"""Regression checks for the maintained historical Phase-I/EMD launcher."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).parents[1] / "scripts" / "full_block12.py"


def _load():
    spec = importlib.util.spec_from_file_location("full_block12_soundness", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_equivalence_worker_timeout_is_unknown_not_non_equivalent(monkeypatch):
    module = _load()
    monkeypatch.setattr(module, "_run_worker", lambda *_args, **_kwargs: None)
    mutant = SimpleNamespace(
        id="m1",
        mutated_code="mutated",
        operator_name="op",
    )
    kernel = SimpleNamespace(
        reference_module_path="reference.py",
        kernel_code="original",
    )

    result = module.check_equiv_isolated(mutant, kernel)

    assert result["validation_status"] == "inconclusive"
    assert result["is_equivalent"] is None


def test_llm_layer_is_explicitly_triage_only():
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"triage_only_no_status_change"' in source
    layer3 = source[source.index('l3_detail["action"] = "triage_only_no_status_change"') :]
    layer3 = layer3[: layer3.index("n_strict =")]
    assert "m.status =" not in layer3


def test_static_equivalence_rules_are_triage_not_ground_truth():
    source = SCRIPT.read_text(encoding="utf-8")
    layer1 = source[source.index('if rule_hit:') :]
    layer1 = layer1[: layer1.index('if not decided:')]

    assert "HEURISTIC_EQUIVALENCE_TRIAGE" in layer1
    assert "m.status = MutantStatus.STRICT_EQUIVALENT" not in layer1
