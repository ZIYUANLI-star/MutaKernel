"""CPU tests for sound three-valued semantics in ``_stress_worker``."""

from __future__ import annotations

import ast
import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from scripts import _stress_worker as worker
from src.validation import StateSyncError


class _ReverseOrderedState(nn.Module):
    def __init__(self, reverse: bool, left: float, right: float):
        super().__init__()
        entries = [("left", left), ("right", right)]
        if reverse:
            entries.reverse()
        for name, value in entries:
            self.register_buffer(name, torch.tensor(value))


class _MutatesInput(nn.Module):
    def forward(self, value):
        value.add_(1)
        return value.clone()


class TestStrictStateSync(unittest.TestCase):
    def test_sync_uses_exact_keys_despite_registration_order(self):
        reference = _ReverseOrderedState(False, left=2.0, right=3.0)
        candidate = _ReverseOrderedState(True, left=20.0, right=30.0)

        report = worker._sync_weights(reference, candidate)

        self.assertEqual(report.keys_synced, 2)
        self.assertEqual(candidate.left.item(), 2.0)
        self.assertEqual(candidate.right.item(), 3.0)

    def test_sync_never_falls_back_to_same_shape_positional_matching(self):
        reference = nn.Linear(2, 2)
        candidate = nn.Sequential(nn.Linear(2, 2))

        with self.assertRaises(StateSyncError):
            worker._sync_weights(reference, candidate)

    def test_sync_failure_becomes_inconclusive_when_run_as_a_mode(self):
        @worker._sound_mode_result
        def invalid_setup():
            worker._sync_weights(nn.Linear(2, 2), nn.Sequential(nn.Linear(2, 2)))
            return {"ref_ok": True, "original_ok": True, "mutant_ok": True}

        result = invalid_setup()

        self.assertEqual(result["validation_status"], "inconclusive")
        self.assertFalse(result["killed"])
        self.assertFalse(result["original_ok"])
        self.assertTrue(result["errors"])

    def test_build_models_syncs_when_config_omits_legacy_flag(self):
        class Reference(_ReverseOrderedState):
            def __init__(self):
                super().__init__(False, left=2.0, right=3.0)

        class Original(_ReverseOrderedState):
            def __init__(self):
                super().__init__(True, left=20.0, right=30.0)

        class Candidate(_ReverseOrderedState):
            def __init__(self):
                super().__init__(True, left=200.0, right=300.0)

        reference_module = SimpleNamespace(
            Model=Reference,
            get_inputs=lambda: [torch.ones(1)],
            get_init_inputs=lambda: [],
        )
        original_module = SimpleNamespace(ModelNew=Original)
        candidate_module = SimpleNamespace(ModelNew=Candidate)
        cfg_without_sync_flag = {
            "kernel_code": "original source",
            "mutated_code": "candidate source",
            "problem_file": "reference.py",
        }

        with mock.patch(
            "src.bridge.eval_bridge._load_module_from_path",
            return_value=reference_module,
        ), mock.patch(
            "src.mutengine.mutant_runner._load_module_from_source",
            side_effect=[original_module, candidate_module],
        ) as source_loader:
            _, original, candidate, _, _, _ = worker._build_models(
                cfg_without_sync_flag,
                "cpu_test",
                "cpu",
            )

        self.assertEqual(original.left.item(), 2.0)
        self.assertEqual(original.right.item(), 3.0)
        self.assertEqual(candidate.left.item(), 2.0)
        self.assertEqual(candidate.right.item(), 3.0)
        self.assertTrue(all(len(call.args) == 3 for call in source_loader.call_args_list))

    def test_run_stress_loader_calls_have_exactly_three_positional_args(self):
        tree = ast.parse(inspect.getsource(inspect.unwrap(worker.run_stress)))
        load_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_load_module_from_source"
        ]
        self.assertEqual(len(load_calls), 2)
        self.assertTrue(all(len(call.args) == 3 for call in load_calls))


class TestSoundOracle(unittest.TestCase):
    def test_dtype_mismatch_is_not_hidden_by_float32_cast(self):
        reference = torch.tensor([1.0], dtype=torch.float32)
        candidate = torch.tensor([1.0], dtype=torch.float64)
        self.assertFalse(worker._allclose(reference, candidate, atol=1e-2, rtol=1e-2))

    def test_nan_inf_integer_bool_and_complex_semantics(self):
        self.assertTrue(worker._allclose(
            torch.tensor([float("nan"), float("inf"), -float("inf")]),
            torch.tensor([float("nan"), float("inf"), -float("inf")]),
            atol=0,
            rtol=0,
        ))
        self.assertFalse(worker._allclose(
            torch.tensor([float("inf")]),
            torch.tensor([-float("inf")]),
            atol=0,
            rtol=0,
        ))
        self.assertFalse(worker._allclose(
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([2], dtype=torch.int64),
            atol=100,
            rtol=100,
        ))
        self.assertFalse(worker._allclose(
            torch.tensor([True]),
            torch.tensor([False]),
            atol=100,
            rtol=100,
        ))
        self.assertTrue(worker._allclose(
            torch.tensor([1 + 2j], dtype=torch.complex64),
            torch.tensor([1 + 2j], dtype=torch.complex64),
            atol=0,
            rtol=0,
        ))

    def test_nested_structure_mismatch_fails(self):
        self.assertFalse(worker._allclose(
            [torch.ones(1)],
            (torch.ones(1),),
            atol=1e-2,
            rtol=1e-2,
        ))


class TestIsolationAndStatus(unittest.TestCase):
    def test_each_call_receives_an_isolated_recursive_clone(self):
        original = torch.tensor([1.0])
        model = _MutatesInput()

        reference_output = worker._call_isolated(model, [original])
        candidate_output = worker._call_isolated(model, [original])

        self.assertEqual(original.item(), 1.0)
        self.assertTrue(torch.equal(reference_output, candidate_output))
        self.assertEqual(reference_output.item(), 2.0)

    def test_reference_crash_is_inconclusive_and_cannot_be_killed(self):
        result = worker._normalize_validation_result({
            "ref_ok": False,
            "original_ok": False,
            "mutant_ok": False,
            "killed": True,
            "error": "ref crash: deliberate",
        })

        self.assertEqual(result["validation_status"], "inconclusive")
        self.assertFalse(result["killed"])
        self.assertTrue(result["errors"])

    def test_candidate_crash_after_valid_baseline_is_fail(self):
        result = worker._normalize_validation_result({
            "ref_ok": True,
            "original_ok": True,
            "mutant_ok": False,
            "error": "candidate crash: illegal memory access",
        })

        self.assertEqual(result["validation_status"], "fail")
        self.assertTrue(result["killed"])
        self.assertTrue(result["original_ok"])

    def test_candidate_oom_is_inconclusive_not_fail(self):
        result = worker._normalize_validation_result({
            "ref_ok": True,
            "original_ok": True,
            "mutant_ok": False,
            "killed": True,
            "error": "candidate crash: CUDA out of memory",
        })

        self.assertEqual(result["validation_status"], "inconclusive")
        self.assertFalse(result["killed"])
        # Conservative legacy encoding prevents old aggregators from inferring a kill.
        self.assertFalse(result["original_ok"])
        self.assertEqual(result["observed_original_ok"], True)

    def test_tolerance_pass_neutralizes_legacy_bitwise_kill_path(self):
        result = worker._normalize_validation_result({
            "ref_ok": True,
            "original_ok": True,
            "mutant_ok": True,
            "bitwise_orig_mut_eq": False,
        })

        self.assertEqual(result["validation_status"], "pass")
        self.assertTrue(result["bitwise_orig_mut_eq"])
        self.assertFalse(result["observed_bitwise_orig_mut_eq"])


if __name__ == "__main__":
    unittest.main()
