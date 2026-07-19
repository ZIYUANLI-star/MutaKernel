"""CPU-only tests for the sound validation core."""

from __future__ import annotations

import random
import unittest

import torch
from torch import nn

from src.validation import (
    OracleConfig,
    RNGSnapshot,
    StateSyncError,
    ValidationStatus,
    clone_call_inputs,
    clone_tree,
    compare_outputs,
    describe_input_tree,
    strict_sync_state_dict,
    validate_pair,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


class _ReverseOrderedState(nn.Module):
    def __init__(self, reverse: bool, first: float, second: float):
        super().__init__()
        entries = [("first", first), ("second", second)]
        if reverse:
            entries.reverse()
        for name, value in entries:
            self.register_buffer(name, torch.tensor(value))

    def forward(self, x):
        return x * self.first + self.second


class TestStateAndRNG(unittest.TestCase):
    def test_state_sync_matches_exact_keys_not_iteration_order(self):
        reference = _ReverseOrderedState(False, first=2.0, second=3.0)
        candidate = _ReverseOrderedState(True, first=20.0, second=30.0)

        report = strict_sync_state_dict(reference, candidate)

        self.assertEqual(report.keys_synced, 2)
        self.assertEqual(candidate.first.item(), 2.0)
        self.assertEqual(candidate.second.item(), 3.0)

    def test_state_sync_rejects_key_mismatch(self):
        reference = nn.Linear(2, 2)
        candidate = nn.Sequential(nn.Linear(2, 2))
        with self.assertRaisesRegex(StateSyncError, "keys differ"):
            strict_sync_state_dict(reference, candidate)

    def test_rng_snapshot_replays_python_numpy_and_torch(self):
        caller_state = RNGSnapshot.capture(include_cuda=False)
        try:
            random.seed(17)
            torch.manual_seed(17)
            if np is not None:
                np.random.seed(17)
            replay = RNGSnapshot.capture(include_cuda=False)

            first = (random.random(), torch.rand(4))
            first_numpy = None if np is None else np.random.rand(4)
            replay.restore()
            second = (random.random(), torch.rand(4))
            second_numpy = None if np is None else np.random.rand(4)

            self.assertEqual(first[0], second[0])
            self.assertTrue(torch.equal(first[1], second[1]))
            if np is not None:
                self.assertTrue(np.array_equal(first_numpy, second_numpy))
        finally:
            caller_state.restore()


class TestExactInputGeometry(unittest.TestCase):
    def test_clone_preserves_non_dense_stride_and_storage_offset(self):
        backing = torch.arange(48, dtype=torch.float32).reshape(3, 16)
        view = backing[:, 3:11:2]

        cloned = clone_tree(view)

        self.assertEqual(cloned.shape, view.shape)
        self.assertEqual(cloned.stride(), view.stride())
        self.assertEqual(cloned.storage_offset(), view.storage_offset())
        self.assertFalse(cloned.is_contiguous())
        self.assertTrue(torch.equal(cloned, view))
        self.assertNotEqual(
            cloned.untyped_storage().data_ptr(),
            view.untyped_storage().data_ptr(),
        )

    def test_distinct_overlapping_views_keep_aliasing_inside_clone(self):
        backing = torch.arange(8, dtype=torch.float32)
        left = backing[1:5]
        right = backing[3:7]

        cloned_left, cloned_right = clone_tree((left, right))
        cloned_left[2] = 999

        self.assertEqual(cloned_right[0].item(), 999)
        self.assertEqual(backing[3].item(), 3)
        self.assertEqual(
            cloned_left.untyped_storage().data_ptr(),
            cloned_right.untyped_storage().data_ptr(),
        )


class TestInputIsolation(unittest.TestCase):
    def test_recursive_clone_preserves_internal_aliases(self):
        tensor = torch.tensor([1.0])
        source = {"left": tensor, "right": [tensor]}

        cloned = clone_tree(source)

        self.assertIs(cloned["left"], cloned["right"][0])
        self.assertIsNot(cloned["left"], tensor)
        cloned["left"].add_(10)
        self.assertEqual(tensor.item(), 1.0)

    def test_args_and_kwargs_share_alias_within_each_isolated_call(self):
        tensor = torch.tensor([2.0])
        args, kwargs = clone_call_inputs((tensor,), {"also": tensor})
        self.assertIs(args[0], kwargs["also"])
        self.assertIsNot(args[0], tensor)

    def test_input_fingerprint_covers_values_geometry_and_aliases(self):
        backing = torch.arange(10, dtype=torch.float32)
        left = backing[1:7:2]
        right = backing[3:9:2]

        first = describe_input_tree((left, right), hash_contents=True)
        replay = describe_input_tree(clone_tree((left, right)), hash_contents=True)
        lost_alias = describe_input_tree((left.clone(), right.clone()), hash_contents=True)
        changed = describe_input_tree((left + 1, right), hash_contents=True)

        tensor_leaves = [leaf for leaf in first["leaves"] if leaf["kind"] == "tensor"]
        self.assertEqual(tensor_leaves[0]["stride"], [2])
        self.assertEqual(tensor_leaves[0]["alias_group"], tensor_leaves[1]["alias_group"])
        self.assertEqual(first["content_sha256"], replay["content_sha256"])
        self.assertNotEqual(first["content_sha256"], lost_alias["content_sha256"])
        self.assertNotEqual(first["content_sha256"], changed["content_sha256"])


class TestOracle(unittest.TestCase):
    def test_nested_numeric_outputs_pass(self):
        reference = {
            "float": torch.tensor([1.0, float("nan"), float("inf"), -float("inf")]),
            "int": torch.tensor([1, 2], dtype=torch.int64),
            "bool": torch.tensor([True, False]),
            "complex": torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex64),
        }
        candidate = clone_tree(reference)

        result = compare_outputs(reference, candidate)

        self.assertEqual(result.status, ValidationStatus.PASS)
        self.assertEqual(result.compared_leaves, 4)

    def test_structure_shape_and_dtype_are_not_coerced(self):
        structure = compare_outputs([torch.ones(2)], (torch.ones(2),))
        shape = compare_outputs(torch.ones(2), torch.ones(3))
        dtype = compare_outputs(
            torch.ones(2, dtype=torch.float32),
            torch.ones(2, dtype=torch.float64),
        )

        self.assertEqual(structure.status, ValidationStatus.FAIL)
        self.assertEqual(structure.mismatches[0].kind, "structure")
        self.assertEqual(shape.mismatches[0].kind, "shape")
        self.assertEqual(dtype.mismatches[0].kind, "dtype")

    def test_nan_positions_and_inf_signs_must_match(self):
        nan_result = compare_outputs(
            torch.tensor([float("nan"), 1.0]),
            torch.tensor([1.0, float("nan")]),
        )
        inf_result = compare_outputs(
            torch.tensor([float("inf")]),
            torch.tensor([-float("inf")]),
        )

        self.assertEqual(nan_result.status, ValidationStatus.FAIL)
        self.assertIn("NaN positions", nan_result.mismatches[0].message)
        self.assertEqual(inf_result.status, ValidationStatus.FAIL)
        self.assertIn("Inf signs", inf_result.mismatches[0].message)

    def test_integer_and_bool_outputs_require_exact_equality(self):
        integer = compare_outputs(torch.tensor([1]), torch.tensor([2]))
        boolean = compare_outputs(torch.tensor([True]), torch.tensor([False]))
        self.assertEqual(integer.status, ValidationStatus.FAIL)
        self.assertEqual(boolean.status, ValidationStatus.FAIL)

    def test_unsupported_leaf_is_inconclusive(self):
        class Unsupported:
            pass

        result = compare_outputs(Unsupported(), Unsupported())
        self.assertEqual(result.status, ValidationStatus.INCONCLUSIVE)
        self.assertEqual(result.mismatches[0].kind, "unsupported")

    def test_diagnostic_cap_never_hides_a_later_definite_mismatch(self):
        class Unsupported:
            pass

        reference = [Unsupported() for _ in range(20)] + [torch.tensor([1])]
        candidate = [Unsupported() for _ in range(20)] + [torch.tensor([2])]

        result = compare_outputs(
            reference,
            candidate,
            OracleConfig(max_mismatches=20),
        )

        self.assertEqual(result.status, ValidationStatus.FAIL)
        self.assertEqual(len(result.mismatches), 20)

    def test_optional_stride_contract_detects_layout_geometry_change(self):
        reference = torch.arange(12).reshape(3, 4).t()
        candidate = reference.contiguous()

        default = compare_outputs(reference, candidate)
        strict = compare_outputs(
            reference,
            candidate,
            OracleConfig(require_stride=True),
        )

        self.assertEqual(default.status, ValidationStatus.PASS)
        self.assertEqual(strict.status, ValidationStatus.FAIL)
        self.assertEqual(strict.mismatches[0].kind, "stride")

    def test_optional_alias_contract_compares_output_alias_topology(self):
        backing = torch.arange(8)
        reference = (backing[:4], backing[2:6])
        candidate = (backing[:4].clone(), backing[2:6].clone())

        strict = compare_outputs(
            reference,
            candidate,
            OracleConfig(require_aliasing=True),
        )

        self.assertEqual(strict.status, ValidationStatus.FAIL)
        self.assertTrue(
            any(item.kind == "aliasing" for item in strict.mismatches)
        )


class _Stochastic(nn.Module):
    def forward(self, x):
        return x + torch.rand_like(x) + random.random()


class _Mutating(nn.Module):
    def forward(self, x):
        x.add_(1)
        return x.clone()


class _Raises(nn.Module):
    def forward(self, x):
        raise RuntimeError("deliberate failure")


class TestValidationExecutor(unittest.TestCase):
    def test_stochastic_calls_receive_identical_rng_state(self):
        result = validate_pair(_Stochastic(), _Stochastic(), args=(torch.zeros(8),))
        self.assertEqual(result.status, ValidationStatus.PASS)

    def test_reference_and_candidate_inputs_are_isolated(self):
        original = torch.tensor([1.0])
        result = validate_pair(_Mutating(), _Mutating(), args=(original,))
        self.assertEqual(result.status, ValidationStatus.PASS)
        self.assertEqual(original.item(), 1.0)

    def test_post_call_input_side_effect_mismatch_is_a_failure(self):
        class Pure(nn.Module):
            def forward(self, value):
                return value + 1

        class Mutates(nn.Module):
            def forward(self, value):
                value.add_(1)
                return value

        result = validate_pair(Pure(), Mutates(), args=(torch.zeros(3),))

        self.assertEqual(result.status, ValidationStatus.FAIL)
        self.assertTrue(
            any("post_call_inputs" in item.path for item in result.oracle.mismatches)
        )

    def test_post_call_buffer_state_mismatch_is_a_failure(self):
        class Stateful(nn.Module):
            def __init__(self, delta):
                super().__init__()
                self.delta = delta
                self.register_buffer("counter", torch.tensor(0.0))

            def forward(self, value):
                self.counter.add_(self.delta)
                return value

        result = validate_pair(
            Stateful(1.0),
            Stateful(2.0),
            args=(torch.zeros(1),),
        )

        self.assertEqual(result.status, ValidationStatus.FAIL)
        self.assertTrue(
            any("post_call_state" in item.path for item in result.oracle.mismatches)
        )

    def test_state_is_synchronized_then_restored(self):
        reference = nn.Linear(2, 2, bias=False)
        candidate = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            reference.weight.fill_(2.0)
            candidate.weight.fill_(7.0)
        candidate_before = candidate.weight.detach().clone()

        result = validate_pair(reference, candidate, args=(torch.ones(1, 2),))

        self.assertEqual(result.status, ValidationStatus.PASS)
        self.assertTrue(torch.equal(candidate.weight, candidate_before))
        self.assertTrue(torch.equal(reference.weight, torch.full_like(reference.weight, 2.0)))

    def test_state_mismatch_is_inconclusive(self):
        result = validate_pair(
            nn.Linear(2, 2),
            nn.Sequential(nn.Linear(2, 2)),
            args=(torch.ones(1, 2),),
        )
        self.assertEqual(result.status, ValidationStatus.INCONCLUSIVE)
        self.assertIn("state_dict keys differ", result.reason)
        self.assertEqual(result.reference_invocations, 0)
        self.assertEqual(result.candidate_invocations, 0)

    def test_candidate_exception_is_fail_but_reference_exception_is_inconclusive(self):
        candidate_failure = validate_pair(nn.Identity(), _Raises(), args=(torch.ones(1),))
        reference_failure = validate_pair(_Raises(), nn.Identity(), args=(torch.ones(1),))

        self.assertEqual(candidate_failure.status, ValidationStatus.FAIL)
        self.assertEqual(reference_failure.status, ValidationStatus.INCONCLUSIVE)

    def test_phase_timings_and_serializable_summary(self):
        result = validate_pair(nn.Identity(), nn.Identity(), args=(torch.ones(1),))
        timings = result.timings.to_dict()

        self.assertEqual(result.status, ValidationStatus.PASS)
        self.assertEqual(
            set(timings),
            {
                "state_snapshot_ms",
                "state_sync_ms",
                "rng_capture_ms",
                "input_isolation_ms",
                "reference_ms",
                "candidate_ms",
                "oracle_ms",
                "cleanup_ms",
                "total_ms",
            },
        )
        self.assertTrue(all(value >= 0 for value in timings.values()))
        self.assertGreater(timings["total_ms"], 0)
        self.assertEqual(result.to_dict()["status"], "pass")


if __name__ == "__main__":
    unittest.main()
