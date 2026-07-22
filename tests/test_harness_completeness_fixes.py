"""Tests for the two E0-diagnosed completeness gaps of the corrected substrate.

Gap 1: strict name-exact state sync rejected candidates whose parameter
naming differs from the reference (all stateful E0 probes).  The fix aligns
state-dict keys by leaf name and structural signature, accepting only
unambiguous bijections.

Gap 2: the oracle's whole-tensor temporaries amplified VRAM usage and turned
large subjects into OOM-INCONCLUSIVE.  The fix compares in bounded chunks and
retries an OOM-failed comparison on CPU copies.
"""

import unittest
from unittest import mock

import torch
from torch import nn

from src.validation import (
    OracleConfig,
    StateSyncError,
    ValidationStatus,
    align_state_keys,
    compare_outputs,
    strict_sync_state_dict,
    validate_pair,
)
from src.validation.oracle import Tolerance, _nonfinite_and_close, _paired_chunks
from src.validation.types import OracleResult


class _NestedReference(nn.Module):
    """KernelBench-style reference: parameters live in named submodules."""

    def __init__(self):
        super().__init__()
        self.gemm = nn.Linear(3, 4)
        self.register_buffer("dummy", torch.zeros(1))
        self.norm = nn.Module()
        self.norm.register_buffer("running_mean", torch.zeros(4))
        self.norm.register_buffer("running_var", torch.ones(4))

    def forward(self, x):
        return self.gemm(x)


class _FlatCandidate(nn.Module):
    """Candidate-kernel style: the same state registered under flat names."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 3))
        self.bias = nn.Parameter(torch.randn(4))
        self.register_buffer("dummy", torch.zeros(1))
        self.register_buffer("running_mean", torch.randn(4))
        self.register_buffer("running_var", torch.rand(4))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


class TestStateKeyAlignment(unittest.TestCase):
    def test_flat_candidate_aligns_to_nested_reference(self):
        reference = _NestedReference()
        candidate = _FlatCandidate()

        mapping = align_state_keys(reference.state_dict(), candidate.state_dict())

        self.assertEqual(mapping["gemm.weight"], "weight")
        self.assertEqual(mapping["gemm.bias"], "bias")
        self.assertEqual(mapping["dummy"], "dummy")
        self.assertEqual(mapping["norm.running_mean"], "running_mean")
        self.assertEqual(mapping["norm.running_var"], "running_var")

    def test_sync_propagates_values_through_the_alignment(self):
        reference = _NestedReference()
        candidate = _FlatCandidate()

        report = strict_sync_state_dict(reference, candidate)

        self.assertEqual(report.keys_synced, 5)
        self.assertEqual(report.remapped_keys, 4)
        self.assertTrue(torch.equal(candidate.weight, reference.gemm.weight))
        self.assertTrue(torch.equal(candidate.bias, reference.gemm.bias))
        self.assertTrue(
            torch.equal(candidate.running_mean, reference.norm.running_mean)
        )

    def test_paired_validation_passes_for_renamed_equivalent_modules(self):
        reference = _NestedReference()
        candidate = _FlatCandidate()

        result = validate_pair(reference, candidate, args=(torch.randn(2, 3),))

        self.assertEqual(result.status, ValidationStatus.PASS)

    def test_leaf_match_with_incompatible_shape_is_refused(self):
        reference = nn.Module()
        reference.sub = nn.Module()
        reference.sub.register_buffer("scale", torch.zeros(3))
        candidate = nn.Module()
        candidate.register_buffer("scale", torch.zeros(5))

        with self.assertRaisesRegex(StateSyncError, "structurally incompatible"):
            align_state_keys(reference.state_dict(), candidate.state_dict())

    def test_ambiguous_signature_groups_are_refused(self):
        reference = nn.Module()
        reference.register_buffer("a", torch.zeros(3))
        reference.register_buffer("b", torch.ones(3))
        candidate = nn.Module()
        candidate.register_buffer("x", torch.zeros(3))
        candidate.register_buffer("y", torch.ones(3))

        with self.assertRaisesRegex(StateSyncError, "cannot be aligned unambiguously"):
            align_state_keys(reference.state_dict(), candidate.state_dict())

    def test_signature_stage_resolves_unique_shapes(self):
        # Leaf names disagree entirely, but each shape occurs exactly once.
        reference = nn.Module()
        reference.register_buffer("first", torch.zeros(2, 3))
        reference.register_buffer("second", torch.zeros(7))
        candidate = nn.Module()
        candidate.register_buffer("alpha", torch.ones(2, 3))
        candidate.register_buffer("beta", torch.ones(7))

        mapping = align_state_keys(reference.state_dict(), candidate.state_dict())

        self.assertEqual(mapping["first"], "alpha")
        self.assertEqual(mapping["second"], "beta")


class TestChunkedOracle(unittest.TestCase):
    def test_paired_chunks_cover_every_element_without_copies(self):
        reference = torch.arange(24.0).reshape(4, 6)
        candidate = reference.clone()

        chunks = list(_paired_chunks(reference, candidate, chunk_numel=5))

        total = sum(ref.numel() for ref, _ in chunks)
        self.assertEqual(total, 24)
        reassembled = torch.cat([ref.reshape(-1) for ref, _ in chunks])
        self.assertTrue(torch.equal(reassembled, torch.arange(24.0)))

    def test_chunked_verdicts_match_whole_tensor_semantics(self):
        tolerance = Tolerance(rtol=0.0, atol=0.1)
        reference = torch.linspace(-2, 2, steps=101)
        candidate = reference.clone()
        candidate[7] += 5.0
        candidate[93] -= 3.0

        whole = _nonfinite_and_close(
            reference, candidate, tolerance, True, chunk_numel=1 << 24
        )
        chunked = _nonfinite_and_close(
            reference, candidate, tolerance, True, chunk_numel=13
        )

        self.assertEqual(whole, chunked)
        self.assertFalse(chunked[0])
        self.assertIn("2 value(s) exceed", chunked[1])

    def test_chunked_nan_and_inf_counts_are_exact(self):
        tolerance = Tolerance(rtol=0.0, atol=0.0)
        reference = torch.zeros(50)
        candidate = torch.zeros(50)
        reference[3] = float("nan")
        reference[40] = float("nan")

        ok, message = _nonfinite_and_close(
            reference, candidate, tolerance, True, chunk_numel=7
        )
        self.assertFalse(ok)
        self.assertIn("NaN positions differ at 2 element(s)", message)

        reference = torch.zeros(50)
        candidate = torch.zeros(50)
        reference[10] = float("inf")
        candidate[10] = -float("inf")
        ok, message = _nonfinite_and_close(
            reference, candidate, tolerance, True, chunk_numel=7
        )
        self.assertFalse(ok)
        self.assertEqual(message, "Inf signs differ")

    def test_compare_outputs_honours_configured_chunk_size(self):
        reference = torch.randn(8, 33)
        candidate = reference.clone()
        config = OracleConfig(compare_chunk_numel=10)

        self.assertEqual(
            compare_outputs(reference, candidate, config).status,
            ValidationStatus.PASS,
        )

        candidate[5, 20] += 1.0
        result = compare_outputs(reference, candidate, config)
        self.assertEqual(result.status, ValidationStatus.FAIL)
        self.assertIn("1 value(s) exceed", result.mismatches[0].message)

    def test_non_contiguous_tensors_are_chunked_correctly(self):
        base = torch.randn(30, 6)
        reference = base.t()  # non-contiguous view (6, 30)
        candidate = base.t().clone()
        candidate[4, 17] += 9.0

        result = compare_outputs(
            reference, candidate, OracleConfig(compare_chunk_numel=11)
        )
        self.assertEqual(result.status, ValidationStatus.FAIL)
        self.assertIn("1 value(s) exceed", result.mismatches[0].message)


class TestCpuFallbackCompare(unittest.TestCase):
    def test_oracle_oom_is_retried_on_cpu_and_can_pass(self):
        real_compare = compare_outputs
        calls = {"count": 0}

        def flaky_compare(reference, candidate, config=None):
            calls["count"] += 1
            if calls["count"] == 1:
                return OracleResult(
                    status=ValidationStatus.INCONCLUSIVE,
                    compared_leaves=0,
                    mismatches=[],
                    reason=(
                        "oracle raised OutOfMemoryError: CUDA out of memory "
                        "(simulated)"
                    ),
                )
            return real_compare(reference, candidate, config)

        with mock.patch(
            "src.validation.executor.compare_outputs", side_effect=flaky_compare
        ):
            result = validate_pair(
                nn.Identity(), nn.Identity(), args=(torch.ones(4),)
            )

        self.assertEqual(calls["count"], 2)
        self.assertEqual(result.status, ValidationStatus.PASS)
        self.assertTrue(
            any(e.phase == "cpu_fallback_compare" for e in result.errors)
        )

    def test_fallback_can_be_disabled(self):
        from src.validation import ExecutionConfig

        def resource_inconclusive(reference, candidate, config=None):
            return OracleResult(
                status=ValidationStatus.INCONCLUSIVE,
                compared_leaves=0,
                mismatches=[],
                reason="oracle raised OutOfMemoryError: CUDA out of memory",
            )

        with mock.patch(
            "src.validation.executor.compare_outputs",
            side_effect=resource_inconclusive,
        ):
            result = validate_pair(
                nn.Identity(),
                nn.Identity(),
                args=(torch.ones(4),),
                execution_config=ExecutionConfig(cpu_fallback_compare=False),
            )

        self.assertEqual(result.status, ValidationStatus.INCONCLUSIVE)
        self.assertIn("OutOfMemoryError", result.reason)


if __name__ == "__main__":
    unittest.main()
