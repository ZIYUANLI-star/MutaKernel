"""Regression tests for value-only stress-policy contracts."""

from __future__ import annotations

import torch
import pytest

from src.stress.policy_bank import STRESS_POLICIES


def test_every_policy_refuses_an_empty_noop_target() -> None:
    template = torch.empty((0, 3), dtype=torch.float32)

    for name, policy in STRESS_POLICIES.items():
        with pytest.raises(ValueError, match="silent identity/no-op"):
            policy([template], 42)


def test_every_policy_is_deterministic_and_does_not_mutate_template() -> None:
    template = torch.linspace(-1, 1, 24, dtype=torch.float32).reshape(4, 6)
    before = template.clone()

    for name, policy in STRESS_POLICIES.items():
        first = policy([template], 123)[0]
        second = policy([template], 123)[0]
        assert torch.equal(first, second), name
        assert torch.equal(template, before), name
        assert first.data_ptr() != template.data_ptr(), name


def test_value_policies_preserve_dtype_stride_and_requires_grad() -> None:
    template = torch.randn(3, 5, dtype=torch.float64).t().requires_grad_()
    assert not template.is_contiguous()

    for name, policy in STRESS_POLICIES.items():
        result = policy([template], 7)[0]
        assert result.shape == template.shape, name
        assert result.dtype == template.dtype, name
        assert result.stride() == template.stride(), name
        assert result.requires_grad, name


def test_value_policies_preserve_gapped_view_geometry() -> None:
    backing = torch.arange(48, dtype=torch.float32).reshape(3, 16)
    template = backing[:, 3:11:2]
    assert template.storage_offset() == 3
    assert template.stride() == (16, 2)

    for name, policy in STRESS_POLICIES.items():
        result = policy([template], 7)[0]
        assert result.shape == template.shape, name
        assert result.stride() == template.stride(), name
        assert result.storage_offset() == template.storage_offset(), name


def test_value_policies_preserve_cross_argument_storage_aliases() -> None:
    backing = torch.arange(8, dtype=torch.float32)
    left = backing[1:5]
    right = backing[3:7]

    for name, policy in STRESS_POLICIES.items():
        result_left, result_right = policy([left, right], 7)
        assert (
            result_left.untyped_storage().data_ptr()
            == result_right.untyped_storage().data_ptr()
        ), name
        assert result_left.storage_offset() == left.storage_offset(), name
        assert result_right.storage_offset() == right.storage_offset(), name


def test_near_overflow_policy_does_not_preinject_nonfinite_values() -> None:
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        result = STRESS_POLICIES["near_overflow"](
            [torch.zeros(4096, dtype=dtype)], 99
        )[0]
        assert bool(torch.isfinite(result).all()), dtype
        assert float(result.abs().max()) <= torch.finfo(dtype).max, dtype


def test_nonfloating_only_targets_are_rejected_instead_of_counted_as_stress() -> None:
    integer = torch.tensor([1, 2, 3], dtype=torch.int64)
    boolean = torch.tensor([True, False])

    for name, policy in STRESS_POLICIES.items():
        with pytest.raises(ValueError, match="silent identity/no-op"):
            policy([integer, boolean], 1)
