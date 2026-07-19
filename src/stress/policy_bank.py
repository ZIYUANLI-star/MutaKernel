"""Stress Policy Bank: 通用的多样化输入策略库。

所有策略保持 shape 和 dtype 不变，仅改变数值分布。
不依赖具体变异算子类型的先验知识。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Set, Tuple

import torch

from src.validation.inputs import clone_tree


def _generation_dtype(dtype: torch.dtype) -> torch.dtype:
    """Use enough host precision without relying on low-precision RNG kernels."""

    return torch.float64 if dtype == torch.float64 else torch.float32


def _apply_to_tensor(
    tensor: torch.Tensor, fn: Callable[[torch.Tensor, torch.Generator], torch.Tensor],
    gen: torch.Generator,
) -> torch.Tensor:
    """Replace values in an already isolated tensor without changing its view."""
    if not tensor.dtype.is_floating_point or tensor.numel() == 0:
        return tensor
    values = fn(tensor, gen)
    if tuple(values.shape) != tuple(tensor.shape):
        raise ValueError(
            f"stress policy changed shape from {tuple(tensor.shape)} "
            f"to {tuple(values.shape)}"
        )
    # ``clone_tree`` has already recreated the exact storage/view geometry.
    # Copying into that view preserves gapped strides, offsets, overlaps, and
    # aliases that ``Tensor.clone(preserve_format)`` is allowed to normalize.
    with torch.no_grad():
        tensor.copy_(values.to(dtype=tensor.dtype, device=tensor.device))
    return tensor


def _make_policy(fn: Callable):
    """Wrap a generation function into a standard policy signature."""
    def policy(template_inputs: List[Any], seed: int) -> List[Any]:
        gen = torch.Generator()
        gen.manual_seed(seed)
        result = clone_tree(template_inputs)
        visited: Set[int] = set()
        applied = 0

        def apply_tree(value: Any) -> None:
            nonlocal applied
            if isinstance(value, torch.Tensor):
                identity = id(value)
                if identity not in visited:
                    visited.add(identity)
                    if value.dtype.is_floating_point and value.numel() > 0:
                        applied += 1
                    _apply_to_tensor(value, fn, gen)
                return
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                for field in dataclasses.fields(value):
                    apply_tree(getattr(value, field.name))
                return
            if isinstance(value, Mapping):
                for item in value.values():
                    apply_tree(item)
                return
            if isinstance(value, (list, tuple, set, frozenset)):
                for item in value:
                    apply_tree(item)

        apply_tree(result)
        if applied == 0:
            raise ValueError(
                "stress policy has no non-empty real-floating tensor target; "
                "refusing a silent identity/no-op case"
            )
        return result
    return policy


def _large_magnitude(t: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    return torch.randn(t.shape, dtype=_generation_dtype(t.dtype), generator=g) * 1000.0


def _near_overflow(t: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    # Bounded values avoid injecting Inf before either implementation runs.
    # A quarter of the dtype limit still stresses additions/scalings that may
    # overflow inside the kernel.
    scale = torch.finfo(t.dtype).max / 4.0
    values = torch.rand(
        t.shape, dtype=_generation_dtype(t.dtype), generator=g
    ) * 2.0 - 1.0
    return values * scale


def _near_zero(t: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    return torch.randn(t.shape, dtype=_generation_dtype(t.dtype), generator=g) * 1e-7


def _denormals(t: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    scale = torch.finfo(t.dtype).tiny / 2.0
    signs = torch.randint(0, 2, t.shape, generator=g, dtype=torch.int64)
    return (signs.to(_generation_dtype(t.dtype)) * 2.0 - 1.0) * scale


def _all_negative(t: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    return -torch.abs(
        torch.randn(t.shape, dtype=_generation_dtype(t.dtype), generator=g)
    ) * 100.0


def _all_positive(t: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    return torch.abs(
        torch.randn(t.shape, dtype=_generation_dtype(t.dtype), generator=g)
    ) * 100.0


def _mixed_extremes(t: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    values = torch.randn(t.shape, dtype=_generation_dtype(t.dtype), generator=g)
    mask = torch.rand(t.shape, generator=g) > 0.5
    values[mask] *= 10000.0
    values[~mask] *= 0.0001
    return values


def _alternating_sign(t: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    dtype = _generation_dtype(t.dtype)
    values = torch.randn(t.shape, dtype=dtype, generator=g).abs() * 100.0
    sign = torch.ones(t.shape, dtype=dtype)
    flat = sign.view(-1)
    flat[1::2] = -1.0
    return values * sign


def _sparse(t: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    dtype = _generation_dtype(t.dtype)
    values = torch.zeros(t.shape, dtype=dtype)
    mask = torch.rand(t.shape, generator=g) > 0.9
    n = mask.sum().item()
    if n > 0:
        values[mask] = torch.randn(n, dtype=dtype, generator=g) * 100.0
    return values


def _uniform_constant(t: torch.Tensor, _g: torch.Generator) -> torch.Tensor:
    return torch.full(t.shape, 88.0, dtype=_generation_dtype(t.dtype))


def _structured_ramp(t: torch.Tensor, _g: torch.Generator) -> torch.Tensor:
    numel = t.numel()
    if numel == 0:
        return torch.empty(t.shape, dtype=_generation_dtype(t.dtype))
    return torch.arange(
        numel, dtype=_generation_dtype(t.dtype)
    ).reshape(t.shape) / numel


def _relop_boundary_hit(t: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Integer-like values that hit relational-op decision boundaries.

    Half the elements are placed at exact integers (0, 1, 2, ...),
    so <=N  vs  <N  will differ when an element equals N.
    """
    numel = t.numel()
    if numel == 0:
        return torch.empty(t.shape, dtype=_generation_dtype(t.dtype))
    base = torch.arange(
        numel, dtype=_generation_dtype(t.dtype)
    ).reshape(t.shape)
    return base % 10.0


def _extreme_magnitude(t: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Values in the ±1e6 range — stresses arith overflow paths."""
    return torch.randn(t.shape, dtype=_generation_dtype(t.dtype), generator=g) * 1e6


def _near_epsilon(t: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Values in [1e-7, 1e-5] — stresses epsilon-dependent branches."""
    return (
        torch.rand(t.shape, dtype=_generation_dtype(t.dtype), generator=g)
        * 9e-6
        + 1e-7
    )


def _reduction_adversarial(t: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Alternating large +/- values to maximise FP reduction error."""
    numel = t.numel()
    if numel == 0:
        return torch.empty(t.shape, dtype=_generation_dtype(t.dtype))
    dtype = _generation_dtype(t.dtype)
    base = torch.ones(numel, dtype=dtype) * 1e4
    base[1::2] = -1e4
    noise = torch.randn(numel, dtype=dtype, generator=g) * 0.01
    return (base + noise).reshape(t.shape)


def _init_sensitive(t: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """All-positive (min-init sensitive) or all-negative (max-init sensitive).

    Randomly picks one mode per call so both paths get coverage.
    """
    v = torch.rand(1, generator=g).item()
    if v > 0.5:
        return torch.abs(
            torch.randn(t.shape, dtype=_generation_dtype(t.dtype), generator=g)
        ) * 100.0
    return -torch.abs(
        torch.randn(t.shape, dtype=_generation_dtype(t.dtype), generator=g)
    ) * 100.0


def _boundary_last_element(t: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Normal randn values with an extreme value (1e4) at the last position.

    Targets off-by-one boundary errors where the mutation only affects
    the last element (e.g., mask_boundary rhs-1, relop_replace).
    """
    values = torch.randn(t.shape, dtype=_generation_dtype(t.dtype), generator=g)
    flat = values.view(-1)
    if flat.numel() > 0:
        flat[-1] = 1e4
    return values


def _head_heavy(t: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Extreme values concentrated in the first quarter; near-zero elsewhere.

    Targets index mutations where a 1-D grid causes all blocks to re-process
    only leading elements (e.g., blockIdx.y/z always 0).  If the mutant
    only sees the head, the aggregated result will diverge from the
    reference that processes the full tensor.
    """
    dtype = _generation_dtype(t.dtype)
    values = torch.randn(t.shape, dtype=dtype, generator=g) * 0.01
    flat = values.view(-1)
    if flat.numel() == 0:
        return values
    n_head = max(1, flat.numel() // 4)
    flat[:n_head] = torch.randn(n_head, dtype=dtype, generator=g) * 10000.0
    return values


def _tail_heavy(t: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Near-zero values in front; extreme values in the last quarter.

    Complements head_heavy: exposes index mutations that skip tail elements
    when thread indexing collapses to the beginning of the tensor.
    """
    dtype = _generation_dtype(t.dtype)
    values = torch.randn(t.shape, dtype=dtype, generator=g) * 0.01
    flat = values.view(-1)
    if flat.numel() == 0:
        return values
    n_tail = max(1, flat.numel() // 4)
    flat[-n_tail:] = torch.randn(n_tail, dtype=dtype, generator=g) * 10000.0
    return values


def _dense_nonzero(t: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """All non-zero values (|randn| + 1.0) to eliminate zero-masking equivalence."""
    return torch.randn(
        t.shape, dtype=_generation_dtype(t.dtype), generator=g
    ).abs() + 1.0


def _sparse_extreme(t: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """99% zeros + 1% extreme values (randn * 1e4) for boundary/propagation testing."""
    dtype = _generation_dtype(t.dtype)
    values = torch.zeros(t.shape, dtype=dtype)
    mask = torch.rand(t.shape, generator=g) > 0.99
    n = mask.sum().item()
    if n > 0:
        values[mask] = torch.randn(n, dtype=dtype, generator=g) * 1e4
    return values


STRESS_POLICIES: Dict[str, Callable[[List[Any], int], List[Any]]] = {
    "large_magnitude":  _make_policy(_large_magnitude),
    "near_overflow":    _make_policy(_near_overflow),
    "near_zero":        _make_policy(_near_zero),
    "denormals":        _make_policy(_denormals),
    "all_negative":     _make_policy(_all_negative),
    "all_positive":     _make_policy(_all_positive),
    "mixed_extremes":   _make_policy(_mixed_extremes),
    "alternating_sign": _make_policy(_alternating_sign),
    "sparse":           _make_policy(_sparse),
    "uniform_constant": _make_policy(_uniform_constant),
    "structured_ramp":          _make_policy(_structured_ramp),
    "boundary_last_element":    _make_policy(_boundary_last_element),
    "head_heavy":               _make_policy(_head_heavy),
    "tail_heavy":               _make_policy(_tail_heavy),
    # operator-directed policies (V2)
    "relop_boundary_hit":       _make_policy(_relop_boundary_hit),
    "extreme_magnitude":        _make_policy(_extreme_magnitude),
    "near_epsilon":             _make_policy(_near_epsilon),
    "reduction_adversarial":    _make_policy(_reduction_adversarial),
    "init_sensitive":           _make_policy(_init_sensitive),
    # sparsity-gradient policies (V2.1)
    "dense_nonzero":            _make_policy(_dense_nonzero),
    "sparse_extreme":           _make_policy(_sparse_extreme),
}


def get_all_policy_names() -> List[str]:
    return list(STRESS_POLICIES.keys())


def generate_stress_inputs(
    get_inputs_fn: Callable,
    policy_name: str,
    seed: int = 30000,
) -> List[Any]:
    """Generate one set of stress inputs using the named policy."""
    torch.manual_seed(seed)
    template = get_inputs_fn()
    policy_fn = STRESS_POLICIES[policy_name]
    return policy_fn(template, seed)
