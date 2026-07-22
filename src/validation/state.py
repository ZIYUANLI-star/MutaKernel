"""Strict module-state synchronization and reproducible RNG replay."""

from __future__ import annotations

import copy
import random
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional, Tuple

import torch

try:  # NumPy is optional for the core package.
    import numpy as np
except ImportError:  # pragma: no cover - exercised only in minimal installs.
    np = None


class StateSyncError(RuntimeError):
    """Raised when two callables cannot be aligned by exact state-dict keys."""


@dataclass(frozen=True)
class StateSyncReport:
    keys_synced: int
    tensor_values_synced: int
    remapped_keys: int = 0
    # reference_key -> candidate_key pairs used for the synchronization;
    # ``None`` when both sides were stateless.
    key_map: Optional[Tuple[Tuple[str, str], ...]] = None


@dataclass
class RNGSnapshot:
    """Python, NumPy, Torch CPU, and (when available) Torch CUDA RNG state."""

    python_state: object
    numpy_state: Optional[Tuple[Any, ...]]
    torch_cpu_state: torch.Tensor
    torch_cuda_states: Optional[Tuple[torch.Tensor, ...]]

    @classmethod
    def capture(cls, include_cuda: bool = True) -> "RNGSnapshot":
        numpy_state = None if np is None else copy.deepcopy(np.random.get_state())
        cuda_states = None
        if include_cuda and torch.cuda.is_available():
            cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        return cls(
            python_state=copy.deepcopy(random.getstate()),
            numpy_state=numpy_state,
            torch_cpu_state=torch.random.get_rng_state().clone(),
            torch_cuda_states=cuda_states,
        )

    def restore(self) -> None:
        random.setstate(copy.deepcopy(self.python_state))
        if self.numpy_state is not None:
            if np is None:
                raise RuntimeError("NumPy RNG state was captured but NumPy is unavailable")
            np.random.set_state(copy.deepcopy(self.numpy_state))
        torch.random.set_rng_state(self.torch_cpu_state.clone())
        if self.torch_cuda_states is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA RNG state was captured but CUDA is unavailable")
            if len(self.torch_cuda_states) != torch.cuda.device_count():
                raise RuntimeError(
                    "CUDA device count changed between RNG capture and replay: "
                    f"captured={len(self.torch_cuda_states)}, "
                    f"current={torch.cuda.device_count()}"
                )
            torch.cuda.set_rng_state_all([state.clone() for state in self.torch_cuda_states])


@contextmanager
def replay_rng(snapshot: RNGSnapshot) -> Iterator[None]:
    """Temporarily replay ``snapshot`` and restore the caller's RNG afterward."""

    caller_state = RNGSnapshot.capture(
        include_cuda=snapshot.torch_cuda_states is not None,
    )
    snapshot.restore()
    try:
        yield
    finally:
        caller_state.restore()


def _has_state_api(value: Any) -> bool:
    return callable(getattr(value, "state_dict", None)) and callable(
        getattr(value, "load_state_dict", None)
    )


def _clone_state_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone(memory_format=torch.preserve_format)
    return copy.deepcopy(value)


def snapshot_state_dict(module: Any) -> Optional["OrderedDict[str, Any]"]:
    """Clone a module state dict, retaining PyTorch's private metadata."""

    if not _has_state_api(module):
        return None
    raw = module.state_dict()
    snapshot = OrderedDict((key, _clone_state_value(value)) for key, value in raw.items())
    if hasattr(raw, "_metadata"):
        snapshot._metadata = copy.deepcopy(raw._metadata)  # type: ignore[attr-defined]
    return snapshot


def _check_state_value_compatibility(key: str, source: Any, target: Any) -> bool:
    source_is_tensor = isinstance(source, torch.Tensor)
    target_is_tensor = isinstance(target, torch.Tensor)
    if source_is_tensor != target_is_tensor:
        raise StateSyncError(
            f"state entry {key!r} changes kind: "
            f"reference={type(source).__name__}, candidate={type(target).__name__}"
        )
    if not source_is_tensor:
        if type(source) is not type(target):
            raise StateSyncError(
                f"state entry {key!r} changes type: "
                f"reference={type(source).__name__}, candidate={type(target).__name__}"
            )
        return False
    if source.shape != target.shape:
        raise StateSyncError(
            f"state entry {key!r} changes shape: "
            f"reference={tuple(source.shape)}, candidate={tuple(target.shape)}"
        )
    if source.dtype != target.dtype:
        raise StateSyncError(
            f"state entry {key!r} changes dtype: "
            f"reference={source.dtype}, candidate={target.dtype}"
        )
    if source.layout != target.layout:
        raise StateSyncError(
            f"state entry {key!r} changes layout: "
            f"reference={source.layout}, candidate={target.layout}"
        )
    return True


def _state_values_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            return False
        expected_on_target = expected.detach().to(actual.device)
        try:
            torch.testing.assert_close(
                actual,
                expected_on_target,
                rtol=0,
                atol=0,
                equal_nan=True,
                check_device=False,
                check_dtype=True,
                check_layout=True,
            )
        except AssertionError:
            return False
        return True
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, Mapping):
        return (
            set(expected.keys()) == set(actual.keys())
            and all(_state_values_equal(expected[key], actual[key]) for key in expected)
        )
    if isinstance(expected, (tuple, list)):
        return len(expected) == len(actual) and all(
            _state_values_equal(left, right) for left, right in zip(expected, actual)
        )
    try:
        return bool(actual == expected)
    except (RuntimeError, TypeError, ValueError):
        return False


def _assert_state_values_equal(key: str, expected: Any, actual: Any) -> None:
    if not _state_values_equal(expected, actual):
        raise StateSyncError(f"state entry {key!r} did not load exactly")


def restore_state_dict(module: Any, snapshot: Optional[Mapping[str, Any]]) -> None:
    """Restore a previously captured state dict with strict key checking."""

    if snapshot is None:
        if _has_state_api(module):
            raise StateSyncError("missing state snapshot for a stateful callable")
        return
    if not _has_state_api(module):
        raise StateSyncError("cannot restore state on a stateless callable")
    result = module.load_state_dict(snapshot, strict=True)
    missing = tuple(getattr(result, "missing_keys", ()))
    unexpected = tuple(getattr(result, "unexpected_keys", ()))
    if missing or unexpected:
        raise StateSyncError(
            f"strict state restoration failed: missing={missing}, unexpected={unexpected}"
        )


def _state_signature(value: Any) -> Tuple[Any, ...]:
    """Structural signature used for unambiguous key alignment."""

    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), str(value.dtype), str(value.layout))
    return ("other", type(value).__name__)


def align_state_keys(
    reference_state: Mapping[str, Any],
    candidate_state: Mapping[str, Any],
) -> "OrderedDict[str, str]":
    """Align reference state keys to candidate state keys by name and structure.

    Returns an ordered mapping ``reference_key -> candidate_key``.  Alignment
    is accepted only when it is unambiguous:

    1. keys present verbatim on both sides map to themselves;
    2. remaining keys are matched by their final dotted component (e.g.
       ``bn.running_mean`` <-> ``running_mean``) when the pairing is unique in
       both directions and the values are structurally compatible;
    3. remaining keys are matched by structural signature (kind, shape,
       dtype, layout) when the signature occurs exactly once on each side.

    Any leftover key on either side, and any ambiguous pairing, raises
    :class:`StateSyncError` — a wrong synchronization would silently corrupt
    the differential comparison, so refusal (INCONCLUSIVE upstream) is the
    only sound fallback.
    """

    reference_keys = list(reference_state.keys())
    candidate_keys = list(candidate_state.keys())
    if set(reference_keys) == set(candidate_keys):
        return OrderedDict((key, key) for key in reference_keys)

    mapping: "OrderedDict[str, str]" = OrderedDict()
    shared = set(reference_keys) & set(candidate_keys)
    for key in reference_keys:
        if key in shared:
            mapping[key] = key
    pending_reference = [key for key in reference_keys if key not in shared]
    pending_candidate = [key for key in candidate_keys if key not in shared]

    def _leaf(key: str) -> str:
        return key.rsplit(".", 1)[-1]

    # Stage 2: unique final-component (leaf name) pairing.
    reference_by_leaf: dict = {}
    for key in pending_reference:
        reference_by_leaf.setdefault(_leaf(key), []).append(key)
    candidate_by_leaf: dict = {}
    for key in pending_candidate:
        candidate_by_leaf.setdefault(_leaf(key), []).append(key)
    matched_reference = set()
    matched_candidate = set()
    for leaf, reference_group in reference_by_leaf.items():
        candidate_group = candidate_by_leaf.get(leaf, [])
        if len(reference_group) == 1 and len(candidate_group) == 1:
            ref_key, cand_key = reference_group[0], candidate_group[0]
            if _state_signature(reference_state[ref_key]) != _state_signature(
                candidate_state[cand_key]
            ):
                raise StateSyncError(
                    f"state keys {ref_key!r} and {cand_key!r} share the leaf name "
                    f"{leaf!r} but are structurally incompatible: "
                    f"reference={_state_signature(reference_state[ref_key])}, "
                    f"candidate={_state_signature(candidate_state[cand_key])}"
                )
            mapping[ref_key] = cand_key
            matched_reference.add(ref_key)
            matched_candidate.add(cand_key)
    pending_reference = [k for k in pending_reference if k not in matched_reference]
    pending_candidate = [k for k in pending_candidate if k not in matched_candidate]

    # Stage 3: unique structural-signature pairing.
    reference_by_signature: dict = {}
    for key in pending_reference:
        reference_by_signature.setdefault(_state_signature(reference_state[key]), []).append(key)
    candidate_by_signature: dict = {}
    for key in pending_candidate:
        candidate_by_signature.setdefault(_state_signature(candidate_state[key]), []).append(key)
    for signature, reference_group in reference_by_signature.items():
        candidate_group = candidate_by_signature.get(signature, [])
        if len(reference_group) == 1 and len(candidate_group) == 1:
            mapping[reference_group[0]] = candidate_group[0]
            matched_reference.add(reference_group[0])
            matched_candidate.add(candidate_group[0])
    pending_reference = [k for k in pending_reference if k not in matched_reference]
    pending_candidate = [k for k in pending_candidate if k not in matched_candidate]

    if pending_reference or pending_candidate:
        raise StateSyncError(
            "state_dict keys differ and cannot be aligned unambiguously by "
            "name normalization (leaf-name and structural matching): "
            f"unmatched_reference={sorted(pending_reference)}, "
            f"unmatched_candidate={sorted(pending_candidate)}"
        )

    # Preserve reference state-dict order for deterministic reporting.
    ordered = OrderedDict()
    for key in reference_keys:
        ordered[key] = mapping[key]
    return ordered


def strict_sync_state_dict(reference: Any, candidate: Any) -> StateSyncReport:
    """Copy state by exact key, never by registration or iteration order.

    When the two state dicts use different (but unambiguously alignable)
    parameter names — e.g. a candidate kernel registering ``weight`` where
    the reference registers ``gemm.weight`` — the keys are first aligned by
    :func:`align_state_keys`; ambiguous alignments still fail explicitly.

    Stateless callables are supported when *both* sides are stateless.  If only
    one side exposes the PyTorch state-dict API, the comparison is unsound and
    synchronization fails explicitly.
    """

    reference_has_state = _has_state_api(reference)
    candidate_has_state = _has_state_api(candidate)
    if not reference_has_state and not candidate_has_state:
        return StateSyncReport(keys_synced=0, tensor_values_synced=0)
    if reference_has_state != candidate_has_state:
        raise StateSyncError(
            "reference and candidate must either both expose state_dict/load_state_dict "
            "or both be stateless"
        )

    reference_state = reference.state_dict()
    candidate_state = candidate.state_dict()
    key_map = align_state_keys(reference_state, candidate_state)
    remapped = sum(1 for ref_key, cand_key in key_map.items() if ref_key != cand_key)

    tensor_count = 0
    for ref_key, cand_key in key_map.items():
        label = ref_key if ref_key == cand_key else f"{ref_key} -> {cand_key}"
        tensor_count += int(
            _check_state_value_compatibility(
                label,
                reference_state[ref_key],
                candidate_state[cand_key],
            )
        )

    source = snapshot_state_dict(reference)
    assert source is not None
    remapped_source = OrderedDict(
        (key_map[ref_key], source[ref_key]) for ref_key in key_map
    )
    if hasattr(source, "_metadata") and remapped == 0:
        remapped_source._metadata = source._metadata  # type: ignore[attr-defined]
    try:
        load_result = candidate.load_state_dict(remapped_source, strict=True)
    except Exception as exc:
        raise StateSyncError(f"strict state_dict load failed: {exc}") from exc
    missing = tuple(getattr(load_result, "missing_keys", ()))
    unexpected = tuple(getattr(load_result, "unexpected_keys", ()))
    if missing or unexpected:
        raise StateSyncError(
            f"strict state synchronization failed: missing={missing}, unexpected={unexpected}"
        )

    synchronized = candidate.state_dict()
    if set(synchronized.keys()) != set(candidate_state.keys()):
        raise StateSyncError("candidate state_dict keys changed during synchronization")
    for ref_key, cand_key in key_map.items():
        label = ref_key if ref_key == cand_key else f"{ref_key} -> {cand_key}"
        _assert_state_values_equal(label, reference_state[ref_key], synchronized[cand_key])

    return StateSyncReport(
        keys_synced=len(key_map),
        tensor_values_synced=tensor_count,
        remapped_keys=remapped,
        key_map=tuple(key_map.items()),
    )
