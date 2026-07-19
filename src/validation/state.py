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


def strict_sync_state_dict(reference: Any, candidate: Any) -> StateSyncReport:
    """Copy state by exact key, never by registration or iteration order.

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
    reference_keys = set(reference_state.keys())
    candidate_keys = set(candidate_state.keys())
    if reference_keys != candidate_keys:
        missing = sorted(reference_keys - candidate_keys)
        unexpected = sorted(candidate_keys - reference_keys)
        raise StateSyncError(
            "state_dict keys differ; refusing positional synchronization: "
            f"missing_in_candidate={missing}, unexpected_in_candidate={unexpected}"
        )

    tensor_count = 0
    for key in reference_state.keys():
        tensor_count += int(
            _check_state_value_compatibility(
                key,
                reference_state[key],
                candidate_state[key],
            )
        )

    source = snapshot_state_dict(reference)
    assert source is not None
    try:
        load_result = candidate.load_state_dict(source, strict=True)
    except Exception as exc:
        raise StateSyncError(f"strict state_dict load failed: {exc}") from exc
    missing = tuple(getattr(load_result, "missing_keys", ()))
    unexpected = tuple(getattr(load_result, "unexpected_keys", ()))
    if missing or unexpected:
        raise StateSyncError(
            f"strict state synchronization failed: missing={missing}, unexpected={unexpected}"
        )

    synchronized = candidate.state_dict()
    if set(synchronized.keys()) != reference_keys:
        raise StateSyncError("candidate state_dict keys changed during synchronization")
    for key in reference_state.keys():
        _assert_state_values_equal(key, reference_state[key], synchronized[key])

    return StateSyncReport(
        keys_synced=len(reference_state),
        tensor_values_synced=tensor_count,
    )
