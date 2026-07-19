"""Recursive input cloning for isolated reference and candidate execution."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from collections.abc import Mapping
from typing import Any, Dict, Hashable, List, Tuple

import torch


class InputIsolationError(RuntimeError):
    """Raised when an input tree cannot be cloned safely."""


def _clone_strided_tensor(
    value: torch.Tensor,
    storage_memo: Dict[Hashable, torch.Tensor],
) -> torch.Tensor:
    """Clone a strided tensor without normalising its view geometry.

    ``Tensor.clone(memory_format=torch.preserve_format)`` is allowed to make a
    non-dense view contiguous.  That silently invalidates stride/layout stress
    tests.  Cloning the underlying typed storage once and rebuilding each view
    preserves size, stride, storage offset, overlap, and storage aliasing among
    distinct tensor arguments in the same call.
    """

    storage = value.untyped_storage()
    storage_identity = getattr(storage, "_cdata", None)
    if storage_identity is None:  # pragma: no cover - fallback for other backends.
        storage_identity = (storage.data_ptr(), storage.nbytes())
    storage_key = (str(value.device), value.dtype, storage_identity)
    cloned_storage = storage_memo.get(storage_key)
    if cloned_storage is None:
        element_size = value.element_size()
        if element_size <= 0 or storage.nbytes() % element_size != 0:
            raise InputIsolationError(
                "tensor storage size is incompatible with its element size"
            )
        storage_elements = storage.nbytes() // element_size
        flat = torch.as_strided(
            value.detach(),
            size=(storage_elements,),
            stride=(1,),
            storage_offset=0,
        )
        cloned_storage = flat.clone()
        storage_memo[storage_key] = cloned_storage
    cloned = torch.as_strided(
        cloned_storage,
        size=value.size(),
        stride=value.stride(),
        storage_offset=value.storage_offset(),
    )
    cloned.requires_grad_(value.requires_grad)
    return cloned


def _clone_tree(
    value: Any,
    memo: Dict[int, Any],
    storage_memo: Dict[Hashable, torch.Tensor],
) -> Any:
    object_id = id(value)
    if object_id in memo:
        return memo[object_id]

    if isinstance(value, torch.Tensor):
        try:
            if value.layout == torch.strided and not value.is_quantized:
                cloned = _clone_strided_tensor(value, storage_memo)
            else:
                cloned = value.detach().clone(memory_format=torch.preserve_format)
                cloned.requires_grad_(value.requires_grad)
        except Exception as exc:
            raise InputIsolationError(
                f"cannot preserve tensor view geometry: {type(exc).__name__}: {exc}"
            ) from exc
        memo[object_id] = cloned
        return cloned

    if value is None or isinstance(value, (str, bytes, int, float, complex, bool)):
        return value

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        cloned = copy.copy(value)
        memo[object_id] = cloned
        for field in dataclasses.fields(value):
            object.__setattr__(
                cloned,
                field.name,
                _clone_tree(getattr(value, field.name), memo, storage_memo),
            )
        return cloned

    if isinstance(value, Mapping):
        try:
            cloned = copy.copy(value)
            cloned.clear()
        except Exception as exc:
            raise InputIsolationError(
                f"cannot clone mapping type {type(value).__name__}: {exc}"
            ) from exc
        memo[object_id] = cloned
        for key, item in value.items():
            cloned[_clone_tree(key, memo, storage_memo)] = _clone_tree(
                item, memo, storage_memo
            )
        return cloned

    if isinstance(value, list):
        cloned_list = []
        memo[object_id] = cloned_list
        cloned_list.extend(_clone_tree(item, memo, storage_memo) for item in value)
        return cloned_list

    if isinstance(value, tuple):
        if hasattr(value, "_fields"):  # namedtuple
            cloned_tuple = type(value)(
                *(_clone_tree(item, memo, storage_memo) for item in value)
            )
        else:
            cloned_tuple = tuple(
                _clone_tree(item, memo, storage_memo) for item in value
            )
        memo[object_id] = cloned_tuple
        return cloned_tuple

    if isinstance(value, set):
        cloned_set = {_clone_tree(item, memo, storage_memo) for item in value}
        memo[object_id] = cloned_set
        return cloned_set

    if isinstance(value, frozenset):
        cloned_frozen = frozenset(
            _clone_tree(item, memo, storage_memo) for item in value
        )
        memo[object_id] = cloned_frozen
        return cloned_frozen

    try:
        cloned = copy.deepcopy(value, memo)
    except Exception as exc:
        raise InputIsolationError(
            f"cannot safely clone input type {type(value).__name__}: {exc}"
        ) from exc
    memo[object_id] = cloned
    return cloned


def clone_tree(value: Any) -> Any:
    """Recursively clone an input tree while preserving repeated aliases."""

    return _clone_tree(value, {}, {})


def clone_call_inputs(
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """Create one isolated ``(*args, **kwargs)`` tree.

    A single memo is shared across args and kwargs so repeated references retain
    their alias relationship *within* a call.  Calling this function twice
    creates fully separate trees for reference and candidate execution.
    """

    cloned = clone_tree((args, dict(kwargs)))
    cloned_args, cloned_kwargs = cloned
    return tuple(cloned_args), dict(cloned_kwargs)


def describe_input_tree(value: Any, *, hash_contents: bool = False) -> Dict[str, Any]:
    """Return replay-oriented tensor metadata and, optionally, an exact digest.

    The digest covers tensor metadata and logical values, not backend-specific
    padding bytes.  Alias groups are recorded separately so replay can verify
    relationships between arguments without persisting potentially huge input
    tensors in every observation.
    """

    leaves: List[Dict[str, Any]] = []
    storage_groups: Dict[Hashable, int] = {}
    hasher = hashlib.sha256() if hash_contents else None

    def update_digest(payload: Any) -> None:
        if hasher is not None:
            hasher.update(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    default=repr,
                ).encode("utf-8")
            )
            hasher.update(b"\0")

    def tensor_bytes(tensor: torch.Tensor) -> bytes:
        detached = tensor.detach()
        if detached.layout != torch.strided:
            detached = detached.to_dense()
        cpu = detached.to(device="cpu").contiguous()
        if cpu.numel() == 0:
            return b""
        return cpu.view(torch.uint8).numpy().tobytes(order="C")

    def visit(item: Any, path: str) -> None:
        if isinstance(item, torch.Tensor):
            alias_group = None
            if item.layout == torch.strided:
                storage = item.untyped_storage()
                storage_identity = getattr(storage, "_cdata", None)
                if storage_identity is None:  # pragma: no cover - backend fallback.
                    storage_identity = (storage.data_ptr(), storage.nbytes())
                storage_key = (str(item.device), storage_identity)
                if storage_key not in storage_groups:
                    storage_groups[storage_key] = len(storage_groups)
                alias_group = storage_groups[storage_key]
            record = {
                "path": path,
                "kind": "tensor",
                "shape": list(item.shape),
                "dtype": str(item.dtype).removeprefix("torch."),
                "device": str(item.device),
                "layout": str(item.layout).removeprefix("torch."),
                "stride": list(item.stride()) if item.layout == torch.strided else None,
                "storage_offset": (
                    int(item.storage_offset()) if item.layout == torch.strided else None
                ),
                "requires_grad": bool(item.requires_grad),
                "alias_group": alias_group,
            }
            leaves.append(record)
            update_digest(record)
            if hasher is not None:
                hasher.update(tensor_bytes(item))
                hasher.update(b"\0")
            return
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            update_digest({"path": path, "container": type(item).__qualname__})
            for field in dataclasses.fields(item):
                visit(getattr(item, field.name), f"{path}.{field.name}")
            return
        if isinstance(item, Mapping):
            update_digest({"path": path, "mapping_type": type(item).__qualname__})
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                visit(item[key], f"{path}[{key!r}]")
            return
        if isinstance(item, (tuple, list)):
            update_digest(
                {"path": path, "sequence_type": type(item).__qualname__, "length": len(item)}
            )
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
            return
        record = {
            "path": path,
            "kind": "scalar",
            "type": type(item).__qualname__,
            "value": repr(item),
        }
        leaves.append(record)
        update_digest(record)

    visit(value, "$")
    result: Dict[str, Any] = {"schema_version": "1.0", "leaves": leaves}
    if hasher is not None:
        result["content_sha256"] = hasher.hexdigest()
    return result
