"""Agent methods for kernel generation."""

from .base import BaseMethod, MethodResult
from .naive_cc import NaiveCCMethod
from .normal_cc import NormalCCMethod
from .naive_opencode import NaiveOpenCodeMethod
from .normal_opencode import NormalOpenCodeMethod

_METHODS = {
    "naive_cc": NaiveCCMethod,
    "normal_cc": NormalCCMethod,
    "naive_opencode": NaiveOpenCodeMethod,
    "normal_opencode": NormalOpenCodeMethod,
}


def get_method(name: str) -> BaseMethod:
    if name not in _METHODS:
        available = ", ".join(_METHODS.keys())
        raise ValueError(f"Unknown method: {name}. Available: {available}")
    return _METHODS[name]()


def list_methods() -> list[str]:
    return list(_METHODS.keys())


__all__ = ["BaseMethod", "MethodResult", "get_method", "list_methods",
           "NaiveCCMethod", "NormalCCMethod", "NaiveOpenCodeMethod", "NormalOpenCodeMethod"]
