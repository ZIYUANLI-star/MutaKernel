"""Stable identities for validation strategies and their planned test cases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Mapping, Optional

from .manifest import stable_json_sha256


STRATEGY_SCHEMA_VERSION = "1.0"
TEST_ID_NAMESPACE = "mutakernel.test-case.v1"


@dataclass(frozen=True)
class StrategySpec:
    """Versioned strategy identity independent of execution order."""

    name: str
    version: str = "1"
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("strategy name and version must not be empty")
        # Hashing here validates that parameters are canonical-JSON compatible.
        stable_json_sha256(self.parameters)

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": STRATEGY_SCHEMA_VERSION,
            "name": self.name,
            "version": self.version,
            "parameters": self.parameters,
        }

    @property
    def strategy_id(self) -> str:
        return stable_json_sha256(self.identity_payload())


def make_test_id(
    *,
    subject_id: str,
    strategy_id: str,
    policy: str,
    seed: int,
    mode: str = "eval",
    scope: str = "in_contract",
    parameters: Optional[Mapping[str, Any]] = None,
    replicate: int = 0,
) -> str:
    """Create a full SHA-256 identity from semantic test-case fields.

    No schedule index or timestamp is included, so reordering a plan leaves all
    test ids unchanged.  ``replicate`` must be explicit when otherwise
    identical executions are intentionally repeated.
    """

    for name, value in (
        ("subject_id", subject_id),
        ("strategy_id", strategy_id),
        ("policy", policy),
        ("mode", mode),
        ("scope", scope),
    ):
        if not value:
            raise ValueError(f"{name} must not be empty")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(replicate, bool) or not isinstance(replicate, int):
        raise TypeError("replicate must be an integer")
    if replicate < 0:
        raise ValueError("replicate must be non-negative")

    payload = {
        "namespace": TEST_ID_NAMESPACE,
        "subject_id": subject_id,
        "strategy_id": strategy_id,
        "policy": policy,
        "seed": seed,
        "mode": mode,
        "scope": scope,
        "parameters": parameters or {},
        "replicate": replicate,
    }
    return stable_json_sha256(payload)


@dataclass(frozen=True)
class TestCaseSpec:
    # The public type name is intentional; prevent pytest from mistaking it for
    # a test container when imported into a test module.
    __test__: ClassVar[bool] = False

    subject_id: str
    strategy: StrategySpec
    policy: str
    seed: int
    mode: str = "eval"
    scope: str = "in_contract"
    parameters: Mapping[str, Any] = field(default_factory=dict)
    replicate: int = 0

    @property
    def candidate_run_cost(self) -> int:
        """Number of candidate invocations consumed by this planned case."""

        if self.mode != "repeated":
            return 1
        repeat_count = self.parameters.get("repeat_count")
        if isinstance(repeat_count, bool) or not isinstance(repeat_count, int):
            raise ValueError("repeated cases require an integer repeat_count")
        if repeat_count < 2:
            raise ValueError("repeated cases require repeat_count >= 2")
        return repeat_count

    @property
    def test_id(self) -> str:
        return make_test_id(
            subject_id=self.subject_id,
            strategy_id=self.strategy.strategy_id,
            policy=self.policy,
            seed=self.seed,
            mode=self.mode,
            scope=self.scope,
            parameters=self.parameters,
            replicate=self.replicate,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "subject_id": self.subject_id,
            "strategy_id": self.strategy.strategy_id,
            "strategy_name": self.strategy.name,
            "strategy_version": self.strategy.version,
            "policy": self.policy,
            "seed": self.seed,
            "mode": self.mode,
            "scope": self.scope,
            "parameters": self.parameters,
            "replicate": self.replicate,
            "candidate_run_cost": self.candidate_run_cost,
        }
