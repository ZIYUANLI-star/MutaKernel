"""Pure planning for compute-matched FSE validation experiments.

The planner consumes a subject-provenance manifest and a strategy matrix.  It
does not open, import, compile, or execute candidate/reference artifacts.  Its
only file operations are reading the two JSON inputs, hashing those JSON files,
and exclusively creating the canonical output plan.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .manifest import canonical_json_bytes, sha256_file, stable_json_sha256
from .strategy import StrategySpec, TestCaseSpec
from .contract import (
    ContractError,
    assert_case_in_contract,
    validate_case_parameters,
    validate_contract,
)


PROTOCOL_SCHEMA_VERSION = "1.0"
SUBJECT_MANIFEST_SCHEMA_VERSION = "1.0"
STRATEGY_MATRIX_SCHEMA_VERSION = "1.0"
ALLOWED_SCOPES = frozenset({"in_contract", "extended_contract"})
ALLOWED_MODES = frozenset({"eval", "train", "repeated", "config"})
ANCHOR_STRATEGY_NAME = "five-iid-historical-anchor"
MATRIX_SCOPE_MARKER = "derived"
REQUIRED_STRATEGY_NAMES = frozenset(
    {
        ANCHOR_STRATEGY_NAME,
        "iid-compute-matched",
        "diversified-value",
        "boundary-values",
        "dtype-mode-config",
        "mutakernel-full",
    }
)

_SUBJECT_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "input_spec",
        "subject_count",
        "subjects",
        "subjects_sha256",
        "manifest_sha256",
    }
)
_SUBJECT_FIELDS = frozenset(
    {
        "subject_id",
        "dataset",
        "task_id",
        "language",
        "candidate",
        "reference",
        "contract",
        "contract_sha256",
        "metadata",
        "source",
        "subject_sha256",
    }
)
_ARTIFACT_FIELDS = frozenset({"logical_path", "sha256", "size_bytes", "role"})
_MATRIX_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "matrix_id",
        "experiment_scope",
        "description",
        "candidate_run_budget",
        "strategies",
    }
)
_STRATEGY_FIELDS = frozenset(
    {"name", "version", "budget_matched", "candidate_runs", "parameters", "cases"}
)
_CASE_FIELDS = frozenset(
    {"policy", "seeds", "mode", "scope", "parameters", "replicates"}
)
_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "matrix_id",
        "experiment_scope",
        "subject_manifest_sha256",
        "subject_manifest_file_sha256",
        "strategy_matrix_sha256",
        "strategy_matrix_file_sha256",
        "candidate_run_budget",
        "subject_count",
        "strategy_count",
        "test_case_count",
        "strategies",
        "schedule",
        "schedule_sha256",
        "plan_sha256",
    }
)
_PLAN_STRATEGY_FIELDS = frozenset(
    {
        "strategy_id",
        "name",
        "version",
        "budget_matched",
        "candidate_runs_per_subject",
        "parameters",
    }
)
_SCHEDULE_FIELDS = frozenset(
    {
        "test_id",
        "subject_id",
        "strategy_id",
        "strategy_name",
        "strategy_version",
        "policy",
        "seed",
        "mode",
        "scope",
        "parameters",
        "replicate",
        "candidate_run_cost",
        "order",
        "budget_matched",
        "strategy_candidate_run_budget",
        "dataset",
        "task_id",
    }
)


class ProtocolError(ValueError):
    """Invalid or internally inconsistent experiment protocol input."""


@dataclass(frozen=True)
class PlannedStrategy:
    spec: StrategySpec
    budget_matched: bool
    candidate_runs: int
    case_templates: Tuple[Mapping[str, Any], ...]


def _require_exact_fields(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    context: str,
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ProtocolError(f"{context} has unknown fields: {sorted(unknown)}")


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{context} must be a non-empty string")
    return value


def _require_positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProtocolError(f"{context} must be a positive integer")
    return value


def _validate_sha256(value: Any, context: str) -> str:
    digest = _require_nonempty_string(value, context)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ProtocolError(f"{context} must be a lowercase SHA-256 digest")
    return digest


def _validate_subject_manifest(data: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
    _require_exact_fields(data, _SUBJECT_TOP_FIELDS, "subject manifest")
    if data.get("schema_version") != SUBJECT_MANIFEST_SCHEMA_VERSION:
        raise ProtocolError("unsupported subject manifest schema_version")

    supplied_manifest_hash = _validate_sha256(
        data.get("manifest_sha256"), "subject manifest manifest_sha256"
    )
    hash_payload = dict(data)
    del hash_payload["manifest_sha256"]
    if stable_json_sha256(hash_payload) != supplied_manifest_hash:
        raise ProtocolError("subject manifest digest mismatch")

    subjects = data.get("subjects")
    if not isinstance(subjects, list) or not subjects:
        raise ProtocolError("subject manifest subjects must be a non-empty list")
    if data.get("subject_count") != len(subjects):
        raise ProtocolError("subject_count does not match subjects")

    validated: List[Mapping[str, Any]] = []
    subject_ids: List[str] = []
    for index, subject in enumerate(subjects):
        if not isinstance(subject, Mapping):
            raise ProtocolError(f"subjects[{index}] must be an object")
        _require_exact_fields(subject, _SUBJECT_FIELDS, f"subjects[{index}]")
        subject_id = _require_nonempty_string(subject.get("subject_id"), f"subjects[{index}].subject_id")
        subject_ids.append(subject_id)
        for name in ("dataset", "task_id", "language"):
            _require_nonempty_string(subject.get(name), f"subjects[{index}].{name}")
        for artifact_name in ("candidate", "reference"):
            artifact = subject.get(artifact_name)
            if not isinstance(artifact, Mapping):
                raise ProtocolError(f"subjects[{index}].{artifact_name} must be an object")
            _require_exact_fields(
                artifact, _ARTIFACT_FIELDS, f"subjects[{index}].{artifact_name}"
            )
            _require_nonempty_string(
                artifact.get("logical_path"),
                f"subjects[{index}].{artifact_name}.logical_path",
            )
            _validate_sha256(
                artifact.get("sha256"), f"subjects[{index}].{artifact_name}.sha256"
            )
            size = artifact.get("size_bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ProtocolError(
                    f"subjects[{index}].{artifact_name}.size_bytes must be non-negative"
                )
        for mapping_name in ("contract", "metadata", "source"):
            if not isinstance(subject.get(mapping_name), Mapping):
                raise ProtocolError(f"subjects[{index}].{mapping_name} must be an object")
        if stable_json_sha256(subject["contract"]) != subject.get("contract_sha256"):
            raise ProtocolError(f"subjects[{index}] contract digest mismatch")
        try:
            normalized_contract = validate_contract(subject["contract"])
        except ContractError as exc:
            raise ProtocolError(f"subjects[{index}] contract is invalid: {exc}") from exc
        if normalized_contract != subject["contract"]:
            raise ProtocolError(f"subjects[{index}] contract is not canonical")
        supplied_subject_hash = _validate_sha256(
            subject.get("subject_sha256"), f"subjects[{index}].subject_sha256"
        )
        subject_payload = dict(subject)
        del subject_payload["subject_sha256"]
        if stable_json_sha256(subject_payload) != supplied_subject_hash:
            raise ProtocolError(f"subjects[{index}] digest mismatch")
        validated.append(subject)

    if len(subject_ids) != len(set(subject_ids)):
        raise ProtocolError("subject_id values must be unique")
    if stable_json_sha256(subjects) != data.get("subjects_sha256"):
        raise ProtocolError("subjects_sha256 does not match subjects")
    return tuple(sorted(validated, key=lambda subject: subject["subject_id"]))


def _validate_case_template(
    raw: Mapping[str, Any],
    strategy_name: str,
    case_index: int,
) -> Tuple[Mapping[str, Any], int]:
    context = f"strategy {strategy_name!r} cases[{case_index}]"
    _require_exact_fields(raw, _CASE_FIELDS, context)
    policy = _require_nonempty_string(raw.get("policy"), f"{context}.policy")
    if policy not in {"iid", "identity"}:
        from src.stress.policy_bank import STRESS_POLICIES

        if policy not in STRESS_POLICIES:
            raise ProtocolError(f"{context}.policy is unknown: {policy!r}")
    mode = _require_nonempty_string(raw.get("mode"), f"{context}.mode")
    if mode not in ALLOWED_MODES:
        raise ProtocolError(f"{context}.mode is invalid: {mode!r}")
    scope = _require_nonempty_string(raw.get("scope"), f"{context}.scope")
    if scope != MATRIX_SCOPE_MARKER:
        raise ProtocolError(
            f"{context}.scope must be {MATRIX_SCOPE_MARKER!r}; "
            "the planner derives the executed scope from the frozen contract"
        )
    parameters = raw.get("parameters", {})
    try:
        parameters = validate_case_parameters(parameters, mode)
    except ContractError as exc:
        raise ProtocolError(f"{context}.parameters are invalid: {exc}") from exc
    stable_json_sha256(parameters)
    repeat_count = parameters.get("repeat_count")
    if mode == "repeated":
        if isinstance(repeat_count, bool) or not isinstance(repeat_count, int):
            raise ProtocolError(f"{context} repeated mode requires integer repeat_count")
        if repeat_count < 2:
            raise ProtocolError(f"{context} repeat_count must be at least two")
        per_case_cost = repeat_count
    else:
        if repeat_count is not None:
            raise ProtocolError(f"{context} repeat_count is valid only in repeated mode")
        per_case_cost = 1

    seeds = raw.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ProtocolError(f"{context}.seeds must be a non-empty list")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ProtocolError(f"{context}.seeds must contain only integers")
    if len(seeds) != len(set(seeds)):
        raise ProtocolError(f"{context}.seeds contains duplicates")
    replicates = _require_positive_int(raw.get("replicates", 1), f"{context}.replicates")
    normalized = {
        "policy": policy,
        "seeds": list(seeds),
        "mode": mode,
        "scope": scope,
        "parameters": dict(parameters),
        "replicates": replicates,
    }
    return normalized, len(seeds) * replicates * per_case_cost


def _validate_strategy_matrix(
    data: Mapping[str, Any],
) -> Tuple[str, str, int, Tuple[PlannedStrategy, ...]]:
    _require_exact_fields(data, _MATRIX_TOP_FIELDS, "strategy matrix")
    if data.get("schema_version") != STRATEGY_MATRIX_SCHEMA_VERSION:
        raise ProtocolError("unsupported strategy matrix schema_version")
    matrix_id = _require_nonempty_string(data.get("matrix_id"), "matrix_id")
    experiment_scope = _require_nonempty_string(
        data.get("experiment_scope"), "experiment_scope"
    )
    if experiment_scope not in ALLOWED_SCOPES:
        raise ProtocolError("experiment_scope is invalid")
    budget = _require_positive_int(data.get("candidate_run_budget"), "candidate_run_budget")
    raw_strategies = data.get("strategies")
    if not isinstance(raw_strategies, list) or not raw_strategies:
        raise ProtocolError("strategies must be a non-empty list")

    strategies: List[PlannedStrategy] = []
    names: List[str] = []
    strategy_ids: List[str] = []
    for strategy_index, raw_strategy in enumerate(raw_strategies):
        if not isinstance(raw_strategy, Mapping):
            raise ProtocolError(f"strategies[{strategy_index}] must be an object")
        _require_exact_fields(raw_strategy, _STRATEGY_FIELDS, f"strategies[{strategy_index}]")
        name = _require_nonempty_string(raw_strategy.get("name"), f"strategies[{strategy_index}].name")
        version = _require_nonempty_string(
            raw_strategy.get("version"), f"strategies[{strategy_index}].version"
        )
        budget_matched = raw_strategy.get("budget_matched")
        if not isinstance(budget_matched, bool):
            raise ProtocolError(f"strategy {name!r}.budget_matched must be boolean")
        candidate_runs = _require_positive_int(
            raw_strategy.get("candidate_runs"), f"strategy {name!r}.candidate_runs"
        )
        parameters = raw_strategy.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ProtocolError(f"strategy {name!r}.parameters must be an object")
        raw_cases = raw_strategy.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ProtocolError(f"strategy {name!r}.cases must be a non-empty list")

        templates: List[Mapping[str, Any]] = []
        expanded_count = 0
        for case_index, raw_case in enumerate(raw_cases):
            if not isinstance(raw_case, Mapping):
                raise ProtocolError(f"strategy {name!r} cases[{case_index}] must be an object")
            template, count = _validate_case_template(raw_case, name, case_index)
            templates.append(template)
            expanded_count += count
        if expanded_count != candidate_runs:
            raise ProtocolError(
                f"strategy {name!r} declares {candidate_runs} candidate runs "
                f"but expands to {expanded_count}"
            )

        if name == ANCHOR_STRATEGY_NAME:
            if budget_matched or candidate_runs != 5:
                raise ProtocolError(
                    f"{ANCHOR_STRATEGY_NAME!r} must be a non-matched five-run anchor"
                )
        elif not budget_matched or candidate_runs != budget:
            raise ProtocolError(
                f"strategy {name!r} must be budget-matched at exactly {budget} candidate runs"
            )

        spec = StrategySpec(
            name=name,
            version=version,
            parameters={
                **dict(parameters),
                "budget_matched": budget_matched,
                "candidate_runs": candidate_runs,
            },
        )
        names.append(name)
        strategy_ids.append(spec.strategy_id)
        strategies.append(
            PlannedStrategy(
                spec=spec,
                budget_matched=budget_matched,
                candidate_runs=candidate_runs,
                case_templates=tuple(templates),
            )
        )

    if len(names) != len(set(names)):
        raise ProtocolError("strategy names must be unique")
    if len(strategy_ids) != len(set(strategy_ids)):
        raise ProtocolError("strategy identities must be unique")
    missing = REQUIRED_STRATEGY_NAMES - set(names)
    if missing:
        raise ProtocolError(f"strategy matrix is missing required strategies: {sorted(missing)}")
    return (
        matrix_id,
        experiment_scope,
        budget,
        tuple(sorted(strategies, key=lambda strategy: strategy.spec.name)),
    )


def _expand_cases(
    subject_id: str,
    contract: Mapping[str, Any],
    experiment_scope: str,
    strategy: PlannedStrategy,
) -> Tuple[TestCaseSpec, ...]:
    expanded: List[TestCaseSpec] = []
    seen_ids = set()
    for template in strategy.case_templates:
        for seed in template["seeds"]:
            for replicate in range(template["replicates"]):
                case_for_contract = {
                    "policy": template["policy"],
                    "mode": template["mode"],
                    "parameters": template["parameters"],
                }
                if experiment_scope == "in_contract":
                    try:
                        assert_case_in_contract(case_for_contract, contract)
                    except ContractError as exc:
                        raise ProtocolError(
                            f"subject {subject_id!r}, strategy {strategy.spec.name!r} "
                            f"contains an out-of-contract case: {exc}"
                        ) from exc
                case = TestCaseSpec(
                    subject_id=subject_id,
                    strategy=strategy.spec,
                    policy=template["policy"],
                    seed=seed,
                    mode=template["mode"],
                    scope=experiment_scope,
                    parameters=template["parameters"],
                    replicate=replicate,
                )
                if case.test_id in seen_ids:
                    raise ProtocolError(
                        f"strategy {strategy.spec.name!r} expands duplicate test case {case.test_id}"
                    )
                seen_ids.add(case.test_id)
                expanded.append(case)
    if sum(case.candidate_run_cost for case in expanded) != strategy.candidate_runs:
        raise ProtocolError(f"strategy {strategy.spec.name!r} expansion changed after validation")
    return tuple(expanded)


def build_experiment_plan(
    *,
    subject_manifest: Mapping[str, Any],
    subject_manifest_file_sha256: str,
    strategy_matrix: Mapping[str, Any],
    strategy_matrix_file_sha256: str,
) -> Dict[str, Any]:
    """Validate inputs and deterministically construct a round-robin plan."""

    _validate_sha256(subject_manifest_file_sha256, "subject_manifest_file_sha256")
    _validate_sha256(strategy_matrix_file_sha256, "strategy_matrix_file_sha256")
    subjects = _validate_subject_manifest(subject_manifest)
    matrix_id, experiment_scope, budget, strategies = _validate_strategy_matrix(
        strategy_matrix
    )

    queues: List[Tuple[Mapping[str, Any], PlannedStrategy, Tuple[TestCaseSpec, ...]]] = []
    for subject in subjects:
        for strategy in strategies:
            queues.append(
                (
                    subject,
                    strategy,
                    _expand_cases(
                        subject["subject_id"],
                        subject["contract"],
                        experiment_scope,
                        strategy,
                    ),
                )
            )
    if not queues:
        raise ProtocolError("schedule is empty")

    schedule: List[Dict[str, Any]] = []
    max_queue_length = max(len(queue[2]) for queue in queues)
    order = 1
    for round_index in range(max_queue_length):
        for subject, strategy, cases in queues:
            if round_index >= len(cases):
                continue
            entry = cases[round_index].to_dict()
            entry.update(
                {
                    "order": order,
                    "candidate_run_cost": cases[round_index].candidate_run_cost,
                    "budget_matched": strategy.budget_matched,
                    "strategy_candidate_run_budget": strategy.candidate_runs,
                    "dataset": subject["dataset"],
                    "task_id": subject["task_id"],
                }
            )
            schedule.append(entry)
            order += 1
    if not schedule:
        raise ProtocolError("schedule is empty")
    test_ids = [entry["test_id"] for entry in schedule]
    if len(test_ids) != len(set(test_ids)):
        raise ProtocolError("schedule contains duplicate test ids")

    strategy_summaries = [
        {
            "strategy_id": strategy.spec.strategy_id,
            "name": strategy.spec.name,
            "version": strategy.spec.version,
            "budget_matched": strategy.budget_matched,
            "candidate_runs_per_subject": strategy.candidate_runs,
            "parameters": strategy.spec.parameters,
        }
        for strategy in strategies
    ]
    payload: Dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "matrix_id": matrix_id,
        "experiment_scope": experiment_scope,
        "subject_manifest_sha256": subject_manifest["manifest_sha256"],
        "subject_manifest_file_sha256": subject_manifest_file_sha256,
        "strategy_matrix_sha256": stable_json_sha256(strategy_matrix),
        "strategy_matrix_file_sha256": strategy_matrix_file_sha256,
        "candidate_run_budget": budget,
        "subject_count": len(subjects),
        "strategy_count": len(strategies),
        "test_case_count": len(schedule),
        "strategies": strategy_summaries,
        "schedule": schedule,
        "schedule_sha256": stable_json_sha256(schedule),
    }
    payload["plan_sha256"] = stable_json_sha256(payload)
    return payload


def validate_frozen_plan(
    plan: Mapping[str, Any],
    subject_manifest: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    """Revalidate a frozen plan as an execution-time safety invariant.

    Embedded hashes alone establish integrity, not semantic validity: a broken
    producer can hash an out-of-contract schedule.  Runner, audit, summary, and
    statistics consumers call this function before trusting test IDs.
    """

    if not isinstance(plan, Mapping):
        raise ProtocolError("experiment plan must be an object")
    _require_exact_fields(plan, _PLAN_FIELDS, "experiment plan")
    if plan.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ProtocolError("unsupported experiment plan schema_version")
    supplied_plan_hash = _validate_sha256(plan.get("plan_sha256"), "plan_sha256")
    plan_payload = dict(plan)
    del plan_payload["plan_sha256"]
    if stable_json_sha256(plan_payload) != supplied_plan_hash:
        raise ProtocolError("experiment plan digest mismatch")

    subjects = _validate_subject_manifest(subject_manifest)
    subject_by_id = {str(subject["subject_id"]): subject for subject in subjects}
    if plan.get("subject_manifest_sha256") != subject_manifest.get("manifest_sha256"):
        raise ProtocolError("plan references a different subject manifest")
    if plan.get("subject_count") != len(subjects):
        raise ProtocolError("plan subject_count does not match the subject manifest")
    _validate_sha256(
        plan.get("subject_manifest_file_sha256"),
        "plan subject_manifest_file_sha256",
    )
    _validate_sha256(plan.get("strategy_matrix_sha256"), "strategy_matrix_sha256")
    _validate_sha256(
        plan.get("strategy_matrix_file_sha256"),
        "strategy_matrix_file_sha256",
    )
    experiment_scope = _require_nonempty_string(
        plan.get("experiment_scope"), "experiment_scope"
    )
    if experiment_scope not in ALLOWED_SCOPES:
        raise ProtocolError("plan experiment_scope is invalid")
    budget = _require_positive_int(plan.get("candidate_run_budget"), "candidate_run_budget")

    raw_strategies = plan.get("strategies")
    if not isinstance(raw_strategies, list) or not raw_strategies:
        raise ProtocolError("plan strategies must be a non-empty list")
    strategy_by_id: Dict[str, Mapping[str, Any]] = {}
    for index, strategy in enumerate(raw_strategies):
        if not isinstance(strategy, Mapping):
            raise ProtocolError(f"plan strategies[{index}] must be an object")
        _require_exact_fields(strategy, _PLAN_STRATEGY_FIELDS, f"plan strategies[{index}]")
        name = _require_nonempty_string(strategy.get("name"), f"plan strategies[{index}].name")
        version = _require_nonempty_string(
            strategy.get("version"), f"plan strategies[{index}].version"
        )
        parameters = strategy.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ProtocolError(f"plan strategy {name!r} parameters must be an object")
        expected_id = StrategySpec(name=name, version=version, parameters=parameters).strategy_id
        if strategy.get("strategy_id") != expected_id:
            raise ProtocolError(f"plan strategy {name!r} identity mismatch")
        if expected_id in strategy_by_id:
            raise ProtocolError("plan strategy IDs must be unique")
        matched = strategy.get("budget_matched")
        if not isinstance(matched, bool):
            raise ProtocolError(f"plan strategy {name!r} budget_matched must be boolean")
        declared_runs = _require_positive_int(
            strategy.get("candidate_runs_per_subject"),
            f"plan strategy {name!r} candidate_runs_per_subject",
        )
        if matched and declared_runs != budget:
            raise ProtocolError(f"budget-matched plan strategy {name!r} has the wrong budget")
        strategy_by_id[expected_id] = strategy
    if plan.get("strategy_count") != len(strategy_by_id):
        raise ProtocolError("plan strategy_count is inconsistent")

    schedule = plan.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        raise ProtocolError("plan schedule must be a non-empty list")
    if plan.get("test_case_count") != len(schedule):
        raise ProtocolError("plan test_case_count is inconsistent")
    if plan.get("schedule_sha256") != stable_json_sha256(schedule):
        raise ProtocolError("plan schedule digest mismatch")

    seen_test_ids = set()
    seen_orders = set()
    cost_by_pair: Dict[Tuple[str, str], int] = {}
    validated_schedule: List[Mapping[str, Any]] = []
    for index, entry in enumerate(schedule):
        context = f"plan schedule[{index}]"
        if not isinstance(entry, Mapping):
            raise ProtocolError(f"{context} must be an object")
        _require_exact_fields(entry, _SCHEDULE_FIELDS, context)
        subject_id = _require_nonempty_string(entry.get("subject_id"), f"{context}.subject_id")
        subject = subject_by_id.get(subject_id)
        if subject is None:
            raise ProtocolError(f"{context} references unknown subject {subject_id!r}")
        strategy_id = _require_nonempty_string(entry.get("strategy_id"), f"{context}.strategy_id")
        strategy = strategy_by_id.get(strategy_id)
        if strategy is None:
            raise ProtocolError(f"{context} references unknown strategy")
        for entry_field, strategy_field in (
            ("strategy_name", "name"),
            ("strategy_version", "version"),
            ("budget_matched", "budget_matched"),
            ("strategy_candidate_run_budget", "candidate_runs_per_subject"),
        ):
            if entry.get(entry_field) != strategy.get(strategy_field):
                raise ProtocolError(f"{context} {entry_field} differs from its strategy")
        if entry.get("dataset") != subject.get("dataset") or entry.get("task_id") != subject.get("task_id"):
            raise ProtocolError(f"{context} dataset/task metadata differs from its subject")
        if entry.get("scope") != experiment_scope:
            raise ProtocolError(f"{context} scope differs from the plan scope")

        case = TestCaseSpec(
            subject_id=subject_id,
            strategy=StrategySpec(
                name=str(strategy["name"]),
                version=str(strategy["version"]),
                parameters=strategy["parameters"],
            ),
            policy=_require_nonempty_string(entry.get("policy"), f"{context}.policy"),
            seed=entry.get("seed"),
            mode=_require_nonempty_string(entry.get("mode"), f"{context}.mode"),
            scope=str(entry.get("scope")),
            parameters=entry.get("parameters", {}),
            replicate=entry.get("replicate"),
        )
        try:
            validate_case_parameters(case.parameters, case.mode)
        except ContractError as exc:
            raise ProtocolError(f"{context} parameters are invalid: {exc}") from exc
        if entry.get("test_id") != case.test_id:
            raise ProtocolError(f"{context} test_id does not match its semantic fields")
        if entry.get("candidate_run_cost") != case.candidate_run_cost:
            raise ProtocolError(f"{context} candidate_run_cost is incorrect")
        if case.test_id in seen_test_ids:
            raise ProtocolError("plan schedule test IDs must be unique")
        seen_test_ids.add(case.test_id)
        order = entry.get("order")
        if isinstance(order, bool) or not isinstance(order, int) or order <= 0:
            raise ProtocolError(f"{context} order must be a positive integer")
        if order in seen_orders:
            raise ProtocolError("plan schedule orders must be unique")
        seen_orders.add(order)
        if experiment_scope == "in_contract":
            try:
                assert_case_in_contract(
                    {
                        "policy": case.policy,
                        "mode": case.mode,
                        "parameters": case.parameters,
                    },
                    subject["contract"],
                )
            except ContractError as exc:
                raise ProtocolError(f"{context} is outside the subject contract: {exc}") from exc
        pair = (subject_id, strategy_id)
        cost_by_pair[pair] = cost_by_pair.get(pair, 0) + case.candidate_run_cost
        validated_schedule.append(entry)

    if seen_orders != set(range(1, len(schedule) + 1)):
        raise ProtocolError("plan schedule orders must be contiguous from one")
    expected_pairs = {
        (subject_id, strategy_id)
        for subject_id in subject_by_id
        for strategy_id in strategy_by_id
    }
    if set(cost_by_pair) != expected_pairs:
        raise ProtocolError("plan schedule does not cover every subject/strategy pair")
    for (subject_id, strategy_id), actual_cost in cost_by_pair.items():
        expected_cost = int(strategy_by_id[strategy_id]["candidate_runs_per_subject"])
        if actual_cost != expected_cost:
            raise ProtocolError(
                f"plan schedule budget mismatch for {subject_id}/{strategy_id}: "
                f"expected {expected_cost}, got {actual_cost}"
            )
    return tuple(validated_schedule)


def _read_json_object(path: Path, context: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid {context} JSON at {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{context} must be a JSON object")
    return value


def plan_from_files(subject_manifest_path: Path, strategy_matrix_path: Path) -> Dict[str, Any]:
    """Read only the two JSON inputs and build a content-addressed plan."""

    subject_manifest_path = subject_manifest_path.resolve()
    strategy_matrix_path = strategy_matrix_path.resolve()
    subject_manifest = _read_json_object(subject_manifest_path, "subject manifest")
    strategy_matrix = _read_json_object(strategy_matrix_path, "strategy matrix")
    return build_experiment_plan(
        subject_manifest=subject_manifest,
        subject_manifest_file_sha256=sha256_file(subject_manifest_path),
        strategy_matrix=strategy_matrix,
        strategy_matrix_file_sha256=sha256_file(strategy_matrix_path),
    )


def write_plan_once(path: Path, plan: Mapping[str, Any]) -> None:
    """Exclusively create an immutable canonical plan."""

    supplied = plan.get("plan_sha256")
    if not isinstance(supplied, str):
        raise ProtocolError("plan_sha256 is missing")
    payload = dict(plan)
    del payload["plan_sha256"]
    if stable_json_sha256(payload) != supplied:
        raise ProtocolError("plan_sha256 does not match plan content")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(plan) + b"\n"
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
