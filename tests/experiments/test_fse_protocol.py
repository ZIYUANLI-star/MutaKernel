import copy
import json
from pathlib import Path

import pytest

from scripts.build_fse_subject_manifest import build_subject_manifest, write_new_canonical_json
from scripts.plan_fse_experiment import main
from src.experiments.manifest import sha256_file, stable_json_sha256
from src.experiments.protocol import (
    ProtocolError,
    build_experiment_plan,
    plan_from_files,
    validate_frozen_plan,
    write_plan_once,
)
from src.experiments.strategy import make_test_id
from tests.experiments.contract_fixture import rich_contract


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = PROJECT_ROOT / "configs" / "fse_strategy_matrix.json"


def _make_subject_manifest(tmp_path: Path, count: int = 2) -> Path:
    rows = []
    for index in range(count):
        candidate = tmp_path / f"candidate_{index}.py"
        reference = tmp_path / f"reference_{index}.py"
        candidate.write_text("this is not imported candidate data !!!\n", encoding="utf-8")
        reference.write_text("this is not imported reference data !!!\n", encoding="utf-8")
        rows.append(
            {
                "subject_id": f"dataset/task-{index}",
                "dataset": "dataset",
                "task_id": f"task-{index}",
                "language": "cuda",
                "candidate_path": candidate.name,
                "reference_path": reference.name,
                "contract": rich_contract(),
                "source": {"revision": "abc"},
                "metadata": {},
            }
        )
    spec = tmp_path / "subjects.jsonl"
    spec.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = build_subject_manifest(input_path=spec, root=tmp_path)
    output = tmp_path / "subject_manifest.json"
    write_new_canonical_json(output, manifest)
    return output


def _plan_with_matrix(subject_manifest_path: Path, matrix):
    subject_manifest = json.loads(subject_manifest_path.read_text(encoding="utf-8"))
    return build_experiment_plan(
        subject_manifest=subject_manifest,
        subject_manifest_file_sha256=sha256_file(subject_manifest_path),
        strategy_matrix=matrix,
        strategy_matrix_file_sha256=stable_json_sha256(matrix),
    )


def test_default_matrix_is_budget_matched_and_deterministic(tmp_path: Path):
    subjects = _make_subject_manifest(tmp_path, count=2)
    # Planning must use only the manifest: deleting source artifacts proves the
    # planner does not reopen or import candidate/reference code.
    for source in tmp_path.glob("candidate_*.py"):
        source.unlink()
    for source in tmp_path.glob("reference_*.py"):
        source.unlink()

    first = plan_from_files(subjects, DEFAULT_MATRIX)
    second = plan_from_files(subjects, DEFAULT_MATRIX)
    assert first == second
    assert first["subject_count"] == 2
    assert first["strategy_count"] == 6
    # The repeated-run case consumes two candidate invocations inside one
    # process, so the number of scheduled cases is one below the execution
    # budget for MutaKernel on each subject.
    assert first["test_case_count"] == 2 * (5 + 4 * 32 + 31)
    assert first["schedule_sha256"] == stable_json_sha256(first["schedule"])

    strategies = {strategy["name"]: strategy for strategy in first["strategies"]}
    anchor = strategies["five-iid-historical-anchor"]
    assert anchor["candidate_runs_per_subject"] == 5
    assert anchor["budget_matched"] is False
    for name, strategy in strategies.items():
        if name != "five-iid-historical-anchor":
            assert strategy["budget_matched"] is True
            assert strategy["candidate_runs_per_subject"] == first["candidate_run_budget"]
    cost_by_subject_strategy = {}
    for entry in first["schedule"]:
        key = (entry["subject_id"], entry["strategy_name"])
        cost_by_subject_strategy[key] = (
            cost_by_subject_strategy.get(key, 0) + entry["candidate_run_cost"]
        )
    for (subject_id, strategy_name), cost in cost_by_subject_strategy.items():
        expected = 5 if strategy_name == "five-iid-historical-anchor" else 32
        assert cost == expected, (subject_id, strategy_name)
    assert {entry["scope"] for entry in first["schedule"]} == {"in_contract"}


def test_round_robin_order_is_stable_but_not_part_of_test_id(tmp_path: Path):
    subjects = _make_subject_manifest(tmp_path, count=2)
    plan = plan_from_files(subjects, DEFAULT_MATRIX)
    schedule = plan["schedule"]
    assert [entry["order"] for entry in schedule] == list(range(1, len(schedule) + 1))
    # The first round visits every deterministic subject/strategy queue once.
    first_round = schedule[: plan["subject_count"] * plan["strategy_count"]]
    assert len({(entry["subject_id"], entry["strategy_name"]) for entry in first_round}) == 12

    entry = schedule[0]
    recomputed = make_test_id(
        subject_id=entry["subject_id"],
        strategy_id=entry["strategy_id"],
        policy=entry["policy"],
        seed=entry["seed"],
        mode=entry["mode"],
        scope=entry["scope"],
        parameters=entry["parameters"],
        replicate=entry["replicate"],
    )
    assert entry["test_id"] == recomputed
    changed_order = dict(entry, order=999999)
    assert changed_order["test_id"] == recomputed


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda matrix: matrix.update({"unknown": True}), "unknown fields"),
        (
            lambda matrix: matrix["strategies"].append(copy.deepcopy(matrix["strategies"][0])),
            "unique",
        ),
        (
            lambda matrix: matrix["strategies"][1]["cases"][0].update({"scope": "out_of_contract"}),
            "must be 'derived'",
        ),
        (lambda matrix: matrix.update({"candidate_run_budget": 0}), "positive integer"),
        (lambda matrix: matrix["strategies"][1].update({"cases": []}), "non-empty"),
        (
            lambda matrix: matrix["strategies"][0].update({"budget_matched": True}),
            "non-matched five-run anchor",
        ),
        (
            lambda matrix: matrix["strategies"][-1]["cases"][-2]["parameters"].update(
                {"repeat_count": 1}
            ),
            "at least two",
        ),
    ],
)
def test_invalid_strategy_matrices_are_rejected(tmp_path: Path, mutate, message):
    subjects = _make_subject_manifest(tmp_path, count=1)
    matrix = json.loads(DEFAULT_MATRIX.read_text(encoding="utf-8"))
    mutate(matrix)
    with pytest.raises(ProtocolError, match=message):
        _plan_with_matrix(subjects, matrix)


def test_plan_write_is_immutable_and_cli_refuses_overwrite(tmp_path: Path):
    subjects = _make_subject_manifest(tmp_path, count=1)
    plan = plan_from_files(subjects, DEFAULT_MATRIX)
    output = tmp_path / "plan.json"
    write_plan_once(output, plan)
    with pytest.raises(FileExistsError):
        write_plan_once(output, plan)

    cli_output = tmp_path / "cli_plan.json"
    assert main(
        [
            "--subjects",
            str(subjects),
            "--strategy-matrix",
            str(DEFAULT_MATRIX),
            "--output",
            str(cli_output),
        ]
    ) == 0
    assert main(
        [
            "--subjects",
            str(subjects),
            "--strategy-matrix",
            str(DEFAULT_MATRIX),
            "--output",
            str(cli_output),
        ]
    ) == 2


def test_frozen_plan_revalidation_rejects_semantically_invalid_rehashed_schedule(
    tmp_path: Path,
):
    subjects_path = _make_subject_manifest(tmp_path, count=1)
    subjects = json.loads(subjects_path.read_text(encoding="utf-8"))
    plan = plan_from_files(subjects_path, DEFAULT_MATRIX)
    plan["schedule"][0]["candidate_run_cost"] = 2
    plan["schedule_sha256"] = stable_json_sha256(plan["schedule"])
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    plan["plan_sha256"] = stable_json_sha256(payload)

    with pytest.raises(ProtocolError, match="candidate_run_cost"):
        validate_frozen_plan(plan, subjects)
