from src.experiments.strategy import StrategySpec, TestCaseSpec, make_test_id


def test_strategy_id_is_stable_across_mapping_order():
    left = StrategySpec("iid-random", version="1", parameters={"seeds": 5, "dtype": "fp32"})
    right = StrategySpec("iid-random", version="1", parameters={"dtype": "fp32", "seeds": 5})
    assert left.strategy_id == right.strategy_id


def test_test_id_is_stable_and_independent_of_schedule_order():
    strategy = StrategySpec("mutakernel-fixed", parameters={"suite": "v1"})
    first_plan = [
        TestCaseSpec("s1", strategy, "near_zero", 42),
        TestCaseSpec("s1", strategy, "sparse", 43),
    ]
    reordered_plan = list(reversed(first_plan))
    assert {case.test_id for case in first_plan} == {case.test_id for case in reordered_plan}
    assert first_plan[0].test_id == TestCaseSpec("s1", strategy, "near_zero", 42).test_id


def test_identity_changes_only_for_semantic_fields():
    strategy = StrategySpec("iid-random")
    base = make_test_id(
        subject_id="s1",
        strategy_id=strategy.strategy_id,
        policy="identity",
        seed=42,
    )
    changed_seed = make_test_id(
        subject_id="s1",
        strategy_id=strategy.strategy_id,
        policy="identity",
        seed=43,
    )
    replicate = make_test_id(
        subject_id="s1",
        strategy_id=strategy.strategy_id,
        policy="identity",
        seed=42,
        replicate=1,
    )
    assert len(base) == 64
    assert base != changed_seed
    assert base != replicate


def test_repeated_case_exposes_its_candidate_execution_cost():
    strategy = StrategySpec("repeat")
    case = TestCaseSpec(
        "s1",
        strategy,
        "identity",
        42,
        mode="repeated",
        parameters={"repeat_count": 3},
    )

    assert case.candidate_run_cost == 3
    assert case.to_dict()["candidate_run_cost"] == 3
