from src.stress.differential_tester import StressSummary, StressTestResult


def _result(mutant_id: str = "m") -> StressTestResult:
    return StressTestResult(
        mutant_id=mutant_id,
        operator_name="op",
        operator_category="A",
        kernel_name="kernel",
    )


def test_main_and_config_tracks_are_reported_separately():
    result = _result()
    result.record_dimension("main", "value_stress", {"killed": False})
    result.record_dimension("config", "config_stress", {"killed": True})

    assert not result.main_track_killed
    assert result.config_track_killed
    assert result.deterministic_killed

    summary = result.get_kill_summary()
    assert summary["main_track_killed"] is False
    assert summary["config_track_killed"] is True


def test_aggregate_does_not_conflate_contract_tracks():
    main = _result("main")
    main.record_dimension("main", "value_stress", {"killed": True})

    config = _result("config")
    config.record_dimension("config", "config_stress", {"killed": True})

    both = _result("both")
    both.record_dimension("main", "dtype", {"killed": True})
    both.record_dimension("config", "batch", {"killed": True})

    aggregate = StressSummary()
    for result in (main, config, both):
        aggregate.add_result(result)

    data = aggregate.to_dict()
    assert data["deterministic_kill_count"] == 3
    assert data["main_track_kill_count"] == 2
    assert data["config_track_kill_count"] == 2
