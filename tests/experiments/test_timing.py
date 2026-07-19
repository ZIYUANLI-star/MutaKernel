from src.experiments.timing import TimingBreakdown


class FakeClock:
    def __init__(self, values):
        self._values = iter(values)

    def __call__(self):
        return next(self._values)


def test_phase_timing_accumulates_repeated_phases():
    timing = TimingBreakdown(clock=FakeClock([100, 1_100, 2_000, 4_500]))
    with timing.phase("candidate"):
        pass
    with timing.phase("candidate"):
        pass
    assert timing.phases_ns == {"candidate": 3_500}
    assert timing.total_ns == 3_500
    assert timing.total_ms == 0.0035


def test_timing_serialization_is_sorted_and_complete():
    timing = TimingBreakdown()
    timing.add_ms("oracle", 2.5)
    timing.add_ns("compile", 1_000_000)
    data = timing.to_dict()
    assert list(data["phases_ns"]) == ["compile", "oracle"]
    assert data["phases_ms"] == {"compile": 1.0, "oracle": 2.5}
    assert data["total_ms"] == 3.5
