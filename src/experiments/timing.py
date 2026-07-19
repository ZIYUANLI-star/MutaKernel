"""Monotonic, phase-level wall-clock timing primitives."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator


Clock = Callable[[], int]


@dataclass
class TimingBreakdown:
    """Accumulate nanosecond durations by phase.

    The clock is injectable so timing behavior can be unit-tested without
    sleeping.  GPU event durations belong in the observation schema as
    separate metrics; this class measures monotonic host wall time only.
    """

    clock: Clock = field(default=time.perf_counter_ns, repr=False, compare=False)
    phases_ns: Dict[str, int] = field(default_factory=dict)

    def add_ns(self, phase: str, duration_ns: int) -> None:
        if not phase:
            raise ValueError("phase must not be empty")
        if isinstance(duration_ns, bool) or not isinstance(duration_ns, int):
            raise TypeError("duration_ns must be an integer")
        if duration_ns < 0:
            raise ValueError("duration_ns must be non-negative")
        self.phases_ns[phase] = self.phases_ns.get(phase, 0) + duration_ns

    def add_ms(self, phase: str, duration_ms: float) -> None:
        if not math.isfinite(duration_ms) or duration_ms < 0:
            raise ValueError("duration_ms must be finite and non-negative")
        self.add_ns(phase, int(round(duration_ms * 1_000_000)))

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started = self.clock()
        try:
            yield
        finally:
            finished = self.clock()
            self.add_ns(name, finished - started)

    @property
    def total_ns(self) -> int:
        return sum(self.phases_ns.values())

    @property
    def total_ms(self) -> float:
        return self.total_ns / 1_000_000.0

    def to_dict(self) -> Dict[str, object]:
        ordered = dict(sorted(self.phases_ns.items()))
        return {
            "clock": "perf_counter_ns",
            "phases_ns": ordered,
            "phases_ms": {name: value / 1_000_000.0 for name, value in ordered.items()},
            "total_ns": self.total_ns,
            "total_ms": self.total_ms,
        }
