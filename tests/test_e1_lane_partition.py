"""Two-lane parallel equiv partitioning (torch-free, CPU-only).

Pins the 2026-07-24 dual-lane contract: kernel sets are mutually exclusive
and complete, every heavy (large-VRAM / slow) kernel lands in lane 0, lane
output paths are isolated, and lane drivers fold the serial-era global
checkpoint into their skip set while checkpointing only their own probes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_e1_probe_study import (
    equiv_lane_paths,
    load_equiv_skip_set,
    plan_equiv_lanes,
)


# ---------------------------------------------------------------------------
# plan_equiv_lanes
# ---------------------------------------------------------------------------

def test_lanes_are_disjoint_and_cover_all_kernels():
    loads = {f"K{i}": float(100 + i) for i in range(11)}
    lanes, _ = plan_equiv_lanes(loads, heavy_kernels=["K3", "K7"])
    flat = [k for lane in lanes for k in lane]
    assert len(flat) == len(set(flat)) == len(loads)
    assert set(flat) == set(loads)
    assert not (set(lanes[0]) & set(lanes[1]))


def test_all_heavy_kernels_pinned_to_lane0():
    loads = {"A": 800.0, "B": 750.0, "C": 300.0, "D": 200.0, "E": 100.0}
    heavy = ["A", "B"]
    lanes, _ = plan_equiv_lanes(loads, heavy)
    assert set(heavy) <= set(lanes[0])
    assert not (set(heavy) & set(lanes[1]))


def test_non_heavy_kernels_balance_lane_totals():
    # Heavy pins 1000s onto lane 0; the greedy pass should push most of the
    # remaining light load onto lane 1 instead of stacking lane 0 further.
    loads = {"H": 1000.0, "a": 400.0, "b": 300.0, "c": 200.0, "d": 100.0}
    lanes, totals = plan_equiv_lanes(loads, heavy_kernels=["H"])
    assert set(lanes[1]) == {"a", "b", "c", "d"}
    assert totals == [1000.0, 1000.0]


def test_plan_is_deterministic():
    loads = {f"K{i}": 100.0 for i in range(9)}  # all ties
    first = plan_equiv_lanes(dict(loads), ["K0"])
    second = plan_equiv_lanes(dict(reversed(list(loads.items()))), ["K0"])
    assert first == second


def test_heavy_kernel_missing_from_loads_is_ignored():
    loads = {"A": 100.0, "B": 200.0}
    lanes, _ = plan_equiv_lanes(loads, heavy_kernels=["Z", "A"])
    assert "A" in lanes[0]
    assert set(lanes[0]) | set(lanes[1]) == {"A", "B"}


# ---------------------------------------------------------------------------
# equiv_lane_paths
# ---------------------------------------------------------------------------

def test_serial_paths_unchanged_and_lane_paths_isolated(tmp_path: Path):
    serial = equiv_lane_paths(tmp_path, None)
    assert serial["obs"].name == "equiv_observations.jsonl"
    assert serial["done"].name == "equiv_completed.json"

    lane0 = equiv_lane_paths(tmp_path, 0)
    lane1 = equiv_lane_paths(tmp_path, 1)
    assert lane0["obs"].name == "equiv_observations_lane0.jsonl"
    assert lane1["done"].name == "equiv_completed_lane1.json"
    all_paths = [p for group in (serial, lane0, lane1) for p in group.values()]
    assert len(all_paths) == len(set(all_paths)), "lane files must never collide"


# ---------------------------------------------------------------------------
# load_equiv_skip_set
# ---------------------------------------------------------------------------

@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    (tmp_path / "equiv_completed.json").write_text(
        json.dumps(["g1", "g2", "g3"]))
    (tmp_path / "equiv_completed_lane0.json").write_text(json.dumps(["l0a"]))
    return tmp_path


def test_lane_skip_set_folds_in_global_checkpoint(out_dir: Path):
    own, skip = load_equiv_skip_set(out_dir, 0)
    assert own == {"l0a"}
    assert skip == {"g1", "g2", "g3", "l0a"}


def test_lane_without_own_checkpoint_skips_global_and_other_lanes(out_dir: Path):
    own, skip = load_equiv_skip_set(out_dir, 1)
    assert own == set()
    # Other lanes' checkpoints are folded in read-only; kernel-set mutual
    # exclusivity makes this a no-op for concurrent lanes and a requirement
    # for re-split lanes inheriting a retired lane's kernels.
    assert skip == {"g1", "g2", "g3", "l0a"}


def test_resplit_lane_inherits_all_prior_lane_checkpoints(out_dir: Path):
    (out_dir / "equiv_completed_lane1.json").write_text(
        json.dumps(["l1a", "l1b"]))
    (out_dir / "equiv_completed_lane2.json").write_text(json.dumps(["l2a"]))
    own, skip = load_equiv_skip_set(out_dir, 2)
    assert own == {"l2a"}
    assert skip == {"g1", "g2", "g3", "l0a", "l1a", "l1b", "l2a"}


# ---------------------------------------------------------------------------
# Tagged (requeue) lanes
# ---------------------------------------------------------------------------

def test_tagged_lane_paths_match_monitoring_globs(tmp_path: Path):
    tagged = equiv_lane_paths(tmp_path, 4, tag="requeue")
    assert tagged["obs"].name == "equiv_observations_lane4_requeue.jsonl"
    assert tagged["done"].name == "equiv_completed_lane4_requeue.json"
    assert tagged["obs"].match("equiv_observations_lane*.jsonl")
    assert tagged["done"].match("equiv_completed_lane*.json")
    assert tagged["obs"] != equiv_lane_paths(tmp_path, 4)["obs"]


def test_tagged_lane_skip_set_folds_all_lanes_and_keeps_own(out_dir: Path):
    (out_dir / "equiv_completed_lane4_requeue.json").write_text(
        json.dumps(["rq1"]))
    own, skip = load_equiv_skip_set(out_dir, 4, tag="requeue")
    assert own == {"rq1"}
    assert skip == {"g1", "g2", "g3", "l0a", "rq1"}


def test_serial_mode_semantics_unchanged(out_dir: Path):
    own, skip = load_equiv_skip_set(out_dir, None)
    assert own == skip == {"g1", "g2", "g3"}


def test_missing_global_checkpoint_is_tolerated(tmp_path: Path):
    own, skip = load_equiv_skip_set(tmp_path, 0)
    assert own == skip == set()
