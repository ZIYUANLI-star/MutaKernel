#!/usr/bin/env python3
"""Scan KernelBench run directories to find the best correct kernel per problem.

For each problem, iterates through all 10 turns, selects the turn with:
  1. correctness == True
  2. Highest speedup

Outputs statistics and a JSON mapping: problem_id -> best_kernel_path
"""
import json
import os
import sys
from pathlib import Path

KB_ROOT = Path("/home/kbuser/projects/KernelBench-0")
LEVELS = {
    "L1": KB_ROOT / "runs" / "iter_full_l1_caesar_paper_v2",
    "L2": KB_ROOT / "runs" / "iter_full_l2_caesar_paper_v2",
}

OUTPUT_FILE = Path(__file__).resolve().parent.parent / "best_kernels.json"


def scan_level(level, run_dir):
    iter_dir = run_dir / "iterations"
    if not iter_dir.exists():
        print(f"  WARN: {iter_dir} not found")
        return []

    results = []
    for prob_dir in sorted(iter_dir.iterdir()):
        summary_file = prob_dir / "problem_summary.json"
        if not summary_file.exists():
            continue

        with open(summary_file) as f:
            data = json.load(f)

        problem_id = data.get("problem_id", prob_dir.name)
        problem_name = data.get("problem_name", "")

        best_turn = None
        best_speedup = -1.0
        best_kernel_path = None

        for turn_info in data.get("turns", []):
            ev = turn_info.get("eval", {})
            if not ev.get("correctness", False):
                continue
            spd = turn_info.get("speedup")
            if spd is None:
                spd = ev.get("metadata", {}).get("observed_speedup", 0)
            if spd is None:
                spd = 0
            if spd > best_speedup:
                best_speedup = spd
                best_turn = turn_info["turn"]
                best_kernel_path = turn_info.get("kernel_path", "")

        results.append({
            "level": level,
            "problem_id": str(problem_id),
            "problem_name": problem_name,
            "best_turn": best_turn,
            "best_speedup": best_speedup if best_turn is not None else None,
            "best_kernel_path": best_kernel_path,
            "num_correct_turns": sum(
                1 for t in data.get("turns", [])
                if t.get("eval", {}).get("correctness", False)
            ),
            "total_turns": len(data.get("turns", [])),
        })

    return results


def main():
    all_results = []
    for level, run_dir in LEVELS.items():
        print(f"\n=== Scanning {level}: {run_dir} ===")
        results = scan_level(level, run_dir)
        all_results.extend(results)

        total = len(results)
        has_correct = sum(1 for r in results if r["best_turn"] is not None)
        no_correct = total - has_correct

        print(f"  Total problems: {total}")
        print(f"  With correct kernel: {has_correct}")
        print(f"  No correct kernel: {no_correct}")

        if has_correct > 0:
            speedups = [r["best_speedup"] for r in results if r["best_speedup"] is not None]
            fast = sum(1 for s in speedups if s >= 1.0)
            print(f"  Speedup >= 1.0 (fastp): {fast}/{has_correct}")
            print(f"  Avg speedup (correct only): {sum(speedups)/len(speedups):.3f}")
            print(f"  Max speedup: {max(speedups):.3f}")

        # Show problems without any correct kernel
        if no_correct > 0:
            ids = [r["problem_id"] for r in results if r["best_turn"] is None]
            print(f"  Problems without correct kernel: {ids}")

    # Build best_kernels mapping
    best_map = {}
    for r in all_results:
        if r["best_turn"] is not None:
            key = f"{r['level']}_P{r['problem_id']}"
            best_map[key] = {
                "level": r["level"],
                "problem_id": r["problem_id"],
                "turn": r["best_turn"],
                "speedup": r["best_speedup"],
                "kernel_path": r["best_kernel_path"],
                "num_correct_turns": r["num_correct_turns"],
            }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(best_map, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"  Total problems: {len(all_results)}")
    print(f"  Correct kernels selected: {len(best_map)}")
    print(f"  Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
