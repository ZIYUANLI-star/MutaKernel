#!/usr/bin/env python3
"""Smoke test: validate operator rule changes (A3/B3/B4/D2) on 20 L1 kernels.

Runs locally (no GPU needed) — only checks find_sites() and apply() logic.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.mutengine.operators.arithmetic import ArithReplace, RelOpReplace, ConstPerturb
from src.mutengine.operators.gpu_parallel import (
    IndexReplace, SyncRemove, MaskBoundary, LaunchConfigMutate,
)
from src.mutengine.operators.ml_semantic import (
    StabRemove, AccDowngrade, EpsilonModify, ScaleModify,
    CastRemove, ReductionReorder, InitModify,
)
from src.mutengine.operators.llm_pattern import BroadcastUnsafe, LayoutAssume

KERNEL_DIR = PROJECT_ROOT / "test_data" / "l1_smoke20"

ALL_OPS = [
    ("A1", ArithReplace()),
    ("A2", RelOpReplace()),
    ("A3", ConstPerturb()),
    ("B1", IndexReplace()),
    ("B2", SyncRemove()),
    ("B3", MaskBoundary()),
    ("B4", LaunchConfigMutate()),
    ("C1", StabRemove()),
    ("C2", AccDowngrade()),
    ("C3", EpsilonModify()),
    ("C4", ScaleModify()),
    ("C5", CastRemove()),
    ("C6", ReductionReorder()),
    ("C7", InitModify()),
    ("D1", BroadcastUnsafe()),
    ("D2", LayoutAssume()),
]

import re

GPU_CONFIG_RE = re.compile(
    r'(?:'
    r'\b(?:BLOCK_?SIZE|block_?size|TILE_?SIZE|tile_?size|'
    r'WARP_?SIZE|warp_?size|num_warps|NUM_WARPS|'
    r'GRID_?SIZE|grid_?size|num_blocks|NUM_BLOCKS|'
    r'num_threads|NUM_THREADS|threads_per_block|blocks_per_grid|'
    r'threadIdx|blockIdx)\b'
    r'|'
    r'\b(?:blockDim|gridDim|dim3)\w*'
    r')'
)


def main():
    kernels = sorted(KERNEL_DIR.glob("*.py"))
    if not kernels:
        print(f"ERROR: no kernel files found in {KERNEL_DIR}")
        sys.exit(1)

    print(f"Testing {len(kernels)} kernels from {KERNEL_DIR.name}\n")

    total_sites = {label: 0 for label, _ in ALL_OPS}
    total_apply_ok = {label: 0 for label, _ in ALL_OPS}
    errors = []

    # Rule-specific counters
    a3_config_leaked = 0
    b3_widen_leaked = 0
    b4_plus1_leaked = 0
    d2_plain_leaked = 0

    for kf in kernels:
        source = kf.read_text(encoding="utf-8", errors="replace")
        lines = source.splitlines()
        kname = kf.stem.split("_kernel")[0].split("problem_")[1]

        for label, op in ALL_OPS:
            try:
                sites = op.find_sites(source)
            except Exception as e:
                errors.append(f"{kname}/{label} find_sites: {e}")
                continue

            total_sites[label] += len(sites)

            # --- Rule validation ---
            if label == "A3":
                for site in sites:
                    line_text = lines[site.line_start - 1] if site.line_start <= len(lines) else ""
                    m_cfg = GPU_CONFIG_RE.search(line_text)
                    if m_cfg:
                        a3_config_leaked += 1
                        errors.append(
                            f"{kname}/A3 LEAKED gpu config: L{site.line_start} "
                            f"'{line_text.strip()[:60]}' (match: {m_cfg.group()})"
                        )

            if label == "B3":
                for site in sites:
                    if site.node_type == "<=":
                        b3_widen_leaked += 1
                        errors.append(
                            f"{kname}/B3 LEAKED widen direction: L{site.line_start} "
                            f"node_type='<=' should be removed"
                        )

            if label == "B4":
                for site in sites:
                    if site.node_type == "+1":
                        b4_plus1_leaked += 1
                        errors.append(
                            f"{kname}/B4 LEAKED +1 direction: L{site.line_start} "
                            f"node_type='+1' should be removed"
                        )

            if label == "D2":
                for site in sites:
                    if site.node_type == "d2:remove_contiguous":
                        line_text = lines[site.line_start - 1] if site.line_start <= len(lines) else ""
                        prefix = line_text[:site.col_start]
                        has_layout_op = any(
                            kw in prefix
                            for kw in [".transpose", ".permute", ".T", ".t(", ".mT", "[:,", "[:"]
                        )
                        if not has_layout_op:
                            d2_plain_leaked += 1
                            errors.append(
                                f"{kname}/D2 LEAKED plain .contiguous(): L{site.line_start} "
                                f"'{line_text.strip()[:60]}'"
                            )

            # --- Test apply() ---
            for si, site in enumerate(sites[:3]):
                try:
                    mutated = op.apply(source, site)
                    if mutated != source:
                        total_apply_ok[label] += 1
                except Exception as e:
                    errors.append(f"{kname}/{label} apply[{si}]: {e}")

    # ===== Summary =====
    print(f"{'='*70}")
    print(f"  SMOKE TEST SUMMARY — Operator Rule Changes")
    print(f"{'='*70}")
    print(f"\n  Sites found per operator:")
    print(f"  {'Op':<6} {'Sites':>7}  {'Apply OK':>9}")
    print(f"  {'-'*26}")
    for label, _ in ALL_OPS:
        print(f"  {label:<6} {total_sites[label]:>7}  {total_apply_ok[label]:>9}")

    print(f"\n  Rule checks:")
    print(f"  A3 GPU-config leaks:   {a3_config_leaked} {'PASS' if a3_config_leaked == 0 else 'FAIL'}")
    print(f"  B3 widen(<= ) leaks:   {b3_widen_leaked} {'PASS' if b3_widen_leaked == 0 else 'FAIL'}")
    print(f"  B4 +1 direction leaks: {b4_plus1_leaked} {'PASS' if b4_plus1_leaked == 0 else 'FAIL'}")
    print(f"  D2 plain .contiguous leaks: {d2_plain_leaked} {'PASS' if d2_plain_leaked == 0 else 'FAIL'}")

    all_pass = (a3_config_leaked == 0 and b3_widen_leaked == 0
                and b4_plus1_leaked == 0 and d2_plain_leaked == 0)

    if errors:
        print(f"\n  Errors/Warnings ({len(errors)}):")
        for e in errors[:20]:
            print(f"    - {e}")
        if len(errors) > 20:
            print(f"    ... and {len(errors)-20} more")

    print(f"\n  Overall: {'ALL RULES PASS' if all_pass else 'SOME RULES FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
