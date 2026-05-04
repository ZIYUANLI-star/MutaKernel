"""Smoke test: run ALL A~D operators on 20 random L1 kernels.

Run:  python -m scripts.smoke_test_all_ops_l1
"""
import os, sys, traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mutengine.operators.arithmetic import ArithReplace, RelOpReplace, ConstPerturb
from src.mutengine.operators.gpu_parallel import (
    IndexReplace, SyncRemove, MaskBoundary, LaunchConfigMutate,
)
from src.mutengine.operators.ml_semantic import (
    StabRemove, AccDowngrade, EpsilonModify, ScaleModify,
    CastRemove, ReductionReorder, InitModify,
)
from src.mutengine.operators.llm_pattern import BroadcastUnsafe, LayoutAssume

KERNEL_DIR = Path(__file__).resolve().parent.parent / "test_data" / "l1_smoke20"
SEP = "=" * 90

ALL_OPS = [
    ("A1 ArithReplace",     "A", ArithReplace()),
    ("A2 RelOpReplace",     "A", RelOpReplace()),
    ("A3 ConstPerturb",     "A", ConstPerturb()),
    ("B1 IndexReplace",     "B", IndexReplace()),
    ("B2 SyncRemove",       "B", SyncRemove()),
    ("B3 MaskBoundary",     "B", MaskBoundary()),
    ("B4 LaunchConfig",     "B", LaunchConfigMutate()),
    ("C1 StabRemove",       "C", StabRemove()),
    ("C2 AccDowngrade",     "C", AccDowngrade()),
    ("C3 EpsilonModify",    "C", EpsilonModify()),
    ("C4 ScaleModify",      "C", ScaleModify()),
    ("C5 CastRemove",       "C", CastRemove()),
    ("C6 RedReorder",       "C", ReductionReorder()),
    ("C7 InitModify",       "C", InitModify()),
    ("D1 BroadcastUnsafe",  "D", BroadcastUnsafe()),
    ("D2 LayoutAssume",     "D", LayoutAssume()),
]


def test_file(filepath: Path):
    source = filepath.read_text(encoding="utf-8", errors="replace")
    loc = len(source.splitlines())
    row = {"name": filepath.stem, "loc": loc}
    
    for label, cat, op in ALL_OPS:
        try:
            sites = op.find_sites(source)
        except Exception as e:
            row[label] = {"count": -1, "error": str(e)[:60]}
            continue
        
        apply_ok = 0
        apply_fail = 0
        if sites:
            for s in sites[:3]:
                try:
                    mutated = op.apply(source, s)
                    if mutated != source:
                        apply_ok += 1
                    else:
                        apply_fail += 1
                except Exception:
                    apply_fail += 1
        
        row[label] = {"count": len(sites), "ok": apply_ok, "fail": apply_fail}
    
    return row


def main():
    files = sorted(KERNEL_DIR.glob("*.py"))
    if not files:
        print(f"ERROR: No .py files found in {KERNEL_DIR}")
        return
    
    print(f"Testing {len(files)} L1 kernels with {len(ALL_OPS)} operators\n")
    
    results = []
    for f in files:
        try:
            row = test_file(f)
            results.append(row)
        except Exception as e:
            print(f"  ERROR on {f.name}: {e}")
            traceback.print_exc()
    
    # --- Per-category summary table ---
    cats = {"A": [], "B": [], "C": [], "D": []}
    for label, cat, _ in ALL_OPS:
        cats[cat].append(label)
    
    for cat_name, cat_labels in cats.items():
        print(SEP)
        print(f"  Category {cat_name}")
        print(SEP)
        
        header = f"  {'File':<45s}" + "".join(f"{l.split()[0]+l.split()[1][:5]:>9s}" for l in cat_labels) + "  CatTotal"
        print(header)
        print("  " + "-" * (len(header) - 2))
        
        cat_totals = {l: 0 for l in cat_labels}
        for r in results:
            counts = []
            for l in cat_labels:
                info = r.get(l, {"count": 0})
                c = info.get("count", 0)
                if c < 0:
                    counts.append("ERR")
                else:
                    counts.append(str(c))
                    cat_totals[l] += c
            total = sum(int(c) for c in counts if c != "ERR")
            print(f"  {r['name']:<45s}" + "".join(f"{c:>9s}" for c in counts) + f"  {total:>8d}")
        
        grand = sum(cat_totals.values())
        print("  " + "-" * (len(header) - 2))
        print(f"  {'TOTAL':<45s}" + "".join(f"{cat_totals[l]:>9d}" for l in cat_labels) + f"  {grand:>8d}")
        print()
    
    # --- Grand summary ---
    print(SEP)
    print("  GRAND SUMMARY — A~D on 20 L1 Kernels")
    print(SEP)
    
    cat_grand = {}
    for cat_name in ("A", "B", "C", "D"):
        total = 0
        for label, c, _ in ALL_OPS:
            if c == cat_name:
                for r in results:
                    info = r.get(label, {"count": 0})
                    cnt = info.get("count", 0)
                    if cnt > 0:
                        total += cnt
        cat_grand[cat_name] = total
    
    files_with_sites = {}
    for cat_name in ("A", "B", "C", "D"):
        count = 0
        for r in results:
            has = False
            for label, c, _ in ALL_OPS:
                if c == cat_name:
                    info = r.get(label, {"count": 0})
                    if info.get("count", 0) > 0:
                        has = True
                        break
            if has:
                count += 1
        files_with_sites[cat_name] = count
    
    total_all = sum(cat_grand.values())
    print(f"  {'Category':<12s} {'Sites':>8s} {'Files(>0)':>10s} {'Coverage':>10s}")
    print(f"  {'-'*42}")
    for cat_name in ("A", "B", "C", "D"):
        cov = f"{files_with_sites[cat_name]}/{len(results)}"
        print(f"  {cat_name:<12s} {cat_grand[cat_name]:>8d} {files_with_sites[cat_name]:>10d} {cov:>10s}")
    print(f"  {'-'*42}")
    print(f"  {'TOTAL':<12s} {total_all:>8d}")
    
    # --- Apply verification ---
    print(f"\n  Apply Verification (up to 3 sites per file):")
    for label, _, op in ALL_OPS:
        ok_total = sum(r.get(label, {}).get("ok", 0) for r in results)
        fail_total = sum(r.get(label, {}).get("fail", 0) for r in results)
        tested = ok_total + fail_total
        status = "PASS" if fail_total == 0 and tested > 0 else ("N/A" if tested == 0 else f"FAIL({fail_total})")
        print(f"    {label:<22s}: {ok_total:>3d}/{tested:>3d} apply OK  [{status}]")


if __name__ == "__main__":
    main()
