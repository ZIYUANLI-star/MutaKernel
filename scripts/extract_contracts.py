#!/usr/bin/env python3
"""Automatic correctness-contract extraction (schema v1) for KernelBench tasks.

Blueprint §5.1.1 contract governance: contracts are extracted automatically
from the benchmark's own artifacts — the reference implementation and its
``get_inputs``/``get_init_inputs`` specification — with only the boundary
clauses (value domain, variable dimensions, determinism) requiring human
judgment.  This extractor:

  * loads each reference module CPU-only (set CUDA_VISIBLE_DEVICES= before
    running; no candidate code is ever executed),
  * calls ``get_inputs()`` once under a fixed seed and records the observed
    tensor shapes/dtypes/layouts,
  * emits a *valid* schema-v1 contract per task with conservative defaults
    for the human-judgment clauses (value_domain=unrestricted_finite,
    fixed shapes, deterministic=True, eval mode) and flags them in
    ``notes`` for the dual review,
  * applies the blueprint's dtype-aware oracle tolerances
    (float32: atol=rtol=1e-4; float16/bfloat16: 1e-2),
  * binds every generic value stress policy to the floating tensor args and
    declares the dtype adapter over them,
  * validates each contract with src.experiments.contract.validate_contract
    and freezes the batch with content hashes (canonical JSON, sha256).

Output: JSONL (one ``{"task_id", "contract", "contract_sha256",
"extraction"}`` per line) + a frozen digest file.  Human review then edits
the flagged clauses and re-freezes — amendments are version-controlled by
contract_id suffix.

Usage (remote, CPU-only):
  CUDA_VISIBLE_DEVICES= python scripts/extract_contracts.py \
      --kernelbench-root KernelBench --levels 1 2 \
      --out /root/mk_v2_runs/e2/contracts_kernelbench_v1.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Blueprint §5.1.1 (following the vendored KernelBench implementation).
DTYPE_TOLERANCES = {
    "float32": {"atol": 1e-4, "rtol": 1e-4},
    "float16": {"atol": 1e-2, "rtol": 1e-2},
    "bfloat16": {"atol": 1e-2, "rtol": 1e-2},
}
EXTRACTION_SEED = 42
LOW_PRECISION_EXTRAS = ["float16", "bfloat16"]

HUMAN_REVIEW_NOTE = (
    "AUTO-EXTRACTED v1 defaults pending dual human review: value_domain "
    "(default unrestricted_finite), variable dimensions (shapes frozen to "
    "the observed get_inputs() sizes), determinism (default True). "
    "Amendments must bump contract_id and record a reason."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_module(path: Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_task_contract(problem_file: Path, task_id: str) -> dict:
    """Extract one schema-v1 contract from a KernelBench-style reference."""
    import torch

    from src.experiments.contract import validate_contract
    from src.stress.policy_bank import get_all_policy_names

    module = _load_module(problem_file, f"contract_ref_{task_id}")
    torch.manual_seed(EXTRACTION_SEED)
    inputs = module.get_inputs()
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    tensor_inputs = []
    float_indices = []
    for index, value in enumerate(inputs):
        if not isinstance(value, torch.Tensor):
            continue  # non-tensor args are passed through untouched
        dtype_name = str(value.dtype).removeprefix("torch.")
        dtypes = [dtype_name]
        if value.is_floating_point() and dtype_name == "float32":
            dtypes.extend(LOW_PRECISION_EXTRAS)
            float_indices.append(index)
        elif value.is_floating_point():
            float_indices.append(index)
        tensor_inputs.append({
            "arg_index": index,
            "dtypes": dtypes,
            "shape": [int(dim) for dim in value.shape],
            "value_domain": {"kind": "unrestricted_finite"},
            "layouts": ["contiguous", "noncontiguous"],
            "requires_grad": bool(value.requires_grad),
            "aliases": [],
        })

    if not tensor_inputs:
        raise ValueError(f"{task_id}: get_inputs() produced no tensor arguments")

    policy_bindings = {
        policy: list(float_indices) or [tensor_inputs[0]["arg_index"]]
        for policy in get_all_policy_names()
    } if float_indices else {}

    input_adapters = {}
    if float_indices:
        input_adapters["dtype"] = {"arg_indices": list(float_indices)}
        input_adapters["layout"] = {
            "arg_indices": list(float_indices),
            "allowed_values": ["noncontiguous"],
        }

    modes = ["eval"]
    contract = {
        "schema_version": "1.0",
        "contract_id": f"{task_id}-auto-v1",
        "tensor_inputs": tensor_inputs,
        "execution": {
            "modes": modes,
            "backward": False,
            "deterministic": True,
            "repeat_count_max": 3,
            "streams": ["default"],
            "compare_input_side_effects": True,
            "compare_module_state": True,
            "backward_vjp_count": 1,
        },
        "oracle": {
            "atol": DTYPE_TOLERANCES["float32"]["atol"],
            "rtol": DTYPE_TOLERANCES["float32"]["rtol"],
            "equal_nan": True,
            "require_dtype": True,
            "require_device": True,
            "require_layout": True,
            "require_stride": False,
            "require_aliasing": False,
            "dtype_tolerances": DTYPE_TOLERANCES,
        },
        "policy_bindings": policy_bindings,
        "input_adapters": input_adapters,
        "candidate_classes": ["ModelNew", "Model"],
        "notes": HUMAN_REVIEW_NOTE,
    }
    normalized = validate_contract(contract)

    extraction = {
        "extracted_at": _now(),
        "seed": EXTRACTION_SEED,
        "source": str(problem_file),
        "source_sha256": hashlib.sha256(problem_file.read_bytes()).hexdigest(),
        "observed_args": len(inputs),
        "observed_tensor_args": len(tensor_inputs),
        "human_review_pending": ["value_domain", "variable_dimensions", "determinism"],
    }
    return {"contract": normalized, "extraction": extraction}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kernelbench-root", required=True, type=Path)
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for level in args.levels:
        level_dir = args.kernelbench_root / "KernelBench" / f"level{level}"
        problem_files = sorted(
            level_dir.glob("*.py"),
            key=lambda p: int(p.name.split("_", 1)[0]),
        )
        if args.limit:
            problem_files = problem_files[: args.limit]
        for problem_file in problem_files:
            task_id = f"L{level}_P{problem_file.name.split('_', 1)[0]}"
            try:
                result = extract_task_contract(problem_file, task_id)
            except Exception as exc:
                failures.append({"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"[:300]})
                print(f"[extract] {task_id}: FAILED {type(exc).__name__}: {str(exc)[:120]}",
                      flush=True)
                continue
            contract_json = json.dumps(result["contract"], sort_keys=True,
                                       separators=(",", ":"))
            rows.append({
                "task_id": task_id,
                "contract": result["contract"],
                "contract_sha256": hashlib.sha256(contract_json.encode()).hexdigest(),
                "extraction": result["extraction"],
            })
            print(f"[extract] {task_id}: ok "
                  f"({result['extraction']['observed_tensor_args']} tensor args)",
                  flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    batch_digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    freeze = {
        "created_at": _now(),
        "contracts": len(rows),
        "failures": failures,
        "batch_sha256": batch_digest,
        "note": "freeze this digest before observing any V2 validator outcome",
    }
    freeze_path = args.out.with_suffix(".freeze.json")
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print(json.dumps({"contracts": len(rows), "failures": len(failures),
                      "batch_sha256": batch_digest}, indent=2))


if __name__ == "__main__":
    main()
