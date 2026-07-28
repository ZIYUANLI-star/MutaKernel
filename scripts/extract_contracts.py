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

C2-C5 extension (E3 readiness, DRAFTS — not frozen):
  * C2/C3/C5 reuse KernelBench task definitions.  ``--frames`` reads E2
    collection-frame JSONLs, collects the included rows' ``KB_L*_P*`` task
    keys and extracts one draft contract per distinct task via the same
    KernelBench extractor (only the benchmark's own reference module is
    loaded/executed; no candidate kernel ever runs).
  * C4 (TritonBench-G) is a Triton-side interface where the corpus file IS
    the candidate, so nothing may be executed at all:
    ``--tritonbench-manifest``/``--tritonbench-dir`` produce *static* drafts
    (schema ``tritonbench-draft-0.1``) by AST-parsing the test section for
    literal tensor-constructor shapes/dtypes.  These drafts are explicitly
    marked non-executable-extraction and require human completion.

Usage (remote, CPU-only):
  CUDA_VISIBLE_DEVICES= python scripts/extract_contracts.py \
      --kernelbench-root KernelBench --levels 1 2 \
      --out /root/mk_v2_runs/e2/contracts_kernelbench_v1.jsonl
  CUDA_VISIBLE_DEVICES= python scripts/extract_contracts.py \
      --kernelbench-root KernelBench \
      --frames C2_collection_frame.jsonl C5_collection_frame.jsonl \
      --out contracts_c2c5_draft.jsonl
  python scripts/extract_contracts.py \
      --tritonbench-manifest external/C4_tritonbench/data/TritonBench_G_v1.json \
      --tritonbench-dir external/C4_tritonbench/data/TritonBench_G_v1 \
      --out contracts_c4_draft.jsonl
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
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


# ---------------------------------------------------------------------------
# C2/C3/C5: task keys from E2 collection frames -> KernelBench extraction
# ---------------------------------------------------------------------------

KB_TASK_RE = re.compile(r"^KB_L(\d)_P(\d+)$")


def collect_frame_task_keys(frame_paths, only_included=True):
    """Distinct KB task keys from collection frames, with source corpora."""
    tasks = {}
    for frame_path in frame_paths:
        with open(frame_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if only_included and not row.get("included", True):
                    continue
                task_key = row.get("task_key", "")
                if not KB_TASK_RE.match(task_key):
                    continue
                tasks.setdefault(task_key, set()).add(row.get("corpus", "?"))
    return {key: sorted(corpora) for key, corpora in sorted(tasks.items())}


def kb_problem_file(kernelbench_root: Path, task_key: str):
    match = KB_TASK_RE.match(task_key)
    level, problem = int(match.group(1)), int(match.group(2))
    level_dir = kernelbench_root / "KernelBench" / f"level{level}"
    if not level_dir.is_dir():
        return None, f"level directory missing: {level_dir}"
    hits = sorted(level_dir.glob(f"{problem}_*.py"))
    if not hits:
        return None, f"no problem file {problem}_*.py under {level_dir}"
    return hits[0], None


# ---------------------------------------------------------------------------
# C4 (TritonBench-G): static AST drafts — never executes the corpus file,
# which IS the candidate kernel (data-separation red line).
# ---------------------------------------------------------------------------

TRITON_DRAFT_SCHEMA = "tritonbench-draft-0.1"
_TENSOR_CTORS = {"randn", "rand", "zeros", "ones", "empty", "full",
                 "randint", "rand_tensor", "arange", "tensor"}


def _literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def _call_name(node: ast.Call):
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def extract_triton_tensor_hints(source: str):
    """Static tensor-constructor hints (shape/dtype/device literals).

    Prefers the test section (functions named ``test_*``); falls back to the
    whole module when no test function exists.
    """
    tree = ast.parse(source)
    test_functions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test")
    ]
    scopes = test_functions or [tree]
    hints = []
    for scope in scopes:
        for node in ast.walk(scope):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name not in _TENSOR_CTORS:
                continue
            shape = []
            for arg in node.args:
                value = _literal(arg)
                if isinstance(value, int):
                    shape.append(value)
                elif isinstance(value, (tuple, list)) and all(
                        isinstance(v, int) for v in value):
                    shape.extend(int(v) for v in value)
            keywords = {}
            for kw in node.keywords:
                if kw.arg in {"dtype", "device", "mode", "low", "high"}:
                    if isinstance(kw.value, ast.Attribute):
                        keywords[kw.arg] = kw.value.attr
                    else:
                        value = _literal(kw.value)
                        if value is not None:
                            keywords[kw.arg] = value
            hints.append({"constructor": name, "shape": shape, **keywords})
    return hints, [fn.name for fn in test_functions]


def extract_c4_draft(manifest_path: Path, kernels_dir: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    failures = []
    for record in manifest:
        file_name = record["file"]
        task_id = f"TBG_{Path(file_name).stem}"
        kernel_file = kernels_dir / file_name
        if not kernel_file.is_file():
            failures.append({"task_id": task_id, "error": "kernel file missing"})
            continue
        source = kernel_file.read_text(encoding="utf-8", errors="replace")
        try:
            hints, test_functions = extract_triton_tensor_hints(source)
        except SyntaxError as exc:
            failures.append({"task_id": task_id,
                             "error": f"SyntaxError: {exc}"[:200]})
            continue
        rows.append({
            "task_id": task_id,
            "draft_schema": TRITON_DRAFT_SCHEMA,
            "language": "triton",
            "interface": "tritonbench_g",
            "entry_file": file_name,
            "source_sha256": hashlib.sha256(
                source.encode("utf-8")).hexdigest(),
            "signature": (record.get("func_inputs") or "")[:2000],
            "tensor_hints": hints,
            "test_functions": test_functions,
            "extraction": {
                "extracted_at": _now(),
                "method": "static_ast_no_execution",
                "note": (
                    "C4 corpus files ARE the candidates; execution is "
                    "forbidden before the fault-to-stress map freeze.  "
                    "Shapes/dtypes are literal constructor hints only; "
                    "human review must complete value domains, tolerances "
                    "and the Triton call adapter before freezing."),
            },
        })
    return rows, failures


def _write_batch(out: Path, rows, failures, draft: bool):
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    batch_digest = hashlib.sha256(out.read_bytes()).hexdigest()
    freeze = {
        "created_at": _now(),
        "contracts": len(rows),
        "failures": failures,
        "batch_sha256": batch_digest,
        "draft": draft,
        "note": ("DRAFT batch - do NOT freeze; pending dual human review"
                 if draft else
                 "freeze this digest before observing any V2 validator outcome"),
    }
    freeze_path = out.with_suffix(".freeze.json")
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print(json.dumps({"contracts": len(rows), "failures": len(failures),
                      "draft": draft, "batch_sha256": batch_digest}, indent=2))


def _extract_kb_tasks(task_items, kernelbench_root: Path):
    """task_items: iterable of (task_id, problem_file, extra_fields)."""
    rows = []
    failures = []
    for task_id, problem_file, extra in task_items:
        try:
            result = extract_task_contract(problem_file, task_id)
        except Exception as exc:
            failures.append({"task_id": task_id,
                             "error": f"{type(exc).__name__}: {exc}"[:300]})
            print(f"[extract] {task_id}: FAILED {type(exc).__name__}: "
                  f"{str(exc)[:120]}", flush=True)
            continue
        contract_json = json.dumps(result["contract"], sort_keys=True,
                                   separators=(",", ":"))
        rows.append({
            "task_id": task_id,
            **extra,
            "contract": result["contract"],
            "contract_sha256": hashlib.sha256(contract_json.encode()).hexdigest(),
            "extraction": result["extraction"],
        })
        print(f"[extract] {task_id}: ok "
              f"({result['extraction']['observed_tensor_args']} tensor args)",
              flush=True)
    return rows, failures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kernelbench-root", type=Path)
    ap.add_argument("--levels", nargs="+", type=int, default=None,
                    help="extract full KernelBench levels (original mode)")
    ap.add_argument("--frames", nargs="+", type=Path, default=None,
                    help="E2 collection frames; extract their KB task keys")
    ap.add_argument("--include-excluded-rows", action="store_true",
                    help="with --frames: also use rows with included=false")
    ap.add_argument("--tritonbench-manifest", type=Path, default=None)
    ap.add_argument("--tritonbench-dir", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.tritonbench_manifest:
        if not args.tritonbench_dir:
            ap.error("--tritonbench-manifest requires --tritonbench-dir")
        rows, failures = extract_c4_draft(args.tritonbench_manifest,
                                          args.tritonbench_dir)
        if args.limit:
            rows = rows[: args.limit]
        _write_batch(args.out, rows, failures, draft=True)
        return

    if not args.kernelbench_root:
        ap.error("--kernelbench-root is required for KernelBench extraction")

    if args.frames:
        tasks = collect_frame_task_keys(
            args.frames, only_included=not args.include_excluded_rows)
        task_items = []
        failures_pre = []
        for task_key, corpora in tasks.items():
            match = KB_TASK_RE.match(task_key)
            task_id = f"L{match.group(1)}_P{match.group(2)}"
            problem_file, error = kb_problem_file(args.kernelbench_root, task_key)
            if error:
                failures_pre.append({"task_id": task_id, "task_key": task_key,
                                     "source_corpora": corpora, "error": error})
                continue
            task_items.append((task_id, problem_file,
                               {"task_key": task_key, "source_corpora": corpora}))
        if args.limit:
            task_items = task_items[: args.limit]
        rows, failures = _extract_kb_tasks(task_items, args.kernelbench_root)
        _write_batch(args.out, rows, failures_pre + failures, draft=True)
        return

    levels = args.levels or [1, 2]
    task_items = []
    for level in levels:
        level_dir = args.kernelbench_root / "KernelBench" / f"level{level}"
        problem_files = sorted(
            level_dir.glob("*.py"),
            key=lambda p: int(p.name.split("_", 1)[0]),
        )
        if args.limit:
            problem_files = problem_files[: args.limit]
        for problem_file in problem_files:
            task_id = f"L{level}_P{problem_file.name.split('_', 1)[0]}"
            task_items.append((task_id, problem_file, {}))
    rows, failures = _extract_kb_tasks(task_items, args.kernelbench_root)
    _write_batch(args.out, rows, failures, draft=False)


if __name__ == "__main__":
    main()
