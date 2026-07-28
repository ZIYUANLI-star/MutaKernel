#!/usr/bin/env python3
"""E3 external-validity harness: C6 corpus + B7/B8/B9 ports + B1u anchor.

Status (2026-07-23):
  IMPLEMENTED  - C6 corpus loader (26 ops: 16 correct controls + 10 seeded
                 bugs) with --dry-run structural check (CPU-only).
  IMPLEMENTED  - unified 32-candidate-invocation budget scheduler
                 (src.experiments.external_ports.budget32): `plan` emits a
                 frozen per-subject case plan for b1u/b7/b8/b9, `run`
                 executes a plan against a KernelBench-style task module +
                 candidate module under the unified judging pipeline,
                 charging an immutable BudgetState before every candidate
                 start.
  IMPLEMENTED  - B7 robust-kbench port (multi-init x multi-input x repeat
                 pair x fwd/bwd + statistical output filters), B8
                 KernelBenchX port (standard/outlier value families), B9
                 seeded-fuzzing port (uniform[-10,10] sampling + fp64 CPU
                 reference + calibrated absolute tolerances), B1u anchor
                 (5 IID draws, unified oracle).
                 Clause-by-clause alignment: MutakernelV2/实验/补充实验数据/
                 E3_port对齐清单_B{7,8,9}.md.
  PENDING      - native-mode executions and native-dataset reproduction
                 (port-fidelity deltas B7/B8/B9) — GPU work, blocked until
                 the A800 is free; C6 GPU runner likewise.

DATA SEPARATION: `run` refuses non-CPU devices unless --allow-gpu is given,
and must only be pointed at C1/KernelBench task modules until the
fault-to-stress map is frozen.  C2-C5 candidates are never executed here.

Usage:
  # structural check of the C6 corpus:
  python scripts/run_e3_external.py c6 --corpus-root external/C6_gpuemu \
      --out-dir /tmp/e3_c6 --dry-run
  # freeze a per-subject plan (CPU-only, no execution):
  python scripts/run_e3_external.py plan --baseline b7 --subject-id L1_P1 \
      --backward --out /tmp/plan_b7.json
  # execute a plan CPU-only (KernelBench smoke):
  python scripts/run_e3_external.py run --plan /tmp/plan_b7.json \
      --task-module KernelBench/KernelBench/level1/1_Square_matrix....py \
      --candidate-module candidate.py --out-dir /tmp/e3_run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# C6: gpuemu seeded-bug corpus (third-party labels; RQ3, Table 9)
# ---------------------------------------------------------------------------

def load_c6_corpus(corpus_root: Path):
    """Load the 26-op gpuemu corpus (meta.json + ref_fp64.py + kernel.py)."""
    data_dir = corpus_root / "gpuemu-corpus" / "gpuemu_corpus" / "data"
    if not data_dir.is_dir():
        candidates = list(corpus_root.rglob("gpuemu_corpus/data"))
        if len(candidates) == 1:
            data_dir = candidates[0]
        else:
            raise FileNotFoundError(f"gpuemu corpus data dir not found under {corpus_root}")
    ops = []
    for op_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        meta_path = op_dir / "meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ref_path = data_dir / str(meta.get("reference", f"{op_dir.name}/ref_fp64.py"))
        kernel_path = data_dir / str(meta.get("kernel", f"{op_dir.name}/kernel.py"))
        ops.append({
            "op_name": op_dir.name,
            "meta": meta,
            "third_party_label": (
                "seeded_bug" if op_dir.name.endswith("_buggy")
                else "correct_control"),
            "source_tag": meta.get("source"),
            "tolerances": meta.get("tolerances"),
            "dtypes": meta.get("dtypes"),
            "ref_path": str(ref_path),
            "kernel_path": str(kernel_path),
            "has_ref": ref_path.is_file(),
            "has_kernel": kernel_path.is_file(),
        })
    return ops


def cmd_c6(args):
    ops = load_c6_corpus(args.corpus_root)
    labels = Counter(op["third_party_label"] for op in ops)
    print(f"[{_now()}] C6: {len(ops)} ops, labels={dict(labels)}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "c6_task_table.json").write_text(json.dumps({
        "created_at": _now(),
        "corpus_root": str(args.corpus_root),
        "ops": [{k: v for k, v in op.items() if k != "meta"} for op in ops],
        "label_counts": dict(labels),
        "label_note": (
            "third_party_label derives from the corpus's own *_buggy naming; "
            "reconciled 16 correct controls + 10 seeded bugs must match the "
            "gpuemu paper's published split before Table 9 is filled"),
    }, indent=2), encoding="utf-8")

    if args.dry_run:
        missing = [op["op_name"] for op in ops if not (op["has_ref"] and op["has_kernel"])]
        print(f"dry-run OK: {len(ops)} ops, incomplete={missing}")
        return

    raise NotImplementedError(
        "C6 GPU runner pending: implement the per-op execution + "
        "unified-oracle judging after the A800 frees up.  Do not run while "
        "E1 owns the GPU.")


# ---------------------------------------------------------------------------
# Unified plan/run scheduler for B1u anchor + B7/B8/B9 ports
# ---------------------------------------------------------------------------

BASELINE_ALIGNMENT_DOCS = {
    "b7": "MutakernelV2/实验/补充实验数据/E3_port对齐清单_B7.md",
    "b8": "MutakernelV2/实验/补充实验数据/E3_port对齐清单_B8.md",
    "b9": "MutakernelV2/实验/补充实验数据/E3_port对齐清单_B9.md",
    "b1u": None,
}


def build_plan(baseline: str, subject_id: str, *, backward: bool = False,
               b9_dtypes=("float32",), b9_batch_values=None):
    from src.experiments.external_ports import (
        plan_b1u,
        plan_b7_port,
        plan_b8_port,
        plan_b9_port,
    )

    if baseline == "b1u":
        cases = plan_b1u(subject_id)
    elif baseline == "b7":
        cases = plan_b7_port(subject_id, backward_supported=backward)
    elif baseline == "b8":
        cases = plan_b8_port(subject_id)
    elif baseline == "b9":
        cases = plan_b9_port(subject_id, dtypes=b9_dtypes,
                             batch_values=b9_batch_values)
    else:
        raise ValueError(f"unknown baseline {baseline!r}")

    strategy = cases[0].strategy
    return {
        "created_at": _now(),
        "baseline": baseline,
        "subject_id": subject_id,
        "strategy": strategy.identity_payload(),
        "strategy_id": strategy.strategy_id,
        "budget_matched": bool(strategy.parameters.get("budget_matched")),
        "candidate_run_budget": sum(c.candidate_run_cost for c in cases),
        "alignment_checklist": BASELINE_ALIGNMENT_DOCS[baseline],
        "cases": [case.to_dict() for case in cases],
    }


def cmd_plan(args):
    plan = build_plan(
        args.baseline, args.subject_id, backward=args.backward,
        b9_dtypes=tuple(args.b9_dtypes), b9_batch_values=args.b9_batch_values,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[{_now()}] {args.baseline} plan for {args.subject_id}: "
          f"{len(plan['cases'])} cases, "
          f"{plan['candidate_run_budget']} candidate invocations -> {args.out}")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _first_class(module, names):
    for name in names:
        cls = getattr(module, name, None)
        if cls is not None:
            return cls
    raise AttributeError(f"none of {names} found in {module.__name__}")


class PlanRunner:
    """Execute a frozen plan against a task module + candidate module.

    KernelBench-style interface: the task module provides ``Model``,
    ``get_inputs()`` and optionally ``get_init_inputs()``; the candidate
    module provides ``ModelNew`` (or ``Model``).  Every candidate start is
    charged against an immutable BudgetState *before* execution.
    """

    DEFAULT_INIT_SEED = 42

    def __init__(self, task_module, candidate_module, device: str = "cpu"):
        import torch  # deferred so `plan` stays torch-free

        self.torch = torch
        self.task_module = task_module
        self.device = device
        self.reference_cls = _first_class(task_module, ("Model",))
        self.candidate_cls = _first_class(candidate_module, ("ModelNew", "Model"))

    # -- construction ------------------------------------------------------
    def _init_inputs(self):
        get_init = getattr(self.task_module, "get_init_inputs", None)
        if get_init is None:
            return []
        values = get_init()
        return list(values) if isinstance(values, (list, tuple)) else [values]

    def _instantiate(self, cls, init_seed):
        self.torch.manual_seed(init_seed)
        model = cls(*self._init_inputs())
        return model.to(self.device).eval()

    def _template_inputs(self, seed):
        self.torch.manual_seed(seed)
        inputs = self.task_module.get_inputs()
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        return list(inputs)

    def _case_inputs(self, case):
        from src.experiments.external_ports import PORT_INPUT_GENERATORS

        policy = case["policy"]
        seed = case["seed"]
        inputs = self._template_inputs(seed)
        if policy != "iid":
            generator = PORT_INPUT_GENERATORS.get(policy)
            if generator is None:
                raise ValueError(f"unknown port input policy {policy!r}")
            inputs = generator(inputs, seed)
        dtype_name = case.get("parameters", {}).get("dtype")
        if dtype_name:
            dtype = getattr(self.torch, dtype_name)
            inputs = [
                value.to(dtype)
                if isinstance(value, self.torch.Tensor) and value.is_floating_point()
                else value
                for value in inputs
            ]
        return [
            value.to(self.device) if isinstance(value, self.torch.Tensor) else value
            for value in inputs
        ]

    # -- execution modes ---------------------------------------------------
    def _run_eval_pair(self, case, inputs, oracle_config):
        from src.validation import validate_pair

        init_seed = case.get("parameters", {}).get("init_seed",
                                                   self.DEFAULT_INIT_SEED)
        reference = self._instantiate(self.reference_cls, init_seed)
        candidate = self._instantiate(self.candidate_cls, init_seed)
        result = validate_pair(reference, candidate, tuple(inputs),
                               oracle_config=oracle_config)
        return {"status": result.status.name, "reason": result.reason}

    def _run_backward_pair(self, case, inputs, oracle_config):
        from src.validation import compare_outputs
        from src.validation.inputs import clone_tree

        torch = self.torch
        init_seed = case.get("parameters", {}).get("init_seed",
                                                   self.DEFAULT_INIT_SEED)

        def run_with_grads(model):
            args = clone_tree(list(inputs))
            leaves = []
            for value in args:
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    value.requires_grad_(True)
                    leaves.append(value)
            output = model(*args)
            if not isinstance(output, torch.Tensor):
                raise TypeError("backward comparison supports tensor outputs")
            grads = torch.autograd.grad(
                output, leaves, grad_outputs=torch.ones_like(output),
                allow_unused=True,
            )
            return {
                "output": output.detach(),
                "grads": [g.detach() if g is not None else None for g in grads],
            }

        reference = self._instantiate(self.reference_cls, init_seed).train()
        candidate = self._instantiate(self.candidate_cls, init_seed).train()
        try:
            reference_obs = run_with_grads(reference)
        except Exception as exc:  # reference failure = undecidable
            return {"status": "INCONCLUSIVE",
                    "reason": f"reference backward failed: {exc}"}
        try:
            candidate_obs = run_with_grads(candidate)
        except Exception as exc:
            return {"status": "FAIL",
                    "reason": f"candidate backward failed: {exc}"}
        oracle = compare_outputs(reference_obs, candidate_obs, oracle_config)
        return {"status": oracle.status.name, "reason": oracle.reason}

    def _run_b9_case(self, case, inputs):
        from src.experiments.external_ports import (
            B9_DEFAULT_TOLERANCES,
            compare_to_fp64,
            run_fp64_cpu_reference,
        )

        torch = self.torch
        init_seed = case.get("parameters", {}).get("init_seed",
                                                   self.DEFAULT_INIT_SEED)
        dtype_name = case.get("parameters", {}).get("dtype", "float32")
        tol = case.get("parameters", {}).get(
            "calibrated_tol", B9_DEFAULT_TOLERANCES[dtype_name])

        candidate = self._instantiate(self.candidate_cls, init_seed)
        try:
            with torch.no_grad():
                candidate_out = candidate(*[
                    v.clone() if isinstance(v, torch.Tensor) else v
                    for v in inputs
                ])
        except Exception as exc:
            return {"status": "FAIL",
                    "reason": f"candidate execution failed: {exc}"}
        if not isinstance(candidate_out, torch.Tensor):
            return {"status": "INCONCLUSIVE",
                    "reason": "B9 port supports single-tensor outputs"}
        reference = self._instantiate(self.reference_cls, init_seed)
        try:
            ideal = run_fp64_cpu_reference(reference, inputs, candidate_out.dtype)
        except Exception as exc:
            return {"status": "INCONCLUSIVE",
                    "reason": f"fp64 reference failed: {exc}"}
        verdict = compare_to_fp64(candidate_out.cpu(), ideal, tol)
        return {
            "status": "PASS" if verdict["passed"] else "FAIL",
            "reason": (None if verdict["passed"]
                       else f"fp64 differential {verdict['failure_kind']}"),
            "b9": {k: verdict[k] for k in
                   ("failure_kind", "max_abs_err", "max_rel_err")},
            "calibrated_tol": tol,
        }

    # -- driver ------------------------------------------------------------
    def run_plan(self, plan, budget_state, oracle_config=None):
        from src.validation import OracleConfig

        oracle_config = oracle_config or OracleConfig()
        b9_strategy = plan["strategy"]["name"].startswith("b9-")
        observations = []
        for case in plan["cases"]:
            cost = int(case["candidate_run_cost"])
            decision = budget_state.can_start(candidate_runs=cost)
            if not decision.allowed:
                raise RuntimeError(
                    f"budget exhausted before case {case['test_id']}: "
                    f"{decision.reason}")
            budget_state = budget_state.charge(candidate_runs=cost)

            inputs = self._case_inputs(case)
            if b9_strategy:
                outcome = self._run_b9_case(case, inputs)
            elif case["mode"] == "train":
                outcome = self._run_backward_pair(case, inputs, oracle_config)
            elif case["mode"] == "repeated":
                repeat_count = int(case["parameters"]["repeat_count"])
                trials = [
                    self._run_eval_pair(case, inputs, oracle_config)
                    for _ in range(repeat_count)
                ]
                statuses = {t["status"] for t in trials}
                if "FAIL" in statuses:
                    status = "FAIL"
                elif "INCONCLUSIVE" in statuses:
                    status = "INCONCLUSIVE"
                else:
                    status = "PASS"
                outcome = {
                    "status": status,
                    "reason": "; ".join(
                        t["reason"] for t in trials if t.get("reason")) or None,
                    "trials": trials,
                }
            else:
                outcome = self._run_eval_pair(case, inputs, oracle_config)

            observations.append({
                "test_id": case["test_id"],
                "policy": case["policy"],
                "seed": case["seed"],
                "mode": case["mode"],
                "parameters": case["parameters"],
                "candidate_run_cost": cost,
                **outcome,
            })
        return observations, budget_state


def cmd_run(args):
    if args.device != "cpu" and not args.allow_gpu:
        raise SystemExit(
            "refusing non-CPU device without --allow-gpu (E1 owns the GPU; "
            "data-separation red line)")

    from src.experiments.external_ports import fresh_budget_state
    from src.experiments.budget import BudgetLimit, BudgetState

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    task_module = _load_module(args.task_module, "e3_task_module")
    candidate_module = _load_module(args.candidate_module, "e3_candidate_module")

    if plan["budget_matched"]:
        budget_state = fresh_budget_state()
    else:
        budget_state = BudgetState(limit=BudgetLimit(
            max_candidate_runs=plan["candidate_run_budget"]))

    runner = PlanRunner(task_module, candidate_module, device=args.device)
    observations, budget_state = runner.run_plan(plan, budget_state)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{plan['baseline']}_{plan['subject_id']}_observations.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for obs in observations:
            fh.write(json.dumps(obs, sort_keys=True, default=str) + "\n")
    summary = {
        "finished_at": _now(),
        "baseline": plan["baseline"],
        "subject_id": plan["subject_id"],
        "strategy_id": plan["strategy_id"],
        "cases": len(observations),
        "candidate_runs_charged": budget_state.candidate_runs,
        "statuses": dict(Counter(obs["status"] for obs in observations)),
        "observations_path": str(out_path),
    }
    (args.out_dir / f"{plan['baseline']}_{plan['subject_id']}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    c6 = sub.add_parser("c6", help="gpuemu seeded-bug corpus (RQ3)")
    c6.add_argument("--corpus-root", required=True, type=Path)
    c6.add_argument("--out-dir", required=True, type=Path)
    c6.add_argument("--dry-run", action="store_true")
    c6.set_defaults(func=cmd_c6)

    plan = sub.add_parser("plan", help="freeze a per-subject case plan")
    plan.add_argument("--baseline", required=True,
                      choices=["b1u", "b7", "b8", "b9"])
    plan.add_argument("--subject-id", required=True)
    plan.add_argument("--backward", action="store_true",
                      help="subject contract authorizes backward (B7)")
    plan.add_argument("--b9-dtypes", nargs="+", default=["float32"])
    plan.add_argument("--b9-batch-values", nargs="+", type=int, default=None)
    plan.add_argument("--out", required=True, type=Path)
    plan.set_defaults(func=cmd_plan)

    run = sub.add_parser("run", help="execute a frozen plan (CPU smoke / GPU later)")
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--task-module", required=True, type=Path)
    run.add_argument("--candidate-module", required=True, type=Path)
    run.add_argument("--device", default="cpu")
    run.add_argument("--allow-gpu", action="store_true")
    run.add_argument("--out-dir", required=True, type=Path)
    run.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
