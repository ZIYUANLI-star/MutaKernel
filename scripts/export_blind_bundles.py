#!/usr/bin/env python3
"""Export policy-neutral blinded evidence bundles for human audit (M9).

Primary annotators must be able to replay a counterexample without learning
which detector/strategy/policy produced it (方法V2_09 §3.3;
docs/HUMAN_CALIBRATION_PROTOCOL.md).  The operator-facing evidence bundles
written by ``run_fse_experiment.py`` expose ``case_config.json`` with policy,
seed and strategy metadata, so they must never be given to primary auditors
directly.  This tool derives, per non-pass observation:

  blind_bundles/<neutral_id>/
      blind_case.json          # neutral execution context only
      inputs.pt                # materialized input tensors (torch.save)
      replay_blind.py          # self-contained replay: load inputs -> run
                               # reference and candidate -> strict compare
  sealed/blind_mapping.json    # neutral_id -> test_id + removed fields
                               # (operator-only; integrity-hashed)

Materialisation executes the *input construction* only (reference module +
policy + seed) — it does not run the candidate.  Run this on the experiment
machine; ``--no-materialize`` produces a structure-only dry run that is NOT
annotation-ready and is marked as such.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Fields that identify the detector/strategy and must never reach primary
# auditors (方法V2_09 §3.3).
SENSITIVE_CASE_FIELDS = (
    "policy",
    "seed",
    "test_id",
    "strategy",
    "strategy_id",
    "strategy_name",
    "strategy_version",
    "scope",
)
SENSITIVE_PARAMETER_FIELDS = (
    "policy_arg_indices",
    "dtype_arg_indices",
    "batch_arg_indices",
    "layout_arg_indices",
    "batch_dimension",
)
# Semantic execution context the auditor legitimately needs to judge whether
# the input is in-contract.  Neutral names, no policy semantics.
NEUTRAL_PARAMETER_FIELDS = (
    "dtype",
    "batch_size",
    "repeat_count",
    "requires_backward",
    "layout",
)


def neutral_id(test_id: str, salt: str) -> str:
    return "blind-" + hashlib.sha256(f"{salt}|{test_id}".encode("utf-8")).hexdigest()[:16]


def neutralize_case_config(
    config: Mapping[str, Any], *, salt: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split an operator case config into (blind view, sealed remainder).

    Pure function; unit-tested without torch.  The blind view contains only
    subject sources, contract, and a neutral execution context.  Every
    removed sensitive value goes into the sealed remainder so the mapping is
    lossless for post-unblinding analysis.
    """
    case = dict(config.get("case") or {})
    parameters = dict(case.get("parameters") or {})
    test_id = str(case.get("test_id", ""))
    if not test_id:
        raise ValueError("case_config has no test_id; cannot build a stable neutral id")

    execution_context = {
        key: parameters[key] for key in NEUTRAL_PARAMETER_FIELDS if key in parameters
    }
    execution_context["mode"] = case.get("mode", "eval")

    sealed: Dict[str, Any] = {"test_id": test_id}
    for field in SENSITIVE_CASE_FIELDS:
        if field in case:
            sealed[field] = case[field]
    sealed_parameters = {
        key: parameters[key]
        for key in parameters
        if key in SENSITIVE_PARAMETER_FIELDS
    }
    if sealed_parameters:
        sealed["parameters"] = sealed_parameters

    blind = {
        "blind_schema_version": "1.0",
        "neutral_id": neutral_id(test_id, salt),
        "subject_id": config.get("subject_id"),
        "reference_path": config.get("reference_path"),
        "candidate_path": config.get("candidate_path"),
        "contract": config.get("contract"),
        "execution_context": execution_context,
        "device": config.get("device"),
    }

    leaked = [
        field for field in SENSITIVE_CASE_FIELDS + SENSITIVE_PARAMETER_FIELDS
        if field in json.dumps(blind, default=str) and field in ("policy", "seed")
        and field in (blind.get("execution_context") or {})
    ]
    if leaked:  # defence in depth; unreachable by construction
        raise AssertionError(f"sensitive fields leaked into blind view: {leaked}")
    return blind, sealed


_REPLAY_TEMPLATE = '''#!/usr/bin/env python3
"""Blind replay: run reference and candidate on the materialized inputs.

This script intentionally contains no policy, seed, or detector identity.
"""
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BLIND = json.loads((HERE / "blind_case.json").read_text(encoding="utf-8"))


def _load_module(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    device = BLIND.get("device", "cuda")
    payload = torch.load(HERE / "inputs.pt", map_location=device, weights_only=True)
    inputs = payload["inputs"]
    ref_mod = _load_module(BLIND["reference_path"], "blind_ref")
    cand_mod = _load_module(BLIND["candidate_path"], "blind_cand")
    init_args = getattr(ref_mod, "get_init_inputs", lambda: [])()
    ctx = BLIND.get("execution_context", {})
    train = ctx.get("mode") == "train"

    ref = ref_mod.Model(*init_args).to(device)
    cand_cls = getattr(cand_mod, "ModelNew", None) or cand_mod.Model
    cand = cand_cls(*init_args).to(device)
    try:
        cand.load_state_dict(ref.state_dict(), strict=True)
    except Exception as exc:
        print(f"STATE_SYNC_FAILURE: {exc}")
        return 2
    ref.train(train); cand.train(train)

    ctx_mgr = torch.enable_grad() if train else torch.no_grad()
    with ctx_mgr:
        ref_out = ref(*[x.clone() if isinstance(x, torch.Tensor) else x for x in inputs])
        cand_out = cand(*[x.clone() if isinstance(x, torch.Tensor) else x for x in inputs])

    tol = payload.get("tolerance", {"atol": 1e-2, "rtol": 1e-2})
    same = torch.allclose(
        ref_out.float().cpu(), cand_out.float().cpu(),
        atol=tol["atol"], rtol=tol["rtol"], equal_nan=True,
    )
    diff = (ref_out.float().cpu() - cand_out.float().cpu()).abs()
    print(json.dumps({
        "within_tolerance": bool(same),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
    }, indent=2))
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def materialize_inputs(
    config: Mapping[str, Any], artifact_root: Path, destination: Path
) -> Dict[str, Any]:
    """Reconstruct the concrete input tensors for a case and persist them.

    Requires torch and the reference module; runs *no* candidate code.
    """
    import torch  # noqa: PLC0415 - optional heavy dependency

    from src.stress.policy_bank import STRESS_POLICIES

    case = config["case"]
    reference_path = artifact_root / str(config["reference_path"])
    import importlib.util

    spec = importlib.util.spec_from_file_location("blind_mat_ref", reference_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    torch.manual_seed(int(case["seed"]))
    template = module.get_inputs()
    policy = str(case["policy"])
    if policy in ("iid", "identity", "__identity__"):
        inputs = template
    else:
        inputs = STRESS_POLICIES[policy](template, int(case["seed"]))

    parameters = case.get("parameters") or {}
    dtype_name = parameters.get("dtype")
    if dtype_name:
        dtype = getattr(torch, str(dtype_name))
        inputs = [
            x.to(dtype) if isinstance(x, torch.Tensor) and x.is_floating_point() else x
            for x in inputs
        ]

    cpu_inputs = [x.cpu() if isinstance(x, torch.Tensor) else x for x in inputs]
    payload = {"inputs": cpu_inputs}
    tolerance = ((config.get("contract") or {}).get("oracle") or {})
    if tolerance:
        payload["tolerance"] = tolerance
    torch.save(payload, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {"inputs_sha256": digest, "tensor_count": len(cpu_inputs)}


def export_run(
    run_dir: Path,
    output_dir: Path,
    *,
    salt: str,
    artifact_root: Path,
    materialize: bool = True,
    only_failures: bool = True,
) -> Dict[str, Any]:
    evidence_dir = run_dir / "evidence"
    if not evidence_dir.is_dir():
        raise FileNotFoundError(f"no evidence directory under {run_dir}")

    bundles_dir = output_dir / "blind_bundles"
    sealed_dir = output_dir / "sealed"
    bundles_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir.mkdir(parents=True, exist_ok=True)

    mapping: Dict[str, Any] = {}
    exported = skipped = 0
    for bundle in sorted(evidence_dir.iterdir()):
        config_path = bundle / "case_config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        result_path = bundle / "replay_result.json"
        if only_failures and result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if str(result.get("verdict", result.get("status", ""))).upper() == "PASS":
                skipped += 1
                continue

        blind, sealed = neutralize_case_config(config, salt=salt)
        nid = blind["neutral_id"]
        target = bundles_dir / nid
        target.mkdir(parents=True, exist_ok=True)
        (target / "blind_case.json").write_text(
            json.dumps(blind, indent=2, sort_keys=True), encoding="utf-8"
        )
        (target / "replay_blind.py").write_text(_REPLAY_TEMPLATE, encoding="utf-8")

        record: Dict[str, Any] = {"sealed": sealed, "source_bundle": bundle.name}
        if materialize:
            record["materialized"] = materialize_inputs(
                config, artifact_root, target / "inputs.pt"
            )
        else:
            (target / "NOT_ANNOTATION_READY").write_text(
                "inputs were not materialized; do not use for formal annotation\n",
                encoding="utf-8",
            )
            record["materialized"] = None
        mapping[nid] = record
        exported += 1

    mapping_payload = json.dumps(mapping, indent=2, sort_keys=True)
    mapping_path = sealed_dir / "blind_mapping.json"
    mapping_path.write_text(mapping_payload, encoding="utf-8")
    integrity = hashlib.sha256(mapping_payload.encode("utf-8")).hexdigest()
    (sealed_dir / "blind_mapping.sha256").write_text(integrity + "\n", encoding="utf-8")
    return {
        "exported": exported,
        "skipped_pass": skipped,
        "annotation_ready": materialize,
        "mapping_sha256": integrity,
    }


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--salt", required=True,
                        help="secret salt for neutral ids; store with the sealed mapping only")
    parser.add_argument("--no-materialize", action="store_true")
    parser.add_argument("--include-passes", action="store_true")
    args = parser.parse_args(argv)

    summary = export_run(
        args.run_dir,
        args.output_dir,
        salt=args.salt,
        artifact_root=args.artifact_root,
        materialize=not args.no_materialize,
        only_failures=not args.include_passes,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
