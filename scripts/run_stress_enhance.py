#!/usr/bin/env python3
"""StressEnhance 实验脚本 — Mutation-Guided Diagnosis and Augmentation.

三层难度框架 + 双轨道 (main fixed-shape + config-stress batch variation)
设计原则: 统一覆盖、跨维度不早停、维度内保留早停
所有 CUDA 操作在子进程中完成。
"""
import gc
import json
import logging
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models import (
    KernelInfo, Mutant, MutantStatus, MutationSite, MutationTestResult,
)
from src.mutengine.mutant_runner import MutantRunner
from src.stress.policy_bank import STRESS_POLICIES, get_all_policy_names
from src.stress.differential_tester import StressTestResult, StressSummary
from src.stress.llm_analyzer import (
    build_analysis_prompt,
    build_reanalysis_prompt,
    parse_llm_response,
    validate_suggested_code,
    OPERATOR_DESCRIPTIONS,
)

KB_ROOT = Path("/home/kbuser/projects/KernelBench-0")
BEST_KERNELS_FILE = PROJECT_ROOT / "best_kernels.json"
PROBLEM_DIRS = {
    "L1": KB_ROOT / "KernelBench" / "level1",
    "L2": KB_ROOT / "KernelBench" / "level2",
}

BLOCK12_RESULT_DIR = PROJECT_ROOT / "第二次实验汇总" / "full_block12_results"
STRESS_RESULT_DIR = PROJECT_ROOT / "第二次实验汇总" / "stress_enhance_results"
WORKER_SCRIPT = SCRIPT_DIR / "_stress_worker.py"

STRESS_TIMEOUT = 180
MAX_CONSECUTIVE_TIMEOUTS = 5
DEVICE = "cuda"
SEED = 42
ATOL = 1e-2
RTOL = 1e-2

N_SEEDS_PER_POLICY = 3
N_SEEDS_DTYPE = 3
N_SEEDS_REPEATED = 3
N_SEEDS_TRAINING = 3

LLM_API_KEY = os.environ.get(
    "LLM_API_KEY", os.environ.get(
        "DEEPSEEK_API_KEY", "sk-b896056753ec440cb735873f0179bb67"))
LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://api.deepseek.com")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-reasoner")
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "16384"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
LLM_MAX_ROUNDS = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("stress_enhance")


def P(msg):
    print(msg, flush=True)


def find_problem_file(problem_dir: Path, problem_id) -> Path | None:
    pid = str(problem_id)
    for f in problem_dir.iterdir():
        if f.name.startswith(f"{pid}_") and f.suffix == ".py":
            return f
    return None


def detect_language(source: str) -> str:
    indicators = ["__global__", "__device__", "load_inline", "cuda_source"]
    return "cuda" if sum(1 for ind in indicators if ind in source) >= 2 else "triton"


def gpu_cleanup():
    stale = [k for k in sys.modules if k.startswith(("mutant_", "ref_", "stress_"))]
    for k in stale:
        del sys.modules[k]
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass


def gpu_health_check() -> bool:
    try:
        t = torch.zeros(1, device="cuda")
        _ = t + 1
        del t
        torch.cuda.synchronize()
        return True
    except Exception as e:
        P(f"    [GPU HEALTH] FAILED: {e}")
        return False


# ---------------------------------------------------------------------------
# Load Block 1-2 results and reconstruct survived mutants
# ---------------------------------------------------------------------------

ENHANCEABLE_STATUSES = {"survived", "candidate_equivalent"}


def load_all_enhanceable(result_dir: Path):
    """Load all SURVIVED + CANDIDATE_EQUIVALENT mutants from Block 1-2 results."""
    details_dir = result_dir / "details"
    if not details_dir.exists():
        return []

    items = []
    for jf in sorted(details_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        kernel_name = data["kernel"]["problem_name"]
        for m in data.get("mutants", []):
            if m.get("status", "") in ENHANCEABLE_STATUSES:
                items.append((kernel_name, data["kernel"], m))
    return items


def classify_tier(mutant_meta: dict) -> int:
    """Classify a mutant into Tier 1, 2, or 3 based on equiv_detail."""
    status = mutant_meta.get("status", "survived")
    ed = mutant_meta.get("equiv_detail", {})

    if status == "candidate_equivalent":
        return 3

    l2 = ed.get("layer2", {})
    if l2 and l2.get("is_equivalent") is False:
        return 1

    l3 = ed.get("layer3", {})
    if l3 and l3.get("verdict") == "possibly_killable":
        return 2

    return 2


def should_challenge_tier3(mutant_meta: dict) -> bool:
    """Filter Tier 3 CANDIDATE_EQ for subset challenge."""
    ed = mutant_meta.get("equiv_detail", {})
    l3 = ed.get("layer3", {})
    confidence = l3.get("confidence", 1.0)
    if confidence < 0.98:
        return True
    op_name = mutant_meta.get("operator_name", "")
    if op_name in ("sync_remove", "launch_config_mutate", "mask_boundary",
                    "index_replace", "relop_replace", "const_perturb",
                    "arith_replace", "cast_remove", "init_modify",
                    "scale_modify"):
        return True
    return False


def reconstruct_mutant_code(kernel_info: KernelInfo, mutant_meta: dict) -> str | None:
    """Regenerate the mutated code by re-running the mutation operator."""
    runner = MutantRunner(categories=["A", "B", "C", "D"])
    all_mutants = runner.generate_mutants(kernel_info)

    target_op = mutant_meta["operator_name"]
    target_line = mutant_meta["site"]["line_start"]
    target_id = mutant_meta["id"]

    for m in all_mutants:
        if m.id == target_id:
            return m.mutated_code
        if (m.operator_name == target_op
                and m.site.line_start == target_line
                and m.site.original_code == mutant_meta["site"].get("original_code", "")):
            return m.mutated_code

    return None


# ---------------------------------------------------------------------------
# Subprocess stress worker
# ---------------------------------------------------------------------------

def _run_stress_worker(cfg: dict, timeout: int) -> dict | None:
    """Run _stress_worker.py in a subprocess."""
    cfg_fd, cfg_path = tempfile.mkstemp(suffix=".json", prefix="stresscfg_")
    res_fd, res_path = tempfile.mkstemp(suffix=".json", prefix="stressres_")
    os.close(cfg_fd)
    os.close(res_fd)

    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    try:
        proc = subprocess.Popen(
            [sys.executable, str(WORKER_SCRIPT), cfg_path, res_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
            start_new_session=True,
        )
        try:
            _, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.wait()
            return None

        if os.path.exists(res_path) and os.path.getsize(res_path) > 2:
            with open(res_path) as f:
                return json.load(f)
        return None

    except Exception:
        return None

    finally:
        for p in [cfg_path, res_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Constants & Strategy
# ---------------------------------------------------------------------------

from src.mutrepair.enhanced_inputs import STRATEGY_MAP

REPEATED_RUN_TRIALS = 10
TIER3_SEEDS_PER_POLICY = 5

TRAINING_TARGET_OPS = frozenset({
    "epsilon_modify", "const_perturb", "init_modify",
    "arith_replace", "cast_remove",
})

CONFIG_STRESS_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
CONFIG_STRESS_SEEDS = [42, 123, 7777]

SHAPE_CHANGE_KEYWORDS = [
    "change shape", "different shape", "different dimension",
    "vary m/n/k", "vary m,n,k", "non-divisible size", "non-divisible",
    "change the input size", "change input dimensions",
    "different batch", "vary batch", "change batch_size",
    "modify the shape", "alter the dimensions",
]


def _get_new_policies(operator_name: str, all_policies: list[str],
                      mutant_meta: dict) -> list[str]:
    """Return policies NOT already tested in EMD Layer 2, with STRATEGY_MAP priority."""
    ed = mutant_meta.get("equiv_detail", {})
    l2 = ed.get("layer2", {})
    tested = set()
    for p in l2.get("tested_policies", []):
        if isinstance(p, dict):
            tested.add(p.get("name", ""))
        elif isinstance(p, str):
            tested.add(p)
    mapped = STRATEGY_MAP.get(operator_name, [])
    mapped_new = [p for p in mapped if p not in tested]
    remaining = [p for p in all_policies if p not in tested and p not in mapped]
    return mapped_new + remaining


# ---------------------------------------------------------------------------
# Dimension runners: each returns a result dict, no shared mutable state
# ---------------------------------------------------------------------------

def _run_value_stress(
    problem_file: Path,
    kernel_code: str,
    mutated_code: str,
    policies: list[str],
    seeds_per_policy: int = None,
) -> dict:
    """Value-Distribution Stress. Within-dimension early-stop on kill."""
    if seeds_per_policy is None:
        seeds_per_policy = N_SEEDS_PER_POLICY
    total_calls = len(policies) * seeds_per_policy
    call_idx = 0
    consecutive_timeouts = 0

    dim = {
        "executed": True, "killed": False,
        "killing_policy": None, "killing_seed": None, "kill_type": None,
        "rounds_executed": 0, "rounds_total": total_calls,
        "policy_results": [], "original_failures": [],
        "aborted_reason": None,
    }

    for pi, policy_name in enumerate(policies):
        for si in range(seeds_per_policy):
            call_idx += 1
            dim["rounds_executed"] = call_idx
            seed = SEED + pi * seeds_per_policy + si
            P(f"    [value_stress {call_idx}/{total_calls}] "
              f"policy={policy_name} seed={seed}")

            cfg = {
                "mode": "value_stress",
                "problem_file": str(problem_file),
                "kernel_code": kernel_code, "mutated_code": mutated_code,
                "policy_name": policy_name, "seed": seed,
                "atol": ATOL, "rtol": RTOL, "device": DEVICE,
            }
            data = _run_stress_worker(cfg, timeout=STRESS_TIMEOUT)

            pr = {"policy": policy_name, "seed": seed}
            if data is None:
                pr.update(original_ok=False, mutant_ok=False, ref_ok=False,
                          error="timeout/crash")
                P(f"      -> TIMEOUT/CRASH")
                consecutive_timeouts += 1
                if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                    P(f"    -> value_stress ABORT: {consecutive_timeouts} "
                      f"consecutive timeouts, skipping dimension")
                    dim["policy_results"].append(pr)
                    dim["aborted_reason"] = (
                        f"consecutive_timeouts_{consecutive_timeouts}")
                    return dim
            else:
                consecutive_timeouts = 0
                bitwise_eq = data.get("bitwise_orig_mut_eq", True)
                pr.update(
                    original_ok=data.get("original_ok", False),
                    mutant_ok=data.get("mutant_ok", False),
                    ref_ok=data.get("ref_ok", True),
                    time_ms=data.get("time_ms", 0),
                    error=data.get("error", ""),
                )

                fallback = data.get("ref_nan_fallback", False)
                if fallback:
                    pr["ref_nan_fallback"] = True

                if not pr["ref_ok"]:
                    P(f"      -> REF_FAIL (skip)")
                elif pr["original_ok"] and not pr["mutant_ok"]:
                    fb_tag = " [ref_nan_fallback]" if fallback else ""
                    P(f"      -> KILLED by {policy_name} (seed={seed}){fb_tag}")
                    dim["killed"] = True
                    dim["killing_policy"] = policy_name
                    dim["killing_seed"] = seed
                    dim["kill_type"] = "allclose"
                elif pr["original_ok"] and pr["mutant_ok"] and not bitwise_eq:
                    P(f"      -> KILLED (bitwise divergence) by "
                      f"{policy_name} (seed={seed})")
                    dim["killed"] = True
                    dim["killing_policy"] = policy_name
                    dim["killing_seed"] = seed
                    dim["kill_type"] = "bitwise"
                elif not pr["original_ok"] and not pr["mutant_ok"]:
                    P(f"      -> ORIG_ALSO_FAILS")
                    if policy_name not in dim["original_failures"]:
                        dim["original_failures"].append(policy_name)
                elif not pr["original_ok"] and pr["mutant_ok"]:
                    P(f"      -> ORIG_FAIL_ONLY")
                else:
                    P(f"      -> both OK (bitwise_eq={bitwise_eq})")

            dim["policy_results"].append(pr)

            if dim["killed"]:
                P(f"    -> value_stress early stop: killed by "
                  f"{policy_name} seed={seed}")
                return dim

    return dim


def _run_dtype_stress(
    problem_file: Path,
    kernel_code: str,
    mutated_code: str,
) -> dict:
    """Precision switch stress (float16, bfloat16). Within-dimension early-stop."""
    dim = {
        "executed": True, "killed": False,
        "killing_dtype": None, "killing_seed": None,
        "results": [],
    }
    for si in range(N_SEEDS_DTYPE):
        seed = SEED + 100 + si
        P(f"    [dtype_stress {si+1}/{N_SEEDS_DTYPE}] seed={seed}")
        cfg = {
            "mode": "dtype_stress",
            "problem_file": str(problem_file),
            "kernel_code": kernel_code, "mutated_code": mutated_code,
            "seed": seed, "atol": ATOL, "rtol": RTOL, "device": DEVICE,
            "target_dtypes": ["float16", "bfloat16"],
        }
        data = _run_stress_worker(cfg, timeout=STRESS_TIMEOUT * 2)

        if data is None:
            P(f"      -> TIMEOUT/CRASH")
            dim["results"].append({"seed": seed, "error": "timeout/crash"})
            continue

        if data.get("killed"):
            killing_dtype = data.get("killing_dtype", "unknown")
            P(f"      -> KILLED by dtype={killing_dtype} (seed={seed})")
            dim["killed"] = True
            dim["killing_dtype"] = killing_dtype
            dim["killing_seed"] = seed
            dim["results"].append({
                "seed": seed, "killed": True,
                "killing_dtype": killing_dtype,
            })
            return dim

        tested_dtypes = data.get("tested_dtypes", cfg.get("target_dtypes", []))
        P(f"      -> not killed (seed={seed})")
        dim["results"].append({
            "seed": seed, "killed": False,
            "tested_dtypes": tested_dtypes,
        })

    return dim


def _run_repeated_run(
    problem_file: Path,
    kernel_code: str,
    mutated_code: str,
) -> dict:
    """Non-determinism detection (repeated execution). Within-dimension early-stop."""
    dim = {
        "executed": True, "killed": False,
        "inconsistency_detected": False,
        "divergent_trial": None, "killing_seed": None,
        "results": [],
    }
    for si in range(N_SEEDS_REPEATED):
        seed = SEED + 200 + si
        P(f"    [repeated_run {si+1}/{N_SEEDS_REPEATED}] "
          f"({REPEATED_RUN_TRIALS} trials) seed={seed}")
        cfg = {
            "mode": "repeated_run",
            "problem_file": str(problem_file),
            "kernel_code": kernel_code, "mutated_code": mutated_code,
            "seed": seed, "atol": ATOL, "rtol": RTOL, "device": DEVICE,
            "n_trials": REPEATED_RUN_TRIALS,
        }
        data = _run_stress_worker(cfg, timeout=STRESS_TIMEOUT * 3)

        if data is None:
            P(f"      -> TIMEOUT/CRASH")
            dim["results"].append({"seed": seed, "error": "timeout/crash"})
            continue

        if data.get("killed"):
            trial = data.get("divergent_trial")
            self_inc = data.get("self_inconsistent", False)
            reason = ("self-inconsistency" if self_inc
                      else f"diverged at trial {trial}")
            P(f"      -> KILLED ({reason}, seed={seed})")
            dim["killed"] = True
            dim["inconsistency_detected"] = self_inc
            dim["divergent_trial"] = trial
            dim["killing_seed"] = seed
            dim["results"].append({
                "seed": seed, "killed": True, "reason": reason,
            })
            return dim

        P(f"      -> not killed (seed={seed})")
        dim["results"].append({"seed": seed, "killed": False})

    return dim


def _run_training_stress(
    problem_file: Path,
    kernel_code: str,
    mutated_code: str,
    operator_name: str,
    all_policies: list[str],
) -> dict:
    """Training-mode stress (.train()). Uses ALL policies (no Layer 2 dedup).

    .train() vs .eval() is a fundamentally different execution mode, so even
    policies already tested in Layer 2 under .eval() are valid here.
    """
    if operator_name not in TRAINING_TARGET_OPS:
        P(f"    [training_stress] skip — operator "
          f"'{operator_name}' not applicable")
        return {
            "executed": False,
            "skipped_reason": "operator_not_applicable",
            "killed": False,
        }

    mapped = STRATEGY_MAP.get(operator_name, [])
    remaining = [p for p in all_policies if p not in mapped]
    priority_policies = mapped + remaining

    total_calls = len(priority_policies) * N_SEEDS_TRAINING
    call_idx = 0
    consecutive_timeouts = 0

    dim = {
        "executed": True, "killed": False, "skipped_reason": None,
        "killing_policy": None, "killing_seed": None,
        "rounds_executed": 0, "rounds_total": total_calls,
        "results": [], "original_failures": [],
        "aborted_reason": None,
    }

    for pi, policy_name in enumerate(priority_policies):
        for si in range(N_SEEDS_TRAINING):
            call_idx += 1
            dim["rounds_executed"] = call_idx
            seed = SEED + 300 + pi * N_SEEDS_TRAINING + si
            P(f"    [training_stress {call_idx}/{total_calls}] "
              f"policy={policy_name} seed={seed}")

            cfg = {
                "mode": "training_stress",
                "problem_file": str(problem_file),
                "kernel_code": kernel_code, "mutated_code": mutated_code,
                "policy_name": policy_name, "seed": seed,
                "atol": ATOL, "rtol": RTOL, "device": DEVICE,
            }
            data = _run_stress_worker(cfg, timeout=STRESS_TIMEOUT)

            if data is None:
                P(f"      -> TIMEOUT/CRASH")
                consecutive_timeouts += 1
                dim["results"].append({
                    "policy": policy_name, "seed": seed,
                    "error": "timeout/crash",
                })
                if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                    P(f"    -> training_stress ABORT: {consecutive_timeouts} "
                      f"consecutive timeouts, skipping dimension")
                    dim["aborted_reason"] = (
                        f"consecutive_timeouts_{consecutive_timeouts}")
                    return dim
                continue

            consecutive_timeouts = 0

            fallback = data.get("ref_nan_fallback", False)

            if not data.get("ref_ok", True):
                P(f"      -> REF_FAIL (skip)")
                dim["results"].append({
                    "policy": policy_name, "seed": seed, "ref_fail": True,
                })
                continue

            orig_ok = data.get("original_ok", False)
            mut_ok = data.get("mutant_ok", False)

            if orig_ok and not mut_ok:
                fb_tag = " [ref_nan_fallback]" if fallback else ""
                P(f"      -> KILLED (training mode) by "
                  f"{policy_name} seed={seed}{fb_tag}")
                dim["killed"] = True
                dim["killing_policy"] = policy_name
                dim["killing_seed"] = seed
                dim["results"].append({
                    "policy": policy_name, "seed": seed, "killed": True,
                })
                return dim
            elif not orig_ok and not mut_ok:
                P(f"      -> ORIG_ALSO_FAILS (train)")
                if policy_name not in dim["original_failures"]:
                    dim["original_failures"].append(policy_name)
            else:
                P(f"      -> both OK (train)")

            dim["results"].append({
                "policy": policy_name, "seed": seed, "killed": False,
                "original_ok": orig_ok, "mutant_ok": mut_ok,
            })

    return dim


def _run_config_stress(
    problem_file: Path,
    kernel_code: str,
    mutated_code: str,
) -> dict:
    """Config-Stress Track (auxiliary): vary batch_size."""
    P(f"    [config_stress] batch_sizes={CONFIG_STRESS_BATCH_SIZES}, "
      f"seeds={CONFIG_STRESS_SEEDS}")
    cfg = {
        "mode": "config_stress",
        "problem_file": str(problem_file),
        "kernel_code": kernel_code, "mutated_code": mutated_code,
        "device": DEVICE,
        "batch_sizes": CONFIG_STRESS_BATCH_SIZES,
        "seeds": CONFIG_STRESS_SEEDS,
    }
    data = _run_stress_worker(cfg, timeout=STRESS_TIMEOUT * 3)

    if data is None:
        P(f"      -> TIMEOUT/CRASH")
        return {"executed": True, "killed": False, "error": "timeout/crash"}

    if data.get("killed"):
        bs = data.get("killing_batch_size")
        seed = data.get("killing_seed")
        kill_type = data.get("kill_type", "unknown")
        P(f"      -> KILLED by config_stress: batch_size={bs}, "
          f"seed={seed}, type={kill_type}")
        return {
            "executed": True, "killed": True,
            "killing_batch_size": bs, "killing_seed": seed,
            "killing_policy": f"batch_size={bs}",
            "kill_type": kill_type,
            "results_per_batch": data.get("results_per_batch", {}),
        }

    P(f"      -> not killed by config_stress")
    return {
        "executed": True, "killed": False,
        "killing_batch_size": None, "kill_type": None,
        "results_per_batch": data.get("results_per_batch", {}),
    }


def _llm_suggestion_violates_fixed_shape(kill_strategy: str) -> bool:
    """Check if LLM kill_strategy suggests changing input shapes."""
    if not kill_strategy:
        return False
    lower = kill_strategy.lower()
    return any(kw in lower for kw in SHAPE_CHANGE_KEYWORDS)


def _get_input_spec_str(equiv_detail: dict) -> str:
    """Extract input_spec as a human-readable string from equiv_detail."""
    spec = equiv_detail.get("input_spec")
    if not spec:
        return "unknown"
    if isinstance(spec, str):
        return spec
    if isinstance(spec, list):
        lines = []
        for i, s in enumerate(spec):
            if isinstance(s, dict):
                lines.append(f"arg[{i}]: shape={s.get('shape')}, "
                             f"dtype={s.get('dtype')}")
            else:
                lines.append(f"arg[{i}]: {s}")
        return "\n".join(lines)
    return str(spec)


def _setup_llm_caller():
    """Create an LLM API caller function, or None if unavailable.

    Returns a callable(prompt) -> dict with keys:
        content: str           — final answer (JSON expected)
        reasoning_content: str — chain-of-thought (R1 only, else "")
        model: str             — actual model returned by API
        usage: dict            — token usage stats
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)
        is_reasoner = "reasoner" in LLM_MODEL.lower()

        def call_llm(prompt: str) -> dict:
            kwargs = dict(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=LLM_MAX_TOKENS,
            )
            if not is_reasoner:
                kwargs["temperature"] = LLM_TEMPERATURE
            resp = client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", "") or ""
            usage = resp.usage
            usage_dict = {}
            if usage:
                usage_dict = {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
                if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                    ctd = usage.completion_tokens_details
                    usage_dict["reasoning_tokens"] = getattr(ctd, "reasoning_tokens", 0) or 0
            return {
                "content": content,
                "reasoning_content": reasoning,
                "model": resp.model or LLM_MODEL,
                "usage": usage_dict,
            }

        return call_llm
    except ImportError:
        log.warning("openai package not installed, LLM iterative analysis disabled")
        return None


def _verify_llm_suggestion(
    problem_file: Path,
    kernel_code: str,
    mutated_code: str,
    python_code: str,
) -> dict:
    """Execute LLM-suggested test code via _stress_worker in llm_verify mode."""
    safety_err = validate_suggested_code(python_code)
    if safety_err:
        return {"killed": False, "ref_ok": True, "original_ok": True,
                "mutant_ok": True, "diff_summary": "",
                "error": f"safety_rejected: {safety_err}"}

    cfg = {
        "mode": "llm_verify",
        "problem_file": str(problem_file),
        "kernel_code": kernel_code, "mutated_code": mutated_code,
        "test_inputs_code": python_code,
        "atol": ATOL, "rtol": RTOL, "device": DEVICE,
    }
    data = _run_stress_worker(cfg, timeout=STRESS_TIMEOUT)

    if data is None:
        return {"killed": False, "ref_ok": False, "original_ok": False,
                "mutant_ok": False, "diff_summary": "",
                "error": "timeout/crash"}
    return {
        "killed": data.get("killed", False),
        "ref_ok": data.get("ref_ok", False),
        "original_ok": data.get("original_ok", False),
        "mutant_ok": data.get("mutant_ok", True),
        "diff_summary": data.get("diff_summary", ""),
        "error": data.get("error", ""),
    }


def _run_llm_iterative_analysis(
    problem_file: Path,
    kernel_code: str,
    mutated_code: str,
    operator_name: str,
    site: dict,
    input_spec: str,
    equiv_detail: dict,
    stress_result: StressTestResult,
) -> dict:
    """Phase 2 Step 3: LLM iterative analysis — up to LLM_MAX_ROUNDS rounds.

    Only called when all deterministic dimensions have failed to kill.
    Generates new suggested_test code via LLM, executes and verifies.
    """
    call_llm = _setup_llm_caller()
    if call_llm is None:
        P(f"    [LLM-Iter] LLM unavailable, skip")
        return {
            "executed": False, "trigger": "llm_unavailable",
            "killed": False, "rounds": [],
        }

    enhanced_data = {
        "main_track": stress_result.main_track,
        "config_track": stress_result.config_track,
    }

    rounds_history: list = []
    killed = False
    killing_round = 0

    for round_num in range(1, LLM_MAX_ROUNDS + 1):
        P(f"    [LLM-Iter] Round {round_num}/{LLM_MAX_ROUNDS}")

        try:
            if round_num == 1:
                prompt = build_analysis_prompt(
                    original_code=kernel_code,
                    mutated_code=mutated_code,
                    operator_name=operator_name,
                    site=site,
                    input_spec=input_spec,
                    equiv_detail=equiv_detail,
                    enhanced_results=enhanced_data,
                )
            else:
                prompt = build_reanalysis_prompt(
                    original_code=kernel_code,
                    mutated_code=mutated_code,
                    operator_name=operator_name,
                    site=site,
                    input_spec=input_spec,
                    previous_rounds=rounds_history,
                    equiv_detail=equiv_detail,
                    enhanced_results=enhanced_data,
                )
        except Exception as e:
            P(f"      -> Prompt build error: {e}")
            rounds_history.append({
                "round": round_num,
                "prompt_type": "error",
                "error": str(e),
                "killed": False,
            })
            break

        prompt_dir = STRESS_RESULT_DIR / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = prompt_dir / f"{stress_result.mutant_id}_r{round_num}.txt"
        try:
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(prompt)
        except Exception:
            pass

        try:
            llm_resp = call_llm(prompt)
        except Exception as e:
            P(f"      -> LLM API error: {e}")
            rounds_history.append({
                "round": round_num,
                "prompt_type": ("ANALYSIS_PROMPT_V2" if round_num == 1
                                else "REANALYSIS_PROMPT_V2"),
                "error": str(e),
                "killed": False,
            })
            break

        raw_response = llm_resp["content"]
        reasoning_content = llm_resp.get("reasoning_content", "")
        llm_usage = llm_resp.get("usage", {})
        llm_model_used = llm_resp.get("model", LLM_MODEL)

        if reasoning_content:
            P(f"      -> R1 reasoning: {len(reasoning_content)} chars")
        P(f"      -> Tokens: {llm_usage}")

        parsed = parse_llm_response(raw_response)

        resp_dir = STRESS_RESULT_DIR / "llm_responses"
        resp_dir.mkdir(parents=True, exist_ok=True)
        resp_file = resp_dir / f"{stress_result.mutant_id}_r{round_num}_response.json"
        try:
            resp_record = {
                "mutant_id": stress_result.mutant_id,
                "round": round_num,
                "model": llm_model_used,
                "killable": parsed.get("killable") if parsed else None,
                "reason_category": parsed.get("reason_category") if parsed else None,
                "proof_sketch": parsed.get("proof_sketch") if parsed else None,
                "content": raw_response,
                "reasoning_content": reasoning_content,
            }
            with open(resp_file, "w", encoding="utf-8") as f:
                json.dump(resp_record, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        if parsed is None:
            P(f"      -> LLM response unparseable")
            rounds_history.append({
                "round": round_num,
                "prompt_type": ("ANALYSIS_PROMPT_V2" if round_num == 1
                                else "REANALYSIS_PROMPT_V2"),
                "survival_reason": "parse_error",
                "killable": False,
                "kill_strategy": "",
                "suggested_code": "",
                "execution_result": None,
                "killed": False,
                "raw_response": raw_response[:500],
                "reasoning_content": reasoning_content[:1000] if reasoning_content else "",
                "model": llm_model_used,
                "usage": llm_usage,
            })
            continue

        survival_reason = parsed.get("survival_reason", "")
        is_killable = parsed.get("killable", False)
        kill_strategy = parsed.get("kill_strategy", "")
        suggested = parsed.get("suggested_test")
        recommendations = parsed.get("recommendations", "")
        reason_category = parsed.get("reason_category", "unknown")
        proof_sketch = parsed.get("proof_sketch", "")

        round_record = {
            "round": round_num,
            "prompt_type": ("ANALYSIS_PROMPT_V2" if round_num == 1
                            else "REANALYSIS_PROMPT_V2"),
            "model": llm_model_used,
            "usage": llm_usage,
            "reason_category": reason_category,
            "proof_sketch": proof_sketch,
            "survival_reason": survival_reason,
            "killable": is_killable,
            "kill_strategy": kill_strategy,
            "recommendations": recommendations,
            "suggested_code": "",
            "execution_result": None,
            "killed": False,
        }

        if not is_killable or not suggested or not isinstance(suggested, dict):
            P(f"      -> LLM says unkillable")
            rounds_history.append(round_record)
            break

        python_code = (suggested.get("python_code") or "").strip()
        if not python_code:
            P(f"      -> LLM gave no code")
            rounds_history.append(round_record)
            continue

        if _llm_suggestion_violates_fixed_shape(kill_strategy):
            P(f"      -> LLM suggestion violates fixed-shape, skip")
            round_record["execution_result"] = {
                "killed": False, "error": "violates_fixed_shape"}
            rounds_history.append(round_record)
            continue

        round_record["suggested_code"] = python_code
        desc = suggested.get("description", "N/A")
        P(f"      -> Executing: {desc[:80]}")

        exec_result = _verify_llm_suggestion(
            problem_file, kernel_code, mutated_code, python_code)
        round_record["execution_result"] = exec_result

        if exec_result.get("killed"):
            fb_tag = " [ref_nan_fallback]" if exec_result.get("ref_nan_fallback") else ""
            P(f"      *** KILLED in round {round_num}!{fb_tag} ***")
            round_record["killed"] = True
            killed = True
            killing_round = round_num
            rounds_history.append(round_record)
            break
        else:
            err = exec_result.get("error", "")
            if err:
                P(f"      -> Not killed (error: {err[:80]})")
            else:
                P(f"      -> Not killed (bitwise identical)")
            rounds_history.append(round_record)

    result = {
        "executed": True,
        "trigger": "all_dimensions_survived",
        "rounds": rounds_history,
        "killed": killed,
        "killing_round": killing_round,
    }

    if killed and killing_round > 0:
        kr = rounds_history[killing_round - 1]
        result["test_construction_rule"] = {
            "kill_strategy": kr.get("kill_strategy", ""),
            "suggested_code": kr.get("suggested_code", ""),
        }
    elif not killed and rounds_history:
        last = rounds_history[-1]
        result["robustness_suggestion"] = last.get("survival_reason", "")

    return result


def _run_tier1_replay(
    problem_file: Path,
    kernel_code: str,
    mutated_code: str,
    equiv_detail: dict,
) -> dict:
    """Tier 1 specific: replay L2 divergence using recorded seed/policy."""
    l2_div = equiv_detail.get("layer2", {}).get("divergence", {})
    replay_seed = l2_div.get("seed")
    replay_policy = l2_div.get("policy")

    if replay_seed is None or not replay_policy:
        P(f"    [tier1_replay] No L2 divergence data to replay")
        return {
            "executed": False, "killed": False,
            "detail": {"reason": "no_divergence_data"},
        }

    P(f"    [tier1_replay] Replaying L2 divergence: "
      f"policy={replay_policy}, seed={replay_seed}")
    cfg = {
        "mode": "value_stress",
        "problem_file": str(problem_file),
        "kernel_code": kernel_code, "mutated_code": mutated_code,
        "policy_name": replay_policy, "seed": replay_seed,
        "atol": ATOL, "rtol": RTOL, "device": DEVICE,
    }
    data = _run_stress_worker(cfg, timeout=STRESS_TIMEOUT)

    if data and data.get("original_ok") and not data.get("mutant_ok"):
        P(f"      -> KILLED by replay (seed={replay_seed}, "
          f"policy={replay_policy})")
        return {
            "executed": True, "killed": True,
            "detail": {
                "seed": replay_seed, "policy": replay_policy,
                "divergence_type": "allclose",
            },
        }

    bitwise_eq = data.get("bitwise_orig_mut_eq", True) if data else True
    if (data and data.get("original_ok") and data.get("mutant_ok")
            and not bitwise_eq):
        P(f"      -> KILLED by replay (bitwise, seed={replay_seed}, "
          f"policy={replay_policy})")
        return {
            "executed": True, "killed": True,
            "detail": {
                "seed": replay_seed, "policy": replay_policy,
                "divergence_type": "bitwise",
            },
        }

    P(f"      -> replay did not kill")
    return {
        "executed": True, "killed": False,
        "detail": {"seed": replay_seed, "policy": replay_policy},
    }


# ---------------------------------------------------------------------------
# Tier 3 confidence calculation
# ---------------------------------------------------------------------------

def _count_passed_rounds(stress_result: StressTestResult) -> int:
    """Count total non-killing, non-error rounds for statistical confidence."""
    n = 0
    all_dims = list(stress_result.main_track.values()) + \
               list(stress_result.config_track.values())
    for dim_r in all_dims:
        if not dim_r.get("executed"):
            continue
        for pr in dim_r.get("policy_results", []):
            if (pr.get("ref_ok", True) and pr.get("original_ok")
                    and pr.get("mutant_ok")):
                n += 1
        for rr in dim_r.get("results", []):
            if (not rr.get("error") and not rr.get("killed")
                    and not rr.get("ref_fail")):
                n += 1
    return max(n, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-mutants", type=int, default=0,
                        help="Limit number of mutants to process (0=all)")
    args = parser.parse_args()

    t_start = time.time()
    STRESS_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with open(BEST_KERNELS_FILE) as f:
        best_kernels = json.load(f)

    P(f"\n{'='*70}")
    P(f"  Mutation-Guided Diagnosis and Augmentation (Three-Tier Framework)")
    P(f"  Dual-Track: Main (fixed-shape) + Config-Stress (batch variation)")
    P(f"  Design: unified coverage, no cross-dimension early stop")
    P(f"{'='*70}")

    enhanceable_list = load_all_enhanceable(BLOCK12_RESULT_DIR)
    if args.max_mutants > 0:
        enhanceable_list = enhanceable_list[:args.max_mutants]
        P(f"  [LIMITED] Processing first {args.max_mutants} mutants")
    all_policies = get_all_policy_names()

    tier_groups = {1: [], 2: [], 3: []}
    for item in enhanceable_list:
        _, _, m = item
        tier = classify_tier(m)
        tier_groups[tier].append(item)

    tier3_filtered = [
        item for item in tier_groups[3]
        if should_challenge_tier3(item[2])
    ]

    P(f"  Total enhanceable mutants: {len(enhanceable_list)}")
    P(f"    Tier 1 (L2-rejected SURVIVED):      {len(tier_groups[1])}")
    P(f"    Tier 2 (LLM-rejected SURVIVED):      {len(tier_groups[2])}")
    P(f"    Tier 3 (CANDIDATE_EQ total):          {len(tier_groups[3])}")
    P(f"    Tier 3 (after filtering for challenge):{len(tier3_filtered)}")
    P(f"  Policies available: {len(all_policies)}")
    P(f"  Config-Stress batch_sizes: {CONFIG_STRESS_BATCH_SIZES}")
    P(f"  Timeout per worker: {STRESS_TIMEOUT}s")
    P("")

    execution_order = (
        [(1, item) for item in tier_groups[1]]
        + [(2, item) for item in tier_groups[2]]
        + [(3, item) for item in tier3_filtered]
    )

    if not execution_order:
        P("  No enhanceable mutants found. Nothing to do.")
        return

    completed_file = STRESS_RESULT_DIR / "completed.json"
    completed = set()
    if completed_file.exists():
        completed = set(json.loads(completed_file.read_text()))

    summary = StressSummary()
    tier_kills = {1: 0, 2: 0, 3: 0}
    tier_tested = {1: 0, 2: 0, 3: 0}

    for idx, (tier, (kernel_name, kernel_meta, mutant_meta)) in enumerate(
            execution_order):
        mutant_id = mutant_meta["id"]
        if mutant_id in completed:
            P(f"  [{idx+1}/{len(execution_order)}] {mutant_id} "
              f"-- already done, skip")
            continue

        equiv_detail = mutant_meta.get("equiv_detail", {})
        op_name = mutant_meta["operator_name"]

        P(f"\n{'_'*70}")
        P(f"  [{idx+1}/{len(execution_order)}] [Tier {tier}] "
          f"{kernel_name} / {op_name} "
          f"({mutant_meta.get('operator_category','?')}) "
          f"@ L{mutant_meta['site']['line_start']}  "
          f"status={mutant_meta['status']}")
        P(f"{'_'*70}")

        # --- Resolve paths ---
        level_key = f"L{kernel_meta['level']}"
        problem_dir = PROBLEM_DIRS.get(level_key)
        if not problem_dir:
            P(f"    ERROR: unknown level {level_key}")
            continue

        bk_info = best_kernels.get(kernel_name)
        if not bk_info:
            P(f"    ERROR: {kernel_name} not in best_kernels.json")
            continue

        kernel_path = Path(bk_info["kernel_path"])
        if not kernel_path.exists():
            P(f"    ERROR: kernel file not found: {kernel_path}")
            continue

        problem_file = find_problem_file(problem_dir, kernel_meta["problem_id"])
        if not problem_file:
            P(f"    ERROR: problem file not found for "
              f"P{kernel_meta['problem_id']}")
            continue

        kernel_code = kernel_path.read_text(encoding="utf-8")
        mutated_code = mutant_meta.get("mutated_code", "")
        if mutated_code:
            P(f"    Loaded mutant code from JSON ({len(mutated_code)} chars)")
        else:
            kernel_info = KernelInfo(
                problem_id=kernel_meta["problem_id"],
                level=kernel_meta["level"],
                problem_name=kernel_name,
                source_path=str(kernel_path),
                kernel_code=kernel_code,
                reference_module_path=str(problem_file),
                language=kernel_meta.get("language",
                                         detect_language(kernel_code)),
            )
            P(f"    Reconstructing mutant code...")
            mutated_code = reconstruct_mutant_code(kernel_info, mutant_meta)
            if mutated_code is None:
                P(f"    ERROR: could not reconstruct mutant code, skipping")
                continue
            P(f"    Mutant code reconstructed ({len(mutated_code)} chars)")

        stress_result = StressTestResult(
            mutant_id=mutant_id,
            operator_name=op_name,
            operator_category=mutant_meta.get("operator_category", ""),
            kernel_name=kernel_name,
            site_node_type=mutant_meta.get("site", {}).get("node_type", ""),
        )

        new_policies = _get_new_policies(op_name, all_policies, mutant_meta)
        seeds_per_policy = (TIER3_SEEDS_PER_POLICY if tier == 3
                            else N_SEEDS_PER_POLICY)

        # =============================================================
        #  Unified coverage: ALL dimensions for ALL tiers
        #  No cross-dimension early stop (§5.1.6)
        #  Execution order varies by Tier (§5.1.3)
        # =============================================================

        if tier == 1:
            P(f"    --- Tier 1: replay → value → dtype → "
              f"training → repeated → config ---")

            r = _run_tier1_replay(
                problem_file, kernel_code, mutated_code, equiv_detail)
            stress_result.record_dimension("main", "tier1_replay", r)

            P(f"    --- value_stress ({len(new_policies)} new policies "
              f"× {seeds_per_policy} seeds) ---")
            r = _run_value_stress(
                problem_file, kernel_code, mutated_code,
                new_policies, seeds_per_policy)
            stress_result.record_dimension("main", "value_stress", r)

            P(f"    --- dtype_stress ---")
            r = _run_dtype_stress(
                problem_file, kernel_code, mutated_code)
            stress_result.record_dimension("main", "dtype_stress", r)

            P(f"    --- training_stress ---")
            r = _run_training_stress(
                problem_file, kernel_code, mutated_code,
                op_name, all_policies)
            stress_result.record_dimension("main", "training_stress", r)

            P(f"    --- repeated_run ---")
            r = _run_repeated_run(
                problem_file, kernel_code, mutated_code)
            stress_result.record_dimension("main", "repeated_run", r)

            P(f"    --- config_stress (auxiliary track) ---")
            r = _run_config_stress(
                problem_file, kernel_code, mutated_code)
            stress_result.record_dimension("config", "config_stress", r)

        elif tier == 2:
            P(f"    --- Tier 2: dtype → training → "
              f"value → repeated → config ---")

            P(f"    --- dtype_stress ---")
            r = _run_dtype_stress(
                problem_file, kernel_code, mutated_code)
            stress_result.record_dimension("main", "dtype_stress", r)

            P(f"    --- training_stress ---")
            r = _run_training_stress(
                problem_file, kernel_code, mutated_code,
                op_name, all_policies)
            stress_result.record_dimension("main", "training_stress", r)

            P(f"    --- value_stress ({len(new_policies)} new policies "
              f"× {seeds_per_policy} seeds) ---")
            r = _run_value_stress(
                problem_file, kernel_code, mutated_code,
                new_policies, seeds_per_policy)
            stress_result.record_dimension("main", "value_stress", r)

            P(f"    --- repeated_run ---")
            r = _run_repeated_run(
                problem_file, kernel_code, mutated_code)
            stress_result.record_dimension("main", "repeated_run", r)

            P(f"    --- config_stress (auxiliary track) ---")
            r = _run_config_stress(
                problem_file, kernel_code, mutated_code)
            stress_result.record_dimension("config", "config_stress", r)

        elif tier == 3:
            P(f"    --- Tier 3: dtype → value(5 seeds) → "
              f"repeated → training → config ---")

            P(f"    --- dtype_stress ---")
            r = _run_dtype_stress(
                problem_file, kernel_code, mutated_code)
            stress_result.record_dimension("main", "dtype_stress", r)

            P(f"    --- value_stress ({len(new_policies)} new policies "
              f"× {seeds_per_policy} seeds) ---")
            r = _run_value_stress(
                problem_file, kernel_code, mutated_code,
                new_policies, seeds_per_policy)
            stress_result.record_dimension("main", "value_stress", r)

            P(f"    --- repeated_run ---")
            r = _run_repeated_run(
                problem_file, kernel_code, mutated_code)
            stress_result.record_dimension("main", "repeated_run", r)

            P(f"    --- training_stress ---")
            r = _run_training_stress(
                problem_file, kernel_code, mutated_code,
                op_name, all_policies)
            stress_result.record_dimension("main", "training_stress", r)

            P(f"    --- config_stress (auxiliary track) ---")
            r = _run_config_stress(
                problem_file, kernel_code, mutated_code)
            stress_result.record_dimension("config", "config_stress", r)

        # --- Step 3: LLM iterative analysis (only if all dimensions survived) ---
        if not stress_result.deterministic_killed:
            P(f"    --- LLM iterative analysis (all deterministic dims survived) ---")
            input_spec_str = _get_input_spec_str(equiv_detail)
            llm_result = _run_llm_iterative_analysis(
                problem_file, kernel_code, mutated_code,
                op_name, mutant_meta.get("site", {}),
                input_spec_str, equiv_detail, stress_result)
            stress_result.record_llm_analysis(llm_result)
            if llm_result.get("killed"):
                P(f"    *** LLM killed in round {llm_result.get('killing_round')} ***")
            else:
                P(f"    LLM analysis: not killed after "
                  f"{len(llm_result.get('rounds', []))} rounds")
        else:
            P(f"    --- LLM analysis skipped (already killed by deterministic dims) ---")
            stress_result.record_llm_analysis({
                "executed": False,
                "trigger": "already_killed",
                "killed": False,
                "rounds": [],
            })

        # --- Result tracking ---
        tier_tested[tier] += 1
        if stress_result.any_killed:
            tier_kills[tier] += 1

        summary.add_result(stress_result)

        # --- Build output JSON (ordered for readability) ---
        sr = stress_result.to_dict()
        detail_data = {
            # 1. 基本信息
            "mutant_id": sr["mutant_id"],
            "operator_name": sr["operator_name"],
            "operator_category": sr["operator_category"],
            "kernel_name": sr["kernel_name"],
            "tier": tier,
            "original_status": mutant_meta["status"],
            "site_node_type": sr["site_node_type"],
            # 2. Phase 1: 等价变异体检测 (EMD Layer 0-3)
            "equiv_detail": equiv_detail,
            # 3. Phase 2: 增强测试
            "main_track": sr["main_track"],
            "config_track": sr["config_track"],
            "original_failures": sr["original_failures"],
            # 4. Phase 2: LLM 迭代分析
            "llm_iterative_analysis": sr["llm_iterative_analysis"],
            # 5. 最终结论
            "kill_summary": sr["kill_summary"],
            "any_killed": sr["any_killed"],
            "first_kill_mode": sr["first_kill_mode"],
            "total_time_ms": sr["total_time_ms"],
        }

        if tier == 3 and not stress_result.any_killed:
            n_passed = _count_passed_rounds(stress_result)
            detail_data["tier3_confidence"] = {
                "total_passed_rounds": n_passed,
                "confidence_equivalent_lower_bound": round(
                    1.0 - (1.0 / (n_passed + 1)), 4),
                "interpretation": (
                    f"After {n_passed} independent non-killing test rounds "
                    f"across all dimensions, at 95%% confidence p < "
                    f"{round(1 - 0.05**(1/max(n_passed,1)), 4)}."
                ),
            }

        detail_path = STRESS_RESULT_DIR / "details" / f"{mutant_id}.json"
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        with open(detail_path, "w") as f:
            json.dump(detail_data, f, indent=2, ensure_ascii=False)

        completed.add(mutant_id)
        with open(completed_file, "w") as f:
            json.dump(sorted(completed), f)

        ks = stress_result.get_kill_summary()
        killed_by = ks["main_track_killed_by"] + ks["config_track_killed_by"]
        status_str = (f"KILLED by {killed_by}" if stress_result.any_killed
                      else "SURVIVED (enhanced)")
        P(f"    Status: {status_str}")
        gpu_cleanup()

        if not gpu_health_check():
            P(f"    [WARNING] GPU unresponsive after mutant, waiting 10s...")
            time.sleep(10)
            if not gpu_health_check():
                P(f"    [FATAL] GPU still unresponsive, aborting")
                break

    elapsed = time.time() - t_start
    P(f"\n{'='*70}")
    P(f"  Three-Tier Enhanced Testing Complete (Unified Coverage)")
    P(f"{'='*70}")
    P(f"  Total tested:           {summary.total_tested}")
    P(f"  Killed:                 {summary.killed_count}")
    P(f"  Survived:               {summary.survived_count}")
    P(f"  Multi-dim kills:        {summary.multi_dimension_kill_count}")
    P(f"  Per-tier tested:        {dict(tier_tested)}")
    P(f"  Per-tier kills:         {dict(tier_kills)}")
    P(f"  Per-dimension kills:    {summary.per_dimension_kills}")
    P(f"  Per-policy kills:       {summary.per_policy_kills}")
    P(f"  Elapsed: {elapsed/60:.1f} min")

    summary_data = summary.to_dict()
    summary_data["tier_tested"] = dict(tier_tested)
    summary_data["tier_kills"] = dict(tier_kills)

    summary_path = STRESS_RESULT_DIR / "stress_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    P(f"  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
