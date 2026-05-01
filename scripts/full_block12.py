#!/usr/bin/env python3
"""Block 1 + Block 2 全量变异测试（L1 + L2）— V2 等价检测版。

第二次全量实验，使用 V2 四层等价检测流水线:
  Layer 0: CUDA 源码归一化 → STRICT_EQUIVALENT
  Layer 1: 算子静态规则 → STRICT_EQUIVALENT
  Layer 2: 动态 bitwise (NaN-aware + CompareResult + 算子定向策略)
           → CANDIDATE_EQUIVALENT
  Layer 3: LLM 验证 (后续单独执行)

设计原则:
  主进程 **永远不执行 CUDA 操作**。
  所有编译 / 运行 / 等价检测均在独立 subprocess 中完成。
  单个变异体的崩溃 / 挂起 / 非法内存访问只影响该子进程，
  主进程标记为 STILLBORN 后继续下一个变异体。
  不跳过任何 kernel — 如果处理异常则重试，绝不放弃。

流程:
  Block 1: MutantRunner.generate_mutants() → 纯 Python AST，无 CUDA
  Block 2: 每个变异体 → subprocess _mutant_worker.py (mode=run)
           等价检测三层流水线 (Layer 0/1 主进程, Layer 2 子进程)
           MutationReporter → 保存 JSON
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

from src.models import KernelInfo, Mutant, MutantStatus, MutationTestResult
from src.mutengine.mutant_runner import MutantRunner
from src.mutengine.report import MutationReporter
from src.bridge.eval_bridge import _load_module_from_path

KB_ROOT = Path("/home/kbuser/projects/KernelBench-0")
BEST_KERNELS_FILE = PROJECT_ROOT / "best_kernels.json"
PROBLEM_DIRS = {
    "L1": KB_ROOT / "KernelBench" / "level1",
    "L2": KB_ROOT / "KernelBench" / "level2",
}

SAMPLE_PER_OP = 3
EQUIV_RUNS = 100
NUM_TEST_INPUTS = 3
DEVICE = "cuda"
SEED = 42
ATOL = 1e-2
RTOL = 1e-2

MUTANT_TIMEOUT = 180   # 3 min per mutant (compile + run)
EQUIV_TIMEOUT = 600    # 10 min per equiv check (compile + N runs + stress)
MAX_KERNEL_RETRIES = 3
MAX_KERNELS = 0        # 0 = unlimited; set >0 to limit kernels for debugging

# Layer 3: LLM equiv verification
LLM_API_KEY = os.environ.get("LLM_API_KEY", os.environ.get(
    "DEEPSEEK_API_KEY", "sk-b896056753ec440cb735873f0179bb67"))
LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://api.deepseek.com")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 4096
LLM_EQUIV_VERIFY = bool(LLM_API_KEY)  # auto-enable if API key is set

RESULT_DIR = PROJECT_ROOT / "第二次实验汇总" / "full_block12_results"
OUTPUT_LOG = PROJECT_ROOT / "第二次实验汇总" / "full_block12_output.txt"
WORKER_SCRIPT = SCRIPT_DIR / "_mutant_worker.py"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("full_block12")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def P(msg):
    print(msg, flush=True)
    with open(OUTPUT_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def find_problem_file(problem_dir: Path, problem_id) -> Path | None:
    pid = str(problem_id)
    for f in problem_dir.iterdir():
        if f.name.startswith(f"{pid}_") and f.suffix == ".py":
            return f
    return None


def detect_language(source: str) -> str:
    indicators = ["__global__", "__device__", "load_inline", "cuda_source"]
    return "cuda" if sum(1 for ind in indicators if ind in source) >= 2 else "triton"


def get_completed_kernels(result_dir: Path) -> set:
    details_dir = result_dir / "details"
    if not details_dir.exists():
        return set()
    return {f.stem for f in details_dir.glob("*.json")}


def load_completed_result(result_dir: Path, key: str):
    path = result_dir / "details" / f"{key}.json"
    if path.exists():
        try:
            return MutationTestResult.load(path)
        except Exception:
            return None
    return None


def gpu_cleanup():
    stale = [k for k in sys.modules
             if k.startswith("mutant_") or k.startswith("ref_")
             or k.startswith("eqchk_") or k.startswith("eqm_")
             or k.startswith("prob_") or k.startswith("ref_w_")
             or k.startswith("ref_eq_")]
    for k in stale:
        del sys.modules[k]
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _normalize_source(source: str) -> str:
    """CUDA-aware normalization delegated to equivalent_detector."""
    from src.mutengine.equivalent_detector import _normalize_source as _ns
    return _ns(source)


def _extract_input_spec(problem_file_path: str) -> str:
    """Extract actual input shapes/dtypes from the reference get_inputs().

    Loads the reference module on CPU (no CUDA), calls get_inputs() once,
    and returns a human-readable description of each tensor argument.
    """
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_input_spec_probe", problem_file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        torch.manual_seed(42)
        inputs = mod.get_inputs()

        parts = []
        for idx, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor):
                parts.append(
                    f"  arg[{idx}]: Tensor shape={list(inp.shape)}, "
                    f"dtype={inp.dtype}")
            else:
                parts.append(
                    f"  arg[{idx}]: {type(inp).__name__} = {repr(inp)}")

        return "\n".join([
            f"forward() expects {len(inputs)} argument(s):",
            *parts,
            "",
            "These shapes are FIXED and NEVER change during testing.",
            "Only the numerical values inside these tensors vary.",
        ])
    except Exception as e:
        return f"(could not extract input spec: {e})"


def _create_llm_caller():
    """Create an OpenAI-compatible LLM caller if API key is available."""
    if not LLM_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)

        is_reasoner = "reasoner" in LLM_MODEL.lower()

        def call_llm(prompt: str) -> str:
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
            if content.strip():
                return content
            return reasoning

        return call_llm
    except ImportError:
        log.warning("openai package not installed, Layer 3 LLM verification disabled")
        return None


# ---------------------------------------------------------------------------
# Subprocess execution helpers
# ---------------------------------------------------------------------------

def _run_worker(cfg: dict, timeout: int) -> dict | None:
    """Run _mutant_worker.py in a subprocess. Returns parsed JSON or None."""
    cfg_fd, cfg_path = tempfile.mkstemp(suffix=".json", prefix="mutcfg_")
    res_fd, res_path = tempfile.mkstemp(suffix=".json", prefix="mutres_")
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
            return None  # caller handles timeout

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


def run_mutant_isolated(mutant: Mutant, kernel: KernelInfo) -> Mutant:
    """Run one mutant in an isolated subprocess."""
    cfg = {
        "mode": "run",
        "problem_id": kernel.problem_id,
        "level": kernel.level,
        "problem_name": kernel.problem_name,
        "source_path": kernel.source_path,
        "kernel_code": kernel.kernel_code,
        "language": kernel.language,
        "problem_file": kernel.reference_module_path,
        "mutant_id": mutant.id,
        "operator_name": mutant.operator_name,
        "operator_category": mutant.operator_category,
        "site": {
            "line_start": mutant.site.line_start,
            "line_end": mutant.site.line_end,
            "col_start": mutant.site.col_start,
            "col_end": mutant.site.col_end,
            "original_code": mutant.site.original_code,
            "node_type": mutant.site.node_type,
        },
        "original_code": mutant.original_code,
        "mutated_code": mutant.mutated_code,
        "description": mutant.description,
        "atol": ATOL,
        "rtol": RTOL,
        "num_test_inputs": NUM_TEST_INPUTS,
        "device": DEVICE,
        "seed": SEED,
    }

    data = _run_worker(cfg, timeout=MUTANT_TIMEOUT)

    if data is None:
        mutant.status = MutantStatus.STILLBORN
        mutant.error_message = f"Subprocess timeout/crash ({MUTANT_TIMEOUT}s)"
        mutant.execution_time_ms = MUTANT_TIMEOUT * 1000
    else:
        mutant.status = MutantStatus(data["status"])
        mutant.execution_time_ms = data.get("time_ms", 0)
        mutant.error_message = data.get("error", "")
        mutant.kill_input_seed = data.get("kill_seed")

    return mutant


def check_equiv_isolated(mutant: Mutant, kernel: KernelInfo) -> dict:
    """Check equivalence for one survived mutant in a subprocess.

    Returns the full worker result dict (not just a bool) so that
    divergence details, seeds, and policies are preserved for
    enhanced testing.
    """
    cfg = {
        "mode": "equiv",
        "problem_file": kernel.reference_module_path,
        "kernel_code": kernel.kernel_code,
        "mutant_id": mutant.id,
        "mutated_code": mutant.mutated_code,
        "operator_name": mutant.operator_name,
        "device": DEVICE,
        "equiv_runs": EQUIV_RUNS,
        "base_seed": 10000,
    }

    data = _run_worker(cfg, timeout=EQUIV_TIMEOUT)

    if data is None:
        return {"is_equivalent": False, "error": "worker_timeout_or_crash"}
    return data


# ---------------------------------------------------------------------------
# Kernel processing
# ---------------------------------------------------------------------------

def _process_one_kernel(runner, reporter, kernel, all_results,
                        llm_caller=None):
    """处理单个 kernel 的完整 Block1+Block2 流程。
    主进程不做任何 CUDA 操作。"""

    key = kernel.problem_name

    # ===== Block 1: generate mutants (pure Python) =====
    t0 = time.time()
    all_mutants = runner.generate_mutants(kernel)
    gen_s = time.time() - t0
    P(f"  [Block1] Generated {len(all_mutants)} mutants ({gen_s:.1f}s)")

    by_op: dict[str, list[Mutant]] = defaultdict(list)
    for m in all_mutants:
        by_op[m.operator_name].append(m)

    by_cat: dict[str, int] = defaultdict(int)
    for m in all_mutants:
        by_cat[m.operator_category] += 1

    for cat in sorted(by_cat):
        ops_in_cat = [
            f"{op}={len(ms)}"
            for op, ms in sorted(by_op.items())
            if ms and ms[0].operator_category == cat
        ]
        P(f"    {cat}: {by_cat[cat]} total  ({', '.join(ops_in_cat)})")

    rng = random.Random(SEED + kernel.problem_id)
    sampled: list[Mutant] = []
    for op_name, muts in by_op.items():
        sampled.extend(rng.sample(muts, min(SAMPLE_PER_OP, len(muts))))
    P(f"  [Sample] {len(sampled)} mutants ({SAMPLE_PER_OP}/op)")

    # ===== Block 2: run mutants (each in subprocess, sequential) =====
    t0 = time.time()
    for i, mutant in enumerate(sampled):
        P(f"  [{i+1}/{len(sampled)}] {mutant.operator_name} "
          f"@ L{mutant.site.line_start}")
        run_mutant_isolated(mutant, kernel)
        tag = mutant.status.value.upper()
        P(f"    → {tag} ({mutant.execution_time_ms:.0f}ms)")
    run_s = time.time() - t0

    result = MutationTestResult(kernel=kernel, mutants=sampled)
    P(f"  [Block2:Runner] {run_s:.1f}s  "
      f"killed={result.killed} survived={result.survived} "
      f"stillborn={result.stillborn}")

    # ===== Equivalence detection (Layer 0-2) =====
    survived = result.survived_mutants()
    if survived:
        P(f"  [Block2:Equiv] Checking {len(survived)} survived mutants...")

        # Layer 1: import static rules (best-effort)
        try:
            from src.mutengine.static_equiv_rules import check_all_rules
        except ImportError:
            check_all_rules = None

        from src.mutengine.equivalent_detector import (
            _extract_cuda_strings, _normalize_cuda_source,
            _normalize_python_source, _analyze_host_diff,
        )

        kname = kernel.problem_name  # e.g. "L1_P1"

        actual_input_spec = _extract_input_spec(
            kernel.reference_module_path)

        for mi, m in enumerate(survived):
            tag = f"[{kname}][{m.operator_name}@L{m.site.line_start}]"
            detail = {
                "mutant_id": m.id,
                "operator": m.operator_name,
                "line_start": m.site.line_start,
                "input_spec": actual_input_spec,
                "kernel_name": kname,
                "problem_file": kernel.reference_module_path,
            }

            lines = [f"  {tag} ── Equiv check ({mi+1}/{len(survived)}) ──"]

            # ── Layer 0: textual equivalence ──
            cuda_orig = _extract_cuda_strings(m.original_code)
            cuda_mut = _extract_cuda_strings(m.mutated_code)
            cuda_eq = (bool(cuda_orig) and bool(cuda_mut) and
                       _normalize_cuda_source(cuda_orig) ==
                       _normalize_cuda_source(cuda_mut))
            py_eq = (_normalize_python_source(m.original_code) ==
                     _normalize_python_source(m.mutated_code))

            import hashlib
            cuda_norm_orig = _normalize_cuda_source(cuda_orig) if cuda_orig else ""
            cuda_norm_mut = _normalize_cuda_source(cuda_mut) if cuda_mut else ""
            py_norm_orig = _normalize_python_source(m.original_code)
            py_norm_mut = _normalize_python_source(m.mutated_code)

            if cuda_eq:
                mutation_domain = "python_host"
            elif py_eq:
                mutation_domain = "cuda_kernel"
            else:
                mutation_domain = "both" if not cuda_eq and not py_eq else "unknown"

            detail["layer0"] = {
                "cuda_strings_equal": cuda_eq,
                "python_host_equal": py_eq,
                "cuda_extracted": bool(cuda_orig) and bool(cuda_mut),
                "mutation_domain": mutation_domain,
                "cuda_norm_hash_orig": hashlib.md5(
                    cuda_norm_orig.encode()).hexdigest()[:16] if cuda_norm_orig else None,
                "cuda_norm_hash_mut": hashlib.md5(
                    cuda_norm_mut.encode()).hexdigest()[:16] if cuda_norm_mut else None,
                "py_norm_hash_orig": hashlib.md5(
                    py_norm_orig.encode()).hexdigest()[:16],
                "py_norm_hash_mut": hashlib.md5(
                    py_norm_mut.encode()).hexdigest()[:16],
                "mutation_site_line": m.site.line_start,
                "original_fragment": m.site.original_code[:200] if m.site.original_code else "",
            }

            if not cuda_eq and cuda_norm_orig and cuda_norm_mut:
                orig_lines = cuda_norm_orig.splitlines()
                mut_lines = cuda_norm_mut.splitlines()
                diff_lines = []
                for li, (ol, ml) in enumerate(
                        zip(orig_lines, mut_lines), start=1):
                    if ol != ml:
                        diff_lines.append({
                            "cuda_line": li,
                            "original": ol.strip()[:200],
                            "mutated": ml.strip()[:200],
                        })
                if len(orig_lines) != len(mut_lines):
                    diff_lines.append({
                        "cuda_line": "length_diff",
                        "original": f"{len(orig_lines)} lines",
                        "mutated": f"{len(mut_lines)} lines",
                    })
                detail["layer0"]["cuda_diff_lines"] = diff_lines[:10]

            if cuda_eq and not py_eq:
                host_diff = _analyze_host_diff(
                    m.mutated_code, m.site.line_start)
                detail["layer0"]["host_diff_analysis"] = host_diff
                lines.append(
                    f"  {tag}   L0: cuda_eq=True, py_eq=False  "
                    f"(mutation at {host_diff['mutation_location']}, "
                    f"var={host_diff.get('mutated_variable', '?')}, "
                    f"used_in_model={host_diff.get('used_in_model', '?')})")
            else:
                lines.append(
                    f"  {tag}   L0: cuda_eq={cuda_eq}, py_eq={py_eq}")

            decided = False
            if cuda_eq and py_eq:
                detail["layer0"]["verdict"] = "STRICT_EQUIVALENT"
                detail["decided_at"] = "layer0"
                m.status = MutantStatus.STRICT_EQUIVALENT
                m.error_message = (
                    "Textually equivalent (full program normalization)")
                m.equiv_detail = detail
                lines.append(
                    f"  {tag}   => STRICT_EQUIVALENT (textual, L0)")
                decided = True

            elif cuda_eq and not py_eq:
                # IMPORTANT: do NOT short-circuit here. CUDA identical
                # but host differs → record evidence, continue to L1/L2
                # for proper static rule check and dynamic verification.
                detail["layer0"]["verdict"] = "cuda_identical_host_differs"
                lines.append(
                    f"  {tag}   L0: CUDA identical, host differs "
                    f"-> continue to L1/L2")

            else:
                detail["layer0"]["verdict"] = "not_equivalent"
                lines.append(f"  {tag}   L0: pass -> continue to L1/L2")

            if not decided:
                # ── Layer 1: static rules ──
                _RULE_DESCRIPTIONS = {
                    "boundary_unreachable": (
                        "threadIdx always < blockDim, so < vs <= on "
                        "blockDim boundary is unreachable"),
                    "dead_write": (
                        "mutated assignment target is overwritten "
                        "before any read"),
                    "mask_noreach": (
                        "mask boundary tightening only affects "
                        "padding threads"),
                    "dead_host_constant": (
                        "module-level constant not used by ModelNew; "
                        "dead code under fixed-shape testing"),
                }
                rule_hit = None
                if check_all_rules is not None:
                    rule_hit = check_all_rules(m)
                rule_detail_info = {}
                if rule_hit == "dead_host_constant":
                    hda = detail.get("layer0", {}).get(
                        "host_diff_analysis", {})
                    rule_detail_info = {
                        "matched_variable": hda.get("mutated_variable"),
                        "mutation_location": hda.get("mutation_location"),
                        "used_in_model": hda.get("used_in_model"),
                        "used_in_get_inputs": hda.get("used_in_get_inputs"),
                        "why_dead": ("variable not referenced in ModelNew; "
                                     "get_inputs() comes from REFERENCE module"),
                    }
                elif rule_hit == "boundary_unreachable":
                    rule_detail_info = {
                        "reason": "threadIdx/blockIdx boundary condition "
                                  "change is unreachable for valid grid dims",
                        "mutation_site_line": m.site.line_start,
                    }
                elif rule_hit:
                    rule_detail_info = {
                        "mutation_site_line": m.site.line_start,
                    }

                detail["layer1"] = {
                    "rules_available": check_all_rules is not None,
                    "rules_checked": ["boundary_unreachable", "dead_write",
                                      "mask_noreach", "dead_host_constant"],
                    "rule_hit": rule_hit,
                    "rule_description": (
                        _RULE_DESCRIPTIONS.get(rule_hit, "")
                        if rule_hit else None),
                    "rule_details": rule_detail_info if rule_hit else None,
                }
                if rule_hit:
                    detail["layer1"]["verdict"] = (
                        f"STRICT_EQUIVALENT ({rule_hit})")
                    detail["decided_at"] = "layer1"
                    m.status = MutantStatus.STRICT_EQUIVALENT
                    m.error_message = f"Static rule: {rule_hit}"
                    m.equiv_detail = detail
                    lines.append(
                        f"  {tag}   L1: rule={rule_hit}")
                    lines.append(
                        f"  {tag}   => STRICT_EQUIVALENT ({rule_hit}, L1)")
                    decided = True
                else:
                    detail["layer1"]["verdict"] = "no_rule_hit"
                    lines.append(f"  {tag}   L1: no rule hit -> pass")

            if not decided:
                # ── Layer 2: statistical equivalence (subprocess) ──
                lines.append(
                    f"  {tag}   L2: running {EQUIV_RUNS} dynamic checks...")
                P("\n".join(lines)); lines = []

                t_l2 = time.time()
                l2_result = check_equiv_isolated(m, kernel)
                l2_ms = (time.time() - t_l2) * 1000
                is_equiv = l2_result.get("is_equivalent", False)

                detail["layer2"] = {
                    "is_equivalent": is_equiv,
                    "equiv_runs": EQUIV_RUNS,
                    "operator_name": m.operator_name,
                    "time_ms": round(l2_ms),
                    "cuda_was_identical": cuda_eq,
                    "tested_random_seeds": l2_result.get(
                        "tested_random_seeds", []),
                    "tested_policies": l2_result.get(
                        "tested_policies", []),
                    "total_rounds": l2_result.get("total_rounds", 0),
                    "first_input_summary": l2_result.get(
                        "first_input_summary"),
                    "last_input_summary": l2_result.get(
                        "last_input_summary"),
                }
                l2_error = l2_result.get("error")
                if l2_error:
                    detail["layer2"]["error"] = l2_error

                if not is_equiv:
                    div = l2_result.get("divergence", {})
                    detail["layer2"]["divergence"] = div

                if is_equiv:
                    detail["layer2"]["verdict"] = "CANDIDATE_EQUIVALENT"
                    detail["decided_at"] = "layer2"
                    m.status = MutantStatus.CANDIDATE_EQUIVALENT
                    n_rounds = l2_result.get("total_rounds", EQUIV_RUNS)
                    evidence_parts = [
                        f"{n_rounds} rounds (random+stress), {l2_ms:.0f}ms"]
                    if cuda_eq:
                        evidence_parts.append("CUDA strings identical")
                    m.error_message = (
                        f"Candidate equivalent ({', '.join(evidence_parts)})")
                    m.equiv_detail = detail
                    lines.append(
                        f"  {tag}   L2: bitwise identical ({l2_ms:.0f}ms)")
                    lines.append(
                        f"  {tag}   => CANDIDATE_EQUIVALENT (L2)")
                else:
                    div = l2_result.get("divergence", {})
                    div_desc = ""
                    if div:
                        rt = div.get("round_type", "?")
                        if rt == "random":
                            div_desc = (f"random round {div.get('round_index')}"
                                        f", seed={div.get('seed')}")
                        else:
                            div_desc = (f"stress {div.get('policy')}"
                                        f"[{div.get('sub_index')}]"
                                        f", seed={div.get('seed')}"
                                        f", {div.get('detail', '')}")
                    detail["layer2"]["verdict"] = "not_equivalent"
                    detail["decided_at"] = "layer2"
                    m.equiv_detail = detail
                    lines.append(
                        f"  {tag}   L2: divergence found ({l2_ms:.0f}ms)"
                        + (f" [{div_desc}]" if div_desc else ""))
                    lines.append(
                        f"  {tag}   => SURVIVED (not equivalent)")

            P("\n".join(lines))

        # ── Layer 3: LLM equiv verification ──
        equiv_mutants = [m for m in survived
                         if m.status in (MutantStatus.STRICT_EQUIVALENT,
                                         MutantStatus.CANDIDATE_EQUIVALENT)]
        if equiv_mutants and llm_caller is not None:
            from src.stress.llm_analyzer import verify_equivalent_with_llm
            P(f"  [{kname}][Layer3:LLM] Verifying {len(equiv_mutants)} "
              f"equiv mutants with {LLM_MODEL}...")
            P(f"  [{kname}][Layer3:LLM] Input spec:\n{actual_input_spec}")
            llm_reverted = 0
            for m in equiv_mutants:
                tag = f"[{kname}][{m.operator_name}@L{m.site.line_start}]"
                P(f"  {tag} LLM reviewing ...")
                t_l3 = time.time()
                llm_result = verify_equivalent_with_llm(
                    mutant_id=m.id,
                    kernel_code=kernel.kernel_code,
                    mutated_code=m.mutated_code,
                    operator_name=m.operator_name,
                    site={
                        "line_start": m.site.line_start,
                        "original_code": m.site.original_code,
                    },
                    equiv_level=m.status.value,
                    equiv_evidence=m.error_message,
                    input_spec=actual_input_spec,
                    call_llm_fn=llm_caller,
                    layer_detail=m.equiv_detail,
                )
                l3_ms = (time.time() - t_l3) * 1000
                verdict = llm_result.get("verdict", "confirmed_equivalent")
                confidence = llm_result.get("confidence", 0.0)
                reasoning = llm_result.get("reasoning", "")

                from src.stress.llm_analyzer import _format_layer_evidence
                layer_evidence_text = _format_layer_evidence(
                    m.equiv_detail) if m.equiv_detail else ""

                l3_detail = {
                    "model": LLM_MODEL,
                    "verdict": verdict,
                    "confidence": confidence,
                    "reason_category": llm_result.get("reason_category", ""),
                    "proof_sketch": llm_result.get("proof_sketch", ""),
                    "reasoning": reasoning,
                    "kill_strategy": llm_result.get("kill_strategy"),
                    "suggested_test": llm_result.get("suggested_test"),
                    "raw_response": llm_result.get("raw", "")[:2000],
                    "time_ms": round(l3_ms),
                    "input_spec": actual_input_spec,
                    "equiv_evidence_sent_to_llm": layer_evidence_text[:3000],
                }

                if verdict == "possibly_killable" and confidence > 0.7:
                    old_status = m.status.value
                    l3_detail["action"] = (
                        f"reverted to SURVIVED (was {old_status})")
                    m.equiv_detail["layer3"] = l3_detail
                    m.equiv_detail["decided_at"] = "layer3"
                    m.status = MutantStatus.SURVIVED
                    m.error_message = (
                        f"LLM rejected equiv (was {old_status}, "
                        f"conf={confidence:.2f}): {reasoning[:200]}")
                    P(f"  {tag}   L3: {verdict} (conf={confidence:.2f})")
                    P(f"  {tag}   L3: {reasoning[:120]}")
                    P(f"  {tag}   => SURVIVED (LLM rejected equiv)")
                    llm_reverted += 1
                else:
                    l3_detail["action"] = "confirmed"
                    m.equiv_detail["layer3"] = l3_detail
                    P(f"  {tag}   L3: {verdict} (conf={confidence:.2f})")
                    P(f"  {tag}   L3: {reasoning[:120]}")
                    P(f"  {tag}   => LLM confirmed equiv")

            if llm_reverted:
                P(f"  [{kname}] LLM rejected {llm_reverted} "
                  f"equiv -> SURVIVED")
        elif equiv_mutants and llm_caller is None:
            P(f"  [{kname}][Layer3:LLM] Skipped (no API key)")
            for m in equiv_mutants:
                m.equiv_detail["layer3"] = {
                    "skipped": True, "reason": "no API key",
                    "input_spec": actual_input_spec,
                }

        n_strict = sum(1 for m in survived
                       if m.status == MutantStatus.STRICT_EQUIVALENT)
        n_cand = sum(1 for m in survived
                     if m.status == MutantStatus.CANDIDATE_EQUIVALENT)
        n_llm_rejected = sum(
            1 for m in survived
            if (m.status == MutantStatus.SURVIVED
                and m.equiv_detail.get("layer3", {}).get("action", "")
                .startswith("reverted")))
        n_truly = sum(1 for m in survived
                      if m.status == MutantStatus.SURVIVED)
        P(f"  [{kname}] Equiv summary: "
          f"strict_eq={n_strict}, candidate_eq={n_cand}, "
          f"llm_rejected={n_llm_rejected}, survived={n_truly}")
    else:
        P(f"  [Block2:Equiv] No survived mutants")

    # ===== Save =====
    reporter.save_kernel_result(result)
    all_results.append(result)

    cons_score = result.mutation_score
    opt_score = result.mutation_score_optimistic
    P(f"  >> Score (conservative): {cons_score:.2%}  "
      f"(killed={result.killed}, survived={result.survived}, "
      f"stillborn={result.stillborn}, "
      f"strict_eq={result.strict_equivalent}, "
      f"cand_eq={result.candidate_equivalent})")
    P(f"  >> Score (optimistic):   {opt_score:.2%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    t_total = time.time()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_LOG.exists():
        OUTPUT_LOG.unlink()

    with open(BEST_KERNELS_FILE) as f:
        best_kernels = json.load(f)

    all_keys = sorted(best_kernels.keys())
    n_l1 = sum(1 for k in all_keys if k.startswith("L1"))
    n_l2 = sum(1 for k in all_keys if k.startswith("L2"))

    P(f"\n{'='*70}")
    P(f"  MutaKernel Block1+Block2 V2 (四层等价检测)")
    P(f"{'='*70}")
    P(f"  GPU            : {torch.cuda.get_device_name(0)}")
    P(f"  Kernels        : {len(all_keys)} (L1={n_l1}, L2={n_l2})")
    P(f"  Sample/op      : {SAMPLE_PER_OP}")
    P(f"  Test inputs    : {NUM_TEST_INPUTS}")
    P(f"  Equiv runs     : {EQUIV_RUNS}")
    P(f"  atol/rtol      : {ATOL} / {RTOL}")
    P(f"  Seed           : {SEED}")
    P(f"  Mutant timeout : {MUTANT_TIMEOUT}s")
    P(f"  Equiv timeout  : {EQUIV_TIMEOUT}s")
    P(f"  Equiv pipeline : Layer0(CUDA-norm) → Layer1(static) → Layer2(dynamic) → Layer3(LLM)")
    P(f"  LLM verify     : {'enabled (' + LLM_MODEL + ')' if LLM_EQUIV_VERIFY else 'disabled (no API key)'}")
    P(f"  Max kernels    : {MAX_KERNELS if MAX_KERNELS > 0 else 'unlimited'}")
    P(f"  Output         : {RESULT_DIR}")
    P("")

    llm_caller = _create_llm_caller()

    runner = MutantRunner(
        atol=ATOL, rtol=RTOL,
        num_test_inputs=NUM_TEST_INPUTS,
        device=DEVICE, seed=SEED,
        categories=["A", "B", "C", "D"],
    )
    reporter = MutationReporter(output_dir=RESULT_DIR)

    completed = get_completed_kernels(RESULT_DIR)
    if completed:
        P(f"  Resuming: {len(completed)} kernels already completed")
        P("")

    all_results: list[MutationTestResult] = []
    kernels_done = 0
    kernels_failed = 0

    if MAX_KERNELS > 0:
        P(f"  *** DEBUG: limited to first {MAX_KERNELS} kernels ***")
        P("")

    for ki, key in enumerate(all_keys):
        if MAX_KERNELS > 0 and kernels_done >= MAX_KERNELS:
            P(f"\n  *** Reached MAX_KERNELS={MAX_KERNELS}, stopping. ***")
            break
        # ---- resume: load previously saved results ----
        if key in completed:
            prev = load_completed_result(RESULT_DIR, key)
            if prev:
                all_results.append(prev)
                kernels_done += 1
            continue

        info = best_kernels[key]
        level_str = info["level"]
        pid = info["problem_id"]
        kpath = Path(info["kernel_path"])
        speedup = info.get("speedup", 0)
        turn = info.get("turn", "?")

        level_int = (int(level_str[1])
                     if isinstance(level_str, str) else int(level_str))
        problem_dir = PROBLEM_DIRS.get(level_str)
        if not problem_dir:
            P(f"  [{ki+1}/{len(all_keys)}] {key}: unknown level {level_str}")
            kernels_failed += 1
            continue

        pfile = find_problem_file(problem_dir, pid)
        if pfile is None:
            P(f"  [{ki+1}/{len(all_keys)}] {key}: problem file not found")
            kernels_failed += 1
            continue

        if not kpath.exists():
            P(f"  [{ki+1}/{len(all_keys)}] {key}: kernel file missing")
            kernels_failed += 1
            continue

        source = kpath.read_text(encoding="utf-8", errors="replace")

        kernel = KernelInfo(
            problem_id=int(pid),
            level=level_int,
            problem_name=key,
            source_path=str(kpath),
            kernel_code=source,
            reference_module_path=str(pfile),
            language=detect_language(source),
        )

        P(f"\n{'─'*70}")
        P(f"  [{ki+1}/{len(all_keys)}] {key}  lang={kernel.language}  "
          f"turn={turn}  speedup={speedup:.3f}")
        P(f"{'─'*70}")

        # ---- retry loop: never give up ----
        success = False
        for attempt in range(1, MAX_KERNEL_RETRIES + 1):
            try:
                _process_one_kernel(runner, reporter, kernel, all_results,
                                    llm_caller=llm_caller)
                kernels_done += 1
                success = True
                break
            except Exception as e:
                P(f"  [RETRY {attempt}/{MAX_KERNEL_RETRIES}] {key}: "
                  f"{str(e)[:150]}")
                gpu_cleanup()
                time.sleep(2)

        if not success:
            P(f"  [FAIL] {key}: all {MAX_KERNEL_RETRIES} attempts failed, "
              f"saving empty result")
            empty = MutationTestResult(kernel=kernel, mutants=[])
            reporter.save_kernel_result(empty)
            all_results.append(empty)
            kernels_failed += 1

        # ---- progress ----
        gpu_cleanup()
        elapsed = time.time() - t_total
        new_done = kernels_done + kernels_failed - len(completed)
        avg = elapsed / max(1, new_done) if new_done > 0 else 300
        remaining = len(all_keys) - ki - 1
        eta = remaining * avg / 60
        P(f"  [Progress] done={kernels_done} fail={kernels_failed} "
          f"/ {len(all_keys)},  elapsed={elapsed/60:.1f}min, "
          f"ETA~{eta:.0f}min")

    # ===== Final summary =====
    total_elapsed = time.time() - t_total

    P(f"\n{'='*70}")
    P(f"  FINAL SUMMARY")
    P(f"{'='*70}")

    summary_path = reporter.save_summary(all_results)
    summary = reporter.generate_summary(all_results)

    P(f"  Kernels tested         : {summary['total_kernels']}")
    P(f"  Kernels failed         : {kernels_failed}")
    P(f"  Total mutants          : {summary['total_mutants']}")
    P(f"  Killed                 : {summary['total_killed']}")
    P(f"  Survived               : {summary['total_survived']}")
    P(f"  Stillborn              : {summary['total_stillborn']}")
    P(f"  Strict Equivalent      : {summary['total_strict_equivalent']}")
    P(f"  Candidate Equivalent   : {summary['total_candidate_equivalent']}")
    P(f"  Conservative Score     : {summary['overall_mutation_score']:.2%}")
    P(f"  Optimistic Score       : {summary['overall_mutation_score_optimistic']:.2%}")

    P(f"\n  By Category:")
    P(f"  {'Cat':<5} {'Name':<30} {'Kill':>5} {'Surv':>5} "
      f"{'SB':>5} {'S-Eq':>5} {'C-Eq':>5} {'Score':>8}")
    P(f"  {'-'*73}")
    for cat in sorted(summary.get("by_category", {})):
        cs = summary["by_category"][cat]
        P(f"  {cat:<5} {cs['name']:<30} {cs['killed']:>5} "
          f"{cs['survived']:>5} {cs['stillborn']:>5} "
          f"{cs['strict_equivalent']:>5} {cs['candidate_equivalent']:>5} "
          f"{cs['score']:>7.1%}")

    P(f"\n  By Operator:")
    P(f"  {'Operator':<26} {'Kill':>5} {'Surv':>5} "
      f"{'SB':>5} {'S-Eq':>5} {'C-Eq':>5} {'Score':>8}")
    P(f"  {'-'*65}")
    for op in sorted(summary.get("by_operator", {})):
        os_ = summary["by_operator"][op]
        P(f"  {op:<26} {os_['killed']:>5} {os_['survived']:>5} "
          f"{os_['stillborn']:>5} {os_['strict_equivalent']:>5} "
          f"{os_['candidate_equivalent']:>5} {os_['score']:>7.1%}")

    P(f"\n  By Level:")
    for level in ["L1", "L2"]:
        level_results = [r for r in all_results
                         if r.kernel.level == int(level[1])]
        if not level_results:
            continue
        k = sum(r.killed for r in level_results)
        sv = sum(r.survived for r in level_results)
        sb = sum(r.stillborn for r in level_results)
        se = sum(r.strict_equivalent for r in level_results)
        ce = sum(r.candidate_equivalent for r in level_results)
        denom = k + sv
        score = k / denom if denom > 0 else 0
        P(f"  {level}: kernels={len(level_results)}, killed={k}, "
          f"survived={sv}, stillborn={sb}, "
          f"strict_eq={se}, cand_eq={ce}, score={score:.1%}")

    P(f"\n  Time: {total_elapsed/60:.1f}min ({total_elapsed/3600:.1f}h)")
    P(f"  Results: {RESULT_DIR}")
    P(f"  Summary: {summary_path}")
    P(f"\n  DONE.")

    runner.cleanup()


if __name__ == "__main__":
    main()
