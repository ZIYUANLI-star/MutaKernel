"""等价变异体检测器 (V2 — 四层流水线)。

Layer 0  文本归一化: CUDA-aware 源码归一化后比较 → STRICT_EQUIVALENT
Layer 1  算子静态规则: boundary_unreachable / dead_write / mask_noreach → STRICT_EQUIVALENT
Layer 2  动态 bitwise: N 次随机 + 算子定向压力策略 → CANDIDATE_EQUIVALENT
Layer 3  LLM 验证:  对判为等价的变异体做二次审查 (外部调用, 此模块不实现)

注意: 统计检测比较的是 original_kernel vs mutant_kernel（两个 GPU kernel），
而非 reference(PyTorch) vs mutant，因为 GPU kernel 与 PyTorch 参考实现之间
本身存在浮点差异，用 ref vs mutant 做 bitwise 比较会产生大量漏检。
"""

from __future__ import annotations

import logging
import re
import time
from enum import Enum
from typing import Any, List, Optional, Tuple

import torch

from ..models import Mutant, MutantStatus
from ..stress.policy_bank import STRESS_POLICIES

logger = logging.getLogger(__name__)

# ── Default stress policies (fallback when no operator-directed set) ──────
EQUIV_STRESS_POLICIES = [
    "large_magnitude", "near_zero", "structured_ramp",
    "all_negative", "sparse", "boundary_last_element",
]

# ── Operator-directed policy selection ────────────────────────────────────
OPERATOR_DIRECTED_POLICIES = {
    "relop_replace": ["relop_boundary_hit", "boundary_last_element", "structured_ramp",
                       "near_zero", "sparse", "large_magnitude"],
    "arith_replace": ["extreme_magnitude", "large_magnitude", "near_zero",
                       "all_negative", "sparse", "boundary_last_element"],
    "epsilon_modify": ["near_epsilon", "near_zero", "denormals",
                        "large_magnitude", "sparse", "boundary_last_element"],
    "mask_boundary": ["boundary_last_element", "structured_ramp", "head_heavy",
                       "tail_heavy", "sparse", "large_magnitude"],
    "index_replace": ["head_heavy", "tail_heavy", "structured_ramp",
                       "large_magnitude", "sparse", "boundary_last_element"],
    "stab_remove": ["extreme_magnitude", "large_magnitude", "all_positive",
                     "near_zero", "sparse", "boundary_last_element"],
    "scale_modify": ["extreme_magnitude", "large_magnitude", "near_zero",
                      "all_negative", "sparse", "boundary_last_element"],
    "acc_downgrade": ["reduction_adversarial", "large_magnitude", "mixed_extremes",
                       "alternating_sign", "sparse", "boundary_last_element"],
    "reduction_reorder": ["reduction_adversarial", "mixed_extremes", "alternating_sign",
                           "large_magnitude", "sparse", "boundary_last_element"],
    "init_modify": ["init_sensitive", "all_negative", "all_positive",
                     "sparse", "near_zero", "boundary_last_element"],
    "cast_remove": ["extreme_magnitude", "near_zero", "mixed_extremes",
                     "large_magnitude", "sparse", "boundary_last_element"],
    "sync_remove": ["structured_ramp", "head_heavy", "tail_heavy",
                     "large_magnitude", "sparse", "boundary_last_element"],
    "const_perturb": ["near_zero", "boundary_last_element", "sparse",
                       "large_magnitude", "structured_ramp", "all_negative"],
    "launch_config_mutate": ["structured_ramp", "head_heavy", "tail_heavy",
                              "large_magnitude", "sparse", "boundary_last_element"],
    "broadcast_unsafe": ["large_magnitude", "sparse", "mixed_extremes",
                          "near_zero", "structured_ramp", "boundary_last_element"],
    "layout_assume": ["structured_ramp", "alternating_sign", "large_magnitude",
                       "near_zero", "sparse", "boundary_last_element"],
}

_INFRA_EXCEPTIONS = (MemoryError, RuntimeError)


# ═══════════════════════════════════════════════════════════════════════════
# CompareResult — structured comparison outcome
# ═══════════════════════════════════════════════════════════════════════════

class CompareResult(Enum):
    SAME_OUTPUT = "same_output"
    DIFFERENT_OUTPUT = "different_output"
    SAME_EXCEPTION = "same_exception"
    DIFFERENT_EXCEPTION = "different_exception"
    ONE_SIDE_EXCEPTION = "one_side_exception"
    INFRA_ERROR = "infra_error"


def _is_infra_error(exc: BaseException) -> bool:
    """CUDA OOM or similar infrastructure failure."""
    msg = str(exc).lower()
    return ("out of memory" in msg or "cuda" in msg and "error" in msg
            or isinstance(exc, MemoryError))


# ═══════════════════════════════════════════════════════════════════════════
# Bitwise comparison (NaN-aware)
# ═══════════════════════════════════════════════════════════════════════════

def _bitwise_identical(a: Any, b: Any) -> bool:
    """NaN-aware bitwise comparison of two outputs."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.is_floating_point():
            nan_a = torch.isnan(a)
            nan_b = torch.isnan(b)
            if not torch.equal(nan_a, nan_b):
                return False
            finite = ~nan_a
            if finite.any():
                return torch.equal(a[finite], b[finite])
            return True
        return torch.equal(a, b)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            return False
        return all(_bitwise_identical(x, y) for x, y in zip(a, b))
    return a == b


# ═══════════════════════════════════════════════════════════════════════════
# Source normalisation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_python_source(source: str) -> str:
    """Python-level: strip blanks, comments, leading/trailing whitespace."""
    lines = source.splitlines()
    normalized = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        normalized.append(stripped)
    return "\n".join(normalized)


def _normalize_cuda_source(cuda_str: str) -> str:
    """C++-level: strip block/line comments, collapse whitespace."""
    text = re.sub(r"/\*.*?\*/", " ", cuda_str, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(re.sub(r"\s+", " ", stripped))
    return "\n".join(lines)


def _extract_cuda_strings(source: str) -> str:
    """Extract concatenated CUDA source strings from a Python file."""
    try:
        from ..mutengine.parser.cuda_parser import CudaParser
        parser = CudaParser()
        result = parser.parse(source)
        if result.is_cuda and result.cuda_blocks:
            return result.all_cuda_source
    except Exception:
        pass
    return ""


def _normalize_source(source: str) -> str:
    """CUDA-aware normalization: prefer CUDA string extraction, fallback Python."""
    cuda = _extract_cuda_strings(source)
    if cuda:
        return _normalize_cuda_source(cuda)
    return _normalize_python_source(source)


def _analyze_host_diff(mutated_code: str, site_line: int) -> dict:
    """Analyze where a host-side mutation falls and whether it affects ModelNew.

    Returns a dict with:
      mutation_location: "module_level" | "inside class XXX" | "inside function XXX"
      mutated_variable:  name of the variable being assigned, or None
      used_in_model:     True/False/None  (is the variable referenced in ModelNew?)
      used_in_get_inputs: True/False/None
    """
    import ast

    info: dict = {
        "mutation_location": "unknown",
        "mutated_variable": None,
        "used_in_model": None,
        "used_in_get_inputs": None,
    }

    try:
        tree = ast.parse(mutated_code)
    except SyntaxError:
        return info

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            end = getattr(node, "end_lineno", None) or (node.lineno + 5000)
            if node.lineno <= site_line <= end:
                info["mutation_location"] = f"inside class {node.name}"
                return info
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", None) or (node.lineno + 5000)
            if node.lineno <= site_line <= end:
                parent_cls = None
                for n2 in ast.walk(tree):
                    if isinstance(n2, ast.ClassDef):
                        for child in ast.iter_child_nodes(n2):
                            if child is node:
                                parent_cls = n2.name
                if parent_cls:
                    info["mutation_location"] = (
                        f"inside {parent_cls}.{node.name}()")
                else:
                    info["mutation_location"] = (
                        f"inside function {node.name}()")
                return info

    info["mutation_location"] = "module_level"

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign) and node.lineno == site_line:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    info["mutated_variable"] = target.id
                    break
            break

    var = info["mutated_variable"]
    if var is None:
        return info

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in ("ModelNew", "Model"):
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and child.id == var:
                    info["used_in_model"] = True
                    break
            if info["used_in_model"]:
                break
    if info["used_in_model"] is None:
        info["used_in_model"] = False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
                "get_inputs", "get_init_inputs"):
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and child.id == var:
                    info["used_in_get_inputs"] = True
                    break
            if info["used_in_get_inputs"]:
                break
    if info["used_in_get_inputs"] is None:
        info["used_in_get_inputs"] = False

    return info


# ═══════════════════════════════════════════════════════════════════════════
# Structured comparison helper
# ═══════════════════════════════════════════════════════════════════════════

def _run_and_compare(
    run_orig_fn: Any,
    run_mut_fn: Any,
    inputs: Any,
) -> CompareResult:
    """Run both sides on *inputs* and return a structured CompareResult."""
    orig_exc: Optional[BaseException] = None
    mut_exc: Optional[BaseException] = None
    orig_out = mut_out = None

    try:
        orig_out = run_orig_fn(inputs)
    except Exception as e:
        orig_exc = e

    try:
        mut_out = run_mut_fn(inputs)
    except Exception as e:
        mut_exc = e

    if orig_exc is not None and mut_exc is not None:
        if _is_infra_error(orig_exc) or _is_infra_error(mut_exc):
            return CompareResult.INFRA_ERROR
        if type(orig_exc) is type(mut_exc):
            return CompareResult.SAME_EXCEPTION
        return CompareResult.DIFFERENT_EXCEPTION

    if orig_exc is not None or mut_exc is not None:
        if (orig_exc and _is_infra_error(orig_exc)) or (mut_exc and _is_infra_error(mut_exc)):
            return CompareResult.INFRA_ERROR
        return CompareResult.ONE_SIDE_EXCEPTION

    if _bitwise_identical(orig_out, mut_out):
        return CompareResult.SAME_OUTPUT
    return CompareResult.DIFFERENT_OUTPUT


# ═══════════════════════════════════════════════════════════════════════════
# Main detector class
# ═══════════════════════════════════════════════════════════════════════════

class EquivalentDetector:
    """Four-layer equivalent mutant detector (V2).

    Layer 0: textual equivalence (CUDA-aware normalization)
    Layer 1: static rules (operator-specific, imported from static_equiv_rules)
    Layer 2: statistical equivalence (bitwise, random + directed stress)
    Layer 3: LLM verification (external — not called here)
    """

    def __init__(
        self,
        num_runs: int = 100,
        device: str = "cuda",
        base_seed: int = 10000,
    ):
        self.num_runs = num_runs
        self.device = device
        self.base_seed = base_seed

    # ── Layer 0 ───────────────────────────────────────────────────────────

    def check_textual_equivalence(self, mutant: Mutant) -> str:
        """Layer 0: CUDA-aware source normalization comparison.

        Returns:
            "strict"   – CUDA strings identical AND Python host identical
                         (true textual equivalence of the whole program)
            "cuda_only" – CUDA strings identical but host code differs
                         (mutation is in host code; CUDA kernel unchanged;
                          NOT sufficient for STRICT — see TCE literature)
            ""         – CUDA strings differ (not textually equivalent)
        """
        cuda_orig = _extract_cuda_strings(mutant.original_code)
        cuda_mut = _extract_cuda_strings(mutant.mutated_code)

        if cuda_orig and cuda_mut:
            cuda_eq = _normalize_cuda_source(cuda_orig) == _normalize_cuda_source(cuda_mut)
        else:
            cuda_eq = False

        if cuda_eq:
            py_orig = _normalize_python_source(mutant.original_code)
            py_mut = _normalize_python_source(mutant.mutated_code)
            if py_orig == py_mut:
                return "strict"
            return "cuda_only"

        py_orig = _normalize_python_source(mutant.original_code)
        py_mut = _normalize_python_source(mutant.mutated_code)
        if py_orig == py_mut:
            return "strict"

        return ""

    # ── Layer 1 ───────────────────────────────────────────────────────────

    def check_static_rules(self, mutant: Mutant) -> Optional[str]:
        """Layer 1: operator-specific static rules.

        Returns the rule name that matched, or None.
        """
        try:
            from .static_equiv_rules import check_all_rules
            return check_all_rules(mutant)
        except ImportError:
            return None

    # ── Layer 2 ───────────────────────────────────────────────────────────

    def _get_stress_policies(self, operator_name: str) -> List[str]:
        """Select stress policies based on mutation operator."""
        return OPERATOR_DIRECTED_POLICIES.get(operator_name, EQUIV_STRESS_POLICIES)

    def check_statistical_equivalence(
        self,
        mutant: Mutant,
        run_original_fn: Any,
        run_mutant_fn: Any,
        get_inputs_fn: Any,
    ) -> Tuple[bool, str]:
        """Layer 2: random + stress bitwise comparison.

        Returns (is_candidate_equiv, detail_message).
        """
        infra_errors = 0
        policies = self._get_stress_policies(mutant.operator_name)

        # --- random phase ---
        for i in range(self.num_runs):
            seed = self.base_seed + i
            try:
                torch.manual_seed(seed)
                inputs_orig = get_inputs_fn()
                torch.manual_seed(seed)
                inputs_mut = get_inputs_fn()

                orig_out = run_original_fn(inputs_orig)
                mut_out = run_mutant_fn(inputs_mut)

                if not _bitwise_identical(orig_out, mut_out):
                    return False, f"diverged at random trial {i}"
            except Exception as e:
                if _is_infra_error(e):
                    infra_errors += 1
                    if infra_errors > 3:
                        return False, f"too many infra errors ({infra_errors})"
                    continue
                return False, f"exception at random trial {i}: {e}"

        # --- stress phase (operator-directed) ---
        for policy_name in policies:
            policy_fn = STRESS_POLICIES.get(policy_name)
            if policy_fn is None:
                continue
            for si in range(2):
                seed = self.base_seed + self.num_runs + si
                torch.manual_seed(seed)

                try:
                    template = get_inputs_fn()
                    stress_inputs = policy_fn(template, seed)
                except Exception:
                    continue

                result = _run_and_compare(run_original_fn, run_mutant_fn, stress_inputs)

                if result == CompareResult.DIFFERENT_OUTPUT:
                    return False, f"diverged at stress {policy_name}[{si}]"
                if result == CompareResult.ONE_SIDE_EXCEPTION:
                    return False, f"one-side exception at stress {policy_name}[{si}]"
                if result == CompareResult.DIFFERENT_EXCEPTION:
                    return False, f"different exception at stress {policy_name}[{si}]"
                if result == CompareResult.INFRA_ERROR:
                    infra_errors += 1

        n_stress = len(policies) * 2
        detail = (f"{self.num_runs} random + {n_stress} stress "
                  f"({','.join(policies[:3])}...), bitwise identical")
        return True, detail

    # ── classify (main entry) ─────────────────────────────────────────────

    def classify_survived_mutants(
        self,
        mutants: List[Mutant],
        run_original_fn: Any,
        run_mutant_fns: dict,
        get_inputs_fn: Any,
    ) -> List[Mutant]:
        """Run the four-layer pipeline on survived mutants (Layer 3 is external)."""
        survived = [m for m in mutants if m.status == MutantStatus.SURVIVED]

        for m in survived:
            # Layer 0 — textual equivalence
            l0_result = self.check_textual_equivalence(m)
            if l0_result == "strict":
                m.status = MutantStatus.STRICT_EQUIVALENT
                m.error_message = "Textually equivalent (full program normalization)"
                logger.info(f"  {m.id}: STRICT_EQUIVALENT (textual)")
                continue
            if l0_result == "cuda_only":
                m.status = MutantStatus.CANDIDATE_EQUIVALENT
                m.error_message = (
                    "CUDA kernel identical but host code differs "
                    "(host-side mutation, not sufficient for strict)"
                )
                logger.info(f"  {m.id}: CANDIDATE_EQUIVALENT (cuda_only, host differs)")
                continue

            # Layer 1 — static rules
            rule_hit = self.check_static_rules(m)
            if rule_hit:
                m.status = MutantStatus.STRICT_EQUIVALENT
                m.error_message = f"Static rule: {rule_hit}"
                logger.info(f"  {m.id}: STRICT_EQUIVALENT (rule: {rule_hit})")
                continue

            # Layer 2 — statistical equivalence
            run_mut = run_mutant_fns.get(m.id)
            if run_mut is None:
                continue

            start = time.time()
            is_equiv, detail = self.check_statistical_equivalence(
                m, run_original_fn, run_mut, get_inputs_fn
            )
            elapsed = (time.time() - start) * 1000

            if is_equiv:
                m.status = MutantStatus.CANDIDATE_EQUIVALENT
                m.error_message = f"Candidate equivalent ({detail}, {elapsed:.0f}ms)"
                logger.info(f"  {m.id}: CANDIDATE_EQUIVALENT ({elapsed:.0f}ms)")
            else:
                logger.info(f"  {m.id}: truly survived — {detail} ({elapsed:.0f}ms)")

        return mutants
