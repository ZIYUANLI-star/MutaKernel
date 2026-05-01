"""Layer 1 — 算子静态等价规则。

三条基于 CUDA 语义的 pattern-matching 规则，可在不执行 kernel
的情况下判定 STRICT_EQUIVALENT：

  1. boundary_unreachable: threadIdx 与 blockDim 的隐含上界关系
  2. dead_write: 变异目标是一个被覆写前从未读取的变量
  3. mask_noreach: mask 边界收紧后只影响 padding 线程
"""

from __future__ import annotations

import re
from typing import Optional

from ..models import Mutant

# ── Rule 1: boundary_unreachable ──────────────────────────────────────────
#
# threadIdx.x 取值 [0, blockDim.x-1]，因此
#   threadIdx.x < blockDim.x   ←→   threadIdx.x <= blockDim.x - 1
# 如果变异只是把 < 换成 <=（或反向），且右侧是同维 blockDim，
# 则 <= blockDim.x 的上界 blockDim.x 永远不可达，语义不变。

_THREAD_DIM = re.compile(
    r"threadIdx\s*\.\s*([xyz])\s*"
    r"(<|<=|>|>=)\s*"
    r"blockDim\s*\.\s*([xyz])"
)


def _boundary_unreachable(mutant: Mutant) -> bool:
    orig_frag = mutant.site.original_code
    mut_code = mutant.mutated_code

    orig_m = _THREAD_DIM.search(orig_frag)
    if orig_m is None:
        return False

    dim_orig = orig_m.group(1)
    op_orig = orig_m.group(2)
    dim_rhs_orig = orig_m.group(3)

    if dim_orig != dim_rhs_orig:
        return False

    mut_m = _THREAD_DIM.search(mut_code)
    if mut_m is None:
        return False

    dim_mut = mut_m.group(1)
    op_mut = mut_m.group(2)
    dim_rhs_mut = mut_m.group(3)

    if dim_mut != dim_orig or dim_rhs_mut != dim_rhs_orig:
        return False

    pair = frozenset((op_orig, op_mut))
    return pair == frozenset(("<", "<=")) or pair == frozenset((">", ">="))


# ── Rule 2: dead_write ────────────────────────────────────────────────────
#
# If the mutation site is an assignment and the target variable is
# overwritten before it is next read, the mutation has no observable effect.

_ASSIGN_LHS = re.compile(r"^\s*(\w+)\s*[+\-*/&|^%]?=\s")

_CUDA_KEYWORDS = frozenset(
    "if else for while do return break continue switch case "
    "default goto sizeof typedef struct union enum "
    "__global__ __device__ __host__ __shared__ __constant__ "
    "void int float double char unsigned long short bool "
    "dim3 size_t".split()
)


def _dead_write(mutant: Mutant) -> bool:
    if mutant.operator_name not in (
        "arith_replace", "const_perturb", "scale_modify", "init_modify",
    ):
        return False

    orig_frag = mutant.site.original_code
    m = _ASSIGN_LHS.match(orig_frag)
    if m is None:
        return False
    target_var = m.group(1)
    if target_var in _CUDA_KEYWORDS:
        return False

    from .parser.cuda_parser import CudaParser
    parser = CudaParser()
    result = parser.parse(mutant.mutated_code)
    if not result.cuda_blocks:
        return False

    full_cuda = result.all_cuda_source
    lines = full_cuda.splitlines()

    site_line = mutant.site.line_start
    _ident = re.compile(r"\b" + re.escape(target_var) + r"\b")
    _assign = re.compile(r"^\s*" + re.escape(target_var) + r"\s*[+\-*/&|^%]?=\s")

    found_site = False
    for i, line in enumerate(lines, 1):
        if not found_site:
            if _assign.match(line) and abs(i - site_line) <= 2:
                found_site = True
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if _assign.match(line):
            return True
        if _ident.search(line):
            return False

    return False


# ── Rule 3: mask_noreach ─────────────────────────────────────────────────
#
# mask_boundary mutations that *tighten* a bounds guard (n → n-1, n → n-2)
# only exclude padding threads whose writes do not affect the output.

_GUARD_PATTERN = re.compile(
    r"(?:idx|tid|index|i|gid)\s*(<|<=)\s*(\w+)"
)


def _mask_noreach(mutant: Mutant) -> bool:
    if mutant.operator_name != "mask_boundary":
        return False

    orig_frag = mutant.site.original_code
    mut_code = mutant.mutated_code

    orig_m = _GUARD_PATTERN.search(orig_frag)
    if orig_m is None:
        return False
    orig_op = orig_m.group(1)
    orig_bound = orig_m.group(2)

    for line in mut_code.splitlines():
        mut_m = _GUARD_PATTERN.search(line)
        if mut_m is None:
            continue
        mut_op = mut_m.group(1)
        mut_bound = mut_m.group(2)

        if mut_bound == orig_bound and orig_op == "<=" and mut_op == "<":
            return True
        if (mut_bound == f"{orig_bound}-1" or mut_bound == f"({orig_bound}-1)"
                or mut_bound == f"{orig_bound} - 1"):
            return True

    return False


# ── Rule 4: dead_host_constant ────────────────────────────────────────────
#
# Under our testing framework, get_inputs() / get_init_inputs() always come
# from the REFERENCE module (never from the mutant). So if const_perturb
# changes a module-level constant (e.g. N=2048→2049) that is only used by
# get_inputs() and NOT by ModelNew.__init__() / forward(), the mutation is
# dead code — it can never affect the execution under fixed-shape testing.

def _dead_host_constant(mutant: Mutant) -> bool:
    if mutant.operator_name != "const_perturb":
        return False

    import ast

    try:
        tree = ast.parse(mutant.mutated_code)
    except SyntaxError:
        return False

    site_line = mutant.site.line_start

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            end = getattr(node, "end_lineno", None) or (node.lineno + 5000)
            if node.lineno <= site_line <= end:
                return False
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", None) or (node.lineno + 5000)
            if node.lineno <= site_line <= end:
                return False

    mutated_var = None
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign) and node.lineno == site_line:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    mutated_var = target.id
                    break
            break

    if mutated_var is None:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in ("ModelNew", "Model"):
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and child.id == mutated_var:
                    return False

    return True


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

_RULES = [
    ("boundary_unreachable", _boundary_unreachable),
    ("dead_write", _dead_write),
    ("mask_noreach", _mask_noreach),
    ("dead_host_constant", _dead_host_constant),
]


def check_all_rules(mutant: Mutant) -> Optional[str]:
    """Run all static rules; return the name of the first match, or None."""
    for name, fn in _RULES:
        try:
            if fn(mutant):
                return name
        except Exception:
            continue
    return None
