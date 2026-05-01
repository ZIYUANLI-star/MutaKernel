"""Category B: GPU parallel semantics mutation operators (Triton / CUDA)."""

from __future__ import annotations

import ast
import re
from typing import List, Optional

from ...models import MutationSite
from .base import MutationOperator, _find_pattern_sites, _replace_in_source


def _is_in_comment_or_string(line: str, col: int) -> bool:
    i = 0
    n = len(line)
    in_single = False
    in_double = False
    in_triple_single = False
    in_triple_double = False
    escape = False

    def at_triple_quote(q: str, start: int) -> bool:
        return start + 2 < n and line[start : start + 3] == q * 3

    while i < col and i < n:
        ch = line[i]
        if not (in_single or in_double or in_triple_single or in_triple_double):
            if ch == "#":
                return True
            if at_triple_quote("'", i):
                in_triple_single = True
                i += 3
                continue
            if at_triple_quote('"', i):
                in_triple_double = True
                i += 3
                continue
            if ch == "'":
                in_single = True
                i += 1
                continue
            if ch == '"':
                in_double = True
                i += 1
                continue
            i += 1
            continue

        if in_triple_single:
            if at_triple_quote("'", i):
                in_triple_single = False
                i += 3
            else:
                i += 1
            continue
        if in_triple_double:
            if at_triple_quote('"', i):
                in_triple_double = False
                i += 3
            else:
                i += 1
            continue

        if escape:
            escape = False
            i += 1
            continue
        if ch == "\\" and (in_single or in_double):
            escape = True
            i += 1
            continue
        if in_single and ch == "'":
            in_single = False
        elif in_double and ch == '"':
            in_double = False
        i += 1

    return in_single or in_double or in_triple_single or in_triple_double


_TRITON_PROG = re.compile(r"tl\.program_id\s*\(\s*([012])\s*\)")
_CUDA_DIM = re.compile(
    r"\b(threadIdx|blockIdx|blockDim|gridDim)\s*\.\s*(x|y|z)\b"
)
_COMP_LT = re.compile(r"\b([A-Za-z_]\w*)\s*<\s*([A-Za-z_]\w*)\b")
_CDIV = re.compile(r"triton\.cdiv\s*\(([^()]*)\)")
_FLOORDIV = re.compile(
    r"\b(?:[A-Za-z_]\w*|\d(?:_?\d)*)\s*//\s*(?:[A-Za-z_]\w*|\d(?:_?\d)*)\b"
)


_COMP_GE = re.compile(r"\b([A-Za-z_]\w*)\s*>=\s*([A-Za-z_]\w*)\b")

# C++ cdiv helper and ceiling division idiom used in CUDA grid sizing
_CUDA_CDIV = re.compile(r"\bcdiv\s*\([^()]*\)")
_CUDA_CEIL_DIV = re.compile(r"\([^()]+\-\s*1\s*\)\s*/\s*[\w.]+")


def _mask_or_where_line(line: str) -> bool:
    return "mask=" in line or "tl.where" in line


def _cuda_idx_boundary_line(line: str) -> bool:
    """Detect CUDA boundary guard ``if`` statements.

    Triggers when the line is a C-style ``if``/``elif`` that either:
    - directly mentions ``threadIdx`` / ``blockIdx``, or
    - is a C-style block (ends with ``{`` or contains ``return;``) with a
      comparison operator, indicating it guards against out-of-bounds access.
    """
    stripped = line.lstrip()
    if not (stripped.startswith("if ") or stripped.startswith("if(")
            or stripped.startswith("elif ")):
        return False
    if "threadIdx" in line or "blockIdx" in line:
        return True
    tail = stripped.rstrip()
    if tail.endswith("{") or "return;" in stripped or "return ;" in stripped:
        if re.search(r"[<>]=?", stripped):
            return True
    return False


class IndexReplace(MutationOperator):
    name = "index_replace"
    category = "B"
    description = (
        "Swap Triton program_id axis or CUDA thread/block dimension index "
        "(e.g. program_id(0)→(1), threadIdx.x→threadIdx.y)"
    )

    def find_sites(self, source: str, tree: Optional[ast.Module] = None) -> List[MutationSite]:
        sites: List[MutationSite] = []

        for li, line in enumerate(source.splitlines(), start=1):
            for m in _TRITON_PROG.finditer(line):
                if _is_in_comment_or_string(line, m.start()):
                    continue
                d = m.group(1)
                for nd in ("0", "1", "2"):
                    if nd == d:
                        continue
                    sites.append(
                        MutationSite(
                            line_start=li,
                            line_end=li,
                            col_start=m.start(),
                            col_end=m.end(),
                            original_code=m.group(0),
                            node_type=f"triton_pi|{nd}",
                        )
                    )

        def cuda_opts(axis: str) -> List[str]:
            order = ("x", "y", "z")
            i = order.index(axis)
            return [order[j] for j in range(3) if j != i]

        for li, line in enumerate(source.splitlines(), start=1):
            for m in _CUDA_DIM.finditer(line):
                if _is_in_comment_or_string(line, m.start()):
                    continue
                ax = m.group(2)
                for na in cuda_opts(ax):
                    sites.append(
                        MutationSite(
                            line_start=li,
                            line_end=li,
                            col_start=m.start(),
                            col_end=m.end(),
                            original_code=m.group(0),
                            node_type=f"cuda_dim|{na}",
                        )
                    )

        sites.sort(key=lambda s: (s.line_start, s.col_start, s.node_type))
        return sites

    def apply(self, source: str, site: MutationSite) -> str:
        nt = site.node_type
        old = site.original_code
        if nt.startswith("triton_pi|"):
            nd = nt.split("|", 1)[1]
            new = re.sub(r"\(\s*[012]\s*\)", f"({nd})", old, count=1)
            if new == old:
                return source
            return _replace_in_source(source, site, old, new)
        if nt.startswith("cuda_dim|"):
            na = nt.split("|", 1)[1]
            new = re.sub(r"\.(x|y|z)\b", f".{na}", old, count=1)
            return _replace_in_source(source, site, old, new)
        return source


class SyncRemove(MutationOperator):
    name = "sync_remove"
    category = "B"
    description = "Remove GPU synchronization barriers (tl.debug_barrier, __syncthreads)"

    _TID0_GUARD = re.compile(
        r"if\s*\(\s*(?:threadIdx\.x|tid)\s*==\s*0\s*\)"
    )
    _SHARED_ACCESS = re.compile(r"\b(?:shared|sdata|s_data|smem)\w*\[")

    @classmethod
    def _is_reduction_tail_sync(cls, source: str, line_no: int) -> bool:
        """Detect __syncthreads() after a reduction loop where only tid==0
        reads shared memory afterward.  Removing such a barrier is benign
        because no cross-warp race exists on the post-reduction read.
        """
        lines = source.splitlines()
        n = len(lines)
        idx = line_no - 1  # 0-based

        # Look ahead: first non-blank / non-brace line should be `if (tid == 0)`
        for i in range(idx + 1, min(idx + 5, n)):
            stripped = lines[i].strip()
            if not stripped or stripped == "{":
                continue
            if cls._TID0_GUARD.search(stripped):
                return True
            break  # first meaningful line does not match

        # Also flag when this is the *last* __syncthreads in the kernel and
        # every subsequent shared-memory access uses index [0] only.
        remaining = "\n".join(lines[idx + 1:])
        if "__syncthreads" not in remaining:
            accesses = re.findall(r"\b(?:shared|sdata|s_data|smem)\w*\[(\w+)\]",
                                  remaining)
            if accesses and all(a == "0" for a in accesses):
                return True

        return False

    def find_sites(self, source: str, tree: Optional[ast.Module] = None) -> List[MutationSite]:
        sites: List[MutationSite] = []
        for pat, nt in (
            (r"tl\.debug_barrier\s*\(\s*\)\s*;?", "triton_barrier"),
            (r"__syncthreads\s*\(\s*\)\s*;?", "cuda_syncthreads"),
        ):
            for s in _find_pattern_sites(source, pat, nt):
                line = source.splitlines()[s.line_start - 1]
                if _is_in_comment_or_string(line, s.col_start):
                    continue
                if nt == "cuda_syncthreads":
                    if self._is_reduction_tail_sync(source, s.line_start):
                        s = MutationSite(
                            line_start=s.line_start, line_end=s.line_end,
                            col_start=s.col_start, col_end=s.col_end,
                            original_code=s.original_code,
                            node_type="cuda_syncthreads:reduction_tail",
                        )
                sites.append(s)
        sites.sort(key=lambda s: (s.line_start, s.col_start))
        return sites

    def apply(self, source: str, site: MutationSite) -> str:
        lines = source.splitlines(keepends=True)
        row = site.line_start - 1
        if 0 <= row < len(lines):
            line = lines[row]
            before = line[: site.col_start]
            after = line[site.col_end :]
            if (before + after).strip() == "":
                return "".join(lines[:row] + lines[row + 1 :])
        return _replace_in_source(source, site, site.original_code, "")


class MaskBoundary(MutationOperator):
    name = "mask_boundary"
    category = "B"
    description = (
        "Weaken or tighten boundary checks in mask/tl.where (Triton) or "
        "thread/block guards (CUDA)"
    )

    def find_sites(self, source: str, tree: Optional[ast.Module] = None) -> List[MutationSite]:
        sites: List[MutationSite] = []

        def add_lt_sites(line: str, li: int) -> None:
            """``word < word`` → ``< word - 1`` (tighten only; ``<=`` widen direction excluded)."""
            for m in _COMP_LT.finditer(line):
                if _is_in_comment_or_string(line, m.start()):
                    continue
                frag = m.group(0)
                sites.append(
                    MutationSite(
                        line_start=li, line_end=li,
                        col_start=m.start(), col_end=m.end(),
                        original_code=frag, node_type="rhs-1",
                    )
                )

        def add_ge_sites(line: str, li: int) -> None:
            """``word >= word`` → ``>`` (tighten) or ``>= word + 1`` (tighten more)."""
            for m in _COMP_GE.finditer(line):
                if _is_in_comment_or_string(line, m.start()):
                    continue
                frag = m.group(0)
                sites.append(
                    MutationSite(
                        line_start=li, line_end=li,
                        col_start=m.start(), col_end=m.end(),
                        original_code=frag, node_type="ge>",
                    )
                )
                sites.append(
                    MutationSite(
                        line_start=li, line_end=li,
                        col_start=m.start(), col_end=m.end(),
                        original_code=frag, node_type="ge_rhs+1",
                    )
                )

        for li, line in enumerate(source.splitlines(), start=1):
            trigger = _mask_or_where_line(line) or _cuda_idx_boundary_line(line)
            if trigger:
                add_lt_sites(line, li)
                add_ge_sites(line, li)

        sites.sort(
            key=lambda s: (s.line_start, s.col_start, s.node_type, s.original_code)
        )
        return sites

    def apply(self, source: str, site: MutationSite) -> str:
        old = site.original_code

        # word < word  →  word <= word
        if site.node_type == "<=":
            if "<" not in old or "<=" in old:
                return source
            new = old.replace("<", "<=", 1)
            return _replace_in_source(source, site, old, new)

        # word < word  →  word < word - 1
        if site.node_type == "rhs-1":
            m = _COMP_LT.fullmatch(old.strip())
            if m is None:
                parts = old.split("<", 1)
                if len(parts) != 2:
                    return source
                lhs, rhs = parts[0].strip(), parts[1].strip()
            else:
                lhs, rhs = m.group(1), m.group(2)
            new = f"{lhs} < {rhs} - 1"
            return _replace_in_source(source, site, old, new)

        # word >= word  →  word > word
        if site.node_type == "ge>":
            if ">=" not in old:
                return source
            new = old.replace(">=", ">", 1)
            return _replace_in_source(source, site, old, new)

        # word >= word  →  word >= word + 1
        if site.node_type == "ge_rhs+1":
            m = _COMP_GE.fullmatch(old.strip())
            if m is None:
                parts = old.split(">=", 1)
                if len(parts) != 2:
                    return source
                lhs, rhs = parts[0].strip(), parts[1].strip()
            else:
                lhs, rhs = m.group(1), m.group(2)
            new = f"{lhs} >= {rhs} + 1"
            return _replace_in_source(source, site, old, new)

        return source


class LaunchConfigMutate(MutationOperator):
    name = "launch_config_mutate"
    category = "B"
    description = (
        "Perturb grid/block sizing expressions (// BLOCK, triton.cdiv) by ±1"
    )

    def find_sites(self, source: str, tree: Optional[ast.Module] = None) -> List[MutationSite]:
        sites: List[MutationSite] = []

        def _add_site(li: int, m: re.Match) -> None:
            whole = m.group(0)
            sites.append(
                MutationSite(
                    line_start=li, line_end=li,
                    col_start=m.start(), col_end=m.end(),
                    original_code=whole, node_type="-1",
                )
            )

        for li, line in enumerate(source.splitlines(), start=1):
            # Triton: triton.cdiv(...)
            for m in _CDIV.finditer(line):
                if _is_in_comment_or_string(line, m.start()):
                    continue
                _add_site(li, m)

            # Python floor division: X // Y
            for m in _FLOORDIV.finditer(line):
                if _is_in_comment_or_string(line, m.start()):
                    continue
                _add_site(li, m)

            # CUDA C++: cdiv(...)  helper function
            for m in _CUDA_CDIV.finditer(line):
                if _is_in_comment_or_string(line, m.start()):
                    continue
                _add_site(li, m)

            # CUDA C++: (expr - 1) / expr  ceiling division idiom
            for m in _CUDA_CEIL_DIV.finditer(line):
                if _is_in_comment_or_string(line, m.start()):
                    continue
                _add_site(li, m)

        sites.sort(key=lambda s: (s.line_start, s.col_start, s.node_type))
        return sites

    def apply(self, source: str, site: MutationSite) -> str:
        old = site.original_code
        if site.node_type == "+1":
            new = f"{old} + 1"
            return _replace_in_source(source, site, old, new)
        if site.node_type == "-1":
            new = f"{old} - 1"
            return _replace_in_source(source, site, old, new)
        return source
