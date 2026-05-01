"""Category C: ML numerical semantics mutation operators.

Supports both:
  - Triton / PyTorch-style kernels (Python syntax)
  - CUDA C++ kernels embedded in Python strings via load_inline
"""

from __future__ import annotations

import ast
import re
from typing import List, Optional, Tuple

from ...models import MutationSite
from .base import (
    MutationOperator,
    _cuda_find_pattern_sites,
    _replace_at_columns,
    _strip_cuda_comment,
)


def _index_before_outside_string_comment(line: str, idx: int) -> bool:
    i = 0
    n = len(line)
    in_triple: Optional[str] = None
    in_string: Optional[str] = None
    while i < idx and i < n:
        c = line[i]
        if in_triple:
            if line.startswith(in_triple, i):
                i += 3
                in_triple = None
            else:
                i += 1
            continue
        if in_string:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == in_string:
                in_string = None
            i += 1
            continue
        if line.startswith("'''", i) or line.startswith('"""', i):
            in_triple = line[i : i + 3]
            i += 3
            continue
        if c in "\"'":
            in_string = c
            i += 1
            continue
        if c == "#":
            return False
        i += 1
    return True


def _line_code_and_comment_start(line: str) -> Tuple[str, Optional[int]]:
    i = 0
    n = len(line)
    in_triple: Optional[str] = None
    in_string: Optional[str] = None
    while i < n:
        c = line[i]
        if in_triple:
            if line.startswith(in_triple, i):
                i += 3
                in_triple = None
            else:
                i += 1
            continue
        if in_string:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == in_string:
                in_string = None
            i += 1
            continue
        if line.startswith("'''", i) or line.startswith('"""', i):
            in_triple = line[i : i + 3]
            i += 3
            continue
        if c in "\"'":
            in_string = c
            i += 1
            continue
        if c == "#":
            return line[:i], i
        i += 1
    return line, None


def _balanced_paren_span(s: str, open_paren_index: int) -> Optional[Tuple[int, int]]:
    if open_paren_index >= len(s) or s[open_paren_index] != "(":
        return None
    depth = 0
    i = open_paren_index
    while i < len(s):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return open_paren_index, i + 1
        i += 1
    return None


def _first_call_arg(inner: str) -> str:
    depth = 0
    start = 0
    i = 0
    while i < len(inner):
        c = inner[i]
        if c in "\"'":
            q = c
            i += 1
            while i < len(inner):
                if inner[i] == "\\":
                    i += 2
                    continue
                if inner[i] == q:
                    i += 1
                    break
                i += 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            return inner[start:i].strip()
        i += 1
    return inner[start:].strip()


def _mutation_site_from_span(
    line_no: int,
    line: str,
    start: int,
    end: int,
    node_type: str,
) -> Optional[MutationSite]:
    if start < 0 or end > len(line) or start >= end:
        return None
    if not _index_before_outside_string_comment(line, start):
        return None
    return MutationSite(
        line_start=line_no,
        line_end=line_no,
        col_start=start,
        col_end=end,
        original_code=line[start:end],
        node_type=node_type,
    )


class StabRemove(MutationOperator):
    name = "stab_remove"
    category = "C"
    description = "Remove max-subtraction numerical stabilization (overflow on large inputs)"

    _TL_MAX_HEAD = re.compile(r"([\w.]+)\s*-\s*tl\.max\s*\(")
    _TORCH_MAX_HEAD = re.compile(r"([\w.]+)\s*-\s*torch\.max\s*\(")
    _METHOD_MAX_HEAD = re.compile(r"([\w.]+)\s*-\s*\1\.max\s*\(")

    # CUDA: expf(val - max_val)  →  expf(val)
    _CUDA_EXPF_MINUS = re.compile(
        r"\bexpf?\s*\(\s*([\w.\[\]]+)\s*-\s*([\w.\[\]]+)\s*\)"
    )
    # CUDA: val - max_val  where max is in the variable name
    _CUDA_MINUS_MAX_VAR = re.compile(
        r"([\w.\[\]]+)\s*-\s*([\w]*(?:max|MAX)[\w]*)"
    )

    def find_sites(self, source: str, tree=None) -> List[MutationSite]:
        sites: List[MutationSite] = []
        for line_no, raw in enumerate(source.splitlines(), start=1):
            code, _ = _line_code_and_comment_start(raw)
            line = raw
            # --- Triton / PyTorch patterns ---
            for rx, kind in (
                (self._TL_MAX_HEAD, "stab:tl_max"),
                (self._TORCH_MAX_HEAD, "stab:torch_max"),
                (self._METHOD_MAX_HEAD, "stab:method_max"),
            ):
                for m in rx.finditer(code):
                    if not _index_before_outside_string_comment(line, m.start()):
                        continue
                    open_i = m.end() - 1
                    span = _balanced_paren_span(code, open_i)
                    if span is None:
                        continue
                    inner = code[span[0] + 1 : span[1] - 1]
                    if _first_call_arg(inner) != m.group(1):
                        continue
                    end = span[1]
                    site = _mutation_site_from_span(line_no, line, m.start(), end, kind)
                    if site:
                        sites.append(site)

            # --- CUDA C++ patterns ---
            cuda_code = _strip_cuda_comment(raw)
            for m in self._CUDA_EXPF_MINUS.finditer(cuda_code):
                sites.append(MutationSite(
                    line_start=line_no, line_end=line_no,
                    col_start=m.start(), col_end=m.end(),
                    original_code=m.group(0),
                    node_type="stab:cuda_expf_minus",
                ))
            for m in self._CUDA_MINUS_MAX_VAR.finditer(cuda_code):
                if any(s.line_start == line_no and s.col_start <= m.start() < s.col_end
                       for s in sites):
                    continue
                sites.append(MutationSite(
                    line_start=line_no, line_end=line_no,
                    col_start=m.start(), col_end=m.end(),
                    original_code=m.group(0),
                    node_type="stab:cuda_minus_max",
                ))
        sites.sort(key=lambda s: (s.line_start, s.col_start))
        return sites

    def apply(self, source: str, site: MutationSite) -> str:
        frag = site.original_code
        nt = site.node_type
        if nt in ("stab:tl_max", "stab:torch_max", "stab:method_max"):
            if "-" not in frag:
                return source
            left = frag.split("-", 1)[0].strip()
            return _replace_at_columns(source, site, left)
        if nt == "stab:cuda_expf_minus":
            m = self._CUDA_EXPF_MINUS.fullmatch(frag)
            if m:
                func = "expf" if frag.lstrip().startswith("expf") else "exp"
                return _replace_at_columns(source, site, f"{func}({m.group(1)})")
        if nt == "stab:cuda_minus_max":
            m = self._CUDA_MINUS_MAX_VAR.fullmatch(frag)
            if m:
                return _replace_at_columns(source, site, m.group(1))
        return source


def _zeros_fp32_call_sites(
    line: str,
    line_no: int,
    prefix: str,
    dtype_needle: str,
    node_type: str,
) -> List[MutationSite]:
    sites: List[MutationSite] = []
    code, _ = _line_code_and_comment_start(line)
    needle = re.compile(rf"dtype\s*=\s*{re.escape(dtype_needle)}\b")
    i = 0
    plen = len(prefix)
    while True:
        j = code.find(prefix, i)
        if j == -1:
            break
        before = code[j - 1] if j > 0 else ""
        if before.isalnum() or before == "_":
            i = j + plen
            continue
        k = j + plen
        while k < len(code) and code[k].isspace():
            k += 1
        if k >= len(code) or code[k] != "(":
            i = j + 1
            continue
        span = _balanced_paren_span(code, k)
        if span is None:
            i = j + 1
            continue
        full_end = span[1]
        fragment = code[j:full_end]
        if not needle.search(fragment):
            i = j + 1
            continue
        st = _mutation_site_from_span(line_no, line, j, full_end, node_type)
        if st:
            sites.append(st)
        i = full_end
    return sites


class AccDowngrade(MutationOperator):
    name = "acc_downgrade"
    category = "C"
    description = "Downgrade FP32 accumulators to FP16 / half precision"

    _TO_FP32 = re.compile(r"\.to\s*\(\s*torch\.float32\s*\)")
    _TO_TL_FP32 = re.compile(r"\.to\s*\(\s*tl\.float32\s*\)")
    _DOT_FLOAT = re.compile(r"\.float\s*\(\s*\)")
    _DOT_DOUBLE = re.compile(r"\.double\s*\(\s*\)")

    # CUDA: static_cast<float>(expr)
    _CUDA_STATIC_CAST_FLOAT = re.compile(
        r"static_cast\s*<\s*float\s*>\s*\("
    )
    # CUDA: (float)expr  C-style cast
    _CUDA_C_CAST_FLOAT = re.compile(
        r"(?<!\w)\(\s*float\s*\)(?=\s*[\w(])"
    )
    # CUDA: __half2float(expr)
    _CUDA_HALF2FLOAT = re.compile(r"\b__half2float\s*\(")

    def find_sites(self, source: str, tree=None) -> List[MutationSite]:
        sites: List[MutationSite] = []
        for line_no, raw in enumerate(source.splitlines(), start=1):
            code, _ = _line_code_and_comment_start(raw)
            line = raw
            # --- Triton / PyTorch patterns ---
            sites.extend(
                _zeros_fp32_call_sites(line, line_no, "tl.zeros", "tl.float32", "acc:tl_zeros_fp32")
            )
            sites.extend(
                _zeros_fp32_call_sites(
                    line, line_no, "torch.zeros", "torch.float32", "acc:torch_zeros_fp32"
                )
            )

            def add_rx(rx: re.Pattern[str], nt: str) -> None:
                for m in rx.finditer(code):
                    if not _index_before_outside_string_comment(line, m.start()):
                        continue
                    st = _mutation_site_from_span(
                        line_no, line, m.start(), m.end(), nt
                    )
                    if st:
                        sites.append(st)

            add_rx(self._TO_FP32, "acc:to_torch_fp32")
            add_rx(self._TO_TL_FP32, "acc:to_tl_fp32")
            add_rx(self._DOT_FLOAT, "acc:dot_float")
            add_rx(self._DOT_DOUBLE, "acc:dot_double")

            # --- CUDA C++ patterns ---
            cuda_code = _strip_cuda_comment(raw)
            for m in self._CUDA_STATIC_CAST_FLOAT.finditer(cuda_code):
                open_i = m.end() - 1
                span = _balanced_paren_span(cuda_code, open_i)
                if span is None:
                    continue
                full = cuda_code[m.start():span[1]]
                sites.append(MutationSite(
                    line_start=line_no, line_end=line_no,
                    col_start=m.start(), col_end=span[1],
                    original_code=full,
                    node_type="acc:cuda_static_cast_float",
                ))
            for m in self._CUDA_C_CAST_FLOAT.finditer(cuda_code):
                sites.append(MutationSite(
                    line_start=line_no, line_end=line_no,
                    col_start=m.start(), col_end=m.end(),
                    original_code=m.group(0),
                    node_type="acc:cuda_c_cast_float",
                ))
            for m in self._CUDA_HALF2FLOAT.finditer(cuda_code):
                open_i = m.end() - 1
                span = _balanced_paren_span(cuda_code, open_i)
                if span is None:
                    continue
                full = cuda_code[m.start():span[1]]
                sites.append(MutationSite(
                    line_start=line_no, line_end=line_no,
                    col_start=m.start(), col_end=span[1],
                    original_code=full,
                    node_type="acc:cuda_half2float",
                ))
        sites.sort(key=lambda s: (s.line_start, s.col_start))
        return sites

    def apply(self, source: str, site: MutationSite) -> str:
        text = site.original_code
        nt = site.node_type
        if nt == "acc:tl_zeros_fp32":
            new_t = re.sub(r"\btl\.float32\b", "tl.float16", text)
            return _replace_at_columns(source, site, new_t) if new_t != text else source
        if nt == "acc:torch_zeros_fp32":
            new_t = re.sub(r"\btorch\.float32\b", "torch.float16", text)
            return _replace_at_columns(source, site, new_t) if new_t != text else source
        if nt == "acc:to_torch_fp32":
            return _replace_at_columns(source, site, ".to(torch.float16)")
        if nt == "acc:to_tl_fp32":
            return _replace_at_columns(source, site, ".to(tl.float16)")
        if nt == "acc:dot_float":
            return _replace_at_columns(source, site, ".half()")
        if nt == "acc:dot_double":
            return _replace_at_columns(source, site, ".float()")
        # --- CUDA ---
        if nt == "acc:cuda_static_cast_float":
            new_t = re.sub(
                r"static_cast\s*<\s*float\s*>",
                "static_cast<__half>",
                text, count=1,
            )
            return _replace_at_columns(source, site, new_t)
        if nt == "acc:cuda_c_cast_float":
            return _replace_at_columns(source, site, "(__half)")
        if nt == "acc:cuda_half2float":
            m = re.match(r"__half2float\s*\(\s*(.+?)\s*\)\s*$", text)
            if m:
                return _replace_at_columns(source, site, m.group(1))
            return source
        return source


class EpsilonModify(MutationOperator):
    name = "epsilon_modify"
    category = "C"
    description = "Alter small epsilon literals (LayerNorm / safe_div / log stability)"

    # Matches scientific-notation floats, with optional f/F/l/L suffix (CUDA uses 'f')
    _EPS_SCI = re.compile(
        r"(?<![\w.])"
        r"[+-]?(?:\d+(?:_\d+)*)?(?:\.\d+(?:_\d+)*)?[eE][+-]?\d+(?:_\d+)*"
        r"[fFlL]?"
        r"(?![\w.])"
    )

    # Matches small decimal floats like 0.00001f, 0.000001 (>=5 leading zeros after dot)
    _EPS_DEC = re.compile(
        r"(?<![\w.])"
        r"0\.0{4,}\d+"
        r"[fFlL]?"
        r"(?![\w.])"
    )

    _EPS_THRESHOLD_ABS = 1e-3

    @classmethod
    def _is_small_epsilon(cls, raw: str) -> bool:
        t = raw.replace("_", "")
        if t and t[-1] in "fFlL":
            t = t[:-1]
        try:
            v = abs(float(t))
        except ValueError:
            return False
        return 0 < v <= cls._EPS_THRESHOLD_ABS

    def find_sites(self, source: str, tree=None) -> List[MutationSite]:
        sites: List[MutationSite] = []
        for line_no, raw in enumerate(source.splitlines(), start=1):
            code_py, _ = _line_code_and_comment_start(raw)
            code_cuda = _strip_cuda_comment(raw)
            searched: set = set()
            for code, is_cuda in ((code_py, False), (code_cuda, True)):
                for rx in (self._EPS_SCI, self._EPS_DEC):
                    for m in rx.finditer(code):
                        pos_key = (line_no, m.start(), m.end())
                        if pos_key in searched:
                            continue
                        searched.add(pos_key)
                        if not is_cuda and not _index_before_outside_string_comment(raw, m.start()):
                            continue
                        raw_lit = m.group(0)
                        if not self._is_small_epsilon(raw_lit):
                            continue
                        for nt in ("eps:to_zero", "eps:to_1e-2"):
                            sites.append(MutationSite(
                                line_start=line_no, line_end=line_no,
                                col_start=m.start(), col_end=m.end(),
                                original_code=raw_lit,
                                node_type=nt,
                            ))
        sites.sort(key=lambda s: (s.line_start, s.col_start, s.node_type))
        return sites

    def apply(self, source: str, site: MutationSite) -> str:
        raw_lit = site.original_code
        has_suffix = raw_lit and raw_lit[-1] in "fFlL"
        suffix = raw_lit[-1] if has_suffix else ""
        if site.node_type == "eps:to_zero":
            return _replace_at_columns(source, site, f"0{suffix}")
        if site.node_type == "eps:to_1e-2":
            return _replace_at_columns(source, site, f"1e-2{suffix}")
        return source


def _inv_sqrt_scale_sites(
    line: str,
    line_no: int,
    module: str,
    node_type: str,
) -> List[MutationSite]:
    sites: List[MutationSite] = []
    code, _ = _line_code_and_comment_start(line)
    tail = rf"{re.escape(module)}\.sqrt\s*\("
    head = re.compile(rf"(?:\b1\.0\b|\b1\b)\s*/\s*{tail}")
    for m in head.finditer(code):
        if not _index_before_outside_string_comment(line, m.start()):
            continue
        open_i = m.end() - 1
        span = _balanced_paren_span(code, open_i)
        if span is None:
            continue
        end = span[1]
        st = _mutation_site_from_span(line_no, line, m.start(), end, node_type)
        if st:
            sites.append(st)
    return sites


def _rsqrt_sites(line: str, line_no: int) -> List[MutationSite]:
    sites: List[MutationSite] = []
    code, _ = _line_code_and_comment_start(line)
    head = re.compile(r"\b(?:math|torch|tl)\.rsqrt\s*\(")
    for m in head.finditer(code):
        if not _index_before_outside_string_comment(line, m.start()):
            continue
        open_i = m.end() - 1
        span = _balanced_paren_span(code, open_i)
        if span is None:
            continue
        st = _mutation_site_from_span(
            line_no, line, m.start(), span[1], "scale:rsqrt_identity"
        )
        if st:
            sites.append(st)
    return sites


class ScaleModify(MutationOperator):
    name = "scale_modify"
    category = "C"
    description = "Distort attention / normalization scaling (inv-sqrt, rsqrt)"

    # CUDA: rsqrtf(expr), rsqrt(expr)
    _CUDA_RSQRT = re.compile(r"\brsqrtf?\s*\(")
    # CUDA: 1.0f / sqrtf(...), 1.0 / sqrt(...)
    _CUDA_INV_SQRT = re.compile(r"(?:\b1\.0f?\b|\b1\b)\s*/\s*sqrtf?\s*\(")

    def find_sites(self, source: str, tree=None) -> List[MutationSite]:
        sites: List[MutationSite] = []
        for line_no, raw in enumerate(source.splitlines(), start=1):
            line = raw
            # --- Triton / PyTorch patterns ---
            sites.extend(
                _inv_sqrt_scale_sites(line, line_no, "math", "scale:drop_inv_sqrt:math")
            )
            sites.extend(
                _inv_sqrt_scale_sites(line, line_no, "torch", "scale:drop_inv_sqrt:torch")
            )
            sites.extend(_inv_sqrt_scale_sites(line, line_no, "tl", "scale:drop_inv_sqrt:tl"))
            sites.extend(_rsqrt_sites(line, line_no))

            # --- CUDA C++ patterns ---
            cuda_code = _strip_cuda_comment(raw)
            # 1/sqrtf(...) → sqrtf(...)   (drop the inverse)
            for m in self._CUDA_INV_SQRT.finditer(cuda_code):
                open_i = m.end() - 1
                span = _balanced_paren_span(cuda_code, open_i)
                if span is None:
                    continue
                full = cuda_code[m.start():span[1]]
                sites.append(MutationSite(
                    line_start=line_no, line_end=line_no,
                    col_start=m.start(), col_end=span[1],
                    original_code=full,
                    node_type="scale:cuda_inv_sqrt",
                ))
            # rsqrtf(expr) → 1.0f  (replace with identity)
            for m in self._CUDA_RSQRT.finditer(cuda_code):
                open_i = m.end() - 1
                span = _balanced_paren_span(cuda_code, open_i)
                if span is None:
                    continue
                full = cuda_code[m.start():span[1]]
                sites.append(MutationSite(
                    line_start=line_no, line_end=line_no,
                    col_start=m.start(), col_end=span[1],
                    original_code=full,
                    node_type="scale:cuda_rsqrt_identity",
                ))
        sites.sort(key=lambda s: (s.line_start, s.col_start))
        return sites

    def apply(self, source: str, site: MutationSite) -> str:
        frag = site.original_code
        nt = site.node_type
        if nt.startswith("scale:drop_inv_sqrt:"):
            inner = re.sub(
                r"^(?:\b1\.0\b|\b1\b)\s*/\s*",
                "",
                frag.strip(),
                count=1,
            )
            return _replace_at_columns(source, site, inner)
        if nt == "scale:rsqrt_identity":
            m = re.search(r"\(\s*([^()]+\s*)\s*\)\s*$", frag)
            if not m:
                return source
            return _replace_at_columns(source, site, m.group(1).strip())
        # --- CUDA ---
        if nt == "scale:cuda_inv_sqrt":
            inner = re.sub(
                r"^(?:\b1\.0f?\b|\b1\b)\s*/\s*", "", frag.strip(), count=1,
            )
            return _replace_at_columns(source, site, inner)
        if nt == "scale:cuda_rsqrt_identity":
            m = re.search(r"\(\s*(.+?)\s*\)\s*$", frag)
            if not m:
                return source
            return _replace_at_columns(source, site, m.group(1).strip())
        return source


class CastRemove(MutationOperator):
    name = "cast_remove"
    category = "C"
    description = "Remove explicit dtype casts (.to, .float, tl.cast, static_cast)"

    _TO_TORCH_DTYPE = re.compile(
        r"\.to\s*\(\s*torch\.(?:float32|float16|bfloat16|int32|int64)\s*\)"
    )
    _TO_TL_DTYPE = re.compile(r"\.to\s*\(\s*tl\.(?:float32|float16|bfloat16|int32)\s*\)")
    _CAST_FLOAT_METHOD = re.compile(r"\.(?:float|half|bfloat16)\s*\(\s*\)")
    _TL_CAST_HEAD = re.compile(r"\btl\.cast\s*\(")

    # CUDA: static_cast<T>(expr) for precision-related types
    _CUDA_STATIC_CAST = re.compile(
        r"static_cast\s*<\s*"
        r"(?:float|double|__half|at::Half|at::BFloat16|__nv_bfloat16|int|int32_t|int64_t)"
        r"\s*>\s*\("
    )
    # CUDA: (float)expr, (double)expr, (__half)expr  C-style casts
    _CUDA_C_CAST = re.compile(
        r"(?<!\w)\(\s*(?:float|double|__half)\s*\)(?=\s*[\w(])"
    )
    # CUDA: __float2half(expr), __half2float(expr), __double2float(expr)
    _CUDA_INTRINSIC_CAST = re.compile(
        r"\b(?:__float2half|__half2float|__double2float|__float2half_rn)\s*\("
    )

    @staticmethod
    def _is_redundant_cuda_static_cast(line: str, cast_target_type: str) -> bool:
        """Detect static_cast<T> that is redundant due to C++ implicit conversion.

        Returns True when the surrounding context already guarantees the same
        type promotion, making the explicit cast semantically unnecessary.
        Removing such a cast produces a mutant that is *always* equivalent.

        Patterns detected:
        1. ``float var = static_cast<float>(expr);``  — assignment LHS matches T
        2. ``const float var = ...``  — const-qualified variant
        3. Compound assign ``var += static_cast<float>(expr);`` in a line that
           contains float literals (e.g. ``0.0f``) or a prior ``float`` keyword.
        """
        if not cast_target_type:
            return False
        stripped = line.lstrip()
        t_esc = re.escape(cast_target_type)
        # Pattern 1/2: variable declaration with same type
        if re.match(rf"(?:const\s+)?{t_esc}\s+\w+\s*[=;]", stripped):
            return True
        # Pattern 3: compound assignment in a float arithmetic context
        if cast_target_type == "float":
            if re.match(r"\w+\s*[+\-*/]?=", stripped):
                if re.search(r"\bfloat\b", line) or re.search(r"\d+\.\d*f\b", line):
                    return True
        return False

    def find_sites(self, source: str, tree=None) -> List[MutationSite]:
        sites: List[MutationSite] = []
        for line_no, raw in enumerate(source.splitlines(), start=1):
            code, _ = _line_code_and_comment_start(raw)
            line = raw
            # --- Triton / PyTorch patterns ---
            for m in self._TO_TORCH_DTYPE.finditer(code):
                if not _index_before_outside_string_comment(line, m.start()):
                    continue
                st = _mutation_site_from_span(
                    line_no, line, m.start(), m.end(), "cast:remove_to_torch"
                )
                if st:
                    sites.append(st)
            for m in self._TO_TL_DTYPE.finditer(code):
                if not _index_before_outside_string_comment(line, m.start()):
                    continue
                st = _mutation_site_from_span(
                    line_no, line, m.start(), m.end(), "cast:remove_to_tl"
                )
                if st:
                    sites.append(st)
            for m in self._CAST_FLOAT_METHOD.finditer(code):
                if not _index_before_outside_string_comment(line, m.start()):
                    continue
                st = _mutation_site_from_span(
                    line_no, line, m.start(), m.end(), "cast:remove_method"
                )
                if st:
                    sites.append(st)
            for m in self._TL_CAST_HEAD.finditer(code):
                if not _index_before_outside_string_comment(line, m.start()):
                    continue
                open_i = m.end() - 1
                outer = _balanced_paren_span(code, open_i)
                if outer is None:
                    continue
                st = _mutation_site_from_span(
                    line_no, line, m.start(), outer[1], "cast:tl_cast",
                )
                if st:
                    sites.append(st)

            # --- CUDA C++ patterns ---
            cuda_code = _strip_cuda_comment(raw)
            for m in self._CUDA_STATIC_CAST.finditer(cuda_code):
                open_i = m.end() - 1
                span = _balanced_paren_span(cuda_code, open_i)
                if span is None:
                    continue
                full = cuda_code[m.start():span[1]]
                # Extract target type for redundancy analysis
                tm = re.match(r"static_cast\s*<\s*(\w[\w:]*)", full)
                target_type = tm.group(1) if tm else ""
                is_redundant = self._is_redundant_cuda_static_cast(raw, target_type)
                nt = ("cast:cuda_static_cast:redundant" if is_redundant
                      else "cast:cuda_static_cast")
                sites.append(MutationSite(
                    line_start=line_no, line_end=line_no,
                    col_start=m.start(), col_end=span[1],
                    original_code=full,
                    node_type=nt,
                ))
            for m in self._CUDA_C_CAST.finditer(cuda_code):
                sites.append(MutationSite(
                    line_start=line_no, line_end=line_no,
                    col_start=m.start(), col_end=m.end(),
                    original_code=m.group(0),
                    node_type="cast:cuda_c_cast",
                ))
            for m in self._CUDA_INTRINSIC_CAST.finditer(cuda_code):
                open_i = m.end() - 1
                span = _balanced_paren_span(cuda_code, open_i)
                if span is None:
                    continue
                full = cuda_code[m.start():span[1]]
                sites.append(MutationSite(
                    line_start=line_no, line_end=line_no,
                    col_start=m.start(), col_end=span[1],
                    original_code=full,
                    node_type="cast:cuda_intrinsic",
                ))
        sites.sort(key=lambda s: (s.line_start, s.col_start, s.col_end))
        return sites

    def apply(self, source: str, site: MutationSite) -> str:
        nt = site.node_type
        frag = site.original_code
        if nt in ("cast:remove_to_torch", "cast:remove_to_tl"):
            return _replace_at_columns(source, site, "")
        if nt == "cast:remove_method":
            return _replace_at_columns(source, site, "")
        if nt == "cast:tl_cast":
            m = re.match(r"tl\.cast\s*\(\s*([^,]+?)\s*,", frag)
            if not m:
                return source
            return _replace_at_columns(source, site, m.group(1).strip())
        # --- CUDA ---
        if nt == "cast:cuda_static_cast":
            m = re.match(r"static_cast\s*<\s*[\w:]+\s*>\s*\(\s*(.+?)\s*\)\s*$", frag)
            if m:
                return _replace_at_columns(source, site, m.group(1))
            return source
        if nt == "cast:cuda_c_cast":
            return _replace_at_columns(source, site, "")
        if nt == "cast:cuda_intrinsic":
            m = re.match(r"__\w+\s*\(\s*(.+?)\s*\)\s*$", frag)
            if m:
                return _replace_at_columns(source, site, m.group(1))
            return source
        return source


class ReductionReorder(MutationOperator):
    name = "reduction_reorder"
    category = "C"
    description = "Reorder reduction inputs to expose non-associative FP accumulation"

    _TL_SUM = re.compile(
        r"\btl\.sum\s*\(\s*([\w.]+)\s*,\s*(axis|dim)\s*=\s*([^,\s)]+)\s*\)"
    )
    _TORCH_SUM = re.compile(
        r"\btorch\.sum\s*\(\s*([\w.]+)\s*,\s*dim\s*=\s*([^,\s)]+)\s*\)"
    )

    def find_sites(self, source: str, tree: Optional[ast.Module] = None) -> List[MutationSite]:
        sites: List[MutationSite] = []
        for line_no, raw in enumerate(source.splitlines(), start=1):
            code, _ = _line_code_and_comment_start(raw)
            line = raw
            for m in self._TL_SUM.finditer(code):
                if not _index_before_outside_string_comment(line, m.start()):
                    continue
                tensor, axname, axval = m.group(1), m.group(2), m.group(3)
                st = _mutation_site_from_span(
                    line_no,
                    line,
                    m.start(),
                    m.end(),
                    f"redtl:{tensor}:{axname}:{axval}",
                )
                if st:
                    sites.append(st)
            for m in self._TORCH_SUM.finditer(code):
                if not _index_before_outside_string_comment(line, m.start()):
                    continue
                tensor, dimv = m.group(1), m.group(2)
                st = _mutation_site_from_span(
                    line_no,
                    line,
                    m.start(),
                    m.end(),
                    f"redtorch:{tensor}:{dimv}",
                )
                if st:
                    sites.append(st)
        sites.sort(key=lambda s: (s.line_start, s.col_start))
        return sites

    def apply(self, source: str, site: MutationSite) -> str:
        nt = site.node_type
        if nt.startswith("redtl:"):
            parts = nt.split(":", 3)
            if len(parts) < 4:
                return source
            _, tensor, axname, axval = parts[0], parts[1], parts[2], parts[3]
            new_call = f"tl.sum({tensor}[::-1], {axname}={axval})"
            return _replace_at_columns(source, site, new_call)
        if nt.startswith("redtorch:"):
            parts = nt.split(":", 2)
            if len(parts) < 3:
                return source
            tensor, dimv = parts[1], parts[2]
            new_call = f"torch.sum(torch.flip({tensor}, ({dimv},)), dim={dimv})"
            return _replace_at_columns(source, site, new_call)
        return source


class InitModify(MutationOperator):
    name = "init_modify"
    category = "C"
    description = "Weaken min/max reduction initializers (inf / -inf)"

    # Python: float('-inf'), float('inf')
    _FLOAT_NEG_INF = re.compile(
        r"float\s*\(\s*(['\"])-inf\1\s*\)"
    )
    _FLOAT_POS_INF = re.compile(
        r"float\s*\(\s*(['\"])\+?inf\1\s*\)"
    )
    _TL_FULL_NEG_INF = re.compile(
        r"\btl\.full\s*\(\s*[^,]+,\s*float\s*\(\s*(['\"])-inf\1\s*\)\s*,"
    )

    # CUDA: -INFINITY, INFINITY  (from <math.h>)
    _CUDA_NEG_INFINITY = re.compile(r"(?<!\w)-\s*INFINITY\b")
    _CUDA_POS_INFINITY = re.compile(r"(?<![-\w])INFINITY\b")
    # CUDA: -FLT_MAX, FLT_MAX  (from <cfloat>)
    _CUDA_NEG_FLT_MAX = re.compile(r"(?<!\w)-\s*FLT_MAX\b")
    _CUDA_POS_FLT_MAX = re.compile(r"(?<![-\w])FLT_MAX\b")
    # CUDA: HUGE_VALF, -HUGE_VALF
    _CUDA_NEG_HUGE = re.compile(r"(?<!\w)-\s*HUGE_VALF?\b")
    _CUDA_POS_HUGE = re.compile(r"(?<![-\w])HUGE_VALF?\b")

    def find_sites(self, source: str, tree=None) -> List[MutationSite]:
        sites: List[MutationSite] = []
        for line_no, raw in enumerate(source.splitlines(), start=1):
            code, _ = _line_code_and_comment_start(raw)
            line = raw
            # --- Python patterns ---
            for m in self._FLOAT_NEG_INF.finditer(code):
                if not _index_before_outside_string_comment(line, m.start()):
                    continue
                for nt in ("init:neginf:-1e10", "init:neginf:0"):
                    st = _mutation_site_from_span(line_no, line, m.start(), m.end(), nt)
                    if st:
                        sites.append(st)
            for m in self._FLOAT_POS_INF.finditer(code):
                if not _index_before_outside_string_comment(line, m.start()):
                    continue
                for nt in ("init:posinf:1e10", "init:posinf:0"):
                    st = _mutation_site_from_span(line_no, line, m.start(), m.end(), nt)
                    if st:
                        sites.append(st)
            for m in self._TL_FULL_NEG_INF.finditer(code):
                if not _index_before_outside_string_comment(line, m.start()):
                    continue
                open_paren = code.find("(", m.start())
                span = _balanced_paren_span(code, open_paren) if open_paren != -1 else None
                if span is None:
                    continue
                for nt in ("init:tl_full_neginf:0", "init:tl_full_neginf:-1e10"):
                    st = _mutation_site_from_span(line_no, line, span[0], span[1], nt)
                    if st:
                        sites.append(st)

            # --- CUDA C++ patterns ---
            cuda_code = _strip_cuda_comment(raw)
            for rx, nt_prefix, replacements in [
                (self._CUDA_NEG_INFINITY, "init:cuda_neginf",
                 [("0.0f", ":0"), ("-1e10f", ":-1e10")]),
                (self._CUDA_POS_INFINITY, "init:cuda_posinf",
                 [("0.0f", ":0"), ("1e10f", ":1e10")]),
                (self._CUDA_NEG_FLT_MAX, "init:cuda_neg_flt_max",
                 [("0.0f", ":0"), ("-1e10f", ":-1e10")]),
                (self._CUDA_POS_FLT_MAX, "init:cuda_pos_flt_max",
                 [("0.0f", ":0"), ("1e10f", ":1e10")]),
                (self._CUDA_NEG_HUGE, "init:cuda_neg_huge",
                 [("0.0f", ":0"), ("-1e10f", ":-1e10")]),
                (self._CUDA_POS_HUGE, "init:cuda_pos_huge",
                 [("0.0f", ":0"), ("1e10f", ":1e10")]),
            ]:
                for m in rx.finditer(cuda_code):
                    for _repl_val, nt_suffix in replacements:
                        sites.append(MutationSite(
                            line_start=line_no, line_end=line_no,
                            col_start=m.start(), col_end=m.end(),
                            original_code=m.group(0),
                            node_type=nt_prefix + nt_suffix,
                        ))
        sites.sort(key=lambda s: (s.line_start, s.col_start, s.node_type))
        return sites

    def apply(self, source: str, site: MutationSite) -> str:
        nt = site.node_type
        # --- Python ---
        if nt == "init:neginf:-1e10":
            return _replace_at_columns(source, site, "float('-1e10')")
        if nt == "init:neginf:0":
            return _replace_at_columns(source, site, "0.0")
        if nt == "init:posinf:1e10":
            return _replace_at_columns(source, site, "float('1e10')")
        if nt == "init:posinf:0":
            return _replace_at_columns(source, site, "0.0")
        if nt == "init:tl_full_neginf:0":
            frag = site.original_code
            new_f = re.sub(
                r"float\s*\(\s*(['\"])-inf\1\s*\)", "0.0", frag, count=1,
            )
            return _replace_at_columns(source, site, new_f)
        if nt == "init:tl_full_neginf:-1e10":
            frag = site.original_code
            new_f = re.sub(
                r"float\s*\(\s*(['\"])-inf\1\s*\)", "float('-1e10')", frag, count=1,
            )
            return _replace_at_columns(source, site, new_f)
        # --- CUDA: extract replacement value from node_type suffix ---
        if nt.startswith("init:cuda_"):
            parts = nt.rsplit(":", 1)
            if len(parts) == 2:
                val = parts[1]
                repl_map = {"0": "0.0f", "1e10": "1e10f", "-1e10": "-1e10f"}
                replacement = repl_map.get(val)
                if replacement:
                    return _replace_at_columns(source, site, replacement)
        return source
