"""Category A: arithmetic and numeric-constant mutation operators for GPU kernel testing."""

from __future__ import annotations

import ast
import io
import math
import re
import tokenize
from typing import List, Optional, Set, Tuple

from ...models import MutationSite
from .base import (
    MutationOperator,
    _find_pattern_sites,
    _replace_in_source,
    _replace_at_columns,
    _strip_cuda_comment,
)

_Pos = Tuple[int, int]

# Characters that, when appearing immediately before an operator (ignoring
# whitespace), indicate the operator is binary rather than unary.
_BINARY_PREV_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "_)]"
)

_CUDA_ARITH_NODE = {"+": "cuda_Add", "-": "cuda_Sub", "*": "cuda_Mult", "/": "cuda_Div"}

_C_TYPE_KEYWORDS = frozenset({
    "float", "double", "int", "char", "void", "short", "long", "unsigned",
    "signed", "bool", "const", "volatile", "__half", "half", "scalar_t",
    "size_t", "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
})


def _preceding_word(code: str, end: int) -> str:
    """Extract the identifier-like word ending at *end* (inclusive)."""
    if end < 0 or not (code[end].isalnum() or code[end] == "_"):
        return ""
    j = end
    while j > 0 and (code[j - 1].isalnum() or code[j - 1] == "_"):
        j -= 1
    return code[j : end + 1]


def _find_cuda_binary_arith_sites(source: str) -> List[MutationSite]:
    """Find binary arithmetic operators (+, -, *, /) in CUDA C++ code.

    Uses a character-scanning heuristic: an operator is considered binary if
    (after skipping whitespace) it is preceded by an identifier, digit, closing
    paren or bracket.  Compound operators (++, --, ->, +=, //, etc.) are excluded.
    Preprocessor ``#include`` lines and pointer declarations are filtered out.
    """
    sites: List[MutationSite] = []
    for lineno, raw_line in enumerate(source.splitlines(), start=1):
        code = _strip_cuda_comment(raw_line)
        stripped = code.lstrip()
        if stripped.startswith("#include"):
            continue
        n = len(code)
        for i, c in enumerate(code):
            if c not in "+-*/":
                continue
            nxt = code[i + 1] if i + 1 < n else ""
            if c == "+" and nxt in "+=":
                continue
            if c == "-" and nxt in "-=>":
                continue
            if c == "*" and nxt in "*=":
                continue
            if c == "/" and nxt in "/=*":
                continue
            j = i - 1
            while j >= 0 and code[j] in " \t":
                j -= 1
            if j < 0 or code[j] not in _BINARY_PREV_CHARS:
                continue
            # Pointer declaration heuristic: ``type * ident`` or ``type *ident``
            if c == "*":
                pre_word = _preceding_word(code, j)
                if pre_word in _C_TYPE_KEYWORDS:
                    continue
            sites.append(
                MutationSite(
                    line_start=lineno,
                    line_end=lineno,
                    col_start=i,
                    col_end=i + 1,
                    original_code=c,
                    node_type=_CUDA_ARITH_NODE[c],
                )
            )
    return sites


# Template-related keywords whose trailing ``<`` is NOT a comparison.
_TEMPLATE_KEYWORDS = (
    "static_cast", "dynamic_cast", "const_cast", "reinterpret_cast",
    "data_ptr", "accessor", "packed_accessor", "template",
)

# Common C++ type names that appear before a closing ``>`` in templates.
_TEMPLATE_TYPES = (
    "float", "double", "int", "__half", "half", "at::Half",
    "at::BFloat16", "int32_t", "int64_t", "uint32_t", "uint64_t",
    "bool", "char", "short", "long", "unsigned", "size_t", "scalar_t",
    "c10::Half", "c10::BFloat16",
)

_CUDA_REL_NODE = {
    "<": "cuda_Lt", "<=": "cuda_LtE",
    ">": "cuda_Gt", ">=": "cuda_GtE",
    "==": "cuda_Eq", "!=": "cuda_NotEq",
}

_CUDA_REL_REPLACE = {
    "<": "<=", "<=": "<",
    ">": ">=", ">=": ">",
    "==": "!=", "!=": "==",
}


def _is_template_open(code: str, pos: int) -> bool:
    """Return True if ``<`` at *pos* is likely a C++ template bracket."""
    prefix = code[:pos].rstrip()
    for kw in _TEMPLATE_KEYWORDS:
        if prefix.endswith(kw):
            return True
    return False


def _is_template_close(code: str, pos: int) -> bool:
    """Return True if ``>`` at *pos* is likely a closing template bracket."""
    if pos + 1 < len(code) and code[pos + 1] == "(":
        return True
    prefix = code[:pos].rstrip()
    for tn in _TEMPLATE_TYPES:
        if prefix.endswith(tn):
            return True
    return False


def _find_cuda_relop_sites(source: str) -> List[MutationSite]:
    """Find relational operators in CUDA C++ code.

    Filters out ``<<`` / ``>>`` (bit-shift), ``<<<`` / ``>>>`` (kernel launch),
    ``#include <…>`` directives, and template angle brackets.
    """
    sites: List[MutationSite] = []
    for lineno, raw_line in enumerate(source.splitlines(), start=1):
        code = _strip_cuda_comment(raw_line)
        stripped = code.lstrip()
        if stripped.startswith("#include"):
            continue
        n = len(code)
        i = 0
        while i < n:
            two = code[i : i + 2]
            if two in ("<=", ">=", "==", "!="):
                sites.append(
                    MutationSite(
                        line_start=lineno,
                        line_end=lineno,
                        col_start=i,
                        col_end=i + 2,
                        original_code=two,
                        node_type=_CUDA_REL_NODE[two],
                    )
                )
                i += 2
                continue
            if code[i] == "<" and two not in ("<=", "<<"):
                # Skip third ``<`` in ``<<<`` kernel launch syntax
                if i > 0 and code[i - 1] == "<":
                    i += 1
                    continue
                if not _is_template_open(code, i):
                    sites.append(
                        MutationSite(
                            line_start=lineno,
                            line_end=lineno,
                            col_start=i,
                            col_end=i + 1,
                            original_code="<",
                            node_type=_CUDA_REL_NODE["<"],
                        )
                    )
            elif code[i] == ">" and two not in (">=", ">>"):
                # Skip third ``>`` in ``>>>`` — already handled by
                # _is_template_close (``>`` followed by ``(``) and
                # preceding ``>`` check.
                if i > 0 and code[i - 1] == ">":
                    i += 1
                    continue
                if not _is_template_close(code, i):
                    sites.append(
                        MutationSite(
                            line_start=lineno,
                            line_end=lineno,
                            col_start=i,
                            col_end=i + 1,
                            original_code=">",
                            node_type=_CUDA_REL_NODE[">"],
                        )
                    )
            i += 1
    return sites


def _parse_tree(source: str, tree: Optional[ast.Module]) -> Optional[ast.Module]:
    if tree is not None:
        return tree
    try:
        return ast.parse(source, mode="exec")
    except SyntaxError:
        return None


def _decorator_subtree_nodes(mod: ast.Module) -> Set[ast.AST]:
    covered: Set[ast.AST] = set()
    for node in ast.walk(mod):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for dec in node.decorator_list:
                for n in ast.walk(dec):
                    covered.add(n)
    return covered


def _pos_le(a: _Pos, b: _Pos) -> bool:
    return a[0] < b[0] or (a[0] == b[0] and a[1] <= b[1])


def _pos_lt(a: _Pos, b: _Pos) -> bool:
    return a[0] < b[0] or (a[0] == b[0] and a[1] < b[1])


def _expr_end(node: ast.expr) -> _Pos:
    el = getattr(node, "end_lineno", None) or node.lineno
    ec = getattr(node, "end_col_offset", None)
    if ec is None:
        return (node.lineno, node.col_offset)
    return (el, ec)


def _expr_start(node: ast.expr) -> _Pos:
    return (node.lineno, node.col_offset)


def _token_op_in_span(
    source: str,
    end_left: _Pos,
    start_right: _Pos,
    allowed_strings: Tuple[str, ...],
) -> Optional[tokenize.TokenInfo]:
    readline = io.StringIO(source).readline
    for tok in tokenize.generate_tokens(readline):
        if tok.type != tokenize.OP or tok.string not in allowed_strings:
            continue
        if tok.string == "/" and tok.line and tok.start[1] > 0:
            i = tok.start[1]
            if i + 1 < len(tok.line) and tok.line[i + 1] == "/":
                continue
        if tok.string == "*" and tok.line and tok.start[1] > 0:
            i = tok.start[1]
            if i + 1 < len(tok.line) and tok.line[i + 1] == "*":
                continue
        if not _pos_le(end_left, tok.start):
            continue
        if not _pos_lt(tok.end, start_right):
            continue
        return tok
    return None


_ARITH_CLASSES = (ast.Add, ast.Sub, ast.Mult, ast.Div)


def _arith_pair(op: ast.operator) -> Tuple[str, str]:
    if isinstance(op, ast.Add):
        return ("+", "-")
    if isinstance(op, ast.Sub):
        return ("-", "+")
    if isinstance(op, ast.Mult):
        return ("*", "/")
    if isinstance(op, ast.Div):
        return ("/", "*")
    raise TypeError(op)


class ArithReplace(MutationOperator):
    name = "arith_replace"
    category = "A"
    description = "Replace binary arithmetic operators (+→-, -→+, *→/, /→*)"

    def find_sites(self, source: str, tree: Optional[ast.Module] = None) -> List[MutationSite]:
        # --- Python/Triton path (AST-based) ---
        ast_sites: List[MutationSite] = []
        mod = _parse_tree(source, tree)
        if mod is not None:
            skip = _decorator_subtree_nodes(mod)
            for node in ast.walk(mod):
                if node in skip:
                    continue
                if not isinstance(node, ast.BinOp) or not isinstance(node.op, _ARITH_CLASSES):
                    continue
                old_s, _ = _arith_pair(node.op)
                end_l = _expr_end(node.left)
                start_r = _expr_start(node.right)
                tok = _token_op_in_span(source, end_l, start_r, (old_s,))
                if tok is None:
                    continue
                ls, cs = tok.start
                le, ce = tok.end
                ast_sites.append(
                    MutationSite(
                        line_start=ls,
                        line_end=le,
                        col_start=cs,
                        col_end=ce,
                        original_code=old_s,
                        node_type=type(node.op).__name__,
                    )
                )

        # --- CUDA C++ path (character scanning) ---
        cuda_sites = _find_cuda_binary_arith_sites(source)

        # Merge, dedup by position (AST sites take priority)
        seen = {(s.line_start, s.col_start) for s in ast_sites}
        for s in cuda_sites:
            if (s.line_start, s.col_start) not in seen:
                ast_sites.append(s)

        ast_sites.sort(key=lambda s: (s.line_start, s.col_start, s.line_end, s.col_end))
        return ast_sites

    def apply(self, source: str, site: MutationSite) -> str:
        new_s = {"+": "-", "-": "+", "*": "/", "/": "*"}.get(site.original_code, "")
        if not new_s:
            return source
        return _replace_at_columns(source, site, new_s)


_REL_PAIR: dict[type, Tuple[str, str]] = {
    ast.Lt: ("<", "<="),
    ast.LtE: ("<=", "<"),
    ast.Gt: (">", ">="),
    ast.GtE: (">=", ">"),
    ast.Eq: ("==", "!="),
    ast.NotEq: ("!=", "=="),
}


def _rel_strings(op: ast.cmpop) -> Optional[Tuple[str, str]]:
    return _REL_PAIR.get(type(op))


class RelOpReplace(MutationOperator):
    name = "relop_replace"
    category = "A"
    description = (
        "Replace relational operators "
        "(<→<=, <=→<, >→>=, >=→>, ==→!=, !=→==)"
    )

    def find_sites(self, source: str, tree: Optional[ast.Module] = None) -> List[MutationSite]:
        # --- Python/Triton path (AST-based) ---
        ast_sites: List[MutationSite] = []
        mod = _parse_tree(source, tree)
        if mod is not None:
            skip = _decorator_subtree_nodes(mod)
            for node in ast.walk(mod):
                if node in skip:
                    continue
                if not isinstance(node, ast.Compare):
                    continue
                parts: List[ast.expr] = [node.left, *node.comparators]
                for i, op in enumerate(node.ops):
                    pair = _rel_strings(op)
                    if pair is None:
                        continue
                    old_s, _ = pair
                    end_l = _expr_end(parts[i])
                    start_r = _expr_start(parts[i + 1])
                    tok = _token_op_in_span(source, end_l, start_r, (old_s,))
                    if tok is None:
                        continue
                    ls, cs = tok.start
                    le, ce = tok.end
                    ast_sites.append(
                        MutationSite(
                            line_start=ls,
                            line_end=le,
                            col_start=cs,
                            col_end=ce,
                            original_code=old_s,
                            node_type=type(op).__name__,
                        )
                    )

        # --- CUDA C++ path (character scanning, with template filtering) ---
        cuda_sites = _find_cuda_relop_sites(source)

        # Merge, dedup by position (AST sites take priority)
        seen = {(s.line_start, s.col_start) for s in ast_sites}
        for s in cuda_sites:
            if (s.line_start, s.col_start) not in seen:
                ast_sites.append(s)

        ast_sites.sort(key=lambda s: (s.line_start, s.col_start, s.line_end, s.col_end))
        return ast_sites

    def apply(self, source: str, site: MutationSite) -> str:
        # CUDA C++ path
        if site.node_type.startswith("cuda_"):
            new_s = _CUDA_REL_REPLACE.get(site.original_code)
            if new_s is None:
                return source
            return _replace_at_columns(source, site, new_s)

        # Python/Triton path (AST-based node_type)
        try:
            cls = getattr(ast, site.node_type)
        except AttributeError:
            return source
        pair = _REL_PAIR.get(cls)
        if pair is None:
            return source
        old_s, new_s = pair
        if site.original_code != old_s:
            return source
        return _replace_at_columns(source, site, new_s)


_DIGITS = r"\d(?:_?\d)*"
_OPT_DIGITS = rf"(?:{_DIGITS})?"
_INT_PATTERN = rf"\b(?!0\b|1\b){_DIGITS}\b(?!\.)"
_FLOAT_PATTERN = (
    rf"(?<![\w.])"
    rf"(?:{_DIGITS}\.{_OPT_DIGITS}(?:[eE][+-]?{_DIGITS})?|\.{_DIGITS}(?:[eE][+-]?{_DIGITS})?|"
    rf"{_DIGITS}[eE][+-]?{_DIGITS})"
    r"[fFlL]?"
    r"(?![\w.])"
)


_GPU_CONFIG_RE = re.compile(
    r'(?:'
    r'\b(?:BLOCK_?SIZE|block_?size|TILE_?SIZE|tile_?size|'
    r'WARP_?SIZE|warp_?size|num_warps|NUM_WARPS|'
    r'GRID_?SIZE|grid_?size|num_blocks|NUM_BLOCKS|'
    r'num_threads|NUM_THREADS|threads_per_block|blocks_per_grid|'
    r'threadIdx|blockIdx)\b'
    r'|'
    r'\b(?:blockDim|gridDim|dim3)\w*'
    r')'
)


def _is_gpu_config_line(source: str, line_idx: int) -> bool:
    """Return True if the line likely assigns or defines a GPU scheduling constant."""
    lines = source.splitlines()
    if line_idx < 0 or line_idx >= len(lines):
        return False
    return _GPU_CONFIG_RE.search(lines[line_idx]) is not None


def _format_mutation_float(x: float) -> str:
    if not math.isfinite(x):
        return repr(x)
    ax = abs(x)
    if 1e-6 <= ax < 1e7:
        return f"{x:.12g}"
    return repr(x)


class ConstPerturb(MutationOperator):
    name = "const_perturb"
    category = "A"
    description = "Perturb numeric literals (integer ±1; float ×1.01 or ×0.99)"

    def find_sites(self, source: str, tree: Optional[ast.Module] = None) -> List[MutationSite]:
        sites: List[MutationSite] = []
        for s in _find_pattern_sites(source, _INT_PATTERN, "literal_int"):
            if _is_gpu_config_line(source, s.line_start - 1):
                continue
            sites.append(
                MutationSite(
                    line_start=s.line_start,
                    line_end=s.line_end,
                    col_start=s.col_start,
                    col_end=s.col_end,
                    original_code=s.original_code,
                    node_type="const:int+1",
                )
            )
            sites.append(
                MutationSite(
                    line_start=s.line_start,
                    line_end=s.line_end,
                    col_start=s.col_start,
                    col_end=s.col_end,
                    original_code=s.original_code,
                    node_type="const:int-1",
                )
            )
        for s in _find_pattern_sites(source, _FLOAT_PATTERN, "literal_float"):
            raw = s.original_code
            body = raw[:-1] if raw and raw[-1] in "fFlL" else raw
            try:
                v = float(body)
            except ValueError:
                continue
            if v == 0.0 or v == 1.0:
                continue
            sites.append(
                MutationSite(
                    line_start=s.line_start,
                    line_end=s.line_end,
                    col_start=s.col_start,
                    col_end=s.col_end,
                    original_code=raw,
                    node_type="const:float*1.01",
                )
            )
            sites.append(
                MutationSite(
                    line_start=s.line_start,
                    line_end=s.line_end,
                    col_start=s.col_start,
                    col_end=s.col_end,
                    original_code=raw,
                    node_type="const:float*0.99",
                )
            )
        sites.sort(key=lambda x: (x.line_start, x.col_start, x.node_type))
        return sites

    def apply(self, source: str, site: MutationSite) -> str:
        nt = site.node_type
        text = site.original_code
        if nt == "const:int+1":
            try:
                n = int(text)
            except ValueError:
                return source
            return _replace_in_source(source, site, text, str(n + 1))
        if nt == "const:int-1":
            try:
                n = int(text)
            except ValueError:
                return source
            return _replace_in_source(source, site, text, str(n - 1))
        if nt == "const:float*1.01":
            new_t = self._perturb_float_text(text, 1.01)
            if new_t is None:
                return source
            return _replace_in_source(source, site, text, new_t)
        if nt == "const:float*0.99":
            new_t = self._perturb_float_text(text, 0.99)
            if new_t is None:
                return source
            return _replace_in_source(source, site, text, new_t)
        return source

    @staticmethod
    def _perturb_float_text(raw: str, factor: float) -> Optional[str]:
        suffix = ""
        body = raw
        if raw and raw[-1] in "fFlL":
            suffix = raw[-1]
            body = raw[:-1]
        try:
            v = float(body)
        except ValueError:
            return None
        nv = v * factor
        return _format_mutation_float(nv) + suffix
