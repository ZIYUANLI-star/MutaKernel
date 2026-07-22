"""Static fault-class site fingerprinting for candidate kernels (M3, Mode B).

Zero-execution analysis: only the mutation operators' ``find_sites()`` static
scanners run (regex + AST).  No probe is compiled or executed.  The output
tells the online validator which fault classes plausibly apply to a candidate
so that stress policies can be *prioritised* (never *excluded*; see
方法V2_08 §3.2).

Schema: MutakernelV2/MutakernelV2方法修正/方法V2_00 §3.7 (SiteFingerprint).
"""

from __future__ import annotations

import ast
import hashlib
import inspect
from functools import lru_cache
from typing import Any, Dict, Optional

from .fault_classes import (
    OPERATOR_TO_FAULT_CLASS,
    TAXONOMY_VERSION,
    is_prior_equivalent_node_type,
)
from .operators import get_all_operators


@lru_cache(maxsize=1)
def fingerprint_version() -> str:
    """Content hash binding fingerprints to the exact operator implementations.

    Any change to an operator's source or to the fault-class taxonomy changes
    this version, which forces a joint version bump with FaultToStressMap
    consumers (方法V2_08 §七.1).
    """
    hasher = hashlib.sha256()
    hasher.update(TAXONOMY_VERSION.encode("utf-8"))
    for name in sorted(OPERATOR_TO_FAULT_CLASS):
        hasher.update(name.encode("utf-8"))
        hasher.update(OPERATOR_TO_FAULT_CLASS[name].encode("utf-8"))
    for operator in sorted(get_all_operators(), key=lambda op: op.name):
        try:
            source = inspect.getsource(type(operator))
        except (OSError, TypeError):  # pragma: no cover - frozen/embedded envs
            source = f"<unavailable:{operator.name}>"
        hasher.update(operator.name.encode("utf-8"))
        hasher.update(source.encode("utf-8"))
    return hasher.hexdigest()[:16]


def build_site_fingerprint(source: str, subject_id: str = "") -> Dict[str, Any]:
    """Build a deterministic fault-class site fingerprint for ``source``.

    The fingerprint never raises for malformed candidate code: a scanner
    failure is recorded per operator and treated as "no informative sites"
    (conservative: absence of evidence, not evidence of absence).
    """
    tree: Optional[ast.Module]
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None

    sites: Dict[str, Dict[str, Any]] = {}
    scan_errors: Dict[str, str] = {}
    present = set()

    for operator in sorted(get_all_operators(), key=lambda op: op.name):
        try:
            found = operator.find_sites(source, tree)
        except Exception as exc:  # noqa: BLE001 - fingerprinting must not crash validation
            scan_errors[operator.name] = f"{type(exc).__name__}: {exc}"[:200]
            found = []
        informative = [
            site for site in found
            if not is_prior_equivalent_node_type(site.node_type)
        ]
        sites[operator.name] = {
            "count": len(informative),
            "total_sites": len(found),
            "node_types": sorted({site.node_type for site in informative}),
        }
        if informative:
            present.add(OPERATOR_TO_FAULT_CLASS[operator.name])

    fingerprint: Dict[str, Any] = {
        "subject_id": subject_id,
        "fingerprint_version": fingerprint_version(),
        "sites": sites,
        "fault_classes_present": sorted(present),
    }
    if scan_errors:
        fingerprint["scan_errors"] = scan_errors
    return fingerprint
