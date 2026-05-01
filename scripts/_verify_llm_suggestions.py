"""Verify LLM-suggested inputs against original and mutant kernels.

Can be used as:
  1. A module imported by _pilot_llm20.py (main use case)
  2. A standalone script for debugging

The verification runs _stress_worker.py in llm_verify mode via subprocess.
No WSL/conda hardcoding — uses the same Python interpreter as the caller.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = SCRIPT_DIR / "_stress_worker.py"
BLOCK12_DIR = PROJECT_ROOT / "full_block12_results" / "details"
BEST_KERNELS_FILE = PROJECT_ROOT / "best_kernels.json"

sys.path.insert(0, str(PROJECT_ROOT))
from src.stress.llm_analyzer import validate_suggested_code


def P(msg):
    print(msg, flush=True)


def verify_one_mutant(
    kernel_code: str,
    mutated_code: str,
    test_code: str,
    problem_file: str,
    python_exe: Optional[str] = None,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    device: str = "cuda",
    timeout: int = 180,
) -> Dict:
    """Run verification of LLM-suggested input code.

    Returns dict with keys: killed, ref_ok, original_ok, mutant_ok,
    diff_summary, error, time_ms.
    """
    safety = validate_suggested_code(test_code)
    if safety:
        return {"killed": False, "error": f"safety_check: {safety}",
                "ref_ok": False, "original_ok": False, "mutant_ok": False}

    if python_exe is None:
        python_exe = sys.executable

    cfg = {
        "mode": "llm_verify",
        "problem_file": problem_file,
        "kernel_code": kernel_code,
        "mutated_code": mutated_code,
        "test_inputs_code": test_code,
        "atol": atol,
        "rtol": rtol,
        "device": device,
    }

    cfg_fd, cfg_path = tempfile.mkstemp(suffix=".json", prefix="vcfg_")
    res_fd, res_path = tempfile.mkstemp(suffix=".json", prefix="vres_")
    os.close(cfg_fd)
    os.close(res_fd)

    try:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False)

        cmd = [python_exe, str(WORKER_SCRIPT), cfg_path, res_path]
        proc = subprocess.run(
            cmd, capture_output=True, timeout=timeout, text=True,
            cwd=str(PROJECT_ROOT),
        )

        if os.path.exists(res_path) and os.path.getsize(res_path) > 2:
            with open(res_path, encoding="utf-8") as f:
                result = json.load(f)
            if "killed" not in result:
                result["killed"] = (
                    result.get("ref_ok", False) and
                    result.get("original_ok", False) and
                    not result.get("mutant_ok", True)
                )
            return result

        stderr = proc.stderr[:500] if proc.stderr else ""
        return {"killed": False, "error": f"worker_no_output: {stderr}",
                "ref_ok": False, "original_ok": False, "mutant_ok": False}

    except subprocess.TimeoutExpired:
        return {"killed": False, "error": "timeout",
                "ref_ok": False, "original_ok": False, "mutant_ok": False}
    except Exception as e:
        return {"killed": False, "error": f"verify_error: {str(e)[:300]}",
                "ref_ok": False, "original_ok": False, "mutant_ok": False}
    finally:
        for p in [cfg_path, res_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


def load_mutant_detail(kernel_name: str, mutant_id: str) -> Optional[Dict]:
    detail_path = BLOCK12_DIR / f"{kernel_name}.json"
    if not detail_path.exists():
        return None
    data = json.loads(detail_path.read_text(encoding="utf-8"))
    for m in data.get("mutants", []):
        if m["id"] == mutant_id:
            return m
    return None


def resolve_problem_file(kernel_name: str, best_kernels: Dict) -> Optional[str]:
    """Resolve the problem file path for a kernel."""
    bk = best_kernels.get(kernel_name, {})
    level_raw = bk.get("level", 1)
    pid = bk.get("problem_id", "")
    if not pid:
        parts = kernel_name.split("_P")
        if len(parts) == 2:
            pid = parts[1]

    # level may be stored as "L1"/"L2" or as int 1/2
    level_str = str(level_raw)
    if level_str.startswith("L"):
        level_num = level_str[1:]
    else:
        level_num = level_str

    kb_root = "/home/kbuser/projects/KernelBench-0"
    problem_dir = f"{kb_root}/KernelBench/level{level_num}"
    pattern = f"{problem_dir}/{pid}_*.py"

    import glob
    matches = glob.glob(pattern)
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Standalone entry point (for debugging)
# ---------------------------------------------------------------------------
def main():
    from collections import defaultdict

    LLM_RESULT_DIR = PROJECT_ROOT / "llm_analysis_results"
    details_dir = LLM_RESULT_DIR / "details"

    if not details_dir.exists():
        P("ERROR: No LLM results found.")
        return

    best_kernels = {}
    if BEST_KERNELS_FILE.exists():
        best_kernels = json.loads(BEST_KERNELS_FILE.read_text(encoding="utf-8"))

    candidates = []
    for jf in sorted(details_dir.glob("*.json")):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if any(r.get("killed") for r in d.get("rounds", [])):
            continue
        code = d.get("suggested_test_code")
        if code and not validate_suggested_code(code):
            candidates.append(d)

    P(f"Found {len(candidates)} candidates to verify")

    for idx, d in enumerate(candidates):
        mid = d["mutant_id"]
        kname = d["kernel_name"]
        P(f"  [{idx+1}/{len(candidates)}] {mid}")

        detail = load_mutant_detail(kname, mid)
        if not detail:
            P(f"    SKIP: no detail")
            continue

        pf = resolve_problem_file(kname, best_kernels)
        if not pf:
            P(f"    SKIP: no problem file")
            continue

        bk = best_kernels.get(kname, {})
        kpath = Path(bk.get("kernel_path", ""))
        kernel_code = kpath.read_text(encoding="utf-8") if kpath.exists() else detail.get("kernel_code", "")
        mutated_code = detail.get("mutated_code", "")

        result = verify_one_mutant(kernel_code, mutated_code,
                                    d["suggested_test_code"], pf)
        P(f"    Result: killed={result.get('killed')}")


if __name__ == "__main__":
    main()
