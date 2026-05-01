#!/usr/bin/env python3
"""为已完成的 JSON 结果补充 mutated_code 和 node_type 信息。

原理:
  对每个 JSON 文件，加载对应 kernel 源码，重跑 generate_mutants()
  通过 mutant.id 精确匹配，将 mutated_code 和 site.node_type 写回 JSON。

  不需要 CUDA / GPU —— 整个过程是纯 Python AST + 字符串操作。
"""
import ast
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import KernelInfo, Mutant, MutationSite
from src.mutengine.operators.base import get_all_operators

BEST_KERNELS_FILE = PROJECT_ROOT / "best_kernels.json"
RESULTS_DIR = PROJECT_ROOT / "full_block12_results" / "details"

SAMPLE_PER_OP = 3
SEED = 42


def load_kernel_source(kernel_path: str) -> str:
    with open(kernel_path, "r", encoding="utf-8") as f:
        return f.read()


def regenerate_all_mutants(source: str, kernel_id: str):
    """重跑所有算子的 find_sites + apply，返回 {mutant_id: Mutant}"""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None

    operators = get_all_operators()
    id_to_mutant = {}

    for op in operators:
        try:
            sites = op.find_sites(source, tree)
            for i, site in enumerate(sites):
                mutated = op.apply(source, site)
                if mutated == source:
                    continue
                mid = f"{kernel_id}__{op.name}__{i}"
                id_to_mutant[mid] = {
                    "mutated_code": mutated,
                    "node_type": site.node_type,
                    "original_code_full": site.original_code,
                }
        except Exception as e:
            print(f"  [WARN] Operator {op.name} failed: {e}")

    return id_to_mutant


def enrich_json_file(json_path: Path, best_kernels: dict) -> bool:
    """给单个 JSON 文件补充信息。返回是否有修改。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    kernel_key = json_path.stem
    if kernel_key not in best_kernels:
        print(f"  [SKIP] {kernel_key} not in best_kernels.json")
        return False

    kernel_path = best_kernels[kernel_key]["kernel_path"]
    if not os.path.exists(kernel_path):
        print(f"  [SKIP] source not found: {kernel_path}")
        return False

    source = load_kernel_source(kernel_path)
    kernel_id = kernel_key
    id_to_info = regenerate_all_mutants(source, kernel_id)

    modified = False
    matched = 0
    unmatched = 0

    for mutant_entry in data.get("mutants", []):
        mid = mutant_entry["id"]
        if mid in id_to_info:
            info = id_to_info[mid]
            mutant_entry["mutated_code"] = info["mutated_code"]
            mutant_entry["site"]["node_type"] = info["node_type"]
            mutant_entry["site"]["original_code"] = info["original_code_full"][:500]
            matched += 1
            modified = True
        else:
            unmatched += 1

    if modified:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  {kernel_key}: matched={matched}, unmatched={unmatched}")
    return modified


def main():
    with open(BEST_KERNELS_FILE, "r", encoding="utf-8") as f:
        best_kernels = json.load(f)

    json_files = sorted(RESULTS_DIR.glob("*.json"))
    print(f"Found {len(json_files)} JSON result files")
    print(f"Best kernels: {len(best_kernels)} entries")
    print()

    total_modified = 0
    for jf in json_files:
        try:
            if enrich_json_file(jf, best_kernels):
                total_modified += 1
        except Exception as e:
            print(f"  [ERROR] {jf.stem}: {e}")

    print(f"\nDone. Modified {total_modified}/{len(json_files)} files.")


if __name__ == "__main__":
    main()
