#!/usr/bin/env python3
"""Fix absolute paths in registry.json files to relative paths.

This allows the project to be moved to another machine without breaking paths.
"""
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def fix_registry(reg_path: Path) -> int:
    """Fix reference_file paths in a registry.json: make relative + use forward slashes.
    Returns count fixed."""
    if not reg_path.exists():
        print(f"  SKIP: {reg_path} not found")
        return 0

    with open(reg_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fixed = 0
    for entry in data:
        ref = entry.get("reference_file", "")
        original = ref

        # Convert absolute paths to relative
        if os.path.isabs(ref):
            try:
                ref = str(Path(ref).relative_to(PROJECT_ROOT))
            except ValueError:
                parts = Path(ref).parts
                idx = None
                for i, p in enumerate(parts):
                    if p == "external_benchmarks":
                        idx = i
                        break
                if idx is not None:
                    ref = str(Path(*parts[idx:]))

        # Always normalize to forward slashes for Linux/WSL compatibility
        ref = ref.replace("\\", "/")

        if ref != original:
            entry["reference_file"] = ref
            fixed += 1

    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  {reg_path.name}: {len(data)} entries, {fixed} paths fixed")
    return fixed


def main():
    ext_dir = PROJECT_ROOT / "external_benchmarks"
    registries = [
        ext_dir / "cuda_l1" / "registry.json",
        ext_dir / "ai_cuda_engineer" / "registry.json",
        ext_dir / "tritonbench_g" / "registry.json",
    ]

    total_fixed = 0
    for reg in registries:
        total_fixed += fix_registry(reg)

    print(f"\nTotal paths fixed: {total_fixed}")


if __name__ == "__main__":
    main()
