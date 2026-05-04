"""KernelBench 集成桥接模块。

职责:
1. 从 KernelBench 的目录结构中加载问题定义
2. 从 runs 目录读取 eval_results.json 和已生成的 kernel 源码
3. 从 iterations 目录读取迭代历史，提取失败/正确配对
4. 提取 Model / ModelNew / get_inputs / get_init_inputs
5. 封装为 MutaKernel 所需的 KernelInfo 格式
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..models import KernelInfo

logger = logging.getLogger(__name__)


@dataclass
class IterationPair:
    """迭代历史中的一个「编译通过但运行错误」vs「最佳正确」配对。"""
    problem_id: int
    level: int
    failed_turn: int
    best_correct_turn: int
    failed_kernel_path: str
    correct_kernel_path: str
    best_speedup: float


def _load_module_from_path(filepath: str, module_name: str) -> Any:
    """从文件路径加载 Python 模块。"""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {filepath}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class KernelBenchBridge:
    """KernelBench 数据读取桥接。

    实际 KernelBench 目录结构:
    KernelBench-0/
    ├── KernelBench/level{1,2,3,4}/
    │   └── {id}_{name}.py                          # 问题定义
    └── runs/{run_name}/
        ├── eval_results.json                        # 评测结果 (key="1","2",...)
        └── level_{level}_problem_{id}_sample_0_kernel.py  # 最终生成的 kernel
    """

    def __init__(
        self,
        kernelbench_root: str | Path,
        run_name: str = "iter_full_l{level}_caesar_paper_v2",
    ):
        self.root = Path(kernelbench_root)
        self.run_name_template = run_name
        self.problems_root = self.root / "KernelBench"

    def get_problem_dir(self, level: int) -> Path:
        return self.problems_root / f"level{level}"

    def get_run_dir(self, level: int) -> Path:
        run_name = self.run_name_template.format(level=level)
        return self.root / "runs" / run_name

    def load_eval_results(self, level: int) -> Dict[str, Any]:
        """加载 eval_results.json。"""
        path = self.get_run_dir(level) / "eval_results.json"
        if not path.exists():
            logger.warning(f"eval_results.json not found: {path}")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_correct_kernels(self, level: int) -> List[Dict[str, Any]]:
        """列出通过 torch.allclose 验证的所有 kernel。"""
        results = self.load_eval_results(level)
        correct = []
        for problem_key, info in results.items():
            if info.get("correctness", False):
                correct.append({
                    "problem_key": problem_key,
                    "level": level,
                    **info,
                })
        return correct

    def list_failed_kernels(self, level: int) -> List[Dict[str, Any]]:
        """列出未通过验证的 kernel（用于 Realism 验证）。"""
        results = self.load_eval_results(level)
        failed = []
        for problem_key, info in results.items():
            if not info.get("correctness", False):
                failed.append({
                    "problem_key": problem_key,
                    "level": level,
                    **info,
                })
        return failed

    def find_problem_file(self, level: int, problem_key: str) -> Optional[Path]:
        """根据 problem_key 找到问题定义文件。"""
        problem_dir = self.get_problem_dir(level)
        if not problem_dir.exists():
            return None

        problem_id = self._extract_problem_id(problem_key)
        if problem_id is None:
            return None

        for f in problem_dir.iterdir():
            if f.name.startswith(f"{problem_id}_") and f.suffix == ".py":
                return f
        return None

    def find_generated_kernel(self, level: int, problem_key: str) -> Optional[Path]:
        """找到最终生成的 kernel 文件。

        实际文件名格式: level_{level}_problem_{id}_sample_0_kernel.py
        直接位于 run_dir 下，不在子目录中。
        """
        run_dir = self.get_run_dir(level)
        problem_id = self._extract_problem_id(problem_key)
        if problem_id is None:
            return None

        kernel_file = run_dir / f"level_{level}_problem_{problem_id}_sample_0_kernel.py"
        if kernel_file.exists():
            return kernel_file

        for f in run_dir.glob(f"level_{level}_problem_{problem_id}_*.py"):
            return f

        return None

    def load_kernel_info(self, level: int, problem_key: str) -> Optional[KernelInfo]:
        """加载一个 kernel 的完整信息。"""
        gen_path = self.find_generated_kernel(level, problem_key)
        if gen_path is None:
            logger.warning(f"Generated kernel not found: L{level}/{problem_key}")
            return None

        ref_path = self.find_problem_file(level, problem_key)
        if ref_path is None:
            logger.warning(f"Problem file not found: L{level}/{problem_key}")
            return None

        try:
            kernel_code = gen_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Cannot read {gen_path}: {e}")
            return None

        problem_id = self._extract_problem_id(problem_key) or 0
        language = "cuda" if self._detect_cuda(kernel_code) else "triton"

        return KernelInfo(
            problem_id=problem_id,
            level=level,
            problem_name=problem_key,
            source_path=str(gen_path),
            kernel_code=kernel_code,
            reference_module_path=str(ref_path),
            language=language,
        )

    def load_runtime_components(
        self, kernel: KernelInfo
    ) -> Tuple[Any, Callable, Callable]:
        """加载运行时组件：参考模块、get_inputs、get_init_inputs。

        Returns:
            (ref_module, get_inputs_fn, get_init_inputs_fn)
        """
        ref_module = _load_module_from_path(
            kernel.reference_module_path,
            f"ref_L{kernel.level}_P{kernel.problem_id}",
        )

        get_inputs_fn = getattr(ref_module, "get_inputs", None)
        get_init_inputs_fn = getattr(ref_module, "get_init_inputs", None)

        if get_inputs_fn is None:
            raise AttributeError(
                f"Problem file missing get_inputs(): {kernel.reference_module_path}"
            )
        if get_init_inputs_fn is None:
            get_init_inputs_fn = lambda: []

        return ref_module, get_inputs_fn, get_init_inputs_fn

    def load_all_correct_kernels(self, levels: List[int] = None) -> List[KernelInfo]:
        """加载所有通过验证的 kernel。"""
        if levels is None:
            levels = [1, 2]

        all_kernels = []
        for level in levels:
            correct_list = self.list_correct_kernels(level)
            logger.info(f"Level {level}: {len(correct_list)} correct kernels")
            for entry in correct_list:
                ki = self.load_kernel_info(level, entry["problem_key"])
                if ki is not None:
                    all_kernels.append(ki)

        logger.info(f"Total loaded: {len(all_kernels)} kernels")
        return all_kernels

    def _extract_problem_id(self, problem_key: str) -> Optional[int]:
        """从 problem_key 中提取数字 ID。"""
        m = re.match(r"(\d+)", problem_key)
        return int(m.group(1)) if m else None

    def _extract_iter_number(self, filename: str) -> int:
        """从 generated_N.py 文件名中提取迭代号。"""
        m = re.search(r"generated_(\d+)", filename)
        return int(m.group(1)) if m else -1

    def _detect_cuda(self, source: str) -> bool:
        """简单检测是否为 CUDA kernel。"""
        cuda_indicators = ["__global__", "__device__", "load_inline", "cuda_source"]
        return sum(1 for ind in cuda_indicators if ind in source) >= 2

    # ------------------------------------------------------------------
    # 迭代历史分析（RealismGuard 数据源）
    # ------------------------------------------------------------------

    def list_iteration_pairs(self, level: int) -> List[IterationPair]:
        """从迭代历史中提取「编译通过但运行错误 vs 最佳正确」配对。

        策略（基于实际 KernelBench 迭代行为）:
        - 迭代过程是错/对交叉的，不是线性递进
        - "对"选 correctness=True 且 speedup 最高的 turn
        - "错"筛选编译通过（kernel 文件存在）但 correctness=False 的 turn
        - 每个"错"都和同一问题的"最佳对"配对，产出一个 diff 样本
        """
        run_dir = self.get_run_dir(level)
        iter_dir = run_dir / "iterations"
        if not iter_dir.exists():
            logger.warning(f"Iterations directory not found: {iter_dir}")
            return []

        pairs: List[IterationPair] = []

        for prob_dir in sorted(iter_dir.iterdir()):
            if not prob_dir.is_dir():
                continue
            summary_file = prob_dir / "problem_summary.json"
            if not summary_file.exists():
                continue

            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.debug(f"Cannot read {summary_file}: {e}")
                continue

            pid = data.get("problem_id")
            if pid is None:
                pid = self._extract_problem_id(prob_dir.name)
            if pid is None:
                continue
            pid = int(pid)

            turns = data.get("turns", [])
            if not turns:
                continue

            best_correct = self._find_best_correct_turn(turns, run_dir, prob_dir)
            if best_correct is None:
                continue

            correct_path, best_speedup, best_turn_num = best_correct

            for turn_info in turns:
                ev = turn_info.get("eval", {})
                if ev.get("correctness", False):
                    continue

                failed_path = self._resolve_turn_kernel_path(
                    turn_info, run_dir, prob_dir,
                )
                if failed_path is None:
                    continue

                pairs.append(IterationPair(
                    problem_id=pid,
                    level=level,
                    failed_turn=turn_info.get("turn", -1),
                    best_correct_turn=best_turn_num,
                    failed_kernel_path=str(failed_path),
                    correct_kernel_path=str(correct_path),
                    best_speedup=best_speedup,
                ))

        logger.info(
            f"Level {level}: found {len(pairs)} iteration pairs "
            f"from {iter_dir}"
        )
        return pairs

    def _find_best_correct_turn(
        self,
        turns: List[Dict],
        run_dir: Path,
        prob_dir: Path,
    ) -> Optional[Tuple[Path, float, int]]:
        """在 turns 中找到 correctness=True 且 speedup 最高的 turn。

        Returns:
            (kernel_path, speedup, turn_number) or None
        """
        best_path: Optional[Path] = None
        best_speedup = -1.0
        best_turn_num = -1

        for turn_info in turns:
            ev = turn_info.get("eval", {})
            if not ev.get("correctness", False):
                continue

            spd = turn_info.get("speedup")
            if spd is None:
                spd = ev.get("metadata", {}).get("observed_speedup", 0)
            if spd is None:
                spd = 0.0
            spd = float(spd)

            if spd > best_speedup:
                kp = self._resolve_turn_kernel_path(turn_info, run_dir, prob_dir)
                if kp is not None:
                    best_speedup = spd
                    best_path = kp
                    best_turn_num = turn_info.get("turn", -1)

        if best_path is None:
            return None
        return best_path, best_speedup, best_turn_num

    def _resolve_turn_kernel_path(
        self,
        turn_info: Dict,
        run_dir: Path,
        prob_dir: Path,
    ) -> Optional[Path]:
        """将 turn_info 中的 kernel_path 解析为实际存在的文件路径。

        kernel_path 可能是绝对路径、相对于 run_dir 的路径、
        相对于 prob_dir 的路径、或只是文件名。逐一尝试。
        """
        raw = turn_info.get("kernel_path", "")
        if not raw:
            return None

        candidates = [
            Path(raw),
            run_dir / raw,
            prob_dir / raw,
            run_dir / Path(raw).name,
            prob_dir / Path(raw).name,
        ]

        for p in candidates:
            if p.is_file():
                return p

        return None
