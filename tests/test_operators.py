"""变异算子单元测试。

不依赖 GPU 即可运行，验证 find_sites() 和 apply() 的正确性。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest

from src.mutengine.operators.arithmetic import ArithReplace, RelOpReplace, ConstPerturb
from src.mutengine.operators.gpu_parallel import IndexReplace, SyncRemove, MaskBoundary, LaunchConfigMutate
from src.mutengine.operators.ml_semantic import (
    StabRemove, AccDowngrade, EpsilonModify,
    ScaleModify, CastRemove, ReductionReorder, InitModify,
)
from src.mutengine.operators.llm_pattern import BroadcastUnsafe, LayoutAssume
from src.mutengine.operators.base import get_all_operators, get_operators_by_category


TRITON_SOFTMAX = """\
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row_start = row_idx * n_cols
    input_ptrs = input_ptr + row_start + col_offsets
    row = tl.load(input_ptrs, mask=mask, other=float('-inf'))
    row_max = tl.max(row, axis=0)
    row = row - tl.max(row, axis=0)
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    output = numerator / (denominator + 1e-6)
    output_ptrs = output_ptr + row_start + col_offsets
    tl.store(output_ptrs, output, mask=mask)
"""


class TestArithReplace(unittest.TestCase):
    def test_find_sites(self):
        op = ArithReplace()
        source = "x = a + b\ny = c * d\n"
        sites = op.find_sites(source)
        self.assertGreater(len(sites), 0)

    def test_apply(self):
        op = ArithReplace()
        source = "x = a + b\n"
        sites = op.find_sites(source)
        self.assertTrue(len(sites) > 0)
        mutated = op.apply(source, sites[0])
        self.assertIn("-", mutated)
        self.assertNotEqual(source, mutated)


class TestRelOpReplace(unittest.TestCase):
    def test_find_sites(self):
        op = RelOpReplace()
        source = "if x < y:\n    pass\n"
        sites = op.find_sites(source)
        self.assertGreater(len(sites), 0)


class TestConstPerturb(unittest.TestCase):
    def test_find_int(self):
        op = ConstPerturb()
        source = "BLOCK_SIZE = 1024\n"
        sites = op.find_sites(source)
        self.assertGreater(len(sites), 0)

    def test_find_float(self):
        op = ConstPerturb()
        source = "scale = 0.125\n"
        sites = op.find_sites(source)
        self.assertGreater(len(sites), 0)


class TestIndexReplace(unittest.TestCase):
    def test_triton(self):
        op = IndexReplace()
        sites = op.find_sites(TRITON_SOFTMAX)
        self.assertGreater(len(sites), 0)
        for s in sites:
            self.assertIn("program_id", s.original_code)

    def test_apply(self):
        op = IndexReplace()
        sites = op.find_sites(TRITON_SOFTMAX)
        mutated = op.apply(TRITON_SOFTMAX, sites[0])
        self.assertNotEqual(TRITON_SOFTMAX, mutated)


class TestSyncRemove(unittest.TestCase):
    def test_cuda(self):
        op = SyncRemove()
        source = "void kernel() { __syncthreads(); }\n"
        sites = op.find_sites(source)
        self.assertGreater(len(sites), 0)


class TestMaskBoundary(unittest.TestCase):
    def test_triton_mask(self):
        op = MaskBoundary()
        source = "row = tl.load(input_ptrs, mask=offsets < n_elements, other=0.0)\n"
        sites = op.find_sites(source)
        self.assertGreater(len(sites), 0, "Should find < boundary in mask= context")


class TestStabRemove(unittest.TestCase):
    def test_find(self):
        op = StabRemove()
        sites = op.find_sites(TRITON_SOFTMAX)
        self.assertGreater(len(sites), 0, "Should find max-subtraction pattern")

    def test_apply(self):
        op = StabRemove()
        sites = op.find_sites(TRITON_SOFTMAX)
        if sites:
            mutated = op.apply(TRITON_SOFTMAX, sites[0])
            self.assertNotEqual(TRITON_SOFTMAX, mutated)


class TestEpsilonModify(unittest.TestCase):
    def test_find(self):
        op = EpsilonModify()
        sites = op.find_sites(TRITON_SOFTMAX)
        eps_sites = [s for s in sites if "1e-6" in s.original_code]
        self.assertGreater(len(eps_sites), 0)

    def test_apply_to_zero(self):
        op = EpsilonModify()
        sites = op.find_sites(TRITON_SOFTMAX)
        zero_sites = [s for s in sites if s.node_type == "eps:to_zero"]
        if zero_sites:
            mutated = op.apply(TRITON_SOFTMAX, zero_sites[0])
            self.assertIn("0", mutated.splitlines()[
                zero_sites[0].line_start - 1
            ])


class TestInitModify(unittest.TestCase):
    def test_find(self):
        op = InitModify()
        sites = op.find_sites(TRITON_SOFTMAX)
        inf_sites = [s for s in sites if "inf" in s.original_code.lower()]
        self.assertGreater(len(inf_sites), 0)


class TestCastRemove(unittest.TestCase):
    def test_find(self):
        op = CastRemove()
        source = "x = y.to(torch.float32)\nz = w.float()\n"
        sites = op.find_sites(source)
        self.assertGreater(len(sites), 0)


class TestLayoutAssume(unittest.TestCase):
    def test_find(self):
        op = LayoutAssume()
        source = "x = tensor.contiguous()\n"
        sites = op.find_sites(source)
        self.assertGreater(len(sites), 0)

    def test_apply(self):
        op = LayoutAssume()
        source = "x = tensor.contiguous()\n"
        sites = op.find_sites(source)
        mutated = op.apply(source, sites[0])
        self.assertNotIn(".contiguous()", mutated)


class TestBroadcastUnsafe(unittest.TestCase):
    def test_find(self):
        op = BroadcastUnsafe()
        source = "y = x.expand(batch, seq_len)\n"
        sites = op.find_sites(source)
        self.assertGreater(len(sites), 0)


class TestOperatorRegistry(unittest.TestCase):
    def test_all_registered(self):
        all_ops = get_all_operators()
        names = {op.name for op in all_ops}
        self.assertIn("arith_replace", names)
        self.assertIn("stab_remove", names)
        self.assertIn("index_replace", names)
        self.assertIn("broadcast_unsafe", names)

    def test_category_filter(self):
        c_ops = get_operators_by_category("C")
        self.assertEqual(len(c_ops), 7)
        for op in c_ops:
            self.assertEqual(op.category, "C")

    def test_a_operators(self):
        a_ops = get_operators_by_category("A")
        self.assertEqual(len(a_ops), 3)

    def test_b_operators(self):
        b_ops = get_operators_by_category("B")
        self.assertEqual(len(b_ops), 4)

    def test_d_operators(self):
        d_ops = get_operators_by_category("D")
        self.assertEqual(len(d_ops), 2)


class TestGenerateMutants(unittest.TestCase):
    def test_softmax(self):
        all_ops = get_all_operators()
        total_mutants = 0
        for op in all_ops:
            mutants = op.generate_mutants(TRITON_SOFTMAX, "test_softmax")
            total_mutants += len(mutants)
            for m in mutants:
                self.assertNotEqual(m.original_code, m.mutated_code)
                self.assertTrue(len(m.mutated_code) > 0)
        self.assertGreater(total_mutants, 5, "Should generate many mutants for softmax")


if __name__ == "__main__":
    unittest.main()
