"""数据模型单元测试。"""

import sys
import json
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest

from src.models import (
    MutantStatus, MutationSite, Mutant, KernelInfo,
    MutationTestResult, RepairResult,
)


class TestMutantStatus(unittest.TestCase):
    def test_values(self):
        self.assertEqual(MutantStatus.KILLED.value, "killed")
        self.assertEqual(MutantStatus.SURVIVED.value, "survived")
        self.assertEqual(MutantStatus.STILLBORN.value, "stillborn")


class TestMutant(unittest.TestCase):
    def test_to_dict(self):
        site = MutationSite(line_start=10, line_end=10, original_code="x + y")
        m = Mutant(
            id="test__arith__0",
            operator_name="arith_replace",
            operator_category="A",
            site=site,
            original_code="x + y",
            mutated_code="x - y",
            status=MutantStatus.KILLED,
        )
        d = m.to_dict()
        self.assertEqual(d["id"], "test__arith__0")
        self.assertEqual(d["status"], "killed")

    def test_from_dict(self):
        d = {
            "id": "test__arith__0",
            "operator_name": "arith_replace",
            "operator_category": "A",
            "status": "survived",
            "site": {"line_start": 10, "line_end": 10, "original_code": "x + y"},
        }
        m = Mutant.from_dict(d)
        self.assertEqual(m.status, MutantStatus.SURVIVED)
        self.assertEqual(m.site.line_start, 10)


class TestMutationTestResult(unittest.TestCase):
    def setUp(self):
        self.kernel = KernelInfo(
            problem_id=1, level=1, problem_name="test",
            source_path="", kernel_code="", reference_module_path="",
        )
        site = MutationSite(line_start=1, line_end=1)
        self.mutants = [
            Mutant(id="1", operator_name="a", operator_category="A",
                   site=site, original_code="", mutated_code="x",
                   status=MutantStatus.KILLED),
            Mutant(id="2", operator_name="a", operator_category="A",
                   site=site, original_code="", mutated_code="y",
                   status=MutantStatus.SURVIVED),
            Mutant(id="3", operator_name="s", operator_category="C",
                   site=site, original_code="", mutated_code="z",
                   status=MutantStatus.STILLBORN),
            Mutant(id="4", operator_name="s", operator_category="C",
                   site=site, original_code="", mutated_code="w",
                   status=MutantStatus.SURVIVED),
        ]

    def test_counts(self):
        r = MutationTestResult(kernel=self.kernel, mutants=self.mutants)
        self.assertEqual(r.total, 4)
        self.assertEqual(r.killed, 1)
        self.assertEqual(r.survived, 2)
        self.assertEqual(r.stillborn, 1)

    def test_mutation_score(self):
        r = MutationTestResult(kernel=self.kernel, mutants=self.mutants)
        self.assertAlmostEqual(r.mutation_score, 1/3)

    def test_score_by_category(self):
        r = MutationTestResult(kernel=self.kernel, mutants=self.mutants)
        scores = r.score_by_category()
        self.assertAlmostEqual(scores["A"], 0.5)
        self.assertAlmostEqual(scores["C"], 0.0)

    def test_save_load(self):
        r = MutationTestResult(kernel=self.kernel, mutants=self.mutants)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            r.save(path)
            loaded = MutationTestResult.load(path)
            self.assertEqual(loaded.total, 4)
            self.assertEqual(loaded.killed, 1)


if __name__ == "__main__":
    unittest.main()
