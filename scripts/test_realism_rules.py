"""测试 RealismGuard 的根因分类规则是否完备。

用人工构造的「正确 vs 错误」代码对，验证每条正则规则和每种根因
是否能被正确识别。不需要 GPU 或 KernelBench 数据。

用法:
    python scripts/test_realism_rules.py
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mutengine.realism_validator import (
    RealismValidator,
    ROOT_CAUSE_TO_OPERATORS,
    DIFF_KEYWORDS_TO_ROOT_CAUSE,
)

# ============================================================
# 测试用例：每种根因对应一个 (correct_code, buggy_code) 配对
# buggy_code 是 LLM 可能写出的错误版本
# ============================================================

TEST_CASES = [
    # --- 1. missing_numerical_stability ---
    {
        "root_cause": "missing_numerical_stability",
        "correct": """
float max_val = -INFINITY;
for (int j = 0; j < D; j++) { if (x[j] > max_val) max_val = x[j]; }
float sum_exp = 0.0f;
for (int j = 0; j < D; j++) {
    float e = expf(x[j] - max_val);  // subtract max for stability
    sum_exp += e;
}
""",
        "buggy": """
float sum_exp = 0.0f;
for (int j = 0; j < D; j++) {
    float e = expf(x[j]);  // forgot to subtract max
    sum_exp += e;
}
""",
    },

    # --- 2. overflow_no_max_subtract ---
    {
        "root_cause": "overflow_no_max_subtract",
        "correct": """
row_m = tl.max(x, axis=1)
p = tl.exp(x - row_m[:, None])
""",
        "buggy": """
p = tl.exp(x)
""",
    },

    # --- 3. precision_loss_fp16_accumulator ---
    {
        "root_cause": "precision_loss_fp16_accumulator",
        "correct": """
accumulator = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
accumulator += tl.dot(a, b)
output = accumulator.to(tl.float16)
""",
        "buggy": """
accumulator = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)
accumulator += tl.dot(a, b)
output = accumulator
""",
    },

    # --- 4. missing_fp32_cast ---
    {
        "root_cause": "missing_fp32_cast",
        "correct": """
x_fp32 = x.float()  # cast to fp32 for precision
result = torch.sum(x_fp32)
output = result.half()
""",
        "buggy": """
result = torch.sum(x)
output = result
""",
    },

    # --- 5. epsilon_missing ---
    {
        "root_cause": "epsilon_missing",
        "correct": """
float eps = 1e-5f;
float inv_std = rsqrtf(var + eps);
""",
        "buggy": """
float inv_std = rsqrtf(var);
""",
    },

    # --- 6. epsilon_wrong_value ---
    {
        "root_cause": "epsilon_wrong_value",
        "correct": """
eps = 1e-5
normalized = x / (std + eps)
""",
        "buggy": """
eps = 0.1
normalized = x / (std + eps)
""",
    },

    # --- 7. scale_factor_missing ---
    {
        "root_cause": "scale_factor_missing",
        "correct": """
scale = 1.0 / math.sqrt(head_dim)
attn = torch.matmul(q, k.T) * scale
""",
        "buggy": """
attn = torch.matmul(q, k.T)
""",
    },

    # --- 8. scale_factor_wrong ---
    {
        "root_cause": "scale_factor_wrong",
        "correct": """
scale = 1.0 / sqrt(d_k)
scores = qk * scale
""",
        "buggy": """
scale = 1.0 / d_k
scores = qk * scale
""",
    },

    # --- 9. missing_type_cast ---
    {
        "root_cause": "missing_type_cast",
        "correct": """
x = input.to(torch.float16)
result = kernel_op(x)
output = result.to(torch.float32)
""",
        "buggy": """
result = kernel_op(input)
output = result
""",
    },

    # --- 10. implicit_type_coercion ---
    {
        "root_cause": "implicit_type_coercion",
        "correct": """
idx = int(offset)
val = float(count) / float(total)
""",
        "buggy": """
val = count / total
""",
    },

    # --- 11. reduction_precision ---
    {
        "root_cause": "reduction_precision",
        "correct": """
row_sum = tl.sum(x.to(tl.float32), axis=1)
output = row_sum.to(tl.float16)
""",
        "buggy": """
row_sum = tl.sum(x, axis=1)
output = row_sum
""",
    },

    # --- 12. wrong_init_value ---
    {
        "root_cause": "wrong_init_value",
        "correct": """
max_val = float('-inf')
for i in range(N):
    if x[i] > max_val: max_val = x[i]
""",
        "buggy": """
max_val = 0.0
for i in range(N):
    if x[i] > max_val: max_val = x[i]
""",
    },

    # --- 13. missing_broadcast ---
    {
        "root_cause": "missing_broadcast",
        "correct": """
mask = mask.unsqueeze(0).expand(batch, seq_len, seq_len)
attn = attn + mask
""",
        "buggy": """
attn = attn + mask
""",
    },

    # --- 14. shape_mismatch_no_expand ---
    {
        "root_cause": "shape_mismatch_no_expand",
        "correct": """
x = x.view(batch, heads, seq_len, dim)
x = x.reshape(batch * heads, seq_len, dim)
""",
        "buggy": """
x = x.contiguous()
""",
    },

    # --- 15. contiguous_assumption ---
    {
        "root_cause": "contiguous_assumption",
        "correct": """
x = x.transpose(1, 2).contiguous()
kernel_op(x.data_ptr())
""",
        "buggy": """
x = x.transpose(1, 2)
kernel_op(x.data_ptr())
""",
    },

    # --- 16. wrong_index_dimension ---
    {
        "root_cause": "wrong_index_dimension",
        "correct": """
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)
""",
        "buggy": """
pid_m = tl.program_id(0)
pid_n = tl.program_id(0)
""",
    },

    # --- 17. off_by_one_boundary ---
    {
        "root_cause": "off_by_one_boundary",
        "correct": """
if (idx < N) {
    output[idx] = input[idx];
}
""",
        "buggy": """
if (idx <= N) {
    output[idx] = input[idx];
}
""",
    },

    # --- 18. wrong_arithmetic_op ---
    {
        "root_cause": "wrong_arithmetic_op",
        "correct": """
result = a + b * scale;
""",
        "buggy": """
result = a - b * scale;
""",
    },

    # --- 19. wrong_comparison_op ---
    {
        "root_cause": "wrong_comparison_op",
        "correct": """
if (val >= threshold) {
    output = val;
}
""",
        "buggy": """
if (val > threshold) {
    output = val;
}
""",
    },

    # --- 20. wrong_constant ---
    {
        "root_cause": "wrong_constant",
        "correct": """
BLOCK_SIZE = 1024
grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
""",
        "buggy": """
BLOCK_SIZE = 512
grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
""",
    },
]


def main():
    validator = RealismValidator()

    print("=" * 70)
    print("  RealismGuard 根因分类规则完备性测试")
    print("=" * 70)

    # 统计所有根因是否都有正则能识别
    all_root_causes = set(ROOT_CAUSE_TO_OPERATORS.keys())
    covered_causes = set()
    failed_cases = []

    for i, tc in enumerate(TEST_CASES, 1):
        expected = tc["root_cause"]
        bug_id = f"test_{i}_{expected}"

        bug = validator.analyze_bug_from_diff(
            bug_id=bug_id,
            problem_id=i,
            level=0,
            correct_code=tc["correct"],
            buggy_code=tc["buggy"],
            source="test",
        )

        match = "OK" if bug.root_cause == expected else "FAIL"
        if bug.root_cause == expected:
            covered_causes.add(expected)
        else:
            failed_cases.append((expected, bug.root_cause))

        ops_str = ", ".join(bug.matched_operators) if bug.matched_operators else "(none)"
        print(f"  [{match:4s}] #{i:2d} expected={expected}")
        if match == "FAIL":
            print(f"         got={bug.root_cause}")
        print(f"         operators={ops_str}  category={bug.error_category}")

    # 报告
    print("\n" + "=" * 70)
    print("  汇总")
    print("=" * 70)

    report = validator.generate_report()
    print(f"  总测试用例:    {len(TEST_CASES)}")
    print(f"  正确识别:      {len(covered_causes)}/{len(all_root_causes)} 根因")
    print(f"  覆盖率:        {len(covered_causes)/len(all_root_causes):.0%}")

    if failed_cases:
        print(f"\n  --- 识别失败的用例 ---")
        for expected, got in failed_cases:
            exp_ops = ROOT_CAUSE_TO_OPERATORS.get(expected, [])
            got_ops = ROOT_CAUSE_TO_OPERATORS.get(got, [])
            print(f"  expected={expected} ({exp_ops})")
            print(f"       got={got} ({got_ops})")

    uncovered = all_root_causes - covered_causes
    if uncovered:
        print(f"\n  --- 未能覆盖的根因 ---")
        for rc in sorted(uncovered):
            ops = ROOT_CAUSE_TO_OPERATORS[rc]
            print(f"  {rc} -> {ops}")
    else:
        print(f"\n  所有 {len(all_root_causes)} 种根因都有对应正则能识别")

    # 额外: 检查正则规则有多少永远不会被触发
    print(f"\n" + "=" * 70)
    print(f"  正则规则触发统计")
    print(f"=" * 70)
    print(f"  DIFF_KEYWORDS_TO_ROOT_CAUSE 共 {len(DIFF_KEYWORDS_TO_ROOT_CAUSE)} 条规则")
    print(f"  ROOT_CAUSE_TO_OPERATORS 共 {len(ROOT_CAUSE_TO_OPERATORS)} 种根因")
    print(f"  可匹配到的根因（通过正则）: {len(set(DIFF_KEYWORDS_TO_ROOT_CAUSE.values()))}")

    regex_targets = set(DIFF_KEYWORDS_TO_ROOT_CAUSE.values())
    root_cause_targets = set(ROOT_CAUSE_TO_OPERATORS.keys())
    no_regex = root_cause_targets - regex_targets
    if no_regex:
        print(f"\n  --- 有算子映射但无正则规则的根因 ---")
        for rc in sorted(no_regex):
            print(f"  {rc} -> {ROOT_CAUSE_TO_OPERATORS[rc]}")
    else:
        print(f"  所有根因都有至少一条正则规则覆盖")


if __name__ == "__main__":
    main()
