"""LLM-Assisted Mutation Analysis (Phase 2).

For mutants that survive the 4-layer stress testing pipeline, the LLM performs
free-form analysis without predefined categories. The workflow is:
  1. Analyze WHY the mutant survives (free-form reasoning)
  2. Decide if the mutant CAN be killed with a targeted input
  3. If killable → suggest 5 inputs → verify on GPU → iterate up to 3 rounds
  4. If not killable → provide robustness suggestions + fixed kernel code
  5. After all mutants analyzed → cluster survival reasons into a taxonomy
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

OPERATOR_DESCRIPTIONS: Dict[str, str] = {
    "arith_replace": "Replace binary arithmetic operator (+→-, *→/, etc.)",
    "relop_replace": "Replace relational operator (<→<=, >→>=, etc.)",
    "const_perturb": "Perturb a numeric constant by a small delta",
    "epsilon_modify": "Change the epsilon value in numerical stability guards",
    "scale_modify": "Alter scaling factors (e.g., 1/sqrt(d) in attention)",
    "stab_remove": "Remove numerical stabilization (e.g., max-subtraction before exp)",
    "cast_remove": "Remove an explicit type cast (static_cast, .float(), etc.)",
    "init_modify": "Change reduction identity elements (-inf → finite, etc.)",
    "acc_downgrade": "Downgrade accumulator precision (float32 → float16)",
    "reduction_reorder": "Reverse the order of a floating-point reduction",
    "index_replace": "Swap thread/block dimension indices (x→y, y→z, etc.)",
    "mask_boundary": "Tighten a boundary guard (idx < N → idx < N-1)",
    "sync_remove": "Remove a GPU synchronization barrier (__syncthreads)",
    "launch_config_mutate": "Perturb grid/block launch sizing by ±1",
    "broadcast_unsafe": "Remove explicit broadcast before element-wise ops",
    "layout_assume": "Remove .contiguous() call before kernel invocations",
}

# ---------------------------------------------------------------------------
# Prompt 1: Initial free-form analysis
# ---------------------------------------------------------------------------
ANALYSIS_PROMPT = """\
You are a GPU kernel expert specializing in CUDA/Triton and numerical computing.
A mutation testing tool applied the following mutation to a GPU kernel. The
mutant has survived ALL of the following testing:
- EMD 4-layer equivalence detection (Layer 0-3, {emd_total_rounds} rounds of bitwise comparison)
- Enhanced deterministic testing ({n_dimensions_executed} dimensions, {enhance_total_rounds} rounds)

Your task: determine whether this mutant is truly unkillable under the
fixed-shape contract, or whether a still-missed value-level test exists.

## Testing Contract

### Main contract (fixed-shape, variable-value)
1. **Shape is FIXED**: Input tensor shapes from `get_inputs()` are NEVER changed.
   Batch size is also fixed.
2. **Values CAN vary**: diverse numerical values within the fixed shapes —
   random seeds, extreme magnitudes, near-zero, sparse, boundary integers,
   adversarial distributions, etc.
3. **Comparison is BITWISE**: NaN-aware bit-for-bit match (stricter than allclose).

### Auxiliary configuration-stress track
We additionally tested alternative batch sizes as a **separate probe** (see
config_stress results below). This does NOT redefine the main contract.
**Do NOT suggest shape or batch size changes in your answer.**

## Full Original Source Code
```
{full_original_code}
```

## Full Mutated Source Code
```
{full_mutated_code}
```

## Mutation Details
- Operator: {operator_name} — {operator_desc}
- Location: line {line_start}
- Original fragment: `{original_fragment}`
- node_type: {node_type}

## Input Specification (FIXED)
{input_spec}

## Equivalence Detection Evidence (EMD Layer 0-3)

NOTE: Previous LLM analysis (Layer 3) may contain mathematical errors.
Reason independently from the source code — do not blindly trust prior
reasoning.

{equiv_evidence}

## Enhanced Deterministic Testing Results (mutant survived ALL)

### Main Track (fixed-shape, variable-value)

#### 1. Value-Distribution Stress
{value_stress_detail}

#### 2. Dtype Stress
{dtype_stress_detail}

#### 3. Repeated Execution
{repeated_run_detail}

#### 4. Training Mode (.train())
{training_stress_detail}

### Auxiliary Configuration-Stress Track (separate from main contract)
{config_stress_detail}

### Policy Semantics Reference
{policy_semantics}

## Mandatory Reasoning Steps

You MUST follow these steps IN ORDER before concluding:

**Step 1 — Reachability analysis**: For every loop variable, thread/block index,
and dimension variable involved in the mutated expression, derive the concrete
value ranges from the constants and launch parameters in the source code.
Do NOT claim a boundary case is reachable unless you can derive it from
the concrete constants and ranges shown in the code.

**Step 2 — Semantic distinguishability**: Given the ranges from Step 1, does
the mutation EVER produce a different result? If the original and mutated
expressions always evaluate identically within the derived ranges, the mutant
is unkillable under this fixed configuration.

**Step 3 — Coverage gap identification**: If a difference IS reachable, review
the tested policies (see Policy Semantics Reference) and explain precisely
which value pattern was missed and WHY it would trigger the difference.

**Step 4 — Conclusion**: Only output `killable: true` if Step 1-3 identified
a concrete, reachable scenario that no previous test covered.

Respond in **strict JSON** (no text before or after):
{{
  "reason_category": "<one of: predicate_unreachable | path_not_triggered | value_insensitive | infection_no_propagation | requires_config_change | unknown>",
  "proof_sketch": "<short bound/range derivation showing why the mutation is or is not distinguishable>",
  "survival_reason": "<detailed technical explanation — 3-8 sentences>",
  "killable": true | false,
  "kill_strategy": "<VALUE-LEVEL strategy or why unkillable>",
  "suggested_test": {{
    "description": "<what this tests — SAME shapes as input_spec>",
    "python_code": "<def generate_inputs(device): ... returning list of tensors>"
  }},
  "recommendations": "<one concrete test-coverage or kernel-robustness lesson>"
}}

CRITICAL rules for suggested_test.python_code:
- Define `def generate_inputs(device): ...` that returns a LIST of tensors
- The returned list MUST have EXACTLY the same number of tensors as forward() expects
- Tensor shapes MUST EXACTLY MATCH those in Input Specification
- Do NOT return multiple scenarios concatenated — return ONE scenario only
- Use ONLY torch and math modules (already imported)
- If you believe the mutant is truly unkillable, set "suggested_test" to null
"""

# ---------------------------------------------------------------------------
# Prompt 2: Re-analysis after failed verification
# ---------------------------------------------------------------------------
REANALYSIS_PROMPT = """\
You are a GPU kernel expert. A mutation testing tool applied a mutation to a GPU
kernel, and the mutant has survived:
- EMD 4-layer equivalence detection ({emd_total_rounds} rounds)
- Enhanced deterministic testing ({n_dimensions_executed} dimensions, {enhance_total_rounds} rounds)
- AND your previous suggested inputs (see below)

Re-analyze with the failure information and try a fundamentally different approach.

## Testing Contract

### Main contract (fixed-shape, variable-value)
- Input tensor shapes from `get_inputs()` are FIXED. Batch size is also fixed.
- Only VALUES can vary. Comparison is BITWISE (NaN-aware).
- **Do NOT suggest shape or batch size changes.**

### Auxiliary configuration-stress track
- Alternative batch sizes were tested separately (see below).
- This does NOT redefine the main contract.

## Full Original Source Code
```
{full_original_code}
```

## Full Mutated Source Code
```
{full_mutated_code}
```

## Mutation Details
- Operator: {operator_name} — {operator_desc}
- Location: line {line_start}
- Original fragment: `{original_fragment}`
- node_type: {node_type}

## Input Specification (FIXED)
{input_spec}

## Equivalence Detection Evidence (EMD Layer 0-3)

NOTE: Previous LLM analysis may contain errors. Reason independently.

{equiv_evidence}

## Enhanced Deterministic Testing Results (mutant survived ALL)
{enhanced_testing_summary}

## Previous Attempts (all FAILED to kill the mutant)
{previous_rounds}

## Mandatory Reasoning Steps

**Step 1 — Reachability**: Derive concrete variable ranges from code constants.
Do NOT claim a boundary case is reachable without deriving it.

**Step 2 — Why previous attempts failed**: For each failed round, explain
the specific mathematical reason it could not expose a difference.

**Step 3 — Novel strategy or conclusion**: Either identify a fundamentally
different value-level approach, or conclude unkillable with proof.

Respond in **strict JSON** (no text before or after):
{{
  "reason_category": "<predicate_unreachable | path_not_triggered | value_insensitive | infection_no_propagation | requires_config_change | unknown>",
  "proof_sketch": "<short bound/range derivation or path argument>",
  "survival_reason": "<updated analysis — 3-8 sentences>",
  "killable": true | false,
  "kill_strategy": "<new VALUE-LEVEL strategy different from ALL previous, or why unkillable>",
  "suggested_test": {{
    "description": "<what this NEW input tests — SAME shapes as input_spec>",
    "python_code": "<def generate_inputs(device): ... DIFFERENT from all previous>"
  }},
  "recommendations": "<one concrete test-coverage or kernel-robustness lesson>"
}}

CRITICAL rules for suggested_test.python_code:
- generate_inputs(device) must return a list with EXACTLY the same tensor count/shapes.
- Shapes MUST EXACTLY MATCH Input Specification.
- Use ONLY torch and math modules. Set "suggested_test" to null if unkillable.
"""

# ---------------------------------------------------------------------------
# Prompt 3: Robustness suggestion + code fix (for unkillable mutants)
# ---------------------------------------------------------------------------
ROBUSTNESS_PROMPT = """\
You are a GPU kernel robustness expert. A mutation was applied to a GPU kernel
and the mutant could NOT be killed despite {num_rounds} rounds of targeted
testing. This means the mutation is either semantically equivalent or reveals
a robustness weakness in the kernel design.

## Original Kernel Code
```
{kernel_code}
```

## Mutation Applied
- Operator: {operator_name} — {operator_desc}
- Location: line {line_start}
- Original fragment: `{original_fragment}`
- Mutated fragment: (mutation removed/changed the above)

## Analysis Summary
{analysis_summary}

## Instructions
1. First, provide a textual robustness suggestion explaining what defensive
   coding practice or correctness check is missing from the kernel.
2. Then, provide a COMPLETE fixed version of the kernel code that would make
   the kernel robust against this type of mutation (i.e., the mutation would
   either be caught by the code or would truly not affect correctness).

Respond in **strict JSON**:
{{
  "robustness_suggestion": "<textual advice on what to fix and why — 3-5 sentences>",
  "robustness_code": "<COMPLETE fixed kernel code — must be valid Python/CUDA>"
}}
"""

# ---------------------------------------------------------------------------
# Prompt 4: Test rule extraction (for mutants killed by LLM suggestions)
# ---------------------------------------------------------------------------
TEST_RULE_PROMPT = """\
You are a test engineering expert for GPU kernels. A mutation was applied to a
GPU kernel, and the mutant was killed using a targeted input. Your task is to
generalize this kill into a reusable test construction rule.

## Testing Principle (CRITICAL)
- Generated policies MUST produce tensors with the SAME shapes as the original
  `get_inputs()`. Only VALUES may vary — shapes, dimensions, and batch size
  are FIXED.
- The `policy(shape, dtype, device, rng)` function receives the ORIGINAL shape
  and must return a tensor of EXACTLY that shape.

## Mutation Details
- Operator: {operator_name} — {operator_desc}
- Kernel: {kernel_name}
- Survival reason (before kill): {survival_reason}

## Killing Input
```python
{killing_code}
```

## Kill Description
{kill_description}

## Instructions
Generalize this specific kill into a REUSABLE RULE that can be added to the
stress test policy bank for future mutation testing campaigns.

The generated policy function receives `shape`, `dtype`, `device`, `rng` and
MUST return a tensor of exactly `shape` with the specified `dtype`. Do NOT
change the shape in any way.

Respond in **strict JSON**:
{{
  "rule_name": "<short snake_case name for this rule, e.g., extreme_exponent_inputs>",
  "rule_description": "<what numerical regime or input pattern this rule targets — 2-3 sentences>",
  "applicable_operators": ["<list of mutation operator names this rule is useful for>"],
  "policy_code": "<Python function body: def policy(shape, dtype, device, rng): ... returning a tensor of EXACTLY shape>"
}}
"""

# ---------------------------------------------------------------------------
# Prompt 5b: Configuration-Stress Track analysis
# ---------------------------------------------------------------------------
CONFIG_STRESS_ANALYSIS_PROMPT = """\
You are a GPU kernel expert. This is the **Configuration-Stress Track** — a
secondary testing track where we vary the **batch dimension** while keeping all
other tensor dimensions fixed. This is separate from our main fixed-shape track.

A mutant survived the main fixed-shape testing. We now want to determine if
varying the batch size (the first dimension of input tensors) could expose
a behavioral difference.

## Mutation Details
- Operator: {operator_name} — {operator_desc}
- Location: line {line_start}
- Original fragment: `{original_fragment}`

## Original Code (around mutation site)
```
{original_context}
```

## Mutated Code (around mutation site)
```
{mutated_context}
```

## Reference Input Specification
{input_spec}

## Equivalence Detection Evidence
{equiv_evidence}

## Instructions

In this track, we CAN vary batch_size (the first dimension of tensors that
have a batch dimension). All other dimensions remain fixed.

Analyze whether varying batch_size could reveal a difference between original
and mutant. Consider:
1. Does the mutation affect thread indexing or block-level synchronization?
2. Does the mutation interact with the batch dimension at all?
3. Would a specific batch size (e.g., 1, prime number, very large) trigger
   different behavior?

Respond in **strict JSON**:
{{
  "batch_sensitive": true | false,
  "reasoning": "<why batch_size variation would or would not help — 2-5 sentences>",
  "suggested_batch_sizes": [<list of integer batch sizes to try, or empty if not sensitive>],
  "priority": "<high | medium | low — how likely batch variation will kill this mutant>"
}}
"""

# ---------------------------------------------------------------------------
# Prompt 6: Post-hoc clustering of survival reasons
# ---------------------------------------------------------------------------
CLUSTER_PROMPT = """\
You are a research analyst specializing in software testing taxonomy design.
Below are {count} survival reason analyses for GPU kernel mutants that could NOT
be killed by any testing method. Your task is to cluster these reasons into a
coherent taxonomy of survival categories.

## Survival Reasons
{reasons_list}

## Instructions
1. Read all reasons carefully
2. Identify recurring themes and patterns
3. Group them into 3-8 categories with clear, non-overlapping definitions
4. Assign each mutant to exactly one category

Respond in **strict JSON**:
{{
  "categories": [
    {{
      "label": "<concise category name in English>",
      "label_cn": "<中文分类名>",
      "description": "<2-3 sentence definition of this category>",
      "mutant_ids": ["<list of mutant_ids belonging to this category>"]
    }}
  ],
  "uncategorized": ["<mutant_ids that don't fit any category>"],
  "taxonomy_rationale": "<1-2 sentences explaining your taxonomy design>"
}}
"""


# ---------------------------------------------------------------------------
# Prompt 6: Equivalence verification (Layer 3 of equiv detection V2)
# ---------------------------------------------------------------------------
EQUIV_VERIFY_PROMPT = """\
You are a GPU kernel equivalence expert. An automated pipeline has classified
the following mutant as **{equiv_level}** (likely equivalent to the original).
Your task is to VERIFY this classification — specifically, to check if the
mutant could actually be KILLED with a cleverly designed input.

## Original Code (around mutation site)
```
{original_context}
```

## Mutated Code (around mutation site)
```
{mutated_context}
```

## Full Original Source Code
Below is the COMPLETE source file (Python host + CUDA kernel) so you can see
all constants, loop bounds, grid/block dimensions, and host-side logic:
```
{full_original_code}
```

## Full Mutated Source Code
```
{full_mutated_code}
```

## Mutation Details
- Operator: {operator_name} — {operator_desc}
- Location: line {line_start}
- Original fragment: `{original_fragment}`

## Multi-Layer Equivalence Evidence (from automated pipeline)
The pipeline checked this mutant through 3 layers before reaching you (Layer 3).
Below is what each layer found:

{layer_evidence}

Final automated classification: **{equiv_level}**

## Testing Principle (CRITICAL — read carefully)

Our mutation testing framework follows a **fixed-shape, variable-value** principle:

1. **Shape is FIXED**: Input tensor shapes come from the REFERENCE problem's
   `get_inputs()` and are NEVER changed. The same shapes are used for both
   original and mutant. Batch size is also fixed.
2. **Values CAN vary**: Within the fixed shapes, we generate diverse numerical
   values — random seeds, extreme magnitudes, near-zero, sparse, boundary
   integers, adversarial distributions, etc. Layer 2 already tested {l2_total_rounds}
   value combinations ({l2_random_seeds} random seeds + operator-directed stress policies).
3. **Comparison is BITWISE**: Original and mutant outputs must match bit-for-bit
   (NaN-aware). This is stricter than allclose.

This means:
- If a mutation changes a module-level constant (e.g., N=2048→2049) but the
  CUDA kernel reads dimensions from `tensor.size()` / `input.shape`, the mutation
  has NO effect on the execution path — it is truly equivalent under fixed-shape
  testing.
- If a mutation changes code that IS on the execution path (e.g., an operator
  inside the CUDA kernel), it MAY be killable by choosing the right input VALUES
  (not shapes).

## Actual Input Specification
{input_spec}

## Mandatory Reasoning Steps

You MUST follow these steps IN ORDER before concluding:

**Step 1 — Kernel dispatch analysis**: Identify ALL kernel functions defined in
the source. Determine which kernel is ACTUALLY LAUNCHED for the given input
dimensions. If the host code has conditional dispatch (e.g., `if M >= 1024: use
kernel_A; else: use kernel_B`), compute the condition using the concrete values
from Input Specification. If the mutation is in a kernel that is NEVER launched,
the mutant is equivalent — stop here.

**Step 2 — Reachability analysis**: For every loop variable, thread/block index,
and dimension variable involved in the mutated expression, derive the CONCRETE
value ranges from the constants and launch parameters in the source code.
Compute grid dimensions, block dimensions, and the resulting thread index ranges
numerically. Do NOT claim a boundary case is reachable unless you can derive it
from the concrete constants and ranges shown in the code.

**Step 3 — Semantic distinguishability**: Given the ranges from Step 2, does
the mutation EVER produce a different result within those ranges? If the
original and mutated expressions always evaluate identically (e.g., `< N` vs
`<= N` when max_index = N-1), the mutant is unkillable — confirm equivalent.

**Step 4 — Value-level kill feasibility**: If a difference IS reachable, can it
be triggered by changing input VALUES (not shapes)? Layer 2 already tried
{l2_total_rounds} value combinations. Describe precisely which value pattern was
missed and WHY it would trigger the difference.

**Step 5 — Conclusion**: Only output `possibly_killable` if Steps 1-4
identified a concrete, reachable scenario that differs AND can be triggered by
value-level input changes. Otherwise, output `confirmed_equivalent`.

**Do NOT suggest changing input shapes, dimensions, or batch size** — our
framework cannot do that. Only suggest value-level strategies.

Respond in **strict JSON** (no extra text before or after the JSON):
{{
  "verdict": "confirmed_equivalent" | "possibly_killable",
  "confidence": <0.0-1.0>,
  "reason_category": "<one of: predicate_unreachable | path_not_triggered | value_insensitive | infection_no_propagation | requires_config_change | unknown>",
  "proof_sketch": "<short bound/range derivation showing why the mutation is or is not distinguishable under the fixed input spec>",
  "reasoning": "<detailed analysis — 3-8 sentences>",
  "kill_strategy": "<if possibly_killable: specific VALUE-LEVEL input strategy to try (not shape changes); else: null>",
  "suggested_test": {{
    "description": "<what this input tests — must use the SAME shapes as input_spec>",
    "python_code": "<generate_inputs(device) code — tensors MUST match the shapes in input_spec>"
  }} | null
}}
"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class LLMAnalysisResult:
    """Result of LLM-assisted analysis for a single survived mutant."""
    mutant_id: str
    operator_name: str
    kernel_name: str

    survival_reason: str = ""
    killable: Optional[bool] = None
    kill_strategy: str = ""
    recommendations: str = ""

    suggested_test_code: Optional[str] = None
    suggested_test_desc: Optional[str] = None

    rounds: List[Dict[str, Any]] = field(default_factory=list)

    killed_by_llm: bool = False
    killing_round: int = 0

    robustness_suggestion: str = ""
    robustness_code: str = ""
    test_construction_rule: Optional[Dict[str, Any]] = None

    cluster_label: str = ""

    raw_llm_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mutant_id": self.mutant_id,
            "operator_name": self.operator_name,
            "kernel_name": self.kernel_name,
            "survival_reason": self.survival_reason,
            "killable": self.killable,
            "kill_strategy": self.kill_strategy,
            "recommendations": self.recommendations,
            "suggested_test_code": self.suggested_test_code,
            "suggested_test_desc": self.suggested_test_desc,
            "rounds": self.rounds,
            "killed_by_llm": self.killed_by_llm,
            "killing_round": self.killing_round,
            "robustness_suggestion": self.robustness_suggestion,
            "robustness_code": self.robustness_code,
            "test_construction_rule": self.test_construction_rule,
            "cluster_label": self.cluster_label,
        }


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def build_stress_detail_table(policy_results: List[Dict[str, Any]]) -> str:
    """Build a markdown table summarizing per-policy stress test results."""
    if not policy_results:
        return "(no detailed stress test data available)"

    POLICY_DESCRIPTIONS = {
        "near_zero": "values ~1e-7",
        "large_magnitude": "values ~1e6",
        "near_overflow": "values ~1e38",
        "denormals": "values ~1e-40 (subnormal)",
        "all_negative": "all values < 0",
        "all_positive": "all values > 0",
        "mixed_extremes": "mix of 1e6 and 1e-6",
        "alternating_sign": "[+big, -big, +big, ...]",
        "sparse": "90% zeros, 10% random",
        "uniform_constant": "all elements = same constant",
        "structured_ramp": "linearly increasing 0→N",
        "boundary_last_element": "last element set to extreme",
        "head_heavy": "first 10% large, rest small",
        "tail_heavy": "last 10% large, rest small",
    }

    from collections import defaultdict
    by_policy = defaultdict(list)
    for pr in policy_results:
        by_policy[pr["policy"]].append(pr)

    lines = ["| Policy | Seeds | Result | Description |",
             "|--------|-------|--------|-------------|"]
    for policy, results in by_policy.items():
        n = len(results)
        ref_ok_all = all(r.get("ref_ok", False) for r in results)
        orig_ok_all = all(r.get("original_ok", False) for r in results)
        mut_ok_all = all(r.get("mutant_ok", True) for r in results)
        any_error = any(r.get("error") for r in results)

        if not ref_ok_all:
            status = "REF_FAIL (ref itself crashes/NaN)"
        elif not orig_ok_all:
            status = "ORIG_ALSO_FAILS (original ≠ ref)"
        elif mut_ok_all:
            status = "both_ok (no difference found)"
        else:
            status = "KILLED"

        desc = POLICY_DESCRIPTIONS.get(policy, policy)
        lines.append(f"| {policy} | ×{n} | {status} | {desc} |")

    return "\n".join(lines)


def _format_equiv_evidence(equiv_detail: Dict[str, Any]) -> str:
    """Format equiv_detail from Block1+2 into readable text for LLM prompts."""
    if not equiv_detail:
        return "No equivalence detection evidence available."
    lines = []
    l0 = equiv_detail.get("layer0", {})
    if l0:
        lines.append("### Layer 0 (Source Normalization)")
        lines.append(f"- CUDA identical: {l0.get('cuda_eq', 'N/A')}")
        lines.append(f"- Python host identical: {l0.get('py_eq', 'N/A')}")
        lines.append(f"- Verdict: {l0.get('verdict', 'N/A')}")
        hda = l0.get("host_diff_analysis", {})
        if hda:
            lines.append(f"- Mutation location: {hda.get('mutation_location', 'N/A')}")
            lines.append(f"- Mutated variable: {hda.get('mutated_variable', 'N/A')}")
            lines.append(f"- Used in ModelNew: {hda.get('used_in_model', 'N/A')}")
            lines.append(f"- Used in get_inputs: {hda.get('used_in_get_inputs', 'N/A')}")
        mut_line = l0.get("mutation_site_line")
        if mut_line:
            lines.append(f"- Mutation site line: {mut_line}")
        orig_frag = l0.get("original_fragment")
        if orig_frag:
            lines.append(f"- Original fragment: `{orig_frag}`")
    l1 = equiv_detail.get("layer1", {})
    if l1:
        lines.append("### Layer 1 (Static Rules)")
        lines.append(f"- Matched rule: {l1.get('rule', 'none')}")
        if l1.get("rule_description"):
            lines.append(f"- Rule description: {l1['rule_description']}")
        if l1.get("rules_checked"):
            lines.append(f"- Rules checked: {', '.join(l1['rules_checked'])}")
    l2 = equiv_detail.get("layer2", {})
    if l2:
        lines.append("### Layer 2 (Dynamic Bitwise Comparison)")
        lines.append(f"- Is equivalent (bitwise): {l2.get('is_equivalent', 'N/A')}")
        lines.append(f"- Total rounds: {l2.get('total_rounds', 'N/A')}")
        lines.append(f"- CUDA was identical (L0): {l2.get('cuda_was_identical', 'N/A')}")
        seeds = l2.get("tested_random_seeds", [])
        if seeds:
            lines.append(f"- Tested seeds: {seeds}")
        policies = l2.get("tested_policies", [])
        if policies:
            lines.append("- Tested stress policies:")
            for p in policies:
                if isinstance(p, dict):
                    pname = p.get("name") or p.get("policy") or "?"
                    lines.append(f"  - {pname}: {p.get('status', '?')}")
                else:
                    lines.append(f"  - {p}")
        div = l2.get("divergence", {})
        if div:
            lines.append(f"- Divergence found: round_type={div.get('round_type')}, "
                         f"seed={div.get('seed')}, policy={div.get('policy')}")
        first_sum = l2.get("first_input_summary")
        if first_sum:
            lines.append(f"- First input summary: {first_sum}")
        last_sum = l2.get("last_input_summary")
        if last_sum:
            lines.append(f"- Last input summary: {last_sum}")
        err = l2.get("error")
        if err:
            lines.append(f"- Error: {err}")
    l3 = equiv_detail.get("layer3", {})
    if l3:
        lines.append("### Layer 3 (LLM Verification — Phase 1)")
        lines.append(f"- Verdict: {l3.get('verdict', 'N/A')}")
        lines.append(f"- Confidence: {l3.get('confidence', 'N/A')}")
        reasoning = l3.get("reasoning", "")
        if reasoning:
            lines.append(f"- Reasoning: {reasoning[:500]}")
        ks = l3.get("kill_strategy")
        if ks:
            lines.append(f"- Kill strategy: {ks}")
        st = l3.get("suggested_test")
        if st and isinstance(st, dict):
            desc = st.get("description", "")
            has_code = bool(st.get("python_code"))
            lines.append(f"- Suggested test: {desc[:200]}")
            lines.append(f"- Had executable code: {has_code}")
    return "\n".join(lines) if lines else "No evidence recorded."


def _format_enhanced_results(enhanced: Dict[str, Any]) -> Dict[str, str]:
    """Format Phase 2 per-dimension results into LLM-readable markdown texts.

    Returns dict with keys matching ANALYSIS_PROMPT V2 placeholders:
      value_stress_detail, dtype_stress_detail, repeated_run_detail,
      training_stress_detail, config_stress_detail
    """
    result = {}

    mt = enhanced.get("main_track", {})

    vs = mt.get("value_stress", {})
    if vs.get("executed"):
        lines = [f"Tested {vs.get('rounds_executed', '?')} / "
                 f"{vs.get('rounds_total', '?')} rounds. "
                 f"Result: {'KILLED by ' + str(vs.get('killing_policy')) if vs.get('killed') else 'ALL SURVIVED (bitwise identical)'}"]
        for pr in (vs.get("policy_results") or [])[:30]:
            if isinstance(pr, dict):
                lines.append(f"  - {pr.get('policy', '?')} seed={pr.get('seed', '?')}: "
                             f"{'KILLED' if pr.get('killed') else 'survived'}")
        result["value_stress_detail"] = "\n".join(lines)
    else:
        result["value_stress_detail"] = "(not executed)"

    ds = mt.get("dtype_stress", {})
    if ds.get("executed"):
        killed_info = f"KILLED by {ds.get('killing_dtype')}" if ds.get("killed") else "ALL SURVIVED"
        lines = [f"Result: {killed_info}"]
        for r in (ds.get("results") or []):
            if isinstance(r, dict):
                dtypes_tested = r.get("tested_dtypes") or r.get("dtype") or "float16/bfloat16"
                if isinstance(dtypes_tested, list):
                    dtypes_tested = ", ".join(dtypes_tested)
                kill_dtype = r.get("killing_dtype", "")
                status = f"KILLED({kill_dtype})" if r.get("killed") else "survived"
                lines.append(f"  - seed={r.get('seed', '?')} dtypes=[{dtypes_tested}]: {status}")
        result["dtype_stress_detail"] = "\n".join(lines)
    else:
        result["dtype_stress_detail"] = "(not executed)"

    rr = mt.get("repeated_run", {})
    if rr.get("executed"):
        result["repeated_run_detail"] = (
            f"Result: {'KILLED (inconsistency detected)' if rr.get('killed') else 'ALL CONSISTENT'}, "
            f"inconsistency_detected={rr.get('inconsistency_detected', False)}")
    else:
        result["repeated_run_detail"] = "(not executed)"

    ts = mt.get("training_stress", {})
    if ts.get("executed"):
        killed_info = f"KILLED by {ts.get('killing_policy')}" if ts.get("killed") else "ALL SURVIVED"
        result["training_stress_detail"] = f"Result: {killed_info}"
    elif ts.get("skipped_reason"):
        result["training_stress_detail"] = f"SKIPPED: {ts['skipped_reason']}"
    else:
        result["training_stress_detail"] = "(not executed)"

    ct = enhanced.get("config_track", {})
    cs = ct.get("config_stress", {})
    if cs.get("executed"):
        if cs.get("killed"):
            result["config_stress_detail"] = (
                f"KILLED at batch_size={cs.get('killing_batch_size')}, "
                f"kill_type={cs.get('kill_type')}")
        else:
            tested = list((cs.get("results_per_batch") or {}).keys())
            result["config_stress_detail"] = (
                f"ALL SURVIVED. Tested batch_sizes: {tested if tested else 'N/A'}")
    else:
        result["config_stress_detail"] = "(not executed)"

    return result


def _format_enhanced_summary(enhanced: Dict[str, Any]) -> str:
    """One-paragraph summary of all enhanced testing results for REANALYSIS_PROMPT."""
    parts = _format_enhanced_results(enhanced)
    lines = [
        "### Value-Distribution Stress",
        parts.get("value_stress_detail", "N/A"),
        "### Dtype Stress",
        parts.get("dtype_stress_detail", "N/A"),
        "### Repeated Execution",
        parts.get("repeated_run_detail", "N/A"),
        "### Training Mode",
        parts.get("training_stress_detail", "N/A"),
        "### Configuration-Stress Track",
        parts.get("config_stress_detail", "N/A"),
    ]
    return "\n".join(lines)


_POLICY_SEMANTICS = {
    "near_zero": "values ~1e-7, tests near-zero sensitivity",
    "large_magnitude": "values ~1e6, tests large-number accumulation",
    "near_overflow": "values ~1e38, tests near-overflow behavior",
    "denormals": "values ~1e-40 (subnormal floats)",
    "all_negative": "all values < 0",
    "all_positive": "all values > 0",
    "mixed_extremes": "mix of 1e6 and 1e-6 values",
    "alternating_sign": "[+big, -big, +big, ...] pattern",
    "sparse": "90% zeros, 10% random non-zero",
    "uniform_constant": "all elements = same constant value",
    "structured_ramp": "linearly increasing 0→N, index-visible pattern",
    "boundary_last_element": "all-zero baseline with last valid element amplified",
    "head_heavy": "first 10% large, rest small",
    "tail_heavy": "last 10% large, rest small",
    "dense_nonzero": "all non-zero (|randn|+1.0), eliminates zero-masking equivalence",
    "sparse_extreme": "99% zeros + 1% extreme (randn*1e4), boundary/propagation test",
    "relop_boundary_hit": "integer-valued tensors emphasizing equality/boundary comparisons",
    "nan_inject": "some elements set to NaN",
    "inf_inject": "some elements set to ±Inf",
    "identity_like": "identity-matrix-like structured input",
    "checkerboard": "alternating 0/1 checkerboard pattern",
}


def _build_policy_semantics(enhanced_results: Dict[str, Any]) -> str:
    """Build a compact policy semantics reference for tested policies."""
    tested = set()
    mt = enhanced_results.get("main_track", {})
    vs = mt.get("value_stress", {})
    for pr in (vs.get("policy_results") or []):
        if isinstance(pr, dict):
            tested.add(pr.get("policy", ""))
    if not tested:
        return "(no detailed policy data available)"
    lines = []
    for name in sorted(tested):
        if not name:
            continue
        desc = _POLICY_SEMANTICS.get(name, name)
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines) if lines else "(no policy semantics available)"


def _compute_testing_stats(
    equiv_detail: Dict[str, Any],
    enhanced_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute dynamic testing statistics from actual data."""
    l2 = (equiv_detail or {}).get("layer2", {})
    emd_total_rounds = l2.get("total_rounds", "~112")

    mt = (enhanced_results or {}).get("main_track", {})
    ct = (enhanced_results or {}).get("config_track", {})

    n_dims = 0
    total_rounds = 0
    for dim_result in list(mt.values()) + list(ct.values()):
        if isinstance(dim_result, dict) and dim_result.get("executed"):
            n_dims += 1
            total_rounds += dim_result.get("rounds_executed",
                            dim_result.get("rounds_total", 0))
    if n_dims == 0:
        n_dims = len(mt) + len(ct)
    if total_rounds == 0:
        total_rounds = "~100-195"

    return {
        "emd_total_rounds": emd_total_rounds,
        "n_dimensions_executed": n_dims,
        "enhance_total_rounds": total_rounds,
    }


def build_analysis_prompt(
    original_code: str,
    mutated_code: str,
    operator_name: str,
    site: Dict[str, Any],
    input_spec: str,
    equiv_detail: Dict[str, Any],
    enhanced_results: Dict[str, Any],
) -> str:
    """Build ANALYSIS_PROMPT V2 for Phase 2 Step 3 LLM iterative analysis."""
    line_start = site.get("line_start", 0)
    original_fragment = site.get("original_code", "")
    node_type = site.get("node_type", "")
    op_desc = OPERATOR_DESCRIPTIONS.get(operator_name, operator_name)

    equiv_evidence = _format_equiv_evidence(equiv_detail or {})
    dim_texts = _format_enhanced_results(enhanced_results or {})
    policy_sem = _build_policy_semantics(enhanced_results or {})
    stats = _compute_testing_stats(equiv_detail, enhanced_results)

    return ANALYSIS_PROMPT.format(
        full_original_code=original_code,
        full_mutated_code=mutated_code,
        operator_name=operator_name,
        operator_desc=op_desc,
        line_start=line_start,
        original_fragment=original_fragment,
        node_type=node_type,
        input_spec=input_spec,
        equiv_evidence=equiv_evidence,
        value_stress_detail=dim_texts.get("value_stress_detail", "N/A"),
        dtype_stress_detail=dim_texts.get("dtype_stress_detail", "N/A"),
        repeated_run_detail=dim_texts.get("repeated_run_detail", "N/A"),
        training_stress_detail=dim_texts.get("training_stress_detail", "N/A"),
        config_stress_detail=dim_texts.get("config_stress_detail", "N/A"),
        policy_semantics=policy_sem,
        emd_total_rounds=stats["emd_total_rounds"],
        n_dimensions_executed=stats["n_dimensions_executed"],
        enhance_total_rounds=stats["enhance_total_rounds"],
    )


def _format_rounds_text(previous_rounds: List[Dict[str, Any]]) -> str:
    """Format previous LLM iterative analysis rounds for REANALYSIS_PROMPT."""
    rounds_text = ""
    for r in previous_rounds:
        rounds_text += f"\n### Round {r['round']}\n"
        rounds_text += f"- Strategy: {r.get('kill_strategy', 'N/A')}\n"
        code = r.get("suggested_code", "N/A")
        if code and code != "N/A":
            rounds_text += f"- Suggested code:\n```python\n{code}\n```\n"
        rounds_text += f"- Verification result: {'KILLED' if r.get('killed') else 'NOT KILLED'}\n"
        detail = r.get("execution_result", r.get("detail", {}))
        if detail:
            rounds_text += (f"- Detail: ref_ok={detail.get('ref_ok')}, "
                            f"original_ok={detail.get('original_ok')}, "
                            f"mutant_ok={detail.get('mutant_ok')}\n")
            if detail.get("diff_summary"):
                rounds_text += f"- Diff summary: {detail['diff_summary']}\n"
            if detail.get("error"):
                rounds_text += f"- Error: {detail['error']}\n"
    return rounds_text


def build_reanalysis_prompt(
    original_code: str,
    mutated_code: str,
    operator_name: str,
    site: Dict[str, Any],
    input_spec: str,
    previous_rounds: List[Dict[str, Any]],
    equiv_detail: Dict[str, Any],
    enhanced_results: Dict[str, Any],
) -> str:
    """Build REANALYSIS_PROMPT V2 for rounds 2+ of LLM iterative analysis."""
    line_start = site.get("line_start", 0)
    original_fragment = site.get("original_code", "")
    node_type = site.get("node_type", "")
    op_desc = OPERATOR_DESCRIPTIONS.get(operator_name, operator_name)

    equiv_evidence = _format_equiv_evidence(equiv_detail or {})
    enhanced_summary = _format_enhanced_summary(enhanced_results or {})
    rounds_text = _format_rounds_text(previous_rounds)
    stats = _compute_testing_stats(equiv_detail, enhanced_results)

    return REANALYSIS_PROMPT.format(
        full_original_code=original_code,
        full_mutated_code=mutated_code,
        operator_name=operator_name,
        operator_desc=op_desc,
        line_start=line_start,
        original_fragment=original_fragment,
        node_type=node_type,
        input_spec=input_spec,
        equiv_evidence=equiv_evidence,
        enhanced_testing_summary=enhanced_summary,
        previous_rounds=rounds_text,
        emd_total_rounds=stats["emd_total_rounds"],
        n_dimensions_executed=stats["n_dimensions_executed"],
        enhance_total_rounds=stats["enhance_total_rounds"],
    )


def build_robustness_prompt(
    kernel_code: str,
    operator_name: str,
    site: Dict[str, Any],
    analysis_summary: str,
    num_rounds: int,
) -> str:
    line_start = site.get("line_start", 0)
    original_fragment = site.get("original_code", "")
    op_desc = OPERATOR_DESCRIPTIONS.get(operator_name, operator_name)

    return ROBUSTNESS_PROMPT.format(
        kernel_code=kernel_code,
        operator_name=operator_name,
        operator_desc=op_desc,
        line_start=line_start,
        original_fragment=original_fragment,
        analysis_summary=analysis_summary,
        num_rounds=num_rounds,
    )


def build_test_rule_prompt(
    operator_name: str,
    kernel_name: str,
    survival_reason: str,
    killing_code: str,
    kill_description: str,
) -> str:
    op_desc = OPERATOR_DESCRIPTIONS.get(operator_name, operator_name)

    return TEST_RULE_PROMPT.format(
        operator_name=operator_name,
        operator_desc=op_desc,
        kernel_name=kernel_name,
        survival_reason=survival_reason,
        killing_code=killing_code,
        kill_description=kill_description,
    )


def _format_layer_evidence(detail: Dict[str, Any]) -> str:
    """Format equiv_detail dict into human-readable layer-by-layer evidence."""
    lines = []
    l0 = detail.get("layer0")
    if l0:
        lines.append("### Layer 0 — Source Code Normalization")
        lines.append(f"- CUDA kernel strings extracted: {l0.get('cuda_extracted', '?')}")
        lines.append(f"- CUDA strings equal after normalization: "
                      f"**{l0.get('cuda_strings_equal', '?')}**")
        lines.append(f"- Python host code equal after normalization: "
                      f"**{l0.get('python_host_equal', '?')}**")
        v = l0.get("verdict", "")
        if v:
            lines.append(f"- Layer 0 verdict: {v}")

        hda = l0.get("host_diff_analysis")
        if hda:
            lines.append(f"\n**Host-code diff analysis** (CUDA strings are "
                          f"identical, so mutation is in Python host code):")
            lines.append(f"- Mutation location: **{hda.get('mutation_location', '?')}**")
            var = hda.get("mutated_variable")
            if var:
                lines.append(f"- Mutated variable: `{var}`")
                lines.append(f"- Used inside ModelNew (forward/init): "
                              f"**{hda.get('used_in_model', '?')}**")
                lines.append(f"- Used inside get_inputs/get_init_inputs: "
                              f"**{hda.get('used_in_get_inputs', '?')}**")
                if hda.get("used_in_model") is False:
                    lines.append(
                        "- **IMPORTANT**: This variable is NOT referenced in "
                        "ModelNew. Under our fixed-shape testing framework, "
                        "get_inputs() always comes from the REFERENCE module, "
                        "so this mutation is effectively dead code.")
    else:
        lines.append("### Layer 0 — Source Code Normalization: (not run)")

    l1 = detail.get("layer1")
    if l1:
        lines.append("\n### Layer 1 — Static Equivalence Rules")
        lines.append(f"- Rules engine available: {l1.get('rules_available', '?')}")
        hit = l1.get("rule_hit")
        lines.append(f"- Rule hit: **{hit if hit else 'none'}**")
        if hit:
            rule_descs = {
                "boundary_unreachable": "threadIdx always < blockDim, so "
                    "< vs <= on blockDim boundary is unreachable",
                "dead_write": "mutated assignment target is overwritten "
                    "before any read",
                "mask_noreach": "mask boundary tightening only affects "
                    "padding threads",
                "dead_host_constant": "module-level constant not used by "
                    "ModelNew; dead code under fixed-shape testing",
            }
            desc = rule_descs.get(hit, "")
            if desc:
                lines.append(f"- Rule explanation: {desc}")
        v = l1.get("verdict", "")
        if v:
            lines.append(f"- Layer 1 verdict: {v}")
    else:
        lines.append("\n### Layer 1 — Static Rules: (not reached — "
                     "decided at Layer 0)")

    l2 = detail.get("layer2")
    if l2:
        lines.append("\n### Layer 2 — Dynamic Bitwise Comparison")
        lines.append(f"- Number of test runs: {l2.get('equiv_runs', '?')} "
                      f"(random seeds + operator-directed stress policies)")
        lines.append(f"- Operator: {l2.get('operator_name', '?')}")
        lines.append(f"- All runs bitwise identical: "
                      f"**{l2.get('is_equivalent', '?')}**")
        lines.append(f"- Time taken: {l2.get('time_ms', '?')}ms")
        if l2.get("cuda_was_identical"):
            lines.append("- Note: CUDA kernel strings were identical "
                          "(mutation in host code only)")
        v = l2.get("verdict", "")
        if v:
            lines.append(f"- Layer 2 verdict: {v}")
    else:
        lines.append("\n### Layer 2 — Dynamic Comparison: "
                     "(not reached — decided at earlier layer)")

    decided = detail.get("decided_at", "unknown")
    lines.append(f"\n**Decision made at**: {decided}")

    return "\n".join(lines)


def build_equiv_verify_prompt(
    kernel_code: str,
    mutated_code: str,
    operator_name: str,
    site: Dict[str, Any],
    equiv_level: str,
    equiv_evidence: str,
    input_spec: str = "unknown",
    layer_detail: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a Layer-3 equivalence verification prompt."""
    line_start = site.get("line_start", 0)
    original_fragment = site.get("original_code", "")

    original_context = _extract_context(kernel_code, line_start, radius=12)
    mutated_context = _extract_context(mutated_code, line_start, radius=12)
    op_desc = OPERATOR_DESCRIPTIONS.get(operator_name, operator_name)

    if layer_detail:
        layer_evidence = _format_layer_evidence(layer_detail)
    else:
        layer_evidence = equiv_evidence

    l2_data = (layer_detail or {}).get("layer2", {})
    l2_total_rounds = l2_data.get("total_rounds", "100+")
    l2_seeds = l2_data.get("tested_random_seeds", [])
    l2_random_seeds = len(l2_seeds) if l2_seeds else "20+"

    max_full_len = 6000
    full_orig = kernel_code if len(kernel_code) <= max_full_len else (
        kernel_code[:max_full_len] + "\n... (truncated)")
    full_mut = mutated_code if len(mutated_code) <= max_full_len else (
        mutated_code[:max_full_len] + "\n... (truncated)")

    return EQUIV_VERIFY_PROMPT.format(
        original_context=original_context,
        mutated_context=mutated_context,
        full_original_code=full_orig,
        full_mutated_code=full_mut,
        operator_name=operator_name,
        operator_desc=op_desc,
        line_start=line_start,
        original_fragment=original_fragment,
        layer_evidence=layer_evidence,
        equiv_level=equiv_level,
        input_spec=input_spec,
        l2_total_rounds=l2_total_rounds,
        l2_random_seeds=l2_random_seeds,
    )


def verify_equivalent_with_llm(
    mutant_id: str,
    kernel_code: str,
    mutated_code: str,
    operator_name: str,
    site: Dict[str, Any],
    equiv_level: str,
    equiv_evidence: str,
    input_spec: str,
    call_llm_fn,
    layer_detail: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Layer 3: verify an equivalent classification using LLM (Plan §7.3).

    *call_llm_fn* is a callable(prompt: str) -> str provided by the caller
    (wrapping any OpenAI-compatible client).

    Returns a dict with keys:
        verdict:     "confirmed_equivalent" | "possibly_killable" | "error"
        confidence:  float 0-1
        reasoning:   str
        kill_strategy: str | None
        suggested_test: dict | None   (description + python_code)
        raw:         str  (raw LLM response)
    """
    prompt = build_equiv_verify_prompt(
        kernel_code=kernel_code,
        mutated_code=mutated_code,
        operator_name=operator_name,
        site=site,
        equiv_level=equiv_level,
        equiv_evidence=equiv_evidence,
        input_spec=input_spec,
        layer_detail=layer_detail,
    )

    try:
        raw = call_llm_fn(prompt)
    except Exception as e:
        logger.warning(f"LLM call failed for {mutant_id}: {e}")
        return {
            "verdict": "error",
            "confidence": 0.0,
            "reasoning": f"LLM call failed: {e}",
            "kill_strategy": None,
            "suggested_test": None,
            "raw": "",
        }

    logger.info(f"LLM raw response for {mutant_id} "
                f"(len={len(raw)}): {raw[:300]!r}")

    parsed = parse_llm_response(raw)
    if parsed is None:
        logger.warning(f"LLM response unparseable for {mutant_id}, "
                       f"full response: {raw[:1000]!r}")
        return {
            "verdict": "error",
            "confidence": 0.0,
            "reasoning": "Could not parse LLM JSON response",
            "kill_strategy": None,
            "suggested_test": None,
            "raw": raw[:500],
        }

    verdict = parsed.get("verdict", "confirmed_equivalent")
    confidence = float(parsed.get("confidence", 0.5))
    reasoning = parsed.get("reasoning", "")
    reason_category = parsed.get("reason_category", "")
    proof_sketch = parsed.get("proof_sketch", "")
    kill_strategy = parsed.get("kill_strategy")
    suggested_test = parsed.get("suggested_test")

    if suggested_test and isinstance(suggested_test, dict):
        code = suggested_test.get("python_code", "")
        safety_err = validate_suggested_code(code) if code else "empty code"
        if safety_err:
            logger.info(f"LLM suggested test rejected for {mutant_id}: {safety_err}")
            suggested_test = None

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reason_category": reason_category,
        "proof_sketch": proof_sketch,
        "reasoning": reasoning,
        "kill_strategy": kill_strategy,
        "suggested_test": suggested_test,
        "raw": raw[:1000],
    }


def build_cluster_prompt(
    survived_results: List[Dict[str, Any]],
) -> str:
    reasons_list = ""
    for r in survived_results:
        mid = r.get("mutant_id", "unknown")
        reason = r.get("survival_reason", "no reason provided")
        op = r.get("operator_name", "")
        reasons_list += f"\n- **{mid}** (op={op}): {reason}\n"

    return CLUSTER_PROMPT.format(
        count=len(survived_results),
        reasons_list=reasons_list,
    )


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def parse_llm_response(raw: str) -> Optional[Dict[str, Any]]:
    """Parse the LLM JSON response into a dict. Returns None on failure."""
    return _extract_json(raw)


def parse_analysis_response(raw: str, mutant_id: str, operator_name: str,
                             kernel_name: str) -> LLMAnalysisResult:
    """Parse initial/re-analysis response into LLMAnalysisResult."""
    result = LLMAnalysisResult(
        mutant_id=mutant_id,
        operator_name=operator_name,
        kernel_name=kernel_name,
        raw_llm_response=raw,
    )

    data = _extract_json(raw)
    if data is None:
        result.survival_reason = "LLM response could not be parsed as JSON"
        return result

    result.survival_reason = data.get("survival_reason", "")
    result.killable = data.get("killable")
    result.kill_strategy = data.get("kill_strategy", "")
    result.recommendations = data.get("recommendations", "")

    st = data.get("suggested_test")
    if isinstance(st, dict) and st.get("python_code"):
        result.suggested_test_code = st["python_code"]
        result.suggested_test_desc = st.get("description", "")

    return result


# ---------------------------------------------------------------------------
# Safety validation
# ---------------------------------------------------------------------------
_FORBIDDEN_PATTERNS = [
    "import os", "import subprocess", "import sys", "import shutil",
    "open(", "__import__", "exec(", "eval(", "compile(",
    "globals(", "locals(", "getattr(", "setattr(",
    "os.system", "os.popen", "subprocess.",
]


def validate_suggested_code(code: str) -> Optional[str]:
    """Check LLM-suggested code for safety. Returns error message or None."""
    if not code or not code.strip():
        return "empty code"
    for pat in _FORBIDDEN_PATTERNS:
        if pat in code:
            return f"forbidden pattern: {pat}"
    if "generate_inputs" not in code:
        return "missing generate_inputs function definition"
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_context(source: str, line_no: int, radius: int = 10) -> str:
    """Extract lines around a given line number with markers."""
    lines = source.splitlines()
    start = max(0, line_no - radius - 1)
    end = min(len(lines), line_no + radius)
    result = []
    for i in range(start, end):
        marker = ">>>" if i == line_no - 1 else "   "
        result.append(f"{marker} {i + 1:4d}| {lines[i]}")
    return "\n".join(result)


def _extract_json(text: str) -> Optional[Dict]:
    """Try to parse JSON from LLM output.

    Handles: pure JSON, markdown-fenced JSON, JSON embedded in reasoning text.
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    cleaned = re.sub(r"```(?:json)?\s*\n?", "", text)
    cleaned = re.sub(r"\n?```", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    candidates = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            depth = 0
            start = i
            in_string = False
            escape_next = False
            for j in range(i, len(text)):
                ch = text[j]
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\' and in_string:
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:j+1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict):
                                candidates.append(obj)
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
            else:
                i += 1
        else:
            i += 1

    if not candidates:
        return None

    for c in reversed(candidates):
        if "verdict" in c and "confidence" in c:
            return c

    for c in reversed(candidates):
        if "survival_reason" in c or "killable" in c:
            return c

    for c in reversed(candidates):
        if "robustness_suggestion" in c or "rule_name" in c or "categories" in c:
            return c

    return candidates[-1]
