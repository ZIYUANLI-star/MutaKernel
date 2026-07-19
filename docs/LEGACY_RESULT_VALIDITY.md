# Legacy Result Validity Register

## 1. Scope

This document records what the existing MutaKernel numbers mean, what they do
not establish, and what must be rerun or independently audited before an FSE
submission. Historical files are valuable pilot evidence, but they are not
automatically final evidence after the runner, oracle, contracts, or data layout
change.

The governing rule is:

> A historical detector alert is not a confirmed defect, an LLM decision is not
> a proof of equivalence, and a selected cohort size is not an independent audit
> size.

## 2. Validity labels

| Label | Meaning |
|---|---|
| `HISTORICAL_REPRODUCIBLE` | The count can be recomputed from preserved historical artifacts, but still describes the old protocol |
| `PILOT_ONLY` | Useful for planning or motivation; not suitable for a final causal or accuracy claim |
| `REQUIRES_RERUN` | Execution semantics, code drift, path drift, or missing provenance require a clean rerun |
| `REQUIRES_HUMAN_AUDIT` | The detector produced an alert, but the correctness label is not independently established |
| `IN_CONTRACT_ONLY` | Valid only after excluding tests outside the frozen task contract |
| `DO_NOT_CLAIM` | The available evidence does not support the stated interpretation |

## 3. Internal mutation funnel

The preserved Phase-I accounting is:

| Outcome | Historical count |
|---|---:|
| Baseline killed | 939 |
| Survived | 270 |
| Candidate equivalent | 264 |
| Stillborn | 163 |
| Strict equivalent | 10 |
| Total mutants | 1,646 |

The arithmetic is internally consistent:

```text
939 + 270 + 264 + 163 + 10 = 1,646
```

The Phase-II selection cohort is:

```text
270 survived + 264 candidate-equivalent = 534
```

### 3.1 Count 939

**Meaning:** 939 mutants were killed by the historical baseline execution
pipeline.

**Validity:** `HISTORICAL_REPRODUCIBLE`, `REQUIRES_RERUN`.

**Permitted use before rerun:** describe it as the old pipeline's recorded
baseline-kill count.

**Not permitted:** treat 939 as immutable ground truth, compare it directly with
numbers produced by a corrected runner, or infer natural LLM-defect prevalence
from it.

Reasons for rerun include duplicated/drifted worker implementations, host-specific
paths and architecture defaults, insufficiently controlled module state and RNG,
input-mutation risk, global tolerances, and coarse failure classification.

The state issue is concrete rather than merely hypothetical: the historical
Phase-I runner instantiated the reference and mutant models consecutively
without restoring initialization RNG or synchronizing parameters and buffers.
The recorded kill distribution was 485/977 for L1 and 454/506 for L2 among
non-stillborn mutants.  Because many L2 tasks contain randomly initialized
Linear/Conv/BatchNorm state, a portion of these differences may be unrelated to
the injected mutation. All 1,646 source mutants must therefore be regenerated
and rerun; it is not sufficient to recheck only the 939 historical kills or
the historical survivors.

### 3.2 Count 534

**Meaning:** 534 is the Phase-II selection cohort: 270 survivors plus 264
candidate-equivalent mutants.

**Validity:** `HISTORICAL_REPRODUCIBLE` as a legacy cohort size,
`REQUIRES_RERUN` for the FSE primary frame.

**Critical limitation:** 534 is not the number independently audited after
MutaKernel produced its Phase-II verdicts. The recorded later audit operated on
368 remaining cases:

```text
534 selected - 166 already stress-killed = 368 later audit targets
```

**Not permitted:** state that an independent blind audit of all 534 established
the ground truth used to evaluate the same detector. Doing so creates a circular
or resubstitution interpretation.

Preserve the historical 534 only as a legacy sensitivity frame. The primary
mutant-audit frame must be derived mechanically after rerunning all 1,646 source
mutants, then frozen before outcomes are revealed and double-labelled in full
or through a preregistered gold subset.

### 3.3 Count 166

**Meaning:** the historical deterministic Phase-II stress suite recorded 166
additional kills among the 534 selected mutants.

**Validity:** `HISTORICAL_REPRODUCIBLE`, `REQUIRES_RERUN`.

The historical Phase-II launcher did not set its worker's optional
`sync_weights` flag.  The old fallback synchronizer was also not scientifically
safe because it could copy same-shaped tensors by order after strict loading
failed.  The revised run must use strict named state synchronization or an
explicit per-subject adapter, and must return `INCONCLUSIVE` when state
alignment cannot be established.

**Contract qualification:** nine of the 166 were historically config-only
findings. If varying that configuration is outside the frozen task contract,
the fixed-contract result is 157, not 166:

```text
166 total stress kills - 9 config-only = 157 in-contract stress kills
```

Correspondingly, the historical combined kill total is 1,105, while the
fixed-contract combined total would be 1,096 under this interpretation:

```text
939 + 166 = 1,105
939 + 157 = 1,096
```

The published 99.82% audited effectiveness is not an independently estimated
accuracy rate because the 166 detector positives were removed before the later
368-case audit. It must not remain a headline result without a blinded frozen
ground truth.

### 3.4 Legacy mutation-score interpretations

The following are protocol-dependent point estimates, not lower/upper bounds on
true correctness:

- `939 / 1,473 = 63.75%` after excluding stillborn and strict-equivalent
  outcomes but retaining candidate-equivalent outcomes;
- `939 / (939 + 270) = 77.67%` after additionally excluding the 264
  candidate-equivalent outcomes.

The second value is not an optimistic upper bound: later audit assumptions can
produce a value outside it. The FSE paper should call these denominator-specific
mutation scores and show unresolved cases explicitly.

### 3.5 KGB supplementary mutation study

The preserved KGB supplement records 54 subjects, 2,664 generated mutants, and
248 cases described by the historical pipeline as stress-rescued/detected.
These values are useful for estimating rerun cost and broadening the task list,
but they are not independent correctness labels.

**Validity:** `PILOT_ONLY`, `REQUIRES_RERUN`, `REQUIRES_HUMAN_AUDIT`.

The supplement used the same broad family of mutable execution/oracle
assumptions and an LLM-heavy equivalence pipeline. Before inclusion in FSE it
needs a content-addressed subject/mutant manifest, corrected state/RNG/input
semantics, the same three-valued oracle, frozen task contracts, and blinded
equivalence/defect calibration. Do not combine `248` with the main 166 or 222
counts: the populations, selection stages, and denominators differ.

Historical KGB claims such as `699 true escapes`, `248/35.5% rescued`, or
`0/272 proves the LLM EMD correct` are not ground-truth statements. The first
and third depend on LLM/heuristic equivalence decisions; the second is a pilot
detector outcome. Absence of a discovered counterexample is not an equivalence
proof.

### 3.6 Alternate 1,663-mutant funnel

Several pre-FSE design/result notes use a different funnel of 1,663 generated
mutants with counts such as 944 baseline kills, 322 survivors, and 234
candidate-equivalent cases. This does not match the preserved primary funnel of
1,646 total, 939 killed, 270 survived, 264 candidate-equivalent, 163 stillborn,
and 10 strict-equivalent mutants.

**Validity:** `PILOT_ONLY`, `DO_NOT_COMBINE`.

Before any alternate funnel is cited, identify its source commit and subject
manifest, account for the 17-mutant total difference, and publish a stable-ID
transition table showing every status change. Historical documents containing
these numbers are not canonical evidence.

## 4. Public-kernel funnel and count 222

The integrated historical RQ4 report gives this funnel:

| Stage | Count |
|---|---:|
| Collected public kernels | 831 |
| Completed | 767 |
| Baseline-positive among completed | 170 |
| Baseline-accepted | 597 |
| Stress-positive among baseline-accepted | 222 |

However, the four preserved per-dataset `summary.json` headers total 834
collected and 67 skipped (`241 + 229 + 142 + 222`, with skips
`10 + 7 + 4 + 46`), while the integrated report gives 831 collected and 64
skipped (`241 + 229 + 141 + 220`). Both retain 767 completed cases. Thus
`831/64` and `834/67` are competing historical cohort definitions. The one
additional TritonBench-G ID and two additional CUDA-Agent IDs must be joined by
stable content ID and explained before either collected count is canonical.

### Count 222

**Meaning:** 222 candidates passed the historical baseline and were flagged by
at least one historical stress dimension.

**Validity:** `HISTORICAL_REPRODUCIBLE`, `REQUIRES_RERUN`,
`REQUIRES_HUMAN_AUDIT`.

**Required terminology:** `222 stress-flagged candidates`.

**Prohibited terminology before audit:** `222 defective kernels`, `222 bugs`, or
an implied zero false-positive rate.

At least 12 alerts were historically dependent on a single contract-changing
dimension: five config-only and seven dtype-only cases. They must not be mixed
with in-contract defects unless the frozen task contract explicitly supports
those configurations or dtypes.

Every flagged case must receive one of the labels in
[HUMAN_CALIBRATION_PROTOCOL.md](HUMAN_CALIBRATION_PROTOCOL.md). A stratified
sample of the 375 stress-negative cases is also needed to estimate misses; alarm
auditing alone can estimate precision but not recall or population prevalence.

## 5. CUDA-Agent TaskD: counts 104 and 101

Two historical cohorts are currently inconsistent:

- RQ4 contains 101 CUDA-Agent candidates that are both baseline-pass and
  stress-positive under its recorded inclusion rules.
- TaskD uses 104 repair targets.

The current evidence indicates that the additional three TaskD targets do not
belong to the same baseline-pass/stress-positive cohort. Therefore:

### Count 101

**Meaning:** canonical historical RQ4 CUDA-Agent stress-positive subset under
the RQ4 filters.

**Validity:** `HISTORICAL_REPRODUCIBLE`, `REQUIRES_HUMAN_AUDIT`.

### Count 104

**Meaning:** the target list passed to the historical TaskD repair workflow.

**Validity:** `PILOT_ONLY` until all target IDs and inclusion reasons are joined
to the RQ4 manifest.

**Required reconciliation:** produce a table with one row per stable candidate
ID and columns for dataset version, baseline result, stress result, TaskD target
status, and exclusion reason. The three additional IDs must be named and
explained; no denominator may alternate between 101 and 104.

The historical repair report additionally records 90 framework-level `FIXED`
outcomes, but strict inspection classified only 15/104 as real custom-CUDA
repairs and 75/90 of the reported fixes as PyTorch fallback or equivalent
validator gaming. These are valuable pilot observations, not final rates, until
the 104/101 cohort mismatch is resolved and the experiment is rerun with a
structural no-fallback guard and an independent oracle.

## 6. Claim matrix

| Historical statement | Current disposition | FSE replacement |
|---|---|---|
| “MutaKernel detects 166 of 168 bugs” | `DO_NOT_CLAIM` without independent ground truth | Report detector outcomes against a frozen blind audit with a confusion matrix |
| “99.82% effectiveness” | `DO_NOT_CLAIM` because of sequential audit/resubstitution risk | Report task-clustered recall/precision with confidence intervals |
| “222 defective public kernels” | `DO_NOT_CLAIM` | “222 historically stress-flagged candidates; X independently confirmed in-contract defects” |
| “349 provably equivalent mutants” | `DO_NOT_CLAIM` unless machine proven | Separate machine-proven, likely equivalent, and inconclusive |
| “five orthogonal dimensions” | Unsupported | “five complementary dimensions,” followed by overlap and leave-one-out analysis |
| “104 CUDA-Agent stress misses repaired” | Cohort mismatch | Use reconciled stable-ID cohort and separate true CUDA repairs from fallback |
| “699 KGB true escapes” | `DO_NOT_CLAIM` because the label depends on heuristic/LLM EMD | Report independently audited KGB equivalence classes |
| “248 / 35.5% KGB rescued” | `PILOT_ONLY`, `REQUIRES_RERUN` | Report corrected, contract-qualified detector outcomes with audited labels |
| “0 / 272 proves LLM EMD correct” | `DO_NOT_CLAIM` | Treat failure to find a witness as inconclusive, not proof |
| “831 public kernels collected” | Cohort mismatch (`831/64` vs `834/67`) | Publish stable-ID reconciliation and one frozen corpus manifest |

## 7. Requirements for accepting a rerun result

A new result supersedes a historical result only when it records:

- immutable candidate/reference IDs and source hashes;
- Git commit and dirty-tree status;
- dataset version and registry hash;
- contract and oracle versions;
- complete environment fingerprint;
- GPU model, driver, CUDA, PyTorch, and Triton versions;
- policy list, seeds, budgets, timeout, and tolerances;
- isolated output directory with no imported old checkpoint;
- structured failure category;
- raw per-trial evidence and summary generated from that evidence;
- a reproducible command and result checksum.

Human-confirmed claims additionally require the double-annotation and
adjudication records defined in the human calibration protocol.

## 8. Preservation policy

Historical result directories must remain immutable. Corrections must be written
to a new run directory and linked through a provenance manifest. Do not edit old
JSON files to make totals agree, and do not combine old and new checkpoint rows
in one summary.
