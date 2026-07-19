# Human Calibration and Independent Audit Protocol

## 1. Objective

This protocol defines how human judgement will establish labels for real
LLM-generated GPU kernels and equivalent-mutant evaluation. Its goals are to
avoid circular validation, quantify annotator uncertainty, separate
in-contract defects from robustness extensions, and leave a reproducible audit
trail.

LLM output may assist search and explanation, but it is not ground truth.

## 2. Audit populations

Human audit has four separate populations and must not merge their labels or
reuse one population as ground truth for another.

### Population A: real generated kernels

- Primary frame: the canonical, contract-qualified rerun of the 597
  historically baseline-accepted public kernels, with missing subjects and
  exclusions reported explicitly.
- Preferred audit: every successfully materialized subject.
- Minimum design: the union of alarms from every compared validator/strategy,
  plus a preregistered stratified sample drawn only from subjects that completed
  every planned test with no alarm. Partial, never-run, and all-inconclusive
  subjects are separate missingness categories, not negative examples.
- Stratification: dataset, generator, CUDA/Triton, task level, and frozen
  contract capabilities.

This population estimates natural false acceptance and validator performance.

### Population B: mutants for EMD calibration

- Frame: the cohort mechanically derived after rerunning all 1,646 source
  mutants with the corrected Phase-I protocol. The historical 534 is a pilot
  frame and must not silently define the new cohort.
- Preferred audit: the complete newly derived cohort.
- Resource-constrained design: a preregistered stratified gold subset covering
  all operators, EMD layers/statuses, task levels, and stress outcomes.
- Auditors must not know whether a mutant belonged to the historical 166
  stress-positive or 368 later-audited subsets.

This population evaluates EMD and mutation representativeness. It must not be
used as a substitute for the natural-kernel population.

### Population C: taxonomy representativeness

Independently double-code a frozen sample of the 1,020 historical failures and
the confirmed natural defects. Publish the open-coding procedure, codebook,
saturation criterion, agreement, adjudication, multiple-class policy, and a
`not represented` category. This population tests whether the 16 mutation
operators resemble natural LLM faults.

### Population D: TaskD cohort reconciliation

Before auditing repair outcomes, construct a stable-ID join between the 101 RQ4
CUDA-Agent stress-positive candidates and the 104 historical TaskD targets. The
three additional TaskD IDs must be explicitly identified and classified. Repair
rates cannot be final until this join is complete.

## 3. Roles and blinding

Each primary item is independently labelled by:

- **Auditor A:** GPU/CUDA-capable reviewer;
- **Auditor B:** a second GPU/CUDA-capable reviewer;
- **Adjudicator C:** resolves disagreements after A and B submit locked labels;
- **Experiment operator:** prepares replay bundles but does not assign the final
  label unless separately acting as a declared auditor.

Auditors A and B must not see:

- which policy or validator raised an alert;
- the other auditor's label or rationale;
- historical paper labels such as “buggy,” “equivalent,” or “killed”;
- the detector's confidence or EMD tier;
- whether an item is expected to support a paper claim.

Items should be presented in randomized order with neutral IDs. The same fixed
pair of primary auditors must label the complete queue because the compiler's
Cohen-kappa calculation enforces a fixed pair. The reference, candidate or
mutant, frozen contract, and policy-neutral replay evidence are visible because
they are necessary for correctness judgement.

The current implementation blinds queue IDs and stores detector identity in a
sealed mapping, but its operator replay config still reveals policy, seed, mode,
and parameters. It also stores an input descriptor/hash rather than an
independently loadable materialized tensor set. Primary annotation must not
start until a separate blinded bundle removes these fields while retaining a
verified, replayable counterexample.

## 4. Contract-first procedure

The task contract must be frozen before revealing stress outcomes. It records:

- valid input shapes and variable dimensions;
- dtypes;
- value domain, NaN/Inf policy, and exceptional cases;
- layout, stride, contiguity, and aliasing rules;
- module mode and backward/gradient requirements;
- RNG, determinism, and stream requirements;
- architecture/device restrictions;
- output structure, shape, dtype, and tolerance policy.

If the public task does not state a property, auditors must not silently infer
an expansive contract merely because a stress policy tests it. Contract disputes
are recorded and adjudicated separately from implementation correctness.

## 5. Label codebook

### 5.1 Real-kernel labels

Each real candidate receives exactly one primary label:

| Label | Definition |
|---|---|
| `CONFIRMED_IN_CONTRACT_DEFECT` | A replayable input allowed by the frozen contract makes the candidate disagree with a valid oracle, crash, violate output structure/dtype, or violate a required deterministic/safety property |
| `EXTENDED_CONTRACT_FAILURE` | Failure occurs only outside the frozen contract, such as an unsupported dtype or shape |
| `REFERENCE_OR_ORACLE_FAILURE` | The reference, tolerance, expected output, or comparison procedure is invalid or insufficient |
| `INFRASTRUCTURE_FAILURE` | Toolchain, timeout, OOM, driver, dependency, or harness failure prevents a correctness conclusion |
| `NO_DEFECT_FOUND` | The reviewed evidence contains no confirmed defect; this is not proof of correctness |
| `INCONCLUSIVE` | Evidence is conflicting or insufficient after the prescribed investigation budget |

Secondary fields record fault class, affected contract clause, confidence, and
whether the issue reproduces on a second GPU.

### 5.2 Alarm-level labels

Each concrete alarm receives exactly one label from the compiler codebook:

| Label | Definition |
|---|---|
| `CONFIRMED_IN_CONTRACT_DISCREPANCY` | The exact alarm witnesses a candidate-attributable violation of the frozen contract |
| `CONFIRMED_EXTENDED_CONTRACT_DISCREPANCY` | The alarm is reproducible only in the explicitly labelled extended-contract scope |
| `INVALID_INPUT` | The test input or transformation is not authorized by the applicable contract |
| `REFERENCE_OR_ORACLE_FAILURE` | Reference or oracle invalidity prevents attributing the alarm to the candidate |
| `INFRASTRUCTURE_FAILURE` | Toolchain, timeout, OOM, driver, dependency, or harness failure produced the alarm |
| `INCONCLUSIVE` | The alarm cannot be classified within the frozen investigation budget |

The compiler rejects an in-contract alarm labelled as extended-contract and the
reverse.

### 5.3 Mutant/EMD labels

| Label | Definition |
|---|---|
| `NON_EQUIVALENT` | A replayable in-contract input distinguishes original and mutant, or a sound semantic argument establishes a difference |
| `EQUIVALENT_MACHINE_PROVEN` | Equivalence is established by a checkable proof or exhaustive argument over the stated finite domain |
| `LIKELY_EQUIVALENT` | Two auditors accept a documented semantic argument, but no machine proof exists |
| `INCONCLUSIVE` | Equivalence or non-equivalence cannot be established within the audit budget |

`provably equivalent` may be used only for
`EQUIVALENT_MACHINE_PROVEN`. LLM agreement, repeated passing tests, or failure to
find a counterexample is insufficient.

### 5.4 Repair labels

TaskD repair outcomes use:

- `REAL_CUSTOM_KERNEL_REPAIR`;
- `PARTIAL_CUSTOM_KERNEL_REPAIR`;
- `PYTORCH_OR_LIBRARY_FALLBACK`;
- `DEAD_OR_UNUSED_CUSTOM_KERNEL`;
- `VALIDATOR_SPECIFIC_GAMING`;
- `NOT_FIXED`;
- `INCONCLUSIVE`.

Framework-reported `FIXED` is retained as an observed field, not used as the
human label.

## 6. Evidence bundle required per item

Before annotation starts, each neutral audit ID must link to an immutable,
policy-neutral bundle containing:

- candidate/reference/original/mutant source hashes as applicable;
- dataset and task stable IDs;
- Git commit and contract version;
- environment manifest;
- materialized/serialized input tensors (including storage/alias metadata) or a
  neutral replay program, plus an input hash; detector, strategy, policy, and
  seed identities remain sealed;
- dtype, shape, stride, layout, mode, and stream metadata;
- raw outputs and structured diff;
- tolerance/oracle configuration;
- stdout/stderr, compiler output, and failure category;
- a one-command replay script;
- minimized counterexample if available;
- results from a second run and, for every headline claim, a second GPU.

The present runner does not yet produce this complete blinded bundle. Its
operator bundle and artifact-root-relative replay command must first pass a
fresh-container qualification; integrity of the operator bundle alone is not
evidence that blinding is complete.

A timeout, OOM, or process crash is not automatically a candidate defect. The
bundle must show that the reference succeeds under the same valid contract and
that infrastructure failure has been excluded.

## 7. Calibration and codebook freeze

1. Select 20–30 pilot items covering datasets, languages, alert statuses, and
   mutation categories.
2. Auditors label them independently using a draft codebook.
3. Discuss disagreements to clarify definitions, not to optimise agreement.
4. Revise the codebook and version it.
5. Re-label the pilot under the frozen codebook.
6. Lock the codebook before the main corpus is opened.

If a genuinely new ambiguity is discovered later, record an amendment, its date,
and all affected items. Revisit the complete affected stratum rather than only
changing convenient cases.

## 8. Main annotation workflow

For each item:

1. verify that the evidence bundle and contract are complete;
2. reproduce the canonical test;
3. inspect source and the claimed counterexample without detector metadata;
4. attempt one independent replay or alternative oracle where required;
5. assign the primary label, fault class, confidence, and rationale;
6. lock the label;
7. after both labels are present, send disagreements to adjudication;
8. preserve A, B, and C labels rather than overwriting them with the final one.

An investigation budget must be fixed in advance, for example a maximum number
of analyst minutes and replay attempts. Exceeding it yields `INCONCLUSIVE`, not a
forced binary decision.

## 9. Agreement and reporting

Report at least:

- raw A/B agreement;
- Cohen's kappa for the primary mutually exclusive labels;
- Krippendorff's alpha if missing labels or multiple auditors require it;
- per-label agreement and the full confusion table;
- number and percentage adjudicated;
- agreement before and after codebook calibration;
- confidence distribution;
- agreement separately for real kernels and mutants.

Do not report agreement only after adjudication. Adjudicated labels form the
analysis ground truth, while the pre-adjudication measurements quantify
subjectivity.

For stratified sampling of stress-negative kernels, publish stratum sizes,
sample sizes, selection seeds, weights, and weighted confidence intervals.

## 10. Permitted use of LLMs

An LLM may:

- suggest counterexample inputs;
- summarize candidate/reference differences;
- propose a fault category;
- help minimize a known counterexample;
- draft a rationale for a human to verify.

An LLM may not:

- assign the final primary label;
- establish `EQUIVALENT_MACHINE_PROVEN`;
- resolve an A/B disagreement;
- silently remove an item from a denominator;
- see hidden detector outcomes during blinded annotation.

Every LLM-assisted action records provider, model/version, prompt hash, raw
response, temperature/sampling parameters, date, and whether the suggestion was
executed successfully. Human auditors must be able to ignore the suggestion and
reach the label from replayable evidence.

## 11. Remote replay safety

Audit replay executes untrusted Python and native GPU code. The following are
mandatory:

- rotate the historically exposed remote credential before access;
- use key-based SSH with pinned host keys;
- run as a non-root user inside an isolated NVIDIA container;
- mount source and datasets read-only;
- expose only a writable per-run result/cache directory;
- disable network access after dependency installation;
- do not mount `.env`, SSH keys, Git credentials, or LLM/cloud tokens;
- use one worker per GPU unless memory isolation is demonstrated;
- detect GPU architecture dynamically;
- apply process, memory, and wall-clock limits;
- distinguish candidate failure from reference, harness, timeout, OOM, and GPU
  failures;
- never use a shared/cloud `nvidia-smi --gpu-reset` recovery action;
- use new checkpoints for every code/config version;
- record hashes before returning evidence to auditors.

The historic remote helper/launcher outputs are not audit evidence unless their
commit, environment, and exact result files are preserved in a compliant bundle.

## 12. Audit data schema

The current Population-A compiler accepts one JSONL row per auditor/item with
at least:

```json
{
  "audit_id": "neutral-id",
  "auditor_id": "pseudonymous-id",
  "annotation_role": "primary",
  "primary_label": "...",
  "confidence": "high|medium|low",
  "fault_class": "...",
  "contract_clause": "...",
  "evidence_reproduced": true,
  "rationale": "..."
}
```

Raw governance records should additionally retain codebook/contract versions,
timestamps, investigation minutes, counterexample ID, and second-GPU status.
The current compiler validates and preserves the correctness fields above but
does not yet implement the mutant/EMD, taxonomy, or repair vocabularies; those
populations require separate versioned tools before their labels are analysed.

Detector outcomes and policy identities live in a separate sealed table and are
joined only after A/B labels are locked.

## 13. Items explicitly deferred to human calibration

Until this protocol is completed, the following must remain marked pending:

- confirmation or rejection of every alarm in the corrected all-strategy union;
- labels for the preregistered fully evaluated non-alarm sample (or complete
  natural-kernel frame);
- final equivalence status for the corrected Phase-II mutant cohort or its
  preregistered gold subset;
- classification of the three IDs causing the TaskD 104 versus RQ4 101 mismatch;
- final classification of reported TaskD fixes as real repair or fallback;
- any contract dispute involving dtype, configuration, mode, layout, backward,
  nondeterminism, or stream semantics;
- any reference/oracle-failure decision.

The final paper and generated tables must fail closed if these fields are absent:
pending items remain pending and are never converted to defects, correct kernels,
or equivalent mutants by default.
