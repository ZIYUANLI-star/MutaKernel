# MutaKernel FSE Revision Plan

## 1. Purpose and submission position

This document defines the evidence and engineering work required to turn the
rejected EuroSys submission into an FSE-style empirical software-engineering
paper. It is a research plan, not a claim that acceptance is guaranteed.

The revised paper will not present mutation as a prerequisite for validating a
new GPU kernel. It will separate two activities:

1. **Validator meta-evaluation:** use controlled mutants and independently
   labelled natural defects to measure which fault classes a validator detects.
2. **Candidate validation:** test a newly generated candidate directly against
   its reference under an explicit correctness contract. No mutation operator,
   original optimized kernel, or EMD label is available or required online.

The proposed paper position is:

> An independently audited empirical study of false acceptance in validators
> for LLM-generated GPU kernels, together with a budget-matched comparison of
> validation strategies and an assessment of whether mutation score predicts
> detection of natural LLM defects.

Suggested title:

> **When Passing Is Not Correct: An Empirical Study of Correctness Validation
> for LLM-Generated GPU Kernels**

## 2. Research questions

### RQ1: Natural false acceptance

Among real, baseline-accepted LLM-generated GPU kernels, how many have a
confirmed in-contract defect, which fault classes occur, and how do results vary
by dataset, language, and task level?

The primary outcome is a **confirmed false-acceptance lower bound**. Absence of
a discovered counterexample must not be described as a proof of correctness.

### RQ2: Budget-matched validator effectiveness

Under equal candidate-execution and GPU-time budgets, how do default random
testing, more random seeds, diversified random testing, boundary testing,
dtype/mode/configuration stress, and the MutaKernel policy suite compare?

Primary metrics are confirmed-defect recall, alarm precision, time to first
counterexample, GPU seconds per confirmed defect, and the detection-versus-cost
Pareto frontier.

### RQ3: Predictive validity of mutation testing

To what extent do the mutation taxonomy and mutation scores predict detection of
natural LLM-generated defects? Which mutation operators are representative,
which are not, and which natural fault classes are missing?

This RQ supplies the missing justification for mutation testing. A negative
result is acceptable if reported honestly: mutation score may be useful for
diagnosis without being a reliable proxy for natural-defect recall.

### Secondary case study: validator-induced repair behaviour

The CUDA-Agent TaskD repair experiment may be retained as an impact case study
only after reconciling the 104-target and 101-target cohorts. It must distinguish
real custom-kernel repair from PyTorch fallback or validator gaming. It is not a
primary RQ unless it is rerun with a frozen cohort, structural guard, multiple
seeds, and an independent final oracle.

## 3. Mapping from EuroSys reviews to FSE evidence

| Review concern | Required change | Evidence required before submission |
|---|---|---|
| A: mutation is insufficiently motivated relative to directly auditing real kernels | Make the real-kernel audit RQ1 and mutation predictive validity RQ3 | Independent labels for real baseline-accepted kernels; mapping between natural faults and mutation operators; held-out correlation analysis |
| A/B: EMD and LLM/human audit are heuristic and subjective | Treat EMD as a triage method to evaluate, not ground truth | Blinded double annotation, adjudication, agreement statistics, per-layer confusion matrix, model/prompt/threshold sensitivity |
| A: Section 5 conflates mutation evaluation with validation of new kernels | Define separate offline meta-evaluation and online candidate-validation paths | Candidate API and experiment protocol that use only candidate, reference, contract, and budget |
| A: baseline is too weak | Add simple and strong budget-matched baselines | Equal-execution and equal-GPU-second experiments; detection-cost curves; paired statistical tests |
| A/B: novelty overlaps robust-kbench, KernelBench-X, and ProofWright | Narrow novelty to empirical measurement, mutation predictive validity, and controlled comparison | Direct feature/protocol comparison; compatible reimplementations or clearly scoped subset comparisons |
| B: claims such as 168 buggy kernels depend on uncertain labels | Remove unaudited defect counts from headline claims | Frozen independent labels and replayable counterexamples; unknowns retained in the denominator |
| B: no effect on a real ADRS loop | Keep TaskD as a disciplined secondary case or omit the systems claim | Same model/token/candidate budget, independent correctness oracle, real-kernel/fallback distinction |
| B: limited to L1/L2 and mostly single kernels | Add a bounded generalisation sample | CUDA and Triton; at least one L3 and one multi-kernel or stateful/backward subset; task-level reporting |
| C: no computational or temporal overhead | Make cost a primary RQ2 outcome | Compile/reference/test breakdown, median/p95, GPU-hours, time to first defect, early-exit savings |
| C: reliance on Claude/DeepSeek harms reproducibility | Constrain LLMs to suggestion/triage and fully log their use | Versioned prompts and raw outputs; deterministic final labels; sensitivity across model/prompt where LLM triage is retained |
| C: taxonomy omits cross-stream ordering and will evolve | State the taxonomy boundary and maintenance protocol | Natural-fault coverage table, explicit unsupported classes, operator versioning, procedure for adding new operators |

## 4. Experimental design

### 4.1 Freeze contracts before observing validator outcomes

Every task must have a versioned contract that records:

- valid shapes and which dimensions may vary;
- supported dtypes;
- value domain and exceptional values;
- layout, contiguity, strides, and aliasing assumptions;
- evaluation/training mode and whether backward is in scope;
- determinism, RNG, and stream requirements;
- allowed device and architecture assumptions;
- output structure and dtype;
- dtype- and operation-aware numerical oracle.

Results must always separate:

1. `IN_CONTRACT_DEFECT`;
2. `EXTENDED_CONTRACT_FAILURE`;
3. `REFERENCE_OR_ORACLE_FAILURE`;
4. `INFRASTRUCTURE_FAILURE`;
5. `INCONCLUSIVE`.

### 4.2 Real-kernel audit corpus

Preferred design: independently audit the complete, stable-ID-reconciled
natural-kernel population corresponding to the historically baseline-accepted
frame (nominally 597), with every missing/excluded subject reported.

Minimum defensible design if resources are constrained:

- audit the union of alarms from every corrected validator/strategy, not only
  the historical 222;
- draw a preregistered stratified random sample only from subjects that complete
  every applicable planned test with no alarm;
- place never-run, partial, unsupported, and inconclusive subjects in separate
  missingness/audit strata;
- stratify by dataset/generator, CUDA/Triton, task level, and contract
  capabilities;
- use sampling weights and confidence intervals for population estimates.

Freeze two different plans. A population/applicability plan covers the complete
natural frame and records applicable, unsupported, missing, and inconclusive
cases per subject. A paired rich-contract plan contains only the preregistered
common subset eligible for every 32-call strategy. The current strict planner
implements the paired plan; per-subject applicability for the full population
remains an implementation gate.

The label protocol is specified in
[HUMAN_CALIBRATION_PROTOCOL.md](HUMAN_CALIBRATION_PROTOCOL.md).

### 4.3 Validator baselines

The comparison matrix should contain at least:

1. KernelBench default validation;
2. 10, 20, and 50 additional IID random seeds;
3. compute-matched IID random testing;
4. diversified value distributions;
5. a simple boundary-value suite;
6. a simple dtype/mode/configuration/repetition grid;
7. a robust-kbench-compatible protocol;
8. a KernelBench-X-compatible protocol;
9. the revised MutaKernel policy suite;
10. LLM self-review as a secondary, separately costed baseline;
11. Compute Sanitizer and ProofWright on supported subsets.

Every strategy must be compared under both an equal execution-count budget and
an equal GPU-second budget. A method must not receive credit merely for running
many more tests.

### 4.4 Mutation study

The mutation study must use task-level clustering and must not treat mutants
from one task as statistically independent programs. It will report:

- operator activation and generated-mutant counts;
- stillborn, killed, surviving, and unresolved counts;
- mutation score without silently removing uncertain equivalents;
- per-natural-fault coverage by mutation operators;
- leave-one-dataset/generator-out prediction of natural-defect recall;
- sensitivity to EMD rules, thresholds, prompts, models, and rounds;
- operators with zero or negligible support.

`provably equivalent` is reserved for machine-checkable proof. Human or LLM
judgement must use `likely equivalent` or `inconclusive`.

### 4.5 Statistics

- Use the task/kernel, not an individual mutant execution, as the main sampling
  unit.
- Use task-cluster bootstrap confidence intervals.
- Use paired tests such as McNemar where the same kernels are evaluated by two
  validators.
- Report effect sizes and 95% confidence intervals.
- Apply Holm correction to families of validator comparisons.
- Estimate dimension contribution with leave-one-dimension-out results, not
  first-kill counts alone.
- Freeze primary metrics and exclusion rules before the full run.

## 5. Engineering prerequisites

No full experiment should start until all of the following are satisfied:

1. one maintained runner implements reference/candidate testing;
2. reference and candidate module states, parameters, buffers, and RNG are
   controlled;
3. each execution receives isolated deep-cloned inputs;
4. the oracle checks structure, shape, dtype, NaN/Inf, and numerical tolerance;
5. timeout, OOM, compile failure, candidate crash, reference failure, and GPU
   health failure are distinct outcomes;
6. paths, output directories, GPU architecture, seeds, and budgets are
   configuration values rather than host-specific constants;
7. every run writes a canonical manifest with commit, data hashes, environment,
   contract version, policy version, and exclusions;
8. paper tables and figures are generated from canonical results;
9. unit and GPU integration tests pass;
10. historical checkpoints are never mixed with rerun results.

The state-control gate invalidates the historical shortcut of rerunning only
the 534 selected Phase-II mutants.  Phase I must be rerun from all 1,646 source
mutants, and the new Phase-II cohort must be derived exclusively from that
corrected run.

## 6. Remote GPU safety and isolation gate

GPU experiments execute generated Python and native GPU code and therefore must
be treated as untrusted-code execution.

- Rotate the historically exposed remote password before any login.
- Remove the secret from Git history; deleting it only from the current tree is
  insufficient.
- Use SSH keys and pinned host keys; do not use `AutoAddPolicy`.
- Do not run candidate kernels as root.
- Require an NVIDIA container with code/data mounted read-only, a writable
  run/cache directory, network disabled after setup, and memory/process limits.
- Execute the untrusted candidate outside the trusted reference/oracle/result
  writer and authenticate the request/result channel; process-per-case execution
  in one Python trust domain is not a reward-hacking sandbox.
- Do not provide Git, SSH, cloud, or LLM credentials to the execution container.
- Use one worker per GPU unless an explicit resource partition is validated.
- Detect compute capability instead of assuming 8.9.
- Disable `nvidia-smi --gpu-reset` on shared/cloud GPUs.
- Use a new run ID and output directory for every commit/configuration.
- Run CPU, CUDA-JIT, Triton, per-dataset, and small stratified smoke tests before
  a full run.
- Transfer results with checksums; do not transfer `.env` or compiler caches.

The existing remote launchers are historical artifacts and must not be used for
the final FSE runs without review: some contain fixed paths, fixed output
directories, destructive cleanup, unsafe concurrency defaults, or GPU-reset
logic.

## 7. Work packages and exit criteria

### P0: Evidence freeze and validity audit

- Create canonical IDs and manifests.
- Reconcile 939/534/166/222/104/101, the RQ4 `831/64` versus `834/67` corpus
  mismatch, the alternate 1,663-mutant funnel, and the KGB 54/2,664/248 study.
- Separate in-contract from extended-contract findings.
- Preserve raw historical files as immutable inputs.

Exit criterion: every historical headline number has an explicit validity label
as defined in [LEGACY_RESULT_VALIDITY.md](LEGACY_RESULT_VALIDITY.md).

### P1: Runner and oracle correction

- Consolidate duplicated workers.
- Correct state, input, RNG, dtype, and failure handling.
- Parameterise environment and result paths.
- Add CPU and GPU regression tests.

Exit criterion: all smoke tests pass and a replay bundle reproduces a known
positive and known negative without reading historical checkpoints.

### P2: Human calibration

- Pilot and freeze the codebook.
- Generate a policy-neutral evidence bundle with materialized counterexamples;
  neutral queue IDs alone do not blind the operator replay config.
- Double-label the real-kernel corpus and EMD gold set.
- Adjudicate disagreements and calculate agreement.

Exit criterion: all primary labels have two independent judgements or a recorded
adjudication; unresolved items remain `INCONCLUSIVE`.

### P3: Budget-matched experiments

- Run all primary baselines and MutaKernel on the same frozen corpus.
- Measure execution count, GPU time, and time to first defect.
- Run leave-one-dimension-out analyses.

Exit criterion: canonical result files produce all RQ2 tables with confidence
intervals.

### P4: Mutation predictive-validity analysis

- Map natural faults to the taxonomy.
- Evaluate held-out prediction and operator coverage.
- Run EMD sensitivity analysis.

Exit criterion: RQ3 can be answered without relying on an LLM label as truth.

### P5: Artifact and paper

- Anonymous artifact, locked environment, smoke mode, and full commands.
- Data-availability and generative-AI-use statements.
- Internal blind review and from-scratch artifact reproduction.

Exit criterion: a clean machine can reproduce the primary tables from released
data and rerun the documented smoke experiment.

## 8. Submission go/no-go gates

Submit the FSE paper only if:

1. the 222 observations are independently classified rather than called defects
   by default;
2. the 534-mutant EMD cohort has a non-circular gold-standard evaluation;
3. 104 versus 101 is reconciled by stable IDs and inclusion rules;
4. all primary comparisons are budget matched;
5. mutation predictive validity is evaluated on held-out natural faults;
6. all headline results are reproducible from one manifest;
7. remote execution uses the safety and isolation controls above;
8. unknown and inconclusive outcomes remain visible in results and denominators.
