# FSE code remediation record

## Status and scope

This record explains the code changes made after the EuroSys rejection, why
each change is necessary, which review concern it addresses, and whether it
invalidates historical results. It is intentionally stricter than a normal
change log: this project evaluates validators, so an unsound harness can turn
its own bugs into paper claims.

The frozen pre-rework version is Git tag `pre-fse-rework-20260719` at commit
`9581c46107df274366550171d9eabd1f871b0b55`. Development is on
`codex/fse-rework`. No document promises FSE acceptance; the objective is a
reviewable, reproducible evidence package that can support an FSE-level claim
after the required GPU runs and independent human labels are complete.

## What the audit found

The most consequential defects in the historical pipeline were:

1. Phase-I reference and mutant models could be initialized from different RNG
   states. Random parameters and buffers, especially in L2, could therefore be
   counted as mutation kills.
2. Phase-II did not enable its optional state synchronization, while its
   fallback could copy same-shaped state entries positionally rather than by
   exact name.
3. Reference and candidate executions could share an input object. An in-place
   reference or candidate could contaminate the other execution.
4. The legacy oracle coerced dtype and flattened/normalized some values, did
   not fail closed on unsupported structures, and used broad global tolerance.
5. Reference failures, candidate defects, timeouts, OOMs, compilation failures,
   and infrastructure failures were often collapsed into binary outcomes.
6. Mutation/EMD logic was entangled with the practical validator, creating the
   conceptual conflation identified by Reviewer A.
7. The 222 public-kernel alerts were not independently audited and included
   tests whose dtype/configuration may be outside the task contract.
8. Experiment paths, GPU architecture, remote credentials, and output
   directories were tied to an author's machine. A live credential was present
   in tracked history.
9. Existing summaries treated detector-selected material as if it were an
   independent gold set, and could incorrectly infer precision/recall at the
   subject level from a different test-level alarm.
10. The old runner discarded compiler logs and had no immutable counterexample
    replay bundle.

Consequently, the historical 1,646-mutant Phase I and its derived Phase-II
cohort require a full rerun. The historical 222 is retained only as
`stress-flagged`, not `defective`, until independent audit.

## Design decisions

### Separate offline meta-evaluation from online validation

`scripts/_candidate_worker.py` and the manifest-driven FSE runner accept only a
candidate, reference, frozen contract, policy case, and budget. They do not use
mutation generation or EMD. Mutation testing remains an offline method for
measuring validator sensitivity and studying whether synthetic mutants predict
natural LLM faults. This directly addresses Reviewer A's Section 5 objection.

### Use three-valued outcomes

Every maintained path uses `PASS`, `FAIL`, or `INCONCLUSIVE`:

- `PASS`: no discrepancy was observed in the executed case; never proof;
- `FAIL`: a concrete candidate-attributable discrepancy was observed after a
  valid reference execution; and
- `INCONCLUSIVE`: reference, oracle, contract, toolchain, resource, or
  infrastructure conditions prevent a sound verdict.

This prevents infrastructure failures from inflating defect counts.

### Make correctness contracts executable

A versioned schema now states supported tensor positions, symbolic/bounded
shapes, dtype, value domain, layout, gradient requirements, aliases, execution
mode, repetition, oracle strictness/tolerance, and explicit per-policy input
bindings. Planning and runtime both reject unauthorized cases. Schema v1 is
deliberately limited to top-level positional tensor arguments and the default
stream; nested/keyword/scalar constraints and concurrent-stream claims are out
of scope until a later schema implements them.

### Compare strategies under explicit budgets

The primary rich-contract matrix gives every budget-matched strategy a planned
budget of 32 candidate invocations per subject; actual invocations are recorded
and validated separately. The five-IID control is labelled a
historical-style anchor, not a native KernelBench reproduction. Repeated mode
performs repeated calls on the same model instance and counts both calls.
Equal-wall-time execution is supported, while native external tools and
protocol ports are kept as different rows.

### Treat human labels as independent data

The audit queue contains the union of all validators' alarms, every unresolved
execution subject, and a frozen stratified sample drawn only from fully
evaluated all-pass subjects. Public queue items use neutral IDs; strategy and
policy metadata live in a sealed mapping. Two primary annotators label before
adjudication. Alarm-level and subject-level labels are separate, and
`NO_DEFECT_FOUND` is never a true negative. Queue-ID blinding is implemented,
but the current replay bundle still exposes policy construction metadata; a
separate blinded evidence export is an annotation-start gate.

## File-by-file remediation

| Files | Change and reasoning | Review/validity link | Historical result impact |
|---|---|---|---|
| `.gitignore` | Excludes credentials, keys, virtual environments, caches, and generated `fse_runs/` artifacts. | Reproducibility and secret hygiene; remote-experiment safety. | None to values; prevents result/source mixing. |
| `config.py` | Replaces author-specific paths with repository defaults and `MUTAKERNEL_*` environment overrides; separates KernelBench task and run roots. | Reviewer B generality; Reviewer C operational detail. | Host-dependent reruns are not comparable without new manifests. |
| `scripts/_kgb_remote.py`, `docs/REMOTE_GPU.md` | Removes plaintext connection data, requires agent/key or ephemeral environment input, rejects unknown host keys, cleans unique temporary scripts. | Artifact safety and reproducibility. | No numerical impact; the leaked historical credential must still be rotated and purged from Git history. |
| `src/validation/types.py` | Defines three-valued results, structured mismatch/error records, and phase timings. | Prevents heuristic/binary overclaiming. | Old binary checkpoints cannot be mixed with new results. |
| `src/validation/state.py` | Captures/restores CPU/CUDA RNG and strictly synchronizes named parameters/buffers with exact key, shape, dtype, and layout checks. | Reviewers A/B: uncertain mutant classification and harness validity. | Invalidates historical Phase I and Phase II execution semantics. |
| `src/validation/inputs.py` | Deep-clones argument trees while preserving non-dense strides, offsets, overlap, and cross-argument aliases; adds replay metadata and exact logical-value hashes for non-passing cases. | Reviewer C's GPU edge cases; independent audit replay. | Old shared-input outcomes require rerun. |
| `src/validation/oracle.py` | Checks nested structure, shape, dtype, device/layout, optional stride and output-alias topology, exact integer/bool values, complex values, and explicit NaN/Inf positions. Unsupported values are inconclusive; mismatch truncation no longer hides later definite failures. | Reviewers A/B: validator novelty must rest on sound differential testing. | Old dtype-coercing/global-tolerance outcomes require rerun. |
| `src/validation/executor.py`, `src/validation/__init__.py` | Provides the paired reference/candidate executor with isolated inputs, replayed RNG, strict state control, attributed failures, actual invocation counters, and composite comparison of return values, post-call input/alias state, and post-call parameters/buffers. | Makes practical validation independent of mutation and closes false negatives caused by hidden side effects. | Defines the new canonical semantics. |
| `scripts/_candidate_worker.py` | New direct online validator; validates contracts before and after transformations, applies policies only to declared arguments, supports dtype/config/layout/train/backward cases, checks three deterministic VJPs, executes repeated calls as a continuous stateful sequence, catches candidate `BaseException`, and emits input evidence. | Reviewer A's mutation/validation separation; Reviewer B end-to-end story; Reviewer C nondeterminism and cost. | New evidence only; no old checkpoint reuse. |
| `scripts/_mutant_worker.py`, `src/mutengine/mutant_runner.py` | Replays constructor RNG, strictly synchronizes state, isolates inputs, uses the strict oracle and three-valued classification. Passing means only no observed divergence. | Reviewers A/B: mutation/EMD reliability. | Requires all 1,646 mutants to be rerun from source. |
| `scripts/_stress_worker.py` | Removes positional state fallback, isolates reference/original/mutant inputs, uses strict oracle and separates reference/original/infra failures from candidate witnesses. | Reviewers A/B heuristic uncertainty. | Historical 166 cannot be reused as final results. |
| `scripts/full_block12.py` | Exact source identity is the only automatic strict-equivalence shortcut; normalized identity and static rules are triage; dynamic timeout is inconclusive; LLM use is opt-in triage and cannot change the final status. SHA-256 replaces MD5. | Reviewers A/B/C: LLM-heavy EMD and reproducibility. | Historical EMD labels remain pilot labels and require sensitivity plus blind calibration. |
| `src/experiments/contract.py` | Adds strict contract schema, canonical normalization, policy applicability, symbolic-dimension/alias checks, and runtime input validation. | Reviewer A's boundary/config concern; prevents off-contract alerts. | Historical dtype/config-only alerts must be reclassified. |
| `src/experiments/strategy.py`, `src/experiments/budget.py`, `configs/fse_strategy_matrix.json` | Creates deterministic strategies and verifies actual candidate-call budgets. All budget-matched rows use 32 calls on a common eligible rich-contract subset. | Reviewer A's weak-baseline criticism; Reviewer C overhead. | Old unequal-budget comparisons are pilot only. |
| `src/experiments/manifest.py`, `src/experiments/protocol.py`, `scripts/build_fse_subject_manifest.py`, `scripts/plan_fse_experiment.py` | Hashes candidate/reference artifacts and contracts, creates stable IDs and an immutable planned schedule, rejects missing/inapplicable contracts, and re-derives every test ID, case cost, strategy identity, schedule order, and digest before execution or analysis. | Reviewers A/B reproducibility and corpus clarity. | Requires new frozen 597-kernel and mutant manifests. |
| `scripts/capture_fse_environment.py` | Captures Git state, Python/OS/dependencies, Torch, NVIDIA driver, NVCC, and selected non-secret environment fields with an embedded digest. | Reviewer C overhead/reproducibility. | Results from unrecorded historical environments remain pilot evidence. |
| `scripts/run_fse_experiment.py` | Verifies live environment, Git cleanliness, subject/plan/source and implementation hashes; isolates caches per strategy; supports resume, filters, wall budgets and online early stop; preserves hashed logs and immutable replay bundles for every non-pass; and emits artifact-root-relative replay commands. | All reviews: end-to-end credible validation and cost measurement. | New canonical result format; old JSON cannot be appended. Fresh-container replay still requires GPU qualification. |
| `src/experiments/timing.py` | Normalizes timing records and summary inputs. | Reviewer C's missing latency/resource analysis. | New runs required. |
| `scripts/summarize_fse_results.py` | Summarizes canonical observations without rerunning; separates test alarms from subject labels; computes detection coverage only from same-strategy confirmed alarms; keeps unaudited/inconclusive counts visible. | Reviewers A/B audit uncertainty. | Published binary claims cannot be regenerated from the corrected schema without labels. |
| `scripts/analyze_fse_statistics.py` | Verifies the run-manifest/plan/observation/audit provenance chain, complete planned budgets, actual invocation-count bounds, complete-audit status, and the report's exact ordered label-set digest; clusters by canonical `(dataset, task_id)`; and computes deterministic bootstrap intervals, exact paired McNemar tests, Holm correction, audited alarm precision, inconclusive sensitivity bounds, and detection/cost Pareto analysis. Early-stopped, wall-censored, incomplete, or partial-audit primary data fail closed. | FSE empirical-method standard; Reviewer A baseline comparison. | Supersedes independent-mutant/Wilson-style inference for primary comparisons. |
| `scripts/build_human_audit_queue.py` | Binds observations to the exact run manifest, frozen plan, subjects, contracts, test identities, and planned costs; distinguishes never/partially/inconclusively/fully evaluated subjects; includes all alarms and unresolved subjects; samples negatives only from fully evaluated all-pass subjects; and seals exact queue, observation, and detector-mapping digests. | Reviewers A/B: independent/blinded audit. | Historical detector-selected audit is not reused as gold; replay evidence still needs a policy-neutral blinded export. |
| `scripts/compile_human_audit.py` | Requires a complete queue by default, verifies exact sealed IDs/digests, enforces a fixed independent primary pair and separate adjudicator, validates scope-specific labels plus fault/contract/replay fields, preserves all A/B/C records, and calculates pre-adjudication agreement/kappa/confusion before emitting immutable analysis labels. | Reviewers A/B: inter-reviewer agreement and subjectivity. | Manual labels still need to be produced. |
| `src/stress/policy_bank.py` | Makes policies dtype-aware, finite, deterministic, recursively storage-preserving, and compatible with gapped/overlapping layouts. Per-argument applicability is enforced by contracts. | Reviewer A stronger simple baselines; Reviewer C GPU edge cases. | Policy behavior changed; stress results require rerun. |
| `src/stress/differential_tester.py` | Uses configurable counts and correct execution timing instead of placeholder zero times. | Reviewer C overhead. | Old time fields are not valid overhead evidence. |
| `src/bridge/eval_bridge.py`, `scripts/run_l1_experiment.py` | Separates problem and runs roots and fixes portable bridge construction. | Reviewer B generality and artifact portability. | Existing local data are preserved; rerun uses resolved manifests. |
| `src/experiments/kernel_registry.py`, `scripts/full_block12.py`, `scripts/run_baselines.py`, `scripts/run_ablation.py`, `scripts/run_stress_enhance.py` | Migrates legacy absolute kernel paths to portable identities under the configured runs root and fails closed on missing targets, traversal, symlink escape, duplicate keys, metadata mismatch, or cross-platform collision. The four maintained legacy experiment entry points use the same validated resolution table. | Reviewer B generality; artifact portability and cohort integrity. | The existing 90-entry registry can be migrated, but this workstation lacks its referenced run artifacts; formal execution therefore stops instead of silently changing the cohort. |
| `scripts/run_baselines.py`, `scripts/run_ablation.py`, `scripts/run_stress_enhance.py`, `scripts/run_external_diff_test.py`, `scripts/run_fullscale_diff_test.py` | Removes maintained hard-coded roots/output assumptions, records real timing, and makes destructive GPU reset opt-in. | Reviewer C systems operation and cost. | These remain legacy launchers; the canonical FSE runner is preferred. |
| `scripts/_recount_taskA_368.py` | Converts an accidentally UTF-16/NUL-containing source file to valid UTF-8 without semantic changes. | Artifact buildability. | No numerical impact. |
| `configs/external_baselines.json`, `docs/RELATED_BASELINE_INTEGRATION.md` | Pins robust-kbench and KernelBenchX official commits, distinguishes native runs from protocol ports, and records that ProofWright has no runnable public artifact at the freeze date. | Reviewers A/B/C novelty and stronger baselines. | No external baseline result is claimed yet. |
| `pytest.ini` | Restricts collection to maintained `tests/test_*.py`, preventing vendored/historical scripts from being mistaken for the test suite. | Reproducible artifact verification. | None. |
| `tests/test_validation_core.py`, `tests/test_candidate_worker.py`, `tests/test_mutant_runner_soundness.py`, `tests/test_mutant_worker_soundness.py`, `tests/test_stress_worker_soundness.py`, `tests/test_full_block12_soundness.py`, `tests/test_policy_bank_soundness.py`, `tests/test_differential_tester.py`, `tests/test_configuration.py`, `tests/test_remote_security.py`, `tests/test_operators.py` | Regression tests for state/RNG/input/oracle semantics, crash attribution, real repetition, EMD downgrading, policy geometry, timing, portability, SSH safety, and operator importability. | Tests the exact failure modes behind the rejected claims. | Required gate before GPU qualification. |
| `tests/experiments/*` | Tests contracts, stable manifests, planning, budgets, provenance/resume, evidence bundles, audit blinding/compilation, summaries, statistics, and timing. | FSE artifact and empirical-analysis quality. | Required gate before primary runs. |

## What is not yet claimed as solved

The following are explicit experiment gates, not hidden implementation details:

- Contract schema v1 excludes nested/keyword/scalar constraints and
  non-default/concurrent stream semantics.
- Backward checks three frozen VJPs, but finite sampling is not a proof of the
  full Jacobian. Broad gradient-equivalence claims still require a qualified
  `gradcheck`/finite-difference subset or formal reasoning.
- Candidate and trusted reference/oracle code still execute in one Python/OS
  trust domain. A malicious candidate can tamper with the harness or evidence.
  Primary reward-hacking claims are blocked until candidate execution uses a
  no-network, no-credential sandbox and a separate trusted controller with
  read-only sources and authenticated IPC/results.
- The process-per-case runner is appropriate for cold-path evidence and crash
  containment, but not a final steady-state ADRS overhead estimate. The overhead
  study needs a compile-once/session runner or must report cold and steady-state
  cost separately.
- Official KernelBench, robust-kbench, and KernelBenchX adapters/parity tests
  are not complete. ProofWright cannot be executed without its upstream
  artifact.
- A complete natural-kernel population plan and the paired rich-contract plan
  must be frozen separately. The current planner fully supports the latter; it
  intentionally rejects a 597-subject manifest if any subject lacks one of the
  matrix capabilities and therefore does not yet implement per-subject
  applicability/missingness for the population census.
- The integrated RQ4 report and preserved per-dataset summaries disagree on
  collected/skipped counts (`831/64` versus `834/67`) while both report 767
  completed cases. The three TritonBench-G/CUDA-Agent IDs require stable-ID
  reconciliation before either funnel is canonical.
- L3/multi-kernel/stateful generalization, a second GPU architecture, mutation
  predictive validity, EMD sensitivity, and TaskD 104/101 reconciliation still
  require data.
- Queue IDs/mapping are blinded, but replay configs expose policy, seed, mode,
  and parameters and do not contain independently loadable materialized input
  tensors. A policy-neutral, replayable blinded bundle must be implemented and
  qualified before annotation begins.
- Current queue/compiler automation covers Population-A natural-kernel subject
  and alarm labels only. Mutant equivalence, taxonomy coding, and TaskD repair
  populations still require dedicated schemas/tools.
- Human annotations have not been performed. See
  `docs/MANUAL_VERIFICATION_TODO.md`.
- Remote GPU execution is blocked until the exposed credential is rotated,
  Git history is cleaned with repository-owner authorization, a non-root
  isolated account/container is available, and the host key is pinned.

These gates are deliberately visible so a future paper cannot turn unfinished
engineering into unsupported prose.
