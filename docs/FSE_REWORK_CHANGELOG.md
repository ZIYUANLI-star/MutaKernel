# FSE rework change log

This document records FSE-rework milestones. The exhaustive file-by-file
rationale and historical-result impact live in
`docs/FSE_CODE_REMEDIATION.md`; the executable sequence and unresolved gates
live in `docs/FSE_EXPERIMENT_RUNBOOK.md`. Historical results are not silently
overwritten; rerun results must carry a new manifest and provenance record.

## Frozen baseline

- Baseline commit: `9581c46107df274366550171d9eabd1f871b0b55`
- Original branch: `main`
- Development branch: `codex/fse-rework`
- Freeze date: 2026-07-19

The baseline commit was already present on `origin/main` before the rework.
All subsequent changes are developed on the dedicated branch so that the
original experiment state remains identifiable.

## 2026-07-19: remote credential and SSH hardening

### Files changed

- `scripts/_kgb_remote.py`
- `.gitignore`
- `docs/REMOTE_GPU.md`

### What changed

- Removed hard-coded remote host, account, and password from source code.
- Read SSH connection settings from process environment variables.
- Prefer SSH agent or an explicit private key; password authentication is
  supported only through a non-persisted environment variable.
- Replaced Paramiko's trust-on-first-use `AutoAddPolicy` with strict host-key
  checking.
- Use a unique temporary remote script name and remove it after execution.
- Ignore common private-key formats, local environment files, and local Claude
  settings.

### Why this change was necessary

The frozen baseline contained a live remote credential in a tracked Python
file.  This is both a security incident and an artifact-reproducibility defect:
reviewers must be able to inspect and reuse the artifact without inheriting an
author-specific account or secret.  The credential must be rotated separately;
removing it in a later commit does not remove it from Git history.

### Relation to the reviews and FSE revision

The EuroSys reviews questioned reproducibility, operational completeness, and
the end-to-end systems story.  FSE's open-science expectations make safe,
portable configuration and explicit provenance mandatory.  This change is a
prerequisite for any remote rerun and for an anonymized replication package.

### Validation

- `python -m py_compile scripts/_kgb_remote.py`
- repository working-tree scan for password/token assignments and private-key
  headers (file-name-only reporting; no secret values logged)

## 2026-07-19: validator, protocol, audit, and statistics reconstruction

### Files changed

- `src/validation/*`, `src/experiments/*`, and `src/stress/*`
- `scripts/_candidate_worker.py`, corrected mutation/stress workers, and the
  manifest-driven planning/runner/audit/statistics scripts
- `configs/fse_strategy_matrix.json` and `configs/external_baselines.json`
- maintained legacy experiment entry points and the portable kernel registry
- FSE revision, validity, human-calibration, baseline, remediation, runbook,
  and manual-work documentation
- CPU regression tests under `tests/` and `tests/experiments/`

### Behavioral change

- Separates mutation/EMD meta-evaluation from direct candidate validation.
- Replaces binary/coercing validation with three-valued, state/RNG-controlled,
  storage-aware structured differential testing, including return values,
  input side effects, state trajectories, aliases, and three backward VJPs.
- Introduces executable contracts, deterministic test identities, strict frozen
  plan revalidation, planned/actual invocation accounting, immutable run
  manifests, hashed logs, and portable non-pass replay bundles.
- Builds run-bound neutral-ID audit queues containing all alarms, execution-
  unresolved subjects, and fully evaluated non-alarm samples; compiles a fixed
  A/B pair plus independent adjudication; binds the exact label collection to
  the run/plan/observation set; and rejects partial audit in paper statistics.
- Adds task-clustered bootstrap intervals, paired exact McNemar tests, Holm
  correction, audited precision, inconclusive sensitivity, and cost/Pareto
  outputs.
- Reclassifies all historical headline counts as pilot/rerun/audit material and
  records the 831/834, 1,646/1,663, 104/101, and KGB cohort conflicts.

### Why and review linkage

This change directly addresses Reviewer A's mutation-versus-online-validation
conflation and weak baselines; Reviewers A/B's EMD and LLM/human ground-truth
uncertainty; Reviewer B's missing end-to-end empirical story; and Reviewer C's
missing overhead, reproducibility, and operational detail. The contribution is
now framed as an independently audited empirical methodology, not a claim of a
fundamentally new proof technique.

### Validation

- `231 passed in 49.92s` for the complete maintained CPU suite
- Python compilation of `config.py`, `scripts`, `src`, and `tests`
- `git diff --check`
- focused evidence/audit/statistics regression suite: `41 passed`

### Result validity and remaining gates

No historical numerical result is promoted by this change. All 1,646 source
mutants, the newly derived Phase-II cohort, public kernels, and any retained KGB
supplement require clean GPU reruns and independent labels. Primary execution
is blocked by the exposed credential, missing trusted-controller sandbox,
unfinished native baseline adapters, missing population/applicability planner,
and missing policy-neutral auditor evidence export.

## 2026-07-21: MutakernelV2 module completion (fingerprint, verdict, audit map, site-directed, blinding)

### Files changed

- `src/mutengine/fault_classes.py`, `src/mutengine/fingerprint.py` (new)
- `src/cse/__init__.py`, `src/cse/verdict.py` (new)
- `src/audit/__init__.py`, `src/audit/ripr.py`, `src/audit/mapbuild.py` (new)
- `src/validator/__init__.py`, `src/validator/site_directed.py` (new)
- `src/stress/policy_metadata.py` (new)
- `scripts/export_blind_bundles.py` (new)
- `tests/test_site_fingerprint.py`, `tests/test_cse_verdict.py`,
  `tests/test_audit_ripr_map.py`, `tests/test_site_directed_plan.py`,
  `tests/test_blind_bundle_neutralize.py`, `tests/test_policy_metadata.py` (new)

### Behavioral change

- Adds the machine-readable 16-operator-to-fault-class taxonomy and a
  zero-execution static site fingerprint for candidate kernels.
- Adds the five-valued three-way differential verdict
  (SPEC_VIOLATION / EXACT_DIVERGENCE_ONLY / INDISTINGUISHED /
  INVALID_INPUT / ACCIDENTAL_REPAIR / INCONCLUSIVE) as a shared torch-free
  function, including a legacy-record reinterpreter for the historical
  impact recount.
- Adds the RIPR escape-mechanism classifier and the FaultToStressMap
  builder that aggregates audited observations into the versioned
  offline-to-online bridge artifact.
- Adds deterministic site-directed stress-plan derivation
  (directed 70% by map closure rate + general 30% fallback; contract-gated;
  stable hash-derived seeds).
- Adds the policy-neutral blinded evidence exporter
  (blind bundle + sealed mapping) that removes policy/seed/strategy identity
  while preserving execution context and materialized inputs.

### Rationale and review linkage

Implements the MutakernelV2 design documents 方法V2_00/03/06/07/08/09.
Site-directed selection answers Reviewer A's "why mutation" with a
per-candidate mechanism at zero online probe cost; the five-valued verdict
separates exact divergence from specification violation (Reviewers A/B
equivalence-noise concern); the blind exporter closes the annotation-start
gate identified in `MANUAL_VERIFICATION_TODO.md`.

### Validation

- `python -m pytest tests/test_site_fingerprint.py tests/test_cse_verdict.py
  tests/test_audit_ripr_map.py tests/test_site_directed_plan.py
  tests/test_blind_bundle_neutralize.py tests/test_policy_metadata.py`
  → 29 passed, 1 skipped (torch unavailable on the authoring workstation;
  the skipped test re-checks policy-registry consistency where torch exists)
- `python -m compileall` over all new modules.

### Result validity

No historical number changes. The legacy reinterpreter enables the P0
bitwise-kill recount but has not yet been run against historical detail
JSONs. Planner integration for a `site-directed` strategy-matrix row and the
audit-harness rewrite remain open (see the pending-work list in
`FSE_CODE_REMEDIATION.md`).

## 2026-07-21 (later): decision closure and RNG double insurance

### Decisions taken (recorded in the V2 method docs)

1. **ref-NaN handling = merged scheme.** Runtime oracle keeps NaN-position-
   aware comparison (`equal_nan=True` with position matching); a kill under
   reference NaN requires the original to reproduce the exact NaN pattern
   while the mutant does not.  Contract value-domain exclusions
   (`ref_nan_inducing`) additionally remove known NaN-inducing inputs at
   planning time.  方法V2_01 §3.5 and 方法V2_06 §3.3 updated to match the
   implemented semantics.
2. **RNG replay double insurance: adopted now.**
3. **`site-directed` planner integration: deferred to P2** together with the
   audit harness, because it consumes the FaultToStressMap artifact which
   only exists after the Phase I/II rerun.  The derivation pure function is
   implemented and test-locked; preregistration freezes (rule version, map
   version, fingerprint version).
4. **70/30 split and the 12-case general fallback sequence: frozen as v1
   defaults** with planned 50/50 and 90/10 sensitivity rows.

### Files changed

- `scripts/_stress_worker.py`: added `_capture_init_rng` /
  `_instantiate_model`; all five paired model-construction sites
  (`run_stress`, `_build_models`, `run_training_stress`, `run_llm_verify`,
  `run_config_stress`) now replay the pre-reference RNG snapshot before
  constructing original/mutant models.  Strict named state sync remains the
  second layer; the replay additionally aligns construction-time randomness
  that never reaches the state dict.
- `tests/test_stress_worker_soundness.py`: new regression test
  `test_paired_construction_replays_identical_rng_entropy` (plain-attribute
  randomness must match across ref/original/mutant).

### Validation

- Full maintained CPU suite in WSL Ubuntu-22.04 (`pilot_env`, torch 2.5.1):
  `262 passed`.
- Fingerprint smoke test on three real CUDA-Agent kernels (pure static,
  no execution): fault classes detected, no scan errors.

## 2026-07-22: E1–E4 readiness build-out (while E0 run5 executes)

### Files changed

- `src/mutengine/static_equiv_rules.py`: versioned the four static
  equivalence rules (`RULE_VERSIONS`, `rules_content_version()` content
  hash) and added `machine_proof()` implementing the Table 10 vocabulary
  (byte-identical | versioned static rule; anything weaker is at most
  LIKELY_EQUIVALENT).
- `src/audit/inconclusive.py` (new): torch-free classifier for refusal
  reasons; the two legitimate E0 families (`state_sync_nonbijective`,
  `cuda_invalid_configuration`) get their own strata.
- `src/audit/crossfit.py` (new): task-level k-fold cross-fitting of the
  FaultToStressMap (content-addressed fold assignment; probes of one task
  never straddle folds; pooled closure within first k planned cases).
- `scripts/run_e1_probe_study.py` (new): E1 driver, four phases
  (generate/baseline/equiv/map) with the E0 patterns built in: per-kernel
  original-control gate, trial-level evidence, classified INCONCLUSIVE,
  serial + checkpointed + manifested.
- `scripts/run_e1_cse_falsify.py` (new): stronger counterexample search
  over LIKELY_EQUIVALENT probes (all 21 policies; dimension cases TODO).
- `scripts/export_e1_blind_equiv_queue.py` (new): blinded equivalence-audit
  queue with sealed mapping.
- `scripts/extract_contracts.py` (new): automatic schema-v1 contract
  extraction from KernelBench references; human-judgment clauses flagged;
  batch content-hash freeze.
- `scripts/reconcile_corpora.py` (new): C2–C5 stable-ID collection frames
  (sha256 content addresses, within/cross-corpus duplicate flags, task
  cluster keys, frozen digests).
- `scripts/run_e3_external.py` (new): C6 gpuemu corpus loader + dry-run
  (26 ops = 16 controls + 10 seeded bugs verified); B7/B8/B9 port entry
  points fail loudly with clause-by-clause alignment checklists.
- `scripts/b11_compute_sanitizer.py` (new): B11 wrapper over
  memcheck/racecheck/synccheck/initcheck with structured alarm records.
- `scripts/build_mdir_strategy_row.py` (new): derives per-subject M-dir
  plans from a frozen map via `src.validator.site_directed` (closes the
  "planner integration" gap up to the strategy-matrix row registration).
- `tests/test_e1_e2_readiness.py` (new): 15 CPU tests over all of the above
  pure logic.

### Validation

- Remote A800 (CPU-only, `CUDA_VISIBLE_DEVICES=`), while E0 run5 kept the
  GPU: full available suite `131 passed`; new tests included.
- E1 `--phase generate` executed on the remote: 1,646 probes regenerated,
  by category A=757 / B=702 / C=178 / D=9 — exact match with the Table 10
  generation-time constants; 90/90 kernels reproduce the historical probe
  ids exactly; 9 probes machine-proven (`dead_host_constant`).
- E2 first artifacts produced: C2/C4/C5 frames locally (241/184/1,724 rows;
  C5 accepted=529 matches the published greedy-baseline count), C3 frame on
  the remote (28,227 rows, 16,459 accepted), contract extraction running on
  the remote KernelBench L1+L2.

### Result validity

No historical number changes. GPU phases (E1 baseline/equiv, C6 runner,
B11) are queued behind E0 run5 completion; see the post-run5 checklist in
`MutakernelV2/实验/重跑实验数据/E1_探针研究_A800/README.md`.

## Entry template

Future entries must include:

1. files changed;
2. behavioral change;
3. rationale;
4. reviewer concern or research-validity threat addressed;
5. tests and experiment manifests used for validation;
6. whether old result numbers remain valid or require rerunning.
