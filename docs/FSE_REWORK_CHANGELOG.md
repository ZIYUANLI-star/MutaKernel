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

## Entry template

Future entries must include:

1. files changed;
2. behavioral change;
3. rationale;
4. reviewer concern or research-validity threat addressed;
5. tests and experiment manifests used for validation;
6. whether old result numbers remain valid or require rerunning.
