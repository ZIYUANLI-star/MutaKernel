# FSE experiment runbook

## Status

This is the operational protocol for the corrected FSE study. It does not
authorize a primary GPU run yet. Two release blockers remain:

1. rotate the credential exposed in the frozen Git history and provision a
   pinned-host-key, key-only, non-root remote account/container; and
2. place candidate execution in a no-network/no-credential sandbox separated
   from the trusted reference, oracle, and result writer.

The current per-case runner is suitable for CPU regression tests and trusted
GPU qualification only. Data produced before both blockers are closed must be
labelled `engineering_smoke`, never a paper result.

## Immutable versions

- Pre-rework freeze: tag `pre-fse-rework-20260719`, commit
  `9581c46107df274366550171d9eabd1f871b0b55`.
- Corrected work: branch `codex/fse-rework`.
- Strategy schema: `configs/fse_strategy_matrix.json`.
- External-baseline freeze: `configs/external_baselines.json`.

Record the final corrected commit after the branch is clean. Never run a paper
experiment with `--allow-dirty`.

## Gate 0: security and isolation

Before copying sources or executing a kernel:

- revoke/rotate the password previously committed to Git;
- use SSH keys and verify the host fingerprint out of band;
- use a non-root account inside an isolated NVIDIA container;
- mount candidate/reference/dataset sources read-only;
- provide a fresh writable run/cache directory only;
- disable network access and do not mount SSH, Git, cloud, or LLM credentials;
- enforce process, memory, disk, and wall limits;
- keep candidate code outside the trusted oracle/result-writer process;
- authenticate the controller/worker request and result digests; and
- never use shared-cloud GPU reset as recovery.

Freeze the container image by digest, not a mutable tag. Add its digest and
sandbox policy hash to the environment and run manifests before the primary
run.

No executable sandbox/trusted-controller backend or automatic container-policy
capture exists in the repository yet. Gate 0 is therefore a code blocker, not a
configuration step that can be completed by invoking the current runner.

## Gate 1: local CPU qualification

From a clean checkout and the intended Python environment:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider tests
python3 -m compileall -q config.py scripts src tests
git diff --check
git status --short
```

Expected result: every test passes, compilation succeeds, `git diff --check`
is empty, and the worktree is clean. A secret scan must report file names only;
never print suspected values into CI logs.

## Gate 2: GPU qualification

Use a disposable, non-paper output directory. Run at least these fixtures on
the exact target image and every reported GPU architecture:

- correct candidate;
- wrong value, shape, dtype, NaN/Inf position, input side effect, buffer side
  effect, and output-alias candidates;
- constructor RNG/state synchronization;
- non-contiguous/gapped/overlapping storage and cross-argument aliases;
- candidate compile error, Python exception, `SystemExit`, CUDA illegal access,
  timeout, OOM, and reference failure;
- a candidate whose first call is correct and second stateful call is wrong;
- backward candidates that fool an all-ones VJP but fail a random VJP; and
- evidence-bundle replay in a fresh container.

Also complete parity fixtures for every native external adapter before that row
can appear in a table. The configuration currently marks unfinished adapters as
`adapter_required_before_primary_run`.

## Gate 3: freeze subjects and contracts

Freeze two different study objects:

1. a complete natural-population/applicability manifest for RQ1, which retains
   unsupported, missing, partial, and inconclusive subjects; and
2. a preregistered rich-contract common subset for the 32-call paired RQ2
   comparison.

The current strict planner implements only item 2: it rejects the entire plan
when any subject cannot execute a matrix case. Do not pass the nominal full 597
frame to that matrix or silently drop incompatible subjects. A separate
applicability-aware population planner is required before RQ1 execution.

Prepare `fse_runs/frozen/paired_subject_spec.jsonl`. Each row must identify one
candidate. The following is a non-executable field checklist; `{}` is not a
valid contract:

```json
{
  "subject_id": "stable-public-id",
  "dataset": "dataset-and-version",
  "task_id": "dataset-local-task-id",
  "language": "cuda-or-triton",
  "candidate_path": "relative/path/to/candidate.py",
  "reference_path": "relative/path/to/reference.py",
  "contract": {"replace_with": "a complete canonical schema-v1 contract"},
  "source": {},
  "metadata": {}
}
```

Paths are resolved under `--root` and content-addressed. Contract schema v1
supports top-level positional tensor arguments and the default stream only.
Nested/keyword/scalar ABI constraints, concurrent streams, and unsupported
task-specific adapters must be excluded and reported, not approximated.

Build and plan the paired rich-contract subset exactly once:

```bash
python3 scripts/build_fse_subject_manifest.py \
  --input fse_runs/frozen/paired_subject_spec.jsonl \
  --root . \
  --output fse_runs/frozen/paired_subjects.json

python3 scripts/plan_fse_experiment.py \
  --subjects fse_runs/frozen/paired_subjects.json \
  --strategy-matrix configs/fse_strategy_matrix.json \
  --output fse_runs/frozen/paired_plan.json
```

Archive the subject specification, subject manifest, plan, their SHA-256 values,
inclusion/exclusion table,
dataset licenses/versions, and the contract-review sign-off. Do not regenerate
them after seeing detector outcomes.

## Gate 4: capture the execution environment

After checking out the clean final commit inside the frozen container:

```bash
python3 scripts/capture_fse_environment.py \
  --repo-root . \
  --output fse_runs/frozen/environment.json
```

Inspect the manifest for Git commit/dirty state, Python, Torch/Triton, CUDA,
driver, NVCC, GPU model, and all allowlisted behavior-affecting environment
values. It must contain no secret value.

## Gate 5: smoke, pilot, then primary execution

Smoke one subject and strategy into a disposable directory:

```bash
python3 scripts/run_fse_experiment.py \
  --plan fse_runs/frozen/paired_plan.json \
  --subjects fse_runs/frozen/paired_subjects.json \
  --environment fse_runs/frozen/environment.json \
  --artifact-root . \
  --output-dir fse_runs/smoke \
  --run-id fse-smoke-001 \
  --device cuda \
  --timeout 180 \
  --subject SUBJECT_ID \
  --strategy five-iid-historical-anchor
```

Inspect all non-pass replay bundles and verify counts before a preregistered
pilot. Pilot subjects must not enter the primary estimand unless the protocol
predeclares this and preserves their blind status.

The paired run uses no filters, no `--max-tests`, no `--early-stop-on-fail`, no
`--allow-dirty`, and no wall budget. Only the strategies marked
`budget_matched` at 32 calls form the compute-matched primary comparison; the
five-call historical anchor is descriptive and must not be included as an
equal-compute row:

```bash
python3 scripts/run_fse_experiment.py \
  --plan fse_runs/frozen/paired_plan.json \
  --subjects fse_runs/frozen/paired_subjects.json \
  --environment fse_runs/frozen/environment.json \
  --artifact-root . \
  --output-dir fse_runs/paired-primary \
  --run-id fse-paired-primary-v1 \
  --device cuda \
  --timeout 180
```

Resume by repeating the identical command. The existing immutable run manifest
must match exactly. Never delete individual observations to force a rerun; use
a new run ID/output directory after a code, environment, source, or protocol
change.

Run equal-wall-time comparisons separately with a preregistered
`--wall-budget-ms`. Budget-exhausted cases remain `INCONCLUSIVE`. Keep cold
per-case cost separate from a later compile-once/warm-session ADRS overhead
study; the current process-isolated runner cannot establish steady-state loop
latency.

## Gate 6: completeness and evidence checks

Before unblinding or statistics:

- verify every planned test ID has exactly one observation;
- verify planned and observed candidate invocation totals;
- retain all `INCONCLUSIVE` categories and denominators;
- replay every headline alarm and a random evidence-bundle sample in a fresh
  container;
- report never-run, partial, reference-failing, unsupported, and missing
  subjects separately; and
- reject any observation whose source, contract, plan, environment, runner, or
  log digest does not match its provenance chain.

The statistical script independently enforces full planned coverage, validates
actual invocation-count bounds (and complete calls for passing cases), and
rejects early-stopped/censored runs.

## Gate 7: blinded human audit

Build this paired-cohort queue only after its observation log is sealed:

```bash
python3 scripts/build_human_audit_queue.py \
  --subjects fse_runs/frozen/paired_subjects.json \
  --plan fse_runs/frozen/paired_plan.json \
  --run-manifest fse_runs/paired-primary/run_manifest.json \
  --observations fse_runs/paired-primary/observations.jsonl \
  --queue fse_runs/audit/public_queue.jsonl \
  --sealed-mapping fse_runs/audit/sealed_mapping.json \
  --selection-seed 20260719 \
  --nonalarm-per-stratum N
```

Replace `N` with the preregistered integer, or omit the option to include every
fully evaluated non-alarm subject. This command binds queue items to the exact
run manifest and includes execution-unresolved subjects, but formal annotation
is still blocked until a policy-neutral evidence exporter exists. RQ1 requires
an analogous queue from the future complete population/applicability plan; the
paired queue alone cannot estimate whole-population false acceptance.

Primary auditors receive only the public queue/evidence allowed by the
codebook. Keep the sealed mapping inaccessible until both primary annotations
are locked. Use two fixed independent GPU-capable primary auditors and a third
independent adjudicator. Every alarm annotation records fault class, violated
contract clause, and whether evidence replay succeeded.

Compile complete annotations (complete is the default and fail-closed mode):

```bash
python3 scripts/compile_human_audit.py \
  --queue fse_runs/audit/public_queue.jsonl \
  --sealed-mapping fse_runs/audit/sealed_mapping.json \
  --annotations fse_runs/audit/annotations.jsonl \
  --report fse_runs/audit/agreement.json \
  --analysis-labels fse_runs/audit/analysis_labels.jsonl
```

`--allow-partial` is development-only and its outputs must not feed paper
tables.

## Gate 8: summaries and statistics

```bash
python3 scripts/summarize_fse_results.py \
  --run-manifest fse_runs/paired-primary/run_manifest.json \
  --observations fse_runs/paired-primary/observations.jsonl \
  --labels fse_runs/audit/analysis_labels.jsonl \
  --output fse_runs/analysis/summary.json

python3 scripts/analyze_fse_statistics.py \
  --run-manifest fse_runs/paired-primary/run_manifest.json \
  --plan fse_runs/frozen/paired_plan.json \
  --observations fse_runs/paired-primary/observations.jsonl \
  --human-audit fse_runs/audit/analysis_labels.jsonl \
  --audit-report fse_runs/audit/agreement.json \
  --output fse_runs/analysis/statistics.json \
  --bootstrap-replicates 10000 \
  --seed 20260719
```

Primary reporting uses task-clustered intervals, paired exact McNemar tests with
Holm correction, audited alarm precision, sensitivity to inconclusive cases,
and detection/cost Pareto results. `NO_DEFECT_FOUND` is not a proof of
correctness. Natural false acceptance is reported as a confirmed lower bound
unless the sampling design supports a weighted prevalence estimate.

`statistics.json` is the authoritative inferential output. `summary.json` is
descriptive convenience only. Neither paired-cohort file substitutes for the
future population/applicability analysis required by RQ1.

## Required experiment families

The paper is not ready until the evidence package contains:

1. corrected natural-kernel rerun and independent audit;
2. equal-call and equal-time strong-baseline comparisons;
3. native KernelBench plus robust-kbench and KernelBenchX native/protocol-port
   rows where applicable, with parity tests;
4. corrected full 1,646-mutant rerun, newly derived Phase-II cohort, blinded EMD
   calibration, and sensitivity to model/prompt/round/threshold choices;
5. predictive validity between mutation/operator scores and confirmed natural
   fault classes;
6. cold and compile-once warm ADRS overhead, peak GPU memory, GPU-hours, and
   monetary cost;
7. stable-ID reconciliation of the competing `831/64` and `834/67` historical
   corpus funnels, followed by missingness analysis of the common 767 completed
   and nominal 597 baseline-accepted frame;
8. TaskD 104-versus-101 stable-ID reconciliation and structural no-fallback
   repair classification;
9. the 54-subject/2,664-mutant/248 historically stress-rescued KGB supplement,
   if retained, rerun and audited under the same protocol; and
10. a disclosed L3/multi-kernel/stateful or second-architecture generalization
    study, or an explicit scope limitation if unavailable.

ProofWright remains a related-work/property comparison until an official
runnable artifact is frozen; unsupported formal-verification cases must not be
scored as passes or failures.

## Stop conditions

Stop the run and create a new versioned protocol if any source/contract/plan
hash changes, a sandbox boundary fails, a reference is discovered incorrect,
an oracle defect is found, a candidate can alter trusted artifacts, or a result
cannot be reproduced from its bundle. Do not patch paper-facing JSON in place.
