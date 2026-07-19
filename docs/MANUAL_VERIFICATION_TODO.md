# Manual verification work that remains

## Current status

No new human correctness label has been assigned during the code remediation.
Historical labels, LLM judgements, and MutaKernel alarms are preserved only as
pilot metadata. They are not silently promoted to FSE ground truth.

Population-A automation can bind an audit queue to the exact run/plan, separate
detector metadata, validate two independent labels plus adjudication, compute
agreement, and join final labels to canonical observations. It is not yet ready
for annotation: replay configs expose policy construction metadata, and a
policy-neutral bundle with independently replayable materialized
counterexamples has not been implemented. Mutant/EMD, taxonomy, and repair
populations still require dedicated queue/compiler schemas. The actual expert
decisions must be made by people.

## Before annotation starts

- [ ] Freeze the real-kernel subject manifest and every task correctness
  contract before revealing new detector outcomes.
- [ ] Freeze the candidate/reference source hashes, separate population and
  paired-comparison plans, run manifest, observation-log digest, environment
  manifest, strategy matrix, audit codebook version, selection seed, and
  per-item review budget.
- [ ] Resolve contract questions with task authors/specifications, especially
  valid dtype, shape, value-domain, layout, backward, determinism, and stream
  clauses. Reference execution success alone does not make an input valid.
- [ ] Ensure every alarm item has an immutable operator bundle containing source
  hashes, deterministic input construction and hash, structured diff,
  compiler/runtime log, oracle settings, and a replay command qualified in a
  fresh container.
- [ ] Export a separate blinded auditor bundle with materialized inputs or a
  policy-neutral replay program. It must remove detector/strategy/policy/seed
  identities while preserving a verified counterexample. Never give the current
  operator `case_config.json` directly to primary auditors.
- [ ] Select two CUDA/GPU-capable primary auditors and a third adjudicator.
  Record conflicts of interest and keep their pseudonymous IDs stable.

## Population A: natural generated kernels

Preferred scope: independently review all 597 historically baseline-accepted
public candidates after they have been rerun by the corrected harness.

If full review is infeasible, the minimum defensible queue is:

1. every alarm in the union of **all** compared validators/strategies, not only
   MutaKernel's historical 222;
2. every subject that raised at least one such alarm; and
3. a preregistered stratified random sample of subjects that completed every
   planned test with `PASS` and no alarm, stratified by dataset/generator,
   language, task level, and other frozen factors; and
4. a separately reported/audited stratum for never-run, partial, and
   inconclusive subjects. These are not non-alarm negatives.

Required human outputs:

- one subject-level label from each primary auditor;
- one alarm-level label for each concrete `(strategy, test_id,
  counterexample)` from each primary auditor;
- contract clause, fault class, confidence, rationale, replay status, and
  investigation minutes (timestamps/minutes are governance fields retained in
  raw records; the current compiler validates the correctness fields only);
- adjudication only after both primary labels are locked; and
- a second-GPU replay for every headline example before it enters the paper.

`NO_DEFECT_FOUND` means only that the allotted review did not confirm a defect.
It is not a true negative or proof of correctness. Population prevalence can be
reported only with the frozen sampling weights and this limitation.

## Population B: mutant/EMD calibration

The historical 534 Phase-II cases must not automatically define the new frame,
because corrected Phase I may produce a different survivor/unresolved cohort.
After all 1,646 source mutants are rerun, derive a new cohort mechanically from
the frozen protocol.

Preferred scope: double-label the full new cohort. A resource-constrained gold
set must cover task level, mutation operator, dynamic outcome, every EMD layer,
and every EMD status. Auditors must not see stress outcome, EMD tier, LLM
answer, or expected paper claim.

Required labels are `NON_EQUIVALENT`, `EQUIVALENT_MACHINE_PROVEN`,
`LIKELY_EQUIVALENT`, or `INCONCLUSIVE`. Passing many tests and LLM agreement
cannot establish machine-proven equivalence.

Separately run and report EMD sensitivity across frozen static-rule variants,
dynamic-round/threshold settings, LLM model and prompt versions, and number of
rounds. Compare each layer with the independent labels; never use a layer's own
decision as its truth.

## Population C: taxonomy and representativeness

- [ ] Re-code a frozen sample of the 1,020 historical LLM compilation/validator
  failures with two independent coders.
- [ ] Publish the codebook, open-coding procedure, saturation criterion,
  disagreements, adjudication, and agreement.
- [ ] Double-code confirmed natural defects against the mutation taxonomy,
  allowing `not represented` and `multiple operators`.
- [ ] Record missing classes, including cross-stream ordering, state/RNG side
  effects, multi-kernel coordination, and other faults outside schema v1.

This work is necessary to answer why mutation is useful and whether injected
faults resemble natural LLM errors, rather than merely showing that a large
stress suite kills synthetic mutants.

## Population D: TaskD repair case study

- [ ] Join the 104 historical repair targets with the 101 RQ4 CUDA-Agent
  stress-positive IDs using stable candidate hashes.
- [ ] Name and explain the three unmatched targets.
- [ ] Reclassify each output as real custom-kernel repair, partial repair,
  PyTorch/library fallback, dead/unused custom kernel, validator-specific
  gaming, not fixed, or inconclusive.
- [ ] If rerun, use a structural no-fallback guard, identical model/token/
  candidate budgets, multiple seeds, and an independent final oracle.

Until this is complete, the historical 90 framework-level `FIXED` outcomes and
15 real custom-CUDA repairs are pilot observations, not an FSE effectiveness
rate.

## Calibration workflow

1. Draw 20–30 diverse pilot items without exposing detector identity.
2. Have auditors A and B label independently under a draft codebook.
3. Discuss ambiguities, revise the codebook, then relabel the pilot.
4. Freeze codebook and investigation budget before opening the main queue.
5. Lock A/B labels before adjudicator C sees disagreements.
6. Publish raw pre-adjudication agreement, Cohen's kappa, full confusion matrix,
   per-label agreement, adjudication rate, and unresolved count.
7. Preserve all A/B/C records; never overwrite disagreement history with the
   final label.

## Commands after the data are frozen

The currently executable command below builds the paired rich-contract queue;
it is not a substitute for the future complete population/applicability queue:

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

Replace `N` with the preregistered integer. Omit the option entirely when every
fully evaluated non-alarm subject will be audited. Do not run this command for
formal annotation until the blinded evidence exporter is implemented.

After two primary labels and any adjudications are complete:

```bash
python3 scripts/compile_human_audit.py \
  --queue fse_runs/audit/public_queue.jsonl \
  --sealed-mapping fse_runs/audit/sealed_mapping.json \
  --annotations fse_runs/audit/annotations.jsonl \
  --report fse_runs/audit/agreement.json \
  --analysis-labels fse_runs/audit/analysis_labels.jsonl
```

Run the statistical analysis only on compiled labels:

```bash
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

The sealed mapping must not be given to primary auditors. Store it separately
with an integrity hash and disclose it only after labels are locked.

## Exit criteria

Human calibration is complete only when:

- every primary analysis item has two independent labels; missing-label reasons
  belong in a separate missingness report and `--allow-partial` output cannot
  enter paper statistics;
- every disagreement is adjudicated or retained as `INCONCLUSIVE`;
- all included alarm labels point to a fresh-container-qualified blinded replay
  bundle;
- agreement and confusion statistics are generated before adjudication;
- sampling weights and exclusions are frozen and published; and
- no LLM decision is used as a final label or proof.
