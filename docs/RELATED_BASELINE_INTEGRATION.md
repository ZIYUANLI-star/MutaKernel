# Strong-baseline integration protocol

## Purpose

The EuroSys reviews correctly observed that comparing only against a weak
KernelBench-style sampler cannot establish that MutaKernel contributes more
than "more tests." The FSE study therefore separates:

1. an upstream tool run on its native candidate/task interface; and
2. a protocol port that applies the upstream testing idea to the same frozen
   MutaKernel subjects.

These are different experiments and must appear in different table rows. The
machine-readable freeze is in `configs/external_baselines.json`.

## KernelBench

The maintained FSE matrix calls its five-seed control
`five-iid-historical-anchor`. It is intentionally **not** labelled a native
KernelBench result. A native row must invoke the vendored
`eval_kernel_against_ref` implementation and preserve its seed derivation,
model lifetime, precision-specific tolerance, compilation result, and metadata.
The vendored implementation uses `1e-4` for float32 and `1e-2` for float16 and
bfloat16. The exact vendored source is frozen by the parent MutaKernel commit
and included in the run manifest's implementation hashes.

Before the primary run, add a subprocess adapter and parity fixtures for:

- five deterministic correctness trials from seed 42;
- candidate compilation/import failure;
- a wrong-value candidate;
- FP32 and FP16 tolerance-boundary candidates;
- a stateful reference task; and
- the exact raw `KernelExecResult` serialization.

Until that adapter and the GPU parity fixtures pass, the native KernelBench row
is marked `adapter_required_before_primary_run` and must not appear in a result
table.

## robust-kbench

The upstream repository is frozen at
`078f5bab29934a822268d59a4e707d449abf9b4e`. Its native protocol enumerates
initialization, input, and shared configurations; runs five deterministic
trials per configuration; executes the candidate twice in a forward trial; and
supports backward checking where the task defines it.

Two tracks are required:

- **Native:** run the upstream artifact only on its compatible task-directory
  and standalone `.cu` candidate interface.
- **Protocol port:** on the common rich-contract MutaKernel cohort, implement
  explicit task adapters for the same configuration Cartesian product,
  repeated calls, and backward checks.

The candidate-execution budget counts both candidate calls per forward trial.
Native and ported results must record upstream commit, adapter commit, candidate
call count, reference call count, timeout, wall time, and applicability. The
current MutaKernel repeated and configuration cases are useful components but
are **not yet** a faithful robust-kbench protocol port.

## KernelBenchX

KernelBenchX is frozen at
`fd4192293bf9a8c645327a9d46aa1e807f1f9cf2`. Its native cohort is the 176 tasks
listed in `data/kernelbenchx_v1.json`; the repository directory contains more
task files and must not be globbed as the cohort. It evaluates standard,
outlier, and task-defined boundary inputs with dtype-aware or task-specific
oracles.

Again, use two tracks:

- **Native:** run the official Triton-facing API on the frozen 176-task
  manifest or a disclosed compatible subset.
- **Protocol port:** run standard, sparse-outlier, and boundary policies on the
  same MutaKernel subjects only when their frozen contracts authorize those
  inputs and oracles.

The port must not be called "KernelBenchX" in plots. Report it as
`KernelBenchX-style protocol port`.

## ProofWright

ProofWright is a property-oriented formal-verification comparison, not a
drop-in binary validator baseline. As of the 2026-07-19 freeze, its modified
verification artifact was not publicly runnable. It is therefore included in
related work and the property/coverage comparison, but excluded from executable
rankings. VerCors alone must not be presented as a ProofWright reproduction.

If an official artifact becomes available, freeze its commit and report its
native categories separately: safety verified, semantic equivalence proved,
partial proof, unsupported/unverified, timeout, and verifier instability.
`unverified` is not a defect and `safety verified` is not semantic equivalence.

## Fairness and reporting gates

Every executable comparison must publish three views:

1. unrestricted native protocol;
2. equal candidate-invocation budget on a common eligible cohort; and
3. equal wall/GPU-time budget on that same cohort.

Required columns are subject/task ID, native versus port mode, artifact commit,
adapter commit, contract ID, case ID, candidate and reference calls, alarm,
inconclusive category, cold compile time, warm execution time, parent wall
time, and peak memory where available. Applicability and missingness stay in
the denominator table. No unsupported subject may be silently counted as a
pass, fail, or miss.

This integration is deliberately fail-closed: the current configuration marks
unfinished adapters as such. Full GPU execution must not start merely because
an upstream repository can be cloned.
