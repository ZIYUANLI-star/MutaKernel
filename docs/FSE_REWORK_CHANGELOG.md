# FSE rework change log

This document records every code and experiment change made for the FSE
revision.  Each entry states the affected files, the reason for the change,
the review concern it addresses, and the validation performed.  Historical
results are not silently overwritten; rerun results must carry a new manifest
and provenance record.

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

## Entry template

Future entries must include:

1. files changed;
2. behavioral change;
3. rationale;
4. reviewer concern or research-validity threat addressed;
5. tests and experiment manifests used for validation;
6. whether old result numbers remain valid or require rerunning.
