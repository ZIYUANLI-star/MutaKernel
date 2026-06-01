"""Scan all Phase II detail JSONs and extract buggy kernels per Task B's
strict definition:

  Kernel is "buggy" iff in any Phase II enhanced-test dimension there exists
  a (policy, seed) such that:
      ref_ok == True  AND  original_ok == False

We aggregate per kernel_name (since the same original kernel is tested under
many mutants, the same failing case may appear repeatedly).
"""
import json, glob, os
from collections import defaultdict

ROOT = '/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/stress_enhance_results/details/'
os.chdir(ROOT)
files = sorted(glob.glob('*.json'))

# {kernel_name: {(policy, seed, mode): [(mutant_id, error, ref_nan_fallback), ...]}}
buggy = defaultdict(lambda: defaultdict(list))

# Diagnostic counters
n_files_scanned = 0
n_total_pr_events = 0
n_pr_eval_value = 0
n_pr_eval_repeat = 0
n_pr_train = 0
n_train_skipped_no_refok = 0


def collect_eval_pr(pr_list, kernel_name, dim_tag, mutant_id):
    """Collect (ref_ok=True, original_ok=False) events from a policy_results array."""
    global n_total_pr_events
    out = 0
    for pr in pr_list:
        n_total_pr_events += 1
        ref_ok = pr.get('ref_ok', None)
        orig_ok = pr.get('original_ok', None)
        if ref_ok is True and orig_ok is False:
            key = (pr.get('policy'), pr.get('seed'), dim_tag)
            buggy[kernel_name][key].append({
                'mutant_id': mutant_id,
                'error': pr.get('error', '')[:120],
                'ref_nan_fallback': pr.get('ref_nan_fallback', False),
                'time_ms': pr.get('time_ms'),
            })
            out += 1
    return out


def collect_train_pr(results_list, kernel_name, mutant_id):
    """training_stress.results lacks ref_ok; we conservatively skip the
    ones that match the buggy criterion only by orig_ok=False, because we
    cannot verify ref_ok."""
    global n_train_skipped_no_refok
    for pr in results_list:
        orig_ok = pr.get('original_ok')
        if orig_ok is False:
            n_train_skipped_no_refok += 1
            # we record this as a "candidate" (ref_ok unknown)
            key = (pr.get('policy'), pr.get('seed'), 'train_candidate_no_refok')
            buggy[kernel_name][key].append({
                'mutant_id': mutant_id,
                'note': 'training_stress: ref_ok not recorded — must rerun to confirm',
            })


for f in files:
    n_files_scanned += 1
    with open(f) as fp:
        d = json.load(fp)
    kname = d.get('kernel_name')
    mid = d.get('mutant_id')
    if not kname:
        continue

    mt = d.get('main_track', {})

    # eval mode: value_stress
    vs = mt.get('value_stress', {})
    if isinstance(vs.get('policy_results'), list):
        added = collect_eval_pr(vs['policy_results'], kname, 'eval_value', mid)
        n_pr_eval_value += added

    # eval mode: repeated_run
    rr = mt.get('repeated_run', {})
    if isinstance(rr.get('policy_results'), list):
        added = collect_eval_pr(rr['policy_results'], kname, 'eval_repeat', mid)
        n_pr_eval_repeat += added

    # train mode: main_track.training_stress  (ref_ok missing → candidate only)
    ts = mt.get('training_stress', {})
    if isinstance(ts.get('results'), list):
        before = n_train_skipped_no_refok
        collect_train_pr(ts['results'], kname, mid)
        n_pr_train += n_train_skipped_no_refok - before


# === Report ===
print(f'Mutant detail files scanned : {n_files_scanned}')
print(f'Total policy_result events  : {n_total_pr_events}')
print(f'Buggy events (eval_value)   : {n_pr_eval_value}  (ref_ok=True ∧ orig_ok=False)')
print(f'Buggy events (eval_repeat)  : {n_pr_eval_repeat} (ref_ok=True ∧ orig_ok=False)')
print(f'Train candidates (no refok) : {n_pr_train}      (orig_ok=False, ref_ok unknown)')
print()
print(f'Unique buggy kernels        : {len(buggy)}')
print()

# === Per-kernel summary ===
# Group: count failing (policy, seed) pairs per kernel, separating eval vs train_candidate
print('--- Per-kernel summary (sorted by #unique eval failing cases) ---')
print(f'{"kernel":<10} | {"eval_cases":<10} | {"train_cand":<10} | top failing eval policies')
print('-' * 100)

ranked = []
for kname, cases in buggy.items():
    eval_cases = {k: v for k, v in cases.items() if not k[2].startswith('train')}
    train_cases = {k: v for k, v in cases.items() if k[2].startswith('train')}
    # collapse seeds to count unique (policy) buckets
    eval_policies = set(k[0] for k in eval_cases.keys())
    ranked.append((kname, len(eval_cases), len(train_cases),
                   eval_policies, eval_cases))

ranked.sort(key=lambda x: -x[1])
for r in ranked:
    kname, n_eval, n_train, ep, _ = r
    top = ', '.join(sorted(ep)[:5])
    print(f'{kname:<10} | {n_eval:>10} | {n_train:>10} | {top}')

# Save full buggy mapping to JSON for downstream Task B
OUT = '/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_b_buggy_kernels_from_existing_data.json'
os.makedirs(os.path.dirname(OUT), exist_ok=True)
serializable = {}
for kname, cases in buggy.items():
    serializable[kname] = []
    for (policy, seed, mode), evidence in cases.items():
        serializable[kname].append({
            'policy': policy, 'seed': seed, 'mode': mode,
            'evidence_count': len(evidence),
            'evidence_samples': evidence[:3],
        })
with open(OUT, 'w', encoding='utf-8') as fp:
    json.dump({
        'summary': {
            'kernels_buggy_total': len(buggy),
            'eval_buggy_events_total': n_pr_eval_value + n_pr_eval_repeat,
            'train_candidate_events_total': n_pr_train,
        },
        'buggy_kernels': serializable,
    }, fp, ensure_ascii=False, indent=2)

print()
print(f'>> saved to {OUT}')
