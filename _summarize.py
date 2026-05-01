import re, sys

data = open('/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/full_block12_output.txt', encoding='utf-8').read()

pattern = r'\[(\S+)\].*?Equiv summary:.*?\n.*?Score \(conservative\):\s+([\d.]+)%\s+\(killed=(\d+),\s*survived=(\d+),\s*stillborn=(\d+),\s*strict_eq=(\d+),\s*cand_eq=(\d+)\)'
matches = re.findall(pattern, data)

eq_pattern = r'\[(\S+)\] Equiv summary: strict_eq=\d+, candidate_eq=\d+, llm_rejected=(\d+), survived=\d+'
eq_matches = dict(re.findall(eq_pattern, data))

total_killed = 0
total_survived = 0
total_stillborn = 0
total_strict = 0
total_cand = 0
total_llm_rejected = 0

header = "{:<12} {:>7} {:>9} {:>10} {:>10} {:>8} {:>8} {:>6} {:>7}".format(
    'Kernel', 'Killed', 'Survived', 'Stillborn', 'Strict_EQ', 'Cand_EQ', 'LLM_Rej', 'Total', 'Score')
print(header)
print('-' * 90)

for m in matches:
    name = m[0]
    score_pct = float(m[1])
    k, s, st, se, ce = int(m[2]), int(m[3]), int(m[4]), int(m[5]), int(m[6])
    lr = int(eq_matches.get(name, 0))
    total = k + s + st + se + ce

    total_killed += k
    total_survived += s
    total_stillborn += st
    total_strict += se
    total_cand += ce
    total_llm_rejected += lr

    print("{:<12} {:>7} {:>9} {:>10} {:>10} {:>8} {:>8} {:>6} {:>6.1f}%".format(
        name, k, s, st, se, ce, lr, total, score_pct))

print('-' * 90)
grand_total = total_killed + total_survived + total_stillborn + total_strict + total_cand
total_equiv = total_strict + total_cand
denom_cons = total_killed + total_survived + total_strict + total_cand
cons_score = total_killed / denom_cons * 100 if denom_cons > 0 else 0
denom_opt = total_killed + total_survived
opt_score = total_killed / denom_opt * 100 if denom_opt > 0 else 0

print("{:<12} {:>7} {:>9} {:>10} {:>10} {:>8} {:>8} {:>6}".format(
    'TOTAL', total_killed, total_survived, total_stillborn, total_strict, total_cand, total_llm_rejected, grand_total))
print()
print("=== Summary ({} kernels) ===".format(len(matches)))
print("  KILLED:              {}".format(total_killed))
print("  SURVIVED:            {}".format(total_survived))
print("  STILLBORN:           {}".format(total_stillborn))
print("  STRICT_EQUIVALENT:   {}".format(total_strict))
print("  CANDIDATE_EQUIVALENT:{}".format(total_cand))
print("  Equiv total:         {}  (strict={} + candidate={})".format(total_equiv, total_strict, total_cand))
print("  LLM rejected->SURV: {}".format(total_llm_rejected))
print("  Grand total:         {}".format(grand_total))
print("  Conservative Score:  {:.2f}%".format(cons_score))
print("  Optimistic Score:    {:.2f}%".format(opt_score))
