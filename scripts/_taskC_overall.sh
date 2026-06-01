#!/bin/bash
TASK_C_DIR=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_c_phase1_direct

echo "=== Task C 全任务总览 ==="
echo

echo "--- 目标范围 ---"
TARGET_TOTAL=534
echo "  目标 mutant 总数:    $TARGET_TOTAL (Phase I survived 270 + candidate_eq 264)"

echo
echo "--- details 文件数（全 Task C 累计产出）---"
DETAIL_TOTAL=$(ls "$TASK_C_DIR/details/" 2>/dev/null | wc -l)
echo "  已产出 detail JSON: $DETAIL_TOTAL / $TARGET_TOTAL"

echo
echo "--- completed.json 统计 ---"
COMPLETED=$(python3 -c "import json; d=json.load(open('$TASK_C_DIR/completed.json')); print(len(d) if isinstance(d, list) else len(d.get('completed', [])))" 2>/dev/null)
echo "  completed.json 收录: $COMPLETED"

echo
echo "--- 全 Task C 的 killed 统计（从所有 detail JSON 聚合）---"
python3 << 'PYEOF'
import json, os, glob
TASK_C_DIR = "/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_c_phase1_direct"
detail_dir = os.path.join(TASK_C_DIR, "details")
files = sorted(glob.glob(os.path.join(detail_dir, "*.json")))

killed = 0
killed_in_round = {1:0, 2:0, 3:0, 4:0, 5:0}
unkillable_judged = 0
no_kill = 0
rounds_executed_total = 0
total_in_tok = 0
total_out_tok = 0
by_phase1 = {"survived": [0,0], "candidate_equivalent": [0,0]}  # [total, killed]

for f in files:
    with open(f) as fp:
        d = json.load(fp)
    is_killed = bool(d.get("killed"))
    kround = d.get("killing_round", 0)
    p1 = d.get("phase1_status", "unknown")

    if p1 in by_phase1:
        by_phase1[p1][0] += 1
        if is_killed:
            by_phase1[p1][1] += 1

    if is_killed:
        killed += 1
        if kround in killed_in_round:
            killed_in_round[kround] += 1
    else:
        # check if LLM said unkillable
        rounds = d.get("rounds", [])
        if rounds and any(r.get("killable") is False for r in rounds):
            unkillable_judged += 1
        else:
            no_kill += 1
        rounds_executed_total += len(rounds)
    # token tally
    for r in d.get("rounds", []):
        u = r.get("usage", {}) or r.get("tokens", {})
        total_in_tok  += u.get("prompt_tokens", 0) or u.get("input", 0)
        total_out_tok += u.get("completion_tokens", 0) or u.get("output", 0)

total = len(files)
print(f"  total details        : {total}")
print(f"  killed               : {killed} ({killed/total*100:.1f}%)" if total else "")
print(f"  unkillable (LLM 判定) : {unkillable_judged}")
print(f"  not killed (其它)     : {no_kill}")
print()
print("  killed by round:")
for r, c in killed_in_round.items():
    if c > 0:
        print(f"    round {r}: {c}")
print()
print("  按 Phase I 状态分桶:")
for k, (tot, kl) in by_phase1.items():
    if tot > 0:
        print(f"    {k:25s}: {kl}/{tot} ({kl/tot*100:.1f}%)")
print()
print(f"  累计 tokens: input={total_in_tok:,} | output={total_out_tok:,} | total={total_in_tok+total_out_tok:,}")
PYEOF

echo
echo "--- 当前补跑（仅 232 个被 throttle 污染的 mutant）---"
DONE_NOW=$(grep -cE 'running kill rate=' /home/kbuser/mutakernel_logs/resume_taskC.log 2>/dev/null)
echo "  本次补跑进度: $DONE_NOW / 232"
grep -E 'running kill rate=' /home/kbuser/mutakernel_logs/resume_taskC.log 2>/dev/null | tail -1

echo
echo "--- 全任务剩余估算 ---"
REMAIN=$((232 - DONE_NOW))
echo "  本次补跑剩余: $REMAIN 个"
echo "  按 ~90s/mutant 估算: $((REMAIN * 90 / 60)) 分钟"

echo
echo "--- 首次 Task C 跑 vs 当前补跑 ---"
# 首次跑的 details 已经写完，补跑会覆盖；我们看下 _archive 是否还有先前版本
ARCHIVE_DIR=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/_archive
if [ -d "$ARCHIVE_DIR" ]; then
    echo "  归档目录存在: $ARCHIVE_DIR"
    ls "$ARCHIVE_DIR" 2>/dev/null | head -10
fi
