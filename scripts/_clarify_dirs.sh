#!/usr/bin/env bash
SUPPL=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充

cnt() { ls "$1" 2>/dev/null | wc -l; }
sz()  { du -sh "$1" 2>/dev/null | awk '{print $1}'; }

echo "==========================================="
echo "  $SUPPL  完整结构"
echo "==========================================="
tree -L 2 -d "$SUPPL" 2>/dev/null || ls -la "$SUPPL"
echo

echo "==========================================="
echo "  1. docs/  (你的文档)"
echo "==========================================="
ls -la "$SUPPL/docs/"
echo

echo "==========================================="
echo "  2. task_a_phase2_rerun/   <-- 新的实验数据写到这里"
echo "==========================================="
echo "  details/        $(cnt "$SUPPL/task_a_phase2_rerun/details") 个 mutant   ($(sz "$SUPPL/task_a_phase2_rerun/details"))"
echo "  prompts/        $(cnt "$SUPPL/task_a_phase2_rerun/prompts") 个文件   ($(sz "$SUPPL/task_a_phase2_rerun/prompts"))"
echo "  llm_responses/  $(cnt "$SUPPL/task_a_phase2_rerun/llm_responses") 个文件   ($(sz "$SUPPL/task_a_phase2_rerun/llm_responses"))"
echo
echo "  最新 3 个 details/ 文件（按时间）："
ls -lt "$SUPPL/task_a_phase2_rerun/details/" | head -4
echo

echo "==========================================="
echo "  3. task_c_phase1_direct/   <-- 新的实验数据写到这里"
echo "==========================================="
echo "  details/        $(cnt "$SUPPL/task_c_phase1_direct/details") 个 mutant   ($(sz "$SUPPL/task_c_phase1_direct/details"))"
echo "  prompts/        $(cnt "$SUPPL/task_c_phase1_direct/prompts") 个文件   ($(sz "$SUPPL/task_c_phase1_direct/prompts"))"
echo "  llm_responses/  $(cnt "$SUPPL/task_c_phase1_direct/llm_responses") 个文件   ($(sz "$SUPPL/task_c_phase1_direct/llm_responses"))"
echo
echo "  最新 3 个 details/ 文件（按时间）："
ls -lt "$SUPPL/task_c_phase1_direct/details/" | head -4
echo

echo "==========================================="
echo "  4. _archive/   <-- 旧的 wrong-protocol 数据（已废，但保留作证据）"
echo "==========================================="
ls "$SUPPL/_archive/" 2>/dev/null
echo
for d in "$SUPPL/_archive"/*/; do
  if [[ -d "$d" ]]; then
    echo "  $(basename $d)"
    echo "    details/  $(cnt "$d/details") 个文件"
  fi
done
