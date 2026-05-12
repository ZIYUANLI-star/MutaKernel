#!/usr/bin/env bash
PROMPT_DIR=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/prompts

echo "--- All occurrences of 'previous LLM' (12 files matched) ---"
grep -nE 'previous LLM' "$PROMPT_DIR"/L1_P100__arith_replace__7_r1.txt | head -5
echo
echo "--- All occurrences of '3.round' (1 file matched) ---"
grep -lE '3.round' "$PROMPT_DIR"/*.txt
echo
echo "--- That file's context ---"
file=$(grep -lE '3.round' "$PROMPT_DIR"/*.txt | head -1)
grep -nE '3.round' "$file" -B 1 -A 1
