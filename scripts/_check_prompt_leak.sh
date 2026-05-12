#!/usr/bin/env bash
# Verify Task A prompts do NOT leak DeepSeek's 3-round iterative analysis.
PROMPT_DIR=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/prompts

echo "============================================"
echo "  Check Task A prompts for DeepSeek leakage"
echo "============================================"
n_prompts=$(ls "$PROMPT_DIR"/*.txt 2>/dev/null | wc -l)
echo "Total prompt files: $n_prompts"
echo

echo "--- Pattern search across ALL Task A prompt files ---"
for pat in 'DeepSeek' 'deep-seek' 'reasoner' 'iterative.analysis' 'previous.LLM' 'previous.attempt' 'prior.attempt' 'llm_iter' 'previous.round' '3.round' '3-round'; do
  n=$(grep -liE "$pat" "$PROMPT_DIR"/*.txt 2>/dev/null | wc -l)
  echo "  Pattern '$pat'  -> $n files contain it"
done
echo

PROMPT=$(ls "$PROMPT_DIR"/*_r1.txt | head -1)
echo "--- Section headers of a representative round-1 prompt ---"
echo "($PROMPT)"
grep -nE '^##' "$PROMPT"
echo

echo "--- Total prompt size ---"
wc -lc "$PROMPT"
