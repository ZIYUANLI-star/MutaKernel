#!/bin/bash
# Full pipeline: Phase 1 (stress test) + Phase 2+3 (LLM analysis + clustering)
# Limited to first 50 survived mutants
set -e

export VENV_PYTHON="/home/kbuser/projects/KernelBench-0/.venv/bin/python"
export PROJECT="/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-sk-b896056753ec440cb735873f0179bb67}"
export LLM_MODEL="${LLM_MODEL:-deepseek-reasoner}"
export LLM_API_BASE="${LLM_API_BASE:-https://api.deepseek.com/v1}"
export LLM_API_KEY="$DEEPSEEK_API_KEY"
export LLM_MAX_MUTANTS="${LLM_MAX_MUTANTS:-0}"
export LLM_MAX_ROUNDS="${LLM_MAX_ROUNDS:-3}"
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-16384}"
export PYTHONPATH="$PROJECT:$PYTHONPATH"

cd "$PROJECT"

echo "============================================================"
echo "  PHASE 1: 4-Layer Stress Enhancement (50 mutants)"
echo "============================================================"
$VENV_PYTHON scripts/run_stress_enhance.py --max-mutants 50

echo ""
echo "============================================================"
echo "  PHASE 2+3: LLM Iterative Analysis + Clustering"
echo "============================================================"
echo "  Model: $LLM_MODEL | Rounds: $LLM_MAX_ROUNDS"
echo ""
$VENV_PYTHON scripts/_pilot_llm20.py

echo ""
echo "============================================================"
echo "  Full Pipeline Complete!"
echo "============================================================"
echo "  Phase 1 results: $PROJECT/stress_enhance_results/"
echo "  Phase 2+3 results: $PROJECT/llm_analysis_results/"
