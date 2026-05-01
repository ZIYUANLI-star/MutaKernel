#!/bin/bash
# Phase 2+3: LLM Iterative Analysis + Clustering (fixed version)
set -e

export VENV_PYTHON="/home/kbuser/projects/KernelBench-0/.venv/bin/python"
export PROJECT="/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel"
export DEEPSEEK_API_KEY="sk-b896056753ec440cb735873f0179bb67"
export LLM_MODEL="deepseek-reasoner"
export LLM_API_BASE="https://api.deepseek.com/v1"
export LLM_API_KEY="$DEEPSEEK_API_KEY"
export LLM_MAX_MUTANTS=0
export LLM_MAX_ROUNDS=3
export LLM_MAX_TOKENS=16384
export PYTHONPATH="$PROJECT:$PYTHONPATH"

cd "$PROJECT"

echo "============================================================"
echo "  PHASE 2+3: LLM Analysis (FIXED: input_spec + prompt)"
echo "  $(date)"
echo "============================================================"
echo "  Model: $LLM_MODEL | Rounds: $LLM_MAX_ROUNDS"
echo ""

$VENV_PYTHON scripts/_pilot_llm20.py

echo ""
echo "============================================================"
echo "  Phase 2+3 Complete! $(date)"
echo "============================================================"
echo "  Results: $PROJECT/llm_analysis_results/"
