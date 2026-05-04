#!/bin/bash
# =============================================================================
#  Full Pipeline: Phase 1 (4-layer stress test) + Phase 2+3 (LLM analysis)
#  ALL 322 survived mutants — incremental (skips already-completed)
# =============================================================================
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
export TORCH_CUDA_ARCH_LIST="8.9"

cd "$PROJECT"

echo ""
echo "============================================================"
echo "  FULL PIPELINE — ALL survived mutants"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# ------------------------------------------------------------------
#  Phase 1: 4-Layer Stress Enhancement (ALL mutants, --max-mutants 0)
# ------------------------------------------------------------------
echo "============================================================"
echo "  PHASE 1: 4-Layer Stress Enhancement (ALL mutants)"
echo "  $(date '+%H:%M:%S')"
echo "============================================================"
$VENV_PYTHON scripts/run_stress_enhance.py --max-mutants 0
PHASE1_EXIT=$?

echo ""
echo "============================================================"
echo "  PHASE 1 Complete (exit=$PHASE1_EXIT) @ $(date '+%H:%M:%S')"
echo "============================================================"

# ------------------------------------------------------------------
#  Phase 2+3: LLM Iterative Analysis + Post-hoc Clustering
# ------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  PHASE 2+3: LLM Iterative Analysis + Clustering"
echo "  Model: $LLM_MODEL | Rounds: $LLM_MAX_ROUNDS"
echo "  $(date '+%H:%M:%S')"
echo "============================================================"
$VENV_PYTHON scripts/_pilot_llm20.py
PHASE23_EXIT=$?

echo ""
echo "============================================================"
echo "  FULL PIPELINE COMPLETE"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Phase 1 exit: $PHASE1_EXIT"
echo "  Phase 2+3 exit: $PHASE23_EXIT"
echo "============================================================"
echo "  Phase 1 results:   $PROJECT/stress_enhance_results/"
echo "  Phase 2+3 results: $PROJECT/llm_analysis_results/"
echo "============================================================"
