@echo off
echo ============================================================
echo  Phase 1: LLM Attribution (all 322 survived mutants)
echo ============================================================
set DEEPSEEK_API_KEY=sk-b896056753ec440cb735873f0179bb67
set LLM_MODEL=deepseek-reasoner
set LLM_API_BASE=https://api.deepseek.com/v1
set LLM_MAX_MUTANTS=0
set NO_PROXY=*

cd /d D:\doctor_learning\Academic_Project\paper_1\MutaKernel
python scripts\_pilot_llm20.py

if %ERRORLEVEL% NEQ 0 (
    echo Phase 1 FAILED
    exit /b 1
)

echo.
echo ============================================================
echo  Phase 2: Verify LLM-Suggested Inputs (WSL + GPU)
echo ============================================================
python scripts\_verify_llm_suggestions.py

echo.
echo ============================================================
echo  Full Experiment Complete!
echo ============================================================
