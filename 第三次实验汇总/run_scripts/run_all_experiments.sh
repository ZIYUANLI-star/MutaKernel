#!/bin/bash
#
# MutaKernel 第三次实验 - 一站式运行脚本（离线版）
#
# 所有外部数据集已预先准备好，无需网络连接。
#
# 用法:
#   bash run_all_experiments.sh [GPU_ARCH]
#
# 参数:
#   GPU_ARCH  - GPU 计算能力 (默认自动检测)
#               例: 8.9 (RTX 40系), 8.6 (RTX 30系), 8.0 (A100), 9.0 (H100)
#
# 示例:
#   bash run_all_experiments.sh           # 自动检测 GPU，全量运行
#   bash run_all_experiments.sh 8.6       # 手动指定 RTX 3090
#   bash run_all_experiments.sh --smoke   # 仅运行冒烟测试
#

set -euo pipefail

# ============================================================
# 配置
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
RESULT_DIR="$PROJECT_ROOT/第三次实验汇总/results"
LOG_DIR="$PROJECT_ROOT/第三次实验汇总/logs"
mkdir -p "$LOG_DIR" "$RESULT_DIR"
SMOKE_ONLY=false

# 解析参数
GPU_ARCH=""
for arg in "$@"; do
    case "$arg" in
        --smoke) SMOKE_ONLY=true ;;
        *) GPU_ARCH="$arg" ;;
    esac
done

# ============================================================
# 工具函数
# ============================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC}  $(date '+%H:%M:%S') $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $(date '+%H:%M:%S') $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%H:%M:%S') $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $*"; }

# ============================================================
# 1. 环境检查
# ============================================================
echo ""
echo "============================================================"
echo "  MutaKernel 第三次实验 - Full-Scale Differential Testing"
echo "  (离线模式 - 所有数据已预备)"
echo "============================================================"
echo ""

log_info "Phase 1: Environment Check"
log_info "Project root: $PROJECT_ROOT"

# Python
if ! command -v "$PYTHON" &>/dev/null; then
    log_error "Python not found: $PYTHON"
    exit 1
fi
PY_VER=$($PYTHON --version 2>&1)
log_ok "Python: $PY_VER"

# CUDA
if ! command -v nvidia-smi &>/dev/null; then
    log_error "nvidia-smi not found. CUDA driver not installed?"
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
log_ok "GPU: $GPU_NAME ($GPU_MEM)"

# 自动检测 GPU 架构
if [ -z "$GPU_ARCH" ]; then
    GPU_ARCH=$($PYTHON -c "
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print(f'{cap[0]}.{cap[1]}')
else:
    print('8.9')
" 2>/dev/null || echo "8.9")
    log_info "Auto-detected GPU arch: $GPU_ARCH"
else
    log_info "Using specified GPU arch: $GPU_ARCH"
fi
export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"

# Python 依赖检查
log_info "Checking Python dependencies..."
MISSING_DEPS=""
for pkg in torch triton tqdm; do
    if ! $PYTHON -c "import $pkg" 2>/dev/null; then
        MISSING_DEPS="$MISSING_DEPS $pkg"
    fi
done
if [ -n "$MISSING_DEPS" ]; then
    log_error "Missing Python packages:$MISSING_DEPS"
    log_error "Install with: pip install$MISSING_DEPS"
    exit 1
fi
log_ok "Core dependencies: torch, triton, tqdm OK"

# 可选依赖 (不阻塞)
for pkg in apex flash_attn; do
    if $PYTHON -c "import $pkg" 2>/dev/null; then
        log_ok "Optional: $pkg available"
    else
        log_warn "Optional: $pkg not installed (some apex kernels may be skipped)"
    fi
done

# nvcc
if command -v nvcc &>/dev/null; then
    NVCC_VER=$(nvcc --version | grep "release" | head -1)
    log_ok "nvcc: $NVCC_VER"
else
    log_warn "nvcc not in PATH; will use PyTorch bundled compiler"
fi

echo ""

# ============================================================
# 2. 数据集完整性检查（离线模式，不下载）
# ============================================================
log_info "Phase 2: Dataset Integrity Check (offline)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

REGISTRIES=(
    "external_benchmarks/cuda_l1/registry.json"
    "external_benchmarks/ai_cuda_engineer/registry.json"
    "external_benchmarks/tritonbench_g/registry.json"
)
REG_NAMES=("CUDA-L1" "AI CUDA Engineer" "TritonBench-G")
DATA_OK=true

for i in "${!REGISTRIES[@]}"; do
    reg="${REGISTRIES[$i]}"
    name="${REG_NAMES[$i]}"
    if [ -f "$PROJECT_ROOT/$reg" ]; then
        COUNT=$($PYTHON -c "
import json
with open('$PROJECT_ROOT/$reg', encoding='utf-8') as f:
    data = json.load(f)
print(len(data))
" 2>/dev/null || echo "0")
        log_ok "  $name: $COUNT kernels (registry OK)"
    else
        log_error "  $name: registry NOT FOUND at $reg"
        DATA_OK=false
    fi
done

# Apex 通过 registry.py 内置
if [ -f "$PROJECT_ROOT/external_benchmarks/registry.py" ]; then
    log_ok "  Apex: built-in (registry.py OK)"
else
    log_error "  Apex: registry.py NOT FOUND"
    DATA_OK=false
fi

if [ "$DATA_OK" = false ]; then
    log_error "Some dataset registries are missing!"
    log_error "Please ensure all data was copied correctly."
    exit 1
fi

# 抽样验证 problem 文件是否存在
log_info "  Verifying sample problem files..."
VERIFY_FAIL=0
for reg in "${REGISTRIES[@]}"; do
    MISSING=$($PYTHON -c "
import json, os
with open('$PROJECT_ROOT/$reg', encoding='utf-8') as f:
    data = json.load(f)
missing = 0
for entry in data[:5]:
    ref = entry.get('reference_file', '')
    full = os.path.join('$PROJECT_ROOT', ref) if not os.path.isabs(ref) else ref
    if not os.path.exists(full):
        missing += 1
print(missing)
" 2>/dev/null || echo "0")
    if [ "$MISSING" -gt 0 ]; then
        log_warn "  $reg: $MISSING of first 5 problem files missing"
        VERIFY_FAIL=$((VERIFY_FAIL + 1))
    fi
done
if [ "$VERIFY_FAIL" -eq 0 ]; then
    log_ok "  Sample problem files verified"
fi

echo ""

# ============================================================
# 3. 冒烟测试 (每个数据集跑 2 个 kernel)
# ============================================================
log_info "Phase 3: Smoke Test (--limit 2 per dataset)"

SMOKE_DATASETS=("apex" "cuda-l1" "sakana" "tritonbench")
SMOKE_PASS=0
SMOKE_FAIL=0

for ds in "${SMOKE_DATASETS[@]}"; do
    log_info "  Smoke testing: $ds ..."
    SMOKE_LOG="$LOG_DIR/smoke_${ds}.log"

    if $PYTHON scripts/run_fullscale_diff_test.py --dataset "$ds" --limit 2 \
        > "$SMOKE_LOG" 2>&1; then
        log_ok "  $ds: PASSED"
        SMOKE_PASS=$((SMOKE_PASS + 1))
    else
        log_warn "  $ds: FAILED (see $SMOKE_LOG)"
        SMOKE_FAIL=$((SMOKE_FAIL + 1))
    fi
done

echo ""
log_info "Smoke results: $SMOKE_PASS passed, $SMOKE_FAIL failed out of ${#SMOKE_DATASETS[@]}"

if [ "$SMOKE_FAIL" -gt 0 ]; then
    log_warn "Some smoke tests failed. Check logs in $LOG_DIR/smoke_*.log"
    log_warn "Failed datasets will still be attempted in full run."
fi

# 清理冒烟测试的 checkpoint（不计入正式结果）
log_info "Cleaning smoke test results..."
for ds_dir in apex_new cuda_l1 ai_cuda_engineer tritonbench_g; do
    SMOKE_CP="$RESULT_DIR/$ds_dir/checkpoint.json"
    if [ -f "$SMOKE_CP" ]; then
        rm -rf "$RESULT_DIR/$ds_dir"
    fi
done
log_ok "Smoke results cleaned"

if [ "$SMOKE_ONLY" = true ]; then
    echo ""
    log_info "Smoke-only mode. Exiting."
    echo ""
    log_ok "All smoke tests completed. Environment is ready for full run."
    log_info "To run full experiments: bash $0"
    exit 0
fi

echo ""

# ============================================================
# 4. 全量测试
# ============================================================
log_info "Phase 4: Full-Scale Testing"
log_info "Results directory: $RESULT_DIR/"
log_info "Supports checkpoint resume: if interrupted, re-run this script to continue"
echo ""

FULL_DATASETS=("apex" "cuda-l1" "sakana" "tritonbench")

for ds in "${FULL_DATASETS[@]}"; do
    FULL_LOG="$LOG_DIR/full_${ds}.log"
    log_info "===== Starting full test: $ds ====="
    log_info "Log file: $FULL_LOG"

    START_TIME=$(date +%s)

    $PYTHON scripts/run_fullscale_diff_test.py --dataset "$ds" 2>&1 | tee "$FULL_LOG" || {
        log_error "$ds: Test script exited with error. Check $FULL_LOG"
        log_warn "Continuing to next dataset..."
        continue
    }

    END_TIME=$(date +%s)
    ELAPSED=$(( END_TIME - START_TIME ))
    HOURS=$(( ELAPSED / 3600 ))
    MINS=$(( (ELAPSED % 3600) / 60 ))
    log_ok "$ds: COMPLETED in ${HOURS}h ${MINS}m"
    echo ""
done

# ============================================================
# 5. 汇总
# ============================================================
echo ""
echo "============================================================"
log_ok "ALL EXPERIMENTS COMPLETED"
echo "============================================================"
log_info "Results: $RESULT_DIR/"
log_info "Logs:    $LOG_DIR/"
echo ""

# 打印各数据集结果摘要
$PYTHON -c "
import json, os
result_base = '$RESULT_DIR'
if not os.path.isdir(result_base):
    print('  No results directory found.')
    exit()
for ds_dir in sorted(os.listdir(result_base)):
    cp = os.path.join(result_base, ds_dir, 'checkpoint.json')
    if os.path.isfile(cp):
        with open(cp, encoding='utf-8') as f:
            data = json.load(f)
        total = len(data)
        completed = sum(1 for v in data.values() if v.get('status') == 'COMPLETED')
        with_disc = sum(1 for v in data.values() if v.get('total_discrepancies', 0) > 0)
        skipped = sum(1 for v in data.values() if v.get('status') == 'SKIPPED')
        print(f'  {ds_dir}: {completed} completed, {with_disc} with discrepancies, {skipped} skipped')
" 2>/dev/null || true

echo ""
log_info "Done. Results are in: $RESULT_DIR/"
log_info "Each dataset has checkpoint.json (summary) and details/ (per-kernel JSON)."
