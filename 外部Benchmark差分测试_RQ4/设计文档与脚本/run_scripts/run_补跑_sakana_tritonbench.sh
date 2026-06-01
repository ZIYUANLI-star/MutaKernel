#!/bin/bash
#
# MutaKernel 第三次实验 - AI CUDA Engineer + TritonBench-G 补跑脚本（离线、安全）
#
# 与 run_all_experiments.sh 的区别：
#   1. 只跑 sakana (AI CUDA Engineer) 和 tritonbench (TritonBench-G) 两个数据集
#   2. 绝对不会清除 cuda_l1/ 和 apex_new/ 已有的实验结果
#   3. 没有"删 checkpoint 重跑冒烟"的逻辑，保证已完成的 CUDA-L1 数据不丢
#   4. 自动支持断点续跑：中断后再次执行，会自动跳过已完成的 kernel
#
# 用法:
#   bash run_补跑_sakana_tritonbench.sh [GPU_ARCH]
#
# 参数:
#   GPU_ARCH  - GPU 计算能力 (默认自动检测)
#               例: 8.9 (RTX 40系), 8.6 (RTX 30系), 8.0 (A100), 9.0 (H100)
#
# 示例:
#   bash run_补跑_sakana_tritonbench.sh           # 自动检测 GPU，全量补跑
#   bash run_补跑_sakana_tritonbench.sh 8.6       # 手动指定 RTX 3090 / A6000
#   bash run_补跑_sakana_tritonbench.sh --skip-smoke   # 跳过冒烟测试，直接全量
#   bash run_补跑_sakana_tritonbench.sh --only sakana       # 只跑 AI CUDA Engineer
#   bash run_补跑_sakana_tritonbench.sh --only tritonbench  # 只跑 TritonBench-G
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

SKIP_SMOKE=false
ONLY=""
GPU_ARCH=""

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-smoke) SKIP_SMOKE=true; shift ;;
        --only)       ONLY="$2"; shift 2 ;;
        *)            GPU_ARCH="$1"; shift ;;
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
echo "  MutaKernel 第三次实验 - 补跑 sakana + tritonbench (离线)"
echo "  ⚠️  本脚本不会触碰 cuda_l1/ 和 apex_new/ 已有结果"
echo "============================================================"
echo ""

log_info "Phase 1: Environment Check"
log_info "Project root: $PROJECT_ROOT"

if ! command -v "$PYTHON" &>/dev/null; then
    log_error "Python not found: $PYTHON"
    exit 1
fi
log_ok "Python: $($PYTHON --version 2>&1)"

if ! command -v nvidia-smi &>/dev/null; then
    log_error "nvidia-smi not found. CUDA driver not installed?"
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
log_ok "GPU: $GPU_NAME ($GPU_MEM)"

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

# 必要依赖
log_info "Checking Python dependencies..."
MISSING_DEPS=""
for pkg in torch triton; do
    if ! $PYTHON -c "import $pkg" 2>/dev/null; then
        MISSING_DEPS="$MISSING_DEPS $pkg"
    fi
done
if [ -n "$MISSING_DEPS" ]; then
    log_error "Missing required packages:$MISSING_DEPS"
    log_error "Install with: pip install$MISSING_DEPS"
    exit 1
fi
log_ok "Required: torch, triton OK"

# nvcc
if command -v nvcc &>/dev/null; then
    log_ok "nvcc: $(nvcc --version | grep release | head -1)"
else
    log_warn "nvcc not in PATH; PyTorch bundled compiler will be used"
fi

# ============================================================
# 2. 数据集完整性检查（离线模式，不下载）
# ============================================================
echo ""
log_info "Phase 2: Dataset Integrity Check (offline)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

declare -A REGISTRIES=(
    ["sakana"]="external_benchmarks/ai_cuda_engineer/registry.json"
    ["tritonbench"]="external_benchmarks/tritonbench_g/registry.json"
)
declare -A REG_NAMES=(
    ["sakana"]="AI CUDA Engineer"
    ["tritonbench"]="TritonBench-G"
)

# 决定要跑哪些数据集
if [ -n "$ONLY" ]; then
    case "$ONLY" in
        sakana|tritonbench) DATASETS=("$ONLY") ;;
        *) log_error "--only must be one of: sakana, tritonbench (got: $ONLY)"; exit 1 ;;
    esac
else
    DATASETS=("sakana" "tritonbench")
fi

DATA_OK=true
for ds in "${DATASETS[@]}"; do
    reg="${REGISTRIES[$ds]}"
    name="${REG_NAMES[$ds]}"
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

if [ "$DATA_OK" = false ]; then
    log_error "Some dataset registries are missing! Re-pull the project from师兄's machine."
    exit 1
fi

# 抽样验证 problem 文件是否存在
log_info "  Verifying sample problem files..."
for ds in "${DATASETS[@]}"; do
    reg="${REGISTRIES[$ds]}"
    MISSING=$($PYTHON -c "
import json, os
with open('$PROJECT_ROOT/$reg', encoding='utf-8') as f:
    data = json.load(f)
miss = 0
for entry in data[:10]:
    ref = entry.get('reference_file', '')
    full = os.path.join('$PROJECT_ROOT', ref) if not os.path.isabs(ref) else ref
    if not os.path.exists(full):
        miss += 1
print(miss)
" 2>/dev/null || echo "0")
    if [ "$MISSING" -gt 0 ]; then
        log_warn "  ${REG_NAMES[$ds]}: $MISSING of first 10 problem files missing"
    else
        log_ok "  ${REG_NAMES[$ds]}: sample problem files present"
    fi
done

# ============================================================
# 3. 已有结果保护性检查
# ============================================================
echo ""
log_info "Phase 3: Existing Results Protection Check"

for protect_ds in cuda_l1 apex_new; do
    pcheckpoint="$RESULT_DIR/$protect_ds/checkpoint.json"
    if [ -f "$pcheckpoint" ]; then
        PROTECT_COUNT=$($PYTHON -c "
import json
with open('$pcheckpoint', encoding='utf-8') as f:
    data = json.load(f)
print(len(data))
" 2>/dev/null || echo "?")
        log_ok "  $protect_ds: $PROTECT_COUNT 已有结果，本脚本不会触碰"
    else
        log_info "  $protect_ds: 无现存结果（无需保护）"
    fi
done

for cur_ds in ai_cuda_engineer tritonbench_g; do
    ccheckpoint="$RESULT_DIR/$cur_ds/checkpoint.json"
    if [ -f "$ccheckpoint" ]; then
        CUR_COUNT=$($PYTHON -c "
import json
with open('$ccheckpoint', encoding='utf-8') as f:
    data = json.load(f)
print(len(data))
" 2>/dev/null || echo "?")
        log_info "  $cur_ds: 已完成 $CUR_COUNT 个 kernel，本次将自动跳过续跑"
    else
        log_info "  $cur_ds: 无现存 checkpoint，本次将从零开始"
    fi
done

# ============================================================
# 4. 冒烟测试（每个数据集 2 个 kernel，可选跳过）
# ============================================================
if [ "$SKIP_SMOKE" = false ]; then
    echo ""
    log_info "Phase 4: Smoke Test (--limit 2 per dataset)"
    log_warn "  ⚠️  冒烟测试不会清除任何 checkpoint，但会真的把 2 个 kernel 写入 checkpoint。"
    log_warn "  如果想完全跳过冒烟，下次加 --skip-smoke 参数。"

    SMOKE_PASS=0
    SMOKE_FAIL=0
    for ds in "${DATASETS[@]}"; do
        log_info "  Smoke testing: $ds ..."
        SMOKE_LOG="$LOG_DIR/smoke_补跑_${ds}.log"
        if $PYTHON scripts/run_fullscale_diff_test.py --dataset "$ds" --limit 2 \
            > "$SMOKE_LOG" 2>&1; then
            log_ok "  $ds: smoke PASSED"
            SMOKE_PASS=$((SMOKE_PASS + 1))
        else
            log_warn "  $ds: smoke FAILED (查看 $SMOKE_LOG)"
            SMOKE_FAIL=$((SMOKE_FAIL + 1))
        fi
    done
    log_info "Smoke results: $SMOKE_PASS passed, $SMOKE_FAIL failed of ${#DATASETS[@]}"
    if [ "$SMOKE_FAIL" -gt 0 ]; then
        log_warn "冒烟测试有失败，但全量测试仍会继续（失败的 kernel 会被记录为 SKIPPED）。"
    fi
else
    echo ""
    log_info "Phase 4: Smoke Test SKIPPED (--skip-smoke)"
fi

# ============================================================
# 5. 全量测试
# ============================================================
echo ""
log_info "Phase 5: Full-Scale Testing (sakana + tritonbench)"
log_info "  Results dir: $RESULT_DIR/{ai_cuda_engineer,tritonbench_g}/"
log_info "  断点续跑：中断后再次执行同一脚本即可继续。"
echo ""

for ds in "${DATASETS[@]}"; do
    FULL_LOG="$LOG_DIR/full_补跑_${ds}.log"
    log_info "===== Starting full test: $ds ====="
    log_info "Log: $FULL_LOG"

    START_TIME=$(date +%s)

    $PYTHON scripts/run_fullscale_diff_test.py --dataset "$ds" 2>&1 | tee "$FULL_LOG" || {
        log_error "$ds: 测试脚本异常退出。Log: $FULL_LOG"
        log_warn "继续下一个数据集..."
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
# 6. 汇总
# ============================================================
echo ""
echo "============================================================"
log_ok "补跑任务结束"
echo "============================================================"
log_info "Results: $RESULT_DIR/"
log_info "Logs:    $LOG_DIR/"
echo ""

$PYTHON -c "
import json, os
result_base = '$RESULT_DIR'
print('  当前所有数据集结果概览：')
for ds_dir in sorted(os.listdir(result_base)):
    cp = os.path.join(result_base, ds_dir, 'checkpoint.json')
    if os.path.isfile(cp):
        with open(cp, encoding='utf-8') as f:
            data = json.load(f)
        total = len(data)
        completed = sum(1 for v in data.values() if isinstance(v, dict) and v.get('status') == 'COMPLETED')
        with_disc = sum(1 for v in data.values() if isinstance(v, dict) and v.get('total_discrepancies', 0) > 0)
        skipped = sum(1 for v in data.values() if isinstance(v, dict) and v.get('status') == 'SKIPPED')
        print(f'    {ds_dir}: {completed} completed, {with_disc} with discrepancies, {skipped} skipped (total checkpoint entries: {total})')
" 2>/dev/null || true

echo ""
log_info "如有未完成 kernel，再次执行本脚本即会从断点续跑。"
