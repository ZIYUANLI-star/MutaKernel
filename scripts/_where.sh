#!/usr/bin/env bash
ROOT=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel
SUMMARY="$ROOT/第二次实验汇总"
SUPPL="$SUMMARY/第二次实验汇总_补充"
LOGDIR=/home/kbuser/mutakernel_logs

cnt() { ls "$1" 2>/dev/null | wc -l; }
human() { du -sh "$1" 2>/dev/null | awk '{print $1}'; }

echo "=========================================="
echo "  1. WSL 实时运行日志"
echo "=========================================="
for f in task_a.log task_c.log task_a.pid task_c.pid; do
  if [[ -f "$LOGDIR/$f" ]]; then
    sz=$(du -h "$LOGDIR/$f" | awk '{print $1}')
    echo "  $LOGDIR/$f   ($sz)"
  fi
done
if [[ -d "$LOGDIR/_archive" ]]; then
  echo "  $LOGDIR/_archive/   (旧 wrong-protocol log)"
  ls "$LOGDIR/_archive/" | sed 's/^/    /'
fi
echo

echo "=========================================="
echo "  2. 原始实验数据（第二次实验汇总/）"
echo "=========================================="
echo "  full_block12_results/   (Phase I — EMD 4 层)"
echo "    details/      $(cnt $SUMMARY/full_block12_results/details) 个 kernel json   $(human $SUMMARY/full_block12_results)"
echo "  stress_enhance_results/ (Phase II — 5 维 stress + DeepSeek 3 轮 LLM)"
echo "    details/      $(cnt $SUMMARY/stress_enhance_results/details) 个 mutant json   $(human $SUMMARY/stress_enhance_results)"
echo

echo "=========================================="
echo "  3. 新实验数据（第二次实验汇总_补充/）"
echo "=========================================="
echo "  docs/  (实验设置 + 提示词全集 + Linux 执行说明)"
ls "$SUPPL/docs/" 2>/dev/null | sed 's/^/    /'
echo
for t in task_a_phase2_rerun task_c_phase1_direct task_b_kernel_strengthen; do
  echo "  $t/"
  if [[ -d "$SUPPL/$t" ]]; then
    for d in details prompts llm_responses run_manifest.json completed.json; do
      if [[ -e "$SUPPL/$t/$d" ]]; then
        if [[ -d "$SUPPL/$t/$d" ]]; then
          n=$(cnt "$SUPPL/$t/$d")
          sz=$(human "$SUPPL/$t/$d")
          echo "    $d/   ($n 个文件, $sz)"
        else
          sz=$(du -h "$SUPPL/$t/$d" | awk '{print $1}')
          echo "    $d   ($sz)"
        fi
      fi
    done
  else
    echo "    (尚未创建 — Task B 待跑)"
  fi
  echo
done

echo "=========================================="
echo "  4. 归档（旧 wrong-protocol 数据，已保留）"
echo "=========================================="
if [[ -d "$SUPPL/_archive" ]]; then
  for d in "$SUPPL/_archive"/*/; do
    if [[ -d "$d" ]]; then
      n_details=$(cnt "$d/details")
      sz=$(human "$d")
      echo "  $(basename $d)"
      echo "    details/    $n_details 个 mutant   $sz"
    fi
  done
fi
echo

echo "=========================================="
echo "  5. 源代码 + 配置（项目根）"
echo "=========================================="
echo "  $ROOT/.env                              (Bedrock API 凭证 — gitignored)"
echo "  $ROOT/best_kernels.json                 (44 个 kernel 路径索引)"
echo "  $ROOT/src/stress/llm_analyzer.py        (所有 prompt 模板)"
echo "  $ROOT/src/stress/llm_clients.py         (Bedrock 客户端)"
echo "  $ROOT/src/stress/evidence_collector.py  (Task B 证据聚合器)"
echo "  $ROOT/scripts/run_taskA_phase2_rerun.py (Task A 主脚本)"
echo "  $ROOT/scripts/run_taskC_phase1_direct.py (Task C 主脚本)"
echo "  $ROOT/scripts/_launch_taskA.sh / _launch_taskC.sh   (后台启动)"
echo "  $ROOT/scripts/_status.sh / _progress_report.py       (进度监控)"
echo "  $ROOT/scripts/_kill_tasks.sh / _archive_and_restart.sh (停 + 回滚)"
