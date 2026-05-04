#!/bin/bash
cd /mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel
LOG="第三次实验汇总/logs/sakana_baseline_all.log"
exec > "$LOG" 2>&1 < /dev/null
exec setsid python3 -u scripts/run_sakana_baseline_all.py
