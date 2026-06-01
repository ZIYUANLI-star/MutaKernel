#!/usr/bin/env bash
set -uo pipefail
rm -f /tmp/bench.log /tmp/bench_keep.log
nohup setsid bash /mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/scripts/_run_bench.sh > /tmp/bench.log 2>&1 < /dev/null &
disown
sleep 1
nohup bash -c 'while true; do touch /tmp/bench_keep.log; sleep 30; done' > /dev/null 2>&1 < /dev/null &
disown
sleep 4
echo "--- log so far ---"
tail -20 /tmp/bench.log 2>&1 || echo "(no log yet)"
echo "--- processes ---"
ps -ef | grep -E 'benchmark_taskB|_run_bench' | grep -v grep || echo "(no proc)"
