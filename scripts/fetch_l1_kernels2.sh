#!/bin/bash
echo "=== Checking KernelBench location ==="
ls -d /home/dog/KernelBench* 2>/dev/null
ls /home/dog/ 2>/dev/null | head -20
echo "=== Checking level1 ==="
find /home/dog -maxdepth 5 -type d -name "level1" 2>/dev/null | head -5
echo "=== Looking for .py kernel files ==="
find /home/dog -maxdepth 6 -type f -name "*.py" -path "*/level1/*" 2>/dev/null | head -10
