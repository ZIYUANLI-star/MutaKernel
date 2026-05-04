#!/bin/bash
echo "HOME=$HOME"
echo "=== home contents ==="
ls -la ~/
echo "=== find KernelBench ==="
find / -maxdepth 4 -type d -name "KernelBench" 2>/dev/null | head -5
echo "=== find level1 ==="
find / -maxdepth 6 -type d -name "level1" 2>/dev/null | head -5
