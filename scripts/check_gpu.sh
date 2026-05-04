#!/bin/bash
echo "=== Looking for conda/torch env ==="
which conda 2>/dev/null && conda env list 2>/dev/null
echo "---"
find /home -maxdepth 3 -name "activate" -path "*/bin/*" 2>/dev/null | head -5
echo "---"
find /home -maxdepth 4 -type d -name "torch" 2>/dev/null | head -5
echo "---"
# Check if there's a venv or conda in kbuser
ls /home/kbuser/.conda/envs/ 2>/dev/null
ls /home/kbuser/miniconda*/envs/ 2>/dev/null
ls /home/kbuser/anaconda*/envs/ 2>/dev/null
echo "=== pip locations ==="
find /home/kbuser -name "pip" -type f 2>/dev/null | head -5
find /home/kbuser -name "python" -o -name "python3" -type f 2>/dev/null | grep -v __pycache__ | head -5
