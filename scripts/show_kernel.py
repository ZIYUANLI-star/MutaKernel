import json
from pathlib import Path
with open("best_kernels.json") as f:
    bk = json.load(f)
info = bk["L1_P1"]
kf = Path(info["kernel_path"])
src = kf.read_text()
print(src)
