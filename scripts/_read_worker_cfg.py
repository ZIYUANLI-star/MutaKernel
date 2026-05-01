#!/usr/bin/env python3
import json, sys, glob, os

for f in sorted(glob.glob("/tmp/fscfg_*.json"), key=os.path.getmtime, reverse=True):
    try:
        with open(f) as fh:
            d = json.load(fh)
        print("Config: " + f)
        print("  keys: " + str(list(d.keys())))
        print("  problem_file: " + os.path.basename(d.get("problem_file", "?")))
        for k in ["test_type", "mode", "policy", "policy_name", "dtype", "batch_sizes", "seeds", "n_repeat", "seed"]:
            if k in d:
                print("  " + k + ": " + str(d[k]))
        break
    except Exception:
        continue
