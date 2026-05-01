#!/usr/bin/env python3
"""Probe which apex.contrib / apex.optimizers modules are available."""
import json, sys

results = {}

probes = [
    ("apex.contrib.group_norm", "GroupNorm"),
    ("apex.contrib.layer_norm", "FastLayerNorm"),
    ("apex.contrib.xentropy", "SoftmaxCrossEntropyLoss"),
    ("apex.contrib.conv_bias_relu", "ConvBiasReLU"),
    ("apex.optimizers", "FusedAdam"),
    ("apex.optimizers", "FusedSGD"),
    ("apex.optimizers", "FusedLAMB"),
    ("apex.contrib.multihead_attn", "SelfMultiheadAttn"),
    ("apex.contrib.optimizers", "FusedAdam"),
    ("apex.mlp", "MLP"),
]

for mod_path, cls_name in probes:
    key = f"{mod_path}.{cls_name}"
    try:
        mod = __import__(mod_path, fromlist=[cls_name])
        cls = getattr(mod, cls_name, None)
        if cls is not None:
            results[key] = {"available": True, "type": str(type(cls).__name__)}
        else:
            results[key] = {"available": False, "reason": f"{cls_name} not in {mod_path}"}
    except ImportError as e:
        results[key] = {"available": False, "reason": str(e)[:200]}
    except Exception as e:
        results[key] = {"available": False, "reason": str(e)[:200]}

print(json.dumps(results, indent=2))
