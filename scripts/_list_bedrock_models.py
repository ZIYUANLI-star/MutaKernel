"""List foundation models available in this Bedrock region via boto3.

Uses the Bedrock API key in env var AWS_BEARER_TOKEN_BEDROCK (or BEDROCK_API_KEY).
"""
from __future__ import annotations
import os
import sys

# boto3 reads AWS_BEARER_TOKEN_BEDROCK for Bedrock API key auth
key = os.environ.get("BEDROCK_API_KEY") or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
if not key:
    print("ERR: BEDROCK_API_KEY env var empty", file=sys.stderr)
    sys.exit(2)
os.environ["AWS_BEARER_TOKEN_BEDROCK"] = key

import boto3

REGION = os.environ.get("AWS_REGION", "us-west-2")

bedrock = boto3.client("bedrock", region_name=REGION)

print(f"=== Foundation models in {REGION} (Anthropic only) ===")
try:
    resp = bedrock.list_foundation_models(byProvider="Anthropic")
    for m in resp.get("modelSummaries", []):
        mid = m.get("modelId", "")
        mods = m.get("inferenceTypesSupported", [])
        status = m.get("modelLifecycle", {}).get("status", "")
        print(f"  {mid}  status={status}  inf={mods}")
except Exception as e:
    print(f"ERR list_foundation_models: {e}")

print(f"\n=== Inference profiles in {REGION} (Anthropic only) ===")
try:
    resp = bedrock.list_inference_profiles()
    for p in resp.get("inferenceProfileSummaries", []):
        pid = p.get("inferenceProfileId", "")
        pname = p.get("inferenceProfileName", "")
        if "anthropic" in pid.lower() or "claude" in pid.lower() or "claude" in pname.lower():
            print(f"  id={pid}  name={pname}")
except Exception as e:
    print(f"ERR list_inference_profiles: {e}")
