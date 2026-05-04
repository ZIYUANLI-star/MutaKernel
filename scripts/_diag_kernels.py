#!/usr/bin/env python3
"""Diagnose external kernel wrapper issues."""
import torch
import torch.nn.functional as F
import sys, os

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
device = "cuda"

print("=" * 60)
print("  Diagnostic: External Kernel Wrappers")
print("=" * 60)

# --- 1. FusedSoftmax ---
print("\n--- FusedSoftmax ---")
try:
    from apex.transformer.functional import FusedScaleMaskSoftmax
    from apex.transformer.enums import AttnMaskType

    x = torch.randn(16, 12, 512, 512, device=device)
    ref_out = F.softmax(x, dim=-1)

    fused = FusedScaleMaskSoftmax(
        input_in_fp16=False,
        input_in_bf16=False,
        attn_mask_type=AttnMaskType.padding,
        scaled_masked_softmax_fusion=True,
        mask_func=lambda val, mask: val,
        softmax_in_fp32=True,
        scale=1.0,
    )
    ext_out = fused(x, None)

    diff = (ref_out - ext_out).abs()
    print(f"  FP32: max_diff={diff.max().item():.6e}, mean_diff={diff.mean().item():.6e}")
    print(f"  allclose(atol=1e-2): {torch.allclose(ref_out, ext_out, atol=1e-2, rtol=1e-2)}")
    print(f"  ref_out[:2,:2,0,0]: {ref_out[0,0,0,:4].tolist()}")
    print(f"  ext_out[:2,:2,0,0]: {ext_out[0,0,0,:4].tolist()}")
    print(f"  shapes: ref={ref_out.shape}, ext={ext_out.shape}")

    # Try with fp16
    x16 = x.half()
    fused16 = FusedScaleMaskSoftmax(
        input_in_fp16=True,
        input_in_bf16=False,
        attn_mask_type=AttnMaskType.padding,
        scaled_masked_softmax_fusion=True,
        mask_func=lambda val, mask: val,
        softmax_in_fp32=True,
        scale=1.0,
    )
    ref16 = F.softmax(x16, dim=-1)
    ext16 = fused16(x16, None)
    diff16 = (ref16.float() - ext16.float()).abs()
    print(f"  FP16: max_diff={diff16.max().item():.6e}, mean_diff={diff16.mean().item():.6e}")
    print(f"  allclose(atol=1e-2): {torch.allclose(ref16, ext16, atol=1e-2, rtol=1e-2)}")

except Exception as e:
    print(f"  ERROR: {e}")

# --- 2. FusedDense weight sync test ---
print("\n--- FusedDense (weight sync) ---")
try:
    from apex.fused_dense import FusedDense

    torch.manual_seed(42)
    ref = torch.nn.Linear(1024, 2048).to(device).eval()
    ext = FusedDense(1024, 2048).to(device).eval()

    x = torch.randn(32, 128, 1024, device=device)

    # Before sync
    ref_out = ref(x)
    ext_out = ext(x)
    diff_no_sync = (ref_out - ext_out).abs().max().item()
    print(f"  Before weight sync: max_diff={diff_no_sync:.6e}")

    # After sync
    ext.load_state_dict(ref.state_dict(), strict=False)
    ext_out2 = ext(x)
    diff_sync = (ref_out - ext_out2).abs().max().item()
    print(f"  After weight sync:  max_diff={diff_sync:.6e}")
    print(f"  allclose(atol=1e-5): {torch.allclose(ref_out, ext_out2, atol=1e-5)}")

except Exception as e:
    print(f"  ERROR: {e}")

# --- 3. FlashAttention ---
print("\n--- FlashAttention ---")
try:
    from flash_attn import flash_attn_func

    B, S, H, D = 4, 256, 8, 64
    q = torch.randn(B, S, H, D, dtype=torch.float16, device=device)
    k = torch.randn(B, S, H, D, dtype=torch.float16, device=device)
    v = torch.randn(B, S, H, D, dtype=torch.float16, device=device)

    # Reference: SDPA (B,H,S,D)
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    ref_out = F.scaled_dot_product_attention(q_t, k_t, v_t).transpose(1, 2)

    # External: flash_attn_func (B,S,H,D)
    ext_out = flash_attn_func(q, k, v)

    diff = (ref_out.float() - ext_out.float()).abs()
    print(f"  max_diff={diff.max().item():.6e}, mean_diff={diff.mean().item():.6e}")
    print(f"  allclose(atol=1e-2): {torch.allclose(ref_out, ext_out, atol=1e-2, rtol=1e-2)}")
    print(f"  allclose(atol=1e-3): {torch.allclose(ref_out, ext_out, atol=1e-3, rtol=1e-3)}")
    print(f"  shapes: ref={ref_out.shape}, ext={ext_out.shape}")

except Exception as e:
    print(f"  ERROR: {e}")

# --- 4. FusedDenseGeluDense ---
print("\n--- FusedDenseGeluDense ---")
try:
    from apex.fused_dense import FusedDenseGeluDense

    IN_F, INTER, OUT_F = 1024, 4096, 1024
    ext = FusedDenseGeluDense(IN_F, INTER, OUT_F).to(device).eval()
    print(f"  Module created OK, params: {sum(p.numel() for p in ext.parameters())}")
    print(f"  state_dict keys: {list(ext.state_dict().keys())}")

    # Check what reference params would look like
    ref_linear1 = torch.nn.Linear(IN_F, INTER)
    ref_linear2 = torch.nn.Linear(INTER, OUT_F)
    print(f"  ref Linear1 keys: {list(ref_linear1.state_dict().keys())}")
    print(f"  ref Linear2 keys: {list(ref_linear2.state_dict().keys())}")

except Exception as e:
    print(f"  ERROR: {e}")

print("\nDone.")
