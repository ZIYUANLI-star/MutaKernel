import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, d_k = Q.shape
        
        # Save original TF32 settings
        orig_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        orig_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        
        try:
            # Disable TF32 for numerical stability
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            
            # Convert to float32 for numerical stability with extreme values
            Q_fp32 = Q.float()
            K_fp32 = K.float()
            V_fp32 = V.float()
            
            # Step 1: Compute Q @ K.transpose(-2, -1) in float32
            qk = Q_fp32 @ K_fp32.transpose(-1, -2)
            
            # Step 2: Scale by sqrt(d_k)
            scale = 1.0 / (d_k ** 0.5)
            qk = qk * scale
            
            # Step 3: Numerically stable softmax
            # Subtract max for numerical stability before exp
            qk_max = qk.max(dim=-1, keepdim=True).values
            qk_stable = qk - qk_max
            exp_qk = torch.exp(qk_stable)
            softmax_qk = exp_qk / (exp_qk.sum(dim=-1, keepdim=True) + 1e-12)
            
            # Step 4: Compute output = softmax_qk @ V
            output = softmax_qk @ V_fp32
            
            # Convert back to original dtype
            output = output.to(Q.dtype)
            
        finally:
            # Restore original TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = orig_matmul_tf32
            torch.backends.cudnn.allow_tf32 = orig_cudnn_tf32
        
        return output