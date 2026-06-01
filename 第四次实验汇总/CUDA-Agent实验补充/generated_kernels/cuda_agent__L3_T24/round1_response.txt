import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_sources = """
// === layer_norm.cu ===
#include <cuda_runtime.h>

__global__ void fused_layer_norm_kernel(const float* input, const float* weight, const float* bias, float* output, int seq_len, int batch_size, int embed_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = seq_len * batch_size;
    
    if (tid < total_elements) {
        int start = tid * embed_dim;
        
        // Calculate mean using Kahan summation for better numerical stability
        double mean = 0.0;
        double c = 0.0;
        for (int i = 0; i < embed_dim; i++) {
            double y = (double)input[start + i] - c;
            double t = mean + y;
            c = (t - mean) - y;
            mean = t;
        }
        mean /= (double)embed_dim;
        
        // Calculate variance using Welford's algorithm for numerical stability
        double var = 0.0;
        for (int i = 0; i < embed_dim; i++) {
            double diff = (double)input[start + i] - mean;
            var += diff * diff;
        }
        var /= (double)embed_dim;
        double inv_std = rsqrt(var + 1e-05);
        
        // Normalize and apply linear transformation
        for (int i = 0; i < embed_dim; i++) {
            double val = (double)input[start + i];
            double normalized_val = (val - mean) * inv_std * (double)weight[i] + (double)bias[i];
            output[start + i] = (float)normalized_val;
        }
    }
}

extern "C" void fused_layer_norm_launcher(const float* input, const float* weight, const float* bias, float* output, int seq_len, int batch_size, int embed_dim, cudaStream_t stream) {
    int total_elements = seq_len * batch_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    fused_layer_norm_kernel<<<blocks, threads, 0, stream>>>(input, weight, bias, output, seq_len, batch_size, embed_dim);
}

"""

_cpp_sources = """
// === layer_norm_binding.cpp ===
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>


extern "C" void fused_layer_norm_launcher(const float* input, const float* weight, const float* bias, float* output, int seq_len, int batch_size, int embed_dim, cudaStream_t stream);

torch::Tensor fused_layer_norm_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int seq_len, int batch_size, int embed_dim) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    TORCH_CHECK(weight.numel() == embed_dim, "Weight must have embed_dim elements");
    
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    TORCH_CHECK(bias.numel() == embed_dim, "Bias must have embed_dim elements");
    
    auto output = torch::empty_like(input);
    
    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Call CUDA launcher
    fused_layer_norm_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        seq_len,
        batch_size,
        embed_dim,
        stream
    );
    
    return output;
}

void register_fused_layer_norm(pybind11::module& m) {
    m.def("fused_layer_norm_forward", &fused_layer_norm_forward,
          "Fused Layer Norm Forward",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("seq_len"),
          py::arg("batch_size"),
          py::arg("embed_dim"));
}



"""

cuda_extension = load_inline(
    name="cuda_extension",
    cpp_sources=[_cpp_sources],
    cuda_sources=[_cuda_sources],
    functions=["fused_layer_norm_forward"],
    verbose=False,
    extra_cuda_cflags=["-O2"],
)

class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Optimized Attention Block using custom CUDA operations.
        :param embed_dim: Embedding dimension (the number of channels)
        :param num_heads: Number of attention heads
        """
        super().__init__()
        # Initialize parameters with names matching the original model
        self.attn = nn.Module()
        self.attn.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.attn.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.attn.out_proj = nn.Module()
        self.attn.out_proj.weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.attn.out_proj.bias = nn.Parameter(torch.empty(embed_dim))
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(torch.empty(embed_dim))
        self.norm.bias = nn.Parameter(torch.empty(embed_dim))
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        Forward pass of the optimized AttentionBlock.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Output tensor of the same shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        seq_len = H * W
        
        # Save original TF32 settings
        orig_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        orig_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        
        # Disable TF32 for numerical precision
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Reshape input to (seq_len, batch_size, embed_dim)
            x_reshaped = x.view(B, C, seq_len).permute(2, 0, 1)
            
            # 1. Compute q, k, v using in_proj
            qkv = x_reshaped @ self.attn.in_proj_weight.t() + self.attn.in_proj_bias
            q, k, v = qkv.chunk(3, dim=-1)
            
            # 2. Reshape for multi-head attention
            head_dim = self.embed_dim // self.num_heads
            
            q = q.reshape(seq_len, B, self.num_heads, head_dim).permute(1, 2, 0, 3)  # (B, num_heads, seq_len, head_dim)
            k = k.reshape(seq_len, B, self.num_heads, head_dim).permute(1, 2, 0, 3)  # (B, num_heads, seq_len, head_dim)
            v = v.reshape(seq_len, B, self.num_heads, head_dim).permute(1, 2, 0, 3)  # (B, num_heads, seq_len, head_dim)
            
            # 3. Compute attention weights
            scale = head_dim ** 0.5
            attn_weights = torch.matmul(q, k.transpose(-1, -2)) / scale
            attn_weights = torch.softmax(attn_weights, dim=-1)
            
            # 4. Compute output
            out = torch.matmul(attn_weights, v)
            out = out.permute(2, 0, 1, 3).reshape(seq_len, B, self.embed_dim)
            
            # 5. Output projection
            out = out @ self.attn.out_proj.weight.t() + self.attn.out_proj.bias
            
            # 6. Residual connection
            out = out + x_reshaped
            
            # 7. Fused Layer norm using custom kernel
            out = out.contiguous().reshape(seq_len * B, self.embed_dim)
            out = cuda_extension.fused_layer_norm_forward(
                out, 
                self.norm.weight.contiguous(), 
                self.norm.bias.contiguous(),
                seq_len,
                B,
                self.embed_dim
            )
            out = out.reshape(seq_len, B, self.embed_dim)
            
            # Reshape back to original shape
            out = out.permute(1, 2, 0).view(B, C, H, W)
            
            return out
        finally:
            # Restore original TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = orig_matmul_tf32
            torch.backends.cudnn.allow_tf32 = orig_cudnn_tf32