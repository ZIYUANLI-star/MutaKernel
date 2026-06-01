import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_sources = """
// === fused_multiply_leaky_relu.cu ===
#include <cuda_runtime.h>

__global__ void fused_multiply_leaky_relu_kernel(float* output, const float* gemm_out, float multiplier, float negative_slope, int total_elems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elems) {
        float val = gemm_out[idx] * multiplier;
        output[idx] = val > 0.0f ? val : negative_slope * val;
    }
}

extern "C" void fused_multiply_leaky_relu_launcher(float* output, const float* gemm_out, float multiplier, float negative_slope, int total_elems, cudaStream_t stream) {
    int threads = 256;
    int blocks = (total_elems + threads - 1) / threads;
    fused_multiply_leaky_relu_kernel<<<blocks, threads, 0, stream>>>(output, gemm_out, multiplier, negative_slope, total_elems);
}
"""

_cpp_sources = """
// === fused_multiply_leaky_relu_binding.cpp ===
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>


extern "C" void fused_multiply_leaky_relu_launcher(float* output, const float* gemm_out, float multiplier, float negative_slope, int total_elems, cudaStream_t stream);

torch::Tensor fused_multiply_leaky_relu_forward(torch::Tensor gemm_out, double multiplier, double negative_slope) {
    TORCH_CHECK(gemm_out.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(gemm_out.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(gemm_out.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(gemm_out);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    int total_elems = gemm_out.numel();

    fused_multiply_leaky_relu_launcher(
        output.data_ptr<float>(), 
        gemm_out.data_ptr<float>(),
        static_cast<float>(multiplier),
        static_cast<float>(negative_slope),
        total_elems,
        stream
    );

    return output;
}

void register_fused_multiply_leaky_relu(pybind11::module& m) {
    m.def("fused_multiply_leaky_relu_forward", &fused_multiply_leaky_relu_forward,
          "Fused multiply and leaky relu forward");
}


"""

cuda_extension = load_inline(
    name="cuda_extension",
    cpp_sources=[_cpp_sources],
    cuda_sources=[_cuda_sources],
    functions=["fused_multiply_leaky_relu_forward"],
    verbose=False,
    extra_cuda_cflags=["-O2"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a GEMM followed by a fused multiply + leaky relu.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        # Use the same name as original model for state_dict compatibility
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.negative_slope = negative_slope

    def forward(self, x):
        # Disable TF32 to ensure numerical precision matches reference
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Use PyTorch's linear layer implementation
            x = self.gemm(x)
            
            # Use fused kernel for multiply + leaky relu with dynamic parameters
            x = cuda_extension.fused_multiply_leaky_relu_forward(x, self.multiplier, self.negative_slope)
        finally:
            # Restore TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        
        return x