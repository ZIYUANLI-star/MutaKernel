import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_sources = """
// === cublas_matmul_fused_post.cu ===
#include <cuda_runtime.h>

__global__ void fused_post_matmul_kernel(float* output, const float* cublas_out, const float* bias,
                                          int batch_size, int out_features, float scaling_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;

    if (idx < total_elements) {
        int out_feature = idx % out_features;
        float val = cublas_out[idx] + bias[out_feature];
        
        // Numerically stable sigmoid computation
        float sig;
        if (val >= 0.0f) {
            float exp_neg_val = expf(-val);
            sig = 1.0f / (1.0f + exp_neg_val);
        } else {
            float exp_val = expf(val);
            sig = exp_val / (1.0f + exp_val);
        }
        
        output[idx] = val * sig * scaling_factor;
    }
}

extern "C" void fused_post_matmul_launcher(float* output, const float* cublas_out, const float* bias,
                                            int batch_size, int out_features, float scaling_factor,
                                            cudaStream_t stream) {
    int total_elements = batch_size * out_features;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_post_matmul_kernel<<<blocks, threads, 0, stream>>>(output, cublas_out, bias,
                                                              batch_size, out_features, scaling_factor);
}

"""

_cpp_sources = """
// === cublas_matmul_fused_post_binding.cpp ===
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>


extern "C" void fused_post_matmul_launcher(float* output, const float* cublas_out, const float* bias,
                                            int batch_size, int out_features, float scaling_factor,
                                            cudaStream_t stream);

torch::Tensor fused_post_matmul_forward(torch::Tensor cublas_out, torch::Tensor bias, float scaling_factor) {
    TORCH_CHECK(cublas_out.is_cuda() && bias.is_cuda(), "All tensors must be CUDA tensors");
    TORCH_CHECK(cublas_out.is_contiguous() && bias.is_contiguous(), "All tensors must be contiguous");
    TORCH_CHECK(cublas_out.dtype() == torch::kFloat32 && bias.dtype() == torch::kFloat32, "All tensors must be float32");
    
    int batch_size = cublas_out.size(0);
    int out_features = cublas_out.size(1);
    
    TORCH_CHECK(bias.size(0) == out_features, "bias size must match out_features");

    auto output = torch::empty_like(cublas_out);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    fused_post_matmul_launcher(output.data_ptr<float>(), cublas_out.data_ptr<float>(), bias.data_ptr<float>(),
                                batch_size, out_features, scaling_factor, stream);

    return output;
}

void register_fused_post_matmul(pybind11::module& m) {
    m.def("fused_post_matmul_forward", &fused_post_matmul_forward,
          "Fused bias addition, swish activation, and scaling after matrix multiplication",
          py::arg("cublas_out"), py::arg("bias"), py::arg("scaling_factor"));
}



"""

cuda_extension = load_inline(
    name="cuda_extension",
    cpp_sources=[_cpp_sources],
    cuda_sources=[_cuda_sources],
    functions=["fused_post_matmul_forward"],
    verbose=False,
    extra_cuda_cflags=["-O2"],
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses bias addition, Swish activation, and scaling.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Disable TF32 to ensure numerical precision matches reference
        old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        old_tf32_cudnn = torch.backends.cudnn.allow_tf32
        
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Perform matrix multiplication without bias
            cublas_out = torch.matmul(x, self.matmul.weight.transpose(0, 1))
            
            # Use fused kernel for bias addition, swish activation, and scaling
            x = cuda_extension.fused_post_matmul_forward(cublas_out, self.matmul.bias, self.scaling_factor)
        finally:
            # Restore original TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
            torch.backends.cudnn.allow_tf32 = old_tf32_cudnn
        
        return x