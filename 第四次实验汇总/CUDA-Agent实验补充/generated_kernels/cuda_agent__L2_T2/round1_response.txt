import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_sources = """
// === fused_post_linear.cu ===
#include <cuda_runtime.h>

__global__ void fused_post_linear_kernel(
    float* output,
    const float* gemm_out,
    const float* bias,
    float subtract_value,
    float multiply_value,
    int batch_size,
    int out_features) {

    int idx = blockIdx.x * 256 + threadIdx.x;
    int total_elements = batch_size * out_features;

    if (idx < total_elements) {
        int col = idx % out_features;

        float val = gemm_out[idx] + bias[col];
        val = (val - subtract_value) * multiply_value;
        output[idx] = fmaxf(val, 0.0f);
    }
}

extern "C" void fused_post_linear_launcher(
    float* output,
    const float* gemm_out,
    const float* bias,
    float subtract_value,
    float multiply_value,
    int batch_size,
    int out_features,
    cudaStream_t stream) {

    int blocks = (batch_size * out_features + 256 - 1) / 256;
    fused_post_linear_kernel<<<blocks, 256, 0, stream>>>(
        output, gemm_out, bias, subtract_value, multiply_value, batch_size, out_features
    );
}

"""

_cpp_sources = """
// === fused_post_linear_binding.cpp ===
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>


extern "C" void fused_post_linear_launcher(
    float* output,
    const float* gemm_out,
    const float* bias,
    float subtract_value,
    float multiply_value,
    int batch_size,
    int out_features,
    cudaStream_t stream);

torch::Tensor fused_post_linear_forward(torch::Tensor gemm_out, torch::Tensor bias, float subtract_value, float multiply_value) {
    TORCH_CHECK(gemm_out.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(gemm_out.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(gemm_out.dtype() == torch::kFloat32, "Input must be float32");

    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");

    auto output = torch::empty_like(gemm_out);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    fused_post_linear_launcher(
        output.data_ptr<float>(),
        gemm_out.data_ptr<float>(),
        bias.data_ptr<float>(),
        subtract_value,
        multiply_value,
        gemm_out.size(0),
        gemm_out.size(1),
        stream
    );

    return output;
}

void register_fused_post_linear(pybind11::module& m) {
    m.def("fused_post_linear_forward", &fused_post_linear_forward,
          "Fused post linear operations (subtract, multiply, relu)",
          py::arg("gemm_out"),
          py::arg("bias"),
          py::arg("subtract_value"),
          py::arg("multiply_value"));
}



"""

cuda_extension = load_inline(
    name="cuda_extension",
    cpp_sources=[_cpp_sources],
    cuda_sources=[_cuda_sources],
    functions=["fused_post_linear_forward"],
    verbose=False,
    extra_cuda_cflags=["-O2"],
)

class ModelNew(nn.Module):
    """
    Model that performs matrix multiplication and fused post operations.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def forward(self, x):
        # Disable TF32 to ensure numerical precision matches reference
        old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        old_tf32_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Matrix multiplication without bias (we'll add it in the fused kernel)
            gemm_out = x @ self.linear.weight.t()
            
            # Fused post operations
            output = cuda_extension.fused_post_linear_forward(
                gemm_out, 
                self.linear.bias, 
                float(self.subtract_value), 
                float(self.multiply_value)
            )
        finally:
            # Restore TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
            torch.backends.cudnn.allow_tf32 = old_tf32_cudnn
        
        return output