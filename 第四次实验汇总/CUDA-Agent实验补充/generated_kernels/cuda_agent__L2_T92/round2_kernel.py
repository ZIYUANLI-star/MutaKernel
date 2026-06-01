import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_sources = """
// === optimized_sum.cu ===
#include <cuda_runtime.h>

__global__ void optimized_sum_kernel(const float* input, const double* weight_sum, const double bias_sum, float* output, int batch_size, int in_features) {
    int idx = blockIdx.x;
    
    if (idx < batch_size) {
        // Use double precision for accumulation to avoid numerical instability
        double dot = 0.0;
        
        // Calculate dot product of input[i] with weight_sum
        const float* x = input + idx * in_features;
        for (int k = 0; k < in_features; k++) {
            dot += (double)x[k] * weight_sum[k];
        }
        
        output[idx] = (float)(dot + bias_sum);
    }
}

extern "C" void optimized_sum_launcher(const float* input, const double* weight_sum, const double bias_sum, float* output, int batch_size, int in_features, cudaStream_t stream) {
    optimized_sum_kernel<<<batch_size, 1, 0, stream>>>(input, weight_sum, bias_sum, output, batch_size, in_features);
}
"""

_cpp_sources = """
// === optimized_sum_binding.cpp ===
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>


extern "C" void optimized_sum_launcher(
    const float* input,
    const double* weight_sum,
    const double bias_sum,
    float* output,
    int batch_size,
    int in_features,
    cudaStream_t stream
);

torch::Tensor optimized_sum_forward(torch::Tensor input, torch::Tensor weight_sum, double bias_sum) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    
    TORCH_CHECK(weight_sum.is_cuda(), "Weight sum must be a CUDA tensor");
    TORCH_CHECK(weight_sum.is_contiguous(), "Weight sum must be contiguous");
    TORCH_CHECK(weight_sum.dtype() == torch::kFloat64, "Weight sum must be float64");
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    
    auto output = torch::empty({batch_size}, input.options());
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    optimized_sum_launcher(
        input.data_ptr<float>(),
        weight_sum.data_ptr<double>(),
        bias_sum,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        stream
    );
    
    return output.unsqueeze(-1);
}

void register_optimized_sum(pybind11::module& m) {
    m.def("optimized_sum_forward", &optimized_sum_forward,
          "Optimized sum forward",
          py::arg("input"),
          py::arg("weight_sum"),
          py::arg("bias_sum"));
}


"""

cuda_extension = load_inline(
    name="cuda_extension",
    cpp_sources=[_cpp_sources],
    cuda_sources=[_cuda_sources],
    functions=["optimized_sum_forward"],
    verbose=False,
    extra_cuda_cflags=["-O2"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
        # Precompute weight sum and bias sum
        self.register_buffer('weight_sum', None)
        self.bias_sum = None

    def forward(self, x):
        # Disable TF32 for numerical stability
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        # Recompute weight_sum each time to handle training mode where weights change
        # Keep in double precision throughout to avoid precision loss
        weight_sum = self.linear.weight.double().sum(dim=0).contiguous()
        bias_sum = self.linear.bias.double().sum().item()
        
        x = x.contiguous()
        x = cuda_extension.optimized_sum_forward(x, weight_sum, bias_sum)
        # All subsequent operations are on a single element per batch, so just return it
        return x

batch_size = 1024
in_features  = 8192  
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]