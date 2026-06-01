import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_sources = """
// === group_norm.cu ===
__global__ void group_norm_kernel(const float* x, const float* gamma, const float* beta, float* output,
                                   int batch_size, int features, int num_groups, int spatial_dim) {
    int tid = threadIdx.x;
    
    const int channels_per_group = features / num_groups;
    const int elements_per_group = channels_per_group * spatial_dim;
    const int threads_per_block = 256;

    __shared__ double sum_sh[256];
    __shared__ double sum_sq_sh[256];

    int block_idx = blockIdx.x;
    int batch = block_idx / num_groups;
    int group = block_idx % num_groups;

    // Initialize shared memory
    sum_sh[tid] = 0.0;
    sum_sq_sh[tid] = 0.0;
    __syncthreads();

    // Step 1: Calculate sum and sum_sq using double precision for accumulation
    double sum_val = 0.0;
    double sum_sq_val = 0.0;
    const float* x_base = x + batch * features * spatial_dim + group * channels_per_group * spatial_dim;

    for (int i = tid; i < elements_per_group; i += threads_per_block) {
        double val = (double)x_base[i];
        sum_val += val;
        sum_sq_val += val * val;
    }

    // Reduce sum and sum_sq across threads
    sum_sh[tid] = sum_val;
    sum_sq_sh[tid] = sum_sq_val;
    __syncthreads();

    for (int s = threads_per_block / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_sh[tid] += sum_sh[tid + s];
            sum_sq_sh[tid] += sum_sq_sh[tid + s];
        }
        __syncthreads();
    }

    double mean = sum_sh[0] / (double)elements_per_group;
    double var = sum_sq_sh[0] / (double)elements_per_group - mean * mean;
    
    // Clamp variance to be non-negative (can be slightly negative due to floating point)
    if (var < 0.0) var = 0.0;
    
    const double eps = 1e-05;
    double inv_std = 1.0 / sqrt(var + eps);

    // Convert to float for the normalization step
    float mean_f = (float)mean;
    float inv_std_f = (float)inv_std;

    __syncthreads();

    // Step 2: Normalize and apply affine transformation
    for (int f = 0; f < channels_per_group; f++) {
        int feature_idx = group * channels_per_group + f;
        float gamma_val = gamma[feature_idx];
        float beta_val = beta[feature_idx];

        const float* x_feature_base = x_base + f * spatial_dim;
        float* out_feature_base = output + batch * features * spatial_dim + feature_idx * spatial_dim;

        float gamma_inv_std = gamma_val * inv_std_f;
        float beta_minus_mean_gamma_inv_std = beta_val - mean_f * gamma_inv_std;

        for (int i = tid; i < spatial_dim; i += threads_per_block) {
            float val = x_feature_base[i];
            out_feature_base[i] = val * gamma_inv_std + beta_minus_mean_gamma_inv_std;
        }
    }
}

extern "C" void group_norm_launcher(const float* x, const float* gamma, const float* beta, float* output,
                                     int batch_size, int features, int num_groups, int spatial_dim,
                                     cudaStream_t stream) {
    int blocks = batch_size * num_groups; // 1 block per batch element per group
    group_norm_kernel<<<blocks, 256, 0, stream>>>(x, gamma, beta, output, batch_size, features, num_groups, spatial_dim);
}
"""

_cpp_sources = """
// === group_norm_binding.cpp ===
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>


extern "C" void group_norm_launcher(const float* x, const float* gamma, const float* beta, float* output,
                                     int batch_size, int features, int num_groups, int spatial_dim,
                                     cudaStream_t stream);

torch::Tensor group_norm_forward(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int num_groups) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    TORCH_CHECK(gamma.is_cuda(), "Gamma must be a CUDA tensor");
    TORCH_CHECK(gamma.is_contiguous(), "Gamma must be contiguous");
    TORCH_CHECK(gamma.dtype() == torch::kFloat32, "Gamma must be float32");

    TORCH_CHECK(beta.is_cuda(), "Beta must be a CUDA tensor");
    TORCH_CHECK(beta.is_contiguous(), "Beta must be contiguous");
    TORCH_CHECK(beta.dtype() == torch::kFloat32, "Beta must be float32");

    auto output = torch::empty_like(x);

    int batch_size = x.size(0);
    int features = x.size(1);
    int spatial_dim = 1;
    for (int i = 2; i < x.dim(); i++) {
        spatial_dim *= x.size(i);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    group_norm_launcher(x.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), 
                        output.data_ptr<float>(), batch_size, features, num_groups, spatial_dim, stream);

    return output;
}

void register_group_norm(pybind11::module& m) {
    m.def("group_norm_forward", &group_norm_forward, "GroupNorm forward",
          py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("num_groups"));
}


"""

cuda_extension = load_inline(
    name="cuda_extension",
    cpp_sources=[_cpp_sources],
    cuda_sources=[_cuda_sources],
    functions=["group_norm_forward"],
    verbose=False,
    extra_cuda_cflags=["-O2"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization using a custom CUDA kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        # Initialize parameters to match the original model's state_dict
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        self.num_groups = num_groups
        # We'll use the parameters from self.gn (weight and bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        gamma = self.gn.weight
        beta = self.gn.bias
        return cuda_extension.group_norm_forward(x, gamma, beta, self.num_groups)