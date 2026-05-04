# 增强测试后被杀死的变异体

**总计**: 50 个变异体在 4 层增强压力测试中被杀死

## 增强测试配置

| 层 | 内容 | 策略 |
|-----|------|------|
| Layer 1 | 值分布压力 | 14 种策略 × 3 种子 = 42 次测试 |
| Layer 2 | dtype 切换 | float16 + bfloat16 × 3 种子 |
| Layer 3 | 重复执行 | 10 次重复 × 3 种子，检测不一致 |
| Layer 4 | 训练模式 | .train() 模式运行 |

---

## 1. `L1_P100__mask_boundary__0`

- **算子**: `mask_boundary` (Category B)
- **描述**: Weaken or tighten boundary checks in mask/tl.where (Triton) or thread/block guards (CUDA) @ L18
- **变异行**: Line 18, 原始片段 `idx < size`, 节点类型 `rhs-1`
- **杀死层**: `layer1_value`
- **杀死策略**: `tail_heavy`
- **杀死种子**: `82`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused hinge loss computation
hinge_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hinge_loss_kernel(const float* predictions, const float* targets, float* loss, int size) {
    __shared__ float shared_mem[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    float thread_loss = 0.0f;
    
    if (idx < size) {
        float margin = 1.0f - predictions[idx] * targets[idx];
        thread_loss = margin > 0.0f ? margin : 0.0f;
    }
    
    shared_mem[tid] = thread_loss;
    __syncthreads();
    
    // Parallel reduction for sum within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    // First thread writes block sum to global memory
    if (tid == 0) {
        loss[blockIdx.x] = shared_mem[0];
    }
}

torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto size = predictions.numel();
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    // Allocate device memory for block sums
    auto block_sums = torch::zeros({num_blocks}, torch::TensorOptions().dtype(torch::kFloat32).device(predictions.device()));
    
    // Launch kernel
    hinge_loss_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_sums.data_ptr<float>(),
        size
    );
    
    // Final reduction on CPU (small number of elements)
    auto final_loss = block_sums.sum() / size;
    
    return final_loss;
}
"""

hinge_loss_cpp_source = "torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"

# Compile the inline CUDA code for hinge loss
hinge_loss_module = load_inline(
    name="hinge_loss",
    cpp_sources=hinge_loss_cpp_source,
    cuda_sources=hinge_loss_source,
    functions=["hinge_loss_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that computes Hinge Loss using custom CUDA kernel.
    Fuses clamp, multiplication, subtraction, and mean operations.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hinge_loss_kernel = hinge_loss_module

    def forward(self, predictions, targets):
        # Ensure tensors are contiguous and have same shape
        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        return self.hinge_loss_kernel.hinge_loss_cuda(predictions, targets)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused hinge loss computation
hinge_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hinge_loss_kernel(const float* predictions, const float* targets, float* loss, int size) {
    __shared__ float shared_mem[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    float thread_loss = 0.0f;
    
    if (idx < size - 1) {
        float margin = 1.0f - predictions[idx] * targets[idx];
        thread_loss = margin > 0.0f ? margin : 0.0f;
    }
    
    shared_mem[tid] = thread_loss;
    __syncthreads();
    
    // Parallel reduction for sum within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    // First thread writes block sum to global memory
    if (tid == 0) {
        loss[blockIdx.x] = shared_mem[0];
    }
}

torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto size = predictions.numel();
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    // Allocate device memory for block sums
    auto block_sums = torch::zeros({num_blocks}, torch::TensorOptions().dtype(torch::kFloat32).device(predictions.device()));
    
    // Launch kernel
    hinge_loss_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_sums.data_ptr<float>(),
        size
    );
    
    // Final reduction on CPU (small number of elements)
    auto final_loss = block_sums.sum() / size;
    
    return final_loss;
}
"""

hinge_loss_cpp_source = "torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"

# Compile the inline CUDA code for hinge loss
hinge_loss_module = load_inline(
    name="hinge_loss",
    cpp_sources=hinge_loss_cpp_source,
    cuda_sources=hinge_loss_source,
    functions=["hinge_loss_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that computes Hinge Loss using custom CUDA kernel.
    Fuses clamp, multiplication, subtraction, and mean operations.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hinge_loss_kernel = hinge_loss_module

    def forward(self, predictions, targets):
        # Ensure tensors are contiguous and have same shape
        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        return self.hinge_loss_kernel.hinge_loss_cuda(predictions, targets)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=tail_heavy, 种子=82, 模式=value_stress

**杀死策略描述**: last 10% large, rest small

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 128
input_shape = (1,)

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randint(0, 2, (batch_size, 1)).float() * 2 - 1]
```

---

## 2. `L1_P23__arith_replace__4`

- **算子**: `arith_replace` (Category A)
- **描述**: Replace binary arithmetic operators (+→-, -→+, *→/, /→*) @ L25
- **变异行**: Line 25, 原始片段 `*`, 节点类型 `cuda_Mult`
- **杀死层**: `layer1_value`
- **杀死策略**: `large_magnitude`
- **杀死种子**: `42`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for numerically stable softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template<typename T>
__global__ void softmax_kernel(
    const T* input,
    T* output,
    const int batch_size,
    const int num_features
) {
    // Each block processes one batch element
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    
    if (batch_idx >= batch_size) return;
    
    const T* batch_input = input + batch_idx * num_features;
    T* batch_output = output + batch_idx * num_features;
    
    // Shared memory for max and sum reductions
    extern __shared__ float shared_mem[];
    float* shared_max = shared_mem;
    float* shared_sum = shared_mem + blockDim.x;
    
    // First pass: find max value in the row
    float thread_max = -INFINITY;
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_input[i]);
        thread_max = fmaxf(thread_max, val);
    }
    
    // Parallel reduction for max
    shared_max[tid] = thread_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    
    float row_max = shared_max[0];
    
    // Second pass: compute exponentials and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_input[i]);
        float exp_val = expf(val - row_max);
        batch_output[i] = static_cast<T>(exp_val);
        thread_sum += exp_val;
    }
    
    // Parallel reduction for sum
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    float row_sum = shared_sum[0];
    
    // Third pass: normalize by sum
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_output[i]);
        batch_output[i] = static_cast<T>(val / row_sum);
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto sizes = input.sizes();
    auto batch_size = sizes[0];
    auto num_features = sizes[1];
    
    auto output = torch::empty_like(input);
    
    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int blocks_per_grid = batch_size;
    
    // Shared memory size: 2 * threads_per_block * sizeof(float)
    size_t shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    // Launch kernel based on data type
    if (input.scalar_type() == torch::kFloat32) {
        softmax_kernel<float><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            num_features
        );
    } else if (input.scalar_type() == torch::kFloat16) {
        softmax_kernel<at::Half><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            batch_size,
            num_features
        );
    } else if (input.scalar_type() == torch::kBFloat16) {
        softmax_kernel<at::BFloat16><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<at::BFloat16>(),
            output.data_ptr<at::BFloat16>(),
            batch_size,
            num_features
        );
    }
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    // Synchronize to ensure kernel completes
    cudaDeviceSynchronize();
    
    return output;
}
"""

softmax_cpp_source = """
torch::Tensor softmax_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
softmax_module = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Softmax activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_cuda = softmax_module.softmax_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor using custom CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).
            
        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        # Use custom kernel for CUDA tensors with 2 dimensions
        if x.is_cuda and x.dim() == 2:
            return self.softmax_cuda(x)
        else:
            # Fallback to PyTorch's implementation for other cases
            return torch.softmax(x, dim=1)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for numerically stable softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template<typename T>
__global__ void softmax_kernel(
    const T* input,
    T* output,
    const int batch_size,
    const int num_features
) {
    // Each block processes one batch element
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    
    if (batch_idx >= batch_size) return;
    
    const T* batch_input = input + batch_idx / num_features;
    T* batch_output = output + batch_idx * num_features;
    
    // Shared memory for max and sum reductions
    extern __shared__ float shared_mem[];
    float* shared_max = shared_mem;
    float* shared_sum = shared_mem + blockDim.x;
    
    // First pass: find max value in the row
    float thread_max = -INFINITY;
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_input[i]);
        thread_max = fmaxf(thread_max, val);
    }
    
    // Parallel reduction for max
    shared_max[tid] = thread_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    
    float row_max = shared_max[0];
    
    // Second pass: compute exponentials and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_input[i]);
        float exp_val = expf(val - row_max);
        batch_output[i] = static_cast<T>(exp_val);
        thread_sum += exp_val;
    }
    
    // Parallel reduction for sum
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    float row_sum = shared_sum[0];
    
    // Third pass: normalize by sum
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_output[i]);
        batch_output[i] = static_cast<T>(val / row_sum);
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto sizes = input.sizes();
    auto batch_size = sizes[0];
    auto num_features = sizes[1];
    
    auto output = torch::empty_like(input);
    
    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int blocks_per_grid = batch_size;
    
    // Shared memory size: 2 * threads_per_block * sizeof(float)
    size_t shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    // Launch kernel based on data type
    if (input.scalar_type() == torch::kFloat32) {
        softmax_kernel<float><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            num_features
        );
    } else if (input.scalar_type() == torch::kFloat16) {
        softmax_kernel<at::Half><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            batch_size,
            num_features
        );
    } else if (input.scalar_type() == torch::kBFloat16) {
        softmax_kernel<at::BFloat16><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<at::BFloat16>(),
            output.data_ptr<at::BFloat16>(),
            batch_size,
            num_features
        );
    }
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    // Synchronize to ensure kernel completes
    cudaDeviceSynchronize();
    
    return output;
}
"""

softmax_cpp_source = """
torch::Tensor softmax_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
softmax_module = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Softmax activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_cuda = softmax_module.softmax_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor using custom CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).
            
        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        # Use custom kernel for CUDA tensors with 2 dimensions
        if x.is_cuda and x.dim() == 2:
            return self.softmax_cuda(x)
        else:
            # Fallback to PyTorch's implementation for other cases
            return torch.softmax(x, dim=1)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=large_magnitude, 种子=42, 模式=value_stress

**杀死策略描述**: values ~1e6

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]
```

---

## 3. `L1_P23__index_replace__5`

- **算子**: `index_replace` (Category B)
- **描述**: Swap Triton program_id axis or CUDA thread/block dimension index (e.g. program_id(0)→(1), threadIdx.x→threadIdx.y) @ L21
- **变异行**: Line 21, 原始片段 `blockDim.x`, 节点类型 `cuda_dim|z`
- **杀死层**: `layer1_value`
- **杀死策略**: `large_magnitude`
- **杀死种子**: `45`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for numerically stable softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template<typename T>
__global__ void softmax_kernel(
    const T* input,
    T* output,
    const int batch_size,
    const int num_features
) {
    // Each block processes one batch element
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    
    if (batch_idx >= batch_size) return;
    
    const T* batch_input = input + batch_idx * num_features;
    T* batch_output = output + batch_idx * num_features;
    
    // Shared memory for max and sum reductions
    extern __shared__ float shared_mem[];
    float* shared_max = shared_mem;
    float* shared_sum = shared_mem + blockDim.x;
    
    // First pass: find max value in the row
    float thread_max = -INFINITY;
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_input[i]);
        thread_max = fmaxf(thread_max, val);
    }
    
    // Parallel reduction for max
    shared_max[tid] = thread_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    
    float row_max = shared_max[0];
    
    // Second pass: compute exponentials and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_input[i]);
        float exp_val = expf(val - row_max);
        batch_output[i] = static_cast<T>(exp_val);
        thread_sum += exp_val;
    }
    
    // Parallel reduction for sum
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    float row_sum = shared_sum[0];
    
    // Third pass: normalize by sum
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_output[i]);
        batch_output[i] = static_cast<T>(val / row_sum);
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto sizes = input.sizes();
    auto batch_size = sizes[0];
    auto num_features = sizes[1];
    
    auto output = torch::empty_like(input);
    
    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int blocks_per_grid = batch_size;
    
    // Shared memory size: 2 * threads_per_block * sizeof(float)
    size_t shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    // Launch kernel based on data type
    if (input.scalar_type() == torch::kFloat32) {
        softmax_kernel<float><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            num_features
        );
    } else if (input.scalar_type() == torch::kFloat16) {
        softmax_kernel<at::Half><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            batch_size,
            num_features
        );
    } else if (input.scalar_type() == torch::kBFloat16) {
        softmax_kernel<at::BFloat16><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<at::BFloat16>(),
            output.data_ptr<at::BFloat16>(),
            batch_size,
            num_features
        );
    }
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    // Synchronize to ensure kernel completes
    cudaDeviceSynchronize();
    
    return output;
}
"""

softmax_cpp_source = """
torch::Tensor softmax_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
softmax_module = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Softmax activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_cuda = softmax_module.softmax_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor using custom CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).
            
        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        # Use custom kernel for CUDA tensors with 2 dimensions
        if x.is_cuda and x.dim() == 2:
            return self.softmax_cuda(x)
        else:
            # Fallback to PyTorch's implementation for other cases
            return torch.softmax(x, dim=1)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for numerically stable softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template<typename T>
__global__ void softmax_kernel(
    const T* input,
    T* output,
    const int batch_size,
    const int num_features
) {
    // Each block processes one batch element
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.z;
    
    if (batch_idx >= batch_size) return;
    
    const T* batch_input = input + batch_idx * num_features;
    T* batch_output = output + batch_idx * num_features;
    
    // Shared memory for max and sum reductions
    extern __shared__ float shared_mem[];
    float* shared_max = shared_mem;
    float* shared_sum = shared_mem + blockDim.x;
    
    // First pass: find max value in the row
    float thread_max = -INFINITY;
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_input[i]);
        thread_max = fmaxf(thread_max, val);
    }
    
    // Parallel reduction for max
    shared_max[tid] = thread_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    
    float row_max = shared_max[0];
    
    // Second pass: compute exponentials and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_input[i]);
        float exp_val = expf(val - row_max);
        batch_output[i] = static_cast<T>(exp_val);
        thread_sum += exp_val;
    }
    
    // Parallel reduction for sum
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    float row_sum = shared_sum[0];
    
    // Third pass: normalize by sum
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_output[i]);
        batch_output[i] = static_cast<T>(val / row_sum);
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto sizes = input.sizes();
    auto batch_size = sizes[0];
    auto num_features = sizes[1];
    
    auto output = torch::empty_like(input);
    
    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int blocks_per_grid = batch_size;
    
    // Shared memory size: 2 * threads_per_block * sizeof(float)
    size_t shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    // Launch kernel based on data type
    if (input.scalar_type() == torch::kFloat32) {
        softmax_kernel<float><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            num_features
        );
    } else if (input.scalar_type() == torch::kFloat16) {
        softmax_kernel<at::Half><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            batch_size,
            num_features
        );
    } else if (input.scalar_type() == torch::kBFloat16) {
        softmax_kernel<at::BFloat16><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<at::BFloat16>(),
            output.data_ptr<at::BFloat16>(),
            batch_size,
            num_features
        );
    }
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    // Synchronize to ensure kernel completes
    cudaDeviceSynchronize();
    
    return output;
}
"""

softmax_cpp_source = """
torch::Tensor softmax_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
softmax_module = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Softmax activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_cuda = softmax_module.softmax_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor using custom CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).
            
        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        # Use custom kernel for CUDA tensors with 2 dimensions
        if x.is_cuda and x.dim() == 2:
            return self.softmax_cuda(x)
        else:
            # Fallback to PyTorch's implementation for other cases
            return torch.softmax(x, dim=1)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=large_magnitude, 种子=45, 模式=value_stress

**杀死策略描述**: values ~1e6

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]
```

---

## 4. `L1_P23__index_replace__8`

- **算子**: `index_replace` (Category B)
- **描述**: Swap Triton program_id axis or CUDA thread/block dimension index (e.g. program_id(0)→(1), threadIdx.x→threadIdx.y) @ L44
- **变异行**: Line 44, 原始片段 `blockDim.x`, 节点类型 `cuda_dim|y`
- **杀死层**: `layer1_value`
- **杀死策略**: `large_magnitude`
- **杀死种子**: `45`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for numerically stable softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template<typename T>
__global__ void softmax_kernel(
    const T* input,
    T* output,
    const int batch_size,
    const int num_features
) {
    // Each block processes one batch element
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    
    if (batch_idx >= batch_size) return;
    
    const T* batch_input = input + batch_idx * num_features;
    T* batch_output = output + batch_idx * num_features;
    
    // Shared memory for max and sum reductions
    extern __shared__ float shared_mem[];
    float* shared_max = shared_mem;
    float* shared_sum = shared_mem + blockDim.x;
    
    // First pass: find max value in the row
    float thread_max = -INFINITY;
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_input[i]);
        thread_max = fmaxf(thread_max, val);
    }
    
    // Parallel reduction for max
    shared_max[tid] = thread_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    
    float row_max = shared_max[0];
    
    // Second pass: compute exponentials and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_input[i]);
        float exp_val = expf(val - row_max);
        batch_output[i] = static_cast<T>(exp_val);
        thread_sum += exp_val;
    }
    
    // Parallel reduction for sum
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    float row_sum = shared_sum[0];
    
    // Third pass: normalize by sum
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_output[i]);
        batch_output[i] = static_cast<T>(val / row_sum);
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto sizes = input.sizes();
    auto batch_size = sizes[0];
    auto num_features = sizes[1];
    
    auto output = torch::empty_like(input);
    
    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int blocks_per_grid = batch_size;
    
    // Shared memory size: 2 * threads_per_block * sizeof(float)
    size_t shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    // Launch kernel based on data type
    if (input.scalar_type() == torch::kFloat32) {
        softmax_kernel<float><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            num_features
        );
    } else if (input.scalar_type() == torch::kFloat16) {
        softmax_kernel<at::Half><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            batch_size,
            num_features
        );
    } else if (input.scalar_type() == torch::kBFloat16) {
        softmax_kernel<at::BFloat16><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<at::BFloat16>(),
            output.data_ptr<at::BFloat16>(),
            batch_size,
            num_features
        );
    }
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    // Synchronize to ensure kernel completes
    cudaDeviceSynchronize();
    
    return output;
}
"""

softmax_cpp_source = """
torch::Tensor softmax_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
softmax_module = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Softmax activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_cuda = softmax_module.softmax_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor using custom CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).
            
        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        # Use custom kernel for CUDA tensors with 2 dimensions
        if x.is_cuda and x.dim() == 2:
            return self.softmax_cuda(x)
        else:
            # Fallback to PyTorch's implementation for other cases
            return torch.softmax(x, dim=1)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for numerically stable softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template<typename T>
__global__ void softmax_kernel(
    const T* input,
    T* output,
    const int batch_size,
    const int num_features
) {
    // Each block processes one batch element
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    
    if (batch_idx >= batch_size) return;
    
    const T* batch_input = input + batch_idx * num_features;
    T* batch_output = output + batch_idx * num_features;
    
    // Shared memory for max and sum reductions
    extern __shared__ float shared_mem[];
    float* shared_max = shared_mem;
    float* shared_sum = shared_mem + blockDim.x;
    
    // First pass: find max value in the row
    float thread_max = -INFINITY;
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_input[i]);
        thread_max = fmaxf(thread_max, val);
    }
    
    // Parallel reduction for max
    shared_max[tid] = thread_max;
    __syncthreads();
    
    for (int s = blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    
    float row_max = shared_max[0];
    
    // Second pass: compute exponentials and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_input[i]);
        float exp_val = expf(val - row_max);
        batch_output[i] = static_cast<T>(exp_val);
        thread_sum += exp_val;
    }
    
    // Parallel reduction for sum
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    float row_sum = shared_sum[0];
    
    // Third pass: normalize by sum
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_output[i]);
        batch_output[i] = static_cast<T>(val / row_sum);
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto sizes = input.sizes();
    auto batch_size = sizes[0];
    auto num_features = sizes[1];
    
    auto output = torch::empty_like(input);
    
    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int blocks_per_grid = batch_size;
    
    // Shared memory size: 2 * threads_per_block * sizeof(float)
    size_t shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    // Launch kernel based on data type
    if (input.scalar_type() == torch::kFloat32) {
        softmax_kernel<float><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            num_features
        );
    } else if (input.scalar_type() == torch::kFloat16) {
        softmax_kernel<at::Half><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            batch_size,
            num_features
        );
    } else if (input.scalar_type() == torch::kBFloat16) {
        softmax_kernel<at::BFloat16><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<at::BFloat16>(),
            output.data_ptr<at::BFloat16>(),
            batch_size,
            num_features
        );
    }
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    // Synchronize to ensure kernel completes
    cudaDeviceSynchronize();
    
    return output;
}
"""

softmax_cpp_source = """
torch::Tensor softmax_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
softmax_module = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Softmax activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_cuda = softmax_module.softmax_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor using custom CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).
            
        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        # Use custom kernel for CUDA tensors with 2 dimensions
        if x.is_cuda and x.dim() == 2:
            return self.softmax_cuda(x)
        else:
            # Fallback to PyTorch's implementation for other cases
            return torch.softmax(x, dim=1)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=large_magnitude, 种子=45, 模式=value_stress

**杀死策略描述**: values ~1e6

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]
```

---

## 5. `L1_P23__mask_boundary__2`

- **算子**: `mask_boundary` (Category B)
- **描述**: Weaken or tighten boundary checks in mask/tl.where (Triton) or thread/block guards (CUDA) @ L45
- **变异行**: Line 45, 原始片段 `tid < s`, 节点类型 `rhs-1`
- **杀死层**: `layer1_value`
- **杀死策略**: `boundary_last_element`
- **杀死种子**: `42`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for numerically stable softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template<typename T>
__global__ void softmax_kernel(
    const T* input,
    T* output,
    const int batch_size,
    const int num_features
) {
    // Each block processes one batch element
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    
    if (batch_idx >= batch_size) return;
    
    const T* batch_input = input + batch_idx * num_features;
    T* batch_output = output + batch_idx * num_features;
    
    // Shared memory for max and sum reductions
    extern __shared__ float shared_mem[];
    float* shared_max = shared_mem;
    float* shared_sum = shared_mem + blockDim.x;
    
    // First pass: find max value in the row
    float thread_max = -INFINITY;
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_input[i]);
        thread_max = fmaxf(thread_max, val);
    }
    
    // Parallel reduction for max
    shared_max[tid] = thread_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    
    float row_max = shared_max[0];
    
    // Second pass: compute exponentials and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_input[i]);
        float exp_val = expf(val - row_max);
        batch_output[i] = static_cast<T>(exp_val);
        thread_sum += exp_val;
    }
    
    // Parallel reduction for sum
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    float row_sum = shared_sum[0];
    
    // Third pass: normalize by sum
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_output[i]);
        batch_output[i] = static_cast<T>(val / row_sum);
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto sizes = input.sizes();
    auto batch_size = sizes[0];
    auto num_features = sizes[1];
    
    auto output = torch::empty_like(input);
    
    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int blocks_per_grid = batch_size;
    
    // Shared memory size: 2 * threads_per_block * sizeof(float)
    size_t shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    // Launch kernel based on data type
    if (input.scalar_type() == torch::kFloat32) {
        softmax_kernel<float><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            num_features
        );
    } else if (input.scalar_type() == torch::kFloat16) {
        softmax_kernel<at::Half><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            batch_size,
            num_features
        );
    } else if (input.scalar_type() == torch::kBFloat16) {
        softmax_kernel<at::BFloat16><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<at::BFloat16>(),
            output.data_ptr<at::BFloat16>(),
            batch_size,
            num_features
        );
    }
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    // Synchronize to ensure kernel completes
    cudaDeviceSynchronize();
    
    return output;
}
"""

softmax_cpp_source = """
torch::Tensor softmax_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
softmax_module = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Softmax activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_cuda = softmax_module.softmax_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor using custom CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).
            
        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        # Use custom kernel for CUDA tensors with 2 dimensions
        if x.is_cuda and x.dim() == 2:
            return self.softmax_cuda(x)
        else:
            # Fallback to PyTorch's implementation for other cases
            return torch.softmax(x, dim=1)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for numerically stable softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template<typename T>
__global__ void softmax_kernel(
    const T* input,
    T* output,
    const int batch_size,
    const int num_features
) {
    // Each block processes one batch element
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    
    if (batch_idx >= batch_size) return;
    
    const T* batch_input = input + batch_idx * num_features;
    T* batch_output = output + batch_idx * num_features;
    
    // Shared memory for max and sum reductions
    extern __shared__ float shared_mem[];
    float* shared_max = shared_mem;
    float* shared_sum = shared_mem + blockDim.x;
    
    // First pass: find max value in the row
    float thread_max = -INFINITY;
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_input[i]);
        thread_max = fmaxf(thread_max, val);
    }
    
    // Parallel reduction for max
    shared_max[tid] = thread_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s - 1) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    
    float row_max = shared_max[0];
    
    // Second pass: compute exponentials and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_input[i]);
        float exp_val = expf(val - row_max);
        batch_output[i] = static_cast<T>(exp_val);
        thread_sum += exp_val;
    }
    
    // Parallel reduction for sum
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    float row_sum = shared_sum[0];
    
    // Third pass: normalize by sum
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_output[i]);
        batch_output[i] = static_cast<T>(val / row_sum);
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto sizes = input.sizes();
    auto batch_size = sizes[0];
    auto num_features = sizes[1];
    
    auto output = torch::empty_like(input);
    
    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int blocks_per_grid = batch_size;
    
    // Shared memory size: 2 * threads_per_block * sizeof(float)
    size_t shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    // Launch kernel based on data type
    if (input.scalar_type() == torch::kFloat32) {
        softmax_kernel<float><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            num_features
        );
    } else if (input.scalar_type() == torch::kFloat16) {
        softmax_kernel<at::Half><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            batch_size,
            num_features
        );
    } else if (input.scalar_type() == torch::kBFloat16) {
        softmax_kernel<at::BFloat16><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<at::BFloat16>(),
            output.data_ptr<at::BFloat16>(),
            batch_size,
            num_features
        );
    }
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    // Synchronize to ensure kernel completes
    cudaDeviceSynchronize();
    
    return output;
}
"""

softmax_cpp_source = """
torch::Tensor softmax_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
softmax_module = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Softmax activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_cuda = softmax_module.softmax_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor using custom CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).
            
        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        # Use custom kernel for CUDA tensors with 2 dimensions
        if x.is_cuda and x.dim() == 2:
            return self.softmax_cuda(x)
        else:
            # Fallback to PyTorch's implementation for other cases
            return torch.softmax(x, dim=1)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=boundary_last_element, 种子=42, 模式=value_stress

**杀死策略描述**: last element set to extreme

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]
```

---

## 6. `L1_P23__stab_remove__0`

- **算子**: `stab_remove` (Category C)
- **描述**: Remove max-subtraction numerical stabilization (overflow on large inputs) @ L57
- **变异行**: Line 57, 原始片段 `expf(val - row_max)`, 节点类型 `stab:cuda_expf_minus`
- **杀死层**: `layer1_value`
- **杀死策略**: `large_magnitude`
- **杀死种子**: `42`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for numerically stable softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template<typename T>
__global__ void softmax_kernel(
    const T* input,
    T* output,
    const int batch_size,
    const int num_features
) {
    // Each block processes one batch element
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    
    if (batch_idx >= batch_size) return;
    
    const T* batch_input = input + batch_idx * num_features;
    T* batch_output = output + batch_idx * num_features;
    
    // Shared memory for max and sum reductions
    extern __shared__ float shared_mem[];
    float* shared_max = shared_mem;
    float* shared_sum = shared_mem + blockDim.x;
    
    // First pass: find max value in the row
    float thread_max = -INFINITY;
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_input[i]);
        thread_max = fmaxf(thread_max, val);
    }
    
    // Parallel reduction for max
    shared_max[tid] = thread_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    
    float row_max = shared_max[0];
    
    // Second pass: compute exponentials and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_input[i]);
        float exp_val = expf(val - row_max);
        batch_output[i] = static_cast<T>(exp_val);
        thread_sum += exp_val;
    }
    
    // Parallel reduction for sum
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    float row_sum = shared_sum[0];
    
    // Third pass: normalize by sum
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_output[i]);
        batch_output[i] = static_cast<T>(val / row_sum);
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto sizes = input.sizes();
    auto batch_size = sizes[0];
    auto num_features = sizes[1];
    
    auto output = torch::empty_like(input);
    
    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int blocks_per_grid = batch_size;
    
    // Shared memory size: 2 * threads_per_block * sizeof(float)
    size_t shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    // Launch kernel based on data type
    if (input.scalar_type() == torch::kFloat32) {
        softmax_kernel<float><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            num_features
        );
    } else if (input.scalar_type() == torch::kFloat16) {
        softmax_kernel<at::Half><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            batch_size,
            num_features
        );
    } else if (input.scalar_type() == torch::kBFloat16) {
        softmax_kernel<at::BFloat16><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<at::BFloat16>(),
            output.data_ptr<at::BFloat16>(),
            batch_size,
            num_features
        );
    }
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    // Synchronize to ensure kernel completes
    cudaDeviceSynchronize();
    
    return output;
}
"""

softmax_cpp_source = """
torch::Tensor softmax_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
softmax_module = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Softmax activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_cuda = softmax_module.softmax_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor using custom CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).
            
        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        # Use custom kernel for CUDA tensors with 2 dimensions
        if x.is_cuda and x.dim() == 2:
            return self.softmax_cuda(x)
        else:
            # Fallback to PyTorch's implementation for other cases
            return torch.softmax(x, dim=1)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for numerically stable softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template<typename T>
__global__ void softmax_kernel(
    const T* input,
    T* output,
    const int batch_size,
    const int num_features
) {
    // Each block processes one batch element
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    
    if (batch_idx >= batch_size) return;
    
    const T* batch_input = input + batch_idx * num_features;
    T* batch_output = output + batch_idx * num_features;
    
    // Shared memory for max and sum reductions
    extern __shared__ float shared_mem[];
    float* shared_max = shared_mem;
    float* shared_sum = shared_mem + blockDim.x;
    
    // First pass: find max value in the row
    float thread_max = -INFINITY;
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_input[i]);
        thread_max = fmaxf(thread_max, val);
    }
    
    // Parallel reduction for max
    shared_max[tid] = thread_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    
    float row_max = shared_max[0];
    
    // Second pass: compute exponentials and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_input[i]);
        float exp_val = expf(val);
        batch_output[i] = static_cast<T>(exp_val);
        thread_sum += exp_val;
    }
    
    // Parallel reduction for sum
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    float row_sum = shared_sum[0];
    
    // Third pass: normalize by sum
    for (int i = tid; i < num_features; i += stride) {
        float val = static_cast<float>(batch_output[i]);
        batch_output[i] = static_cast<T>(val / row_sum);
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto sizes = input.sizes();
    auto batch_size = sizes[0];
    auto num_features = sizes[1];
    
    auto output = torch::empty_like(input);
    
    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int blocks_per_grid = batch_size;
    
    // Shared memory size: 2 * threads_per_block * sizeof(float)
    size_t shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    // Launch kernel based on data type
    if (input.scalar_type() == torch::kFloat32) {
        softmax_kernel<float><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            num_features
        );
    } else if (input.scalar_type() == torch::kFloat16) {
        softmax_kernel<at::Half><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            batch_size,
            num_features
        );
    } else if (input.scalar_type() == torch::kBFloat16) {
        softmax_kernel<at::BFloat16><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            input.data_ptr<at::BFloat16>(),
            output.data_ptr<at::BFloat16>(),
            batch_size,
            num_features
        );
    }
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    // Synchronize to ensure kernel completes
    cudaDeviceSynchronize();
    
    return output;
}
"""

softmax_cpp_source = """
torch::Tensor softmax_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
softmax_module = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Softmax activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_cuda = softmax_module.softmax_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor using custom CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).
            
        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        # Use custom kernel for CUDA tensors with 2 dimensions
        if x.is_cuda and x.dim() == 2:
            return self.softmax_cuda(x)
        else:
            # Fallback to PyTorch's implementation for other cases
            return torch.softmax(x, dim=1)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=large_magnitude, 种子=42, 模式=value_stress

**杀死策略描述**: values ~1e6

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]
```

---

## 7. `L1_P24__mask_boundary__0`

- **算子**: `mask_boundary` (Category B)
- **描述**: Weaken or tighten boundary checks in mask/tl.where (Triton) or thread/block guards (CUDA) @ L61
- **变异行**: Line 61, 原始片段 `tid < s`, 节点类型 `rhs-1`
- **杀死层**: `layer1_value`
- **杀死策略**: `boundary_last_element`
- **杀死种子**: `42`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized LogSoftmax
log_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<__half>(__half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ __half from_float<__half>(float val) {
    return __float2half(val);
}

template<typename T>
__global__ void log_softmax_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int batch_size,
    const int dim_size
) {
    // Each block handles one batch element
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    const T* batch_input = input + batch_idx * dim_size;
    T* batch_output = output + batch_idx * dim_size;
    
    // Shared memory for reductions
    __shared__ float sdata[1024];
    
    // Step 1: Find max using block reduction
    float thread_max = -INFINITY;
    for (int i = tid; i < dim_size; i += blockDim.x) {
        float val = to_float<T>(batch_input[i]);
        thread_max = fmaxf(thread_max, val);
    }
    
    sdata[tid] = thread_max;
    __syncthreads();
    
    // Block reduction for max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    float global_max = sdata[0];
    __syncthreads();
    
    // Step 2: Compute sum(exp(x - max)) with block reduction
    float thread_sum = 0.0f;
    for (int i = tid; i < dim_size; i += blockDim.x) {
        float val = to_float<T>(batch_input[i]);
        thread_sum += expf(val - global_max);
    }
    
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Block reduction for sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float total_sum = sdata[0];
    float log_sum_exp = logf(total_sum) + global_max;
    
    // Step 3: Compute final log_softmax values
    for (int i = tid; i < dim_size; i += blockDim.x) {
        float val = to_float<T>(batch_input[i]);
        batch_output[i] = from_float<T>(val - log_sum_exp);
    }
}

torch::Tensor log_softmax_cuda(torch::Tensor input, int dim) {
    auto input_contiguous = input.contiguous();
    auto output = torch::empty_like(input_contiguous);
    
    // Only support 2D tensors with dim=1 for now
    TORCH_CHECK(input_contiguous.dim() == 2, "Input must be 2D tensor");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");
    
    int batch_size = input_contiguous.size(0);
    int dim_size = input_contiguous.size(1);
    
    // Optimize block size
    int BLOCK_SIZE = 256;
    if (dim_size <= 256) {
        BLOCK_SIZE = 128;
    } else if (dim_size <= 512) {
        BLOCK_SIZE = 256;
    } else {
        BLOCK_SIZE = 512;
    }
    
    // Ensure block size is power of 2 for reduction
    BLOCK_SIZE = min(BLOCK_SIZE, 1024);
    BLOCK_SIZE = 1 << (int)log2f(BLOCK_SIZE);
    
    dim3 grid(batch_size);
    dim3 block(BLOCK_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input_contiguous.scalar_type(), "log_softmax_cuda", ([&] {
            log_softmax_kernel<scalar_t><<<grid, block>>>(
                input_contiguous.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                dim_size
            );
        })
    );
    
    return output;
}
"""

log_softmax_cpp_source = """
torch::Tensor log_softmax_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code
log_softmax_module = load_inline(
    name="log_softmax",
    cpp_sources=log_softmax_cpp_source,
    cuda_sources=log_softmax_source,
    functions=["log_softmax_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs LogSoftmax activation using custom CUDA kernel.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.log_softmax_cuda = log_softmax_module.log_softmax_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation to the input tensor using custom CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied.
        """
        # Use custom kernel for 2D tensors with dim=1
        if x.dim() == 2 and self.dim == 1:
            return self.log_softmax_cuda(x, self.dim)
        else:
            # Fall back to PyTorch's implementation for other cases
            return torch.log_softmax(x, dim=self.dim)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized LogSoftmax
log_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<__half>(__half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ __half from_float<__half>(float val) {
    return __float2half(val);
}

template<typename T>
__global__ void log_softmax_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int batch_size,
    const int dim_size
) {
    // Each block handles one batch element
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    const T* batch_input = input + batch_idx * dim_size;
    T* batch_output = output + batch_idx * dim_size;
    
    // Shared memory for reductions
    __shared__ float sdata[1024];
    
    // Step 1: Find max using block reduction
    float thread_max = -INFINITY;
    for (int i = tid; i < dim_size; i += blockDim.x) {
        float val = to_float<T>(batch_input[i]);
        thread_max = fmaxf(thread_max, val);
    }
    
    sdata[tid] = thread_max;
    __syncthreads();
    
    // Block reduction for max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s - 1) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    float global_max = sdata[0];
    __syncthreads();
    
    // Step 2: Compute sum(exp(x - max)) with block reduction
    float thread_sum = 0.0f;
    for (int i = tid; i < dim_size; i += blockDim.x) {
        float val = to_float<T>(batch_input[i]);
        thread_sum += expf(val - global_max);
    }
    
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Block reduction for sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float total_sum = sdata[0];
    float log_sum_exp = logf(total_sum) + global_max;
    
    // Step 3: Compute final log_softmax values
    for (int i = tid; i < dim_size; i += blockDim.x) {
        float val = to_float<T>(batch_input[i]);
        batch_output[i] = from_float<T>(val - log_sum_exp);
    }
}

torch::Tensor log_softmax_cuda(torch::Tensor input, int dim) {
    auto input_contiguous = input.contiguous();
    auto output = torch::empty_like(input_contiguous);
    
    // Only support 2D tensors with dim=1 for now
    TORCH_CHECK(input_contiguous.dim() == 2, "Input must be 2D tensor");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");
    
    int batch_size = input_contiguous.size(0);
    int dim_size = input_contiguous.size(1);
    
    // Optimize block size
    int BLOCK_SIZE = 256;
    if (dim_size <= 256) {
        BLOCK_SIZE = 128;
    } else if (dim_size <= 512) {
        BLOCK_SIZE = 256;
    } else {
        BLOCK_SIZE = 512;
    }
    
    // Ensure block size is power of 2 for reduction
    BLOCK_SIZE = min(BLOCK_SIZE, 1024);
    BLOCK_SIZE = 1 << (int)log2f(BLOCK_SIZE);
    
    dim3 grid(batch_size);
    dim3 block(BLOCK_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input_contiguous.scalar_type(), "log_softmax_cuda", ([&] {
            log_softmax_kernel<scalar_t><<<grid, block>>>(
                input_contiguous.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                dim_size
            );
        })
    );
    
    return output;
}
"""

log_softmax_cpp_source = """
torch::Tensor log_softmax_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code
log_softmax_module = load_inline(
    name="log_softmax",
    cpp_sources=log_softmax_cpp_source,
    cuda_sources=log_softmax_source,
    functions=["log_softmax_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs LogSoftmax activation using custom CUDA kernel.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.log_softmax_cuda = log_softmax_module.log_softmax_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation to the input tensor using custom CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied.
        """
        # Use custom kernel for 2D tensors with dim=1
        if x.dim() == 2 and self.dim == 1:
            return self.log_softmax_cuda(x, self.dim)
        else:
            # Fall back to PyTorch's implementation for other cases
            return torch.log_softmax(x, dim=self.dim)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=boundary_last_element, 种子=42, 模式=value_stress

**杀死策略描述**: last element set to extreme

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]
```

---

## 8. `L1_P24__relop_replace__7`

- **算子**: `relop_replace` (Category A)
- **描述**: Replace relational operators (<→<=, <=→<, >→>=, >=→>, ==→!=, !=→==) @ L51
- **变异行**: Line 51, 原始片段 `<`, 节点类型 `cuda_Lt`
- **杀死层**: `layer1_value`
- **杀死策略**: `tail_heavy`
- **杀死种子**: `83`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized LogSoftmax
log_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<__half>(__half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ __half from_float<__half>(float val) {
    return __float2half(val);
}

template<typename T>
__global__ void log_softmax_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int batch_size,
    const int dim_size
) {
    // Each block handles one batch element
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    const T* batch_input = input + batch_idx * dim_size;
    T* batch_output = output + batch_idx * dim_size;
    
    // Shared memory for reductions
    __shared__ float sdata[1024];
    
    // Step 1: Find max using block reduction
    float thread_max = -INFINITY;
    for (int i = tid; i < dim_size; i += blockDim.x) {
        float val = to_float<T>(batch_input[i]);
        thread_max = fmaxf(thread_max, val);
    }
    
    sdata[tid] = thread_max;
    __syncthreads();
    
    // Block reduction for max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    float global_max = sdata[0];
    __syncthreads();
    
    // Step 2: Compute sum(exp(x - max)) with block reduction
    float thread_sum = 0.0f;
    for (int i = tid; i < dim_size; i += blockDim.x) {
        float val = to_float<T>(batch_input[i]);
        thread_sum += expf(val - global_max);
    }
    
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Block reduction for sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float total_sum = sdata[0];
    float log_sum_exp = logf(total_sum) + global_max;
    
    // Step 3: Compute final log_softmax values
    for (int i = tid; i < dim_size; i += blockDim.x) {
        float val = to_float<T>(batch_input[i]);
        batch_output[i] = from_float<T>(val - log_sum_exp);
    }
}

torch::Tensor log_softmax_cuda(torch::Tensor input, int dim) {
    auto input_contiguous = input.contiguous();
    auto output = torch::empty_like(input_contiguous);
    
    // Only support 2D tensors with dim=1 for now
    TORCH_CHECK(input_contiguous.dim() == 2, "Input must be 2D tensor");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");
    
    int batch_size = input_contiguous.size(0);
    int dim_size = input_contiguous.size(1);
    
    // Optimize block size
    int BLOCK_SIZE = 256;
    if (dim_size <= 256) {
        BLOCK_SIZE = 128;
    } else if (dim_size <= 512) {
        BLOCK_SIZE = 256;
    } else {
        BLOCK_SIZE = 512;
    }
    
    // Ensure block size is power of 2 for reduction
    BLOCK_SIZE = min(BLOCK_SIZE, 1024);
    BLOCK_SIZE = 1 << (int)log2f(BLOCK_SIZE);
    
    dim3 grid(batch_size);
    dim3 block(BLOCK_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input_contiguous.scalar_type(), "log_softmax_cuda", ([&] {
            log_softmax_kernel<scalar_t><<<grid, block>>>(
                input_contiguous.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                dim_size
            );
        })
    );
    
    return output;
}
"""

log_softmax_cpp_source = """
torch::Tensor log_softmax_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code
log_softmax_module = load_inline(
    name="log_softmax",
    cpp_sources=log_softmax_cpp_source,
    cuda_sources=log_softmax_source,
    functions=["log_softmax_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs LogSoftmax activation using custom CUDA kernel.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.log_softmax_cuda = log_softmax_module.log_softmax_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation to the input tensor using custom CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied.
        """
        # Use custom kernel for 2D tensors with dim=1
        if x.dim() == 2 and self.dim == 1:
            return self.log_softmax_cuda(x, self.dim)
        else:
            # Fall back to PyTorch's implementation for other cases
            return torch.log_softmax(x, dim=self.dim)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized LogSoftmax
log_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<__half>(__half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ __half from_float<__half>(float val) {
    return __float2half(val);
}

template<typename T>
__global__ void log_softmax_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int batch_size,
    const int dim_size
) {
    // Each block handles one batch element
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    const T* batch_input = input + batch_idx * dim_size;
    T* batch_output = output + batch_idx * dim_size;
    
    // Shared memory for reductions
    __shared__ float sdata[1024];
    
    // Step 1: Find max using block reduction
    float thread_max = -INFINITY;
    for (int i = tid; i <= dim_size; i += blockDim.x) {
        float val = to_float<T>(batch_input[i]);
        thread_max = fmaxf(thread_max, val);
    }
    
    sdata[tid] = thread_max;
    __syncthreads();
    
    // Block reduction for max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    float global_max = sdata[0];
    __syncthreads();
    
    // Step 2: Compute sum(exp(x - max)) with block reduction
    float thread_sum = 0.0f;
    for (int i = tid; i < dim_size; i += blockDim.x) {
        float val = to_float<T>(batch_input[i]);
        thread_sum += expf(val - global_max);
    }
    
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Block reduction for sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float total_sum = sdata[0];
    float log_sum_exp = logf(total_sum) + global_max;
    
    // Step 3: Compute final log_softmax values
    for (int i = tid; i < dim_size; i += blockDim.x) {
        float val = to_float<T>(batch_input[i]);
        batch_output[i] = from_float<T>(val - log_sum_exp);
    }
}

torch::Tensor log_softmax_cuda(torch::Tensor input, int dim) {
    auto input_contiguous = input.contiguous();
    auto output = torch::empty_like(input_contiguous);
    
    // Only support 2D tensors with dim=1 for now
    TORCH_CHECK(input_contiguous.dim() == 2, "Input must be 2D tensor");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");
    
    int batch_size = input_contiguous.size(0);
    int dim_size = input_contiguous.size(1);
    
    // Optimize block size
    int BLOCK_SIZE = 256;
    if (dim_size <= 256) {
        BLOCK_SIZE = 128;
    } else if (dim_size <= 512) {
        BLOCK_SIZE = 256;
    } else {
        BLOCK_SIZE = 512;
    }
    
    // Ensure block size is power of 2 for reduction
    BLOCK_SIZE = min(BLOCK_SIZE, 1024);
    BLOCK_SIZE = 1 << (int)log2f(BLOCK_SIZE);
    
    dim3 grid(batch_size);
    dim3 block(BLOCK_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input_contiguous.scalar_type(), "log_softmax_cuda", ([&] {
            log_softmax_kernel<scalar_t><<<grid, block>>>(
                input_contiguous.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                dim_size
            );
        })
    );
    
    return output;
}
"""

log_softmax_cpp_source = """
torch::Tensor log_softmax_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code
log_softmax_module = load_inline(
    name="log_softmax",
    cpp_sources=log_softmax_cpp_source,
    cuda_sources=log_softmax_source,
    functions=["log_softmax_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs LogSoftmax activation using custom CUDA kernel.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.log_softmax_cuda = log_softmax_module.log_softmax_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation to the input tensor using custom CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied.
        """
        # Use custom kernel for 2D tensors with dim=1
        if x.dim() == 2 and self.dim == 1:
            return self.log_softmax_cuda(x, self.dim)
        else:
            # Fall back to PyTorch's implementation for other cases
            return torch.log_softmax(x, dim=self.dim)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=tail_heavy, 种子=83, 模式=value_stress

**杀死策略描述**: last 10% large, rest small

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]
```

---

## 9. `L1_P27__arith_replace__6`

- **算子**: `arith_replace` (Category A)
- **描述**: Replace binary arithmetic operators (+→-, -→+, *→/, /→*) @ L26
- **变异行**: Line 26, 原始片段 `*`, 节点类型 `cuda_Mult`
- **杀死层**: `layer2_dtype`
- **杀死策略**: `None`
- **杀死种子**: `142`
- **杀死模式**: `dtype_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized SELU activation
selu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// Constants for SELU
constexpr float ALPHA = 1.6732632423543772848170429916717f;
constexpr float SCALE = 1.0507009873554804934193349852946f;
constexpr float LAMBDA = 1.0507009873554804934193349852946f;
constexpr float NEG_ALPHA_LAMBDA = -1.7580993408473768599402175208123f;

template<typename T>
__device__ __forceinline__ T selu_compute(T x) {
    return (x >= 0) ? (SCALE * x) : (LAMBDA * (ALPHA * expf(x) - ALPHA));
}

template<>
__device__ __forceinline__ half selu_compute(half x) {
    float fx = __half2float(x);
    float result = (fx >= 0) ? (SCALE * fx) : (LAMBDA * (ALPHA * expf(fx) - ALPHA));
    return __float2half(result);
}

template<typename T>
__global__ void selu_kernel_optimized(const T* __restrict__ input, T* __restrict__ output, int64_t size) {
    // Process 8 elements per thread for better occupancy
    const int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int64_t element_idx = idx + i;
        if (element_idx < size) {
            T x = input[element_idx];
            output[element_idx] = selu_compute(x);
        }
    }
}

// Specialized kernel for float32 with warp-level optimization
__global__ void selu_kernel_float32(const float* __restrict__ input, float* __restrict__ output, int64_t size) {
    const int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
    
    // Process 4 elements per thread with vectorized loads/stores
    if (idx * 4 + 3 < size) {
        // Load 4 floats at once
        float4 in = *reinterpret_cast<const float4*>(&input[idx * 4]);
        
        // Compute SELU
        float4 out;
        out.x = (in.x >= 0) ? (SCALE * in.x) : (LAMBDA * (ALPHA * expf(in.x) - ALPHA));
        out.y = (in.y >= 0) ? (SCALE * in.y) : (LAMBDA * (ALPHA * expf(in.y) - ALPHA));
        out.z = (in.z >= 0) ? (SCALE * in.z) : (LAMBDA * (ALPHA * expf(in.z) - ALPHA));
        out.w = (in.w >= 0) ? (SCALE * in.w) : (LAMBDA * (ALPHA * expf(in.w) - ALPHA));
        
        // Store 4 floats at once
        *reinterpret_cast<float4*>(&output[idx * 4]) = out;
    } else {
        // Handle remaining elements
        for (int64_t i = idx * 4; i < size; i++) {
            float x = input[i];
            output[i] = (x >= 0) ? (SCALE * x) : (LAMBDA * (ALPHA * expf(x) - ALPHA));
        }
    }
}

torch::Tensor selu_cuda_optimized(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int64_t size = input.numel();
    
    // Optimized launch configuration based on tensor size
    if (input.scalar_type() == torch::kFloat32) {
        // Use vectorized kernel for float32
        const int block_size = 256;
        int grid_size = (size / 4 + block_size - 1) / block_size;
        
        // Ensure sufficient occupancy
        if (grid_size < 32) grid_size = 32;
        
        selu_kernel_float32<<<grid_size, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size
        );
    } else {
        // Generic kernel for other types
        const int block_size = 256;
        int grid_size = (size / 8 + block_size - 1) / block_size;
        
        if (grid_size < 32) grid_size = 32;
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "selu_cuda_optimized", [&] {
            if (input.scalar_type() == torch::kFloat16) {
                selu_kernel_optimized<half><<<grid_size, block_size>>>(
                    reinterpret_cast<half*>(input.data_ptr<torch::Half>()),
                    reinterpret_cast<half*>(output.data_ptr<torch::Half>()),
                    size
                );
            } else if (input.scalar_type() == torch::kBFloat16) {
                // For BF16, convert to float, process, convert back
                auto input_float = input.to(torch::kFloat32);
                auto output_float = torch::empty_like(input_float);
                
                const int block_size_f = 256;
                int grid_size_f = (size / 4 + block_size_f - 1) / block_size_f;
                if (grid_size_f < 32) grid_size_f = 32;
                
                selu_kernel_float32<<<grid_size_f, block_size_f>>>(
                    input_float.data_ptr<float>(),
                    output_float.data_ptr<float>(),
                    size
                );
                
                output = output_float.to(torch::kBFloat16);
            }
        });
    }
    
    return output;
}
"""

selu_cpp_source = "torch::Tensor selu_cuda_optimized(torch::Tensor input);"

# Compile the inline CUDA code for optimized SELU
selu_extension = load_inline(
    name="selu_extension_optimized",
    cpp_sources=selu_cpp_source,
    cuda_sources=selu_source,
    functions=["selu_cuda_optimized"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas=-v"],
    extra_cflags=["-O3", "-march=native"],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs SELU activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.selu_cuda = selu_extension.selu_cuda_optimized
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized SELU activation to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with SELU applied, same shape as input.
        """
        # Only use custom kernel if tensor is on CUDA
        if x.is_cuda:
            return self.selu_cuda(x)
        else:
            # Fallback to PyTorch's implementation for CPU
            return torch.selu(x)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized SELU activation
selu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// Constants for SELU
constexpr float ALPHA = 1.6732632423543772848170429916717f;
constexpr float SCALE = 1.0507009873554804934193349852946f;
constexpr float LAMBDA = 1.0507009873554804934193349852946f;
constexpr float NEG_ALPHA_LAMBDA = -1.7580993408473768599402175208123f;

template<typename T>
__device__ __forceinline__ T selu_compute(T x) {
    return (x >= 0) ? (SCALE * x) : (LAMBDA * (ALPHA * expf(x) - ALPHA));
}

template<>
__device__ __forceinline__ half selu_compute(half x) {
    float fx = __half2float(x);
    float result = (fx >= 0) ? (SCALE * fx) : (LAMBDA * (ALPHA / expf(fx) - ALPHA));
    return __float2half(result);
}

template<typename T>
__global__ void selu_kernel_optimized(const T* __restrict__ input, T* __restrict__ output, int64_t size) {
    // Process 8 elements per thread for better occupancy
    const int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int64_t element_idx = idx + i;
        if (element_idx < size) {
            T x = input[element_idx];
            output[element_idx] = selu_compute(x);
        }
    }
}

// Specialized kernel for float32 with warp-level optimization
__global__ void selu_kernel_float32(const float* __restrict__ input, float* __restrict__ output, int64_t size) {
    const int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
    
    // Process 4 elements per thread with vectorized loads/stores
    if (idx * 4 + 3 < size) {
        // Load 4 floats at once
        float4 in = *reinterpret_cast<const float4*>(&input[idx * 4]);
        
        // Compute SELU
        float4 out;
        out.x = (in.x >= 0) ? (SCALE * in.x) : (LAMBDA * (ALPHA * expf(in.x) - ALPHA));
        out.y = (in.y >= 0) ? (SCALE * in.y) : (LAMBDA * (ALPHA * expf(in.y) - ALPHA));
        out.z = (in.z >= 0) ? (SCALE * in.z) : (LAMBDA * (ALPHA * expf(in.z) - ALPHA));
        out.w = (in.w >= 0) ? (SCALE * in.w) : (LAMBDA * (ALPHA * expf(in.w) - ALPHA));
        
        // Store 4 floats at once
        *reinterpret_cast<float4*>(&output[idx * 4]) = out;
    } else {
        // Handle remaining elements
        for (int64_t i = idx * 4; i < size; i++) {
            float x = input[i];
            output[i] = (x >= 0) ? (SCALE * x) : (LAMBDA * (ALPHA * expf(x) - ALPHA));
        }
    }
}

torch::Tensor selu_cuda_optimized(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int64_t size = input.numel();
    
    // Optimized launch configuration based on tensor size
    if (input.scalar_type() == torch::kFloat32) {
        // Use vectorized kernel for float32
        const int block_size = 256;
        int grid_size = (size / 4 + block_size - 1) / block_size;
        
        // Ensure sufficient occupancy
        if (grid_size < 32) grid_size = 32;
        
        selu_kernel_float32<<<grid_size, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size
        );
    } else {
        // Generic kernel for other types
        const int block_size = 256;
        int grid_size = (size / 8 + block_size - 1) / block_size;
        
        if (grid_size < 32) grid_size = 32;
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "selu_cuda_optimized", [&] {
            if (input.scalar_type() == torch::kFloat16) {
                selu_kernel_optimized<half><<<grid_size, block_size>>>(
                    reinterpret_cast<half*>(input.data_ptr<torch::Half>()),
                    reinterpret_cast<half*>(output.data_ptr<torch::Half>()),
                    size
                );
            } else if (input.scalar_type() == torch::kBFloat16) {
                // For BF16, convert to float, process, convert back
                auto input_float = input.to(torch::kFloat32);
                auto output_float = torch::empty_like(input_float);
                
                const int block_size_f = 256;
                int grid_size_f = (size / 4 + block_size_f - 1) / block_size_f;
                if (grid_size_f < 32) grid_size_f = 32;
                
                selu_kernel_float32<<<grid_size_f, block_size_f>>>(
                    input_float.data_ptr<float>(),
                    output_float.data_ptr<float>(),
                    size
                );
                
                output = output_float.to(torch::kBFloat16);
            }
        });
    }
    
    return output;
}
"""

selu_cpp_source = "torch::Tensor selu_cuda_optimized(torch::Tensor input);"

# Compile the inline CUDA code for optimized SELU
selu_extension = load_inline(
    name="selu_extension_optimized",
    cpp_sources=selu_cpp_source,
    cuda_sources=selu_source,
    functions=["selu_cuda_optimized"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas=-v"],
    extra_cflags=["-O3", "-march=native"],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs SELU activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.selu_cuda = selu_extension.selu_cuda_optimized
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized SELU activation to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with SELU applied, same shape as input.
        """
        # Only use custom kernel if tensor is on CUDA
        if x.is_cuda:
            return self.selu_cuda(x)
        else:
            # Fallback to PyTorch's implementation for CPU
            return torch.selu(x)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer2_dtype, 策略=None, 种子=142, 模式=dtype_stress

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]
```

---

## 10. `L1_P27__launch_config_mutate__1`

- **算子**: `launch_config_mutate` (Category B)
- **描述**: Perturb grid/block sizing expressions (// BLOCK, triton.cdiv) by ±1 @ L93
- **变异行**: Line 93, 原始片段 `(size / 8 + block_size - 1) / block_size`, 节点类型 `-1`
- **杀死层**: `layer2_dtype`
- **杀死策略**: `None`
- **杀死种子**: `142`
- **杀死模式**: `dtype_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized SELU activation
selu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// Constants for SELU
constexpr float ALPHA = 1.6732632423543772848170429916717f;
constexpr float SCALE = 1.0507009873554804934193349852946f;
constexpr float LAMBDA = 1.0507009873554804934193349852946f;
constexpr float NEG_ALPHA_LAMBDA = -1.7580993408473768599402175208123f;

template<typename T>
__device__ __forceinline__ T selu_compute(T x) {
    return (x >= 0) ? (SCALE * x) : (LAMBDA * (ALPHA * expf(x) - ALPHA));
}

template<>
__device__ __forceinline__ half selu_compute(half x) {
    float fx = __half2float(x);
    float result = (fx >= 0) ? (SCALE * fx) : (LAMBDA * (ALPHA * expf(fx) - ALPHA));
    return __float2half(result);
}

template<typename T>
__global__ void selu_kernel_optimized(const T* __restrict__ input, T* __restrict__ output, int64_t size) {
    // Process 8 elements per thread for better occupancy
    const int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int64_t element_idx = idx + i;
        if (element_idx < size) {
            T x = input[element_idx];
            output[element_idx] = selu_compute(x);
        }
    }
}

// Specialized kernel for float32 with warp-level optimization
__global__ void selu_kernel_float32(const float* __restrict__ input, float* __restrict__ output, int64_t size) {
    const int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
    
    // Process 4 elements per thread with vectorized loads/stores
    if (idx * 4 + 3 < size) {
        // Load 4 floats at once
        float4 in = *reinterpret_cast<const float4*>(&input[idx * 4]);
        
        // Compute SELU
        float4 out;
        out.x = (in.x >= 0) ? (SCALE * in.x) : (LAMBDA * (ALPHA * expf(in.x) - ALPHA));
        out.y = (in.y >= 0) ? (SCALE * in.y) : (LAMBDA * (ALPHA * expf(in.y) - ALPHA));
        out.z = (in.z >= 0) ? (SCALE * in.z) : (LAMBDA * (ALPHA * expf(in.z) - ALPHA));
        out.w = (in.w >= 0) ? (SCALE * in.w) : (LAMBDA * (ALPHA * expf(in.w) - ALPHA));
        
        // Store 4 floats at once
        *reinterpret_cast<float4*>(&output[idx * 4]) = out;
    } else {
        // Handle remaining elements
        for (int64_t i = idx * 4; i < size; i++) {
            float x = input[i];
            output[i] = (x >= 0) ? (SCALE * x) : (LAMBDA * (ALPHA * expf(x) - ALPHA));
        }
    }
}

torch::Tensor selu_cuda_optimized(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int64_t size = input.numel();
    
    // Optimized launch configuration based on tensor size
    if (input.scalar_type() == torch::kFloat32) {
        // Use vectorized kernel for float32
        const int block_size = 256;
        int grid_size = (size / 4 + block_size - 1) / block_size;
        
        // Ensure sufficient occupancy
        if (grid_size < 32) grid_size = 32;
        
        selu_kernel_float32<<<grid_size, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size
        );
    } else {
        // Generic kernel for other types
        const int block_size = 256;
        int grid_size = (size / 8 + block_size - 1) / block_size;
        
        if (grid_size < 32) grid_size = 32;
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "selu_cuda_optimized", [&] {
            if (input.scalar_type() == torch::kFloat16) {
                selu_kernel_optimized<half><<<grid_size, block_size>>>(
                    reinterpret_cast<half*>(input.data_ptr<torch::Half>()),
                    reinterpret_cast<half*>(output.data_ptr<torch::Half>()),
                    size
                );
            } else if (input.scalar_type() == torch::kBFloat16) {
                // For BF16, convert to float, process, convert back
                auto input_float = input.to(torch::kFloat32);
                auto output_float = torch::empty_like(input_float);
                
                const int block_size_f = 256;
                int grid_size_f = (size / 4 + block_size_f - 1) / block_size_f;
                if (grid_size_f < 32) grid_size_f = 32;
                
                selu_kernel_float32<<<grid_size_f, block_size_f>>>(
                    input_float.data_ptr<float>(),
                    output_float.data_ptr<float>(),
                    size
                );
                
                output = output_float.to(torch::kBFloat16);
            }
        });
    }
    
    return output;
}
"""

selu_cpp_source = "torch::Tensor selu_cuda_optimized(torch::Tensor input);"

# Compile the inline CUDA code for optimized SELU
selu_extension = load_inline(
    name="selu_extension_optimized",
    cpp_sources=selu_cpp_source,
    cuda_sources=selu_source,
    functions=["selu_cuda_optimized"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas=-v"],
    extra_cflags=["-O3", "-march=native"],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs SELU activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.selu_cuda = selu_extension.selu_cuda_optimized
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized SELU activation to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with SELU applied, same shape as input.
        """
        # Only use custom kernel if tensor is on CUDA
        if x.is_cuda:
            return self.selu_cuda(x)
        else:
            # Fallback to PyTorch's implementation for CPU
            return torch.selu(x)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized SELU activation
selu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// Constants for SELU
constexpr float ALPHA = 1.6732632423543772848170429916717f;
constexpr float SCALE = 1.0507009873554804934193349852946f;
constexpr float LAMBDA = 1.0507009873554804934193349852946f;
constexpr float NEG_ALPHA_LAMBDA = -1.7580993408473768599402175208123f;

template<typename T>
__device__ __forceinline__ T selu_compute(T x) {
    return (x >= 0) ? (SCALE * x) : (LAMBDA * (ALPHA * expf(x) - ALPHA));
}

template<>
__device__ __forceinline__ half selu_compute(half x) {
    float fx = __half2float(x);
    float result = (fx >= 0) ? (SCALE * fx) : (LAMBDA * (ALPHA * expf(fx) - ALPHA));
    return __float2half(result);
}

template<typename T>
__global__ void selu_kernel_optimized(const T* __restrict__ input, T* __restrict__ output, int64_t size) {
    // Process 8 elements per thread for better occupancy
    const int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int64_t element_idx = idx + i;
        if (element_idx < size) {
            T x = input[element_idx];
            output[element_idx] = selu_compute(x);
        }
    }
}

// Specialized kernel for float32 with warp-level optimization
__global__ void selu_kernel_float32(const float* __restrict__ input, float* __restrict__ output, int64_t size) {
    const int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
    
    // Process 4 elements per thread with vectorized loads/stores
    if (idx * 4 + 3 < size) {
        // Load 4 floats at once
        float4 in = *reinterpret_cast<const float4*>(&input[idx * 4]);
        
        // Compute SELU
        float4 out;
        out.x = (in.x >= 0) ? (SCALE * in.x) : (LAMBDA * (ALPHA * expf(in.x) - ALPHA));
        out.y = (in.y >= 0) ? (SCALE * in.y) : (LAMBDA * (ALPHA * expf(in.y) - ALPHA));
        out.z = (in.z >= 0) ? (SCALE * in.z) : (LAMBDA * (ALPHA * expf(in.z) - ALPHA));
        out.w = (in.w >= 0) ? (SCALE * in.w) : (LAMBDA * (ALPHA * expf(in.w) - ALPHA));
        
        // Store 4 floats at once
        *reinterpret_cast<float4*>(&output[idx * 4]) = out;
    } else {
        // Handle remaining elements
        for (int64_t i = idx * 4; i < size; i++) {
            float x = input[i];
            output[i] = (x >= 0) ? (SCALE * x) : (LAMBDA * (ALPHA * expf(x) - ALPHA));
        }
    }
}

torch::Tensor selu_cuda_optimized(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int64_t size = input.numel();
    
    // Optimized launch configuration based on tensor size
    if (input.scalar_type() == torch::kFloat32) {
        // Use vectorized kernel for float32
        const int block_size = 256;
        int grid_size = (size / 4 + block_size - 1) / block_size;
        
        // Ensure sufficient occupancy
        if (grid_size < 32) grid_size = 32;
        
        selu_kernel_float32<<<grid_size, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size
        );
    } else {
        // Generic kernel for other types
        const int block_size = 256;
        int grid_size = (size / 8 + block_size - 1) / block_size - 1;
        
        if (grid_size < 32) grid_size = 32;
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "selu_cuda_optimized", [&] {
            if (input.scalar_type() == torch::kFloat16) {
                selu_kernel_optimized<half><<<grid_size, block_size>>>(
                    reinterpret_cast<half*>(input.data_ptr<torch::Half>()),
                    reinterpret_cast<half*>(output.data_ptr<torch::Half>()),
                    size
                );
            } else if (input.scalar_type() == torch::kBFloat16) {
                // For BF16, convert to float, process, convert back
                auto input_float = input.to(torch::kFloat32);
                auto output_float = torch::empty_like(input_float);
                
                const int block_size_f = 256;
                int grid_size_f = (size / 4 + block_size_f - 1) / block_size_f;
                if (grid_size_f < 32) grid_size_f = 32;
                
                selu_kernel_float32<<<grid_size_f, block_size_f>>>(
                    input_float.data_ptr<float>(),
                    output_float.data_ptr<float>(),
                    size
                );
                
                output = output_float.to(torch::kBFloat16);
            }
        });
    }
    
    return output;
}
"""

selu_cpp_source = "torch::Tensor selu_cuda_optimized(torch::Tensor input);"

# Compile the inline CUDA code for optimized SELU
selu_extension = load_inline(
    name="selu_extension_optimized",
    cpp_sources=selu_cpp_source,
    cuda_sources=selu_source,
    functions=["selu_cuda_optimized"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas=-v"],
    extra_cflags=["-O3", "-march=native"],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs SELU activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.selu_cuda = selu_extension.selu_cuda_optimized
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized SELU activation to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with SELU applied, same shape as input.
        """
        # Only use custom kernel if tensor is on CUDA
        if x.is_cuda:
            return self.selu_cuda(x)
        else:
            # Fallback to PyTorch's implementation for CPU
            return torch.selu(x)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer2_dtype, 策略=None, 种子=142, 模式=dtype_stress

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]
```

---

## 11. `L1_P27__mask_boundary__0`

- **算子**: `mask_boundary` (Category B)
- **描述**: Weaken or tighten boundary checks in mask/tl.where (Triton) or thread/block guards (CUDA) @ L38
- **变异行**: Line 38, 原始片段 `element_idx < size`, 节点类型 `rhs-1`
- **杀死层**: `layer2_dtype`
- **杀死策略**: `None`
- **杀死种子**: `142`
- **杀死模式**: `dtype_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized SELU activation
selu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// Constants for SELU
constexpr float ALPHA = 1.6732632423543772848170429916717f;
constexpr float SCALE = 1.0507009873554804934193349852946f;
constexpr float LAMBDA = 1.0507009873554804934193349852946f;
constexpr float NEG_ALPHA_LAMBDA = -1.7580993408473768599402175208123f;

template<typename T>
__device__ __forceinline__ T selu_compute(T x) {
    return (x >= 0) ? (SCALE * x) : (LAMBDA * (ALPHA * expf(x) - ALPHA));
}

template<>
__device__ __forceinline__ half selu_compute(half x) {
    float fx = __half2float(x);
    float result = (fx >= 0) ? (SCALE * fx) : (LAMBDA * (ALPHA * expf(fx) - ALPHA));
    return __float2half(result);
}

template<typename T>
__global__ void selu_kernel_optimized(const T* __restrict__ input, T* __restrict__ output, int64_t size) {
    // Process 8 elements per thread for better occupancy
    const int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int64_t element_idx = idx + i;
        if (element_idx < size) {
            T x = input[element_idx];
            output[element_idx] = selu_compute(x);
        }
    }
}

// Specialized kernel for float32 with warp-level optimization
__global__ void selu_kernel_float32(const float* __restrict__ input, float* __restrict__ output, int64_t size) {
    const int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
    
    // Process 4 elements per thread with vectorized loads/stores
    if (idx * 4 + 3 < size) {
        // Load 4 floats at once
        float4 in = *reinterpret_cast<const float4*>(&input[idx * 4]);
        
        // Compute SELU
        float4 out;
        out.x = (in.x >= 0) ? (SCALE * in.x) : (LAMBDA * (ALPHA * expf(in.x) - ALPHA));
        out.y = (in.y >= 0) ? (SCALE * in.y) : (LAMBDA * (ALPHA * expf(in.y) - ALPHA));
        out.z = (in.z >= 0) ? (SCALE * in.z) : (LAMBDA * (ALPHA * expf(in.z) - ALPHA));
        out.w = (in.w >= 0) ? (SCALE * in.w) : (LAMBDA * (ALPHA * expf(in.w) - ALPHA));
        
        // Store 4 floats at once
        *reinterpret_cast<float4*>(&output[idx * 4]) = out;
    } else {
        // Handle remaining elements
        for (int64_t i = idx * 4; i < size; i++) {
            float x = input[i];
            output[i] = (x >= 0) ? (SCALE * x) : (LAMBDA * (ALPHA * expf(x) - ALPHA));
        }
    }
}

torch::Tensor selu_cuda_optimized(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int64_t size = input.numel();
    
    // Optimized launch configuration based on tensor size
    if (input.scalar_type() == torch::kFloat32) {
        // Use vectorized kernel for float32
        const int block_size = 256;
        int grid_size = (size / 4 + block_size - 1) / block_size;
        
        // Ensure sufficient occupancy
        if (grid_size < 32) grid_size = 32;
        
        selu_kernel_float32<<<grid_size, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size
        );
    } else {
        // Generic kernel for other types
        const int block_size = 256;
        int grid_size = (size / 8 + block_size - 1) / block_size;
        
        if (grid_size < 32) grid_size = 32;
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "selu_cuda_optimized", [&] {
            if (input.scalar_type() == torch::kFloat16) {
                selu_kernel_optimized<half><<<grid_size, block_size>>>(
                    reinterpret_cast<half*>(input.data_ptr<torch::Half>()),
                    reinterpret_cast<half*>(output.data_ptr<torch::Half>()),
                    size
                );
            } else if (input.scalar_type() == torch::kBFloat16) {
                // For BF16, convert to float, process, convert back
                auto input_float = input.to(torch::kFloat32);
                auto output_float = torch::empty_like(input_float);
                
                const int block_size_f = 256;
                int grid_size_f = (size / 4 + block_size_f - 1) / block_size_f;
                if (grid_size_f < 32) grid_size_f = 32;
                
                selu_kernel_float32<<<grid_size_f, block_size_f>>>(
                    input_float.data_ptr<float>(),
                    output_float.data_ptr<float>(),
                    size
                );
                
                output = output_float.to(torch::kBFloat16);
            }
        });
    }
    
    return output;
}
"""

selu_cpp_source = "torch::Tensor selu_cuda_optimized(torch::Tensor input);"

# Compile the inline CUDA code for optimized SELU
selu_extension = load_inline(
    name="selu_extension_optimized",
    cpp_sources=selu_cpp_source,
    cuda_sources=selu_source,
    functions=["selu_cuda_optimized"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas=-v"],
    extra_cflags=["-O3", "-march=native"],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs SELU activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.selu_cuda = selu_extension.selu_cuda_optimized
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized SELU activation to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with SELU applied, same shape as input.
        """
        # Only use custom kernel if tensor is on CUDA
        if x.is_cuda:
            return self.selu_cuda(x)
        else:
            # Fallback to PyTorch's implementation for CPU
            return torch.selu(x)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized SELU activation
selu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// Constants for SELU
constexpr float ALPHA = 1.6732632423543772848170429916717f;
constexpr float SCALE = 1.0507009873554804934193349852946f;
constexpr float LAMBDA = 1.0507009873554804934193349852946f;
constexpr float NEG_ALPHA_LAMBDA = -1.7580993408473768599402175208123f;

template<typename T>
__device__ __forceinline__ T selu_compute(T x) {
    return (x >= 0) ? (SCALE * x) : (LAMBDA * (ALPHA * expf(x) - ALPHA));
}

template<>
__device__ __forceinline__ half selu_compute(half x) {
    float fx = __half2float(x);
    float result = (fx >= 0) ? (SCALE * fx) : (LAMBDA * (ALPHA * expf(fx) - ALPHA));
    return __float2half(result);
}

template<typename T>
__global__ void selu_kernel_optimized(const T* __restrict__ input, T* __restrict__ output, int64_t size) {
    // Process 8 elements per thread for better occupancy
    const int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int64_t element_idx = idx + i;
        if (element_idx < size - 1) {
            T x = input[element_idx];
            output[element_idx] = selu_compute(x);
        }
    }
}

// Specialized kernel for float32 with warp-level optimization
__global__ void selu_kernel_float32(const float* __restrict__ input, float* __restrict__ output, int64_t size) {
    const int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
    
    // Process 4 elements per thread with vectorized loads/stores
    if (idx * 4 + 3 < size) {
        // Load 4 floats at once
        float4 in = *reinterpret_cast<const float4*>(&input[idx * 4]);
        
        // Compute SELU
        float4 out;
        out.x = (in.x >= 0) ? (SCALE * in.x) : (LAMBDA * (ALPHA * expf(in.x) - ALPHA));
        out.y = (in.y >= 0) ? (SCALE * in.y) : (LAMBDA * (ALPHA * expf(in.y) - ALPHA));
        out.z = (in.z >= 0) ? (SCALE * in.z) : (LAMBDA * (ALPHA * expf(in.z) - ALPHA));
        out.w = (in.w >= 0) ? (SCALE * in.w) : (LAMBDA * (ALPHA * expf(in.w) - ALPHA));
        
        // Store 4 floats at once
        *reinterpret_cast<float4*>(&output[idx * 4]) = out;
    } else {
        // Handle remaining elements
        for (int64_t i = idx * 4; i < size; i++) {
            float x = input[i];
            output[i] = (x >= 0) ? (SCALE * x) : (LAMBDA * (ALPHA * expf(x) - ALPHA));
        }
    }
}

torch::Tensor selu_cuda_optimized(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int64_t size = input.numel();
    
    // Optimized launch configuration based on tensor size
    if (input.scalar_type() == torch::kFloat32) {
        // Use vectorized kernel for float32
        const int block_size = 256;
        int grid_size = (size / 4 + block_size - 1) / block_size;
        
        // Ensure sufficient occupancy
        if (grid_size < 32) grid_size = 32;
        
        selu_kernel_float32<<<grid_size, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size
        );
    } else {
        // Generic kernel for other types
        const int block_size = 256;
        int grid_size = (size / 8 + block_size - 1) / block_size;
        
        if (grid_size < 32) grid_size = 32;
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "selu_cuda_optimized", [&] {
            if (input.scalar_type() == torch::kFloat16) {
                selu_kernel_optimized<half><<<grid_size, block_size>>>(
                    reinterpret_cast<half*>(input.data_ptr<torch::Half>()),
                    reinterpret_cast<half*>(output.data_ptr<torch::Half>()),
                    size
                );
            } else if (input.scalar_type() == torch::kBFloat16) {
                // For BF16, convert to float, process, convert back
                auto input_float = input.to(torch::kFloat32);
                auto output_float = torch::empty_like(input_float);
                
                const int block_size_f = 256;
                int grid_size_f = (size / 4 + block_size_f - 1) / block_size_f;
                if (grid_size_f < 32) grid_size_f = 32;
                
                selu_kernel_float32<<<grid_size_f, block_size_f>>>(
                    input_float.data_ptr<float>(),
                    output_float.data_ptr<float>(),
                    size
                );
                
                output = output_float.to(torch::kBFloat16);
            }
        });
    }
    
    return output;
}
"""

selu_cpp_source = "torch::Tensor selu_cuda_optimized(torch::Tensor input);"

# Compile the inline CUDA code for optimized SELU
selu_extension = load_inline(
    name="selu_extension_optimized",
    cpp_sources=selu_cpp_source,
    cuda_sources=selu_source,
    functions=["selu_cuda_optimized"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas=-v"],
    extra_cflags=["-O3", "-march=native"],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs SELU activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.selu_cuda = selu_extension.selu_cuda_optimized
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized SELU activation to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with SELU applied, same shape as input.
        """
        # Only use custom kernel if tensor is on CUDA
        if x.is_cuda:
            return self.selu_cuda(x)
        else:
            # Fallback to PyTorch's implementation for CPU
            return torch.selu(x)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer2_dtype, 策略=None, 种子=142, 模式=dtype_stress

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]
```

---

## 12. `L1_P33__epsilon_modify__0`

- **算子**: `epsilon_modify` (Category C)
- **描述**: Alter small epsilon literals (LayerNorm / safe_div / log stability) @ L267
- **变异行**: Line 267, 原始片段 `1e-5`, 节点类型 `eps:to_1e-2`
- **杀死层**: `layer4_training`
- **杀死策略**: `None`
- **杀死种子**: `372`
- **杀死模式**: `training_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for BatchNorm2d
batchnorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void batchnorm2d_forward_inference_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ running_mean,
    const scalar_t* __restrict__ running_var,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t eps,
    int batch_size,
    int num_features,
    int spatial_size,
    int total_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    int spatial_idx = idx % spatial_size;
    int feature_idx = (idx / spatial_size) % num_features;
    int batch_idx = idx / (spatial_size * num_features);
    
    scalar_t mean = running_mean[feature_idx];
    scalar_t var = running_var[feature_idx];
    scalar_t gamma = weight ? weight[feature_idx] : scalar_t(1.0);
    scalar_t beta = bias ? bias[feature_idx] : scalar_t(0.0);
    
    scalar_t inv_std = rsqrt(var + eps);
    scalar_t scale = gamma * inv_std;
    scalar_t shift = beta - mean * scale;
    
    int input_idx = ((batch_idx * num_features + feature_idx) * spatial_size) + spatial_idx;
    output[input_idx] = input[input_idx] * scale + shift;
}

template<typename scalar_t>
__global__ void compute_mean_var_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ batch_mean,
    scalar_t* __restrict__ batch_var,
    int batch_size,
    int num_features,
    int spatial_size) {
    
    extern __shared__ char shared_mem_raw[];
    scalar_t* shared_sum = reinterpret_cast<scalar_t*>(shared_mem_raw);
    scalar_t* shared_sqsum = &shared_sum[blockDim.x];
    
    int feature_idx = blockIdx.x;
    int tid = threadIdx.x;
    int total_elements = batch_size * spatial_size;
    
    scalar_t sum = scalar_t(0);
    scalar_t sqsum = scalar_t(0);
    
    // Each thread processes multiple elements
    for (int i = tid; i < total_elements; i += blockDim.x) {
        int batch_idx = i / spatial_size;
        int spatial_idx = i % spatial_size;
        int input_idx = ((batch_idx * num_features + feature_idx) * spatial_size) + spatial_idx;
        scalar_t val = input[input_idx];
        sum += val;
        sqsum += val * val;
    }
    
    shared_sum[tid] = sum;
    shared_sqsum[tid] = sqsum;
    __syncthreads();
    
    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sqsum[tid] += shared_sqsum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        scalar_t mean_val = shared_sum[0] / total_elements;
        scalar_t var_val = shared_sqsum[0] / total_elements - mean_val * mean_val;
        batch_mean[feature_idx] = mean_val;
        batch_var[feature_idx] = var_val;
    }
}

template<typename scalar_t>
__global__ void apply_batchnorm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ batch_mean,
    const scalar_t* __restrict__ batch_var,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t eps,
    int batch_size,
    int num_features,
    int spatial_size,
    int total_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    int spatial_idx = idx % spatial_size;
    int feature_idx = (idx / spatial_size) % num_features;
    int batch_idx = idx / (spatial_size * num_features);
    
    scalar_t mean_val = batch_mean[feature_idx];
    scalar_t var_val = batch_var[feature_idx];
    scalar_t gamma = weight ? weight[feature_idx] : scalar_t(1.0);
    scalar_t beta = bias ? bias[feature_idx] : scalar_t(0.0);
    
    scalar_t inv_std = rsqrt(var_val + eps);
    scalar_t scale = gamma * inv_std;
    scalar_t shift = beta - mean_val * scale;
    
    int input_idx = ((batch_idx * num_features + feature_idx) * spatial_size) + spatial_idx;
    output[input_idx] = input[input_idx] * scale + shift;
}

template<typename scalar_t>
__global__ void update_running_stats_kernel(
    scalar_t* __restrict__ running_mean,
    scalar_t* __restrict__ running_var,
    const scalar_t* __restrict__ batch_mean,
    const scalar_t* __restrict__ batch_var,
    scalar_t momentum,
    int num_features) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_features) return;
    
    scalar_t new_running_mean = (scalar_t(1) - momentum) * running_mean[idx] + momentum * batch_mean[idx];
    scalar_t new_running_var = (scalar_t(1) - momentum) * running_var[idx] + momentum * batch_var[idx];
    
    running_mean[idx] = new_running_mean;
    running_var[idx] = new_running_var;
}

torch::Tensor batchnorm2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps,
    bool training,
    float momentum) {
    
    auto batch_size = input.size(0);
    auto num_features = input.size(1);
    auto spatial_size = input.size(2) * input.size(3);
    auto total_elements = batch_size * num_features * spatial_size;
    
    auto output = torch::empty_like(input);
    
    if (!training) {
        // Inference mode - fully parallel
        const int block_size = 256;
        const int num_blocks = (total_elements + block_size - 1) / block_size;
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batchnorm2d_forward_inference_cuda", [&]() {
            batchnorm2d_forward_inference_kernel<scalar_t><<<num_blocks, block_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                running_mean.data_ptr<scalar_t>(),
                running_var.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<scalar_t>() : nullptr,
                bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
                scalar_t(eps),
                batch_size,
                num_features,
                spatial_size,
                total_elements);
        });
    } else {
        // Training mode - optimized 3-step approach
        auto batch_mean = torch::empty(num_features, input.options());
        auto batch_var = torch::empty(num_features, input.options());
        
        // Step 1: Compute mean and variance per feature
        const int threads_per_feature = 256;
        const size_t shared_mem_size = 2 * threads_per_feature * sizeof(float);
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "compute_mean_var_cuda", [&]() {
            compute_mean_var_kernel<scalar_t><<<num_features, threads_per_feature, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                batch_mean.data_ptr<scalar_t>(),
                batch_var.data_ptr<scalar_t>(),
                batch_size,
                num_features,
                spatial_size);
        });
        
        // Step 2: Apply batch normalization
        const int block_size = 256;
        const int num_blocks = (total_elements + block_size - 1) / block_size;
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "apply_batchnorm_cuda", [&]() {
            apply_batchnorm_kernel<scalar_t><<<num_blocks, block_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_mean.data_ptr<scalar_t>(),
                batch_var.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<scalar_t>() : nullptr,
                bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
                scalar_t(eps),
                batch_size,
                num_features,
                spatial_size,
                total_elements);
        });
        
        // Step 3: Update running statistics
        const int update_block_size = 256;
        const int update_blocks = (num_features + update_block_size - 1) / update_block_size;
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "update_running_stats_cuda", [&]() {
            update_running_stats_kernel<scalar_t><<<update_blocks, update_block_size>>>(
                running_mean.data_ptr<scalar_t>(),
                running_var.data_ptr<scalar_t>(),
                batch_mean.data_ptr<scalar_t>(),
                batch_var.data_ptr<scalar_t>(),
                scalar_t(momentum),
                num_features);
        });
    }
    
    cudaDeviceSynchronize();
    return output;
}
"""

batchnorm_cpp_source = """
torch::Tensor batchnorm2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps,
    bool training,
    float momentum);
"""

# Compile the inline CUDA code
custom_batchnorm = load_inline(
    name="custom_batchnorm",
    cpp_sources=batchnorm_cpp_source,
    cuda_sources=batchnorm_source,
    functions=["batchnorm2d_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    with_cuda=True,
)

class CustomBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super(CustomBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        self.batchnorm_op = custom_batchnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.batchnorm_op.batchnorm2d_forward_cuda(
            x, 
            self.running_mean, 
            self.running_var,
            self.weight, 
            self.bias, 
            self.eps,
            self.training,
            self.momentum
        )

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.bn = CustomBatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for BatchNorm2d
batchnorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void batchnorm2d_forward_inference_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ running_mean,
    const scalar_t* __restrict__ running_var,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t eps,
    int batch_size,
    int num_features,
    int spatial_size,
    int total_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    int spatial_idx = idx % spatial_size;
    int feature_idx = (idx / spatial_size) % num_features;
    int batch_idx = idx / (spatial_size * num_features);
    
    scalar_t mean = running_mean[feature_idx];
    scalar_t var = running_var[feature_idx];
    scalar_t gamma = weight ? weight[feature_idx] : scalar_t(1.0);
    scalar_t beta = bias ? bias[feature_idx] : scalar_t(0.0);
    
    scalar_t inv_std = rsqrt(var + eps);
    scalar_t scale = gamma * inv_std;
    scalar_t shift = beta - mean * scale;
    
    int input_idx = ((batch_idx * num_features + feature_idx) * spatial_size) + spatial_idx;
    output[input_idx] = input[input_idx] * scale + shift;
}

template<typename scalar_t>
__global__ void compute_mean_var_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ batch_mean,
    scalar_t* __restrict__ batch_var,
    int batch_size,
    int num_features,
    int spatial_size) {
    
    extern __shared__ char shared_mem_raw[];
    scalar_t* shared_sum = reinterpret_cast<scalar_t*>(shared_mem_raw);
    scalar_t* shared_sqsum = &shared_sum[blockDim.x];
    
    int feature_idx = blockIdx.x;
    int tid = threadIdx.x;
    int total_elements = batch_size * spatial_size;
    
    scalar_t sum = scalar_t(0);
    scalar_t sqsum = scalar_t(0);
    
    // Each thread processes multiple elements
    for (int i = tid; i < total_elements; i += blockDim.x) {
        int batch_idx = i / spatial_size;
        int spatial_idx = i % spatial_size;
        int input_idx = ((batch_idx * num_features + feature_idx) * spatial_size) + spatial_idx;
        scalar_t val = input[input_idx];
        sum += val;
        sqsum += val * val;
    }
    
    shared_sum[tid] = sum;
    shared_sqsum[tid] = sqsum;
    __syncthreads();
    
    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sqsum[tid] += shared_sqsum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        scalar_t mean_val = shared_sum[0] / total_elements;
        scalar_t var_val = shared_sqsum[0] / total_elements - mean_val * mean_val;
        batch_mean[feature_idx] = mean_val;
        batch_var[feature_idx] = var_val;
    }
}

template<typename scalar_t>
__global__ void apply_batchnorm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ batch_mean,
    const scalar_t* __restrict__ batch_var,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t eps,
    int batch_size,
    int num_features,
    int spatial_size,
    int total_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    int spatial_idx = idx % spatial_size;
    int feature_idx = (idx / spatial_size) % num_features;
    int batch_idx = idx / (spatial_size * num_features);
    
    scalar_t mean_val = batch_mean[feature_idx];
    scalar_t var_val = batch_var[feature_idx];
    scalar_t gamma = weight ? weight[feature_idx] : scalar_t(1.0);
    scalar_t beta = bias ? bias[feature_idx] : scalar_t(0.0);
    
    scalar_t inv_std = rsqrt(var_val + eps);
    scalar_t scale = gamma * inv_std;
    scalar_t shift = beta - mean_val * scale;
    
    int input_idx = ((batch_idx * num_features + feature_idx) * spatial_size) + spatial_idx;
    output[input_idx] = input[input_idx] * scale + shift;
}

template<typename scalar_t>
__global__ void update_running_stats_kernel(
    scalar_t* __restrict__ running_mean,
    scalar_t* __restrict__ running_var,
    const scalar_t* __restrict__ batch_mean,
    const scalar_t* __restrict__ batch_var,
    scalar_t momentum,
    int num_features) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_features) return;
    
    scalar_t new_running_mean = (scalar_t(1) - momentum) * running_mean[idx] + momentum * batch_mean[idx];
    scalar_t new_running_var = (scalar_t(1) - momentum) * running_var[idx] + momentum * batch_var[idx];
    
    running_mean[idx] = new_running_mean;
    running_var[idx] = new_running_var;
}

torch::Tensor batchnorm2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps,
    bool training,
    float momentum) {
    
    auto batch_size = input.size(0);
    auto num_features = input.size(1);
    auto spatial_size = input.size(2) * input.size(3);
    auto total_elements = batch_size * num_features * spatial_size;
    
    auto output = torch::empty_like(input);
    
    if (!training) {
        // Inference mode - fully parallel
        const int block_size = 256;
        const int num_blocks = (total_elements + block_size - 1) / block_size;
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batchnorm2d_forward_inference_cuda", [&]() {
            batchnorm2d_forward_inference_kernel<scalar_t><<<num_blocks, block_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                running_mean.data_ptr<scalar_t>(),
                running_var.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<scalar_t>() : nullptr,
                bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
                scalar_t(eps),
                batch_size,
                num_features,
                spatial_size,
                total_elements);
        });
    } else {
        // Training mode - optimized 3-step approach
        auto batch_mean = torch::empty(num_features, input.options());
        auto batch_var = torch::empty(num_features, input.options());
        
        // Step 1: Compute mean and variance per feature
        const int threads_per_feature = 256;
        const size_t shared_mem_size = 2 * threads_per_feature * sizeof(float);
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "compute_mean_var_cuda", [&]() {
            compute_mean_var_kernel<scalar_t><<<num_features, threads_per_feature, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                batch_mean.data_ptr<scalar_t>(),
                batch_var.data_ptr<scalar_t>(),
                batch_size,
                num_features,
                spatial_size);
        });
        
        // Step 2: Apply batch normalization
        const int block_size = 256;
        const int num_blocks = (total_elements + block_size - 1) / block_size;
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "apply_batchnorm_cuda", [&]() {
            apply_batchnorm_kernel<scalar_t><<<num_blocks, block_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_mean.data_ptr<scalar_t>(),
                batch_var.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<scalar_t>() : nullptr,
                bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
                scalar_t(eps),
                batch_size,
                num_features,
                spatial_size,
                total_elements);
        });
        
        // Step 3: Update running statistics
        const int update_block_size = 256;
        const int update_blocks = (num_features + update_block_size - 1) / update_block_size;
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "update_running_stats_cuda", [&]() {
            update_running_stats_kernel<scalar_t><<<update_blocks, update_block_size>>>(
                running_mean.data_ptr<scalar_t>(),
                running_var.data_ptr<scalar_t>(),
                batch_mean.data_ptr<scalar_t>(),
                batch_var.data_ptr<scalar_t>(),
                scalar_t(momentum),
                num_features);
        });
    }
    
    cudaDeviceSynchronize();
    return output;
}
"""

batchnorm_cpp_source = """
torch::Tensor batchnorm2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps,
    bool training,
    float momentum);
"""

# Compile the inline CUDA code
custom_batchnorm = load_inline(
    name="custom_batchnorm",
    cpp_sources=batchnorm_cpp_source,
    cuda_sources=batchnorm_source,
    functions=["batchnorm2d_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    with_cuda=True,
)

class CustomBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-2, momentum: float = 0.1):
        super(CustomBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        self.batchnorm_op = custom_batchnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.batchnorm_op.batchnorm2d_forward_cuda(
            x, 
            self.running_mean, 
            self.running_var,
            self.weight, 
            self.bias, 
            self.eps,
            self.training,
            self.momentum
        )

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.bn = CustomBatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer4_training, 策略=None, 种子=372, 模式=training_stress

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]
```

---

## 13. `L1_P33__epsilon_modify__1`

- **算子**: `epsilon_modify` (Category C)
- **描述**: Alter small epsilon literals (LayerNorm / safe_div / log stability) @ L267
- **变异行**: Line 267, 原始片段 `1e-5`, 节点类型 `eps:to_zero`
- **杀死层**: `layer4_training`
- **杀死策略**: `None`
- **杀死种子**: `342`
- **杀死模式**: `training_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for BatchNorm2d
batchnorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void batchnorm2d_forward_inference_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ running_mean,
    const scalar_t* __restrict__ running_var,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t eps,
    int batch_size,
    int num_features,
    int spatial_size,
    int total_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    int spatial_idx = idx % spatial_size;
    int feature_idx = (idx / spatial_size) % num_features;
    int batch_idx = idx / (spatial_size * num_features);
    
    scalar_t mean = running_mean[feature_idx];
    scalar_t var = running_var[feature_idx];
    scalar_t gamma = weight ? weight[feature_idx] : scalar_t(1.0);
    scalar_t beta = bias ? bias[feature_idx] : scalar_t(0.0);
    
    scalar_t inv_std = rsqrt(var + eps);
    scalar_t scale = gamma * inv_std;
    scalar_t shift = beta - mean * scale;
    
    int input_idx = ((batch_idx * num_features + feature_idx) * spatial_size) + spatial_idx;
    output[input_idx] = input[input_idx] * scale + shift;
}

template<typename scalar_t>
__global__ void compute_mean_var_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ batch_mean,
    scalar_t* __restrict__ batch_var,
    int batch_size,
    int num_features,
    int spatial_size) {
    
    extern __shared__ char shared_mem_raw[];
    scalar_t* shared_sum = reinterpret_cast<scalar_t*>(shared_mem_raw);
    scalar_t* shared_sqsum = &shared_sum[blockDim.x];
    
    int feature_idx = blockIdx.x;
    int tid = threadIdx.x;
    int total_elements = batch_size * spatial_size;
    
    scalar_t sum = scalar_t(0);
    scalar_t sqsum = scalar_t(0);
    
    // Each thread processes multiple elements
    for (int i = tid; i < total_elements; i += blockDim.x) {
        int batch_idx = i / spatial_size;
        int spatial_idx = i % spatial_size;
        int input_idx = ((batch_idx * num_features + feature_idx) * spatial_size) + spatial_idx;
        scalar_t val = input[input_idx];
        sum += val;
        sqsum += val * val;
    }
    
    shared_sum[tid] = sum;
    shared_sqsum[tid] = sqsum;
    __syncthreads();
    
    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sqsum[tid] += shared_sqsum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        scalar_t mean_val = shared_sum[0] / total_elements;
        scalar_t var_val = shared_sqsum[0] / total_elements - mean_val * mean_val;
        batch_mean[feature_idx] = mean_val;
        batch_var[feature_idx] = var_val;
    }
}

template<typename scalar_t>
__global__ void apply_batchnorm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ batch_mean,
    const scalar_t* __restrict__ batch_var,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t eps,
    int batch_size,
    int num_features,
    int spatial_size,
    int total_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    int spatial_idx = idx % spatial_size;
    int feature_idx = (idx / spatial_size) % num_features;
    int batch_idx = idx / (spatial_size * num_features);
    
    scalar_t mean_val = batch_mean[feature_idx];
    scalar_t var_val = batch_var[feature_idx];
    scalar_t gamma = weight ? weight[feature_idx] : scalar_t(1.0);
    scalar_t beta = bias ? bias[feature_idx] : scalar_t(0.0);
    
    scalar_t inv_std = rsqrt(var_val + eps);
    scalar_t scale = gamma * inv_std;
    scalar_t shift = beta - mean_val * scale;
    
    int input_idx = ((batch_idx * num_features + feature_idx) * spatial_size) + spatial_idx;
    output[input_idx] = input[input_idx] * scale + shift;
}

template<typename scalar_t>
__global__ void update_running_stats_kernel(
    scalar_t* __restrict__ running_mean,
    scalar_t* __restrict__ running_var,
    const scalar_t* __restrict__ batch_mean,
    const scalar_t* __restrict__ batch_var,
    scalar_t momentum,
    int num_features) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_features) return;
    
    scalar_t new_running_mean = (scalar_t(1) - momentum) * running_mean[idx] + momentum * batch_mean[idx];
    scalar_t new_running_var = (scalar_t(1) - momentum) * running_var[idx] + momentum * batch_var[idx];
    
    running_mean[idx] = new_running_mean;
    running_var[idx] = new_running_var;
}

torch::Tensor batchnorm2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps,
    bool training,
    float momentum) {
    
    auto batch_size = input.size(0);
    auto num_features = input.size(1);
    auto spatial_size = input.size(2) * input.size(3);
    auto total_elements = batch_size * num_features * spatial_size;
    
    auto output = torch::empty_like(input);
    
    if (!training) {
        // Inference mode - fully parallel
        const int block_size = 256;
        const int num_blocks = (total_elements + block_size - 1) / block_size;
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batchnorm2d_forward_inference_cuda", [&]() {
            batchnorm2d_forward_inference_kernel<scalar_t><<<num_blocks, block_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                running_mean.data_ptr<scalar_t>(),
                running_var.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<scalar_t>() : nullptr,
                bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
                scalar_t(eps),
                batch_size,
                num_features,
                spatial_size,
                total_elements);
        });
    } else {
        // Training mode - optimized 3-step approach
        auto batch_mean = torch::empty(num_features, input.options());
        auto batch_var = torch::empty(num_features, input.options());
        
        // Step 1: Compute mean and variance per feature
        const int threads_per_feature = 256;
        const size_t shared_mem_size = 2 * threads_per_feature * sizeof(float);
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "compute_mean_var_cuda", [&]() {
            compute_mean_var_kernel<scalar_t><<<num_features, threads_per_feature, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                batch_mean.data_ptr<scalar_t>(),
                batch_var.data_ptr<scalar_t>(),
                batch_size,
                num_features,
                spatial_size);
        });
        
        // Step 2: Apply batch normalization
        const int block_size = 256;
        const int num_blocks = (total_elements + block_size - 1) / block_size;
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "apply_batchnorm_cuda", [&]() {
            apply_batchnorm_kernel<scalar_t><<<num_blocks, block_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_mean.data_ptr<scalar_t>(),
                batch_var.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<scalar_t>() : nullptr,
                bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
                scalar_t(eps),
                batch_size,
                num_features,
                spatial_size,
                total_elements);
        });
        
        // Step 3: Update running statistics
        const int update_block_size = 256;
        const int update_blocks = (num_features + update_block_size - 1) / update_block_size;
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "update_running_stats_cuda", [&]() {
            update_running_stats_kernel<scalar_t><<<update_blocks, update_block_size>>>(
                running_mean.data_ptr<scalar_t>(),
                running_var.data_ptr<scalar_t>(),
                batch_mean.data_ptr<scalar_t>(),
                batch_var.data_ptr<scalar_t>(),
                scalar_t(momentum),
                num_features);
        });
    }
    
    cudaDeviceSynchronize();
    return output;
}
"""

batchnorm_cpp_source = """
torch::Tensor batchnorm2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps,
    bool training,
    float momentum);
"""

# Compile the inline CUDA code
custom_batchnorm = load_inline(
    name="custom_batchnorm",
    cpp_sources=batchnorm_cpp_source,
    cuda_sources=batchnorm_source,
    functions=["batchnorm2d_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    with_cuda=True,
)

class CustomBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super(CustomBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        self.batchnorm_op = custom_batchnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.batchnorm_op.batchnorm2d_forward_cuda(
            x, 
            self.running_mean, 
            self.running_var,
            self.weight, 
            self.bias, 
            self.eps,
            self.training,
            self.momentum
        )

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.bn = CustomBatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for BatchNorm2d
batchnorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void batchnorm2d_forward_inference_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ running_mean,
    const scalar_t* __restrict__ running_var,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t eps,
    int batch_size,
    int num_features,
    int spatial_size,
    int total_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    int spatial_idx = idx % spatial_size;
    int feature_idx = (idx / spatial_size) % num_features;
    int batch_idx = idx / (spatial_size * num_features);
    
    scalar_t mean = running_mean[feature_idx];
    scalar_t var = running_var[feature_idx];
    scalar_t gamma = weight ? weight[feature_idx] : scalar_t(1.0);
    scalar_t beta = bias ? bias[feature_idx] : scalar_t(0.0);
    
    scalar_t inv_std = rsqrt(var + eps);
    scalar_t scale = gamma * inv_std;
    scalar_t shift = beta - mean * scale;
    
    int input_idx = ((batch_idx * num_features + feature_idx) * spatial_size) + spatial_idx;
    output[input_idx] = input[input_idx] * scale + shift;
}

template<typename scalar_t>
__global__ void compute_mean_var_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ batch_mean,
    scalar_t* __restrict__ batch_var,
    int batch_size,
    int num_features,
    int spatial_size) {
    
    extern __shared__ char shared_mem_raw[];
    scalar_t* shared_sum = reinterpret_cast<scalar_t*>(shared_mem_raw);
    scalar_t* shared_sqsum = &shared_sum[blockDim.x];
    
    int feature_idx = blockIdx.x;
    int tid = threadIdx.x;
    int total_elements = batch_size * spatial_size;
    
    scalar_t sum = scalar_t(0);
    scalar_t sqsum = scalar_t(0);
    
    // Each thread processes multiple elements
    for (int i = tid; i < total_elements; i += blockDim.x) {
        int batch_idx = i / spatial_size;
        int spatial_idx = i % spatial_size;
        int input_idx = ((batch_idx * num_features + feature_idx) * spatial_size) + spatial_idx;
        scalar_t val = input[input_idx];
        sum += val;
        sqsum += val * val;
    }
    
    shared_sum[tid] = sum;
    shared_sqsum[tid] = sqsum;
    __syncthreads();
    
    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sqsum[tid] += shared_sqsum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        scalar_t mean_val = shared_sum[0] / total_elements;
        scalar_t var_val = shared_sqsum[0] / total_elements - mean_val * mean_val;
        batch_mean[feature_idx] = mean_val;
        batch_var[feature_idx] = var_val;
    }
}

template<typename scalar_t>
__global__ void apply_batchnorm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ batch_mean,
    const scalar_t* __restrict__ batch_var,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t eps,
    int batch_size,
    int num_features,
    int spatial_size,
    int total_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    int spatial_idx = idx % spatial_size;
    int feature_idx = (idx / spatial_size) % num_features;
    int batch_idx = idx / (spatial_size * num_features);
    
    scalar_t mean_val = batch_mean[feature_idx];
    scalar_t var_val = batch_var[feature_idx];
    scalar_t gamma = weight ? weight[feature_idx] : scalar_t(1.0);
    scalar_t beta = bias ? bias[feature_idx] : scalar_t(0.0);
    
    scalar_t inv_std = rsqrt(var_val + eps);
    scalar_t scale = gamma * inv_std;
    scalar_t shift = beta - mean_val * scale;
    
    int input_idx = ((batch_idx * num_features + feature_idx) * spatial_size) + spatial_idx;
    output[input_idx] = input[input_idx] * scale + shift;
}

template<typename scalar_t>
__global__ void update_running_stats_kernel(
    scalar_t* __restrict__ running_mean,
    scalar_t* __restrict__ running_var,
    const scalar_t* __restrict__ batch_mean,
    const scalar_t* __restrict__ batch_var,
    scalar_t momentum,
    int num_features) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_features) return;
    
    scalar_t new_running_mean = (scalar_t(1) - momentum) * running_mean[idx] + momentum * batch_mean[idx];
    scalar_t new_running_var = (scalar_t(1) - momentum) * running_var[idx] + momentum * batch_var[idx];
    
    running_mean[idx] = new_running_mean;
    running_var[idx] = new_running_var;
}

torch::Tensor batchnorm2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps,
    bool training,
    float momentum) {
    
    auto batch_size = input.size(0);
    auto num_features = input.size(1);
    auto spatial_size = input.size(2) * input.size(3);
    auto total_elements = batch_size * num_features * spatial_size;
    
    auto output = torch::empty_like(input);
    
    if (!training) {
        // Inference mode - fully parallel
        const int block_size = 256;
        const int num_blocks = (total_elements + block_size - 1) / block_size;
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batchnorm2d_forward_inference_cuda", [&]() {
            batchnorm2d_forward_inference_kernel<scalar_t><<<num_blocks, block_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                running_mean.data_ptr<scalar_t>(),
                running_var.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<scalar_t>() : nullptr,
                bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
                scalar_t(eps),
                batch_size,
                num_features,
                spatial_size,
                total_elements);
        });
    } else {
        // Training mode - optimized 3-step approach
        auto batch_mean = torch::empty(num_features, input.options());
        auto batch_var = torch::empty(num_features, input.options());
        
        // Step 1: Compute mean and variance per feature
        const int threads_per_feature = 256;
        const size_t shared_mem_size = 2 * threads_per_feature * sizeof(float);
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "compute_mean_var_cuda", [&]() {
            compute_mean_var_kernel<scalar_t><<<num_features, threads_per_feature, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                batch_mean.data_ptr<scalar_t>(),
                batch_var.data_ptr<scalar_t>(),
                batch_size,
                num_features,
                spatial_size);
        });
        
        // Step 2: Apply batch normalization
        const int block_size = 256;
        const int num_blocks = (total_elements + block_size - 1) / block_size;
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "apply_batchnorm_cuda", [&]() {
            apply_batchnorm_kernel<scalar_t><<<num_blocks, block_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_mean.data_ptr<scalar_t>(),
                batch_var.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<scalar_t>() : nullptr,
                bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
                scalar_t(eps),
                batch_size,
                num_features,
                spatial_size,
                total_elements);
        });
        
        // Step 3: Update running statistics
        const int update_block_size = 256;
        const int update_blocks = (num_features + update_block_size - 1) / update_block_size;
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "update_running_stats_cuda", [&]() {
            update_running_stats_kernel<scalar_t><<<update_blocks, update_block_size>>>(
                running_mean.data_ptr<scalar_t>(),
                running_var.data_ptr<scalar_t>(),
                batch_mean.data_ptr<scalar_t>(),
                batch_var.data_ptr<scalar_t>(),
                scalar_t(momentum),
                num_features);
        });
    }
    
    cudaDeviceSynchronize();
    return output;
}
"""

batchnorm_cpp_source = """
torch::Tensor batchnorm2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps,
    bool training,
    float momentum);
"""

# Compile the inline CUDA code
custom_batchnorm = load_inline(
    name="custom_batchnorm",
    cpp_sources=batchnorm_cpp_source,
    cuda_sources=batchnorm_source,
    functions=["batchnorm2d_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    with_cuda=True,
)

class CustomBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 0, momentum: float = 0.1):
        super(CustomBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        self.batchnorm_op = custom_batchnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.batchnorm_op.batchnorm2d_forward_cuda(
            x, 
            self.running_mean, 
            self.running_var,
            self.weight, 
            self.bias, 
            self.eps,
            self.training,
            self.momentum
        )

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.bn = CustomBatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer4_training, 策略=None, 种子=342, 模式=training_stress

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]
```

---

## 14. `L1_P34__const_perturb__15`

- **算子**: `const_perturb` (Category A)
- **描述**: Perturb numeric literals (integer ±1; float ×1.01 or ×0.99) @ L257
- **变异行**: Line 257, 原始片段 `5`, 节点类型 `const:int-1`
- **杀死层**: `layer1_value`
- **杀死策略**: `structured_ramp`
- **杀死种子**: `72`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused InstanceNorm2d
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__global__ void instance_norm_forward_kernel(
    const T* input,
    T* output,
    const int batch_size,
    const int num_features,
    const int height,
    const int width,
    const float eps
) {
    const int feature_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    const int spatial_size = height * width;
    const int feature_offset = feature_idx * spatial_size;
    const int batch_offset = batch_idx * num_features * spatial_size;
    const int start_idx = batch_offset + feature_offset;
    
    // Shared memory for reduction
    __shared__ float shared_sum[256];
    __shared__ float shared_sq_sum[256];
    
    // First pass: compute mean
    float sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        sum += static_cast<float>(input[idx]);
    }
    
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Reduction to compute mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / spatial_size;
    __syncthreads();
    
    // Second pass: compute variance
    float sq_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float diff = static_cast<float>(input[idx]) - mean;
        sq_sum += diff * diff;
    }
    
    shared_sq_sum[tid] = sq_sum;
    __syncthreads();
    
    // Reduction to compute variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_sq_sum[0] / spatial_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Third pass: normalize
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float normalized = (static_cast<float>(input[idx]) - mean) * inv_std;
        output[idx] = static_cast<T>(normalized);
    }
}

template<typename T>
__global__ void instance_norm_forward_kernel_with_affine(
    const T* input,
    T* output,
    const T* weight,
    const T* bias,
    const int batch_size,
    const int num_features,
    const int height,
    const int width,
    const float eps
) {
    const int feature_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    const int spatial_size = height * width;
    const int feature_offset = feature_idx * spatial_size;
    const int batch_offset = batch_idx * num_features * spatial_size;
    const int start_idx = batch_offset + feature_offset;
    
    // Shared memory for reduction
    __shared__ float shared_sum[256];
    __shared__ float shared_sq_sum[256];
    
    // First pass: compute mean
    float sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        sum += static_cast<float>(input[idx]);
    }
    
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Reduction to compute mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / spatial_size;
    __syncthreads();
    
    // Second pass: compute variance
    float sq_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float diff = static_cast<float>(input[idx]) - mean;
        sq_sum += diff * diff;
    }
    
    shared_sq_sum[tid] = sq_sum;
    __syncthreads();
    
    // Reduction to compute variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_sq_sum[0] / spatial_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Apply affine transformation parameters
    float w = static_cast<float>(weight[feature_idx]);
    float b = static_cast<float>(bias[feature_idx]);
    
    // Third pass: normalize and apply affine transform
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float normalized = (static_cast<float>(input[idx]) - mean) * inv_std;
        output[idx] = static_cast<T>(normalized * w + b);
    }
}

torch::Tensor instance_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    const int threads_per_block = 256;
    dim3 grid_size(num_features, batch_size);
    
    if (input.scalar_type() == torch::kFloat32) {
        if (weight.defined() && bias.defined()) {
            instance_norm_forward_kernel_with_affine<float><<<grid_size, threads_per_block>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                weight.data_ptr<float>(),
                bias.data_ptr<float>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        } else {
            instance_norm_forward_kernel<float><<<grid_size, threads_per_block>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        }
    } else if (input.scalar_type() == torch::kFloat16) {
        if (weight.defined() && bias.defined()) {
            instance_norm_forward_kernel_with_affine<at::Half><<<grid_size, threads_per_block>>>(
                input.data_ptr<at::Half>(),
                output.data_ptr<at::Half>(),
                weight.data_ptr<at::Half>(),
                bias.data_ptr<at::Half>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        } else {
            instance_norm_forward_kernel<at::Half><<<grid_size, threads_per_block>>>(
                input.data_ptr<at::Half>(),
                output.data_ptr<at::Half>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        }
    }
    
    return output;
}
"""

instance_norm_cpp_source = """
torch::Tensor instance_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
);
"""

# Compile the inline CUDA code
instance_norm_cuda = load_inline(
    name="instance_norm_cuda",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class InstanceNorm2dCustom(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super(InstanceNorm2dCustom, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return instance_norm_cuda.instance_norm_forward_cuda(
            x,
            self.weight if self.affine else torch.Tensor(),
            self.bias if self.affine else torch.Tensor(),
            self.eps
        )

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.inorm = InstanceNorm2dCustom(num_features=num_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inorm(x)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused InstanceNorm2d
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__global__ void instance_norm_forward_kernel(
    const T* input,
    T* output,
    const int batch_size,
    const int num_features,
    const int height,
    const int width,
    const float eps
) {
    const int feature_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    const int spatial_size = height * width;
    const int feature_offset = feature_idx * spatial_size;
    const int batch_offset = batch_idx * num_features * spatial_size;
    const int start_idx = batch_offset + feature_offset;
    
    // Shared memory for reduction
    __shared__ float shared_sum[256];
    __shared__ float shared_sq_sum[256];
    
    // First pass: compute mean
    float sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        sum += static_cast<float>(input[idx]);
    }
    
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Reduction to compute mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / spatial_size;
    __syncthreads();
    
    // Second pass: compute variance
    float sq_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float diff = static_cast<float>(input[idx]) - mean;
        sq_sum += diff * diff;
    }
    
    shared_sq_sum[tid] = sq_sum;
    __syncthreads();
    
    // Reduction to compute variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_sq_sum[0] / spatial_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Third pass: normalize
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float normalized = (static_cast<float>(input[idx]) - mean) * inv_std;
        output[idx] = static_cast<T>(normalized);
    }
}

template<typename T>
__global__ void instance_norm_forward_kernel_with_affine(
    const T* input,
    T* output,
    const T* weight,
    const T* bias,
    const int batch_size,
    const int num_features,
    const int height,
    const int width,
    const float eps
) {
    const int feature_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    const int spatial_size = height * width;
    const int feature_offset = feature_idx * spatial_size;
    const int batch_offset = batch_idx * num_features * spatial_size;
    const int start_idx = batch_offset + feature_offset;
    
    // Shared memory for reduction
    __shared__ float shared_sum[256];
    __shared__ float shared_sq_sum[256];
    
    // First pass: compute mean
    float sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        sum += static_cast<float>(input[idx]);
    }
    
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Reduction to compute mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / spatial_size;
    __syncthreads();
    
    // Second pass: compute variance
    float sq_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float diff = static_cast<float>(input[idx]) - mean;
        sq_sum += diff * diff;
    }
    
    shared_sq_sum[tid] = sq_sum;
    __syncthreads();
    
    // Reduction to compute variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_sq_sum[0] / spatial_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Apply affine transformation parameters
    float w = static_cast<float>(weight[feature_idx]);
    float b = static_cast<float>(bias[feature_idx]);
    
    // Third pass: normalize and apply affine transform
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float normalized = (static_cast<float>(input[idx]) - mean) * inv_std;
        output[idx] = static_cast<T>(normalized * w + b);
    }
}

torch::Tensor instance_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    const int threads_per_block = 256;
    dim3 grid_size(num_features, batch_size);
    
    if (input.scalar_type() == torch::kFloat32) {
        if (weight.defined() && bias.defined()) {
            instance_norm_forward_kernel_with_affine<float><<<grid_size, threads_per_block>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                weight.data_ptr<float>(),
                bias.data_ptr<float>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        } else {
            instance_norm_forward_kernel<float><<<grid_size, threads_per_block>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        }
    } else if (input.scalar_type() == torch::kFloat16) {
        if (weight.defined() && bias.defined()) {
            instance_norm_forward_kernel_with_affine<at::Half><<<grid_size, threads_per_block>>>(
                input.data_ptr<at::Half>(),
                output.data_ptr<at::Half>(),
                weight.data_ptr<at::Half>(),
                bias.data_ptr<at::Half>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        } else {
            instance_norm_forward_kernel<at::Half><<<grid_size, threads_per_block>>>(
                input.data_ptr<at::Half>(),
                output.data_ptr<at::Half>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        }
    }
    
    return output;
}
"""

instance_norm_cpp_source = """
torch::Tensor instance_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
);
"""

# Compile the inline CUDA code
instance_norm_cuda = load_inline(
    name="instance_norm_cuda",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class InstanceNorm2dCustom(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-4, affine: bool = True):
        super(InstanceNorm2dCustom, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return instance_norm_cuda.instance_norm_forward_cuda(
            x,
            self.weight if self.affine else torch.Tensor(),
            self.bias if self.affine else torch.Tensor(),
            self.eps
        )

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.inorm = InstanceNorm2dCustom(num_features=num_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inorm(x)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=structured_ramp, 种子=72, 模式=value_stress

**杀死策略描述**: linearly increasing 0→N

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]
```

---

## 15. `L1_P34__epsilon_modify__0`

- **算子**: `epsilon_modify` (Category C)
- **描述**: Alter small epsilon literals (LayerNorm / safe_div / log stability) @ L257
- **变异行**: Line 257, 原始片段 `1e-5`, 节点类型 `eps:to_1e-2`
- **杀死层**: `layer1_value`
- **杀死策略**: `structured_ramp`
- **杀死种子**: `72`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused InstanceNorm2d
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__global__ void instance_norm_forward_kernel(
    const T* input,
    T* output,
    const int batch_size,
    const int num_features,
    const int height,
    const int width,
    const float eps
) {
    const int feature_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    const int spatial_size = height * width;
    const int feature_offset = feature_idx * spatial_size;
    const int batch_offset = batch_idx * num_features * spatial_size;
    const int start_idx = batch_offset + feature_offset;
    
    // Shared memory for reduction
    __shared__ float shared_sum[256];
    __shared__ float shared_sq_sum[256];
    
    // First pass: compute mean
    float sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        sum += static_cast<float>(input[idx]);
    }
    
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Reduction to compute mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / spatial_size;
    __syncthreads();
    
    // Second pass: compute variance
    float sq_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float diff = static_cast<float>(input[idx]) - mean;
        sq_sum += diff * diff;
    }
    
    shared_sq_sum[tid] = sq_sum;
    __syncthreads();
    
    // Reduction to compute variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_sq_sum[0] / spatial_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Third pass: normalize
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float normalized = (static_cast<float>(input[idx]) - mean) * inv_std;
        output[idx] = static_cast<T>(normalized);
    }
}

template<typename T>
__global__ void instance_norm_forward_kernel_with_affine(
    const T* input,
    T* output,
    const T* weight,
    const T* bias,
    const int batch_size,
    const int num_features,
    const int height,
    const int width,
    const float eps
) {
    const int feature_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    const int spatial_size = height * width;
    const int feature_offset = feature_idx * spatial_size;
    const int batch_offset = batch_idx * num_features * spatial_size;
    const int start_idx = batch_offset + feature_offset;
    
    // Shared memory for reduction
    __shared__ float shared_sum[256];
    __shared__ float shared_sq_sum[256];
    
    // First pass: compute mean
    float sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        sum += static_cast<float>(input[idx]);
    }
    
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Reduction to compute mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / spatial_size;
    __syncthreads();
    
    // Second pass: compute variance
    float sq_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float diff = static_cast<float>(input[idx]) - mean;
        sq_sum += diff * diff;
    }
    
    shared_sq_sum[tid] = sq_sum;
    __syncthreads();
    
    // Reduction to compute variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_sq_sum[0] / spatial_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Apply affine transformation parameters
    float w = static_cast<float>(weight[feature_idx]);
    float b = static_cast<float>(bias[feature_idx]);
    
    // Third pass: normalize and apply affine transform
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float normalized = (static_cast<float>(input[idx]) - mean) * inv_std;
        output[idx] = static_cast<T>(normalized * w + b);
    }
}

torch::Tensor instance_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    const int threads_per_block = 256;
    dim3 grid_size(num_features, batch_size);
    
    if (input.scalar_type() == torch::kFloat32) {
        if (weight.defined() && bias.defined()) {
            instance_norm_forward_kernel_with_affine<float><<<grid_size, threads_per_block>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                weight.data_ptr<float>(),
                bias.data_ptr<float>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        } else {
            instance_norm_forward_kernel<float><<<grid_size, threads_per_block>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        }
    } else if (input.scalar_type() == torch::kFloat16) {
        if (weight.defined() && bias.defined()) {
            instance_norm_forward_kernel_with_affine<at::Half><<<grid_size, threads_per_block>>>(
                input.data_ptr<at::Half>(),
                output.data_ptr<at::Half>(),
                weight.data_ptr<at::Half>(),
                bias.data_ptr<at::Half>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        } else {
            instance_norm_forward_kernel<at::Half><<<grid_size, threads_per_block>>>(
                input.data_ptr<at::Half>(),
                output.data_ptr<at::Half>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        }
    }
    
    return output;
}
"""

instance_norm_cpp_source = """
torch::Tensor instance_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
);
"""

# Compile the inline CUDA code
instance_norm_cuda = load_inline(
    name="instance_norm_cuda",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class InstanceNorm2dCustom(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super(InstanceNorm2dCustom, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return instance_norm_cuda.instance_norm_forward_cuda(
            x,
            self.weight if self.affine else torch.Tensor(),
            self.bias if self.affine else torch.Tensor(),
            self.eps
        )

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.inorm = InstanceNorm2dCustom(num_features=num_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inorm(x)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused InstanceNorm2d
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__global__ void instance_norm_forward_kernel(
    const T* input,
    T* output,
    const int batch_size,
    const int num_features,
    const int height,
    const int width,
    const float eps
) {
    const int feature_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    const int spatial_size = height * width;
    const int feature_offset = feature_idx * spatial_size;
    const int batch_offset = batch_idx * num_features * spatial_size;
    const int start_idx = batch_offset + feature_offset;
    
    // Shared memory for reduction
    __shared__ float shared_sum[256];
    __shared__ float shared_sq_sum[256];
    
    // First pass: compute mean
    float sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        sum += static_cast<float>(input[idx]);
    }
    
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Reduction to compute mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / spatial_size;
    __syncthreads();
    
    // Second pass: compute variance
    float sq_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float diff = static_cast<float>(input[idx]) - mean;
        sq_sum += diff * diff;
    }
    
    shared_sq_sum[tid] = sq_sum;
    __syncthreads();
    
    // Reduction to compute variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_sq_sum[0] / spatial_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Third pass: normalize
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float normalized = (static_cast<float>(input[idx]) - mean) * inv_std;
        output[idx] = static_cast<T>(normalized);
    }
}

template<typename T>
__global__ void instance_norm_forward_kernel_with_affine(
    const T* input,
    T* output,
    const T* weight,
    const T* bias,
    const int batch_size,
    const int num_features,
    const int height,
    const int width,
    const float eps
) {
    const int feature_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    const int spatial_size = height * width;
    const int feature_offset = feature_idx * spatial_size;
    const int batch_offset = batch_idx * num_features * spatial_size;
    const int start_idx = batch_offset + feature_offset;
    
    // Shared memory for reduction
    __shared__ float shared_sum[256];
    __shared__ float shared_sq_sum[256];
    
    // First pass: compute mean
    float sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        sum += static_cast<float>(input[idx]);
    }
    
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Reduction to compute mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / spatial_size;
    __syncthreads();
    
    // Second pass: compute variance
    float sq_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float diff = static_cast<float>(input[idx]) - mean;
        sq_sum += diff * diff;
    }
    
    shared_sq_sum[tid] = sq_sum;
    __syncthreads();
    
    // Reduction to compute variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_sq_sum[0] / spatial_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Apply affine transformation parameters
    float w = static_cast<float>(weight[feature_idx]);
    float b = static_cast<float>(bias[feature_idx]);
    
    // Third pass: normalize and apply affine transform
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float normalized = (static_cast<float>(input[idx]) - mean) * inv_std;
        output[idx] = static_cast<T>(normalized * w + b);
    }
}

torch::Tensor instance_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    const int threads_per_block = 256;
    dim3 grid_size(num_features, batch_size);
    
    if (input.scalar_type() == torch::kFloat32) {
        if (weight.defined() && bias.defined()) {
            instance_norm_forward_kernel_with_affine<float><<<grid_size, threads_per_block>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                weight.data_ptr<float>(),
                bias.data_ptr<float>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        } else {
            instance_norm_forward_kernel<float><<<grid_size, threads_per_block>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        }
    } else if (input.scalar_type() == torch::kFloat16) {
        if (weight.defined() && bias.defined()) {
            instance_norm_forward_kernel_with_affine<at::Half><<<grid_size, threads_per_block>>>(
                input.data_ptr<at::Half>(),
                output.data_ptr<at::Half>(),
                weight.data_ptr<at::Half>(),
                bias.data_ptr<at::Half>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        } else {
            instance_norm_forward_kernel<at::Half><<<grid_size, threads_per_block>>>(
                input.data_ptr<at::Half>(),
                output.data_ptr<at::Half>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        }
    }
    
    return output;
}
"""

instance_norm_cpp_source = """
torch::Tensor instance_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
);
"""

# Compile the inline CUDA code
instance_norm_cuda = load_inline(
    name="instance_norm_cuda",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class InstanceNorm2dCustom(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-2, affine: bool = True):
        super(InstanceNorm2dCustom, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return instance_norm_cuda.instance_norm_forward_cuda(
            x,
            self.weight if self.affine else torch.Tensor(),
            self.bias if self.affine else torch.Tensor(),
            self.eps
        )

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.inorm = InstanceNorm2dCustom(num_features=num_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inorm(x)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=structured_ramp, 种子=72, 模式=value_stress

**杀死策略描述**: linearly increasing 0→N

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]
```

---

## 16. `L1_P34__epsilon_modify__1`

- **算子**: `epsilon_modify` (Category C)
- **描述**: Alter small epsilon literals (LayerNorm / safe_div / log stability) @ L257
- **变异行**: Line 257, 原始片段 `1e-5`, 节点类型 `eps:to_zero`
- **杀死层**: `layer1_value`
- **杀死策略**: `near_zero`
- **杀死种子**: `42`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused InstanceNorm2d
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__global__ void instance_norm_forward_kernel(
    const T* input,
    T* output,
    const int batch_size,
    const int num_features,
    const int height,
    const int width,
    const float eps
) {
    const int feature_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    const int spatial_size = height * width;
    const int feature_offset = feature_idx * spatial_size;
    const int batch_offset = batch_idx * num_features * spatial_size;
    const int start_idx = batch_offset + feature_offset;
    
    // Shared memory for reduction
    __shared__ float shared_sum[256];
    __shared__ float shared_sq_sum[256];
    
    // First pass: compute mean
    float sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        sum += static_cast<float>(input[idx]);
    }
    
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Reduction to compute mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / spatial_size;
    __syncthreads();
    
    // Second pass: compute variance
    float sq_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float diff = static_cast<float>(input[idx]) - mean;
        sq_sum += diff * diff;
    }
    
    shared_sq_sum[tid] = sq_sum;
    __syncthreads();
    
    // Reduction to compute variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_sq_sum[0] / spatial_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Third pass: normalize
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float normalized = (static_cast<float>(input[idx]) - mean) * inv_std;
        output[idx] = static_cast<T>(normalized);
    }
}

template<typename T>
__global__ void instance_norm_forward_kernel_with_affine(
    const T* input,
    T* output,
    const T* weight,
    const T* bias,
    const int batch_size,
    const int num_features,
    const int height,
    const int width,
    const float eps
) {
    const int feature_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    const int spatial_size = height * width;
    const int feature_offset = feature_idx * spatial_size;
    const int batch_offset = batch_idx * num_features * spatial_size;
    const int start_idx = batch_offset + feature_offset;
    
    // Shared memory for reduction
    __shared__ float shared_sum[256];
    __shared__ float shared_sq_sum[256];
    
    // First pass: compute mean
    float sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        sum += static_cast<float>(input[idx]);
    }
    
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Reduction to compute mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / spatial_size;
    __syncthreads();
    
    // Second pass: compute variance
    float sq_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float diff = static_cast<float>(input[idx]) - mean;
        sq_sum += diff * diff;
    }
    
    shared_sq_sum[tid] = sq_sum;
    __syncthreads();
    
    // Reduction to compute variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_sq_sum[0] / spatial_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Apply affine transformation parameters
    float w = static_cast<float>(weight[feature_idx]);
    float b = static_cast<float>(bias[feature_idx]);
    
    // Third pass: normalize and apply affine transform
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float normalized = (static_cast<float>(input[idx]) - mean) * inv_std;
        output[idx] = static_cast<T>(normalized * w + b);
    }
}

torch::Tensor instance_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    const int threads_per_block = 256;
    dim3 grid_size(num_features, batch_size);
    
    if (input.scalar_type() == torch::kFloat32) {
        if (weight.defined() && bias.defined()) {
            instance_norm_forward_kernel_with_affine<float><<<grid_size, threads_per_block>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                weight.data_ptr<float>(),
                bias.data_ptr<float>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        } else {
            instance_norm_forward_kernel<float><<<grid_size, threads_per_block>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        }
    } else if (input.scalar_type() == torch::kFloat16) {
        if (weight.defined() && bias.defined()) {
            instance_norm_forward_kernel_with_affine<at::Half><<<grid_size, threads_per_block>>>(
                input.data_ptr<at::Half>(),
                output.data_ptr<at::Half>(),
                weight.data_ptr<at::Half>(),
                bias.data_ptr<at::Half>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        } else {
            instance_norm_forward_kernel<at::Half><<<grid_size, threads_per_block>>>(
                input.data_ptr<at::Half>(),
                output.data_ptr<at::Half>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        }
    }
    
    return output;
}
"""

instance_norm_cpp_source = """
torch::Tensor instance_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
);
"""

# Compile the inline CUDA code
instance_norm_cuda = load_inline(
    name="instance_norm_cuda",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class InstanceNorm2dCustom(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super(InstanceNorm2dCustom, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return instance_norm_cuda.instance_norm_forward_cuda(
            x,
            self.weight if self.affine else torch.Tensor(),
            self.bias if self.affine else torch.Tensor(),
            self.eps
        )

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.inorm = InstanceNorm2dCustom(num_features=num_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inorm(x)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused InstanceNorm2d
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__global__ void instance_norm_forward_kernel(
    const T* input,
    T* output,
    const int batch_size,
    const int num_features,
    const int height,
    const int width,
    const float eps
) {
    const int feature_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    const int spatial_size = height * width;
    const int feature_offset = feature_idx * spatial_size;
    const int batch_offset = batch_idx * num_features * spatial_size;
    const int start_idx = batch_offset + feature_offset;
    
    // Shared memory for reduction
    __shared__ float shared_sum[256];
    __shared__ float shared_sq_sum[256];
    
    // First pass: compute mean
    float sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        sum += static_cast<float>(input[idx]);
    }
    
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Reduction to compute mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / spatial_size;
    __syncthreads();
    
    // Second pass: compute variance
    float sq_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float diff = static_cast<float>(input[idx]) - mean;
        sq_sum += diff * diff;
    }
    
    shared_sq_sum[tid] = sq_sum;
    __syncthreads();
    
    // Reduction to compute variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_sq_sum[0] / spatial_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Third pass: normalize
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float normalized = (static_cast<float>(input[idx]) - mean) * inv_std;
        output[idx] = static_cast<T>(normalized);
    }
}

template<typename T>
__global__ void instance_norm_forward_kernel_with_affine(
    const T* input,
    T* output,
    const T* weight,
    const T* bias,
    const int batch_size,
    const int num_features,
    const int height,
    const int width,
    const float eps
) {
    const int feature_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    const int spatial_size = height * width;
    const int feature_offset = feature_idx * spatial_size;
    const int batch_offset = batch_idx * num_features * spatial_size;
    const int start_idx = batch_offset + feature_offset;
    
    // Shared memory for reduction
    __shared__ float shared_sum[256];
    __shared__ float shared_sq_sum[256];
    
    // First pass: compute mean
    float sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        sum += static_cast<float>(input[idx]);
    }
    
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Reduction to compute mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / spatial_size;
    __syncthreads();
    
    // Second pass: compute variance
    float sq_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float diff = static_cast<float>(input[idx]) - mean;
        sq_sum += diff * diff;
    }
    
    shared_sq_sum[tid] = sq_sum;
    __syncthreads();
    
    // Reduction to compute variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_sq_sum[0] / spatial_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Apply affine transformation parameters
    float w = static_cast<float>(weight[feature_idx]);
    float b = static_cast<float>(bias[feature_idx]);
    
    // Third pass: normalize and apply affine transform
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float normalized = (static_cast<float>(input[idx]) - mean) * inv_std;
        output[idx] = static_cast<T>(normalized * w + b);
    }
}

torch::Tensor instance_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    const int threads_per_block = 256;
    dim3 grid_size(num_features, batch_size);
    
    if (input.scalar_type() == torch::kFloat32) {
        if (weight.defined() && bias.defined()) {
            instance_norm_forward_kernel_with_affine<float><<<grid_size, threads_per_block>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                weight.data_ptr<float>(),
                bias.data_ptr<float>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        } else {
            instance_norm_forward_kernel<float><<<grid_size, threads_per_block>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        }
    } else if (input.scalar_type() == torch::kFloat16) {
        if (weight.defined() && bias.defined()) {
            instance_norm_forward_kernel_with_affine<at::Half><<<grid_size, threads_per_block>>>(
                input.data_ptr<at::Half>(),
                output.data_ptr<at::Half>(),
                weight.data_ptr<at::Half>(),
                bias.data_ptr<at::Half>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        } else {
            instance_norm_forward_kernel<at::Half><<<grid_size, threads_per_block>>>(
                input.data_ptr<at::Half>(),
                output.data_ptr<at::Half>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        }
    }
    
    return output;
}
"""

instance_norm_cpp_source = """
torch::Tensor instance_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
);
"""

# Compile the inline CUDA code
instance_norm_cuda = load_inline(
    name="instance_norm_cuda",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class InstanceNorm2dCustom(nn.Module):
    def __init__(self, num_features: int, eps: float = 0, affine: bool = True):
        super(InstanceNorm2dCustom, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return instance_norm_cuda.instance_norm_forward_cuda(
            x,
            self.weight if self.affine else torch.Tensor(),
            self.bias if self.affine else torch.Tensor(),
            self.eps
        )

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.inorm = InstanceNorm2dCustom(num_features=num_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inorm(x)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=near_zero, 种子=42, 模式=value_stress

**杀死策略描述**: values ~1e-7

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]
```

---

## 17. `L1_P34__mask_boundary__2`

- **算子**: `mask_boundary` (Category B)
- **描述**: Weaken or tighten boundary checks in mask/tl.where (Triton) or thread/block guards (CUDA) @ L123
- **变异行**: Line 123, 原始片段 `tid < stride`, 节点类型 `rhs-1`
- **杀死层**: `layer1_value`
- **杀死策略**: `sparse`
- **杀死种子**: `45`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused InstanceNorm2d
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__global__ void instance_norm_forward_kernel(
    const T* input,
    T* output,
    const int batch_size,
    const int num_features,
    const int height,
    const int width,
    const float eps
) {
    const int feature_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    const int spatial_size = height * width;
    const int feature_offset = feature_idx * spatial_size;
    const int batch_offset = batch_idx * num_features * spatial_size;
    const int start_idx = batch_offset + feature_offset;
    
    // Shared memory for reduction
    __shared__ float shared_sum[256];
    __shared__ float shared_sq_sum[256];
    
    // First pass: compute mean
    float sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        sum += static_cast<float>(input[idx]);
    }
    
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Reduction to compute mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / spatial_size;
    __syncthreads();
    
    // Second pass: compute variance
    float sq_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float diff = static_cast<float>(input[idx]) - mean;
        sq_sum += diff * diff;
    }
    
    shared_sq_sum[tid] = sq_sum;
    __syncthreads();
    
    // Reduction to compute variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_sq_sum[0] / spatial_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Third pass: normalize
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float normalized = (static_cast<float>(input[idx]) - mean) * inv_std;
        output[idx] = static_cast<T>(normalized);
    }
}

template<typename T>
__global__ void instance_norm_forward_kernel_with_affine(
    const T* input,
    T* output,
    const T* weight,
    const T* bias,
    const int batch_size,
    const int num_features,
    const int height,
    const int width,
    const float eps
) {
    const int feature_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    const int spatial_size = height * width;
    const int feature_offset = feature_idx * spatial_size;
    const int batch_offset = batch_idx * num_features * spatial_size;
    const int start_idx = batch_offset + feature_offset;
    
    // Shared memory for reduction
    __shared__ float shared_sum[256];
    __shared__ float shared_sq_sum[256];
    
    // First pass: compute mean
    float sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        sum += static_cast<float>(input[idx]);
    }
    
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Reduction to compute mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / spatial_size;
    __syncthreads();
    
    // Second pass: compute variance
    float sq_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float diff = static_cast<float>(input[idx]) - mean;
        sq_sum += diff * diff;
    }
    
    shared_sq_sum[tid] = sq_sum;
    __syncthreads();
    
    // Reduction to compute variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_sq_sum[0] / spatial_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Apply affine transformation parameters
    float w = static_cast<float>(weight[feature_idx]);
    float b = static_cast<float>(bias[feature_idx]);
    
    // Third pass: normalize and apply affine transform
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float normalized = (static_cast<float>(input[idx]) - mean) * inv_std;
        output[idx] = static_cast<T>(normalized * w + b);
    }
}

torch::Tensor instance_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    const int threads_per_block = 256;
    dim3 grid_size(num_features, batch_size);
    
    if (input.scalar_type() == torch::kFloat32) {
        if (weight.defined() && bias.defined()) {
            instance_norm_forward_kernel_with_affine<float><<<grid_size, threads_per_block>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                weight.data_ptr<float>(),
                bias.data_ptr<float>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        } else {
            instance_norm_forward_kernel<float><<<grid_size, threads_per_block>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        }
    } else if (input.scalar_type() == torch::kFloat16) {
        if (weight.defined() && bias.defined()) {
            instance_norm_forward_kernel_with_affine<at::Half><<<grid_size, threads_per_block>>>(
                input.data_ptr<at::Half>(),
                output.data_ptr<at::Half>(),
                weight.data_ptr<at::Half>(),
                bias.data_ptr<at::Half>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        } else {
            instance_norm_forward_kernel<at::Half><<<grid_size, threads_per_block>>>(
                input.data_ptr<at::Half>(),
                output.data_ptr<at::Half>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        }
    }
    
    return output;
}
"""

instance_norm_cpp_source = """
torch::Tensor instance_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
);
"""

# Compile the inline CUDA code
instance_norm_cuda = load_inline(
    name="instance_norm_cuda",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class InstanceNorm2dCustom(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super(InstanceNorm2dCustom, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return instance_norm_cuda.instance_norm_forward_cuda(
            x,
            self.weight if self.affine else torch.Tensor(),
            self.bias if self.affine else torch.Tensor(),
            self.eps
        )

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.inorm = InstanceNorm2dCustom(num_features=num_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inorm(x)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused InstanceNorm2d
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__global__ void instance_norm_forward_kernel(
    const T* input,
    T* output,
    const int batch_size,
    const int num_features,
    const int height,
    const int width,
    const float eps
) {
    const int feature_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    const int spatial_size = height * width;
    const int feature_offset = feature_idx * spatial_size;
    const int batch_offset = batch_idx * num_features * spatial_size;
    const int start_idx = batch_offset + feature_offset;
    
    // Shared memory for reduction
    __shared__ float shared_sum[256];
    __shared__ float shared_sq_sum[256];
    
    // First pass: compute mean
    float sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        sum += static_cast<float>(input[idx]);
    }
    
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Reduction to compute mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / spatial_size;
    __syncthreads();
    
    // Second pass: compute variance
    float sq_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float diff = static_cast<float>(input[idx]) - mean;
        sq_sum += diff * diff;
    }
    
    shared_sq_sum[tid] = sq_sum;
    __syncthreads();
    
    // Reduction to compute variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_sq_sum[0] / spatial_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Third pass: normalize
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float normalized = (static_cast<float>(input[idx]) - mean) * inv_std;
        output[idx] = static_cast<T>(normalized);
    }
}

template<typename T>
__global__ void instance_norm_forward_kernel_with_affine(
    const T* input,
    T* output,
    const T* weight,
    const T* bias,
    const int batch_size,
    const int num_features,
    const int height,
    const int width,
    const float eps
) {
    const int feature_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    const int spatial_size = height * width;
    const int feature_offset = feature_idx * spatial_size;
    const int batch_offset = batch_idx * num_features * spatial_size;
    const int start_idx = batch_offset + feature_offset;
    
    // Shared memory for reduction
    __shared__ float shared_sum[256];
    __shared__ float shared_sq_sum[256];
    
    // First pass: compute mean
    float sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        sum += static_cast<float>(input[idx]);
    }
    
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Reduction to compute mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride - 1) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / spatial_size;
    __syncthreads();
    
    // Second pass: compute variance
    float sq_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float diff = static_cast<float>(input[idx]) - mean;
        sq_sum += diff * diff;
    }
    
    shared_sq_sum[tid] = sq_sum;
    __syncthreads();
    
    // Reduction to compute variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_sq_sum[0] / spatial_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Apply affine transformation parameters
    float w = static_cast<float>(weight[feature_idx]);
    float b = static_cast<float>(bias[feature_idx]);
    
    // Third pass: normalize and apply affine transform
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = start_idx + i;
        float normalized = (static_cast<float>(input[idx]) - mean) * inv_std;
        output[idx] = static_cast<T>(normalized * w + b);
    }
}

torch::Tensor instance_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    const int threads_per_block = 256;
    dim3 grid_size(num_features, batch_size);
    
    if (input.scalar_type() == torch::kFloat32) {
        if (weight.defined() && bias.defined()) {
            instance_norm_forward_kernel_with_affine<float><<<grid_size, threads_per_block>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                weight.data_ptr<float>(),
                bias.data_ptr<float>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        } else {
            instance_norm_forward_kernel<float><<<grid_size, threads_per_block>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        }
    } else if (input.scalar_type() == torch::kFloat16) {
        if (weight.defined() && bias.defined()) {
            instance_norm_forward_kernel_with_affine<at::Half><<<grid_size, threads_per_block>>>(
                input.data_ptr<at::Half>(),
                output.data_ptr<at::Half>(),
                weight.data_ptr<at::Half>(),
                bias.data_ptr<at::Half>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        } else {
            instance_norm_forward_kernel<at::Half><<<grid_size, threads_per_block>>>(
                input.data_ptr<at::Half>(),
                output.data_ptr<at::Half>(),
                batch_size,
                num_features,
                height,
                width,
                eps
            );
        }
    }
    
    return output;
}
"""

instance_norm_cpp_source = """
torch::Tensor instance_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
);
"""

# Compile the inline CUDA code
instance_norm_cuda = load_inline(
    name="instance_norm_cuda",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class InstanceNorm2dCustom(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super(InstanceNorm2dCustom, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return instance_norm_cuda.instance_norm_forward_cuda(
            x,
            self.weight if self.affine else torch.Tensor(),
            self.bias if self.affine else torch.Tensor(),
            self.eps
        )

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.inorm = InstanceNorm2dCustom(num_features=num_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inorm(x)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=sparse, 种子=45, 模式=value_stress

**杀死策略描述**: 90% zeros, 10% random

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]
```

---

## 18. `L1_P35__arith_replace__16`

- **算子**: `arith_replace` (Category A)
- **描述**: Replace binary arithmetic operators (+→-, -→+, *→/, /→*) @ L91
- **变异行**: Line 91, 原始片段 `*`, 节点类型 `cuda_Mult`
- **杀死层**: `layer1_value`
- **杀死策略**: `sparse`
- **杀死种子**: `66`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half from_float<half>(float val) {
    return __float2half(val);
}

template<typename T>
__global__ void group_norm_forward_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int batch_size,
    const int num_channels,
    const int num_groups,
    const int channels_per_group,
    const int spatial_size,
    const float eps
) {
    // Each block handles one group in one sample
    const int group_idx = blockIdx.x % num_groups;
    const int sample_idx = blockIdx.x / num_groups;
    
    if (sample_idx >= batch_size || group_idx >= num_groups) return;
    
    const int group_start = group_idx * channels_per_group;
    const int group_elements = channels_per_group * spatial_size;
    
    // Shared memory for reductions
    __shared__ float shared_data[256];
    
    // First pass: compute mean
    float thread_sum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        thread_sum += to_float(input[input_idx]);
    }
    
    // Block reduction for sum
    shared_data[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float mean = 0.0f;
    if (threadIdx.x == 0) {
        mean = shared_data[0] / group_elements;
        shared_data[0] = mean;
    }
    __syncthreads();
    
    mean = shared_data[0];
    
    // Second pass: compute variance
    float thread_sqsum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        float val = to_float(input[input_idx]);
        float diff = val - mean;
        thread_sqsum += diff * diff;
    }
    
    // Block reduction for variance
    shared_data[threadIdx.x] = thread_sqsum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float inv_std = 0.0f;
    if (threadIdx.x == 0) {
        float var = shared_data[0] / group_elements;
        inv_std = rsqrtf(var + eps);
        shared_data[0] = inv_std;
    }
    __syncthreads();
    
    inv_std = shared_data[0];
    
    // Apply normalization with affine transformation
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int output_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        
        float val = to_float(input[output_idx]);
        float normalized = (val - mean) * inv_std;
        
        if (weight && bias) {
            normalized = normalized * weight[c] + bias[c];
        }
        
        output[output_idx] = from_float<T>(normalized);
    }
}

torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
) {
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    
    // Calculate spatial size
    int spatial_size = 1;
    for (int i = 2; i < input.dim(); ++i) {
        spatial_size *= input.size(i);
    }
    
    auto channels_per_group = num_channels / num_groups;
    auto output = torch::empty_like(input);
    
    // Configure kernel launch
    const int threads_per_block = 256;
    dim3 blocks(batch_size * num_groups);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "group_norm_forward_cuda",
        ([&] {
            group_norm_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<float>() : nullptr,
                bias.defined() ? bias.data_ptr<float>() : nullptr,
                batch_size,
                num_channels,
                num_groups,
                channels_per_group,
                spatial_size,
                eps
            );
        })
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
);
"""

# Compile the inline CUDA code
group_norm_cuda = load_inline(
    name="group_norm_cuda",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization with custom CUDA kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the optimized GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Small epsilon for numerical stability
        self.eps = 1e-5
        
        # CUDA kernel
        self.group_norm_kernel = group_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Validate input
        if x.dim() < 2:
            raise ValueError(f"Expected at least 2D input (got {x.dim()}D input)")
        
        if x.size(1) != self.num_features:
            raise ValueError(
                f"Expected input channels {self.num_features}, got {x.size(1)} channels"
            )
        
        if self.num_features % self.num_groups != 0:
            raise ValueError(
                f"Number of channels ({self.num_features}) must be divisible by number of groups ({self.num_groups})"
            )
        
        # Use custom CUDA kernel
        return self.group_norm_kernel.group_norm_forward_cuda(
            x,
            self.weight,
            self.bias,
            self.num_groups,
            self.eps
        )
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half from_float<half>(float val) {
    return __float2half(val);
}

template<typename T>
__global__ void group_norm_forward_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int batch_size,
    const int num_channels,
    const int num_groups,
    const int channels_per_group,
    const int spatial_size,
    const float eps
) {
    // Each block handles one group in one sample
    const int group_idx = blockIdx.x % num_groups;
    const int sample_idx = blockIdx.x / num_groups;
    
    if (sample_idx >= batch_size || group_idx >= num_groups) return;
    
    const int group_start = group_idx * channels_per_group;
    const int group_elements = channels_per_group * spatial_size;
    
    // Shared memory for reductions
    __shared__ float shared_data[256];
    
    // First pass: compute mean
    float thread_sum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        thread_sum += to_float(input[input_idx]);
    }
    
    // Block reduction for sum
    shared_data[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float mean = 0.0f;
    if (threadIdx.x == 0) {
        mean = shared_data[0] / group_elements;
        shared_data[0] = mean;
    }
    __syncthreads();
    
    mean = shared_data[0];
    
    // Second pass: compute variance
    float thread_sqsum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx / num_channels + c) * spatial_size + s);
        float val = to_float(input[input_idx]);
        float diff = val - mean;
        thread_sqsum += diff * diff;
    }
    
    // Block reduction for variance
    shared_data[threadIdx.x] = thread_sqsum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float inv_std = 0.0f;
    if (threadIdx.x == 0) {
        float var = shared_data[0] / group_elements;
        inv_std = rsqrtf(var + eps);
        shared_data[0] = inv_std;
    }
    __syncthreads();
    
    inv_std = shared_data[0];
    
    // Apply normalization with affine transformation
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int output_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        
        float val = to_float(input[output_idx]);
        float normalized = (val - mean) * inv_std;
        
        if (weight && bias) {
            normalized = normalized * weight[c] + bias[c];
        }
        
        output[output_idx] = from_float<T>(normalized);
    }
}

torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
) {
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    
    // Calculate spatial size
    int spatial_size = 1;
    for (int i = 2; i < input.dim(); ++i) {
        spatial_size *= input.size(i);
    }
    
    auto channels_per_group = num_channels / num_groups;
    auto output = torch::empty_like(input);
    
    // Configure kernel launch
    const int threads_per_block = 256;
    dim3 blocks(batch_size * num_groups);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "group_norm_forward_cuda",
        ([&] {
            group_norm_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<float>() : nullptr,
                bias.defined() ? bias.data_ptr<float>() : nullptr,
                batch_size,
                num_channels,
                num_groups,
                channels_per_group,
                spatial_size,
                eps
            );
        })
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
);
"""

# Compile the inline CUDA code
group_norm_cuda = load_inline(
    name="group_norm_cuda",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization with custom CUDA kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the optimized GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Small epsilon for numerical stability
        self.eps = 1e-5
        
        # CUDA kernel
        self.group_norm_kernel = group_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Validate input
        if x.dim() < 2:
            raise ValueError(f"Expected at least 2D input (got {x.dim()}D input)")
        
        if x.size(1) != self.num_features:
            raise ValueError(
                f"Expected input channels {self.num_features}, got {x.size(1)} channels"
            )
        
        if self.num_features % self.num_groups != 0:
            raise ValueError(
                f"Number of channels ({self.num_features}) must be divisible by number of groups ({self.num_groups})"
            )
        
        # Use custom CUDA kernel
        return self.group_norm_kernel.group_norm_forward_cuda(
            x,
            self.weight,
            self.bias,
            self.num_groups,
            self.eps
        )
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=sparse, 种子=66, 模式=value_stress

**杀死策略描述**: 90% zeros, 10% random

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]
```

---

## 19. `L1_P35__arith_replace__20`

- **算子**: `arith_replace` (Category A)
- **描述**: Replace binary arithmetic operators (+→-, -→+, *→/, /→*) @ L93
- **变异行**: Line 93, 原始片段 `-`, 节点类型 `cuda_Sub`
- **杀死层**: `layer1_value`
- **杀死策略**: `all_negative`
- **杀死种子**: `57`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half from_float<half>(float val) {
    return __float2half(val);
}

template<typename T>
__global__ void group_norm_forward_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int batch_size,
    const int num_channels,
    const int num_groups,
    const int channels_per_group,
    const int spatial_size,
    const float eps
) {
    // Each block handles one group in one sample
    const int group_idx = blockIdx.x % num_groups;
    const int sample_idx = blockIdx.x / num_groups;
    
    if (sample_idx >= batch_size || group_idx >= num_groups) return;
    
    const int group_start = group_idx * channels_per_group;
    const int group_elements = channels_per_group * spatial_size;
    
    // Shared memory for reductions
    __shared__ float shared_data[256];
    
    // First pass: compute mean
    float thread_sum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        thread_sum += to_float(input[input_idx]);
    }
    
    // Block reduction for sum
    shared_data[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float mean = 0.0f;
    if (threadIdx.x == 0) {
        mean = shared_data[0] / group_elements;
        shared_data[0] = mean;
    }
    __syncthreads();
    
    mean = shared_data[0];
    
    // Second pass: compute variance
    float thread_sqsum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        float val = to_float(input[input_idx]);
        float diff = val - mean;
        thread_sqsum += diff * diff;
    }
    
    // Block reduction for variance
    shared_data[threadIdx.x] = thread_sqsum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float inv_std = 0.0f;
    if (threadIdx.x == 0) {
        float var = shared_data[0] / group_elements;
        inv_std = rsqrtf(var + eps);
        shared_data[0] = inv_std;
    }
    __syncthreads();
    
    inv_std = shared_data[0];
    
    // Apply normalization with affine transformation
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int output_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        
        float val = to_float(input[output_idx]);
        float normalized = (val - mean) * inv_std;
        
        if (weight && bias) {
            normalized = normalized * weight[c] + bias[c];
        }
        
        output[output_idx] = from_float<T>(normalized);
    }
}

torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
) {
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    
    // Calculate spatial size
    int spatial_size = 1;
    for (int i = 2; i < input.dim(); ++i) {
        spatial_size *= input.size(i);
    }
    
    auto channels_per_group = num_channels / num_groups;
    auto output = torch::empty_like(input);
    
    // Configure kernel launch
    const int threads_per_block = 256;
    dim3 blocks(batch_size * num_groups);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "group_norm_forward_cuda",
        ([&] {
            group_norm_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<float>() : nullptr,
                bias.defined() ? bias.data_ptr<float>() : nullptr,
                batch_size,
                num_channels,
                num_groups,
                channels_per_group,
                spatial_size,
                eps
            );
        })
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
);
"""

# Compile the inline CUDA code
group_norm_cuda = load_inline(
    name="group_norm_cuda",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization with custom CUDA kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the optimized GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Small epsilon for numerical stability
        self.eps = 1e-5
        
        # CUDA kernel
        self.group_norm_kernel = group_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Validate input
        if x.dim() < 2:
            raise ValueError(f"Expected at least 2D input (got {x.dim()}D input)")
        
        if x.size(1) != self.num_features:
            raise ValueError(
                f"Expected input channels {self.num_features}, got {x.size(1)} channels"
            )
        
        if self.num_features % self.num_groups != 0:
            raise ValueError(
                f"Number of channels ({self.num_features}) must be divisible by number of groups ({self.num_groups})"
            )
        
        # Use custom CUDA kernel
        return self.group_norm_kernel.group_norm_forward_cuda(
            x,
            self.weight,
            self.bias,
            self.num_groups,
            self.eps
        )
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half from_float<half>(float val) {
    return __float2half(val);
}

template<typename T>
__global__ void group_norm_forward_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int batch_size,
    const int num_channels,
    const int num_groups,
    const int channels_per_group,
    const int spatial_size,
    const float eps
) {
    // Each block handles one group in one sample
    const int group_idx = blockIdx.x % num_groups;
    const int sample_idx = blockIdx.x / num_groups;
    
    if (sample_idx >= batch_size || group_idx >= num_groups) return;
    
    const int group_start = group_idx * channels_per_group;
    const int group_elements = channels_per_group * spatial_size;
    
    // Shared memory for reductions
    __shared__ float shared_data[256];
    
    // First pass: compute mean
    float thread_sum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        thread_sum += to_float(input[input_idx]);
    }
    
    // Block reduction for sum
    shared_data[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float mean = 0.0f;
    if (threadIdx.x == 0) {
        mean = shared_data[0] / group_elements;
        shared_data[0] = mean;
    }
    __syncthreads();
    
    mean = shared_data[0];
    
    // Second pass: compute variance
    float thread_sqsum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        float val = to_float(input[input_idx]);
        float diff = val + mean;
        thread_sqsum += diff * diff;
    }
    
    // Block reduction for variance
    shared_data[threadIdx.x] = thread_sqsum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float inv_std = 0.0f;
    if (threadIdx.x == 0) {
        float var = shared_data[0] / group_elements;
        inv_std = rsqrtf(var + eps);
        shared_data[0] = inv_std;
    }
    __syncthreads();
    
    inv_std = shared_data[0];
    
    // Apply normalization with affine transformation
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int output_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        
        float val = to_float(input[output_idx]);
        float normalized = (val - mean) * inv_std;
        
        if (weight && bias) {
            normalized = normalized * weight[c] + bias[c];
        }
        
        output[output_idx] = from_float<T>(normalized);
    }
}

torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
) {
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    
    // Calculate spatial size
    int spatial_size = 1;
    for (int i = 2; i < input.dim(); ++i) {
        spatial_size *= input.size(i);
    }
    
    auto channels_per_group = num_channels / num_groups;
    auto output = torch::empty_like(input);
    
    // Configure kernel launch
    const int threads_per_block = 256;
    dim3 blocks(batch_size * num_groups);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "group_norm_forward_cuda",
        ([&] {
            group_norm_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<float>() : nullptr,
                bias.defined() ? bias.data_ptr<float>() : nullptr,
                batch_size,
                num_channels,
                num_groups,
                channels_per_group,
                spatial_size,
                eps
            );
        })
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
);
"""

# Compile the inline CUDA code
group_norm_cuda = load_inline(
    name="group_norm_cuda",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization with custom CUDA kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the optimized GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Small epsilon for numerical stability
        self.eps = 1e-5
        
        # CUDA kernel
        self.group_norm_kernel = group_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Validate input
        if x.dim() < 2:
            raise ValueError(f"Expected at least 2D input (got {x.dim()}D input)")
        
        if x.size(1) != self.num_features:
            raise ValueError(
                f"Expected input channels {self.num_features}, got {x.size(1)} channels"
            )
        
        if self.num_features % self.num_groups != 0:
            raise ValueError(
                f"Number of channels ({self.num_features}) must be divisible by number of groups ({self.num_groups})"
            )
        
        # Use custom CUDA kernel
        return self.group_norm_kernel.group_norm_forward_cuda(
            x,
            self.weight,
            self.bias,
            self.num_groups,
            self.eps
        )
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=all_negative, 种子=57, 模式=value_stress

**杀死策略描述**: all values < 0

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]
```

---

## 20. `L1_P35__const_perturb__7`

- **算子**: `const_perturb` (Category A)
- **描述**: Perturb numeric literals (integer ±1; float ×1.01 or ×0.99) @ L229
- **变异行**: Line 229, 原始片段 `5`, 节点类型 `const:int-1`
- **杀死层**: `layer1_value`
- **杀死策略**: `structured_ramp`
- **杀死种子**: `72`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half from_float<half>(float val) {
    return __float2half(val);
}

template<typename T>
__global__ void group_norm_forward_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int batch_size,
    const int num_channels,
    const int num_groups,
    const int channels_per_group,
    const int spatial_size,
    const float eps
) {
    // Each block handles one group in one sample
    const int group_idx = blockIdx.x % num_groups;
    const int sample_idx = blockIdx.x / num_groups;
    
    if (sample_idx >= batch_size || group_idx >= num_groups) return;
    
    const int group_start = group_idx * channels_per_group;
    const int group_elements = channels_per_group * spatial_size;
    
    // Shared memory for reductions
    __shared__ float shared_data[256];
    
    // First pass: compute mean
    float thread_sum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        thread_sum += to_float(input[input_idx]);
    }
    
    // Block reduction for sum
    shared_data[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float mean = 0.0f;
    if (threadIdx.x == 0) {
        mean = shared_data[0] / group_elements;
        shared_data[0] = mean;
    }
    __syncthreads();
    
    mean = shared_data[0];
    
    // Second pass: compute variance
    float thread_sqsum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        float val = to_float(input[input_idx]);
        float diff = val - mean;
        thread_sqsum += diff * diff;
    }
    
    // Block reduction for variance
    shared_data[threadIdx.x] = thread_sqsum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float inv_std = 0.0f;
    if (threadIdx.x == 0) {
        float var = shared_data[0] / group_elements;
        inv_std = rsqrtf(var + eps);
        shared_data[0] = inv_std;
    }
    __syncthreads();
    
    inv_std = shared_data[0];
    
    // Apply normalization with affine transformation
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int output_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        
        float val = to_float(input[output_idx]);
        float normalized = (val - mean) * inv_std;
        
        if (weight && bias) {
            normalized = normalized * weight[c] + bias[c];
        }
        
        output[output_idx] = from_float<T>(normalized);
    }
}

torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
) {
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    
    // Calculate spatial size
    int spatial_size = 1;
    for (int i = 2; i < input.dim(); ++i) {
        spatial_size *= input.size(i);
    }
    
    auto channels_per_group = num_channels / num_groups;
    auto output = torch::empty_like(input);
    
    // Configure kernel launch
    const int threads_per_block = 256;
    dim3 blocks(batch_size * num_groups);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "group_norm_forward_cuda",
        ([&] {
            group_norm_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<float>() : nullptr,
                bias.defined() ? bias.data_ptr<float>() : nullptr,
                batch_size,
                num_channels,
                num_groups,
                channels_per_group,
                spatial_size,
                eps
            );
        })
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
);
"""

# Compile the inline CUDA code
group_norm_cuda = load_inline(
    name="group_norm_cuda",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization with custom CUDA kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the optimized GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Small epsilon for numerical stability
        self.eps = 1e-5
        
        # CUDA kernel
        self.group_norm_kernel = group_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Validate input
        if x.dim() < 2:
            raise ValueError(f"Expected at least 2D input (got {x.dim()}D input)")
        
        if x.size(1) != self.num_features:
            raise ValueError(
                f"Expected input channels {self.num_features}, got {x.size(1)} channels"
            )
        
        if self.num_features % self.num_groups != 0:
            raise ValueError(
                f"Number of channels ({self.num_features}) must be divisible by number of groups ({self.num_groups})"
            )
        
        # Use custom CUDA kernel
        return self.group_norm_kernel.group_norm_forward_cuda(
            x,
            self.weight,
            self.bias,
            self.num_groups,
            self.eps
        )
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half from_float<half>(float val) {
    return __float2half(val);
}

template<typename T>
__global__ void group_norm_forward_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int batch_size,
    const int num_channels,
    const int num_groups,
    const int channels_per_group,
    const int spatial_size,
    const float eps
) {
    // Each block handles one group in one sample
    const int group_idx = blockIdx.x % num_groups;
    const int sample_idx = blockIdx.x / num_groups;
    
    if (sample_idx >= batch_size || group_idx >= num_groups) return;
    
    const int group_start = group_idx * channels_per_group;
    const int group_elements = channels_per_group * spatial_size;
    
    // Shared memory for reductions
    __shared__ float shared_data[256];
    
    // First pass: compute mean
    float thread_sum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        thread_sum += to_float(input[input_idx]);
    }
    
    // Block reduction for sum
    shared_data[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float mean = 0.0f;
    if (threadIdx.x == 0) {
        mean = shared_data[0] / group_elements;
        shared_data[0] = mean;
    }
    __syncthreads();
    
    mean = shared_data[0];
    
    // Second pass: compute variance
    float thread_sqsum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        float val = to_float(input[input_idx]);
        float diff = val - mean;
        thread_sqsum += diff * diff;
    }
    
    // Block reduction for variance
    shared_data[threadIdx.x] = thread_sqsum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float inv_std = 0.0f;
    if (threadIdx.x == 0) {
        float var = shared_data[0] / group_elements;
        inv_std = rsqrtf(var + eps);
        shared_data[0] = inv_std;
    }
    __syncthreads();
    
    inv_std = shared_data[0];
    
    // Apply normalization with affine transformation
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int output_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        
        float val = to_float(input[output_idx]);
        float normalized = (val - mean) * inv_std;
        
        if (weight && bias) {
            normalized = normalized * weight[c] + bias[c];
        }
        
        output[output_idx] = from_float<T>(normalized);
    }
}

torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
) {
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    
    // Calculate spatial size
    int spatial_size = 1;
    for (int i = 2; i < input.dim(); ++i) {
        spatial_size *= input.size(i);
    }
    
    auto channels_per_group = num_channels / num_groups;
    auto output = torch::empty_like(input);
    
    // Configure kernel launch
    const int threads_per_block = 256;
    dim3 blocks(batch_size * num_groups);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "group_norm_forward_cuda",
        ([&] {
            group_norm_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<float>() : nullptr,
                bias.defined() ? bias.data_ptr<float>() : nullptr,
                batch_size,
                num_channels,
                num_groups,
                channels_per_group,
                spatial_size,
                eps
            );
        })
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
);
"""

# Compile the inline CUDA code
group_norm_cuda = load_inline(
    name="group_norm_cuda",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization with custom CUDA kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the optimized GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Small epsilon for numerical stability
        self.eps = 1e-4
        
        # CUDA kernel
        self.group_norm_kernel = group_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Validate input
        if x.dim() < 2:
            raise ValueError(f"Expected at least 2D input (got {x.dim()}D input)")
        
        if x.size(1) != self.num_features:
            raise ValueError(
                f"Expected input channels {self.num_features}, got {x.size(1)} channels"
            )
        
        if self.num_features % self.num_groups != 0:
            raise ValueError(
                f"Number of channels ({self.num_features}) must be divisible by number of groups ({self.num_groups})"
            )
        
        # Use custom CUDA kernel
        return self.group_norm_kernel.group_norm_forward_cuda(
            x,
            self.weight,
            self.bias,
            self.num_groups,
            self.eps
        )
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=structured_ramp, 种子=72, 模式=value_stress

**杀死策略描述**: linearly increasing 0→N

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]
```

---

## 21. `L1_P35__epsilon_modify__0`

- **算子**: `epsilon_modify` (Category C)
- **描述**: Alter small epsilon literals (LayerNorm / safe_div / log stability) @ L229
- **变异行**: Line 229, 原始片段 `1e-5`, 节点类型 `eps:to_1e-2`
- **杀死层**: `layer1_value`
- **杀死策略**: `structured_ramp`
- **杀死种子**: `72`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half from_float<half>(float val) {
    return __float2half(val);
}

template<typename T>
__global__ void group_norm_forward_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int batch_size,
    const int num_channels,
    const int num_groups,
    const int channels_per_group,
    const int spatial_size,
    const float eps
) {
    // Each block handles one group in one sample
    const int group_idx = blockIdx.x % num_groups;
    const int sample_idx = blockIdx.x / num_groups;
    
    if (sample_idx >= batch_size || group_idx >= num_groups) return;
    
    const int group_start = group_idx * channels_per_group;
    const int group_elements = channels_per_group * spatial_size;
    
    // Shared memory for reductions
    __shared__ float shared_data[256];
    
    // First pass: compute mean
    float thread_sum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        thread_sum += to_float(input[input_idx]);
    }
    
    // Block reduction for sum
    shared_data[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float mean = 0.0f;
    if (threadIdx.x == 0) {
        mean = shared_data[0] / group_elements;
        shared_data[0] = mean;
    }
    __syncthreads();
    
    mean = shared_data[0];
    
    // Second pass: compute variance
    float thread_sqsum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        float val = to_float(input[input_idx]);
        float diff = val - mean;
        thread_sqsum += diff * diff;
    }
    
    // Block reduction for variance
    shared_data[threadIdx.x] = thread_sqsum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float inv_std = 0.0f;
    if (threadIdx.x == 0) {
        float var = shared_data[0] / group_elements;
        inv_std = rsqrtf(var + eps);
        shared_data[0] = inv_std;
    }
    __syncthreads();
    
    inv_std = shared_data[0];
    
    // Apply normalization with affine transformation
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int output_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        
        float val = to_float(input[output_idx]);
        float normalized = (val - mean) * inv_std;
        
        if (weight && bias) {
            normalized = normalized * weight[c] + bias[c];
        }
        
        output[output_idx] = from_float<T>(normalized);
    }
}

torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
) {
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    
    // Calculate spatial size
    int spatial_size = 1;
    for (int i = 2; i < input.dim(); ++i) {
        spatial_size *= input.size(i);
    }
    
    auto channels_per_group = num_channels / num_groups;
    auto output = torch::empty_like(input);
    
    // Configure kernel launch
    const int threads_per_block = 256;
    dim3 blocks(batch_size * num_groups);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "group_norm_forward_cuda",
        ([&] {
            group_norm_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<float>() : nullptr,
                bias.defined() ? bias.data_ptr<float>() : nullptr,
                batch_size,
                num_channels,
                num_groups,
                channels_per_group,
                spatial_size,
                eps
            );
        })
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
);
"""

# Compile the inline CUDA code
group_norm_cuda = load_inline(
    name="group_norm_cuda",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization with custom CUDA kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the optimized GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Small epsilon for numerical stability
        self.eps = 1e-5
        
        # CUDA kernel
        self.group_norm_kernel = group_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Validate input
        if x.dim() < 2:
            raise ValueError(f"Expected at least 2D input (got {x.dim()}D input)")
        
        if x.size(1) != self.num_features:
            raise ValueError(
                f"Expected input channels {self.num_features}, got {x.size(1)} channels"
            )
        
        if self.num_features % self.num_groups != 0:
            raise ValueError(
                f"Number of channels ({self.num_features}) must be divisible by number of groups ({self.num_groups})"
            )
        
        # Use custom CUDA kernel
        return self.group_norm_kernel.group_norm_forward_cuda(
            x,
            self.weight,
            self.bias,
            self.num_groups,
            self.eps
        )
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half from_float<half>(float val) {
    return __float2half(val);
}

template<typename T>
__global__ void group_norm_forward_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int batch_size,
    const int num_channels,
    const int num_groups,
    const int channels_per_group,
    const int spatial_size,
    const float eps
) {
    // Each block handles one group in one sample
    const int group_idx = blockIdx.x % num_groups;
    const int sample_idx = blockIdx.x / num_groups;
    
    if (sample_idx >= batch_size || group_idx >= num_groups) return;
    
    const int group_start = group_idx * channels_per_group;
    const int group_elements = channels_per_group * spatial_size;
    
    // Shared memory for reductions
    __shared__ float shared_data[256];
    
    // First pass: compute mean
    float thread_sum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        thread_sum += to_float(input[input_idx]);
    }
    
    // Block reduction for sum
    shared_data[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float mean = 0.0f;
    if (threadIdx.x == 0) {
        mean = shared_data[0] / group_elements;
        shared_data[0] = mean;
    }
    __syncthreads();
    
    mean = shared_data[0];
    
    // Second pass: compute variance
    float thread_sqsum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        float val = to_float(input[input_idx]);
        float diff = val - mean;
        thread_sqsum += diff * diff;
    }
    
    // Block reduction for variance
    shared_data[threadIdx.x] = thread_sqsum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float inv_std = 0.0f;
    if (threadIdx.x == 0) {
        float var = shared_data[0] / group_elements;
        inv_std = rsqrtf(var + eps);
        shared_data[0] = inv_std;
    }
    __syncthreads();
    
    inv_std = shared_data[0];
    
    // Apply normalization with affine transformation
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int output_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        
        float val = to_float(input[output_idx]);
        float normalized = (val - mean) * inv_std;
        
        if (weight && bias) {
            normalized = normalized * weight[c] + bias[c];
        }
        
        output[output_idx] = from_float<T>(normalized);
    }
}

torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
) {
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    
    // Calculate spatial size
    int spatial_size = 1;
    for (int i = 2; i < input.dim(); ++i) {
        spatial_size *= input.size(i);
    }
    
    auto channels_per_group = num_channels / num_groups;
    auto output = torch::empty_like(input);
    
    // Configure kernel launch
    const int threads_per_block = 256;
    dim3 blocks(batch_size * num_groups);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "group_norm_forward_cuda",
        ([&] {
            group_norm_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<float>() : nullptr,
                bias.defined() ? bias.data_ptr<float>() : nullptr,
                batch_size,
                num_channels,
                num_groups,
                channels_per_group,
                spatial_size,
                eps
            );
        })
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
);
"""

# Compile the inline CUDA code
group_norm_cuda = load_inline(
    name="group_norm_cuda",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization with custom CUDA kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the optimized GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Small epsilon for numerical stability
        self.eps = 1e-2
        
        # CUDA kernel
        self.group_norm_kernel = group_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Validate input
        if x.dim() < 2:
            raise ValueError(f"Expected at least 2D input (got {x.dim()}D input)")
        
        if x.size(1) != self.num_features:
            raise ValueError(
                f"Expected input channels {self.num_features}, got {x.size(1)} channels"
            )
        
        if self.num_features % self.num_groups != 0:
            raise ValueError(
                f"Number of channels ({self.num_features}) must be divisible by number of groups ({self.num_groups})"
            )
        
        # Use custom CUDA kernel
        return self.group_norm_kernel.group_norm_forward_cuda(
            x,
            self.weight,
            self.bias,
            self.num_groups,
            self.eps
        )
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=structured_ramp, 种子=72, 模式=value_stress

**杀死策略描述**: linearly increasing 0→N

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]
```

---

## 22. `L1_P35__epsilon_modify__1`

- **算子**: `epsilon_modify` (Category C)
- **描述**: Alter small epsilon literals (LayerNorm / safe_div / log stability) @ L229
- **变异行**: Line 229, 原始片段 `1e-5`, 节点类型 `eps:to_zero`
- **杀死层**: `layer1_value`
- **杀死策略**: `near_zero`
- **杀死种子**: `42`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half from_float<half>(float val) {
    return __float2half(val);
}

template<typename T>
__global__ void group_norm_forward_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int batch_size,
    const int num_channels,
    const int num_groups,
    const int channels_per_group,
    const int spatial_size,
    const float eps
) {
    // Each block handles one group in one sample
    const int group_idx = blockIdx.x % num_groups;
    const int sample_idx = blockIdx.x / num_groups;
    
    if (sample_idx >= batch_size || group_idx >= num_groups) return;
    
    const int group_start = group_idx * channels_per_group;
    const int group_elements = channels_per_group * spatial_size;
    
    // Shared memory for reductions
    __shared__ float shared_data[256];
    
    // First pass: compute mean
    float thread_sum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        thread_sum += to_float(input[input_idx]);
    }
    
    // Block reduction for sum
    shared_data[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float mean = 0.0f;
    if (threadIdx.x == 0) {
        mean = shared_data[0] / group_elements;
        shared_data[0] = mean;
    }
    __syncthreads();
    
    mean = shared_data[0];
    
    // Second pass: compute variance
    float thread_sqsum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        float val = to_float(input[input_idx]);
        float diff = val - mean;
        thread_sqsum += diff * diff;
    }
    
    // Block reduction for variance
    shared_data[threadIdx.x] = thread_sqsum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float inv_std = 0.0f;
    if (threadIdx.x == 0) {
        float var = shared_data[0] / group_elements;
        inv_std = rsqrtf(var + eps);
        shared_data[0] = inv_std;
    }
    __syncthreads();
    
    inv_std = shared_data[0];
    
    // Apply normalization with affine transformation
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int output_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        
        float val = to_float(input[output_idx]);
        float normalized = (val - mean) * inv_std;
        
        if (weight && bias) {
            normalized = normalized * weight[c] + bias[c];
        }
        
        output[output_idx] = from_float<T>(normalized);
    }
}

torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
) {
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    
    // Calculate spatial size
    int spatial_size = 1;
    for (int i = 2; i < input.dim(); ++i) {
        spatial_size *= input.size(i);
    }
    
    auto channels_per_group = num_channels / num_groups;
    auto output = torch::empty_like(input);
    
    // Configure kernel launch
    const int threads_per_block = 256;
    dim3 blocks(batch_size * num_groups);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "group_norm_forward_cuda",
        ([&] {
            group_norm_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<float>() : nullptr,
                bias.defined() ? bias.data_ptr<float>() : nullptr,
                batch_size,
                num_channels,
                num_groups,
                channels_per_group,
                spatial_size,
                eps
            );
        })
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
);
"""

# Compile the inline CUDA code
group_norm_cuda = load_inline(
    name="group_norm_cuda",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization with custom CUDA kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the optimized GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Small epsilon for numerical stability
        self.eps = 1e-5
        
        # CUDA kernel
        self.group_norm_kernel = group_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Validate input
        if x.dim() < 2:
            raise ValueError(f"Expected at least 2D input (got {x.dim()}D input)")
        
        if x.size(1) != self.num_features:
            raise ValueError(
                f"Expected input channels {self.num_features}, got {x.size(1)} channels"
            )
        
        if self.num_features % self.num_groups != 0:
            raise ValueError(
                f"Number of channels ({self.num_features}) must be divisible by number of groups ({self.num_groups})"
            )
        
        # Use custom CUDA kernel
        return self.group_norm_kernel.group_norm_forward_cuda(
            x,
            self.weight,
            self.bias,
            self.num_groups,
            self.eps
        )
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half from_float<half>(float val) {
    return __float2half(val);
}

template<typename T>
__global__ void group_norm_forward_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int batch_size,
    const int num_channels,
    const int num_groups,
    const int channels_per_group,
    const int spatial_size,
    const float eps
) {
    // Each block handles one group in one sample
    const int group_idx = blockIdx.x % num_groups;
    const int sample_idx = blockIdx.x / num_groups;
    
    if (sample_idx >= batch_size || group_idx >= num_groups) return;
    
    const int group_start = group_idx * channels_per_group;
    const int group_elements = channels_per_group * spatial_size;
    
    // Shared memory for reductions
    __shared__ float shared_data[256];
    
    // First pass: compute mean
    float thread_sum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        thread_sum += to_float(input[input_idx]);
    }
    
    // Block reduction for sum
    shared_data[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float mean = 0.0f;
    if (threadIdx.x == 0) {
        mean = shared_data[0] / group_elements;
        shared_data[0] = mean;
    }
    __syncthreads();
    
    mean = shared_data[0];
    
    // Second pass: compute variance
    float thread_sqsum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        float val = to_float(input[input_idx]);
        float diff = val - mean;
        thread_sqsum += diff * diff;
    }
    
    // Block reduction for variance
    shared_data[threadIdx.x] = thread_sqsum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float inv_std = 0.0f;
    if (threadIdx.x == 0) {
        float var = shared_data[0] / group_elements;
        inv_std = rsqrtf(var + eps);
        shared_data[0] = inv_std;
    }
    __syncthreads();
    
    inv_std = shared_data[0];
    
    // Apply normalization with affine transformation
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int output_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        
        float val = to_float(input[output_idx]);
        float normalized = (val - mean) * inv_std;
        
        if (weight && bias) {
            normalized = normalized * weight[c] + bias[c];
        }
        
        output[output_idx] = from_float<T>(normalized);
    }
}

torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
) {
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    
    // Calculate spatial size
    int spatial_size = 1;
    for (int i = 2; i < input.dim(); ++i) {
        spatial_size *= input.size(i);
    }
    
    auto channels_per_group = num_channels / num_groups;
    auto output = torch::empty_like(input);
    
    // Configure kernel launch
    const int threads_per_block = 256;
    dim3 blocks(batch_size * num_groups);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "group_norm_forward_cuda",
        ([&] {
            group_norm_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<float>() : nullptr,
                bias.defined() ? bias.data_ptr<float>() : nullptr,
                batch_size,
                num_channels,
                num_groups,
                channels_per_group,
                spatial_size,
                eps
            );
        })
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
);
"""

# Compile the inline CUDA code
group_norm_cuda = load_inline(
    name="group_norm_cuda",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization with custom CUDA kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the optimized GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Small epsilon for numerical stability
        self.eps = 0
        
        # CUDA kernel
        self.group_norm_kernel = group_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Validate input
        if x.dim() < 2:
            raise ValueError(f"Expected at least 2D input (got {x.dim()}D input)")
        
        if x.size(1) != self.num_features:
            raise ValueError(
                f"Expected input channels {self.num_features}, got {x.size(1)} channels"
            )
        
        if self.num_features % self.num_groups != 0:
            raise ValueError(
                f"Number of channels ({self.num_features}) must be divisible by number of groups ({self.num_groups})"
            )
        
        # Use custom CUDA kernel
        return self.group_norm_kernel.group_norm_forward_cuda(
            x,
            self.weight,
            self.bias,
            self.num_groups,
            self.eps
        )
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=near_zero, 种子=42, 模式=value_stress

**杀死策略描述**: values ~1e-7

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]
```

---

## 23. `L1_P35__index_replace__15`

- **算子**: `index_replace` (Category B)
- **描述**: Swap Triton program_id axis or CUDA thread/block dimension index (e.g. program_id(0)→(1), threadIdx.x→threadIdx.y) @ L72
- **变异行**: Line 72, 原始片段 `threadIdx.x`, 节点类型 `cuda_dim|z`
- **杀死层**: `layer1_value`
- **杀死策略**: `structured_ramp`
- **杀死种子**: `42`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half from_float<half>(float val) {
    return __float2half(val);
}

template<typename T>
__global__ void group_norm_forward_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int batch_size,
    const int num_channels,
    const int num_groups,
    const int channels_per_group,
    const int spatial_size,
    const float eps
) {
    // Each block handles one group in one sample
    const int group_idx = blockIdx.x % num_groups;
    const int sample_idx = blockIdx.x / num_groups;
    
    if (sample_idx >= batch_size || group_idx >= num_groups) return;
    
    const int group_start = group_idx * channels_per_group;
    const int group_elements = channels_per_group * spatial_size;
    
    // Shared memory for reductions
    __shared__ float shared_data[256];
    
    // First pass: compute mean
    float thread_sum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        thread_sum += to_float(input[input_idx]);
    }
    
    // Block reduction for sum
    shared_data[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float mean = 0.0f;
    if (threadIdx.x == 0) {
        mean = shared_data[0] / group_elements;
        shared_data[0] = mean;
    }
    __syncthreads();
    
    mean = shared_data[0];
    
    // Second pass: compute variance
    float thread_sqsum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        float val = to_float(input[input_idx]);
        float diff = val - mean;
        thread_sqsum += diff * diff;
    }
    
    // Block reduction for variance
    shared_data[threadIdx.x] = thread_sqsum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float inv_std = 0.0f;
    if (threadIdx.x == 0) {
        float var = shared_data[0] / group_elements;
        inv_std = rsqrtf(var + eps);
        shared_data[0] = inv_std;
    }
    __syncthreads();
    
    inv_std = shared_data[0];
    
    // Apply normalization with affine transformation
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int output_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        
        float val = to_float(input[output_idx]);
        float normalized = (val - mean) * inv_std;
        
        if (weight && bias) {
            normalized = normalized * weight[c] + bias[c];
        }
        
        output[output_idx] = from_float<T>(normalized);
    }
}

torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
) {
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    
    // Calculate spatial size
    int spatial_size = 1;
    for (int i = 2; i < input.dim(); ++i) {
        spatial_size *= input.size(i);
    }
    
    auto channels_per_group = num_channels / num_groups;
    auto output = torch::empty_like(input);
    
    // Configure kernel launch
    const int threads_per_block = 256;
    dim3 blocks(batch_size * num_groups);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "group_norm_forward_cuda",
        ([&] {
            group_norm_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<float>() : nullptr,
                bias.defined() ? bias.data_ptr<float>() : nullptr,
                batch_size,
                num_channels,
                num_groups,
                channels_per_group,
                spatial_size,
                eps
            );
        })
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
);
"""

# Compile the inline CUDA code
group_norm_cuda = load_inline(
    name="group_norm_cuda",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization with custom CUDA kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the optimized GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Small epsilon for numerical stability
        self.eps = 1e-5
        
        # CUDA kernel
        self.group_norm_kernel = group_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Validate input
        if x.dim() < 2:
            raise ValueError(f"Expected at least 2D input (got {x.dim()}D input)")
        
        if x.size(1) != self.num_features:
            raise ValueError(
                f"Expected input channels {self.num_features}, got {x.size(1)} channels"
            )
        
        if self.num_features % self.num_groups != 0:
            raise ValueError(
                f"Number of channels ({self.num_features}) must be divisible by number of groups ({self.num_groups})"
            )
        
        # Use custom CUDA kernel
        return self.group_norm_kernel.group_norm_forward_cuda(
            x,
            self.weight,
            self.bias,
            self.num_groups,
            self.eps
        )
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half from_float<half>(float val) {
    return __float2half(val);
}

template<typename T>
__global__ void group_norm_forward_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int batch_size,
    const int num_channels,
    const int num_groups,
    const int channels_per_group,
    const int spatial_size,
    const float eps
) {
    // Each block handles one group in one sample
    const int group_idx = blockIdx.x % num_groups;
    const int sample_idx = blockIdx.x / num_groups;
    
    if (sample_idx >= batch_size || group_idx >= num_groups) return;
    
    const int group_start = group_idx * channels_per_group;
    const int group_elements = channels_per_group * spatial_size;
    
    // Shared memory for reductions
    __shared__ float shared_data[256];
    
    // First pass: compute mean
    float thread_sum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        thread_sum += to_float(input[input_idx]);
    }
    
    // Block reduction for sum
    shared_data[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.z] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float mean = 0.0f;
    if (threadIdx.x == 0) {
        mean = shared_data[0] / group_elements;
        shared_data[0] = mean;
    }
    __syncthreads();
    
    mean = shared_data[0];
    
    // Second pass: compute variance
    float thread_sqsum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        float val = to_float(input[input_idx]);
        float diff = val - mean;
        thread_sqsum += diff * diff;
    }
    
    // Block reduction for variance
    shared_data[threadIdx.x] = thread_sqsum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float inv_std = 0.0f;
    if (threadIdx.x == 0) {
        float var = shared_data[0] / group_elements;
        inv_std = rsqrtf(var + eps);
        shared_data[0] = inv_std;
    }
    __syncthreads();
    
    inv_std = shared_data[0];
    
    // Apply normalization with affine transformation
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int output_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        
        float val = to_float(input[output_idx]);
        float normalized = (val - mean) * inv_std;
        
        if (weight && bias) {
            normalized = normalized * weight[c] + bias[c];
        }
        
        output[output_idx] = from_float<T>(normalized);
    }
}

torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
) {
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    
    // Calculate spatial size
    int spatial_size = 1;
    for (int i = 2; i < input.dim(); ++i) {
        spatial_size *= input.size(i);
    }
    
    auto channels_per_group = num_channels / num_groups;
    auto output = torch::empty_like(input);
    
    // Configure kernel launch
    const int threads_per_block = 256;
    dim3 blocks(batch_size * num_groups);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "group_norm_forward_cuda",
        ([&] {
            group_norm_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<float>() : nullptr,
                bias.defined() ? bias.data_ptr<float>() : nullptr,
                batch_size,
                num_channels,
                num_groups,
                channels_per_group,
                spatial_size,
                eps
            );
        })
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
);
"""

# Compile the inline CUDA code
group_norm_cuda = load_inline(
    name="group_norm_cuda",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization with custom CUDA kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the optimized GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Small epsilon for numerical stability
        self.eps = 1e-5
        
        # CUDA kernel
        self.group_norm_kernel = group_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Validate input
        if x.dim() < 2:
            raise ValueError(f"Expected at least 2D input (got {x.dim()}D input)")
        
        if x.size(1) != self.num_features:
            raise ValueError(
                f"Expected input channels {self.num_features}, got {x.size(1)} channels"
            )
        
        if self.num_features % self.num_groups != 0:
            raise ValueError(
                f"Number of channels ({self.num_features}) must be divisible by number of groups ({self.num_groups})"
            )
        
        # Use custom CUDA kernel
        return self.group_norm_kernel.group_norm_forward_cuda(
            x,
            self.weight,
            self.bias,
            self.num_groups,
            self.eps
        )
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=structured_ramp, 种子=42, 模式=value_stress

**杀死策略描述**: linearly increasing 0→N

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]
```

---

## 24. `L1_P35__scale_modify__0`

- **算子**: `scale_modify` (Category C)
- **描述**: Distort attention / normalization scaling (inv-sqrt, rsqrt) @ L111
- **变异行**: Line 111, 原始片段 `rsqrtf(var + eps)`, 节点类型 `scale:cuda_rsqrt_identity`
- **杀死层**: `layer1_value`
- **杀死策略**: `structured_ramp`
- **杀死种子**: `45`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half from_float<half>(float val) {
    return __float2half(val);
}

template<typename T>
__global__ void group_norm_forward_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int batch_size,
    const int num_channels,
    const int num_groups,
    const int channels_per_group,
    const int spatial_size,
    const float eps
) {
    // Each block handles one group in one sample
    const int group_idx = blockIdx.x % num_groups;
    const int sample_idx = blockIdx.x / num_groups;
    
    if (sample_idx >= batch_size || group_idx >= num_groups) return;
    
    const int group_start = group_idx * channels_per_group;
    const int group_elements = channels_per_group * spatial_size;
    
    // Shared memory for reductions
    __shared__ float shared_data[256];
    
    // First pass: compute mean
    float thread_sum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        thread_sum += to_float(input[input_idx]);
    }
    
    // Block reduction for sum
    shared_data[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float mean = 0.0f;
    if (threadIdx.x == 0) {
        mean = shared_data[0] / group_elements;
        shared_data[0] = mean;
    }
    __syncthreads();
    
    mean = shared_data[0];
    
    // Second pass: compute variance
    float thread_sqsum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        float val = to_float(input[input_idx]);
        float diff = val - mean;
        thread_sqsum += diff * diff;
    }
    
    // Block reduction for variance
    shared_data[threadIdx.x] = thread_sqsum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float inv_std = 0.0f;
    if (threadIdx.x == 0) {
        float var = shared_data[0] / group_elements;
        inv_std = rsqrtf(var + eps);
        shared_data[0] = inv_std;
    }
    __syncthreads();
    
    inv_std = shared_data[0];
    
    // Apply normalization with affine transformation
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int output_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        
        float val = to_float(input[output_idx]);
        float normalized = (val - mean) * inv_std;
        
        if (weight && bias) {
            normalized = normalized * weight[c] + bias[c];
        }
        
        output[output_idx] = from_float<T>(normalized);
    }
}

torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
) {
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    
    // Calculate spatial size
    int spatial_size = 1;
    for (int i = 2; i < input.dim(); ++i) {
        spatial_size *= input.size(i);
    }
    
    auto channels_per_group = num_channels / num_groups;
    auto output = torch::empty_like(input);
    
    // Configure kernel launch
    const int threads_per_block = 256;
    dim3 blocks(batch_size * num_groups);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "group_norm_forward_cuda",
        ([&] {
            group_norm_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<float>() : nullptr,
                bias.defined() ? bias.data_ptr<float>() : nullptr,
                batch_size,
                num_channels,
                num_groups,
                channels_per_group,
                spatial_size,
                eps
            );
        })
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
);
"""

# Compile the inline CUDA code
group_norm_cuda = load_inline(
    name="group_norm_cuda",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization with custom CUDA kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the optimized GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Small epsilon for numerical stability
        self.eps = 1e-5
        
        # CUDA kernel
        self.group_norm_kernel = group_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Validate input
        if x.dim() < 2:
            raise ValueError(f"Expected at least 2D input (got {x.dim()}D input)")
        
        if x.size(1) != self.num_features:
            raise ValueError(
                f"Expected input channels {self.num_features}, got {x.size(1)} channels"
            )
        
        if self.num_features % self.num_groups != 0:
            raise ValueError(
                f"Number of channels ({self.num_features}) must be divisible by number of groups ({self.num_groups})"
            )
        
        # Use custom CUDA kernel
        return self.group_norm_kernel.group_norm_forward_cuda(
            x,
            self.weight,
            self.bias,
            self.num_groups,
            self.eps
        )
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half from_float<half>(float val) {
    return __float2half(val);
}

template<typename T>
__global__ void group_norm_forward_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int batch_size,
    const int num_channels,
    const int num_groups,
    const int channels_per_group,
    const int spatial_size,
    const float eps
) {
    // Each block handles one group in one sample
    const int group_idx = blockIdx.x % num_groups;
    const int sample_idx = blockIdx.x / num_groups;
    
    if (sample_idx >= batch_size || group_idx >= num_groups) return;
    
    const int group_start = group_idx * channels_per_group;
    const int group_elements = channels_per_group * spatial_size;
    
    // Shared memory for reductions
    __shared__ float shared_data[256];
    
    // First pass: compute mean
    float thread_sum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        thread_sum += to_float(input[input_idx]);
    }
    
    // Block reduction for sum
    shared_data[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float mean = 0.0f;
    if (threadIdx.x == 0) {
        mean = shared_data[0] / group_elements;
        shared_data[0] = mean;
    }
    __syncthreads();
    
    mean = shared_data[0];
    
    // Second pass: compute variance
    float thread_sqsum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        float val = to_float(input[input_idx]);
        float diff = val - mean;
        thread_sqsum += diff * diff;
    }
    
    // Block reduction for variance
    shared_data[threadIdx.x] = thread_sqsum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float inv_std = 0.0f;
    if (threadIdx.x == 0) {
        float var = shared_data[0] / group_elements;
        inv_std = var + eps;
        shared_data[0] = inv_std;
    }
    __syncthreads();
    
    inv_std = shared_data[0];
    
    // Apply normalization with affine transformation
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int output_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        
        float val = to_float(input[output_idx]);
        float normalized = (val - mean) * inv_std;
        
        if (weight && bias) {
            normalized = normalized * weight[c] + bias[c];
        }
        
        output[output_idx] = from_float<T>(normalized);
    }
}

torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
) {
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    
    // Calculate spatial size
    int spatial_size = 1;
    for (int i = 2; i < input.dim(); ++i) {
        spatial_size *= input.size(i);
    }
    
    auto channels_per_group = num_channels / num_groups;
    auto output = torch::empty_like(input);
    
    // Configure kernel launch
    const int threads_per_block = 256;
    dim3 blocks(batch_size * num_groups);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "group_norm_forward_cuda",
        ([&] {
            group_norm_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<float>() : nullptr,
                bias.defined() ? bias.data_ptr<float>() : nullptr,
                batch_size,
                num_channels,
                num_groups,
                channels_per_group,
                spatial_size,
                eps
            );
        })
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
);
"""

# Compile the inline CUDA code
group_norm_cuda = load_inline(
    name="group_norm_cuda",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization with custom CUDA kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the optimized GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Small epsilon for numerical stability
        self.eps = 1e-5
        
        # CUDA kernel
        self.group_norm_kernel = group_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Validate input
        if x.dim() < 2:
            raise ValueError(f"Expected at least 2D input (got {x.dim()}D input)")
        
        if x.size(1) != self.num_features:
            raise ValueError(
                f"Expected input channels {self.num_features}, got {x.size(1)} channels"
            )
        
        if self.num_features % self.num_groups != 0:
            raise ValueError(
                f"Number of channels ({self.num_features}) must be divisible by number of groups ({self.num_groups})"
            )
        
        # Use custom CUDA kernel
        return self.group_norm_kernel.group_norm_forward_cuda(
            x,
            self.weight,
            self.bias,
            self.num_groups,
            self.eps
        )
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=structured_ramp, 种子=45, 模式=value_stress

**杀死策略描述**: linearly increasing 0→N

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]
```

---

## 25. `L1_P35__sync_remove__0`

- **算子**: `sync_remove` (Category B)
- **描述**: Remove GPU synchronization barriers (tl.debug_barrier, __syncthreads) @ L68
- **变异行**: Line 68, 原始片段 `__syncthreads();`, 节点类型 `cuda_syncthreads`
- **杀死层**: `layer1_value`
- **杀死策略**: `all_negative`
- **杀死种子**: `57`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half from_float<half>(float val) {
    return __float2half(val);
}

template<typename T>
__global__ void group_norm_forward_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int batch_size,
    const int num_channels,
    const int num_groups,
    const int channels_per_group,
    const int spatial_size,
    const float eps
) {
    // Each block handles one group in one sample
    const int group_idx = blockIdx.x % num_groups;
    const int sample_idx = blockIdx.x / num_groups;
    
    if (sample_idx >= batch_size || group_idx >= num_groups) return;
    
    const int group_start = group_idx * channels_per_group;
    const int group_elements = channels_per_group * spatial_size;
    
    // Shared memory for reductions
    __shared__ float shared_data[256];
    
    // First pass: compute mean
    float thread_sum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        thread_sum += to_float(input[input_idx]);
    }
    
    // Block reduction for sum
    shared_data[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float mean = 0.0f;
    if (threadIdx.x == 0) {
        mean = shared_data[0] / group_elements;
        shared_data[0] = mean;
    }
    __syncthreads();
    
    mean = shared_data[0];
    
    // Second pass: compute variance
    float thread_sqsum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        float val = to_float(input[input_idx]);
        float diff = val - mean;
        thread_sqsum += diff * diff;
    }
    
    // Block reduction for variance
    shared_data[threadIdx.x] = thread_sqsum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float inv_std = 0.0f;
    if (threadIdx.x == 0) {
        float var = shared_data[0] / group_elements;
        inv_std = rsqrtf(var + eps);
        shared_data[0] = inv_std;
    }
    __syncthreads();
    
    inv_std = shared_data[0];
    
    // Apply normalization with affine transformation
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int output_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        
        float val = to_float(input[output_idx]);
        float normalized = (val - mean) * inv_std;
        
        if (weight && bias) {
            normalized = normalized * weight[c] + bias[c];
        }
        
        output[output_idx] = from_float<T>(normalized);
    }
}

torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
) {
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    
    // Calculate spatial size
    int spatial_size = 1;
    for (int i = 2; i < input.dim(); ++i) {
        spatial_size *= input.size(i);
    }
    
    auto channels_per_group = num_channels / num_groups;
    auto output = torch::empty_like(input);
    
    // Configure kernel launch
    const int threads_per_block = 256;
    dim3 blocks(batch_size * num_groups);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "group_norm_forward_cuda",
        ([&] {
            group_norm_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<float>() : nullptr,
                bias.defined() ? bias.data_ptr<float>() : nullptr,
                batch_size,
                num_channels,
                num_groups,
                channels_per_group,
                spatial_size,
                eps
            );
        })
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
);
"""

# Compile the inline CUDA code
group_norm_cuda = load_inline(
    name="group_norm_cuda",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization with custom CUDA kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the optimized GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Small epsilon for numerical stability
        self.eps = 1e-5
        
        # CUDA kernel
        self.group_norm_kernel = group_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Validate input
        if x.dim() < 2:
            raise ValueError(f"Expected at least 2D input (got {x.dim()}D input)")
        
        if x.size(1) != self.num_features:
            raise ValueError(
                f"Expected input channels {self.num_features}, got {x.size(1)} channels"
            )
        
        if self.num_features % self.num_groups != 0:
            raise ValueError(
                f"Number of channels ({self.num_features}) must be divisible by number of groups ({self.num_groups})"
            )
        
        # Use custom CUDA kernel
        return self.group_norm_kernel.group_norm_forward_cuda(
            x,
            self.weight,
            self.bias,
            self.num_groups,
            self.eps
        )
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half from_float<half>(float val) {
    return __float2half(val);
}

template<typename T>
__global__ void group_norm_forward_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int batch_size,
    const int num_channels,
    const int num_groups,
    const int channels_per_group,
    const int spatial_size,
    const float eps
) {
    // Each block handles one group in one sample
    const int group_idx = blockIdx.x % num_groups;
    const int sample_idx = blockIdx.x / num_groups;
    
    if (sample_idx >= batch_size || group_idx >= num_groups) return;
    
    const int group_start = group_idx * channels_per_group;
    const int group_elements = channels_per_group * spatial_size;
    
    // Shared memory for reductions
    __shared__ float shared_data[256];
    
    // First pass: compute mean
    float thread_sum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        thread_sum += to_float(input[input_idx]);
    }
    
    // Block reduction for sum
    shared_data[threadIdx.x] = thread_sum;
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float mean = 0.0f;
    if (threadIdx.x == 0) {
        mean = shared_data[0] / group_elements;
        shared_data[0] = mean;
    }
    __syncthreads();
    
    mean = shared_data[0];
    
    // Second pass: compute variance
    float thread_sqsum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        float val = to_float(input[input_idx]);
        float diff = val - mean;
        thread_sqsum += diff * diff;
    }
    
    // Block reduction for variance
    shared_data[threadIdx.x] = thread_sqsum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float inv_std = 0.0f;
    if (threadIdx.x == 0) {
        float var = shared_data[0] / group_elements;
        inv_std = rsqrtf(var + eps);
        shared_data[0] = inv_std;
    }
    __syncthreads();
    
    inv_std = shared_data[0];
    
    // Apply normalization with affine transformation
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int output_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        
        float val = to_float(input[output_idx]);
        float normalized = (val - mean) * inv_std;
        
        if (weight && bias) {
            normalized = normalized * weight[c] + bias[c];
        }
        
        output[output_idx] = from_float<T>(normalized);
    }
}

torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
) {
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    
    // Calculate spatial size
    int spatial_size = 1;
    for (int i = 2; i < input.dim(); ++i) {
        spatial_size *= input.size(i);
    }
    
    auto channels_per_group = num_channels / num_groups;
    auto output = torch::empty_like(input);
    
    // Configure kernel launch
    const int threads_per_block = 256;
    dim3 blocks(batch_size * num_groups);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "group_norm_forward_cuda",
        ([&] {
            group_norm_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<float>() : nullptr,
                bias.defined() ? bias.data_ptr<float>() : nullptr,
                batch_size,
                num_channels,
                num_groups,
                channels_per_group,
                spatial_size,
                eps
            );
        })
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
);
"""

# Compile the inline CUDA code
group_norm_cuda = load_inline(
    name="group_norm_cuda",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization with custom CUDA kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the optimized GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Small epsilon for numerical stability
        self.eps = 1e-5
        
        # CUDA kernel
        self.group_norm_kernel = group_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Validate input
        if x.dim() < 2:
            raise ValueError(f"Expected at least 2D input (got {x.dim()}D input)")
        
        if x.size(1) != self.num_features:
            raise ValueError(
                f"Expected input channels {self.num_features}, got {x.size(1)} channels"
            )
        
        if self.num_features % self.num_groups != 0:
            raise ValueError(
                f"Number of channels ({self.num_features}) must be divisible by number of groups ({self.num_groups})"
            )
        
        # Use custom CUDA kernel
        return self.group_norm_kernel.group_norm_forward_cuda(
            x,
            self.weight,
            self.bias,
            self.num_groups,
            self.eps
        )
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=all_negative, 种子=57, 模式=value_stress

**杀死策略描述**: all values < 0

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]
```

---

## 26. `L1_P35__sync_remove__1`

- **算子**: `sync_remove` (Category B)
- **描述**: Remove GPU synchronization barriers (tl.debug_barrier, __syncthreads) @ L74
- **变异行**: Line 74, 原始片段 `__syncthreads();`, 节点类型 `cuda_syncthreads`
- **杀死层**: `layer1_value`
- **杀死策略**: `all_negative`
- **杀死种子**: `57`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half from_float<half>(float val) {
    return __float2half(val);
}

template<typename T>
__global__ void group_norm_forward_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int batch_size,
    const int num_channels,
    const int num_groups,
    const int channels_per_group,
    const int spatial_size,
    const float eps
) {
    // Each block handles one group in one sample
    const int group_idx = blockIdx.x % num_groups;
    const int sample_idx = blockIdx.x / num_groups;
    
    if (sample_idx >= batch_size || group_idx >= num_groups) return;
    
    const int group_start = group_idx * channels_per_group;
    const int group_elements = channels_per_group * spatial_size;
    
    // Shared memory for reductions
    __shared__ float shared_data[256];
    
    // First pass: compute mean
    float thread_sum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        thread_sum += to_float(input[input_idx]);
    }
    
    // Block reduction for sum
    shared_data[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float mean = 0.0f;
    if (threadIdx.x == 0) {
        mean = shared_data[0] / group_elements;
        shared_data[0] = mean;
    }
    __syncthreads();
    
    mean = shared_data[0];
    
    // Second pass: compute variance
    float thread_sqsum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        float val = to_float(input[input_idx]);
        float diff = val - mean;
        thread_sqsum += diff * diff;
    }
    
    // Block reduction for variance
    shared_data[threadIdx.x] = thread_sqsum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float inv_std = 0.0f;
    if (threadIdx.x == 0) {
        float var = shared_data[0] / group_elements;
        inv_std = rsqrtf(var + eps);
        shared_data[0] = inv_std;
    }
    __syncthreads();
    
    inv_std = shared_data[0];
    
    // Apply normalization with affine transformation
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int output_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        
        float val = to_float(input[output_idx]);
        float normalized = (val - mean) * inv_std;
        
        if (weight && bias) {
            normalized = normalized * weight[c] + bias[c];
        }
        
        output[output_idx] = from_float<T>(normalized);
    }
}

torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
) {
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    
    // Calculate spatial size
    int spatial_size = 1;
    for (int i = 2; i < input.dim(); ++i) {
        spatial_size *= input.size(i);
    }
    
    auto channels_per_group = num_channels / num_groups;
    auto output = torch::empty_like(input);
    
    // Configure kernel launch
    const int threads_per_block = 256;
    dim3 blocks(batch_size * num_groups);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "group_norm_forward_cuda",
        ([&] {
            group_norm_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<float>() : nullptr,
                bias.defined() ? bias.data_ptr<float>() : nullptr,
                batch_size,
                num_channels,
                num_groups,
                channels_per_group,
                spatial_size,
                eps
            );
        })
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
);
"""

# Compile the inline CUDA code
group_norm_cuda = load_inline(
    name="group_norm_cuda",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization with custom CUDA kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the optimized GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Small epsilon for numerical stability
        self.eps = 1e-5
        
        # CUDA kernel
        self.group_norm_kernel = group_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Validate input
        if x.dim() < 2:
            raise ValueError(f"Expected at least 2D input (got {x.dim()}D input)")
        
        if x.size(1) != self.num_features:
            raise ValueError(
                f"Expected input channels {self.num_features}, got {x.size(1)} channels"
            )
        
        if self.num_features % self.num_groups != 0:
            raise ValueError(
                f"Number of channels ({self.num_features}) must be divisible by number of groups ({self.num_groups})"
            )
        
        # Use custom CUDA kernel
        return self.group_norm_kernel.group_norm_forward_cuda(
            x,
            self.weight,
            self.bias,
            self.num_groups,
            self.eps
        )
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half from_float<half>(float val) {
    return __float2half(val);
}

template<typename T>
__global__ void group_norm_forward_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int batch_size,
    const int num_channels,
    const int num_groups,
    const int channels_per_group,
    const int spatial_size,
    const float eps
) {
    // Each block handles one group in one sample
    const int group_idx = blockIdx.x % num_groups;
    const int sample_idx = blockIdx.x / num_groups;
    
    if (sample_idx >= batch_size || group_idx >= num_groups) return;
    
    const int group_start = group_idx * channels_per_group;
    const int group_elements = channels_per_group * spatial_size;
    
    // Shared memory for reductions
    __shared__ float shared_data[256];
    
    // First pass: compute mean
    float thread_sum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        thread_sum += to_float(input[input_idx]);
    }
    
    // Block reduction for sum
    shared_data[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
    }
    
    float mean = 0.0f;
    if (threadIdx.x == 0) {
        mean = shared_data[0] / group_elements;
        shared_data[0] = mean;
    }
    __syncthreads();
    
    mean = shared_data[0];
    
    // Second pass: compute variance
    float thread_sqsum = 0.0f;
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int input_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        float val = to_float(input[input_idx]);
        float diff = val - mean;
        thread_sqsum += diff * diff;
    }
    
    // Block reduction for variance
    shared_data[threadIdx.x] = thread_sqsum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float inv_std = 0.0f;
    if (threadIdx.x == 0) {
        float var = shared_data[0] / group_elements;
        inv_std = rsqrtf(var + eps);
        shared_data[0] = inv_std;
    }
    __syncthreads();
    
    inv_std = shared_data[0];
    
    // Apply normalization with affine transformation
    for (int idx = threadIdx.x; idx < group_elements; idx += blockDim.x) {
        int c = group_start + (idx / spatial_size);
        int s = idx % spatial_size;
        int output_idx = ((sample_idx * num_channels + c) * spatial_size + s);
        
        float val = to_float(input[output_idx]);
        float normalized = (val - mean) * inv_std;
        
        if (weight && bias) {
            normalized = normalized * weight[c] + bias[c];
        }
        
        output[output_idx] = from_float<T>(normalized);
    }
}

torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
) {
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    
    // Calculate spatial size
    int spatial_size = 1;
    for (int i = 2; i < input.dim(); ++i) {
        spatial_size *= input.size(i);
    }
    
    auto channels_per_group = num_channels / num_groups;
    auto output = torch::empty_like(input);
    
    // Configure kernel launch
    const int threads_per_block = 256;
    dim3 blocks(batch_size * num_groups);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "group_norm_forward_cuda",
        ([&] {
            group_norm_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<float>() : nullptr,
                bias.defined() ? bias.data_ptr<float>() : nullptr,
                batch_size,
                num_channels,
                num_groups,
                channels_per_group,
                spatial_size,
                eps
            );
        })
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
);
"""

# Compile the inline CUDA code
group_norm_cuda = load_inline(
    name="group_norm_cuda",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization with custom CUDA kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the optimized GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Small epsilon for numerical stability
        self.eps = 1e-5
        
        # CUDA kernel
        self.group_norm_kernel = group_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Validate input
        if x.dim() < 2:
            raise ValueError(f"Expected at least 2D input (got {x.dim()}D input)")
        
        if x.size(1) != self.num_features:
            raise ValueError(
                f"Expected input channels {self.num_features}, got {x.size(1)} channels"
            )
        
        if self.num_features % self.num_groups != 0:
            raise ValueError(
                f"Number of channels ({self.num_features}) must be divisible by number of groups ({self.num_groups})"
            )
        
        # Use custom CUDA kernel
        return self.group_norm_kernel.group_norm_forward_cuda(
            x,
            self.weight,
            self.bias,
            self.num_groups,
            self.eps
        )
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=all_negative, 种子=57, 模式=value_stress

**杀死策略描述**: all values < 0

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]
```

---

## 27. `L1_P37__const_perturb__7`

- **算子**: `const_perturb` (Category A)
- **描述**: Perturb numeric literals (integer ±1; float ×1.01 or ×0.99) @ L52
- **变异行**: Line 52, 原始片段 `8`, 节点类型 `const:int-1`
- **杀死层**: `layer1_value`
- **杀死策略**: `all_negative`
- **杀死种子**: `56`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Frobenius norm normalization
frobenius_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__global__ void frobenius_norm_kernel(const T* input, T* output, T* norm, int total_elements) {
    __shared__ T shared_mem[1024];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Phase 1: Parallel reduction for norm calculation
    T local_sum = 0.0;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < total_elements; i += stride) {
        T val = input[i];
        local_sum += val * val;
    }
    
    shared_mem[tid] = local_sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    // Store block sum to global memory
    if (tid == 0) {
        norm[blockIdx.x] = shared_mem[0];
    }
    __syncthreads();
    
    // Phase 2: Wait for all blocks to complete and compute final norm
    __shared__ T final_norm;
    if (tid == 0) {
        T total_sum = 0.0;
        for (int i = 0; i < gridDim.x; i++) {
            total_sum += norm[i];
        }
        final_norm = sqrt(total_sum + 1e-8);  // Add epsilon for numerical stability
    }
    __syncthreads();
    
    T inv_norm = 1.0 / final_norm;
    
    // Phase 3: Normalize elements
    for (int i = idx; i < total_elements; i += stride) {
        output[i] = input[i] * inv_norm;
    }
}

torch::Tensor frobenius_norm_cuda(torch::Tensor input) {
    auto total_elements = input.numel();
    auto output = torch::empty_like(input);
    
    // Determine optimal block and grid sizes
    const int block_size = 1024;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    num_blocks = min(num_blocks, 1024);  // Limit number of blocks
    
    // Allocate temporary memory for block sums
    auto block_sums = torch::zeros({num_blocks}, torch::TensorOptions()
                                   .dtype(input.dtype())
                                   .device(input.device()));
    
    // Launch kernel based on data type
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "frobenius_norm_kernel", ([&] {
        frobenius_norm_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            block_sums.data_ptr<scalar_t>(),
            total_elements
        );
    }));
    
    cudaDeviceSynchronize();
    return output;
}
"""

frobenius_norm_cpp_source = """
torch::Tensor frobenius_norm_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
frobenius_norm = load_inline(
    name="frobenius_norm",
    cpp_sources=frobenius_norm_cpp_source,
    cuda_sources=frobenius_norm_source,
    functions=["frobenius_norm_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Frobenius norm normalization using custom CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the optimized Frobenius norm normalization layer.
        """
        super(ModelNew, self).__init__()
        self.frobenius_norm = frobenius_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies fused Frobenius norm normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        # Use custom CUDA kernel for GPU tensors, fallback to PyTorch for CPU
        if x.is_cuda:
            return self.frobenius_norm.frobenius_norm_cuda(x)
        else:
            norm = torch.norm(x, p='fro')
            return x / norm
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Frobenius norm normalization
frobenius_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__global__ void frobenius_norm_kernel(const T* input, T* output, T* norm, int total_elements) {
    __shared__ T shared_mem[1024];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Phase 1: Parallel reduction for norm calculation
    T local_sum = 0.0;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < total_elements; i += stride) {
        T val = input[i];
        local_sum += val * val;
    }
    
    shared_mem[tid] = local_sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    // Store block sum to global memory
    if (tid == 0) {
        norm[blockIdx.x] = shared_mem[0];
    }
    __syncthreads();
    
    // Phase 2: Wait for all blocks to complete and compute final norm
    __shared__ T final_norm;
    if (tid == 0) {
        T total_sum = 0.0;
        for (int i = 0; i < gridDim.x; i++) {
            total_sum += norm[i];
        }
        final_norm = sqrt(total_sum + 1e-7);  // Add epsilon for numerical stability
    }
    __syncthreads();
    
    T inv_norm = 1.0 / final_norm;
    
    // Phase 3: Normalize elements
    for (int i = idx; i < total_elements; i += stride) {
        output[i] = input[i] * inv_norm;
    }
}

torch::Tensor frobenius_norm_cuda(torch::Tensor input) {
    auto total_elements = input.numel();
    auto output = torch::empty_like(input);
    
    // Determine optimal block and grid sizes
    const int block_size = 1024;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    num_blocks = min(num_blocks, 1024);  // Limit number of blocks
    
    // Allocate temporary memory for block sums
    auto block_sums = torch::zeros({num_blocks}, torch::TensorOptions()
                                   .dtype(input.dtype())
                                   .device(input.device()));
    
    // Launch kernel based on data type
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "frobenius_norm_kernel", ([&] {
        frobenius_norm_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            block_sums.data_ptr<scalar_t>(),
            total_elements
        );
    }));
    
    cudaDeviceSynchronize();
    return output;
}
"""

frobenius_norm_cpp_source = """
torch::Tensor frobenius_norm_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
frobenius_norm = load_inline(
    name="frobenius_norm",
    cpp_sources=frobenius_norm_cpp_source,
    cuda_sources=frobenius_norm_source,
    functions=["frobenius_norm_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Frobenius norm normalization using custom CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the optimized Frobenius norm normalization layer.
        """
        super(ModelNew, self).__init__()
        self.frobenius_norm = frobenius_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies fused Frobenius norm normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        # Use custom CUDA kernel for GPU tensors, fallback to PyTorch for CPU
        if x.is_cuda:
            return self.frobenius_norm.frobenius_norm_cuda(x)
        else:
            norm = torch.norm(x, p='fro')
            return x / norm
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=all_negative, 种子=56, 模式=value_stress

**杀死策略描述**: all values < 0

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]
```

---

## 28. `L1_P37__index_replace__2`

- **算子**: `index_replace` (Category B)
- **描述**: Swap Triton program_id axis or CUDA thread/block dimension index (e.g. program_id(0)→(1), threadIdx.x→threadIdx.y) @ L17
- **变异行**: Line 17, 原始片段 `blockIdx.x`, 节点类型 `cuda_dim|y`
- **杀死层**: `layer3_repeated`
- **杀死策略**: `None`
- **杀死种子**: `242`
- **杀死模式**: `repeated_run`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Frobenius norm normalization
frobenius_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__global__ void frobenius_norm_kernel(const T* input, T* output, T* norm, int total_elements) {
    __shared__ T shared_mem[1024];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Phase 1: Parallel reduction for norm calculation
    T local_sum = 0.0;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < total_elements; i += stride) {
        T val = input[i];
        local_sum += val * val;
    }
    
    shared_mem[tid] = local_sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    // Store block sum to global memory
    if (tid == 0) {
        norm[blockIdx.x] = shared_mem[0];
    }
    __syncthreads();
    
    // Phase 2: Wait for all blocks to complete and compute final norm
    __shared__ T final_norm;
    if (tid == 0) {
        T total_sum = 0.0;
        for (int i = 0; i < gridDim.x; i++) {
            total_sum += norm[i];
        }
        final_norm = sqrt(total_sum + 1e-8);  // Add epsilon for numerical stability
    }
    __syncthreads();
    
    T inv_norm = 1.0 / final_norm;
    
    // Phase 3: Normalize elements
    for (int i = idx; i < total_elements; i += stride) {
        output[i] = input[i] * inv_norm;
    }
}

torch::Tensor frobenius_norm_cuda(torch::Tensor input) {
    auto total_elements = input.numel();
    auto output = torch::empty_like(input);
    
    // Determine optimal block and grid sizes
    const int block_size = 1024;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    num_blocks = min(num_blocks, 1024);  // Limit number of blocks
    
    // Allocate temporary memory for block sums
    auto block_sums = torch::zeros({num_blocks}, torch::TensorOptions()
                                   .dtype(input.dtype())
                                   .device(input.device()));
    
    // Launch kernel based on data type
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "frobenius_norm_kernel", ([&] {
        frobenius_norm_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            block_sums.data_ptr<scalar_t>(),
            total_elements
        );
    }));
    
    cudaDeviceSynchronize();
    return output;
}
"""

frobenius_norm_cpp_source = """
torch::Tensor frobenius_norm_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
frobenius_norm = load_inline(
    name="frobenius_norm",
    cpp_sources=frobenius_norm_cpp_source,
    cuda_sources=frobenius_norm_source,
    functions=["frobenius_norm_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Frobenius norm normalization using custom CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the optimized Frobenius norm normalization layer.
        """
        super(ModelNew, self).__init__()
        self.frobenius_norm = frobenius_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies fused Frobenius norm normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        # Use custom CUDA kernel for GPU tensors, fallback to PyTorch for CPU
        if x.is_cuda:
            return self.frobenius_norm.frobenius_norm_cuda(x)
        else:
            norm = torch.norm(x, p='fro')
            return x / norm
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Frobenius norm normalization
frobenius_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__global__ void frobenius_norm_kernel(const T* input, T* output, T* norm, int total_elements) {
    __shared__ T shared_mem[1024];
    
    int tid = threadIdx.x;
    int idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    // Phase 1: Parallel reduction for norm calculation
    T local_sum = 0.0;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < total_elements; i += stride) {
        T val = input[i];
        local_sum += val * val;
    }
    
    shared_mem[tid] = local_sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    // Store block sum to global memory
    if (tid == 0) {
        norm[blockIdx.x] = shared_mem[0];
    }
    __syncthreads();
    
    // Phase 2: Wait for all blocks to complete and compute final norm
    __shared__ T final_norm;
    if (tid == 0) {
        T total_sum = 0.0;
        for (int i = 0; i < gridDim.x; i++) {
            total_sum += norm[i];
        }
        final_norm = sqrt(total_sum + 1e-8);  // Add epsilon for numerical stability
    }
    __syncthreads();
    
    T inv_norm = 1.0 / final_norm;
    
    // Phase 3: Normalize elements
    for (int i = idx; i < total_elements; i += stride) {
        output[i] = input[i] * inv_norm;
    }
}

torch::Tensor frobenius_norm_cuda(torch::Tensor input) {
    auto total_elements = input.numel();
    auto output = torch::empty_like(input);
    
    // Determine optimal block and grid sizes
    const int block_size = 1024;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    num_blocks = min(num_blocks, 1024);  // Limit number of blocks
    
    // Allocate temporary memory for block sums
    auto block_sums = torch::zeros({num_blocks}, torch::TensorOptions()
                                   .dtype(input.dtype())
                                   .device(input.device()));
    
    // Launch kernel based on data type
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "frobenius_norm_kernel", ([&] {
        frobenius_norm_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            block_sums.data_ptr<scalar_t>(),
            total_elements
        );
    }));
    
    cudaDeviceSynchronize();
    return output;
}
"""

frobenius_norm_cpp_source = """
torch::Tensor frobenius_norm_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
frobenius_norm = load_inline(
    name="frobenius_norm",
    cpp_sources=frobenius_norm_cpp_source,
    cuda_sources=frobenius_norm_source,
    functions=["frobenius_norm_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Frobenius norm normalization using custom CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the optimized Frobenius norm normalization layer.
        """
        super(ModelNew, self).__init__()
        self.frobenius_norm = frobenius_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies fused Frobenius norm normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        # Use custom CUDA kernel for GPU tensors, fallback to PyTorch for CPU
        if x.is_cuda:
            return self.frobenius_norm.frobenius_norm_cuda(x)
        else:
            norm = torch.norm(x, p='fro')
            return x / norm
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer3_repeated, 策略=None, 种子=242, 模式=repeated_run

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]
```

---

## 29. `L1_P38__const_perturb__23`

- **算子**: `const_perturb` (Category A)
- **描述**: Perturb numeric literals (integer ±1; float ×1.01 or ×0.99) @ L48
- **变异行**: Line 48, 原始片段 `16`, 节点类型 `const:int-1`
- **杀死层**: `layer1_value`
- **杀死策略**: `boundary_last_element`
- **杀死种子**: `75`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for L1 normalization with warp-level reduction
l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void l1_normalize_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim
) {
    // Use 2D grid: x for batch, y for elements
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Shared memory for warp-level reduction results
    __shared__ float warp_sums[32];  // 32 warps max for 1024 threads
    
    // Each thread processes multiple elements with vectorized loads
    float thread_sum = 0.0f;
    
    // Process 4 elements at a time for better memory throughput
    const int stride = blockDim.x * 4;
    int idx = batch_idx * dim + tid * 4;
    
    for (int i = 0; i < dim; i += stride) {
        if (idx + 3 < (batch_idx + 1) * dim) {
            // Load 4 elements at once
            scalar_t val0 = input[idx];
            scalar_t val1 = input[idx + 1];
            scalar_t val2 = input[idx + 2];
            scalar_t val3 = input[idx + 3];
            
            thread_sum += fabsf((float)val0) + fabsf((float)val1) + 
                         fabsf((float)val2) + fabsf((float)val3);
            idx += stride;
        }
    }
    
    // Warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // First thread in each warp stores the warp sum
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction across warps (only first warp does this)
    if (warp_id == 0) {
        float warp_sum = lane_id < (blockDim.x / 32) ? warp_sums[lane_id] : 0.0f;
        
        // Reduce across warps
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        // Broadcast final sum to all threads
        if (lane_id == 0) {
            warp_sums[0] = warp_sum;
        }
    }
    __syncthreads();
    
    // All threads get the final sum
    float batch_sum = warp_sums[0];
    batch_sum = batch_sum == 0.0f ? 1.0f : batch_sum;
    float inv_batch_sum = 1.0f / batch_sum;
    
    // Normalize and write output with vectorized stores
    idx = batch_idx * dim + tid * 4;
    for (int i = 0; i < dim; i += stride) {
        if (idx + 3 < (batch_idx + 1) * dim) {
            output[idx] = input[idx] * inv_batch_sum;
            output[idx + 1] = input[idx + 1] * inv_batch_sum;
            output[idx + 2] = input[idx + 2] * inv_batch_sum;
            output[idx + 3] = input[idx + 3] * inv_batch_sum;
            idx += stride;
        }
    }
    
    // Handle remaining elements
    for (int i = tid; i < dim; i += blockDim.x) {
        int pos = batch_idx * dim + i;
        output[pos] = input[pos] * inv_batch_sum;
    }
}

torch::Tensor l1_normalize_cuda(torch::Tensor input) {
    // Check input dimensions
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");
    
    // Make sure tensor is contiguous
    auto input_contig = input.contiguous();
    
    auto sizes = input_contig.sizes();
    auto batch_size = sizes[0];
    auto dim = sizes[1];
    
    // Create output tensor
    auto output = torch::empty_like(input_contig);
    
    // Optimized configuration - use 256 threads for good occupancy
    const int block_size = 256;
    const int grid_size = batch_size;
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(
        input_contig.scalar_type(),
        "l1_normalize_kernel_optimized",
        [&] {
            l1_normalize_kernel_optimized<scalar_t><<<grid_size, block_size>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                dim
            );
        }
    );
    
    return output;
}
"""

l1_norm_cpp_source = "torch::Tensor l1_normalize_cuda(torch::Tensor input);"

# Compile the inline CUDA code
l1_normalize = load_inline(
    name="l1_normalize",
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=["l1_normalize_cuda"],
    verbose=False,
    extra_cflags=["-O3", "-DNDEBUG"],
    extra_ldflags=["-lcudart"],
    with_cuda=True
)

class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using custom CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the optimized L1 normalization layer.
        """
        super(ModelNew, self).__init__()
        self.l1_normalize = l1_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        # Ensure tensor is on GPU
        if not x.is_cuda:
            x = x.cuda()
        
        # Apply fused L1 normalization kernel
        return self.l1_normalize.l1_normalize_cuda(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for L1 normalization with warp-level reduction
l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void l1_normalize_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim
) {
    // Use 2D grid: x for batch, y for elements
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Shared memory for warp-level reduction results
    __shared__ float warp_sums[32];  // 32 warps max for 1024 threads
    
    // Each thread processes multiple elements with vectorized loads
    float thread_sum = 0.0f;
    
    // Process 4 elements at a time for better memory throughput
    const int stride = blockDim.x * 4;
    int idx = batch_idx * dim + tid * 4;
    
    for (int i = 0; i < dim; i += stride) {
        if (idx + 3 < (batch_idx + 1) * dim) {
            // Load 4 elements at once
            scalar_t val0 = input[idx];
            scalar_t val1 = input[idx + 1];
            scalar_t val2 = input[idx + 2];
            scalar_t val3 = input[idx + 3];
            
            thread_sum += fabsf((float)val0) + fabsf((float)val1) + 
                         fabsf((float)val2) + fabsf((float)val3);
            idx += stride;
        }
    }
    
    // Warp-level reduction using shuffle instructions
    for (int offset = 15; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // First thread in each warp stores the warp sum
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction across warps (only first warp does this)
    if (warp_id == 0) {
        float warp_sum = lane_id < (blockDim.x / 32) ? warp_sums[lane_id] : 0.0f;
        
        // Reduce across warps
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        // Broadcast final sum to all threads
        if (lane_id == 0) {
            warp_sums[0] = warp_sum;
        }
    }
    __syncthreads();
    
    // All threads get the final sum
    float batch_sum = warp_sums[0];
    batch_sum = batch_sum == 0.0f ? 1.0f : batch_sum;
    float inv_batch_sum = 1.0f / batch_sum;
    
    // Normalize and write output with vectorized stores
    idx = batch_idx * dim + tid * 4;
    for (int i = 0; i < dim; i += stride) {
        if (idx + 3 < (batch_idx + 1) * dim) {
            output[idx] = input[idx] * inv_batch_sum;
            output[idx + 1] = input[idx + 1] * inv_batch_sum;
            output[idx + 2] = input[idx + 2] * inv_batch_sum;
            output[idx + 3] = input[idx + 3] * inv_batch_sum;
            idx += stride;
        }
    }
    
    // Handle remaining elements
    for (int i = tid; i < dim; i += blockDim.x) {
        int pos = batch_idx * dim + i;
        output[pos] = input[pos] * inv_batch_sum;
    }
}

torch::Tensor l1_normalize_cuda(torch::Tensor input) {
    // Check input dimensions
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");
    
    // Make sure tensor is contiguous
    auto input_contig = input.contiguous();
    
    auto sizes = input_contig.sizes();
    auto batch_size = sizes[0];
    auto dim = sizes[1];
    
    // Create output tensor
    auto output = torch::empty_like(input_contig);
    
    // Optimized configuration - use 256 threads for good occupancy
    const int block_size = 256;
    const int grid_size = batch_size;
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(
        input_contig.scalar_type(),
        "l1_normalize_kernel_optimized",
        [&] {
            l1_normalize_kernel_optimized<scalar_t><<<grid_size, block_size>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                dim
            );
        }
    );
    
    return output;
}
"""

l1_norm_cpp_source = "torch::Tensor l1_normalize_cuda(torch::Tensor input);"

# Compile the inline CUDA code
l1_normalize = load_inline(
    name="l1_normalize",
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=["l1_normalize_cuda"],
    verbose=False,
    extra_cflags=["-O3", "-DNDEBUG"],
    extra_ldflags=["-lcudart"],
    with_cuda=True
)

class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using custom CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the optimized L1 normalization layer.
        """
        super(ModelNew, self).__init__()
        self.l1_normalize = l1_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        # Ensure tensor is on GPU
        if not x.is_cuda:
            x = x.cuda()
        
        # Apply fused L1 normalization kernel
        return self.l1_normalize.l1_normalize_cuda(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=boundary_last_element, 种子=75, 模式=value_stress

**杀死策略描述**: last element set to extreme

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]
```

---

## 30. `L1_P38__const_perturb__25`

- **算子**: `const_perturb` (Category A)
- **描述**: Perturb numeric literals (integer ±1; float ×1.01 or ×0.99) @ L63
- **变异行**: Line 63, 原始片段 `16`, 节点类型 `const:int-1`
- **杀死层**: `layer1_value`
- **杀死策略**: `boundary_last_element`
- **杀死种子**: `75`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for L1 normalization with warp-level reduction
l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void l1_normalize_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim
) {
    // Use 2D grid: x for batch, y for elements
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Shared memory for warp-level reduction results
    __shared__ float warp_sums[32];  // 32 warps max for 1024 threads
    
    // Each thread processes multiple elements with vectorized loads
    float thread_sum = 0.0f;
    
    // Process 4 elements at a time for better memory throughput
    const int stride = blockDim.x * 4;
    int idx = batch_idx * dim + tid * 4;
    
    for (int i = 0; i < dim; i += stride) {
        if (idx + 3 < (batch_idx + 1) * dim) {
            // Load 4 elements at once
            scalar_t val0 = input[idx];
            scalar_t val1 = input[idx + 1];
            scalar_t val2 = input[idx + 2];
            scalar_t val3 = input[idx + 3];
            
            thread_sum += fabsf((float)val0) + fabsf((float)val1) + 
                         fabsf((float)val2) + fabsf((float)val3);
            idx += stride;
        }
    }
    
    // Warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // First thread in each warp stores the warp sum
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction across warps (only first warp does this)
    if (warp_id == 0) {
        float warp_sum = lane_id < (blockDim.x / 32) ? warp_sums[lane_id] : 0.0f;
        
        // Reduce across warps
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        // Broadcast final sum to all threads
        if (lane_id == 0) {
            warp_sums[0] = warp_sum;
        }
    }
    __syncthreads();
    
    // All threads get the final sum
    float batch_sum = warp_sums[0];
    batch_sum = batch_sum == 0.0f ? 1.0f : batch_sum;
    float inv_batch_sum = 1.0f / batch_sum;
    
    // Normalize and write output with vectorized stores
    idx = batch_idx * dim + tid * 4;
    for (int i = 0; i < dim; i += stride) {
        if (idx + 3 < (batch_idx + 1) * dim) {
            output[idx] = input[idx] * inv_batch_sum;
            output[idx + 1] = input[idx + 1] * inv_batch_sum;
            output[idx + 2] = input[idx + 2] * inv_batch_sum;
            output[idx + 3] = input[idx + 3] * inv_batch_sum;
            idx += stride;
        }
    }
    
    // Handle remaining elements
    for (int i = tid; i < dim; i += blockDim.x) {
        int pos = batch_idx * dim + i;
        output[pos] = input[pos] * inv_batch_sum;
    }
}

torch::Tensor l1_normalize_cuda(torch::Tensor input) {
    // Check input dimensions
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");
    
    // Make sure tensor is contiguous
    auto input_contig = input.contiguous();
    
    auto sizes = input_contig.sizes();
    auto batch_size = sizes[0];
    auto dim = sizes[1];
    
    // Create output tensor
    auto output = torch::empty_like(input_contig);
    
    // Optimized configuration - use 256 threads for good occupancy
    const int block_size = 256;
    const int grid_size = batch_size;
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(
        input_contig.scalar_type(),
        "l1_normalize_kernel_optimized",
        [&] {
            l1_normalize_kernel_optimized<scalar_t><<<grid_size, block_size>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                dim
            );
        }
    );
    
    return output;
}
"""

l1_norm_cpp_source = "torch::Tensor l1_normalize_cuda(torch::Tensor input);"

# Compile the inline CUDA code
l1_normalize = load_inline(
    name="l1_normalize",
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=["l1_normalize_cuda"],
    verbose=False,
    extra_cflags=["-O3", "-DNDEBUG"],
    extra_ldflags=["-lcudart"],
    with_cuda=True
)

class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using custom CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the optimized L1 normalization layer.
        """
        super(ModelNew, self).__init__()
        self.l1_normalize = l1_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        # Ensure tensor is on GPU
        if not x.is_cuda:
            x = x.cuda()
        
        # Apply fused L1 normalization kernel
        return self.l1_normalize.l1_normalize_cuda(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for L1 normalization with warp-level reduction
l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void l1_normalize_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim
) {
    // Use 2D grid: x for batch, y for elements
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Shared memory for warp-level reduction results
    __shared__ float warp_sums[32];  // 32 warps max for 1024 threads
    
    // Each thread processes multiple elements with vectorized loads
    float thread_sum = 0.0f;
    
    // Process 4 elements at a time for better memory throughput
    const int stride = blockDim.x * 4;
    int idx = batch_idx * dim + tid * 4;
    
    for (int i = 0; i < dim; i += stride) {
        if (idx + 3 < (batch_idx + 1) * dim) {
            // Load 4 elements at once
            scalar_t val0 = input[idx];
            scalar_t val1 = input[idx + 1];
            scalar_t val2 = input[idx + 2];
            scalar_t val3 = input[idx + 3];
            
            thread_sum += fabsf((float)val0) + fabsf((float)val1) + 
                         fabsf((float)val2) + fabsf((float)val3);
            idx += stride;
        }
    }
    
    // Warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // First thread in each warp stores the warp sum
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction across warps (only first warp does this)
    if (warp_id == 0) {
        float warp_sum = lane_id < (blockDim.x / 32) ? warp_sums[lane_id] : 0.0f;
        
        // Reduce across warps
        for (int offset = 15; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        // Broadcast final sum to all threads
        if (lane_id == 0) {
            warp_sums[0] = warp_sum;
        }
    }
    __syncthreads();
    
    // All threads get the final sum
    float batch_sum = warp_sums[0];
    batch_sum = batch_sum == 0.0f ? 1.0f : batch_sum;
    float inv_batch_sum = 1.0f / batch_sum;
    
    // Normalize and write output with vectorized stores
    idx = batch_idx * dim + tid * 4;
    for (int i = 0; i < dim; i += stride) {
        if (idx + 3 < (batch_idx + 1) * dim) {
            output[idx] = input[idx] * inv_batch_sum;
            output[idx + 1] = input[idx + 1] * inv_batch_sum;
            output[idx + 2] = input[idx + 2] * inv_batch_sum;
            output[idx + 3] = input[idx + 3] * inv_batch_sum;
            idx += stride;
        }
    }
    
    // Handle remaining elements
    for (int i = tid; i < dim; i += blockDim.x) {
        int pos = batch_idx * dim + i;
        output[pos] = input[pos] * inv_batch_sum;
    }
}

torch::Tensor l1_normalize_cuda(torch::Tensor input) {
    // Check input dimensions
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");
    
    // Make sure tensor is contiguous
    auto input_contig = input.contiguous();
    
    auto sizes = input_contig.sizes();
    auto batch_size = sizes[0];
    auto dim = sizes[1];
    
    // Create output tensor
    auto output = torch::empty_like(input_contig);
    
    // Optimized configuration - use 256 threads for good occupancy
    const int block_size = 256;
    const int grid_size = batch_size;
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(
        input_contig.scalar_type(),
        "l1_normalize_kernel_optimized",
        [&] {
            l1_normalize_kernel_optimized<scalar_t><<<grid_size, block_size>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                dim
            );
        }
    );
    
    return output;
}
"""

l1_norm_cpp_source = "torch::Tensor l1_normalize_cuda(torch::Tensor input);"

# Compile the inline CUDA code
l1_normalize = load_inline(
    name="l1_normalize",
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=["l1_normalize_cuda"],
    verbose=False,
    extra_cflags=["-O3", "-DNDEBUG"],
    extra_ldflags=["-lcudart"],
    with_cuda=True
)

class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using custom CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the optimized L1 normalization layer.
        """
        super(ModelNew, self).__init__()
        self.l1_normalize = l1_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        # Ensure tensor is on GPU
        if not x.is_cuda:
            x = x.cuda()
        
        # Apply fused L1 normalization kernel
        return self.l1_normalize.l1_normalize_cuda(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=boundary_last_element, 种子=75, 模式=value_stress

**杀死策略描述**: last element set to extreme

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]
```

---

## 31. `L1_P38__const_perturb__34`

- **算子**: `const_perturb` (Category A)
- **描述**: Perturb numeric literals (integer ±1; float ×1.01 or ×0.99) @ L86
- **变异行**: Line 86, 原始片段 `3`, 节点类型 `const:int+1`
- **杀死层**: `layer3_repeated`
- **杀死策略**: `None`
- **杀死种子**: `242`
- **杀死模式**: `repeated_run`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for L1 normalization with warp-level reduction
l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void l1_normalize_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim
) {
    // Use 2D grid: x for batch, y for elements
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Shared memory for warp-level reduction results
    __shared__ float warp_sums[32];  // 32 warps max for 1024 threads
    
    // Each thread processes multiple elements with vectorized loads
    float thread_sum = 0.0f;
    
    // Process 4 elements at a time for better memory throughput
    const int stride = blockDim.x * 4;
    int idx = batch_idx * dim + tid * 4;
    
    for (int i = 0; i < dim; i += stride) {
        if (idx + 3 < (batch_idx + 1) * dim) {
            // Load 4 elements at once
            scalar_t val0 = input[idx];
            scalar_t val1 = input[idx + 1];
            scalar_t val2 = input[idx + 2];
            scalar_t val3 = input[idx + 3];
            
            thread_sum += fabsf((float)val0) + fabsf((float)val1) + 
                         fabsf((float)val2) + fabsf((float)val3);
            idx += stride;
        }
    }
    
    // Warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // First thread in each warp stores the warp sum
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction across warps (only first warp does this)
    if (warp_id == 0) {
        float warp_sum = lane_id < (blockDim.x / 32) ? warp_sums[lane_id] : 0.0f;
        
        // Reduce across warps
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        // Broadcast final sum to all threads
        if (lane_id == 0) {
            warp_sums[0] = warp_sum;
        }
    }
    __syncthreads();
    
    // All threads get the final sum
    float batch_sum = warp_sums[0];
    batch_sum = batch_sum == 0.0f ? 1.0f : batch_sum;
    float inv_batch_sum = 1.0f / batch_sum;
    
    // Normalize and write output with vectorized stores
    idx = batch_idx * dim + tid * 4;
    for (int i = 0; i < dim; i += stride) {
        if (idx + 3 < (batch_idx + 1) * dim) {
            output[idx] = input[idx] * inv_batch_sum;
            output[idx + 1] = input[idx + 1] * inv_batch_sum;
            output[idx + 2] = input[idx + 2] * inv_batch_sum;
            output[idx + 3] = input[idx + 3] * inv_batch_sum;
            idx += stride;
        }
    }
    
    // Handle remaining elements
    for (int i = tid; i < dim; i += blockDim.x) {
        int pos = batch_idx * dim + i;
        output[pos] = input[pos] * inv_batch_sum;
    }
}

torch::Tensor l1_normalize_cuda(torch::Tensor input) {
    // Check input dimensions
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");
    
    // Make sure tensor is contiguous
    auto input_contig = input.contiguous();
    
    auto sizes = input_contig.sizes();
    auto batch_size = sizes[0];
    auto dim = sizes[1];
    
    // Create output tensor
    auto output = torch::empty_like(input_contig);
    
    // Optimized configuration - use 256 threads for good occupancy
    const int block_size = 256;
    const int grid_size = batch_size;
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(
        input_contig.scalar_type(),
        "l1_normalize_kernel_optimized",
        [&] {
            l1_normalize_kernel_optimized<scalar_t><<<grid_size, block_size>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                dim
            );
        }
    );
    
    return output;
}
"""

l1_norm_cpp_source = "torch::Tensor l1_normalize_cuda(torch::Tensor input);"

# Compile the inline CUDA code
l1_normalize = load_inline(
    name="l1_normalize",
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=["l1_normalize_cuda"],
    verbose=False,
    extra_cflags=["-O3", "-DNDEBUG"],
    extra_ldflags=["-lcudart"],
    with_cuda=True
)

class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using custom CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the optimized L1 normalization layer.
        """
        super(ModelNew, self).__init__()
        self.l1_normalize = l1_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        # Ensure tensor is on GPU
        if not x.is_cuda:
            x = x.cuda()
        
        # Apply fused L1 normalization kernel
        return self.l1_normalize.l1_normalize_cuda(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for L1 normalization with warp-level reduction
l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void l1_normalize_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim
) {
    // Use 2D grid: x for batch, y for elements
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Shared memory for warp-level reduction results
    __shared__ float warp_sums[32];  // 32 warps max for 1024 threads
    
    // Each thread processes multiple elements with vectorized loads
    float thread_sum = 0.0f;
    
    // Process 4 elements at a time for better memory throughput
    const int stride = blockDim.x * 4;
    int idx = batch_idx * dim + tid * 4;
    
    for (int i = 0; i < dim; i += stride) {
        if (idx + 3 < (batch_idx + 1) * dim) {
            // Load 4 elements at once
            scalar_t val0 = input[idx];
            scalar_t val1 = input[idx + 1];
            scalar_t val2 = input[idx + 2];
            scalar_t val3 = input[idx + 3];
            
            thread_sum += fabsf((float)val0) + fabsf((float)val1) + 
                         fabsf((float)val2) + fabsf((float)val3);
            idx += stride;
        }
    }
    
    // Warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // First thread in each warp stores the warp sum
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction across warps (only first warp does this)
    if (warp_id == 0) {
        float warp_sum = lane_id < (blockDim.x / 32) ? warp_sums[lane_id] : 0.0f;
        
        // Reduce across warps
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        // Broadcast final sum to all threads
        if (lane_id == 0) {
            warp_sums[0] = warp_sum;
        }
    }
    __syncthreads();
    
    // All threads get the final sum
    float batch_sum = warp_sums[0];
    batch_sum = batch_sum == 0.0f ? 1.0f : batch_sum;
    float inv_batch_sum = 1.0f / batch_sum;
    
    // Normalize and write output with vectorized stores
    idx = batch_idx * dim + tid * 4;
    for (int i = 0; i < dim; i += stride) {
        if (idx + 3 < (batch_idx + 1) * dim) {
            output[idx] = input[idx] * inv_batch_sum;
            output[idx + 1] = input[idx + 1] * inv_batch_sum;
            output[idx + 2] = input[idx + 2] * inv_batch_sum;
            output[idx + 4] = input[idx + 3] * inv_batch_sum;
            idx += stride;
        }
    }
    
    // Handle remaining elements
    for (int i = tid; i < dim; i += blockDim.x) {
        int pos = batch_idx * dim + i;
        output[pos] = input[pos] * inv_batch_sum;
    }
}

torch::Tensor l1_normalize_cuda(torch::Tensor input) {
    // Check input dimensions
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");
    
    // Make sure tensor is contiguous
    auto input_contig = input.contiguous();
    
    auto sizes = input_contig.sizes();
    auto batch_size = sizes[0];
    auto dim = sizes[1];
    
    // Create output tensor
    auto output = torch::empty_like(input_contig);
    
    // Optimized configuration - use 256 threads for good occupancy
    const int block_size = 256;
    const int grid_size = batch_size;
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(
        input_contig.scalar_type(),
        "l1_normalize_kernel_optimized",
        [&] {
            l1_normalize_kernel_optimized<scalar_t><<<grid_size, block_size>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                dim
            );
        }
    );
    
    return output;
}
"""

l1_norm_cpp_source = "torch::Tensor l1_normalize_cuda(torch::Tensor input);"

# Compile the inline CUDA code
l1_normalize = load_inline(
    name="l1_normalize",
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=["l1_normalize_cuda"],
    verbose=False,
    extra_cflags=["-O3", "-DNDEBUG"],
    extra_ldflags=["-lcudart"],
    with_cuda=True
)

class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using custom CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the optimized L1 normalization layer.
        """
        super(ModelNew, self).__init__()
        self.l1_normalize = l1_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        # Ensure tensor is on GPU
        if not x.is_cuda:
            x = x.cuda()
        
        # Apply fused L1 normalization kernel
        return self.l1_normalize.l1_normalize_cuda(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer3_repeated, 策略=None, 种子=242, 模式=repeated_run

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]
```

---

## 32. `L1_P38__relop_replace__13`

- **算子**: `relop_replace` (Category A)
- **描述**: Replace relational operators (<→<=, <=→<, >→>=, >=→>, ==→!=, !=→==) @ L92
- **变异行**: Line 92, 原始片段 `<`, 节点类型 `cuda_Lt`
- **杀死层**: `layer1_value`
- **杀死策略**: `tail_heavy`
- **杀死种子**: `81`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for L1 normalization with warp-level reduction
l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void l1_normalize_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim
) {
    // Use 2D grid: x for batch, y for elements
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Shared memory for warp-level reduction results
    __shared__ float warp_sums[32];  // 32 warps max for 1024 threads
    
    // Each thread processes multiple elements with vectorized loads
    float thread_sum = 0.0f;
    
    // Process 4 elements at a time for better memory throughput
    const int stride = blockDim.x * 4;
    int idx = batch_idx * dim + tid * 4;
    
    for (int i = 0; i < dim; i += stride) {
        if (idx + 3 < (batch_idx + 1) * dim) {
            // Load 4 elements at once
            scalar_t val0 = input[idx];
            scalar_t val1 = input[idx + 1];
            scalar_t val2 = input[idx + 2];
            scalar_t val3 = input[idx + 3];
            
            thread_sum += fabsf((float)val0) + fabsf((float)val1) + 
                         fabsf((float)val2) + fabsf((float)val3);
            idx += stride;
        }
    }
    
    // Warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // First thread in each warp stores the warp sum
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction across warps (only first warp does this)
    if (warp_id == 0) {
        float warp_sum = lane_id < (blockDim.x / 32) ? warp_sums[lane_id] : 0.0f;
        
        // Reduce across warps
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        // Broadcast final sum to all threads
        if (lane_id == 0) {
            warp_sums[0] = warp_sum;
        }
    }
    __syncthreads();
    
    // All threads get the final sum
    float batch_sum = warp_sums[0];
    batch_sum = batch_sum == 0.0f ? 1.0f : batch_sum;
    float inv_batch_sum = 1.0f / batch_sum;
    
    // Normalize and write output with vectorized stores
    idx = batch_idx * dim + tid * 4;
    for (int i = 0; i < dim; i += stride) {
        if (idx + 3 < (batch_idx + 1) * dim) {
            output[idx] = input[idx] * inv_batch_sum;
            output[idx + 1] = input[idx + 1] * inv_batch_sum;
            output[idx + 2] = input[idx + 2] * inv_batch_sum;
            output[idx + 3] = input[idx + 3] * inv_batch_sum;
            idx += stride;
        }
    }
    
    // Handle remaining elements
    for (int i = tid; i < dim; i += blockDim.x) {
        int pos = batch_idx * dim + i;
        output[pos] = input[pos] * inv_batch_sum;
    }
}

torch::Tensor l1_normalize_cuda(torch::Tensor input) {
    // Check input dimensions
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");
    
    // Make sure tensor is contiguous
    auto input_contig = input.contiguous();
    
    auto sizes = input_contig.sizes();
    auto batch_size = sizes[0];
    auto dim = sizes[1];
    
    // Create output tensor
    auto output = torch::empty_like(input_contig);
    
    // Optimized configuration - use 256 threads for good occupancy
    const int block_size = 256;
    const int grid_size = batch_size;
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(
        input_contig.scalar_type(),
        "l1_normalize_kernel_optimized",
        [&] {
            l1_normalize_kernel_optimized<scalar_t><<<grid_size, block_size>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                dim
            );
        }
    );
    
    return output;
}
"""

l1_norm_cpp_source = "torch::Tensor l1_normalize_cuda(torch::Tensor input);"

# Compile the inline CUDA code
l1_normalize = load_inline(
    name="l1_normalize",
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=["l1_normalize_cuda"],
    verbose=False,
    extra_cflags=["-O3", "-DNDEBUG"],
    extra_ldflags=["-lcudart"],
    with_cuda=True
)

class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using custom CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the optimized L1 normalization layer.
        """
        super(ModelNew, self).__init__()
        self.l1_normalize = l1_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        # Ensure tensor is on GPU
        if not x.is_cuda:
            x = x.cuda()
        
        # Apply fused L1 normalization kernel
        return self.l1_normalize.l1_normalize_cuda(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for L1 normalization with warp-level reduction
l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void l1_normalize_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim
) {
    // Use 2D grid: x for batch, y for elements
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Shared memory for warp-level reduction results
    __shared__ float warp_sums[32];  // 32 warps max for 1024 threads
    
    // Each thread processes multiple elements with vectorized loads
    float thread_sum = 0.0f;
    
    // Process 4 elements at a time for better memory throughput
    const int stride = blockDim.x * 4;
    int idx = batch_idx * dim + tid * 4;
    
    for (int i = 0; i < dim; i += stride) {
        if (idx + 3 < (batch_idx + 1) * dim) {
            // Load 4 elements at once
            scalar_t val0 = input[idx];
            scalar_t val1 = input[idx + 1];
            scalar_t val2 = input[idx + 2];
            scalar_t val3 = input[idx + 3];
            
            thread_sum += fabsf((float)val0) + fabsf((float)val1) + 
                         fabsf((float)val2) + fabsf((float)val3);
            idx += stride;
        }
    }
    
    // Warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // First thread in each warp stores the warp sum
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction across warps (only first warp does this)
    if (warp_id == 0) {
        float warp_sum = lane_id < (blockDim.x / 32) ? warp_sums[lane_id] : 0.0f;
        
        // Reduce across warps
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        // Broadcast final sum to all threads
        if (lane_id == 0) {
            warp_sums[0] = warp_sum;
        }
    }
    __syncthreads();
    
    // All threads get the final sum
    float batch_sum = warp_sums[0];
    batch_sum = batch_sum == 0.0f ? 1.0f : batch_sum;
    float inv_batch_sum = 1.0f / batch_sum;
    
    // Normalize and write output with vectorized stores
    idx = batch_idx * dim + tid * 4;
    for (int i = 0; i < dim; i += stride) {
        if (idx + 3 < (batch_idx + 1) * dim) {
            output[idx] = input[idx] * inv_batch_sum;
            output[idx + 1] = input[idx + 1] * inv_batch_sum;
            output[idx + 2] = input[idx + 2] * inv_batch_sum;
            output[idx + 3] = input[idx + 3] * inv_batch_sum;
            idx += stride;
        }
    }
    
    // Handle remaining elements
    for (int i = tid; i <= dim; i += blockDim.x) {
        int pos = batch_idx * dim + i;
        output[pos] = input[pos] * inv_batch_sum;
    }
}

torch::Tensor l1_normalize_cuda(torch::Tensor input) {
    // Check input dimensions
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");
    
    // Make sure tensor is contiguous
    auto input_contig = input.contiguous();
    
    auto sizes = input_contig.sizes();
    auto batch_size = sizes[0];
    auto dim = sizes[1];
    
    // Create output tensor
    auto output = torch::empty_like(input_contig);
    
    // Optimized configuration - use 256 threads for good occupancy
    const int block_size = 256;
    const int grid_size = batch_size;
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(
        input_contig.scalar_type(),
        "l1_normalize_kernel_optimized",
        [&] {
            l1_normalize_kernel_optimized<scalar_t><<<grid_size, block_size>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                dim
            );
        }
    );
    
    return output;
}
"""

l1_norm_cpp_source = "torch::Tensor l1_normalize_cuda(torch::Tensor input);"

# Compile the inline CUDA code
l1_normalize = load_inline(
    name="l1_normalize",
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=["l1_normalize_cuda"],
    verbose=False,
    extra_cflags=["-O3", "-DNDEBUG"],
    extra_ldflags=["-lcudart"],
    with_cuda=True
)

class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using custom CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the optimized L1 normalization layer.
        """
        super(ModelNew, self).__init__()
        self.l1_normalize = l1_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        # Ensure tensor is on GPU
        if not x.is_cuda:
            x = x.cuda()
        
        # Apply fused L1 normalization kernel
        return self.l1_normalize.l1_normalize_cuda(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=tail_heavy, 种子=81, 模式=value_stress

**杀死策略描述**: last 10% large, rest small

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]
```

---

## 33. `L1_P38__sync_remove__0`

- **算子**: `sync_remove` (Category B)
- **描述**: Remove GPU synchronization barriers (tl.debug_barrier, __syncthreads) @ L56
- **变异行**: Line 56, 原始片段 `__syncthreads();`, 节点类型 `cuda_syncthreads`
- **杀死层**: `layer1_value`
- **杀死策略**: `sparse`
- **杀死种子**: `66`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for L1 normalization with warp-level reduction
l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void l1_normalize_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim
) {
    // Use 2D grid: x for batch, y for elements
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Shared memory for warp-level reduction results
    __shared__ float warp_sums[32];  // 32 warps max for 1024 threads
    
    // Each thread processes multiple elements with vectorized loads
    float thread_sum = 0.0f;
    
    // Process 4 elements at a time for better memory throughput
    const int stride = blockDim.x * 4;
    int idx = batch_idx * dim + tid * 4;
    
    for (int i = 0; i < dim; i += stride) {
        if (idx + 3 < (batch_idx + 1) * dim) {
            // Load 4 elements at once
            scalar_t val0 = input[idx];
            scalar_t val1 = input[idx + 1];
            scalar_t val2 = input[idx + 2];
            scalar_t val3 = input[idx + 3];
            
            thread_sum += fabsf((float)val0) + fabsf((float)val1) + 
                         fabsf((float)val2) + fabsf((float)val3);
            idx += stride;
        }
    }
    
    // Warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // First thread in each warp stores the warp sum
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction across warps (only first warp does this)
    if (warp_id == 0) {
        float warp_sum = lane_id < (blockDim.x / 32) ? warp_sums[lane_id] : 0.0f;
        
        // Reduce across warps
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        // Broadcast final sum to all threads
        if (lane_id == 0) {
            warp_sums[0] = warp_sum;
        }
    }
    __syncthreads();
    
    // All threads get the final sum
    float batch_sum = warp_sums[0];
    batch_sum = batch_sum == 0.0f ? 1.0f : batch_sum;
    float inv_batch_sum = 1.0f / batch_sum;
    
    // Normalize and write output with vectorized stores
    idx = batch_idx * dim + tid * 4;
    for (int i = 0; i < dim; i += stride) {
        if (idx + 3 < (batch_idx + 1) * dim) {
            output[idx] = input[idx] * inv_batch_sum;
            output[idx + 1] = input[idx + 1] * inv_batch_sum;
            output[idx + 2] = input[idx + 2] * inv_batch_sum;
            output[idx + 3] = input[idx + 3] * inv_batch_sum;
            idx += stride;
        }
    }
    
    // Handle remaining elements
    for (int i = tid; i < dim; i += blockDim.x) {
        int pos = batch_idx * dim + i;
        output[pos] = input[pos] * inv_batch_sum;
    }
}

torch::Tensor l1_normalize_cuda(torch::Tensor input) {
    // Check input dimensions
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");
    
    // Make sure tensor is contiguous
    auto input_contig = input.contiguous();
    
    auto sizes = input_contig.sizes();
    auto batch_size = sizes[0];
    auto dim = sizes[1];
    
    // Create output tensor
    auto output = torch::empty_like(input_contig);
    
    // Optimized configuration - use 256 threads for good occupancy
    const int block_size = 256;
    const int grid_size = batch_size;
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(
        input_contig.scalar_type(),
        "l1_normalize_kernel_optimized",
        [&] {
            l1_normalize_kernel_optimized<scalar_t><<<grid_size, block_size>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                dim
            );
        }
    );
    
    return output;
}
"""

l1_norm_cpp_source = "torch::Tensor l1_normalize_cuda(torch::Tensor input);"

# Compile the inline CUDA code
l1_normalize = load_inline(
    name="l1_normalize",
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=["l1_normalize_cuda"],
    verbose=False,
    extra_cflags=["-O3", "-DNDEBUG"],
    extra_ldflags=["-lcudart"],
    with_cuda=True
)

class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using custom CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the optimized L1 normalization layer.
        """
        super(ModelNew, self).__init__()
        self.l1_normalize = l1_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        # Ensure tensor is on GPU
        if not x.is_cuda:
            x = x.cuda()
        
        # Apply fused L1 normalization kernel
        return self.l1_normalize.l1_normalize_cuda(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for L1 normalization with warp-level reduction
l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void l1_normalize_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim
) {
    // Use 2D grid: x for batch, y for elements
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Shared memory for warp-level reduction results
    __shared__ float warp_sums[32];  // 32 warps max for 1024 threads
    
    // Each thread processes multiple elements with vectorized loads
    float thread_sum = 0.0f;
    
    // Process 4 elements at a time for better memory throughput
    const int stride = blockDim.x * 4;
    int idx = batch_idx * dim + tid * 4;
    
    for (int i = 0; i < dim; i += stride) {
        if (idx + 3 < (batch_idx + 1) * dim) {
            // Load 4 elements at once
            scalar_t val0 = input[idx];
            scalar_t val1 = input[idx + 1];
            scalar_t val2 = input[idx + 2];
            scalar_t val3 = input[idx + 3];
            
            thread_sum += fabsf((float)val0) + fabsf((float)val1) + 
                         fabsf((float)val2) + fabsf((float)val3);
            idx += stride;
        }
    }
    
    // Warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // First thread in each warp stores the warp sum
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    
    // Final reduction across warps (only first warp does this)
    if (warp_id == 0) {
        float warp_sum = lane_id < (blockDim.x / 32) ? warp_sums[lane_id] : 0.0f;
        
        // Reduce across warps
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        // Broadcast final sum to all threads
        if (lane_id == 0) {
            warp_sums[0] = warp_sum;
        }
    }
    __syncthreads();
    
    // All threads get the final sum
    float batch_sum = warp_sums[0];
    batch_sum = batch_sum == 0.0f ? 1.0f : batch_sum;
    float inv_batch_sum = 1.0f / batch_sum;
    
    // Normalize and write output with vectorized stores
    idx = batch_idx * dim + tid * 4;
    for (int i = 0; i < dim; i += stride) {
        if (idx + 3 < (batch_idx + 1) * dim) {
            output[idx] = input[idx] * inv_batch_sum;
            output[idx + 1] = input[idx + 1] * inv_batch_sum;
            output[idx + 2] = input[idx + 2] * inv_batch_sum;
            output[idx + 3] = input[idx + 3] * inv_batch_sum;
            idx += stride;
        }
    }
    
    // Handle remaining elements
    for (int i = tid; i < dim; i += blockDim.x) {
        int pos = batch_idx * dim + i;
        output[pos] = input[pos] * inv_batch_sum;
    }
}

torch::Tensor l1_normalize_cuda(torch::Tensor input) {
    // Check input dimensions
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");
    
    // Make sure tensor is contiguous
    auto input_contig = input.contiguous();
    
    auto sizes = input_contig.sizes();
    auto batch_size = sizes[0];
    auto dim = sizes[1];
    
    // Create output tensor
    auto output = torch::empty_like(input_contig);
    
    // Optimized configuration - use 256 threads for good occupancy
    const int block_size = 256;
    const int grid_size = batch_size;
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(
        input_contig.scalar_type(),
        "l1_normalize_kernel_optimized",
        [&] {
            l1_normalize_kernel_optimized<scalar_t><<<grid_size, block_size>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                dim
            );
        }
    );
    
    return output;
}
"""

l1_norm_cpp_source = "torch::Tensor l1_normalize_cuda(torch::Tensor input);"

# Compile the inline CUDA code
l1_normalize = load_inline(
    name="l1_normalize",
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=["l1_normalize_cuda"],
    verbose=False,
    extra_cflags=["-O3", "-DNDEBUG"],
    extra_ldflags=["-lcudart"],
    with_cuda=True
)

class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using custom CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the optimized L1 normalization layer.
        """
        super(ModelNew, self).__init__()
        self.l1_normalize = l1_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        # Ensure tensor is on GPU
        if not x.is_cuda:
            x = x.cuda()
        
        # Apply fused L1 normalization kernel
        return self.l1_normalize.l1_normalize_cuda(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=sparse, 种子=66, 模式=value_stress

**杀死策略描述**: 90% zeros, 10% random

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]
```

---

## 34. `L1_P38__sync_remove__1`

- **算子**: `sync_remove` (Category B)
- **描述**: Remove GPU synchronization barriers (tl.debug_barrier, __syncthreads) @ L72
- **变异行**: Line 72, 原始片段 `__syncthreads();`, 节点类型 `cuda_syncthreads`
- **杀死层**: `layer1_value`
- **杀死策略**: `sparse`
- **杀死种子**: `66`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for L1 normalization with warp-level reduction
l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void l1_normalize_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim
) {
    // Use 2D grid: x for batch, y for elements
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Shared memory for warp-level reduction results
    __shared__ float warp_sums[32];  // 32 warps max for 1024 threads
    
    // Each thread processes multiple elements with vectorized loads
    float thread_sum = 0.0f;
    
    // Process 4 elements at a time for better memory throughput
    const int stride = blockDim.x * 4;
    int idx = batch_idx * dim + tid * 4;
    
    for (int i = 0; i < dim; i += stride) {
        if (idx + 3 < (batch_idx + 1) * dim) {
            // Load 4 elements at once
            scalar_t val0 = input[idx];
            scalar_t val1 = input[idx + 1];
            scalar_t val2 = input[idx + 2];
            scalar_t val3 = input[idx + 3];
            
            thread_sum += fabsf((float)val0) + fabsf((float)val1) + 
                         fabsf((float)val2) + fabsf((float)val3);
            idx += stride;
        }
    }
    
    // Warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // First thread in each warp stores the warp sum
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction across warps (only first warp does this)
    if (warp_id == 0) {
        float warp_sum = lane_id < (blockDim.x / 32) ? warp_sums[lane_id] : 0.0f;
        
        // Reduce across warps
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        // Broadcast final sum to all threads
        if (lane_id == 0) {
            warp_sums[0] = warp_sum;
        }
    }
    __syncthreads();
    
    // All threads get the final sum
    float batch_sum = warp_sums[0];
    batch_sum = batch_sum == 0.0f ? 1.0f : batch_sum;
    float inv_batch_sum = 1.0f / batch_sum;
    
    // Normalize and write output with vectorized stores
    idx = batch_idx * dim + tid * 4;
    for (int i = 0; i < dim; i += stride) {
        if (idx + 3 < (batch_idx + 1) * dim) {
            output[idx] = input[idx] * inv_batch_sum;
            output[idx + 1] = input[idx + 1] * inv_batch_sum;
            output[idx + 2] = input[idx + 2] * inv_batch_sum;
            output[idx + 3] = input[idx + 3] * inv_batch_sum;
            idx += stride;
        }
    }
    
    // Handle remaining elements
    for (int i = tid; i < dim; i += blockDim.x) {
        int pos = batch_idx * dim + i;
        output[pos] = input[pos] * inv_batch_sum;
    }
}

torch::Tensor l1_normalize_cuda(torch::Tensor input) {
    // Check input dimensions
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");
    
    // Make sure tensor is contiguous
    auto input_contig = input.contiguous();
    
    auto sizes = input_contig.sizes();
    auto batch_size = sizes[0];
    auto dim = sizes[1];
    
    // Create output tensor
    auto output = torch::empty_like(input_contig);
    
    // Optimized configuration - use 256 threads for good occupancy
    const int block_size = 256;
    const int grid_size = batch_size;
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(
        input_contig.scalar_type(),
        "l1_normalize_kernel_optimized",
        [&] {
            l1_normalize_kernel_optimized<scalar_t><<<grid_size, block_size>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                dim
            );
        }
    );
    
    return output;
}
"""

l1_norm_cpp_source = "torch::Tensor l1_normalize_cuda(torch::Tensor input);"

# Compile the inline CUDA code
l1_normalize = load_inline(
    name="l1_normalize",
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=["l1_normalize_cuda"],
    verbose=False,
    extra_cflags=["-O3", "-DNDEBUG"],
    extra_ldflags=["-lcudart"],
    with_cuda=True
)

class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using custom CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the optimized L1 normalization layer.
        """
        super(ModelNew, self).__init__()
        self.l1_normalize = l1_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        # Ensure tensor is on GPU
        if not x.is_cuda:
            x = x.cuda()
        
        # Apply fused L1 normalization kernel
        return self.l1_normalize.l1_normalize_cuda(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for L1 normalization with warp-level reduction
l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void l1_normalize_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim
) {
    // Use 2D grid: x for batch, y for elements
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Shared memory for warp-level reduction results
    __shared__ float warp_sums[32];  // 32 warps max for 1024 threads
    
    // Each thread processes multiple elements with vectorized loads
    float thread_sum = 0.0f;
    
    // Process 4 elements at a time for better memory throughput
    const int stride = blockDim.x * 4;
    int idx = batch_idx * dim + tid * 4;
    
    for (int i = 0; i < dim; i += stride) {
        if (idx + 3 < (batch_idx + 1) * dim) {
            // Load 4 elements at once
            scalar_t val0 = input[idx];
            scalar_t val1 = input[idx + 1];
            scalar_t val2 = input[idx + 2];
            scalar_t val3 = input[idx + 3];
            
            thread_sum += fabsf((float)val0) + fabsf((float)val1) + 
                         fabsf((float)val2) + fabsf((float)val3);
            idx += stride;
        }
    }
    
    // Warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // First thread in each warp stores the warp sum
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction across warps (only first warp does this)
    if (warp_id == 0) {
        float warp_sum = lane_id < (blockDim.x / 32) ? warp_sums[lane_id] : 0.0f;
        
        // Reduce across warps
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        // Broadcast final sum to all threads
        if (lane_id == 0) {
            warp_sums[0] = warp_sum;
        }
    }
    
    // All threads get the final sum
    float batch_sum = warp_sums[0];
    batch_sum = batch_sum == 0.0f ? 1.0f : batch_sum;
    float inv_batch_sum = 1.0f / batch_sum;
    
    // Normalize and write output with vectorized stores
    idx = batch_idx * dim + tid * 4;
    for (int i = 0; i < dim; i += stride) {
        if (idx + 3 < (batch_idx + 1) * dim) {
            output[idx] = input[idx] * inv_batch_sum;
            output[idx + 1] = input[idx + 1] * inv_batch_sum;
            output[idx + 2] = input[idx + 2] * inv_batch_sum;
            output[idx + 3] = input[idx + 3] * inv_batch_sum;
            idx += stride;
        }
    }
    
    // Handle remaining elements
    for (int i = tid; i < dim; i += blockDim.x) {
        int pos = batch_idx * dim + i;
        output[pos] = input[pos] * inv_batch_sum;
    }
}

torch::Tensor l1_normalize_cuda(torch::Tensor input) {
    // Check input dimensions
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");
    
    // Make sure tensor is contiguous
    auto input_contig = input.contiguous();
    
    auto sizes = input_contig.sizes();
    auto batch_size = sizes[0];
    auto dim = sizes[1];
    
    // Create output tensor
    auto output = torch::empty_like(input_contig);
    
    // Optimized configuration - use 256 threads for good occupancy
    const int block_size = 256;
    const int grid_size = batch_size;
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(
        input_contig.scalar_type(),
        "l1_normalize_kernel_optimized",
        [&] {
            l1_normalize_kernel_optimized<scalar_t><<<grid_size, block_size>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                dim
            );
        }
    );
    
    return output;
}
"""

l1_norm_cpp_source = "torch::Tensor l1_normalize_cuda(torch::Tensor input);"

# Compile the inline CUDA code
l1_normalize = load_inline(
    name="l1_normalize",
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=["l1_normalize_cuda"],
    verbose=False,
    extra_cflags=["-O3", "-DNDEBUG"],
    extra_ldflags=["-lcudart"],
    with_cuda=True
)

class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using custom CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the optimized L1 normalization layer.
        """
        super(ModelNew, self).__init__()
        self.l1_normalize = l1_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        # Ensure tensor is on GPU
        if not x.is_cuda:
            x = x.cuda()
        
        # Apply fused L1 normalization kernel
        return self.l1_normalize.l1_normalize_cuda(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=sparse, 种子=66, 模式=value_stress

**杀死策略描述**: 90% zeros, 10% random

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]
```

---

## 35. `L1_P40__const_perturb__7`

- **算子**: `const_perturb` (Category A)
- **描述**: Perturb numeric literals (integer ±1; float ×1.01 or ×0.99) @ L196
- **变异行**: Line 196, 原始片段 `5`, 节点类型 `const:int-1`
- **杀死层**: `layer1_value`
- **杀死策略**: `structured_ramp`
- **杀死种子**: `72`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused LayerNorm
layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float convert_to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float convert_to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T convert_from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half convert_from_float<half>(float val) {
    return __float2half_rn(val);
}

template<typename T>
__global__ void layernorm_fused_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const float eps,
    const int num_elements,
    const int feature_size
) {
    extern __shared__ float shared_mem[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    const T* batch_input = input + batch_idx * feature_size;
    T* batch_output = output + batch_idx * feature_size;
    
    // Phase 1: Compute mean using parallel reduction
    float sum = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        sum += convert_to_float(batch_input[i]);
    }
    
    shared_mem[tid] = sum;
    __syncthreads();
    
    // Parallel reduction for mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_mem[0] / feature_size;
    __syncthreads();
    
    // Phase 2: Compute variance using parallel reduction
    float var_sum = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float diff = convert_to_float(batch_input[i]) - mean;
        var_sum += diff * diff;
    }
    
    shared_mem[tid] = var_sum;
    __syncthreads();
    
    // Parallel reduction for variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_mem[0] / feature_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Phase 3: Apply normalization with gamma and beta
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float normalized = (convert_to_float(batch_input[i]) - mean) * inv_std;
        float gamma_val = convert_to_float(gamma[i % num_elements]);
        float beta_val = convert_to_float(beta[i % num_elements]);
        normalized = normalized * gamma_val + beta_val;
        
        batch_output[i] = convert_from_float<T>(normalized);
    }
}

torch::Tensor layernorm_fused_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    auto sizes = input.sizes();
    int batch_size = 1;
    for (int i = 0; i < sizes.size() - gamma.sizes().size(); i++) {
        batch_size *= sizes[i];
    }
    
    int feature_size = 1;
    for (int i = sizes.size() - gamma.sizes().size(); i < sizes.size(); i++) {
        feature_size *= sizes[i];
    }
    
    int num_elements = gamma.numel();
    
    auto output = torch::empty_like(input);
    
    // Configure grid and block
    dim3 grid(batch_size);
    int block_size = 256;
    while (block_size > feature_size && block_size > 32) {
        block_size >>= 1;
    }
    
    // Ensure block_size is power of 2 for reduction
    block_size = 1 << static_cast<int>(log2f(static_cast<float>(block_size)));
    
    // Calculate shared memory size
    size_t shared_mem_size = block_size * sizeof(float);
    
    // Launch kernel based on data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "layernorm_fused_kernel",
        ([&] {
            layernorm_fused_kernel<scalar_t><<<grid, block_size, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                gamma.data_ptr<scalar_t>(),
                beta.data_ptr<scalar_t>(),
                eps,
                num_elements,
                feature_size
            );
        })
    );
    
    return output;
}
"""

layernorm_cpp_source = """
torch::Tensor layernorm_fused_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
);
"""

# Compile the inline CUDA code
layernorm_fused = load_inline(
    name="layernorm_fused",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_source,
    functions=["layernorm_fused_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using custom CUDA kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the optimized LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5
        
        # Custom CUDA operator
        self.layernorm_fused = layernorm_fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Layer Normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        # Flatten the normalized dimensions for kernel processing
        normalized_dims = len(self.normalized_shape)
        input_shape = x.shape
        
        # Reshape to [batch_size, feature_size] where feature_size = product(normalized_shape)
        batch_size = 1
        for i in range(len(input_shape) - normalized_dims):
            batch_size *= input_shape[i]
        
        feature_size = 1
        for i in range(normalized_dims):
            feature_size *= input_shape[-normalized_dims + i]
        
        x_reshaped = x.reshape(batch_size, feature_size)
        
        # Apply custom kernel
        output_reshaped = self.layernorm_fused.layernorm_fused_cuda(
            x_reshaped, 
            self.gamma.flatten(), 
            self.beta.flatten(), 
            self.eps
        )
        
        # Reshape back to original shape
        return output_reshaped.reshape(input_shape)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused LayerNorm
layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float convert_to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float convert_to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T convert_from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half convert_from_float<half>(float val) {
    return __float2half_rn(val);
}

template<typename T>
__global__ void layernorm_fused_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const float eps,
    const int num_elements,
    const int feature_size
) {
    extern __shared__ float shared_mem[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    const T* batch_input = input + batch_idx * feature_size;
    T* batch_output = output + batch_idx * feature_size;
    
    // Phase 1: Compute mean using parallel reduction
    float sum = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        sum += convert_to_float(batch_input[i]);
    }
    
    shared_mem[tid] = sum;
    __syncthreads();
    
    // Parallel reduction for mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_mem[0] / feature_size;
    __syncthreads();
    
    // Phase 2: Compute variance using parallel reduction
    float var_sum = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float diff = convert_to_float(batch_input[i]) - mean;
        var_sum += diff * diff;
    }
    
    shared_mem[tid] = var_sum;
    __syncthreads();
    
    // Parallel reduction for variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_mem[0] / feature_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Phase 3: Apply normalization with gamma and beta
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float normalized = (convert_to_float(batch_input[i]) - mean) * inv_std;
        float gamma_val = convert_to_float(gamma[i % num_elements]);
        float beta_val = convert_to_float(beta[i % num_elements]);
        normalized = normalized * gamma_val + beta_val;
        
        batch_output[i] = convert_from_float<T>(normalized);
    }
}

torch::Tensor layernorm_fused_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    auto sizes = input.sizes();
    int batch_size = 1;
    for (int i = 0; i < sizes.size() - gamma.sizes().size(); i++) {
        batch_size *= sizes[i];
    }
    
    int feature_size = 1;
    for (int i = sizes.size() - gamma.sizes().size(); i < sizes.size(); i++) {
        feature_size *= sizes[i];
    }
    
    int num_elements = gamma.numel();
    
    auto output = torch::empty_like(input);
    
    // Configure grid and block
    dim3 grid(batch_size);
    int block_size = 256;
    while (block_size > feature_size && block_size > 32) {
        block_size >>= 1;
    }
    
    // Ensure block_size is power of 2 for reduction
    block_size = 1 << static_cast<int>(log2f(static_cast<float>(block_size)));
    
    // Calculate shared memory size
    size_t shared_mem_size = block_size * sizeof(float);
    
    // Launch kernel based on data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "layernorm_fused_kernel",
        ([&] {
            layernorm_fused_kernel<scalar_t><<<grid, block_size, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                gamma.data_ptr<scalar_t>(),
                beta.data_ptr<scalar_t>(),
                eps,
                num_elements,
                feature_size
            );
        })
    );
    
    return output;
}
"""

layernorm_cpp_source = """
torch::Tensor layernorm_fused_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
);
"""

# Compile the inline CUDA code
layernorm_fused = load_inline(
    name="layernorm_fused",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_source,
    functions=["layernorm_fused_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using custom CUDA kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the optimized LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-4
        
        # Custom CUDA operator
        self.layernorm_fused = layernorm_fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Layer Normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        # Flatten the normalized dimensions for kernel processing
        normalized_dims = len(self.normalized_shape)
        input_shape = x.shape
        
        # Reshape to [batch_size, feature_size] where feature_size = product(normalized_shape)
        batch_size = 1
        for i in range(len(input_shape) - normalized_dims):
            batch_size *= input_shape[i]
        
        feature_size = 1
        for i in range(normalized_dims):
            feature_size *= input_shape[-normalized_dims + i]
        
        x_reshaped = x.reshape(batch_size, feature_size)
        
        # Apply custom kernel
        output_reshaped = self.layernorm_fused.layernorm_fused_cuda(
            x_reshaped, 
            self.gamma.flatten(), 
            self.beta.flatten(), 
            self.eps
        )
        
        # Reshape back to original shape
        return output_reshaped.reshape(input_shape)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=structured_ramp, 种子=72, 模式=value_stress

**杀死策略描述**: linearly increasing 0→N

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]
```

---

## 36. `L1_P40__epsilon_modify__0`

- **算子**: `epsilon_modify` (Category C)
- **描述**: Alter small epsilon literals (LayerNorm / safe_div / log stability) @ L196
- **变异行**: Line 196, 原始片段 `1e-5`, 节点类型 `eps:to_1e-2`
- **杀死层**: `layer1_value`
- **杀死策略**: `structured_ramp`
- **杀死种子**: `72`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused LayerNorm
layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float convert_to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float convert_to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T convert_from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half convert_from_float<half>(float val) {
    return __float2half_rn(val);
}

template<typename T>
__global__ void layernorm_fused_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const float eps,
    const int num_elements,
    const int feature_size
) {
    extern __shared__ float shared_mem[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    const T* batch_input = input + batch_idx * feature_size;
    T* batch_output = output + batch_idx * feature_size;
    
    // Phase 1: Compute mean using parallel reduction
    float sum = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        sum += convert_to_float(batch_input[i]);
    }
    
    shared_mem[tid] = sum;
    __syncthreads();
    
    // Parallel reduction for mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_mem[0] / feature_size;
    __syncthreads();
    
    // Phase 2: Compute variance using parallel reduction
    float var_sum = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float diff = convert_to_float(batch_input[i]) - mean;
        var_sum += diff * diff;
    }
    
    shared_mem[tid] = var_sum;
    __syncthreads();
    
    // Parallel reduction for variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_mem[0] / feature_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Phase 3: Apply normalization with gamma and beta
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float normalized = (convert_to_float(batch_input[i]) - mean) * inv_std;
        float gamma_val = convert_to_float(gamma[i % num_elements]);
        float beta_val = convert_to_float(beta[i % num_elements]);
        normalized = normalized * gamma_val + beta_val;
        
        batch_output[i] = convert_from_float<T>(normalized);
    }
}

torch::Tensor layernorm_fused_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    auto sizes = input.sizes();
    int batch_size = 1;
    for (int i = 0; i < sizes.size() - gamma.sizes().size(); i++) {
        batch_size *= sizes[i];
    }
    
    int feature_size = 1;
    for (int i = sizes.size() - gamma.sizes().size(); i < sizes.size(); i++) {
        feature_size *= sizes[i];
    }
    
    int num_elements = gamma.numel();
    
    auto output = torch::empty_like(input);
    
    // Configure grid and block
    dim3 grid(batch_size);
    int block_size = 256;
    while (block_size > feature_size && block_size > 32) {
        block_size >>= 1;
    }
    
    // Ensure block_size is power of 2 for reduction
    block_size = 1 << static_cast<int>(log2f(static_cast<float>(block_size)));
    
    // Calculate shared memory size
    size_t shared_mem_size = block_size * sizeof(float);
    
    // Launch kernel based on data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "layernorm_fused_kernel",
        ([&] {
            layernorm_fused_kernel<scalar_t><<<grid, block_size, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                gamma.data_ptr<scalar_t>(),
                beta.data_ptr<scalar_t>(),
                eps,
                num_elements,
                feature_size
            );
        })
    );
    
    return output;
}
"""

layernorm_cpp_source = """
torch::Tensor layernorm_fused_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
);
"""

# Compile the inline CUDA code
layernorm_fused = load_inline(
    name="layernorm_fused",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_source,
    functions=["layernorm_fused_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using custom CUDA kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the optimized LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5
        
        # Custom CUDA operator
        self.layernorm_fused = layernorm_fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Layer Normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        # Flatten the normalized dimensions for kernel processing
        normalized_dims = len(self.normalized_shape)
        input_shape = x.shape
        
        # Reshape to [batch_size, feature_size] where feature_size = product(normalized_shape)
        batch_size = 1
        for i in range(len(input_shape) - normalized_dims):
            batch_size *= input_shape[i]
        
        feature_size = 1
        for i in range(normalized_dims):
            feature_size *= input_shape[-normalized_dims + i]
        
        x_reshaped = x.reshape(batch_size, feature_size)
        
        # Apply custom kernel
        output_reshaped = self.layernorm_fused.layernorm_fused_cuda(
            x_reshaped, 
            self.gamma.flatten(), 
            self.beta.flatten(), 
            self.eps
        )
        
        # Reshape back to original shape
        return output_reshaped.reshape(input_shape)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused LayerNorm
layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float convert_to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float convert_to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T convert_from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half convert_from_float<half>(float val) {
    return __float2half_rn(val);
}

template<typename T>
__global__ void layernorm_fused_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const float eps,
    const int num_elements,
    const int feature_size
) {
    extern __shared__ float shared_mem[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    const T* batch_input = input + batch_idx * feature_size;
    T* batch_output = output + batch_idx * feature_size;
    
    // Phase 1: Compute mean using parallel reduction
    float sum = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        sum += convert_to_float(batch_input[i]);
    }
    
    shared_mem[tid] = sum;
    __syncthreads();
    
    // Parallel reduction for mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_mem[0] / feature_size;
    __syncthreads();
    
    // Phase 2: Compute variance using parallel reduction
    float var_sum = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float diff = convert_to_float(batch_input[i]) - mean;
        var_sum += diff * diff;
    }
    
    shared_mem[tid] = var_sum;
    __syncthreads();
    
    // Parallel reduction for variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_mem[0] / feature_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Phase 3: Apply normalization with gamma and beta
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float normalized = (convert_to_float(batch_input[i]) - mean) * inv_std;
        float gamma_val = convert_to_float(gamma[i % num_elements]);
        float beta_val = convert_to_float(beta[i % num_elements]);
        normalized = normalized * gamma_val + beta_val;
        
        batch_output[i] = convert_from_float<T>(normalized);
    }
}

torch::Tensor layernorm_fused_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    auto sizes = input.sizes();
    int batch_size = 1;
    for (int i = 0; i < sizes.size() - gamma.sizes().size(); i++) {
        batch_size *= sizes[i];
    }
    
    int feature_size = 1;
    for (int i = sizes.size() - gamma.sizes().size(); i < sizes.size(); i++) {
        feature_size *= sizes[i];
    }
    
    int num_elements = gamma.numel();
    
    auto output = torch::empty_like(input);
    
    // Configure grid and block
    dim3 grid(batch_size);
    int block_size = 256;
    while (block_size > feature_size && block_size > 32) {
        block_size >>= 1;
    }
    
    // Ensure block_size is power of 2 for reduction
    block_size = 1 << static_cast<int>(log2f(static_cast<float>(block_size)));
    
    // Calculate shared memory size
    size_t shared_mem_size = block_size * sizeof(float);
    
    // Launch kernel based on data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "layernorm_fused_kernel",
        ([&] {
            layernorm_fused_kernel<scalar_t><<<grid, block_size, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                gamma.data_ptr<scalar_t>(),
                beta.data_ptr<scalar_t>(),
                eps,
                num_elements,
                feature_size
            );
        })
    );
    
    return output;
}
"""

layernorm_cpp_source = """
torch::Tensor layernorm_fused_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
);
"""

# Compile the inline CUDA code
layernorm_fused = load_inline(
    name="layernorm_fused",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_source,
    functions=["layernorm_fused_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using custom CUDA kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the optimized LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-2
        
        # Custom CUDA operator
        self.layernorm_fused = layernorm_fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Layer Normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        # Flatten the normalized dimensions for kernel processing
        normalized_dims = len(self.normalized_shape)
        input_shape = x.shape
        
        # Reshape to [batch_size, feature_size] where feature_size = product(normalized_shape)
        batch_size = 1
        for i in range(len(input_shape) - normalized_dims):
            batch_size *= input_shape[i]
        
        feature_size = 1
        for i in range(normalized_dims):
            feature_size *= input_shape[-normalized_dims + i]
        
        x_reshaped = x.reshape(batch_size, feature_size)
        
        # Apply custom kernel
        output_reshaped = self.layernorm_fused.layernorm_fused_cuda(
            x_reshaped, 
            self.gamma.flatten(), 
            self.beta.flatten(), 
            self.eps
        )
        
        # Reshape back to original shape
        return output_reshaped.reshape(input_shape)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=structured_ramp, 种子=72, 模式=value_stress

**杀死策略描述**: linearly increasing 0→N

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]
```

---

## 37. `L1_P40__epsilon_modify__1`

- **算子**: `epsilon_modify` (Category C)
- **描述**: Alter small epsilon literals (LayerNorm / safe_div / log stability) @ L196
- **变异行**: Line 196, 原始片段 `1e-5`, 节点类型 `eps:to_zero`
- **杀死层**: `layer1_value`
- **杀死策略**: `near_zero`
- **杀死种子**: `42`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused LayerNorm
layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float convert_to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float convert_to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T convert_from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half convert_from_float<half>(float val) {
    return __float2half_rn(val);
}

template<typename T>
__global__ void layernorm_fused_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const float eps,
    const int num_elements,
    const int feature_size
) {
    extern __shared__ float shared_mem[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    const T* batch_input = input + batch_idx * feature_size;
    T* batch_output = output + batch_idx * feature_size;
    
    // Phase 1: Compute mean using parallel reduction
    float sum = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        sum += convert_to_float(batch_input[i]);
    }
    
    shared_mem[tid] = sum;
    __syncthreads();
    
    // Parallel reduction for mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_mem[0] / feature_size;
    __syncthreads();
    
    // Phase 2: Compute variance using parallel reduction
    float var_sum = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float diff = convert_to_float(batch_input[i]) - mean;
        var_sum += diff * diff;
    }
    
    shared_mem[tid] = var_sum;
    __syncthreads();
    
    // Parallel reduction for variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_mem[0] / feature_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Phase 3: Apply normalization with gamma and beta
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float normalized = (convert_to_float(batch_input[i]) - mean) * inv_std;
        float gamma_val = convert_to_float(gamma[i % num_elements]);
        float beta_val = convert_to_float(beta[i % num_elements]);
        normalized = normalized * gamma_val + beta_val;
        
        batch_output[i] = convert_from_float<T>(normalized);
    }
}

torch::Tensor layernorm_fused_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    auto sizes = input.sizes();
    int batch_size = 1;
    for (int i = 0; i < sizes.size() - gamma.sizes().size(); i++) {
        batch_size *= sizes[i];
    }
    
    int feature_size = 1;
    for (int i = sizes.size() - gamma.sizes().size(); i < sizes.size(); i++) {
        feature_size *= sizes[i];
    }
    
    int num_elements = gamma.numel();
    
    auto output = torch::empty_like(input);
    
    // Configure grid and block
    dim3 grid(batch_size);
    int block_size = 256;
    while (block_size > feature_size && block_size > 32) {
        block_size >>= 1;
    }
    
    // Ensure block_size is power of 2 for reduction
    block_size = 1 << static_cast<int>(log2f(static_cast<float>(block_size)));
    
    // Calculate shared memory size
    size_t shared_mem_size = block_size * sizeof(float);
    
    // Launch kernel based on data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "layernorm_fused_kernel",
        ([&] {
            layernorm_fused_kernel<scalar_t><<<grid, block_size, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                gamma.data_ptr<scalar_t>(),
                beta.data_ptr<scalar_t>(),
                eps,
                num_elements,
                feature_size
            );
        })
    );
    
    return output;
}
"""

layernorm_cpp_source = """
torch::Tensor layernorm_fused_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
);
"""

# Compile the inline CUDA code
layernorm_fused = load_inline(
    name="layernorm_fused",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_source,
    functions=["layernorm_fused_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using custom CUDA kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the optimized LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5
        
        # Custom CUDA operator
        self.layernorm_fused = layernorm_fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Layer Normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        # Flatten the normalized dimensions for kernel processing
        normalized_dims = len(self.normalized_shape)
        input_shape = x.shape
        
        # Reshape to [batch_size, feature_size] where feature_size = product(normalized_shape)
        batch_size = 1
        for i in range(len(input_shape) - normalized_dims):
            batch_size *= input_shape[i]
        
        feature_size = 1
        for i in range(normalized_dims):
            feature_size *= input_shape[-normalized_dims + i]
        
        x_reshaped = x.reshape(batch_size, feature_size)
        
        # Apply custom kernel
        output_reshaped = self.layernorm_fused.layernorm_fused_cuda(
            x_reshaped, 
            self.gamma.flatten(), 
            self.beta.flatten(), 
            self.eps
        )
        
        # Reshape back to original shape
        return output_reshaped.reshape(input_shape)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused LayerNorm
layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float convert_to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float convert_to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T convert_from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half convert_from_float<half>(float val) {
    return __float2half_rn(val);
}

template<typename T>
__global__ void layernorm_fused_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const float eps,
    const int num_elements,
    const int feature_size
) {
    extern __shared__ float shared_mem[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    const T* batch_input = input + batch_idx * feature_size;
    T* batch_output = output + batch_idx * feature_size;
    
    // Phase 1: Compute mean using parallel reduction
    float sum = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        sum += convert_to_float(batch_input[i]);
    }
    
    shared_mem[tid] = sum;
    __syncthreads();
    
    // Parallel reduction for mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_mem[0] / feature_size;
    __syncthreads();
    
    // Phase 2: Compute variance using parallel reduction
    float var_sum = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float diff = convert_to_float(batch_input[i]) - mean;
        var_sum += diff * diff;
    }
    
    shared_mem[tid] = var_sum;
    __syncthreads();
    
    // Parallel reduction for variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_mem[0] / feature_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Phase 3: Apply normalization with gamma and beta
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float normalized = (convert_to_float(batch_input[i]) - mean) * inv_std;
        float gamma_val = convert_to_float(gamma[i % num_elements]);
        float beta_val = convert_to_float(beta[i % num_elements]);
        normalized = normalized * gamma_val + beta_val;
        
        batch_output[i] = convert_from_float<T>(normalized);
    }
}

torch::Tensor layernorm_fused_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    auto sizes = input.sizes();
    int batch_size = 1;
    for (int i = 0; i < sizes.size() - gamma.sizes().size(); i++) {
        batch_size *= sizes[i];
    }
    
    int feature_size = 1;
    for (int i = sizes.size() - gamma.sizes().size(); i < sizes.size(); i++) {
        feature_size *= sizes[i];
    }
    
    int num_elements = gamma.numel();
    
    auto output = torch::empty_like(input);
    
    // Configure grid and block
    dim3 grid(batch_size);
    int block_size = 256;
    while (block_size > feature_size && block_size > 32) {
        block_size >>= 1;
    }
    
    // Ensure block_size is power of 2 for reduction
    block_size = 1 << static_cast<int>(log2f(static_cast<float>(block_size)));
    
    // Calculate shared memory size
    size_t shared_mem_size = block_size * sizeof(float);
    
    // Launch kernel based on data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "layernorm_fused_kernel",
        ([&] {
            layernorm_fused_kernel<scalar_t><<<grid, block_size, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                gamma.data_ptr<scalar_t>(),
                beta.data_ptr<scalar_t>(),
                eps,
                num_elements,
                feature_size
            );
        })
    );
    
    return output;
}
"""

layernorm_cpp_source = """
torch::Tensor layernorm_fused_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
);
"""

# Compile the inline CUDA code
layernorm_fused = load_inline(
    name="layernorm_fused",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_source,
    functions=["layernorm_fused_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using custom CUDA kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the optimized LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 0
        
        # Custom CUDA operator
        self.layernorm_fused = layernorm_fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Layer Normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        # Flatten the normalized dimensions for kernel processing
        normalized_dims = len(self.normalized_shape)
        input_shape = x.shape
        
        # Reshape to [batch_size, feature_size] where feature_size = product(normalized_shape)
        batch_size = 1
        for i in range(len(input_shape) - normalized_dims):
            batch_size *= input_shape[i]
        
        feature_size = 1
        for i in range(normalized_dims):
            feature_size *= input_shape[-normalized_dims + i]
        
        x_reshaped = x.reshape(batch_size, feature_size)
        
        # Apply custom kernel
        output_reshaped = self.layernorm_fused.layernorm_fused_cuda(
            x_reshaped, 
            self.gamma.flatten(), 
            self.beta.flatten(), 
            self.eps
        )
        
        # Reshape back to original shape
        return output_reshaped.reshape(input_shape)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=near_zero, 种子=42, 模式=value_stress

**杀死策略描述**: values ~1e-7

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]
```

---

## 38. `L1_P40__mask_boundary__0`

- **算子**: `mask_boundary` (Category B)
- **描述**: Weaken or tighten boundary checks in mask/tl.where (Triton) or thread/block guards (CUDA) @ L61
- **变异行**: Line 61, 原始片段 `tid < stride`, 节点类型 `rhs-1`
- **杀死层**: `layer1_value`
- **杀死策略**: `all_negative`
- **杀死种子**: `60`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused LayerNorm
layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float convert_to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float convert_to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T convert_from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half convert_from_float<half>(float val) {
    return __float2half_rn(val);
}

template<typename T>
__global__ void layernorm_fused_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const float eps,
    const int num_elements,
    const int feature_size
) {
    extern __shared__ float shared_mem[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    const T* batch_input = input + batch_idx * feature_size;
    T* batch_output = output + batch_idx * feature_size;
    
    // Phase 1: Compute mean using parallel reduction
    float sum = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        sum += convert_to_float(batch_input[i]);
    }
    
    shared_mem[tid] = sum;
    __syncthreads();
    
    // Parallel reduction for mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_mem[0] / feature_size;
    __syncthreads();
    
    // Phase 2: Compute variance using parallel reduction
    float var_sum = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float diff = convert_to_float(batch_input[i]) - mean;
        var_sum += diff * diff;
    }
    
    shared_mem[tid] = var_sum;
    __syncthreads();
    
    // Parallel reduction for variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_mem[0] / feature_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Phase 3: Apply normalization with gamma and beta
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float normalized = (convert_to_float(batch_input[i]) - mean) * inv_std;
        float gamma_val = convert_to_float(gamma[i % num_elements]);
        float beta_val = convert_to_float(beta[i % num_elements]);
        normalized = normalized * gamma_val + beta_val;
        
        batch_output[i] = convert_from_float<T>(normalized);
    }
}

torch::Tensor layernorm_fused_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    auto sizes = input.sizes();
    int batch_size = 1;
    for (int i = 0; i < sizes.size() - gamma.sizes().size(); i++) {
        batch_size *= sizes[i];
    }
    
    int feature_size = 1;
    for (int i = sizes.size() - gamma.sizes().size(); i < sizes.size(); i++) {
        feature_size *= sizes[i];
    }
    
    int num_elements = gamma.numel();
    
    auto output = torch::empty_like(input);
    
    // Configure grid and block
    dim3 grid(batch_size);
    int block_size = 256;
    while (block_size > feature_size && block_size > 32) {
        block_size >>= 1;
    }
    
    // Ensure block_size is power of 2 for reduction
    block_size = 1 << static_cast<int>(log2f(static_cast<float>(block_size)));
    
    // Calculate shared memory size
    size_t shared_mem_size = block_size * sizeof(float);
    
    // Launch kernel based on data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "layernorm_fused_kernel",
        ([&] {
            layernorm_fused_kernel<scalar_t><<<grid, block_size, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                gamma.data_ptr<scalar_t>(),
                beta.data_ptr<scalar_t>(),
                eps,
                num_elements,
                feature_size
            );
        })
    );
    
    return output;
}
"""

layernorm_cpp_source = """
torch::Tensor layernorm_fused_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
);
"""

# Compile the inline CUDA code
layernorm_fused = load_inline(
    name="layernorm_fused",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_source,
    functions=["layernorm_fused_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using custom CUDA kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the optimized LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5
        
        # Custom CUDA operator
        self.layernorm_fused = layernorm_fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Layer Normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        # Flatten the normalized dimensions for kernel processing
        normalized_dims = len(self.normalized_shape)
        input_shape = x.shape
        
        # Reshape to [batch_size, feature_size] where feature_size = product(normalized_shape)
        batch_size = 1
        for i in range(len(input_shape) - normalized_dims):
            batch_size *= input_shape[i]
        
        feature_size = 1
        for i in range(normalized_dims):
            feature_size *= input_shape[-normalized_dims + i]
        
        x_reshaped = x.reshape(batch_size, feature_size)
        
        # Apply custom kernel
        output_reshaped = self.layernorm_fused.layernorm_fused_cuda(
            x_reshaped, 
            self.gamma.flatten(), 
            self.beta.flatten(), 
            self.eps
        )
        
        # Reshape back to original shape
        return output_reshaped.reshape(input_shape)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused LayerNorm
layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float convert_to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float convert_to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T convert_from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half convert_from_float<half>(float val) {
    return __float2half_rn(val);
}

template<typename T>
__global__ void layernorm_fused_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const float eps,
    const int num_elements,
    const int feature_size
) {
    extern __shared__ float shared_mem[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    const T* batch_input = input + batch_idx * feature_size;
    T* batch_output = output + batch_idx * feature_size;
    
    // Phase 1: Compute mean using parallel reduction
    float sum = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        sum += convert_to_float(batch_input[i]);
    }
    
    shared_mem[tid] = sum;
    __syncthreads();
    
    // Parallel reduction for mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride - 1) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_mem[0] / feature_size;
    __syncthreads();
    
    // Phase 2: Compute variance using parallel reduction
    float var_sum = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float diff = convert_to_float(batch_input[i]) - mean;
        var_sum += diff * diff;
    }
    
    shared_mem[tid] = var_sum;
    __syncthreads();
    
    // Parallel reduction for variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_mem[0] / feature_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Phase 3: Apply normalization with gamma and beta
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float normalized = (convert_to_float(batch_input[i]) - mean) * inv_std;
        float gamma_val = convert_to_float(gamma[i % num_elements]);
        float beta_val = convert_to_float(beta[i % num_elements]);
        normalized = normalized * gamma_val + beta_val;
        
        batch_output[i] = convert_from_float<T>(normalized);
    }
}

torch::Tensor layernorm_fused_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    auto sizes = input.sizes();
    int batch_size = 1;
    for (int i = 0; i < sizes.size() - gamma.sizes().size(); i++) {
        batch_size *= sizes[i];
    }
    
    int feature_size = 1;
    for (int i = sizes.size() - gamma.sizes().size(); i < sizes.size(); i++) {
        feature_size *= sizes[i];
    }
    
    int num_elements = gamma.numel();
    
    auto output = torch::empty_like(input);
    
    // Configure grid and block
    dim3 grid(batch_size);
    int block_size = 256;
    while (block_size > feature_size && block_size > 32) {
        block_size >>= 1;
    }
    
    // Ensure block_size is power of 2 for reduction
    block_size = 1 << static_cast<int>(log2f(static_cast<float>(block_size)));
    
    // Calculate shared memory size
    size_t shared_mem_size = block_size * sizeof(float);
    
    // Launch kernel based on data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "layernorm_fused_kernel",
        ([&] {
            layernorm_fused_kernel<scalar_t><<<grid, block_size, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                gamma.data_ptr<scalar_t>(),
                beta.data_ptr<scalar_t>(),
                eps,
                num_elements,
                feature_size
            );
        })
    );
    
    return output;
}
"""

layernorm_cpp_source = """
torch::Tensor layernorm_fused_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
);
"""

# Compile the inline CUDA code
layernorm_fused = load_inline(
    name="layernorm_fused",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_source,
    functions=["layernorm_fused_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using custom CUDA kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the optimized LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5
        
        # Custom CUDA operator
        self.layernorm_fused = layernorm_fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Layer Normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        # Flatten the normalized dimensions for kernel processing
        normalized_dims = len(self.normalized_shape)
        input_shape = x.shape
        
        # Reshape to [batch_size, feature_size] where feature_size = product(normalized_shape)
        batch_size = 1
        for i in range(len(input_shape) - normalized_dims):
            batch_size *= input_shape[i]
        
        feature_size = 1
        for i in range(normalized_dims):
            feature_size *= input_shape[-normalized_dims + i]
        
        x_reshaped = x.reshape(batch_size, feature_size)
        
        # Apply custom kernel
        output_reshaped = self.layernorm_fused.layernorm_fused_cuda(
            x_reshaped, 
            self.gamma.flatten(), 
            self.beta.flatten(), 
            self.eps
        )
        
        # Reshape back to original shape
        return output_reshaped.reshape(input_shape)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=all_negative, 种子=60, 模式=value_stress

**杀死策略描述**: all values < 0

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]
```

---

## 39. `L1_P40__scale_modify__0`

- **算子**: `scale_modify` (Category C)
- **描述**: Distort attention / normalization scaling (inv-sqrt, rsqrt) @ L89
- **变异行**: Line 89, 原始片段 `rsqrtf(variance + eps)`, 节点类型 `scale:cuda_rsqrt_identity`
- **杀死层**: `layer1_value`
- **杀死策略**: `structured_ramp`
- **杀死种子**: `45`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused LayerNorm
layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float convert_to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float convert_to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T convert_from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half convert_from_float<half>(float val) {
    return __float2half_rn(val);
}

template<typename T>
__global__ void layernorm_fused_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const float eps,
    const int num_elements,
    const int feature_size
) {
    extern __shared__ float shared_mem[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    const T* batch_input = input + batch_idx * feature_size;
    T* batch_output = output + batch_idx * feature_size;
    
    // Phase 1: Compute mean using parallel reduction
    float sum = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        sum += convert_to_float(batch_input[i]);
    }
    
    shared_mem[tid] = sum;
    __syncthreads();
    
    // Parallel reduction for mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_mem[0] / feature_size;
    __syncthreads();
    
    // Phase 2: Compute variance using parallel reduction
    float var_sum = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float diff = convert_to_float(batch_input[i]) - mean;
        var_sum += diff * diff;
    }
    
    shared_mem[tid] = var_sum;
    __syncthreads();
    
    // Parallel reduction for variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_mem[0] / feature_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Phase 3: Apply normalization with gamma and beta
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float normalized = (convert_to_float(batch_input[i]) - mean) * inv_std;
        float gamma_val = convert_to_float(gamma[i % num_elements]);
        float beta_val = convert_to_float(beta[i % num_elements]);
        normalized = normalized * gamma_val + beta_val;
        
        batch_output[i] = convert_from_float<T>(normalized);
    }
}

torch::Tensor layernorm_fused_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    auto sizes = input.sizes();
    int batch_size = 1;
    for (int i = 0; i < sizes.size() - gamma.sizes().size(); i++) {
        batch_size *= sizes[i];
    }
    
    int feature_size = 1;
    for (int i = sizes.size() - gamma.sizes().size(); i < sizes.size(); i++) {
        feature_size *= sizes[i];
    }
    
    int num_elements = gamma.numel();
    
    auto output = torch::empty_like(input);
    
    // Configure grid and block
    dim3 grid(batch_size);
    int block_size = 256;
    while (block_size > feature_size && block_size > 32) {
        block_size >>= 1;
    }
    
    // Ensure block_size is power of 2 for reduction
    block_size = 1 << static_cast<int>(log2f(static_cast<float>(block_size)));
    
    // Calculate shared memory size
    size_t shared_mem_size = block_size * sizeof(float);
    
    // Launch kernel based on data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "layernorm_fused_kernel",
        ([&] {
            layernorm_fused_kernel<scalar_t><<<grid, block_size, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                gamma.data_ptr<scalar_t>(),
                beta.data_ptr<scalar_t>(),
                eps,
                num_elements,
                feature_size
            );
        })
    );
    
    return output;
}
"""

layernorm_cpp_source = """
torch::Tensor layernorm_fused_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
);
"""

# Compile the inline CUDA code
layernorm_fused = load_inline(
    name="layernorm_fused",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_source,
    functions=["layernorm_fused_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using custom CUDA kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the optimized LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5
        
        # Custom CUDA operator
        self.layernorm_fused = layernorm_fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Layer Normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        # Flatten the normalized dimensions for kernel processing
        normalized_dims = len(self.normalized_shape)
        input_shape = x.shape
        
        # Reshape to [batch_size, feature_size] where feature_size = product(normalized_shape)
        batch_size = 1
        for i in range(len(input_shape) - normalized_dims):
            batch_size *= input_shape[i]
        
        feature_size = 1
        for i in range(normalized_dims):
            feature_size *= input_shape[-normalized_dims + i]
        
        x_reshaped = x.reshape(batch_size, feature_size)
        
        # Apply custom kernel
        output_reshaped = self.layernorm_fused.layernorm_fused_cuda(
            x_reshaped, 
            self.gamma.flatten(), 
            self.beta.flatten(), 
            self.eps
        )
        
        # Reshape back to original shape
        return output_reshaped.reshape(input_shape)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused LayerNorm
layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ __forceinline__ float convert_to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float convert_to_float<half>(half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ T convert_from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half convert_from_float<half>(float val) {
    return __float2half_rn(val);
}

template<typename T>
__global__ void layernorm_fused_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const float eps,
    const int num_elements,
    const int feature_size
) {
    extern __shared__ float shared_mem[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    const T* batch_input = input + batch_idx * feature_size;
    T* batch_output = output + batch_idx * feature_size;
    
    // Phase 1: Compute mean using parallel reduction
    float sum = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        sum += convert_to_float(batch_input[i]);
    }
    
    shared_mem[tid] = sum;
    __syncthreads();
    
    // Parallel reduction for mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_mem[0] / feature_size;
    __syncthreads();
    
    // Phase 2: Compute variance using parallel reduction
    float var_sum = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float diff = convert_to_float(batch_input[i]) - mean;
        var_sum += diff * diff;
    }
    
    shared_mem[tid] = var_sum;
    __syncthreads();
    
    // Parallel reduction for variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_mem[0] / feature_size;
    float inv_std = variance + eps;
    
    // Phase 3: Apply normalization with gamma and beta
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float normalized = (convert_to_float(batch_input[i]) - mean) * inv_std;
        float gamma_val = convert_to_float(gamma[i % num_elements]);
        float beta_val = convert_to_float(beta[i % num_elements]);
        normalized = normalized * gamma_val + beta_val;
        
        batch_output[i] = convert_from_float<T>(normalized);
    }
}

torch::Tensor layernorm_fused_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    auto sizes = input.sizes();
    int batch_size = 1;
    for (int i = 0; i < sizes.size() - gamma.sizes().size(); i++) {
        batch_size *= sizes[i];
    }
    
    int feature_size = 1;
    for (int i = sizes.size() - gamma.sizes().size(); i < sizes.size(); i++) {
        feature_size *= sizes[i];
    }
    
    int num_elements = gamma.numel();
    
    auto output = torch::empty_like(input);
    
    // Configure grid and block
    dim3 grid(batch_size);
    int block_size = 256;
    while (block_size > feature_size && block_size > 32) {
        block_size >>= 1;
    }
    
    // Ensure block_size is power of 2 for reduction
    block_size = 1 << static_cast<int>(log2f(static_cast<float>(block_size)));
    
    // Calculate shared memory size
    size_t shared_mem_size = block_size * sizeof(float);
    
    // Launch kernel based on data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "layernorm_fused_kernel",
        ([&] {
            layernorm_fused_kernel<scalar_t><<<grid, block_size, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                gamma.data_ptr<scalar_t>(),
                beta.data_ptr<scalar_t>(),
                eps,
                num_elements,
                feature_size
            );
        })
    );
    
    return output;
}
"""

layernorm_cpp_source = """
torch::Tensor layernorm_fused_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
);
"""

# Compile the inline CUDA code
layernorm_fused = load_inline(
    name="layernorm_fused",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_source,
    functions=["layernorm_fused_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using custom CUDA kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the optimized LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5
        
        # Custom CUDA operator
        self.layernorm_fused = layernorm_fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Layer Normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        # Flatten the normalized dimensions for kernel processing
        normalized_dims = len(self.normalized_shape)
        input_shape = x.shape
        
        # Reshape to [batch_size, feature_size] where feature_size = product(normalized_shape)
        batch_size = 1
        for i in range(len(input_shape) - normalized_dims):
            batch_size *= input_shape[i]
        
        feature_size = 1
        for i in range(normalized_dims):
            feature_size *= input_shape[-normalized_dims + i]
        
        x_reshaped = x.reshape(batch_size, feature_size)
        
        # Apply custom kernel
        output_reshaped = self.layernorm_fused.layernorm_fused_cuda(
            x_reshaped, 
            self.gamma.flatten(), 
            self.beta.flatten(), 
            self.eps
        )
        
        # Reshape back to original shape
        return output_reshaped.reshape(input_shape)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=structured_ramp, 种子=45, 模式=value_stress

**杀死策略描述**: linearly increasing 0→N

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]
```

---

## 40. `L1_P50__index_replace__21`

- **算子**: `index_replace` (Category B)
- **描述**: Swap Triton program_id axis or CUDA thread/block dimension index (e.g. program_id(0)→(1), threadIdx.x→threadIdx.y) @ L69
- **变异行**: Line 69, 原始片段 `blockDim.x`, 节点类型 `cuda_dim|z`
- **杀死层**: `layer1_value`
- **杀死策略**: `structured_ramp`
- **杀死种子**: `42`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for product reduction
prod_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_product(scalar_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val *= __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<typename scalar_t>
__global__ void prod_reduction_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t reduction_size,
    const int64_t inner_size
) {
    extern __shared__ char shared_memory[];
    scalar_t* shared = reinterpret_cast<scalar_t*>(shared_memory);
    
    const int64_t outer_idx = blockIdx.x;
    const int64_t inner_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (inner_idx >= inner_size) return;
    
    const int64_t input_offset = outer_idx * reduction_size * inner_size + inner_idx;
    const int64_t output_offset = outer_idx * inner_size + inner_idx;
    
    scalar_t thread_product = scalar_t(1.0);
    
    // Optimized memory access pattern
    const int64_t elements_per_thread = 32;
    const int64_t thread_id = threadIdx.x;
    const int64_t block_dim = blockDim.x;
    
    // Process elements with better memory coalescing
    for (int64_t base = 0; base < reduction_size; base += elements_per_thread * block_dim) {
        int64_t idx = base + thread_id;
        
        #pragma unroll
        for (int64_t j = 0; j < elements_per_thread; ++j) {
            if (idx < reduction_size) {
                thread_product *= input[input_offset + idx * inner_size];
            }
            idx += block_dim;
        }
    }
    
    // Warp-level reduction
    scalar_t warp_product = warp_reduce_product(thread_product);
    
    // Write warp result to shared memory
    if (threadIdx.x % 32 == 0) {
        shared[threadIdx.x / 32] = warp_product;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (threadIdx.x < 32) {
        scalar_t val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[threadIdx.x] : scalar_t(1.0);
        scalar_t block_product = warp_reduce_product(val);
        
        if (threadIdx.x == 0) {
            output[output_offset] = block_product;
        }
    }
}

template<typename scalar_t>
__global__ void prod_reduction_kernel_small(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t reduction_size,
    const int64_t inner_size
) {
    const int64_t outer_idx = blockIdx.x;
    const int64_t inner_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (inner_idx >= inner_size) return;
    
    const int64_t input_offset = outer_idx * reduction_size * inner_size + inner_idx;
    const int64_t output_offset = outer_idx * inner_size + inner_idx;
    
    scalar_t product = scalar_t(1.0);
    
    // Simple reduction for small sizes
    for (int64_t i = 0; i < reduction_size; ++i) {
        product *= input[input_offset + i * inner_size];
    }
    
    if (threadIdx.x == 0) {
        output[output_offset] = product;
    }
}

torch::Tensor prod_reduction_cuda(torch::Tensor input, int64_t dim) {
    auto sizes = input.sizes();
    
    // Handle negative dimension
    if (dim < 0) {
        dim = input.dim() + dim;
    }
    
    // Calculate output shape
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < input.dim(); ++i) {
        if (i != dim) {
            output_shape.push_back(sizes[i]);
        }
    }
    auto output = torch::empty(output_shape, input.options());
    
    // Calculate dimensions for kernel launch
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= sizes[i];
    }
    
    int64_t reduction_size = sizes[dim];
    
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < input.dim(); ++i) {
        inner_size *= sizes[i];
    }
    
    // Choose kernel based on reduction size
    const int64_t block_size = 256;
    const int64_t grid_x = outer_size;
    const int64_t grid_y = (inner_size + block_size - 1) / block_size;
    
    dim3 grid(grid_x, grid_y);
    dim3 block(block_size);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "prod_reduction_cuda", [&] {
        if (reduction_size <= 32) {
            // Use simple kernel for small reductions
            prod_reduction_kernel_small<scalar_t><<<grid, block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                outer_size,
                reduction_size,
                inner_size
            );
        } else {
            // Use optimized kernel with warp reduction for larger reductions
            size_t shared_mem_size = ((block_size + 31) / 32) * sizeof(scalar_t);
            prod_reduction_kernel_optimized<scalar_t><<<grid, block, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                outer_size,
                reduction_size,
                inner_size
            );
        }
    });
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}
"""

prod_reduction_cpp_source = """
torch::Tensor prod_reduction_cuda(torch::Tensor input, int64_t dim);
"""

# Compile the inline CUDA code
prod_reduction_module = load_inline(
    name="prod_reduction",
    cpp_sources=prod_reduction_cpp_source,
    cuda_sources=prod_reduction_source,
    functions=["prod_reduction_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs product reduction over a dimension using custom CUDA kernels.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.prod_reduction = prod_reduction_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs product reduction over the specified dimension using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with product reduction applied.
        """
        if x.is_cuda:
            return self.prod_reduction.prod_reduction_cuda(x, self.dim)
        else:
            # Fallback to PyTorch implementation for CPU tensors
            return torch.prod(x, dim=self.dim)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for product reduction
prod_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_product(scalar_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val *= __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<typename scalar_t>
__global__ void prod_reduction_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t reduction_size,
    const int64_t inner_size
) {
    extern __shared__ char shared_memory[];
    scalar_t* shared = reinterpret_cast<scalar_t*>(shared_memory);
    
    const int64_t outer_idx = blockIdx.x;
    const int64_t inner_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (inner_idx >= inner_size) return;
    
    const int64_t input_offset = outer_idx * reduction_size * inner_size + inner_idx;
    const int64_t output_offset = outer_idx * inner_size + inner_idx;
    
    scalar_t thread_product = scalar_t(1.0);
    
    // Optimized memory access pattern
    const int64_t elements_per_thread = 32;
    const int64_t thread_id = threadIdx.x;
    const int64_t block_dim = blockDim.x;
    
    // Process elements with better memory coalescing
    for (int64_t base = 0; base < reduction_size; base += elements_per_thread * block_dim) {
        int64_t idx = base + thread_id;
        
        #pragma unroll
        for (int64_t j = 0; j < elements_per_thread; ++j) {
            if (idx < reduction_size) {
                thread_product *= input[input_offset + idx * inner_size];
            }
            idx += block_dim;
        }
    }
    
    // Warp-level reduction
    scalar_t warp_product = warp_reduce_product(thread_product);
    
    // Write warp result to shared memory
    if (threadIdx.x % 32 == 0) {
        shared[threadIdx.x / 32] = warp_product;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (threadIdx.x < 32) {
        scalar_t val = (threadIdx.x < (blockDim.z + 31) / 32) ? shared[threadIdx.x] : scalar_t(1.0);
        scalar_t block_product = warp_reduce_product(val);
        
        if (threadIdx.x == 0) {
            output[output_offset] = block_product;
        }
    }
}

template<typename scalar_t>
__global__ void prod_reduction_kernel_small(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t reduction_size,
    const int64_t inner_size
) {
    const int64_t outer_idx = blockIdx.x;
    const int64_t inner_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (inner_idx >= inner_size) return;
    
    const int64_t input_offset = outer_idx * reduction_size * inner_size + inner_idx;
    const int64_t output_offset = outer_idx * inner_size + inner_idx;
    
    scalar_t product = scalar_t(1.0);
    
    // Simple reduction for small sizes
    for (int64_t i = 0; i < reduction_size; ++i) {
        product *= input[input_offset + i * inner_size];
    }
    
    if (threadIdx.x == 0) {
        output[output_offset] = product;
    }
}

torch::Tensor prod_reduction_cuda(torch::Tensor input, int64_t dim) {
    auto sizes = input.sizes();
    
    // Handle negative dimension
    if (dim < 0) {
        dim = input.dim() + dim;
    }
    
    // Calculate output shape
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < input.dim(); ++i) {
        if (i != dim) {
            output_shape.push_back(sizes[i]);
        }
    }
    auto output = torch::empty(output_shape, input.options());
    
    // Calculate dimensions for kernel launch
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= sizes[i];
    }
    
    int64_t reduction_size = sizes[dim];
    
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < input.dim(); ++i) {
        inner_size *= sizes[i];
    }
    
    // Choose kernel based on reduction size
    const int64_t block_size = 256;
    const int64_t grid_x = outer_size;
    const int64_t grid_y = (inner_size + block_size - 1) / block_size;
    
    dim3 grid(grid_x, grid_y);
    dim3 block(block_size);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "prod_reduction_cuda", [&] {
        if (reduction_size <= 32) {
            // Use simple kernel for small reductions
            prod_reduction_kernel_small<scalar_t><<<grid, block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                outer_size,
                reduction_size,
                inner_size
            );
        } else {
            // Use optimized kernel with warp reduction for larger reductions
            size_t shared_mem_size = ((block_size + 31) / 32) * sizeof(scalar_t);
            prod_reduction_kernel_optimized<scalar_t><<<grid, block, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                outer_size,
                reduction_size,
                inner_size
            );
        }
    });
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}
"""

prod_reduction_cpp_source = """
torch::Tensor prod_reduction_cuda(torch::Tensor input, int64_t dim);
"""

# Compile the inline CUDA code
prod_reduction_module = load_inline(
    name="prod_reduction",
    cpp_sources=prod_reduction_cpp_source,
    cuda_sources=prod_reduction_source,
    functions=["prod_reduction_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs product reduction over a dimension using custom CUDA kernels.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.prod_reduction = prod_reduction_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs product reduction over the specified dimension using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with product reduction applied.
        """
        if x.is_cuda:
            return self.prod_reduction.prod_reduction_cuda(x, self.dim)
        else:
            # Fallback to PyTorch implementation for CPU tensors
            return torch.prod(x, dim=self.dim)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=structured_ramp, 种子=42, 模式=value_stress

**杀死策略描述**: linearly increasing 0→N

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 16
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]
```

---

## 41. `L1_P94__relop_replace__7`

- **算子**: `relop_replace` (Category A)
- **描述**: Replace relational operators (<→<=, <=→<, >→>=, >=→>, ==→!=, !=→==) @ L68
- **变异行**: Line 68, 原始片段 `<`, 节点类型 `cuda_Lt`
- **杀死层**: `layer3_repeated`
- **杀死策略**: `None`
- **杀死种子**: `242`
- **杀死模式**: `repeated_run`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused MSE computation
mse_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void fused_mse_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    int batch_size,
    int element_count
) {
    // Shared memory for block-level reduction
    __shared__ float shared_sum[BLOCK_SIZE];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const float* pred_batch = predictions + batch_idx * element_count;
    const float* target_batch = targets + batch_idx * element_count;
    
    // Thread-local sum
    float thread_sum = 0.0f;
    
    // Process multiple elements per thread for better memory coalescing
    for (int i = tid; i < element_count; i += BLOCK_SIZE) {
        float diff = pred_batch[i] - target_batch[i];
        thread_sum += diff * diff;
    }
    
    // Block-level reduction using shared memory
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    // First thread writes the batch result
    if (tid == 0) {
        output[batch_idx] = shared_sum[0] / element_count;
    }
}

__global__ void final_mean_kernel(
    const float* __restrict__ batch_means,
    float* __restrict__ final_output,
    int batch_size
) {
    // Single block reduction for final mean
    __shared__ float shared_sum[256];
    
    int tid = threadIdx.x;
    shared_sum[tid] = 0.0f;
    
    // Accumulate batch means
    for (int i = tid; i < batch_size; i += 256) {
        shared_sum[tid] += batch_means[i];
    }
    __syncthreads();
    
    // Parallel reduction
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    // Write scalar output
    if (tid == 0) {
        *final_output = shared_sum[0] / batch_size;
    }
}

torch::Tensor fused_mse_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto batch_size = predictions.size(0);
    auto element_count = predictions.numel() / batch_size;
    
    // Allocate intermediate tensor for batch means
    auto batch_means = torch::empty({batch_size}, predictions.options());
    auto final_output = torch::zeros({}, predictions.options());  // Scalar tensor
    
    // Configure kernel launch parameters
    const int BLOCK_SIZE = 256;
    dim3 grid_dims(batch_size);
    dim3 block_dims(BLOCK_SIZE);
    
    // Launch fused MSE kernel
    fused_mse_kernel<BLOCK_SIZE><<<grid_dims, block_dims>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        batch_means.data_ptr<float>(),
        batch_size,
        element_count
    );
    
    // Launch final reduction kernel (single block)
    final_mean_kernel<<<1, 256>>>(
        batch_means.data_ptr<float>(),
        final_output.data_ptr<float>(),
        batch_size
    );
    
    return final_output;
}
"""

mse_cpp_source = "torch::Tensor fused_mse_cuda(torch::Tensor predictions, torch::Tensor targets);"

# Compile the inline CUDA code
fused_mse = load_inline(
    name="fused_mse",
    cpp_sources=mse_cpp_source,
    cuda_sources=mse_kernel_source,
    functions=["fused_mse_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for Mean Squared Error computation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.fused_mse = fused_mse

    def forward(self, predictions, targets):
        # Use fused CUDA kernel for MSE computation
        return self.fused_mse.fused_mse_cuda(predictions, targets)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused MSE computation
mse_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void fused_mse_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    int batch_size,
    int element_count
) {
    // Shared memory for block-level reduction
    __shared__ float shared_sum[BLOCK_SIZE];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const float* pred_batch = predictions + batch_idx * element_count;
    const float* target_batch = targets + batch_idx * element_count;
    
    // Thread-local sum
    float thread_sum = 0.0f;
    
    // Process multiple elements per thread for better memory coalescing
    for (int i = tid; i < element_count; i += BLOCK_SIZE) {
        float diff = pred_batch[i] - target_batch[i];
        thread_sum += diff * diff;
    }
    
    // Block-level reduction using shared memory
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    // First thread writes the batch result
    if (tid == 0) {
        output[batch_idx] = shared_sum[0] / element_count;
    }
}

__global__ void final_mean_kernel(
    const float* __restrict__ batch_means,
    float* __restrict__ final_output,
    int batch_size
) {
    // Single block reduction for final mean
    __shared__ float shared_sum[256];
    
    int tid = threadIdx.x;
    shared_sum[tid] = 0.0f;
    
    // Accumulate batch means
    for (int i = tid; i <= batch_size; i += 256) {
        shared_sum[tid] += batch_means[i];
    }
    __syncthreads();
    
    // Parallel reduction
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    // Write scalar output
    if (tid == 0) {
        *final_output = shared_sum[0] / batch_size;
    }
}

torch::Tensor fused_mse_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto batch_size = predictions.size(0);
    auto element_count = predictions.numel() / batch_size;
    
    // Allocate intermediate tensor for batch means
    auto batch_means = torch::empty({batch_size}, predictions.options());
    auto final_output = torch::zeros({}, predictions.options());  // Scalar tensor
    
    // Configure kernel launch parameters
    const int BLOCK_SIZE = 256;
    dim3 grid_dims(batch_size);
    dim3 block_dims(BLOCK_SIZE);
    
    // Launch fused MSE kernel
    fused_mse_kernel<BLOCK_SIZE><<<grid_dims, block_dims>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        batch_means.data_ptr<float>(),
        batch_size,
        element_count
    );
    
    // Launch final reduction kernel (single block)
    final_mean_kernel<<<1, 256>>>(
        batch_means.data_ptr<float>(),
        final_output.data_ptr<float>(),
        batch_size
    );
    
    return final_output;
}
"""

mse_cpp_source = "torch::Tensor fused_mse_cuda(torch::Tensor predictions, torch::Tensor targets);"

# Compile the inline CUDA code
fused_mse = load_inline(
    name="fused_mse",
    cpp_sources=mse_cpp_source,
    cuda_sources=mse_kernel_source,
    functions=["fused_mse_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for Mean Squared Error computation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.fused_mse = fused_mse

    def forward(self, predictions, targets):
        # Use fused CUDA kernel for MSE computation
        return self.fused_mse.fused_mse_cuda(predictions, targets)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer3_repeated, 策略=None, 种子=242, 模式=repeated_run

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 128
input_shape = (4096, )

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]
```

---

## 42. `L1_P96__sync_remove__0`

- **算子**: `sync_remove` (Category B)
- **描述**: Remove GPU synchronization barriers (tl.debug_barrier, __syncthreads) @ L48
- **变异行**: Line 48, 原始片段 `__syncthreads();`, 节点类型 `cuda_syncthreads`
- **杀死层**: `layer1_value`
- **杀死策略**: `uniform_constant`
- **杀死种子**: `69`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Smooth L1 Loss
smooth_l1_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void smooth_l1_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    int total_elements,
    float beta
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        float diff = predictions[idx] - targets[idx];
        float abs_diff = fabsf(diff);
        
        if (abs_diff < beta) {
            output[idx] = 0.5f * diff * diff / beta;
        } else {
            output[idx] = abs_diff - 0.5f * beta;
        }
    }
}

__global__ void sum_reduction_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (idx < size) {
        shared_mem[tid] = input[idx];
    } else {
        shared_mem[tid] = 0.0f;
    }
    __syncthreads();
    
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    // Write block sum to global memory
    if (tid == 0) {
        output[blockIdx.x] = shared_mem[0];
    }
}

torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets, float beta=1.0) {
    // Check inputs
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.is_cuda() && targets.is_cuda(), "Inputs must be CUDA tensors");
    
    auto total_elements = predictions.numel();
    
    // First kernel: compute element-wise losses
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    auto element_losses = torch::empty({total_elements}, 
                                       torch::TensorOptions()
                                       .dtype(torch::kFloat32)
                                       .device(torch::kCUDA));
    
    smooth_l1_loss_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        element_losses.data_ptr<float>(),
        total_elements,
        beta
    );
    
    // Second kernel: reduction to get sum
    const int reduction_block_size = 256;
    const int reduction_blocks = (total_elements + reduction_block_size - 1) / reduction_block_size;
    
    auto block_sums = torch::zeros({reduction_blocks}, 
                                   torch::TensorOptions()
                                   .dtype(torch::kFloat32)
                                   .device(torch::kCUDA));
    
    sum_reduction_kernel<<<reduction_blocks, reduction_block_size, reduction_block_size * sizeof(float)>>>(
        element_losses.data_ptr<float>(),
        block_sums.data_ptr<float>(),
        total_elements
    );
    
    // Final reduction on CPU (simpler and fine for small number of blocks)
    auto block_sums_cpu = block_sums.cpu();
    float total_loss = 0.0f;
    for (int i = 0; i < reduction_blocks; i++) {
        total_loss += block_sums_cpu.data_ptr<float>()[i];
    }
    
    // Compute mean loss
    float mean_loss = total_loss / total_elements;
    return torch::tensor(mean_loss, torch::TensorOptions().device(torch::kCUDA));
}
"""

smooth_l1_cpp_source = """
torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets, float beta=1.0);
"""

# Compile the inline CUDA code
smooth_l1_cuda = load_inline(
    name="smooth_l1_loss",
    cpp_sources=smooth_l1_cpp_source,
    cuda_sources=smooth_l1_source,
    functions=["smooth_l1_loss_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that computes Smooth L1 (Huber) Loss using custom CUDA kernels.
    
    Parameters:
        beta (float): Specifies the threshold at which to change between L1 and L2 loss.
                      Default: 1.0
    """
    def __init__(self, beta=1.0):
        super(ModelNew, self).__init__()
        self.beta = beta
        self.smooth_l1_loss_cuda = smooth_l1_cuda
    
    def forward(self, predictions, targets):
        return self.smooth_l1_loss_cuda.smooth_l1_loss_cuda(predictions, targets, self.beta)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Smooth L1 Loss
smooth_l1_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void smooth_l1_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    int total_elements,
    float beta
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        float diff = predictions[idx] - targets[idx];
        float abs_diff = fabsf(diff);
        
        if (abs_diff < beta) {
            output[idx] = 0.5f * diff * diff / beta;
        } else {
            output[idx] = abs_diff - 0.5f * beta;
        }
    }
}

__global__ void sum_reduction_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (idx < size) {
        shared_mem[tid] = input[idx];
    } else {
        shared_mem[tid] = 0.0f;
    }
    
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    // Write block sum to global memory
    if (tid == 0) {
        output[blockIdx.x] = shared_mem[0];
    }
}

torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets, float beta=1.0) {
    // Check inputs
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.is_cuda() && targets.is_cuda(), "Inputs must be CUDA tensors");
    
    auto total_elements = predictions.numel();
    
    // First kernel: compute element-wise losses
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    auto element_losses = torch::empty({total_elements}, 
                                       torch::TensorOptions()
                                       .dtype(torch::kFloat32)
                                       .device(torch::kCUDA));
    
    smooth_l1_loss_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        element_losses.data_ptr<float>(),
        total_elements,
        beta
    );
    
    // Second kernel: reduction to get sum
    const int reduction_block_size = 256;
    const int reduction_blocks = (total_elements + reduction_block_size - 1) / reduction_block_size;
    
    auto block_sums = torch::zeros({reduction_blocks}, 
                                   torch::TensorOptions()
                                   .dtype(torch::kFloat32)
                                   .device(torch::kCUDA));
    
    sum_reduction_kernel<<<reduction_blocks, reduction_block_size, reduction_block_size * sizeof(float)>>>(
        element_losses.data_ptr<float>(),
        block_sums.data_ptr<float>(),
        total_elements
    );
    
    // Final reduction on CPU (simpler and fine for small number of blocks)
    auto block_sums_cpu = block_sums.cpu();
    float total_loss = 0.0f;
    for (int i = 0; i < reduction_blocks; i++) {
        total_loss += block_sums_cpu.data_ptr<float>()[i];
    }
    
    // Compute mean loss
    float mean_loss = total_loss / total_elements;
    return torch::tensor(mean_loss, torch::TensorOptions().device(torch::kCUDA));
}
"""

smooth_l1_cpp_source = """
torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets, float beta=1.0);
"""

# Compile the inline CUDA code
smooth_l1_cuda = load_inline(
    name="smooth_l1_loss",
    cpp_sources=smooth_l1_cpp_source,
    cuda_sources=smooth_l1_source,
    functions=["smooth_l1_loss_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that computes Smooth L1 (Huber) Loss using custom CUDA kernels.
    
    Parameters:
        beta (float): Specifies the threshold at which to change between L1 and L2 loss.
                      Default: 1.0
    """
    def __init__(self, beta=1.0):
        super(ModelNew, self).__init__()
        self.beta = beta
        self.smooth_l1_loss_cuda = smooth_l1_cuda
    
    def forward(self, predictions, targets):
        return self.smooth_l1_loss_cuda.smooth_l1_loss_cuda(predictions, targets, self.beta)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=uniform_constant, 种子=69, 模式=value_stress

**杀死策略描述**: all elements = same constant

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 128
input_shape = (4096, )

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]
```

---

## 43. `L1_P97__sync_remove__0`

- **算子**: `sync_remove` (Category B)
- **描述**: Remove GPU synchronization barriers (tl.debug_barrier, __syncthreads) @ L57
- **变异行**: Line 57, 原始片段 `__syncthreads();`, 节点类型 `cuda_syncthreads`
- **杀死层**: `layer3_repeated`
- **杀死策略**: `None`
- **杀死种子**: `242`
- **杀死模式**: `repeated_run`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fully fused cosine similarity loss with vectorized memory access
cosine_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template<int THREADS_PER_BLOCK, int ELEMENTS_PER_THREAD>
__global__ void cosine_loss_fused_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    int batch_size,
    int feature_size
) {
    __shared__ float shared_dot[THREADS_PER_BLOCK];
    __shared__ float shared_pred_norm[THREADS_PER_BLOCK];
    __shared__ float shared_target_norm[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int batch_idx = blockIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const float* pred_ptr = predictions + batch_idx * feature_size;
    const float* target_ptr = targets + batch_idx * feature_size;
    
    // Initialize accumulators
    float dot_product = 0.0f;
    float norm_pred = 0.0f;
    float norm_target = 0.0f;
    
    // Vectorized processing for better memory bandwidth utilization
    for (int base = 0; base < feature_size; base += THREADS_PER_BLOCK * ELEMENTS_PER_THREAD) {
        int idx = base + tid * ELEMENTS_PER_THREAD;
        
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            int element_idx = idx + i;
            if (element_idx < feature_size) {
                float pred_val = pred_ptr[element_idx];
                float target_val = target_ptr[element_idx];
                dot_product += pred_val * target_val;
                norm_pred += pred_val * pred_val;
                norm_target += target_val * target_val;
            }
        }
    }
    
    // Store thread results in shared memory
    shared_dot[tid] = dot_product;
    shared_pred_norm[tid] = norm_pred;
    shared_target_norm[tid] = norm_target;
    __syncthreads();
    
    // Parallel reduction using warp shuffles for better efficiency
    for (int stride = THREADS_PER_BLOCK / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            shared_dot[tid] += shared_dot[tid + stride];
            shared_pred_norm[tid] += shared_pred_norm[tid + stride];
            shared_target_norm[tid] += shared_target_norm[tid + stride];
        }
        __syncthreads();
    }
    
    // Warp-level reduction
    if (tid < 32) {
        shared_dot[tid] += __shfl_down_sync(0xffffffff, shared_dot[tid], 16);
        shared_pred_norm[tid] += __shfl_down_sync(0xffffffff, shared_pred_norm[tid], 16);
        shared_target_norm[tid] += __shfl_down_sync(0xffffffff, shared_target_norm[tid], 16);
        
        shared_dot[tid] += __shfl_down_sync(0xffffffff, shared_dot[tid], 8);
        shared_pred_norm[tid] += __shfl_down_sync(0xffffffff, shared_pred_norm[tid], 8);
        shared_target_norm[tid] += __shfl_down_sync(0xffffffff, shared_target_norm[tid], 8);
        
        shared_dot[tid] += __shfl_down_sync(0xffffffff, shared_dot[tid], 4);
        shared_pred_norm[tid] += __shfl_down_sync(0xffffffff, shared_pred_norm[tid], 4);
        shared_target_norm[tid] += __shfl_down_sync(0xffffffff, shared_target_norm[tid], 4);
        
        shared_dot[tid] += __shfl_down_sync(0xffffffff, shared_dot[tid], 2);
        shared_pred_norm[tid] += __shfl_down_sync(0xffffffff, shared_pred_norm[tid], 2);
        shared_target_norm[tid] += __shfl_down_sync(0xffffffff, shared_target_norm[tid], 2);
        
        shared_dot[tid] += __shfl_down_sync(0xffffffff, shared_dot[tid], 1);
        shared_pred_norm[tid] += __shfl_down_sync(0xffffffff, shared_pred_norm[tid], 1);
        shared_target_norm[tid] += __shfl_down_sync(0xffffffff, shared_target_norm[tid], 1);
    }
    __syncthreads();
    
    // First thread computes final loss for this batch
    if (tid == 0) {
        const float epsilon = 1e-8f;
        float norm_product = sqrtf(shared_pred_norm[0] + epsilon) * sqrtf(shared_target_norm[0] + epsilon);
        float cosine_sim = shared_dot[0] / (norm_product + epsilon);
        float loss = 1.0f - cosine_sim;
        
        // Atomic add to accumulate mean
        atomicAdd(output, loss / batch_size);
    }
}

torch::Tensor cosine_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto batch_size = predictions.size(0);
    auto feature_size = predictions.size(1);
    
    // Allocate output tensor (single value)
    auto options = torch::TensorOptions()
        .dtype(predictions.dtype())
        .device(predictions.device());
    auto output = torch::zeros({1}, options);
    
    // Choose optimal configuration based on feature size
    const int THREADS = 256;
    const int ELEMENTS_PER_THREAD = 4;  // Process 4 elements per thread for better memory coalescing
    
    dim3 grid_size(batch_size, 1, 1);
    dim3 block_size(THREADS, 1, 1);
    
    // Launch fused kernel - one block per batch element
    cosine_loss_fused_kernel<THREADS, ELEMENTS_PER_THREAD><<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        feature_size
    );
    
    return output.squeeze();
}
"""

cosine_loss_cpp_source = "torch::Tensor cosine_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"

# Compile the inline CUDA code - fixed compiler flags
cosine_loss = load_inline(
    name="cosine_loss",
    cpp_sources=cosine_loss_cpp_source,
    cuda_sources=cosine_loss_source,
    functions=["cosine_loss_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fully fused custom CUDA kernel for cosine similarity loss.
    Features vectorized memory access and warp-level reductions.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cosine_loss = cosine_loss

    def forward(self, predictions, targets):
        # Use custom CUDA kernel for fully fused cosine similarity loss computation
        return self.cosine_loss.cosine_loss_cuda(predictions, targets)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fully fused cosine similarity loss with vectorized memory access
cosine_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template<int THREADS_PER_BLOCK, int ELEMENTS_PER_THREAD>
__global__ void cosine_loss_fused_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    int batch_size,
    int feature_size
) {
    __shared__ float shared_dot[THREADS_PER_BLOCK];
    __shared__ float shared_pred_norm[THREADS_PER_BLOCK];
    __shared__ float shared_target_norm[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int batch_idx = blockIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const float* pred_ptr = predictions + batch_idx * feature_size;
    const float* target_ptr = targets + batch_idx * feature_size;
    
    // Initialize accumulators
    float dot_product = 0.0f;
    float norm_pred = 0.0f;
    float norm_target = 0.0f;
    
    // Vectorized processing for better memory bandwidth utilization
    for (int base = 0; base < feature_size; base += THREADS_PER_BLOCK * ELEMENTS_PER_THREAD) {
        int idx = base + tid * ELEMENTS_PER_THREAD;
        
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            int element_idx = idx + i;
            if (element_idx < feature_size) {
                float pred_val = pred_ptr[element_idx];
                float target_val = target_ptr[element_idx];
                dot_product += pred_val * target_val;
                norm_pred += pred_val * pred_val;
                norm_target += target_val * target_val;
            }
        }
    }
    
    // Store thread results in shared memory
    shared_dot[tid] = dot_product;
    shared_pred_norm[tid] = norm_pred;
    shared_target_norm[tid] = norm_target;
    
    // Parallel reduction using warp shuffles for better efficiency
    for (int stride = THREADS_PER_BLOCK / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            shared_dot[tid] += shared_dot[tid + stride];
            shared_pred_norm[tid] += shared_pred_norm[tid + stride];
            shared_target_norm[tid] += shared_target_norm[tid + stride];
        }
        __syncthreads();
    }
    
    // Warp-level reduction
    if (tid < 32) {
        shared_dot[tid] += __shfl_down_sync(0xffffffff, shared_dot[tid], 16);
        shared_pred_norm[tid] += __shfl_down_sync(0xffffffff, shared_pred_norm[tid], 16);
        shared_target_norm[tid] += __shfl_down_sync(0xffffffff, shared_target_norm[tid], 16);
        
        shared_dot[tid] += __shfl_down_sync(0xffffffff, shared_dot[tid], 8);
        shared_pred_norm[tid] += __shfl_down_sync(0xffffffff, shared_pred_norm[tid], 8);
        shared_target_norm[tid] += __shfl_down_sync(0xffffffff, shared_target_norm[tid], 8);
        
        shared_dot[tid] += __shfl_down_sync(0xffffffff, shared_dot[tid], 4);
        shared_pred_norm[tid] += __shfl_down_sync(0xffffffff, shared_pred_norm[tid], 4);
        shared_target_norm[tid] += __shfl_down_sync(0xffffffff, shared_target_norm[tid], 4);
        
        shared_dot[tid] += __shfl_down_sync(0xffffffff, shared_dot[tid], 2);
        shared_pred_norm[tid] += __shfl_down_sync(0xffffffff, shared_pred_norm[tid], 2);
        shared_target_norm[tid] += __shfl_down_sync(0xffffffff, shared_target_norm[tid], 2);
        
        shared_dot[tid] += __shfl_down_sync(0xffffffff, shared_dot[tid], 1);
        shared_pred_norm[tid] += __shfl_down_sync(0xffffffff, shared_pred_norm[tid], 1);
        shared_target_norm[tid] += __shfl_down_sync(0xffffffff, shared_target_norm[tid], 1);
    }
    __syncthreads();
    
    // First thread computes final loss for this batch
    if (tid == 0) {
        const float epsilon = 1e-8f;
        float norm_product = sqrtf(shared_pred_norm[0] + epsilon) * sqrtf(shared_target_norm[0] + epsilon);
        float cosine_sim = shared_dot[0] / (norm_product + epsilon);
        float loss = 1.0f - cosine_sim;
        
        // Atomic add to accumulate mean
        atomicAdd(output, loss / batch_size);
    }
}

torch::Tensor cosine_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto batch_size = predictions.size(0);
    auto feature_size = predictions.size(1);
    
    // Allocate output tensor (single value)
    auto options = torch::TensorOptions()
        .dtype(predictions.dtype())
        .device(predictions.device());
    auto output = torch::zeros({1}, options);
    
    // Choose optimal configuration based on feature size
    const int THREADS = 256;
    const int ELEMENTS_PER_THREAD = 4;  // Process 4 elements per thread for better memory coalescing
    
    dim3 grid_size(batch_size, 1, 1);
    dim3 block_size(THREADS, 1, 1);
    
    // Launch fused kernel - one block per batch element
    cosine_loss_fused_kernel<THREADS, ELEMENTS_PER_THREAD><<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        feature_size
    );
    
    return output.squeeze();
}
"""

cosine_loss_cpp_source = "torch::Tensor cosine_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"

# Compile the inline CUDA code - fixed compiler flags
cosine_loss = load_inline(
    name="cosine_loss",
    cpp_sources=cosine_loss_cpp_source,
    cuda_sources=cosine_loss_source,
    functions=["cosine_loss_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fully fused custom CUDA kernel for cosine similarity loss.
    Features vectorized memory access and warp-level reductions.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cosine_loss = cosine_loss

    def forward(self, predictions, targets):
        # Use custom CUDA kernel for fully fused cosine similarity loss computation
        return self.cosine_loss.cosine_loss_cuda(predictions, targets)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer3_repeated, 策略=None, 种子=242, 模式=repeated_run

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 128
input_shape = (4096, )

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]
```

---

## 44. `L1_P97__sync_remove__1`

- **算子**: `sync_remove` (Category B)
- **描述**: Remove GPU synchronization barriers (tl.debug_barrier, __syncthreads) @ L66
- **变异行**: Line 66, 原始片段 `__syncthreads();`, 节点类型 `cuda_syncthreads`
- **杀死层**: `layer1_value`
- **杀死策略**: `all_negative`
- **杀死种子**: `57`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fully fused cosine similarity loss with vectorized memory access
cosine_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template<int THREADS_PER_BLOCK, int ELEMENTS_PER_THREAD>
__global__ void cosine_loss_fused_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    int batch_size,
    int feature_size
) {
    __shared__ float shared_dot[THREADS_PER_BLOCK];
    __shared__ float shared_pred_norm[THREADS_PER_BLOCK];
    __shared__ float shared_target_norm[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int batch_idx = blockIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const float* pred_ptr = predictions + batch_idx * feature_size;
    const float* target_ptr = targets + batch_idx * feature_size;
    
    // Initialize accumulators
    float dot_product = 0.0f;
    float norm_pred = 0.0f;
    float norm_target = 0.0f;
    
    // Vectorized processing for better memory bandwidth utilization
    for (int base = 0; base < feature_size; base += THREADS_PER_BLOCK * ELEMENTS_PER_THREAD) {
        int idx = base + tid * ELEMENTS_PER_THREAD;
        
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            int element_idx = idx + i;
            if (element_idx < feature_size) {
                float pred_val = pred_ptr[element_idx];
                float target_val = target_ptr[element_idx];
                dot_product += pred_val * target_val;
                norm_pred += pred_val * pred_val;
                norm_target += target_val * target_val;
            }
        }
    }
    
    // Store thread results in shared memory
    shared_dot[tid] = dot_product;
    shared_pred_norm[tid] = norm_pred;
    shared_target_norm[tid] = norm_target;
    __syncthreads();
    
    // Parallel reduction using warp shuffles for better efficiency
    for (int stride = THREADS_PER_BLOCK / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            shared_dot[tid] += shared_dot[tid + stride];
            shared_pred_norm[tid] += shared_pred_norm[tid + stride];
            shared_target_norm[tid] += shared_target_norm[tid + stride];
        }
        __syncthreads();
    }
    
    // Warp-level reduction
    if (tid < 32) {
        shared_dot[tid] += __shfl_down_sync(0xffffffff, shared_dot[tid], 16);
        shared_pred_norm[tid] += __shfl_down_sync(0xffffffff, shared_pred_norm[tid], 16);
        shared_target_norm[tid] += __shfl_down_sync(0xffffffff, shared_target_norm[tid], 16);
        
        shared_dot[tid] += __shfl_down_sync(0xffffffff, shared_dot[tid], 8);
        shared_pred_norm[tid] += __shfl_down_sync(0xffffffff, shared_pred_norm[tid], 8);
        shared_target_norm[tid] += __shfl_down_sync(0xffffffff, shared_target_norm[tid], 8);
        
        shared_dot[tid] += __shfl_down_sync(0xffffffff, shared_dot[tid], 4);
        shared_pred_norm[tid] += __shfl_down_sync(0xffffffff, shared_pred_norm[tid], 4);
        shared_target_norm[tid] += __shfl_down_sync(0xffffffff, shared_target_norm[tid], 4);
        
        shared_dot[tid] += __shfl_down_sync(0xffffffff, shared_dot[tid], 2);
        shared_pred_norm[tid] += __shfl_down_sync(0xffffffff, shared_pred_norm[tid], 2);
        shared_target_norm[tid] += __shfl_down_sync(0xffffffff, shared_target_norm[tid], 2);
        
        shared_dot[tid] += __shfl_down_sync(0xffffffff, shared_dot[tid], 1);
        shared_pred_norm[tid] += __shfl_down_sync(0xffffffff, shared_pred_norm[tid], 1);
        shared_target_norm[tid] += __shfl_down_sync(0xffffffff, shared_target_norm[tid], 1);
    }
    __syncthreads();
    
    // First thread computes final loss for this batch
    if (tid == 0) {
        const float epsilon = 1e-8f;
        float norm_product = sqrtf(shared_pred_norm[0] + epsilon) * sqrtf(shared_target_norm[0] + epsilon);
        float cosine_sim = shared_dot[0] / (norm_product + epsilon);
        float loss = 1.0f - cosine_sim;
        
        // Atomic add to accumulate mean
        atomicAdd(output, loss / batch_size);
    }
}

torch::Tensor cosine_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto batch_size = predictions.size(0);
    auto feature_size = predictions.size(1);
    
    // Allocate output tensor (single value)
    auto options = torch::TensorOptions()
        .dtype(predictions.dtype())
        .device(predictions.device());
    auto output = torch::zeros({1}, options);
    
    // Choose optimal configuration based on feature size
    const int THREADS = 256;
    const int ELEMENTS_PER_THREAD = 4;  // Process 4 elements per thread for better memory coalescing
    
    dim3 grid_size(batch_size, 1, 1);
    dim3 block_size(THREADS, 1, 1);
    
    // Launch fused kernel - one block per batch element
    cosine_loss_fused_kernel<THREADS, ELEMENTS_PER_THREAD><<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        feature_size
    );
    
    return output.squeeze();
}
"""

cosine_loss_cpp_source = "torch::Tensor cosine_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"

# Compile the inline CUDA code - fixed compiler flags
cosine_loss = load_inline(
    name="cosine_loss",
    cpp_sources=cosine_loss_cpp_source,
    cuda_sources=cosine_loss_source,
    functions=["cosine_loss_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fully fused custom CUDA kernel for cosine similarity loss.
    Features vectorized memory access and warp-level reductions.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cosine_loss = cosine_loss

    def forward(self, predictions, targets):
        # Use custom CUDA kernel for fully fused cosine similarity loss computation
        return self.cosine_loss.cosine_loss_cuda(predictions, targets)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fully fused cosine similarity loss with vectorized memory access
cosine_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template<int THREADS_PER_BLOCK, int ELEMENTS_PER_THREAD>
__global__ void cosine_loss_fused_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    int batch_size,
    int feature_size
) {
    __shared__ float shared_dot[THREADS_PER_BLOCK];
    __shared__ float shared_pred_norm[THREADS_PER_BLOCK];
    __shared__ float shared_target_norm[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int batch_idx = blockIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const float* pred_ptr = predictions + batch_idx * feature_size;
    const float* target_ptr = targets + batch_idx * feature_size;
    
    // Initialize accumulators
    float dot_product = 0.0f;
    float norm_pred = 0.0f;
    float norm_target = 0.0f;
    
    // Vectorized processing for better memory bandwidth utilization
    for (int base = 0; base < feature_size; base += THREADS_PER_BLOCK * ELEMENTS_PER_THREAD) {
        int idx = base + tid * ELEMENTS_PER_THREAD;
        
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            int element_idx = idx + i;
            if (element_idx < feature_size) {
                float pred_val = pred_ptr[element_idx];
                float target_val = target_ptr[element_idx];
                dot_product += pred_val * target_val;
                norm_pred += pred_val * pred_val;
                norm_target += target_val * target_val;
            }
        }
    }
    
    // Store thread results in shared memory
    shared_dot[tid] = dot_product;
    shared_pred_norm[tid] = norm_pred;
    shared_target_norm[tid] = norm_target;
    __syncthreads();
    
    // Parallel reduction using warp shuffles for better efficiency
    for (int stride = THREADS_PER_BLOCK / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            shared_dot[tid] += shared_dot[tid + stride];
            shared_pred_norm[tid] += shared_pred_norm[tid + stride];
            shared_target_norm[tid] += shared_target_norm[tid + stride];
        }
    }
    
    // Warp-level reduction
    if (tid < 32) {
        shared_dot[tid] += __shfl_down_sync(0xffffffff, shared_dot[tid], 16);
        shared_pred_norm[tid] += __shfl_down_sync(0xffffffff, shared_pred_norm[tid], 16);
        shared_target_norm[tid] += __shfl_down_sync(0xffffffff, shared_target_norm[tid], 16);
        
        shared_dot[tid] += __shfl_down_sync(0xffffffff, shared_dot[tid], 8);
        shared_pred_norm[tid] += __shfl_down_sync(0xffffffff, shared_pred_norm[tid], 8);
        shared_target_norm[tid] += __shfl_down_sync(0xffffffff, shared_target_norm[tid], 8);
        
        shared_dot[tid] += __shfl_down_sync(0xffffffff, shared_dot[tid], 4);
        shared_pred_norm[tid] += __shfl_down_sync(0xffffffff, shared_pred_norm[tid], 4);
        shared_target_norm[tid] += __shfl_down_sync(0xffffffff, shared_target_norm[tid], 4);
        
        shared_dot[tid] += __shfl_down_sync(0xffffffff, shared_dot[tid], 2);
        shared_pred_norm[tid] += __shfl_down_sync(0xffffffff, shared_pred_norm[tid], 2);
        shared_target_norm[tid] += __shfl_down_sync(0xffffffff, shared_target_norm[tid], 2);
        
        shared_dot[tid] += __shfl_down_sync(0xffffffff, shared_dot[tid], 1);
        shared_pred_norm[tid] += __shfl_down_sync(0xffffffff, shared_pred_norm[tid], 1);
        shared_target_norm[tid] += __shfl_down_sync(0xffffffff, shared_target_norm[tid], 1);
    }
    __syncthreads();
    
    // First thread computes final loss for this batch
    if (tid == 0) {
        const float epsilon = 1e-8f;
        float norm_product = sqrtf(shared_pred_norm[0] + epsilon) * sqrtf(shared_target_norm[0] + epsilon);
        float cosine_sim = shared_dot[0] / (norm_product + epsilon);
        float loss = 1.0f - cosine_sim;
        
        // Atomic add to accumulate mean
        atomicAdd(output, loss / batch_size);
    }
}

torch::Tensor cosine_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto batch_size = predictions.size(0);
    auto feature_size = predictions.size(1);
    
    // Allocate output tensor (single value)
    auto options = torch::TensorOptions()
        .dtype(predictions.dtype())
        .device(predictions.device());
    auto output = torch::zeros({1}, options);
    
    // Choose optimal configuration based on feature size
    const int THREADS = 256;
    const int ELEMENTS_PER_THREAD = 4;  // Process 4 elements per thread for better memory coalescing
    
    dim3 grid_size(batch_size, 1, 1);
    dim3 block_size(THREADS, 1, 1);
    
    // Launch fused kernel - one block per batch element
    cosine_loss_fused_kernel<THREADS, ELEMENTS_PER_THREAD><<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        feature_size
    );
    
    return output.squeeze();
}
"""

cosine_loss_cpp_source = "torch::Tensor cosine_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"

# Compile the inline CUDA code - fixed compiler flags
cosine_loss = load_inline(
    name="cosine_loss",
    cpp_sources=cosine_loss_cpp_source,
    cuda_sources=cosine_loss_source,
    functions=["cosine_loss_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fully fused custom CUDA kernel for cosine similarity loss.
    Features vectorized memory access and warp-level reductions.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cosine_loss = cosine_loss

    def forward(self, predictions, targets):
        # Use custom CUDA kernel for fully fused cosine similarity loss computation
        return self.cosine_loss.cosine_loss_cuda(predictions, targets)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=all_negative, 种子=57, 模式=value_stress

**杀死策略描述**: all values < 0

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 128
input_shape = (4096, )

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]
```

---

## 45. `L1_P98__arith_replace__0`

- **算子**: `arith_replace` (Category A)
- **描述**: Replace binary arithmetic operators (+→-, -→+, *→/, /→*) @ L26
- **变异行**: Line 26, 原始片段 `+`, 节点类型 `cuda_Add`
- **杀死层**: `layer1_value`
- **杀死策略**: `uniform_constant`
- **杀死种子**: `69`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for fused KL divergence
kl_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

template<typename scalar_t>
__global__ void fused_kl_div_kernel(
    const scalar_t* __restrict__ predictions,
    const scalar_t* __restrict__ targets,
    scalar_t* __restrict__ output,
    int batch_size,
    int seq_len
) {
    // Each block processes one batch element
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const scalar_t* pred_batch = predictions + batch_idx * seq_len;
    const scalar_t* target_batch = targets + batch_idx * seq_len;
    
    // Thread block and warp setup
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    float thread_sum = 0.0f;
    
    // Process elements with stride equal to total threads in block
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    
    for (int idx = tid; idx < seq_len; idx += total_threads) {
        float p = static_cast<float>(pred_batch[idx]);
        float t = static_cast<float>(target_batch[idx]);
        
        // KL divergence: t * log(t / p) = t * (log(t) - log(p))
        // But we compute directly to match PyTorch's kl_div
        if (t > 0.0f && p > 0.0f) {
            thread_sum += t * (__logf(t) - __logf(p));
        }
        // Note: PyTorch's kl_div expects log(predictions) as input,
        // but we're computing the full KL divergence here
    }
    
    // Warp reduction
    float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>());
    
    // First thread in warp writes to shared memory
    __shared__ float block_sums[32];
    if (warp.thread_rank() == 0) {
        block_sums[warp.meta_group_rank()] = warp_sum;
    }
    block.sync();
    
    // First warp reduces all warp sums
    if (warp.meta_group_rank() == 0) {
        float block_sum = 0.0f;
        int num_warps = (total_threads + 31) / 32;
        if (warp.thread_rank() < num_warps) {
            block_sum = block_sums[warp.thread_rank()];
        }
        
        float final_sum = cg::reduce(warp, block_sum, cg::plus<float>());
        
        if (warp.thread_rank() == 0) {
            // Store per-batch result
            output[batch_idx] = static_cast<scalar_t>(final_sum);
        }
    }
}

torch::Tensor fused_kl_div_cuda(torch::Tensor predictions, torch::Tensor targets) {
    // Ensure inputs are on GPU and contiguous
    predictions = predictions.contiguous();
    targets = targets.contiguous();
    
    auto batch_size = predictions.size(0);
    auto seq_len = predictions.size(1);
    
    // Create output tensor for per-batch results
    auto output = torch::zeros({batch_size}, predictions.options());
    
    // Optimized kernel launch configuration
    int threads = 256;
    int blocks = batch_size;
    
    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "fused_kl_div_kernel", ([&] {
        fused_kl_div_kernel<scalar_t><<<blocks, threads>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            seq_len
        );
    }));
    
    // Return batchmean reduction
    return output.mean();
}
"""

kl_div_cpp_source = """
torch::Tensor fused_kl_div_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

# Compile the inline CUDA code
fused_kl_div = load_inline(
    name="fused_kl_div",
    cpp_sources=kl_div_cpp_source,
    cuda_sources=kl_div_source,
    functions=["fused_kl_div_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

class ModelNew(nn.Module):
    """
    Optimized model that computes Kullback-Leibler Divergence using fused CUDA kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.fused_kl_div = fused_kl_div
        
    def forward(self, predictions, targets):
        # Input validation and preprocessing
        predictions = predictions.contiguous()
        targets = targets.contiguous()
        
        # Ensure numerical stability (matching PyTorch's behavior)
        predictions = torch.clamp(predictions, min=1e-12)
        targets = torch.clamp(targets, min=1e-12)
        
        # Normalize to probability distributions
        predictions = predictions / predictions.sum(dim=-1, keepdim=True)
        targets = targets / targets.sum(dim=-1, keepdim=True)
        
        # Use fused CUDA kernel
        return self.fused_kl_div.fused_kl_div_cuda(predictions, targets)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for fused KL divergence
kl_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

template<typename scalar_t>
__global__ void fused_kl_div_kernel(
    const scalar_t* __restrict__ predictions,
    const scalar_t* __restrict__ targets,
    scalar_t* __restrict__ output,
    int batch_size,
    int seq_len
) {
    // Each block processes one batch element
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const scalar_t* pred_batch = predictions - batch_idx * seq_len;
    const scalar_t* target_batch = targets + batch_idx * seq_len;
    
    // Thread block and warp setup
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    float thread_sum = 0.0f;
    
    // Process elements with stride equal to total threads in block
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    
    for (int idx = tid; idx < seq_len; idx += total_threads) {
        float p = static_cast<float>(pred_batch[idx]);
        float t = static_cast<float>(target_batch[idx]);
        
        // KL divergence: t * log(t / p) = t * (log(t) - log(p))
        // But we compute directly to match PyTorch's kl_div
        if (t > 0.0f && p > 0.0f) {
            thread_sum += t * (__logf(t) - __logf(p));
        }
        // Note: PyTorch's kl_div expects log(predictions) as input,
        // but we're computing the full KL divergence here
    }
    
    // Warp reduction
    float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>());
    
    // First thread in warp writes to shared memory
    __shared__ float block_sums[32];
    if (warp.thread_rank() == 0) {
        block_sums[warp.meta_group_rank()] = warp_sum;
    }
    block.sync();
    
    // First warp reduces all warp sums
    if (warp.meta_group_rank() == 0) {
        float block_sum = 0.0f;
        int num_warps = (total_threads + 31) / 32;
        if (warp.thread_rank() < num_warps) {
            block_sum = block_sums[warp.thread_rank()];
        }
        
        float final_sum = cg::reduce(warp, block_sum, cg::plus<float>());
        
        if (warp.thread_rank() == 0) {
            // Store per-batch result
            output[batch_idx] = static_cast<scalar_t>(final_sum);
        }
    }
}

torch::Tensor fused_kl_div_cuda(torch::Tensor predictions, torch::Tensor targets) {
    // Ensure inputs are on GPU and contiguous
    predictions = predictions.contiguous();
    targets = targets.contiguous();
    
    auto batch_size = predictions.size(0);
    auto seq_len = predictions.size(1);
    
    // Create output tensor for per-batch results
    auto output = torch::zeros({batch_size}, predictions.options());
    
    // Optimized kernel launch configuration
    int threads = 256;
    int blocks = batch_size;
    
    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "fused_kl_div_kernel", ([&] {
        fused_kl_div_kernel<scalar_t><<<blocks, threads>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            seq_len
        );
    }));
    
    // Return batchmean reduction
    return output.mean();
}
"""

kl_div_cpp_source = """
torch::Tensor fused_kl_div_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

# Compile the inline CUDA code
fused_kl_div = load_inline(
    name="fused_kl_div",
    cpp_sources=kl_div_cpp_source,
    cuda_sources=kl_div_source,
    functions=["fused_kl_div_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

class ModelNew(nn.Module):
    """
    Optimized model that computes Kullback-Leibler Divergence using fused CUDA kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.fused_kl_div = fused_kl_div
        
    def forward(self, predictions, targets):
        # Input validation and preprocessing
        predictions = predictions.contiguous()
        targets = targets.contiguous()
        
        # Ensure numerical stability (matching PyTorch's behavior)
        predictions = torch.clamp(predictions, min=1e-12)
        targets = torch.clamp(targets, min=1e-12)
        
        # Normalize to probability distributions
        predictions = predictions / predictions.sum(dim=-1, keepdim=True)
        targets = targets / targets.sum(dim=-1, keepdim=True)
        
        # Use fused CUDA kernel
        return self.fused_kl_div.fused_kl_div_cuda(predictions, targets)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=uniform_constant, 种子=69, 模式=value_stress

**杀死策略描述**: all elements = same constant

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape).softmax(dim=-1), torch.randn(batch_size, *input_shape).softmax(dim=-1)]
```

---

## 46. `L1_P98__arith_replace__12`

- **算子**: `arith_replace` (Category A)
- **描述**: Replace binary arithmetic operators (+→-, -→+, *→/, /→*) @ L143
- **变异行**: Line 143, 原始片段 `/`, 节点类型 `Div`
- **杀死层**: `layer1_value`
- **杀死策略**: `uniform_constant`
- **杀死种子**: `69`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for fused KL divergence
kl_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

template<typename scalar_t>
__global__ void fused_kl_div_kernel(
    const scalar_t* __restrict__ predictions,
    const scalar_t* __restrict__ targets,
    scalar_t* __restrict__ output,
    int batch_size,
    int seq_len
) {
    // Each block processes one batch element
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const scalar_t* pred_batch = predictions + batch_idx * seq_len;
    const scalar_t* target_batch = targets + batch_idx * seq_len;
    
    // Thread block and warp setup
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    float thread_sum = 0.0f;
    
    // Process elements with stride equal to total threads in block
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    
    for (int idx = tid; idx < seq_len; idx += total_threads) {
        float p = static_cast<float>(pred_batch[idx]);
        float t = static_cast<float>(target_batch[idx]);
        
        // KL divergence: t * log(t / p) = t * (log(t) - log(p))
        // But we compute directly to match PyTorch's kl_div
        if (t > 0.0f && p > 0.0f) {
            thread_sum += t * (__logf(t) - __logf(p));
        }
        // Note: PyTorch's kl_div expects log(predictions) as input,
        // but we're computing the full KL divergence here
    }
    
    // Warp reduction
    float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>());
    
    // First thread in warp writes to shared memory
    __shared__ float block_sums[32];
    if (warp.thread_rank() == 0) {
        block_sums[warp.meta_group_rank()] = warp_sum;
    }
    block.sync();
    
    // First warp reduces all warp sums
    if (warp.meta_group_rank() == 0) {
        float block_sum = 0.0f;
        int num_warps = (total_threads + 31) / 32;
        if (warp.thread_rank() < num_warps) {
            block_sum = block_sums[warp.thread_rank()];
        }
        
        float final_sum = cg::reduce(warp, block_sum, cg::plus<float>());
        
        if (warp.thread_rank() == 0) {
            // Store per-batch result
            output[batch_idx] = static_cast<scalar_t>(final_sum);
        }
    }
}

torch::Tensor fused_kl_div_cuda(torch::Tensor predictions, torch::Tensor targets) {
    // Ensure inputs are on GPU and contiguous
    predictions = predictions.contiguous();
    targets = targets.contiguous();
    
    auto batch_size = predictions.size(0);
    auto seq_len = predictions.size(1);
    
    // Create output tensor for per-batch results
    auto output = torch::zeros({batch_size}, predictions.options());
    
    // Optimized kernel launch configuration
    int threads = 256;
    int blocks = batch_size;
    
    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "fused_kl_div_kernel", ([&] {
        fused_kl_div_kernel<scalar_t><<<blocks, threads>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            seq_len
        );
    }));
    
    // Return batchmean reduction
    return output.mean();
}
"""

kl_div_cpp_source = """
torch::Tensor fused_kl_div_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

# Compile the inline CUDA code
fused_kl_div = load_inline(
    name="fused_kl_div",
    cpp_sources=kl_div_cpp_source,
    cuda_sources=kl_div_source,
    functions=["fused_kl_div_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

class ModelNew(nn.Module):
    """
    Optimized model that computes Kullback-Leibler Divergence using fused CUDA kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.fused_kl_div = fused_kl_div
        
    def forward(self, predictions, targets):
        # Input validation and preprocessing
        predictions = predictions.contiguous()
        targets = targets.contiguous()
        
        # Ensure numerical stability (matching PyTorch's behavior)
        predictions = torch.clamp(predictions, min=1e-12)
        targets = torch.clamp(targets, min=1e-12)
        
        # Normalize to probability distributions
        predictions = predictions / predictions.sum(dim=-1, keepdim=True)
        targets = targets / targets.sum(dim=-1, keepdim=True)
        
        # Use fused CUDA kernel
        return self.fused_kl_div.fused_kl_div_cuda(predictions, targets)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for fused KL divergence
kl_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

template<typename scalar_t>
__global__ void fused_kl_div_kernel(
    const scalar_t* __restrict__ predictions,
    const scalar_t* __restrict__ targets,
    scalar_t* __restrict__ output,
    int batch_size,
    int seq_len
) {
    // Each block processes one batch element
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const scalar_t* pred_batch = predictions + batch_idx * seq_len;
    const scalar_t* target_batch = targets + batch_idx * seq_len;
    
    // Thread block and warp setup
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    float thread_sum = 0.0f;
    
    // Process elements with stride equal to total threads in block
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    
    for (int idx = tid; idx < seq_len; idx += total_threads) {
        float p = static_cast<float>(pred_batch[idx]);
        float t = static_cast<float>(target_batch[idx]);
        
        // KL divergence: t * log(t / p) = t * (log(t) - log(p))
        // But we compute directly to match PyTorch's kl_div
        if (t > 0.0f && p > 0.0f) {
            thread_sum += t * (__logf(t) - __logf(p));
        }
        // Note: PyTorch's kl_div expects log(predictions) as input,
        // but we're computing the full KL divergence here
    }
    
    // Warp reduction
    float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>());
    
    // First thread in warp writes to shared memory
    __shared__ float block_sums[32];
    if (warp.thread_rank() == 0) {
        block_sums[warp.meta_group_rank()] = warp_sum;
    }
    block.sync();
    
    // First warp reduces all warp sums
    if (warp.meta_group_rank() == 0) {
        float block_sum = 0.0f;
        int num_warps = (total_threads + 31) / 32;
        if (warp.thread_rank() < num_warps) {
            block_sum = block_sums[warp.thread_rank()];
        }
        
        float final_sum = cg::reduce(warp, block_sum, cg::plus<float>());
        
        if (warp.thread_rank() == 0) {
            // Store per-batch result
            output[batch_idx] = static_cast<scalar_t>(final_sum);
        }
    }
}

torch::Tensor fused_kl_div_cuda(torch::Tensor predictions, torch::Tensor targets) {
    // Ensure inputs are on GPU and contiguous
    predictions = predictions.contiguous();
    targets = targets.contiguous();
    
    auto batch_size = predictions.size(0);
    auto seq_len = predictions.size(1);
    
    // Create output tensor for per-batch results
    auto output = torch::zeros({batch_size}, predictions.options());
    
    // Optimized kernel launch configuration
    int threads = 256;
    int blocks = batch_size;
    
    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "fused_kl_div_kernel", ([&] {
        fused_kl_div_kernel<scalar_t><<<blocks, threads>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            seq_len
        );
    }));
    
    // Return batchmean reduction
    return output.mean();
}
"""

kl_div_cpp_source = """
torch::Tensor fused_kl_div_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

# Compile the inline CUDA code
fused_kl_div = load_inline(
    name="fused_kl_div",
    cpp_sources=kl_div_cpp_source,
    cuda_sources=kl_div_source,
    functions=["fused_kl_div_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

class ModelNew(nn.Module):
    """
    Optimized model that computes Kullback-Leibler Divergence using fused CUDA kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.fused_kl_div = fused_kl_div
        
    def forward(self, predictions, targets):
        # Input validation and preprocessing
        predictions = predictions.contiguous()
        targets = targets.contiguous()
        
        # Ensure numerical stability (matching PyTorch's behavior)
        predictions = torch.clamp(predictions, min=1e-12)
        targets = torch.clamp(targets, min=1e-12)
        
        # Normalize to probability distributions
        predictions = predictions / predictions.sum(dim=-1, keepdim=True)
        targets = targets * targets.sum(dim=-1, keepdim=True)
        
        # Use fused CUDA kernel
        return self.fused_kl_div.fused_kl_div_cuda(predictions, targets)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=uniform_constant, 种子=69, 模式=value_stress

**杀死策略描述**: all elements = same constant

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape).softmax(dim=-1), torch.randn(batch_size, *input_shape).softmax(dim=-1)]
```

---

## 47. `L1_P99__arith_replace__15`

- **算子**: `arith_replace` (Category A)
- **描述**: Replace binary arithmetic operators (+→-, -→+, *→/, /→*) @ L77
- **变异行**: Line 77, 原始片段 `/`, 节点类型 `cuda_Div`
- **杀死层**: `layer3_repeated`
- **杀死策略**: `None`
- **杀死种子**: `242`
- **杀死模式**: `repeated_run`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized triplet margin loss with fused operations
triplet_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

constexpr int THREADS_PER_BLOCK = 256;
constexpr int ELEMENTS_PER_THREAD = 8;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void triplet_margin_loss_fused_kernel(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float* __restrict__ output,
    float margin,
    int batch_size,
    int feature_dim
) {
    __shared__ float shared_pos[32];
    __shared__ float shared_neg[32];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (batch_idx >= batch_size) return;
    
    const float* anchor_ptr = anchor + batch_idx * feature_dim;
    const float* positive_ptr = positive + batch_idx * feature_dim;
    const float* negative_ptr = negative + batch_idx * feature_dim;
    
    // Vectorized distance computation with fused sqrt and margin
    float pos_dist = 0.0f;
    float neg_dist = 0.0f;
    
    // Process multiple elements per thread for better memory coalescing
    for (int i = tid * ELEMENTS_PER_THREAD; i < feature_dim; i += blockDim.x * ELEMENTS_PER_THREAD) {
        #pragma unroll
        for (int j = 0; j < ELEMENTS_PER_THREAD; j++) {
            int idx = i + j;
            if (idx < feature_dim) {
                float diff_pos = anchor_ptr[idx] - positive_ptr[idx];
                float diff_neg = anchor_ptr[idx] - negative_ptr[idx];
                pos_dist += diff_pos * diff_pos;
                neg_dist += diff_neg * diff_neg;
            }
        }
    }
    
    // Warp-level reduction
    pos_dist = warp_reduce_sum(pos_dist);
    neg_dist = warp_reduce_sum(neg_dist);
    
    // Store warp results in shared memory
    if (lane_id == 0) {
        shared_pos[warp_id] = pos_dist;
        shared_neg[warp_id] = neg_dist;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (warp_id == 0) {
        float final_pos = lane_id < (blockDim.x / 32) ? shared_pos[lane_id] : 0.0f;
        float final_neg = lane_id < (blockDim.x / 32) ? shared_neg[lane_id] : 0.0f;
        
        final_pos = warp_reduce_sum(final_pos);
        final_neg = warp_reduce_sum(final_neg);
        
        // Thread 0 computes final loss with sqrt and margin
        if (lane_id == 0) {
            float pos_sqrt = sqrtf(final_pos);
            float neg_sqrt = sqrtf(final_neg);
            float loss = fmaxf(0.0f, pos_sqrt - neg_sqrt + margin);
            output[batch_idx] = loss;
        }
    }
}

torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin
) {
    // Input validation
    TORCH_CHECK(anchor.device().is_cuda(), "anchor must be a CUDA tensor");
    TORCH_CHECK(positive.device().is_cuda(), "positive must be a CUDA tensor");
    TORCH_CHECK(negative.device().is_cuda(), "negative must be a CUDA tensor");
    TORCH_CHECK(anchor.dtype() == torch::kFloat32, "anchor must be float32");
    TORCH_CHECK(positive.dtype() == torch::kFloat32, "positive must be float32");
    TORCH_CHECK(negative.dtype() == torch::kFloat32, "negative must be float32");
    
    auto batch_size = anchor.size(0);
    auto feature_dim = anchor.numel() / batch_size;
    
    TORCH_CHECK(positive.sizes() == anchor.sizes(),
                "positive must have same shape as anchor");
    TORCH_CHECK(negative.sizes() == anchor.sizes(),
                "negative must have same shape as anchor");
    
    // Allocate output buffer
    auto output = torch::empty({batch_size}, anchor.options());
    
    // Launch fused kernel: computes distances, sqrt, and margin in one pass
    dim3 grid(batch_size);
    dim3 block(THREADS_PER_BLOCK);
    
    triplet_margin_loss_fused_kernel<<<grid, block>>>(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        output.data_ptr<float>(),
        margin,
        batch_size,
        feature_dim
    );
    
    // Return mean of all losses
    return output.mean();
}
"""

triplet_loss_cpp_source = """
torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin
);
"""

# Compile the inline CUDA code
triplet_loss_module = load_inline(
    name="triplet_margin_loss",
    cpp_sources=triplet_loss_cpp_source,
    cuda_sources=triplet_loss_source,
    functions=["triplet_margin_loss_cuda"],
    verbose=False,
    extra_cflags=["-O3", "-ffast-math"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that computes Triplet Margin Loss with fully fused custom CUDA kernel.
    
    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin
        self.triplet_loss_cuda = triplet_loss_module.triplet_margin_loss_cuda

    def forward(self, anchor, positive, negative):
        # Ensure tensors are on CUDA and have correct dtype
        anchor = anchor.contiguous().cuda().float()
        positive = positive.contiguous().cuda().float()
        negative = negative.contiguous().cuda().float()
        
        return self.triplet_loss_cuda(anchor, positive, negative, self.margin)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized triplet margin loss with fused operations
triplet_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

constexpr int THREADS_PER_BLOCK = 256;
constexpr int ELEMENTS_PER_THREAD = 8;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void triplet_margin_loss_fused_kernel(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float* __restrict__ output,
    float margin,
    int batch_size,
    int feature_dim
) {
    __shared__ float shared_pos[32];
    __shared__ float shared_neg[32];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (batch_idx >= batch_size) return;
    
    const float* anchor_ptr = anchor + batch_idx * feature_dim;
    const float* positive_ptr = positive + batch_idx * feature_dim;
    const float* negative_ptr = negative + batch_idx * feature_dim;
    
    // Vectorized distance computation with fused sqrt and margin
    float pos_dist = 0.0f;
    float neg_dist = 0.0f;
    
    // Process multiple elements per thread for better memory coalescing
    for (int i = tid * ELEMENTS_PER_THREAD; i < feature_dim; i += blockDim.x * ELEMENTS_PER_THREAD) {
        #pragma unroll
        for (int j = 0; j < ELEMENTS_PER_THREAD; j++) {
            int idx = i + j;
            if (idx < feature_dim) {
                float diff_pos = anchor_ptr[idx] - positive_ptr[idx];
                float diff_neg = anchor_ptr[idx] - negative_ptr[idx];
                pos_dist += diff_pos * diff_pos;
                neg_dist += diff_neg * diff_neg;
            }
        }
    }
    
    // Warp-level reduction
    pos_dist = warp_reduce_sum(pos_dist);
    neg_dist = warp_reduce_sum(neg_dist);
    
    // Store warp results in shared memory
    if (lane_id == 0) {
        shared_pos[warp_id] = pos_dist;
        shared_neg[warp_id] = neg_dist;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (warp_id == 0) {
        float final_pos = lane_id < (blockDim.x / 32) ? shared_pos[lane_id] : 0.0f;
        float final_neg = lane_id < (blockDim.x * 32) ? shared_neg[lane_id] : 0.0f;
        
        final_pos = warp_reduce_sum(final_pos);
        final_neg = warp_reduce_sum(final_neg);
        
        // Thread 0 computes final loss with sqrt and margin
        if (lane_id == 0) {
            float pos_sqrt = sqrtf(final_pos);
            float neg_sqrt = sqrtf(final_neg);
            float loss = fmaxf(0.0f, pos_sqrt - neg_sqrt + margin);
            output[batch_idx] = loss;
        }
    }
}

torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin
) {
    // Input validation
    TORCH_CHECK(anchor.device().is_cuda(), "anchor must be a CUDA tensor");
    TORCH_CHECK(positive.device().is_cuda(), "positive must be a CUDA tensor");
    TORCH_CHECK(negative.device().is_cuda(), "negative must be a CUDA tensor");
    TORCH_CHECK(anchor.dtype() == torch::kFloat32, "anchor must be float32");
    TORCH_CHECK(positive.dtype() == torch::kFloat32, "positive must be float32");
    TORCH_CHECK(negative.dtype() == torch::kFloat32, "negative must be float32");
    
    auto batch_size = anchor.size(0);
    auto feature_dim = anchor.numel() / batch_size;
    
    TORCH_CHECK(positive.sizes() == anchor.sizes(),
                "positive must have same shape as anchor");
    TORCH_CHECK(negative.sizes() == anchor.sizes(),
                "negative must have same shape as anchor");
    
    // Allocate output buffer
    auto output = torch::empty({batch_size}, anchor.options());
    
    // Launch fused kernel: computes distances, sqrt, and margin in one pass
    dim3 grid(batch_size);
    dim3 block(THREADS_PER_BLOCK);
    
    triplet_margin_loss_fused_kernel<<<grid, block>>>(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        output.data_ptr<float>(),
        margin,
        batch_size,
        feature_dim
    );
    
    // Return mean of all losses
    return output.mean();
}
"""

triplet_loss_cpp_source = """
torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin
);
"""

# Compile the inline CUDA code
triplet_loss_module = load_inline(
    name="triplet_margin_loss",
    cpp_sources=triplet_loss_cpp_source,
    cuda_sources=triplet_loss_source,
    functions=["triplet_margin_loss_cuda"],
    verbose=False,
    extra_cflags=["-O3", "-ffast-math"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that computes Triplet Margin Loss with fully fused custom CUDA kernel.
    
    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin
        self.triplet_loss_cuda = triplet_loss_module.triplet_margin_loss_cuda

    def forward(self, anchor, positive, negative):
        # Ensure tensors are on CUDA and have correct dtype
        anchor = anchor.contiguous().cuda().float()
        positive = positive.contiguous().cuda().float()
        negative = negative.contiguous().cuda().float()
        
        return self.triplet_loss_cuda(anchor, positive, negative, self.margin)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer3_repeated, 策略=None, 种子=242, 模式=repeated_run

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 128
input_shape = (4096, )

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]
```

---

## 48. `L1_P99__const_perturb__0`

- **算子**: `const_perturb` (Category A)
- **描述**: Perturb numeric literals (integer ±1; float ×1.01 or ×0.99) @ L11
- **变异行**: Line 11, 原始片段 `256`, 节点类型 `const:int+1`
- **杀死层**: `layer1_value`
- **杀死策略**: `all_negative`
- **杀死种子**: `54`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized triplet margin loss with fused operations
triplet_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

constexpr int THREADS_PER_BLOCK = 256;
constexpr int ELEMENTS_PER_THREAD = 8;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void triplet_margin_loss_fused_kernel(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float* __restrict__ output,
    float margin,
    int batch_size,
    int feature_dim
) {
    __shared__ float shared_pos[32];
    __shared__ float shared_neg[32];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (batch_idx >= batch_size) return;
    
    const float* anchor_ptr = anchor + batch_idx * feature_dim;
    const float* positive_ptr = positive + batch_idx * feature_dim;
    const float* negative_ptr = negative + batch_idx * feature_dim;
    
    // Vectorized distance computation with fused sqrt and margin
    float pos_dist = 0.0f;
    float neg_dist = 0.0f;
    
    // Process multiple elements per thread for better memory coalescing
    for (int i = tid * ELEMENTS_PER_THREAD; i < feature_dim; i += blockDim.x * ELEMENTS_PER_THREAD) {
        #pragma unroll
        for (int j = 0; j < ELEMENTS_PER_THREAD; j++) {
            int idx = i + j;
            if (idx < feature_dim) {
                float diff_pos = anchor_ptr[idx] - positive_ptr[idx];
                float diff_neg = anchor_ptr[idx] - negative_ptr[idx];
                pos_dist += diff_pos * diff_pos;
                neg_dist += diff_neg * diff_neg;
            }
        }
    }
    
    // Warp-level reduction
    pos_dist = warp_reduce_sum(pos_dist);
    neg_dist = warp_reduce_sum(neg_dist);
    
    // Store warp results in shared memory
    if (lane_id == 0) {
        shared_pos[warp_id] = pos_dist;
        shared_neg[warp_id] = neg_dist;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (warp_id == 0) {
        float final_pos = lane_id < (blockDim.x / 32) ? shared_pos[lane_id] : 0.0f;
        float final_neg = lane_id < (blockDim.x / 32) ? shared_neg[lane_id] : 0.0f;
        
        final_pos = warp_reduce_sum(final_pos);
        final_neg = warp_reduce_sum(final_neg);
        
        // Thread 0 computes final loss with sqrt and margin
        if (lane_id == 0) {
            float pos_sqrt = sqrtf(final_pos);
            float neg_sqrt = sqrtf(final_neg);
            float loss = fmaxf(0.0f, pos_sqrt - neg_sqrt + margin);
            output[batch_idx] = loss;
        }
    }
}

torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin
) {
    // Input validation
    TORCH_CHECK(anchor.device().is_cuda(), "anchor must be a CUDA tensor");
    TORCH_CHECK(positive.device().is_cuda(), "positive must be a CUDA tensor");
    TORCH_CHECK(negative.device().is_cuda(), "negative must be a CUDA tensor");
    TORCH_CHECK(anchor.dtype() == torch::kFloat32, "anchor must be float32");
    TORCH_CHECK(positive.dtype() == torch::kFloat32, "positive must be float32");
    TORCH_CHECK(negative.dtype() == torch::kFloat32, "negative must be float32");
    
    auto batch_size = anchor.size(0);
    auto feature_dim = anchor.numel() / batch_size;
    
    TORCH_CHECK(positive.sizes() == anchor.sizes(),
                "positive must have same shape as anchor");
    TORCH_CHECK(negative.sizes() == anchor.sizes(),
                "negative must have same shape as anchor");
    
    // Allocate output buffer
    auto output = torch::empty({batch_size}, anchor.options());
    
    // Launch fused kernel: computes distances, sqrt, and margin in one pass
    dim3 grid(batch_size);
    dim3 block(THREADS_PER_BLOCK);
    
    triplet_margin_loss_fused_kernel<<<grid, block>>>(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        output.data_ptr<float>(),
        margin,
        batch_size,
        feature_dim
    );
    
    // Return mean of all losses
    return output.mean();
}
"""

triplet_loss_cpp_source = """
torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin
);
"""

# Compile the inline CUDA code
triplet_loss_module = load_inline(
    name="triplet_margin_loss",
    cpp_sources=triplet_loss_cpp_source,
    cuda_sources=triplet_loss_source,
    functions=["triplet_margin_loss_cuda"],
    verbose=False,
    extra_cflags=["-O3", "-ffast-math"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that computes Triplet Margin Loss with fully fused custom CUDA kernel.
    
    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin
        self.triplet_loss_cuda = triplet_loss_module.triplet_margin_loss_cuda

    def forward(self, anchor, positive, negative):
        # Ensure tensors are on CUDA and have correct dtype
        anchor = anchor.contiguous().cuda().float()
        positive = positive.contiguous().cuda().float()
        negative = negative.contiguous().cuda().float()
        
        return self.triplet_loss_cuda(anchor, positive, negative, self.margin)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized triplet margin loss with fused operations
triplet_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

constexpr int THREADS_PER_BLOCK = 257;
constexpr int ELEMENTS_PER_THREAD = 8;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void triplet_margin_loss_fused_kernel(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float* __restrict__ output,
    float margin,
    int batch_size,
    int feature_dim
) {
    __shared__ float shared_pos[32];
    __shared__ float shared_neg[32];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (batch_idx >= batch_size) return;
    
    const float* anchor_ptr = anchor + batch_idx * feature_dim;
    const float* positive_ptr = positive + batch_idx * feature_dim;
    const float* negative_ptr = negative + batch_idx * feature_dim;
    
    // Vectorized distance computation with fused sqrt and margin
    float pos_dist = 0.0f;
    float neg_dist = 0.0f;
    
    // Process multiple elements per thread for better memory coalescing
    for (int i = tid * ELEMENTS_PER_THREAD; i < feature_dim; i += blockDim.x * ELEMENTS_PER_THREAD) {
        #pragma unroll
        for (int j = 0; j < ELEMENTS_PER_THREAD; j++) {
            int idx = i + j;
            if (idx < feature_dim) {
                float diff_pos = anchor_ptr[idx] - positive_ptr[idx];
                float diff_neg = anchor_ptr[idx] - negative_ptr[idx];
                pos_dist += diff_pos * diff_pos;
                neg_dist += diff_neg * diff_neg;
            }
        }
    }
    
    // Warp-level reduction
    pos_dist = warp_reduce_sum(pos_dist);
    neg_dist = warp_reduce_sum(neg_dist);
    
    // Store warp results in shared memory
    if (lane_id == 0) {
        shared_pos[warp_id] = pos_dist;
        shared_neg[warp_id] = neg_dist;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (warp_id == 0) {
        float final_pos = lane_id < (blockDim.x / 32) ? shared_pos[lane_id] : 0.0f;
        float final_neg = lane_id < (blockDim.x / 32) ? shared_neg[lane_id] : 0.0f;
        
        final_pos = warp_reduce_sum(final_pos);
        final_neg = warp_reduce_sum(final_neg);
        
        // Thread 0 computes final loss with sqrt and margin
        if (lane_id == 0) {
            float pos_sqrt = sqrtf(final_pos);
            float neg_sqrt = sqrtf(final_neg);
            float loss = fmaxf(0.0f, pos_sqrt - neg_sqrt + margin);
            output[batch_idx] = loss;
        }
    }
}

torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin
) {
    // Input validation
    TORCH_CHECK(anchor.device().is_cuda(), "anchor must be a CUDA tensor");
    TORCH_CHECK(positive.device().is_cuda(), "positive must be a CUDA tensor");
    TORCH_CHECK(negative.device().is_cuda(), "negative must be a CUDA tensor");
    TORCH_CHECK(anchor.dtype() == torch::kFloat32, "anchor must be float32");
    TORCH_CHECK(positive.dtype() == torch::kFloat32, "positive must be float32");
    TORCH_CHECK(negative.dtype() == torch::kFloat32, "negative must be float32");
    
    auto batch_size = anchor.size(0);
    auto feature_dim = anchor.numel() / batch_size;
    
    TORCH_CHECK(positive.sizes() == anchor.sizes(),
                "positive must have same shape as anchor");
    TORCH_CHECK(negative.sizes() == anchor.sizes(),
                "negative must have same shape as anchor");
    
    // Allocate output buffer
    auto output = torch::empty({batch_size}, anchor.options());
    
    // Launch fused kernel: computes distances, sqrt, and margin in one pass
    dim3 grid(batch_size);
    dim3 block(THREADS_PER_BLOCK);
    
    triplet_margin_loss_fused_kernel<<<grid, block>>>(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        output.data_ptr<float>(),
        margin,
        batch_size,
        feature_dim
    );
    
    // Return mean of all losses
    return output.mean();
}
"""

triplet_loss_cpp_source = """
torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin
);
"""

# Compile the inline CUDA code
triplet_loss_module = load_inline(
    name="triplet_margin_loss",
    cpp_sources=triplet_loss_cpp_source,
    cuda_sources=triplet_loss_source,
    functions=["triplet_margin_loss_cuda"],
    verbose=False,
    extra_cflags=["-O3", "-ffast-math"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that computes Triplet Margin Loss with fully fused custom CUDA kernel.
    
    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin
        self.triplet_loss_cuda = triplet_loss_module.triplet_margin_loss_cuda

    def forward(self, anchor, positive, negative):
        # Ensure tensors are on CUDA and have correct dtype
        anchor = anchor.contiguous().cuda().float()
        positive = positive.contiguous().cuda().float()
        negative = negative.contiguous().cuda().float()
        
        return self.triplet_loss_cuda(anchor, positive, negative, self.margin)
```

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=all_negative, 种子=54, 模式=value_stress

**杀死策略描述**: all values < 0

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 128
input_shape = (4096, )

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]
```

---

## 49. `L2_P41__scale_modify__1`

- **算子**: `scale_modify` (Category C)
- **描述**: Distort attention / normalization scaling (inv-sqrt, rsqrt) @ L184
- **变异行**: Line 184, 原始片段 `rsqrtf(var + eps)`, 节点类型 ``
- **杀死层**: `layer1_value`
- **杀死策略**: `large_magnitude`
- **杀死种子**: `48`
- **杀死模式**: `value_stress`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define fused GEMM + BatchNorm + GELU kernel with tiling optimization
fused_gemm_bn_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

constexpr int TILE_SIZE = 32;
constexpr int BLOCK_SIZE = 256;

template<typename T>
__device__ __forceinline__ T gelu_forward(T x) {
    const T sqrt_2_over_pi = 0.7978845608028654;
    const T coeff = 0.044715;
    T x_cube = x * x * x;
    T inner = sqrt_2_over_pi * (x + coeff * x_cube);
    return 0.5 * x * (1.0 + tanhf(inner));
}

template<typename T>
__global__ void fused_gemm_bn_gelu_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    const T* __restrict__ running_mean,
    const T* __restrict__ running_var,
    const T* __restrict__ weight_bn,
    const T* __restrict__ bias_bn,
    T* __restrict__ output,
    int M, int N, int K,
    float eps) {
    
    __shared__ T tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ T tile_b[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    T acc = 0.0;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int tiled_k = t * TILE_SIZE;
        
        // Load tile from input matrix A
        if (row < M && (tiled_k + threadIdx.x) < K) {
            tile_a[threadIdx.y][threadIdx.x] = input[row * K + tiled_k + threadIdx.x];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        // Load tile from weight matrix B (transposed)
        if (col < N && (tiled_k + threadIdx.y) < K) {
            tile_b[threadIdx.y][threadIdx.x] = weight[col * K + tiled_k + threadIdx.y];
        } else {
            tile_b[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        // Add bias if present
        if (bias) {
            acc += bias[col];
        }
        
        // BatchNorm
        T mean = running_mean[col];
        T var = running_var[col];
        T inv_std = rsqrtf(var + eps);
        T normalized = (acc - mean) * inv_std;
        
        // Scale and shift
        T scaled = normalized * weight_bn[col] + bias_bn[col];
        
        // GELU activation
        output[row * N + col] = gelu_forward(scaled);
    }
}

torch::Tensor fused_gemm_bn_gelu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    torch::Tensor weight_bn, torch::Tensor bias_bn,
    float eps) {
    
    int M = input.size(0);
    int K = input.size(1);
    int N = weight.size(0);
    
    auto output = torch::empty({M, N}, input.options());
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_gemm_bn_gelu", ([&] {
        fused_gemm_bn_gelu_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            running_mean.data_ptr<scalar_t>(),
            running_var.data_ptr<scalar_t>(),
            weight_bn.data_ptr<scalar_t>(),
            bias_bn.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            M, N, K, eps);
    }));
    
    return output;
}
"""

# Define optimized GroupNorm + Mean + ReLU kernel
fused_group_norm_mean_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__global__ void fused_group_norm_mean_relu_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ weight_gn,
    const T* __restrict__ bias_gn,
    int batch_size, int channels, int num_groups,
    float eps) {
    
    // Each thread block handles one batch element
    int batch = blockIdx.x;
    const int group_size = channels / num_groups;
    
    // Shared memory for group statistics
    extern __shared__ float shared_mem[];
    float* group_means = shared_mem;
    float* group_vars = &shared_mem[num_groups];
    
    // Phase 1: Compute group statistics
    for (int group = threadIdx.x; group < num_groups; group += blockDim.x) {
        int start_idx = batch * channels + group * group_size;
        
        float sum = 0.0f;
        float sqsum = 0.0f;
        
        for (int i = 0; i < group_size; i++) {
            float val = static_cast<float>(input[start_idx + i]);
            sum += val;
            sqsum += val * val;
        }
        
        float mean = sum / group_size;
        float variance = (sqsum / group_size) - (mean * mean);
        
        group_means[group] = mean;
        group_vars[group] = variance;
    }
    
    __syncthreads();
    
    // Phase 2: Apply GroupNorm and compute mean across channels
    float channel_sum = 0.0f;
    
    for (int i = threadIdx.x; i < channels; i += blockDim.x) {
        int group = i / group_size;
        
        int idx = batch * channels + i;
        float val = static_cast<float>(input[idx]);
        
        // GroupNorm normalization
        float mean = group_means[group];
        float var = group_vars[group];
        float inv_std = rsqrtf(var + eps);
        float normalized = (val - mean) * inv_std;
        
        // Scale and shift
        float scaled = normalized * static_cast<float>(weight_gn[i]) + 
                      static_cast<float>(bias_gn[i]);
        
        channel_sum += scaled;
    }
    
    // Parallel reduction for channel sum
    __shared__ float channel_sum_shared[256];
    channel_sum_shared[threadIdx.x] = channel_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            channel_sum_shared[threadIdx.x] += channel_sum_shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        float mean_across_channels = channel_sum_shared[0] / channels;
        output[batch] = fmaxf(mean_across_channels, 0.0f);
    }
}

torch::Tensor fused_group_norm_mean_relu_cuda(
    torch::Tensor input, torch::Tensor weight_gn, torch::Tensor bias_gn,
    int num_groups, float eps) {
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    
    auto output = torch::empty({batch_size, 1}, input.options());
    
    // Use optimal block size
    const int block_size = min(256, channels);
    dim3 block(block_size);
    dim3 grid(batch_size);
    
    // Shared memory size
    int shared_mem_size = 2 * num_groups * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_group_norm_mean_relu", ([&] {
        fused_group_norm_mean_relu_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            weight_gn.data_ptr<scalar_t>(),
            bias_gn.data_ptr<scalar_t>(),
            batch_size, channels, num_groups, eps);
    }));
    
    return output;
}
"""

# Compile the fused kernels
fused_gemm_bn_gelu = load_inline(
    name="fused_gemm_bn_gelu",
    cpp_sources="torch::Tensor fused_gemm_bn_gelu_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float);",
    cuda_sources=fused_gemm_bn_gelu_source,
    functions=["fused_gemm_bn_gelu_cuda"],
    verbose=False
)

fused_group_norm_mean_relu = load_inline(
    name="fused_group_norm_mean_relu",
    cpp_sources="torch::Tensor fused_group_norm_mean_relu_cuda(torch::Tensor, torch::Tensor, torch::Tensor, int, float);",
    cuda_sources=fused_group_norm_mean_relu_source,
    functions=["fused_group_norm_mean_relu_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        
        # GEMM parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # BatchNorm parameters
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        self.weight_bn = nn.Parameter(torch.ones(out_features))
        self.bias_bn = nn.Parameter(torch.zeros(out_features))
        
        # GroupNorm parameters
        self.weight_gn = nn.Parameter(torch.ones(out_features))
        self.bias_gn = nn.Parameter(torch.zeros(out_features))
        
        # BN hyperparameters
        self.eps = 1e-5
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
        nn.init.ones_(self.weight_bn)
        nn.init.zeros_(self.bias_bn)
        nn.init.ones_(self.weight_gn)
        nn.init.zeros_(self.bias_gn)
    
    def forward(self, x):
        # Fused GEMM + BatchNorm + GELU
        x = fused_gemm_bn_gelu.fused_gemm_bn_gelu_cuda(
            x, self.weight, self.bias,
            self.running_mean, self.running_var,
            self.weight_bn, self.bias_bn,
            self.eps
        )
        
        # Fused GroupNorm + Mean + ReLU
        x = fused_group_norm_mean_relu.fused_group_norm_mean_relu_cuda(
            x, self.weight_gn, self.bias_gn,
            self.num_groups, self.eps
        )
        
        return x
```

### Part 2: 变异后的完整源代码

*(无变异代码数据)*

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer1_value, 策略=large_magnitude, 种子=48, 模式=value_stress

**杀死策略描述**: values ~1e6

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 128
in_features = 512

def get_inputs():
    return [torch.randn(batch_size, in_features)]
```

---

## 50. `L2_P9__index_replace__8`

- **算子**: `index_replace` (Category B)
- **描述**: Swap Triton program_id axis or CUDA thread/block dimension index (e.g. program_id(0)→(1), threadIdx.x→threadIdx.y) @ L21
- **变异行**: Line 21, 原始片段 `blockDim.x`, 节点类型 ``
- **杀死层**: `layer3_repeated`
- **杀死策略**: `None`
- **杀死种子**: `244`
- **杀死模式**: `repeated_run`

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for linear + subtract + multiply + relu
fused_operation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename T>
__global__ void fused_linear_sub_mul_relu_kernel(
    const T* __restrict__ input, 
    const T* __restrict__ weight, 
    const T* __restrict__ bias,
    T* __restrict__ output, 
    int batch_size, int in_features, int out_features,
    T subtract_value, T multiply_value) {
    
    // Use 2D thread blocks for better memory coalescing
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        T sum = bias ? bias[col] : static_cast<T>(0);
        
        // Unroll the inner loop for better performance
        #pragma unroll(4)
        for (int k = 0; k < in_features; ++k) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        
        // Fused operations: subtract, multiply, and ReLU
        T result = sum - subtract_value;
        result *= multiply_value;
        result = max(result, static_cast<T>(0));
        
        output[row * out_features + col] = result;
    }
}

torch::Tensor fused_linear_sub_mul_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float subtract_value, float multiply_value) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, 
                               torch::TensorOptions()
                               .dtype(input.dtype())
                               .device(input.device()));
    
    // Optimized block and grid dimensions
    dim3 block_size(32, 8);  // 256 threads total
    dim3 grid_size((out_features + block_size.x - 1) / block_size.x,
                   (batch_size + block_size.y - 1) / block_size.y);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_linear_sub_mul_relu_cuda", [&] {
        fused_linear_sub_mul_relu_kernel<scalar_t><<<grid_size, block_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size, in_features, out_features,
            static_cast<scalar_t>(subtract_value),
            static_cast<scalar_t>(multiply_value));
    });
    
    return output;
}
"""

fused_operation_cpp_source = """
torch::Tensor fused_linear_sub_mul_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float subtract_value, float multiply_value);
"""

# Compile the inline CUDA code
fused_operation = load_inline(
    name="fused_operation",
    cpp_sources=fused_operation_cpp_source,
    cuda_sources=fused_operation_source,
    functions=["fused_linear_sub_mul_relu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for linear + subtract + multiply + ReLU.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        
        # Initialize weights and bias like nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Initialize parameters using PyTorch's standard initialization
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
        
        self.fused_op = fused_operation

    def forward(self, x):
        return self.fused_op.fused_linear_sub_mul_relu_cuda(
            x, self.weight, self.bias, self.subtract_value, self.multiply_value)
```

### Part 2: 变异后的完整源代码

*(无变异代码数据)*

### Part 3: 杀死该变异体的输入

**杀死方式**: 层=layer3_repeated, 策略=None, 种子=244, 模式=repeated_run

**基础输入生成代码** (压力策略在此基础上修改值分布)：

```python
batch_size = 128
in_features = 10

def get_inputs():
    return [torch.randn(batch_size, in_features)]
```

---