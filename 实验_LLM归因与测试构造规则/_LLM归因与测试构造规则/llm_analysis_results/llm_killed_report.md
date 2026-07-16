# LLM 分析后新杀死的变异体

**总计**: 31 个变异体
**模型**: DeepSeek-R1 (deepseek-reasoner)
**最大迭代轮数**: 3

---

## 1. `L1_P100__index_replace__1`

- **Kernel**: `L1_P100`
- **算子**: `index_replace` (Category B)
- **描述**: Swap Triton program_id axis or CUDA thread/block dimension index (e.g. program_id(0)→(1), threadIdx.x→threadIdx.y) @ L13
- **变异行**: Line 13, 原始片段 `blockIdx.x`, 节点类型 `cuda_dim|z`
- **杀死轮次**: Round 1

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
    
    int idx = blockIdx.z * blockDim.x + threadIdx.x;
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

### Part 3: 杀死该变异体的输入

**杀死输入代码 (Round 1)：**

```python
import torch

def generate_inputs(device):
    # Create tensors with 512 elements (exceeding block size 256)
    # Shape [512, 1] maintains the same rank as original spec
    predictions = torch.zeros(512, 1, dtype=torch.float32, device=device)
    targets = torch.zeros(512, 1, dtype=torch.float32, device=device)
    
    # First 256 elements: margin = 1 - 1*1 = 0 → zero loss
    predictions[:256] = 1.0
    targets[:256] = 1.0
    
    # Elements 257-512: margin = 1 - (-1)*1 = 2 → positive loss
    predictions[256:] = -1.0
    targets[256:] = 1.0
    
    return [predictions, targets]
```

**输出差异**: `max_diff=1.000000e+00, mean_diff=1.000000e+00, ref_range=[1.0000e+00,1.0000e+00], mut_range=[0.0000e+00,0.0000e+00]`

### Part 4: 大模型诊断流程

共 1 轮迭代分析：

#### Round 1 (**KILLED**)

**LLM 存活原因分析：**

> The mutant survived because the original input shape [128, 1] (128 elements) fits within a single CUDA block (256 threads). With blockIdx.z always 0 in a 1D grid launch, the mutated calculation 'blockIdx.z * blockDim.x + threadIdx.x' equals 'threadIdx.x', which for block 0 is identical to the original 'blockIdx.x * blockDim.x + threadIdx.x' (since blockIdx.x=0). For inputs ≤256 elements, only one block is launched, making the mutant mathematically equivalent to the original. All stress tests varied only values, not shapes, so they never triggered the scenario where multiple blocks are needed.

**LLM 杀死策略：**

> Use input sizes larger than the thread block dimension (256) to force multiple blocks. The mutant will incorrectly compute indices for blocks beyond the first, causing data access patterns to wrap around within the first block's range. Specifically, set the input size >256 so blockIdx.x > 0, making blockIdx.z (always 0) diverge from blockIdx.x.

**LLM 建议的输入代码：**

```python
import torch

def generate_inputs(device):
    # Create tensors with 512 elements (exceeding block size 256)
    # Shape [512, 1] maintains the same rank as original spec
    predictions = torch.zeros(512, 1, dtype=torch.float32, device=device)
    targets = torch.zeros(512, 1, dtype=torch.float32, device=device)
    
    # First 256 elements: margin = 1 - 1*1 = 0 → zero loss
    predictions[:256] = 1.0
    targets[:256] = 1.0
    
    # Elements 257-512: margin = 1 - (-1)*1 = 2 → positive loss
    predictions[256:] = -1.0
    targets[256:] = 1.0
    
    return [predictions, targets]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=1.000000e+00, mean_diff=1.000000e+00, ref_range=[1.0000e+00,1.0000e+00], mut_range=[0.0000e+00,0.0000e+00]`

#### 测试构造规则

- **规则名**: `force_multiple_blocks`
- **描述**: Generates inputs that exceed a single thread block's capacity, requiring multiple CUDA blocks for full processing. This exposes mutations that affect block/thread indexing by creating scenarios where incorrect index calculations access wrong memory regions. The input ensures different behavior between first and subsequent blocks by setting distinct values in different block regions.
- **适用算子**: index_replace, thread_replace, block_replace, grid_replace

```python
def policy(shape, dtype, device, rng):
    import torch
    import math
    
    # Target at least 2 blocks worth of data (assuming typical 256-thread blocks)
    # Keep original tensor rank but expand first dimension
    base_elements = 512  # Enough for 2 blocks of 256 threads
    
    # Calculate expanded shape: replace first dimension with at least base_elements
    if shape[0] < base_elements:
        expanded_shape = (base_elements,) + shape[1:]
    else:
        # If already large, keep original shape but ensure it's multiple of block size
        expanded_shape = shape
    
    # Create two tensors for typical loss function inputs
    # First tensor: predictions
    predictions = torch.zeros(expanded_shape, dtype=dtype, device=device)
    targets = torch.zeros(expanded_shape, dtype=dtype, device=device)
    
    # Block size assumption: typical 256 threads/block
    # Adjust if kernel metadata indicates different block size
    block_size = 256  # Conservative default
    
    # First block region: set to produce zero contribution
    # (e.g., identical values for loss functions)
    first_block_end = min(block_size, expanded_shape[0])
    predictions[:first_block_end] = 1.0
    targets[:first_block_end] = 1.0
    
    # Second block region: set to produce non-zero contribution
    if expanded_shape[0] > block_size:
        second_block_start = block_size
        second_block_end = min(2 * block_size, expanded_shape[0])
        predictions[second_block_start:second_block_end] = -1.0
        targets[second_block_start:second_block_end] = 1.0
        
        # Fill any remaining elements beyond 2 blocks with non-zero values
        if expanded_shape[0] > 2 * block_size:
            predictions[2 * block_size:] = -1.0
            targets[2 * block_size:] = 1.0
    
    return [predictions, targets]
```


---

## 2. `L1_P17__relop_replace__7`

- **Kernel**: `L1_P17`
- **算子**: `relop_replace` (Category A)
- **描述**: Replace relational operators (<→<=, <=→<, >→>=, >=→>, ==→!=, !=→==) @ L52
- **变异行**: Line 52, 原始片段 `<`, 节点类型 `cuda_Lt`
- **杀死轮次**: Round 1

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Tile size for shared memory optimization
#define TILE_SIZE 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                              int M, int N, int K) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile from A
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from B (transposed)
        if (col < N && (t * TILE_SIZE + threadIdx.y) < K) {
            Bs[threadIdx.y][threadIdx.x] = B[col * K + t * TILE_SIZE + threadIdx.y];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Get dimensions
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);
    
    // Create output tensor
    auto C = torch::zeros({M, N}, A.options());
    
    // Configure kernel launch parameters
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, 
                 (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // Launch kernel
    matmul_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for matrix multiplication
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_cuda = matmul_cuda
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized matrix multiplication using custom CUDA kernel.
        
        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (N, K).
            
        Returns:
            Output tensor of shape (M, N).
        """
        return self.matmul_cuda.matmul_cuda(A, B)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Tile size for shared memory optimization
#define TILE_SIZE 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                              int M, int N, int K) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile from A
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from B (transposed)
        if (col < N && (t * TILE_SIZE + threadIdx.y) < K) {
            Bs[threadIdx.y][threadIdx.x] = B[col * K + t * TILE_SIZE + threadIdx.y];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col <= N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Get dimensions
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);
    
    // Create output tensor
    auto C = torch::zeros({M, N}, A.options());
    
    // Configure kernel launch parameters
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, 
                 (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // Launch kernel
    matmul_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for matrix multiplication
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_cuda = matmul_cuda
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized matrix multiplication using custom CUDA kernel.
        
        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (N, K).
            
        Returns:
            Output tensor of shape (M, N).
        """
        return self.matmul_cuda.matmul_cuda(A, B)
```

### Part 3: 杀死该变异体的输入

**杀死输入代码 (Round 1)：**

```python
import torch
import math

def generate_inputs(device):
    M = 1024
    K = 4096
    N = 2047  # Changed to break tile-size alignment
    # Use random values to ensure non‑zero writes; the values themselves are not critical.
    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(N, K, device=device, dtype=torch.float32)  # Note: shape [2047, 4096]
    return [A, B]
```

**输出差异**: `max_diff=2.175872e+02, mean_diff=2.422097e-02, ref_range=[-3.2816e+02,3.2297e+02], mut_range=[-3.2816e+02,3.2297e+02]`

### Part 4: 大模型诊断流程

共 1 轮迭代分析：

#### Round 1 (**KILLED**)

**LLM 存活原因分析：**

> The mutant changes the boundary condition from `col < N` to `col <= N`, allowing writes at column index `col = N`. However, in CUDA grid-stride loops, the thread's column index is computed as `col = blockIdx.x * TILE_SIZE + threadIdx.x`. With the original input shape (N=2048) and typical TILE_SIZE values (like 16, 32, 64), N is exactly divisible by TILE_SIZE. This means the maximum computed `col` value is `N - 1`, so the condition `col <= N` behaves identically to `col < N`—no thread ever has `col == N`. Thus, the mutant is semantically equivalent to the original for these specific dimensions.

**LLM 杀死策略：**

> Use input dimensions where N is NOT a multiple of TILE_SIZE. This ensures that the last block in the x‑direction contains threads with `col = N` (due to overshoot), which the mutant will write to (causing out‑of‑bounds access), while the original kernel correctly excludes them. The exact TILE_SIZE must be known or guessed; common values are 16, 32, 64. Choosing N = 2047 (one less than a power‑of‑two) avoids being a multiple of most tile sizes.

**LLM 建议的输入代码：**

```python
import torch
import math

def generate_inputs(device):
    M = 1024
    K = 4096
    N = 2047  # Changed to break tile-size alignment
    # Use random values to ensure non‑zero writes; the values themselves are not critical.
    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(N, K, device=device, dtype=torch.float32)  # Note: shape [2047, 4096]
    return [A, B]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=2.175872e+02, mean_diff=2.422097e-02, ref_range=[-3.2816e+02,3.2297e+02], mut_range=[-3.2816e+02,3.2297e+02]`

#### 测试构造规则

- **规则名**: `misaligned_grid_boundary`
- **描述**: Targets kernels that use grid-stride loops with power‑of‑two tile sizes. By setting problem dimensions to be one less than a multiple of common tile sizes (e.g., 16, 32, 64), the last block of threads will compute indices that exactly equal the dimension boundary. This exposes relational-operator mutants (like < → ≤) that cause out‑of‑bounds accesses.
- **适用算子**: relop_replace, relop_negation, relop_exchange

```python
import torch
import numpy as np

def policy(shape, dtype, device, rng):
    # This rule assumes the kernel is a matrix multiplication that expects two inputs:
    #   A of shape (M, K)
    #   B of shape (N, K)
    # The critical dimension is N, which controls the column index boundary.
    
    # Choose N so it is NOT a multiple of common tile sizes.
    # 2047 = 2048 - 1, where 2048 is divisible by 16, 32, 64, 128, 256, 512, 1024, 2048.
    # This ensures that for any of these tile sizes, the last block will have a thread
    # with column index exactly equal to N.
    M = 1024
    K = 4096
    N = 2047  # critical: one less than a power‑of‑two multiple of common tile sizes
    
    # Use random data; exact values are irrelevant as long as they are non‑zero.
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(N, K, device=device, dtype=dtype)
    return [A, B]
```


---

## 3. `L1_P18__relop_replace__3`

- **Kernel**: `L1_P18`
- **算子**: `relop_replace` (Category A)
- **描述**: Replace relational operators (<→<=, <=→<, >→>=, >=→>, ==→!=, !=→==) @ L26
- **变异行**: Line 26, 原始片段 `<`, 节点类型 `cuda_Lt`
- **杀死轮次**: Round 1

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized matrix multiplication
matmul_optimized_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Kernel for transposing matrix A (K, M) -> (M, K)
__global__ void transpose_A_kernel(const float* A, float* A_T, int K, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < M && idy < K) {
        A_T[idx * K + idy] = A[idy * M + idx];
    }
}

// Kernel for transposing matrix B (N, K) -> (K, N)
__global__ void transpose_B_kernel(const float* B, float* B_T, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < K && idy < N) {
        B_T[idx * N + idy] = B[idy * K + idx];
    }
}

torch::Tensor matmul_optimized_cuda(torch::Tensor A, torch::Tensor B) {
    // Get dimensions
    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(0);
    
    // Create output tensor
    auto C = torch::zeros({M, N}, A.options());
    
    // Create transposed versions
    auto A_T = torch::zeros({M, K}, A.options());
    auto B_T = torch::zeros({K, N}, B.options());
    
    // Transpose A (K, M) -> (M, K)
    dim3 blockDim1(16, 16);
    dim3 gridDim1((M + 15) / 16, (K + 15) / 16);
    transpose_A_kernel<<<gridDim1, blockDim1>>>(
        A.data_ptr<float>(),
        A_T.data_ptr<float>(),
        K, M
    );
    
    // Transpose B (N, K) -> (K, N)
    dim3 gridDim2((K + 15) / 16, (N + 15) / 16);
    transpose_B_kernel<<<gridDim2, blockDim1>>>(
        B.data_ptr<float>(),
        B_T.data_ptr<float>(),
        N, K
    );
    
    // Use cuBLAS for matrix multiplication: C = A_T * B_T
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,  // Note: dimensions are swapped for column-major
                &alpha,
                B_T.data_ptr<float>(), N,
                A_T.data_ptr<float>(), K,
                &beta,
                C.data_ptr<float>(), N);
    
    cublasDestroy(handle);
    
    return C;
}
"""

matmul_optimized_cpp_source = """
torch::Tensor matmul_optimized_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code
matmul_optimized = load_inline(
    name="matmul_optimized",
    cpp_sources=matmul_optimized_cpp_source,
    cuda_sources=matmul_optimized_source,
    functions=["matmul_optimized_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for matrix multiplication
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_optimized = matmul_optimized
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized matrix multiplication A.T * B.T
        
        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (N, K).
            
        Returns:
            Output tensor of shape (M, N).
        """
        return self.matmul_optimized.matmul_optimized_cuda(A, B)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized matrix multiplication
matmul_optimized_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Kernel for transposing matrix A (K, M) -> (M, K)
__global__ void transpose_A_kernel(const float* A, float* A_T, int K, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < M && idy < K) {
        A_T[idx * K + idy] = A[idy * M + idx];
    }
}

// Kernel for transposing matrix B (N, K) -> (K, N)
__global__ void transpose_B_kernel(const float* B, float* B_T, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < K && idy <= N) {
        B_T[idx * N + idy] = B[idy * K + idx];
    }
}

torch::Tensor matmul_optimized_cuda(torch::Tensor A, torch::Tensor B) {
    // Get dimensions
    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(0);
    
    // Create output tensor
    auto C = torch::zeros({M, N}, A.options());
    
    // Create transposed versions
    auto A_T = torch::zeros({M, K}, A.options());
    auto B_T = torch::zeros({K, N}, B.options());
    
    // Transpose A (K, M) -> (M, K)
    dim3 blockDim1(16, 16);
    dim3 gridDim1((M + 15) / 16, (K + 15) / 16);
    transpose_A_kernel<<<gridDim1, blockDim1>>>(
        A.data_ptr<float>(),
        A_T.data_ptr<float>(),
        K, M
    );
    
    // Transpose B (N, K) -> (K, N)
    dim3 gridDim2((K + 15) / 16, (N + 15) / 16);
    transpose_B_kernel<<<gridDim2, blockDim1>>>(
        B.data_ptr<float>(),
        B_T.data_ptr<float>(),
        N, K
    );
    
    // Use cuBLAS for matrix multiplication: C = A_T * B_T
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,  // Note: dimensions are swapped for column-major
                &alpha,
                B_T.data_ptr<float>(), N,
                A_T.data_ptr<float>(), K,
                &beta,
                C.data_ptr<float>(), N);
    
    cublasDestroy(handle);
    
    return C;
}
"""

matmul_optimized_cpp_source = """
torch::Tensor matmul_optimized_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code
matmul_optimized = load_inline(
    name="matmul_optimized",
    cpp_sources=matmul_optimized_cpp_source,
    cuda_sources=matmul_optimized_source,
    functions=["matmul_optimized_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for matrix multiplication
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_optimized = matmul_optimized
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized matrix multiplication A.T * B.T
        
        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (N, K).
            
        Returns:
            Output tensor of shape (M, N).
        """
        return self.matmul_optimized.matmul_optimized_cuda(A, B)
```

### Part 3: 杀死该变异体的输入

**杀死输入代码 (Round 1)：**

```python
import torch
def generate_inputs(device):
    # Keep inner dimension K=4096 consistent with original, but change N to 2049 (not multiple of 16)
    K = 4096
    M = 1024  # unchanged
    N = 2049  # 2048 + 1 to cause over-provisioning
    A = torch.randn(K, M, device=device, dtype=torch.float32)
    B = torch.randn(N, K, device=device, dtype=torch.float32)
    return [A, B]
```

**输出差异**: `max_diff=2.304568e+02, mean_diff=2.481521e-02, ref_range=[-3.4798e+02,3.1339e+02], mut_range=[-3.4798e+02,3.1339e+02]`

### Part 4: 大模型诊断流程

共 1 轮迭代分析：

#### Round 1 (**KILLED**)

**LLM 存活原因分析：**

> The mutant survived because the kernel launch parameters (grid/block dimensions) are set to exactly cover the matrix dimensions with no over-provisioning. Specifically, the input shape used in stress tests (N=2048) is a multiple of the typical block size (16), so gridDim.y = N/blockDim.y = 128 exactly. Thus, the maximum thread idy is N-1, making the condition `idy <= N` equivalent to `idy < N` (since idy never reaches N). No out-of-bounds access occurs, so the mutant behaves identically to the original.

**LLM 杀死策略：**

> Change the input shape so that N (first dimension of B) is not a multiple of the thread block size in the y-dimension (likely 16). This causes the grid to be over-provisioned, creating threads with idy = N. The original condition (`idy < N`) rejects these threads, but the mutant (`idy <= N`) accepts them, leading to out-of-bounds writes in B_T and out-of-bounds reads from B.

**LLM 建议的输入代码：**

```python
import torch
def generate_inputs(device):
    # Keep inner dimension K=4096 consistent with original, but change N to 2049 (not multiple of 16)
    K = 4096
    M = 1024  # unchanged
    N = 2049  # 2048 + 1 to cause over-provisioning
    A = torch.randn(K, M, device=device, dtype=torch.float32)
    B = torch.randn(N, K, device=device, dtype=torch.float32)
    return [A, B]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=2.304568e+02, mean_diff=2.481521e-02, ref_range=[-3.4798e+02,3.1339e+02], mut_range=[-3.4798e+02,3.1339e+02]`

#### 测试构造规则

- **规则名**: `non_multiple_block_shape_dimension`
- **描述**: This rule creates inputs where matrix dimensions are NOT multiples of typical GPU block sizes (e.g., 16, 32). This triggers grid over-provisioning where extra threads are launched beyond the actual data boundaries. This exposes boundary-condition mutations by allowing threads with indices equal to the dimension size to execute, which would cause out-of-bounds access when relational operators are mutated.
- **适用算子**: relop_replace, boundary_condition

```python
def policy(shape, dtype, device, rng):
    import torch
    import numpy as np
    
    # Common GPU block sizes to avoid multiples of
    common_block_sizes = [16, 32, 64, 128, 256]
    target_block_size = common_block_sizes[rng.randint(0, len(common_block_sizes))]
    
    # For matrix multiplication kernels like L1_P18, we expect shapes:
    # A: [K, M], B: [N, K]
    if len(shape) == 2:
        # If this is likely the B matrix (N, K)
        N, K = shape
        # Make N not a multiple of target_block_size
        # Choose a prime-ish offset (1, 3, 5, 7, 11) to avoid multiples of other block sizes too
        offsets = [1, 3, 5, 7, 11]
        offset = offsets[rng.randint(0, len(offsets))]
        new_N = N + offset
        
        # Generate random data with the new shape
        return torch.randn(new_N, K, device=device, dtype=dtype)
    else:
        # For other tensors, return original shape with random data
        return torch.randn(*shape, device=device, dtype=dtype)
```


---

## 4. `L1_P1__relop_replace__7`

- **Kernel**: `L1_P1`
- **算子**: `relop_replace` (Category A)
- **描述**: Replace relational operators (<→<=, <=→<, >→>=, >=→>, ==→!=, !=→==) @ L50
- **变异行**: Line 50, 原始片段 `<`, 节点类型 `cuda_Lt`
- **杀死轮次**: Round 2

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Using shared memory for tile-based matrix multiplication
__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    const int TILE_SIZE = 32;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile from A
        if (row < N && (tile * TILE_SIZE + tx) < N) {
            As[ty][tx] = A[row * N + (tile * TILE_SIZE + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile from B
        if ((tile * TILE_SIZE + ty) < N && col < N) {
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());
    
    const int TILE_SIZE = 32;
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
    
    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for matrix multiplication
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_cuda = matmul_cuda
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized matrix multiplication using custom CUDA kernel.
        
        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).
            
        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return self.matmul_cuda.matmul_cuda(A, B)

N = 2048

def get_inputs():
    A = torch.randn(N, N).cuda()
    B = torch.randn(N, N).cuda()
    return [A, B]

def get_init_inputs():
    return []
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Using shared memory for tile-based matrix multiplication
__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    const int TILE_SIZE = 32;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile from A
        if (row < N && (tile * TILE_SIZE + tx) < N) {
            As[ty][tx] = A[row * N + (tile * TILE_SIZE + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile from B
        if ((tile * TILE_SIZE + ty) < N && col < N) {
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < N && col <= N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());
    
    const int TILE_SIZE = 32;
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
    
    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for matrix multiplication
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_cuda = matmul_cuda
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized matrix multiplication using custom CUDA kernel.
        
        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).
            
        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return self.matmul_cuda.matmul_cuda(A, B)

N = 2048

def get_inputs():
    A = torch.randn(N, N).cuda()
    B = torch.randn(N, N).cuda()
    return [A, B]

def get_init_inputs():
    return []
```

### Part 3: 杀死该变异体的输入

**杀死输入代码 (Round 2)：**

```python
import torch
def generate_inputs(device):
    # N chosen to cause integer overflow in index calculation when col=N
    # N = 46341 -> N*N = 2,147,488,281 > 2^31-1
    N = 46341
    torch.manual_seed(123)
    # Use smaller matrices to fit memory, but the kernel will still compute with N
    # Actually, we need to allocate NxN matrices which is ~8.6GB for float32
    # This is too large for most GPUs. Let's instead use a smaller N that still causes
    # significant out-of-bounds writes beyond the allocation.
    # Alternative: Use N = 8193 (not multiple of 32) which creates many out-of-bounds writes
    # The allocation is 8193*8193 = 67.1M elements = 268MB
    # Threads with col=8193 write to indices row*8193 + 8193 = (row+1)*8193
    # For row=8192 (last row), index = 8193*8193 = 67,117,649 which is one beyond allocation
    # This single out-of-bounds write might still be within padding.
    # Let's cause MANY out-of-bounds writes by making N just above a multiple of 32
    # so that many threads have col=N.
    N = 33  # Actually go back to small N but verify exact output values
    # We'll compute reference CPU result and compare bit-exact
    A = torch.ones(N, N, device=device, dtype=torch.float32)
    B = torch.eye(N, device=device, dtype=torch.float32) * 2.0
    # With these inputs, C should be exactly 2*A (since B is 2*identity)
    # The mutant may write incorrect values at (row, N) positions
    # We'll run both original and mutant and compare full output
    return [A, B]

```

**输出差异**: `max_diff=2.000000e+00, mean_diff=7.346189e-03, ref_range=[2.0000e+00,2.0000e+00], mut_range=[0.0000e+00,2.0000e+00]`

### Part 4: 大模型诊断流程

共 2 轮迭代分析：

#### Round 1 (未杀死)

**LLM 存活原因分析：**

> The mutant survived because all stress tests used input dimensions N=2048, which is a multiple of the TILE_SIZE (32). When N is a multiple of TILE_SIZE, the grid launching mechanism ensures that the maximum column index computed by any thread is N-1. Thus, the condition `col <= N` is always true (since col ≤ N-1 < N) for all launched threads, making the mutant behavior identical to the original (`col < N`). No value-distribution changes could expose the mutation because the condition's branch is never taken differently when N is a multiple of TILE_SIZE.

**LLM 杀死策略：**

> Use an input dimension N that is NOT a multiple of TILE_SIZE (32). When N is not a multiple of 32, the grid launches extra threads with column indices up to ceil(N/32)*32 - 1, which includes indices equal to N. For these threads, the mutant's condition `col <= N` will be true (allowing out-of-bounds writes at linear index `row * N + N`), while the original's `col < N` correctly filters them out. This will cause either a CUDA memory error (crash) or silent corruption of adjacent memory, distinguishing the mutant.

**LLM 建议的输入代码：**

```python
import torch

def generate_inputs(device):
    N = 33  # Not a multiple of TILE_SIZE (32)
    torch.manual_seed(123)  # For reproducibility
    A = torch.randn(N, N, device=device, dtype=torch.float32)
    B = torch.randn(N, N, device=device, dtype=torch.float32)
    return [A, B]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `True`

#### Round 2 (**KILLED**)

**LLM 存活原因分析：**

> The mutant survives because even when N is not a multiple of TILE_SIZE, the out-of-bounds write at index `row * N + N` (when col=N) falls within the same memory page as the valid allocation due to PyTorch's memory alignment and CUDA's memory protection granularity. This doesn't cause a crash or detectable corruption because: (1) The GPU memory allocator may pad allocations beyond requested size, (2) The single extra write may not cross page boundaries, and (3) The stress tests may not verify exact output values beyond checking for crashes. The previous attempt with N=33 failed because the write at index 33 was still within padded memory bounds.

**LLM 杀死策略：**

> Force a larger out-of-bounds write that clearly crosses page boundaries by using an input size where the extra thread writes far beyond the allocated memory region. Use N = 2047 (not multiple of 32) so that thread with col=2047 writes at index `row*2047 + 2047`. For row=2046, this writes to index ~4.19M, but the allocation is only 2047² ≈ 4.19M elements. However, due to padding, this may still be safe. Instead, we need to cause a write so far out of bounds that it clearly accesses unallocated memory. We'll achieve this by exploiting that the kernel uses 32-bit indexing, and we can cause an index overflow/wrap-around by making N large enough that `row * N + col` exceeds 32-bit range when col=N.

**LLM 建议的输入代码：**

```python
import torch
def generate_inputs(device):
    # N chosen to cause integer overflow in index calculation when col=N
    # N = 46341 -> N*N = 2,147,488,281 > 2^31-1
    N = 46341
    torch.manual_seed(123)
    # Use smaller matrices to fit memory, but the kernel will still compute with N
    # Actually, we need to allocate NxN matrices which is ~8.6GB for float32
    # This is too large for most GPUs. Let's instead use a smaller N that still causes
    # significant out-of-bounds writes beyond the allocation.
    # Alternative: Use N = 8193 (not multiple of 32) which creates many out-of-bounds writes
    # The allocation is 8193*8193 = 67.1M elements = 268MB
    # Threads with col=8193 write to indices row*8193 + 8193 = (row+1)*8193
    # For row=8192 (last row), index = 8193*8193 = 67,117,649 which is one beyond allocation
    # This single out-of-bounds write might still be within padding.
    # Let's cause MANY out-of-bounds writes by making N just above a multiple of 32
    # so that many threads have col=N.
    N = 33  # Actually go back to small N but verify exact output values
    # We'll compute reference CPU result and compare bit-exact
    A = torch.ones(N, N, device=device, dtype=torch.float32)
    B = torch.eye(N, device=device, dtype=torch.float32) * 2.0
    # With these inputs, C should be exactly 2*A (since B is 2*identity)
    # The mutant may write incorrect values at (row, N) positions
    # We'll run both original and mutant and compare full output
    return [A, B]

```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=2.000000e+00, mean_diff=7.346189e-03, ref_range=[2.0000e+00,2.0000e+00], mut_range=[0.0000e+00,2.0000e+00]`

#### 测试构造规则

- **规则名**: `index_overflow_due_to_extended_bound`
- **描述**: Targets kernels where relational operator mutations extend loop bounds beyond intended limits. Creates inputs where the index calculation for the extended bound causes integer overflow in 32-bit arithmetic, or produces values far beyond allocated memory bounds, making out-of-bounds writes detectable.
- **适用算子**: relop_replace, loop_boundary, arithmetic_replace

```python
def policy(shape, dtype, device, rng):
    import torch
    import math
    
    # For 2D matrix kernels, target square dimensions near sqrt(2^31) to cause overflow
    # when index = row*N + N with row near N-1
    if len(shape) == 2 and shape[0] == shape[1]:
        N = shape[0]
        # Choose dimension that ensures index overflow when computing (N-1)*N + N = N*N
        # Smallest N where N*N > 2^31-1 is 46341
        overflow_N = 46341
        
        # If requested N is smaller, still use a non-multiple of tile size (32)
        # to ensure extended bounds produce out-of-bounds writes
        if N < overflow_N:
            # Adjust to be non-multiple of 32 to get extra threads
            adjusted_N = N
            if adjusted_N % 32 == 0:
                adjusted_N += 1
            
            # Create matrices with predictable pattern for value comparison
            A = torch.ones((adjusted_N, adjusted_N), device=device, dtype=dtype)
            B = torch.eye(adjusted_N, device=device, dtype=dtype) * 2.0
            return [A, B]
        else:
            # For large N near overflow boundary, use identity-like pattern
            # but be mindful of memory constraints
            A = torch.ones((N, N), device=device, dtype=dtype)
            B = torch.eye(N, device=device, dtype=dtype)
            return [A, B]
    
    # For 1D kernels, target size near 2^31 for overflow in index + 1 calculations
    elif len(shape) == 1:
        N = shape[0]
        # Create input with pattern for easy validation
        data = torch.arange(N, device=device, dtype=dtype) / N
        return [data]
    
    # Default: return random data with seed for reproducibility
    torch.manual_seed(123)
    return [torch.randn(shape, device=device, dtype=dtype)]
```


---

## 5. `L1_P23__init_modify__0`

- **Kernel**: `L1_P23`
- **算子**: `init_modify` (Category C)
- **描述**: Weaken min/max reduction initializers (inf / -inf) @ L34
- **变异行**: Line 34, 原始片段 `-INFINITY`, 节点类型 `init:cuda_neginf:-1e10`
- **杀死轮次**: Round 1

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
    float thread_max = -1e10f;
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

**杀死输入代码 (Round 1)：**

```python
import torch
import math

def generate_inputs(device):
    batch_size, num_features = 16, 16384
    # Initialize with normal values (e.g., zeros)
    x = torch.zeros(batch_size, num_features, dtype=torch.float32, device=device)
    # Set first row to -2e10 (well below -1e10)
    x[0, :] = -2e10
    return [x]
```

**输出差异**: `max_diff=nan, mean_diff=nan, ref_range=[6.1035e-05,6.1035e-05], mut_range=[nan,nan]`

### Part 4: 大模型诊断流程

共 1 轮迭代分析：

#### Round 1 (**KILLED**)

**LLM 存活原因分析：**

> The mutant replaces -INFINITY with -1e10f as the initial value for max reduction. This only produces incorrect behavior when ALL elements in a row are less than -1e10 (i.e., < -1.0 × 10¹⁰). The stress tests failed to kill it because they never generated rows where every single element was below -1e10. Even 'all_negative' tests likely used values in ranges like [-1, 0) or [-1000, 0), which are > -1e10. The reduction's parallel nature means a single element ≥ -1e10 would propagate correctly through fmaxf, making the mutant equivalent to the original.

**LLM 杀死策略：**

> Create a row where ALL elements are < -1e10 (e.g., -2e10). This forces the mutant's initial -1e10 to remain as the 'maximum' since no element exceeds it, while the original correctly identifies the true maximum (which is < -1e10 but > -INF). This difference propagates through the subsequent softmax computations.

**LLM 建议的输入代码：**

```python
import torch
import math

def generate_inputs(device):
    batch_size, num_features = 16, 16384
    # Initialize with normal values (e.g., zeros)
    x = torch.zeros(batch_size, num_features, dtype=torch.float32, device=device)
    # Set first row to -2e10 (well below -1e10)
    x[0, :] = -2e10
    return [x]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=nan, mean_diff=nan, ref_range=[6.1035e-05,6.1035e-05], mut_range=[nan,nan]`

#### 测试构造规则

- **规则名**: `reduction_init_identity_extremes`
- **描述**: Targets reduction operations with mutated identity elements by creating inputs where entire reduction sets (rows/columns) contain only values outside the mutant's valid range. For max reductions with finite negative identity, sets all elements in at least one reduction dimension to values below the mutant threshold. For min reductions with finite positive identity, sets all elements above the threshold.
- **适用算子**: init_modify, reduce_identity, reduction_initialization

```python
import torch
import numpy as np

def policy(shape, dtype, device, rng):
    """Generate tensor with at least one full reduction dimension containing extreme values."""
    # Use numpy RNG for reproducibility
    np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
    
    # Create base tensor with normal values
    tensor = torch.zeros(shape, dtype=dtype, device=device)
    
    if len(shape) >= 2:
        # Choose a reduction dimension (typically last dimension for row-wise reductions)
        # but could be any dimension; here we use the last dimension for common patterns
        reduce_dim = -1
        
        # Choose at least one slice in the other dimension to be extreme
        # For 2D: set entire rows to extreme values
        slice_dim = 0 if len(shape) == 2 else np_rng.choice(len(shape) - 1)
        
        # Number of slices to make extreme (1 to 25% of slices)
        num_slices = max(1, int(shape[slice_dim] * np_rng.uniform(0.01, 0.25)))
        slice_indices = np_rng.choice(shape[slice_dim], size=num_slices, replace=False)
        
        # Determine extreme value based on dtype
        if dtype in (torch.float32, torch.float64):
            # Values significantly below typical finite identity replacements
            extreme_value = -1e20 if dtype == torch.float64 else -1e10
        elif dtype == torch.float16:
            extreme_value = -60000.0  # Near fp16 limits
        else:
            # For integer types, use near-minimum values
            info = torch.iinfo(dtype) if dtype.is_integer else torch.finfo(dtype)
            extreme_value = info.min + 1 if hasattr(info, 'min') else -1e6
        
        # Apply extreme values to selected slices
        for idx in slice_indices:
            if len(shape) == 2:
                tensor[idx, :] = extreme_value
            else:
                # General n-dimensional case: set entire hyperplane
                idx_tuple = [slice(None)] * len(shape)
                idx_tuple[slice_dim] = idx
                tensor[tuple(idx_tuple)] = extreme_value
    
    return tensor
```


---

## 6. `L1_P28__arith_replace__9`

- **Kernel**: `L1_P28`
- **算子**: `arith_replace` (Category A)
- **描述**: Replace binary arithmetic operators (+→-, -→+, *→/, /→*) @ L38
- **变异行**: Line 38, 原始片段 `/`, 节点类型 `cuda_Div`
- **杀死轮次**: Round 1

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized HardSigmoid with vectorized loads
hardsigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Vectorized kernel using float4 for better memory bandwidth utilization
__global__ void hardsigmoid_kernel_vectorized(const float* __restrict__ input, 
                                              float* __restrict__ output, 
                                              int size) {
    const int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
    
    if (idx + 3 < size) {
        // Load 4 elements at once using float4
        float4 in_vec = *reinterpret_cast<const float4*>(&input[idx]);
        float4 out_vec;
        
        // Apply HardSigmoid to each element: max(0, min(1, (x + 3) / 6))
        const float one_sixth = 1.0f / 6.0f;
        const float half = 0.5f;
        
        out_vec.x = fmaxf(0.0f, fminf(1.0f, __fmaf_rd(in_vec.x, one_sixth, half)));
        out_vec.y = fmaxf(0.0f, fminf(1.0f, __fmaf_rd(in_vec.y, one_sixth, half)));
        out_vec.z = fmaxf(0.0f, fminf(1.0f, __fmaf_rd(in_vec.z, one_sixth, half)));
        out_vec.w = fmaxf(0.0f, fminf(1.0f, __fmaf_rd(in_vec.w, one_sixth, half)));
        
        // Store 4 elements at once
        *reinterpret_cast<float4*>(&output[idx]) = out_vec;
    } else {
        // Handle remaining elements (less than 4)
        for (int i = 0; i < 4; i++) {
            int elem_idx = idx + i;
            if (elem_idx < size) {
                float val = input[elem_idx];
                output[elem_idx] = fmaxf(0.0f, fminf(1.0f, __fmaf_rd(val, 1.0f/6.0f, 0.5f)));
            }
        }
    }
}

// Fallback kernel for non-vectorized cases
__global__ void hardsigmoid_kernel_scalar(const float* __restrict__ input, 
                                          float* __restrict__ output, 
                                          int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        float val = input[i];
        output[i] = fmaxf(0.0f, fminf(1.0f, __fmaf_rd(val, 1.0f/6.0f, 0.5f)));
    }
}

torch::Tensor hardsigmoid_cuda_optimized(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    // Use optimal configuration based on tensor size
    if (size >= 1024) {
        // Use vectorized kernel for large tensors
        const int block_size = 256;  // Optimal for vectorized loads
        int num_blocks = (size + (block_size * 4) - 1) / (block_size * 4);
        
        // Ensure we have enough blocks for good occupancy
        num_blocks = min(max(num_blocks, 32), 1024);
        
        hardsigmoid_kernel_vectorized<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size
        );
    } else {
        // Use scalar kernel for small tensors
        const int block_size = 256;
        int num_blocks = (size + block_size - 1) / block_size;
        num_blocks = min(max(num_blocks, 1), 1024);
        
        hardsigmoid_kernel_scalar<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size
        );
    }
    
    return output;
}
"""

hardsigmoid_cpp_source = "torch::Tensor hardsigmoid_cuda_optimized(torch::Tensor input);"

# Compile the inline CUDA code with optimization flags
hardsigmoid_cuda_optimized = load_inline(
    name="hardsigmoid_optimized",
    cpp_sources=hardsigmoid_cpp_source,
    cuda_sources=hardsigmoid_source,
    functions=["hardsigmoid_cuda_optimized"],
    verbose=False,
    extra_cflags=["-O3", "-march=native"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas=-v"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs HardSigmoid activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hardsigmoid_op = hardsigmoid_cuda_optimized
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardSigmoid activation to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardSigmoid applied, same shape as input.
        """
        if x.is_cuda and x.dtype == torch.float32:
            return self.hardsigmoid_op.hardsigmoid_cuda_optimized(x)
        else:
            # Fallback to PyTorch implementation for CPU tensors or other dtypes
            return torch.nn.functional.hardsigmoid(x)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized HardSigmoid with vectorized loads
hardsigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Vectorized kernel using float4 for better memory bandwidth utilization
__global__ void hardsigmoid_kernel_vectorized(const float* __restrict__ input, 
                                              float* __restrict__ output, 
                                              int size) {
    const int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
    
    if (idx + 3 < size) {
        // Load 4 elements at once using float4
        float4 in_vec = *reinterpret_cast<const float4*>(&input[idx]);
        float4 out_vec;
        
        // Apply HardSigmoid to each element: max(0, min(1, (x + 3) / 6))
        const float one_sixth = 1.0f / 6.0f;
        const float half = 0.5f;
        
        out_vec.x = fmaxf(0.0f, fminf(1.0f, __fmaf_rd(in_vec.x, one_sixth, half)));
        out_vec.y = fmaxf(0.0f, fminf(1.0f, __fmaf_rd(in_vec.y, one_sixth, half)));
        out_vec.z = fmaxf(0.0f, fminf(1.0f, __fmaf_rd(in_vec.z, one_sixth, half)));
        out_vec.w = fmaxf(0.0f, fminf(1.0f, __fmaf_rd(in_vec.w, one_sixth, half)));
        
        // Store 4 elements at once
        *reinterpret_cast<float4*>(&output[idx]) = out_vec;
    } else {
        // Handle remaining elements (less than 4)
        for (int i = 0; i < 4; i++) {
            int elem_idx = idx + i;
            if (elem_idx < size) {
                float val = input[elem_idx];
                output[elem_idx] = fmaxf(0.0f, fminf(1.0f, __fmaf_rd(val, 1.0f*6.0f, 0.5f)));
            }
        }
    }
}

// Fallback kernel for non-vectorized cases
__global__ void hardsigmoid_kernel_scalar(const float* __restrict__ input, 
                                          float* __restrict__ output, 
                                          int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        float val = input[i];
        output[i] = fmaxf(0.0f, fminf(1.0f, __fmaf_rd(val, 1.0f/6.0f, 0.5f)));
    }
}

torch::Tensor hardsigmoid_cuda_optimized(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    // Use optimal configuration based on tensor size
    if (size >= 1024) {
        // Use vectorized kernel for large tensors
        const int block_size = 256;  // Optimal for vectorized loads
        int num_blocks = (size + (block_size * 4) - 1) / (block_size * 4);
        
        // Ensure we have enough blocks for good occupancy
        num_blocks = min(max(num_blocks, 32), 1024);
        
        hardsigmoid_kernel_vectorized<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size
        );
    } else {
        // Use scalar kernel for small tensors
        const int block_size = 256;
        int num_blocks = (size + block_size - 1) / block_size;
        num_blocks = min(max(num_blocks, 1), 1024);
        
        hardsigmoid_kernel_scalar<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size
        );
    }
    
    return output;
}
"""

hardsigmoid_cpp_source = "torch::Tensor hardsigmoid_cuda_optimized(torch::Tensor input);"

# Compile the inline CUDA code with optimization flags
hardsigmoid_cuda_optimized = load_inline(
    name="hardsigmoid_optimized",
    cpp_sources=hardsigmoid_cpp_source,
    cuda_sources=hardsigmoid_source,
    functions=["hardsigmoid_cuda_optimized"],
    verbose=False,
    extra_cflags=["-O3", "-march=native"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas=-v"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs HardSigmoid activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hardsigmoid_op = hardsigmoid_cuda_optimized
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardSigmoid activation to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardSigmoid applied, same shape as input.
        """
        if x.is_cuda and x.dtype == torch.float32:
            return self.hardsigmoid_op.hardsigmoid_cuda_optimized(x)
        else:
            # Fallback to PyTorch implementation for CPU tensors or other dtypes
            return torch.nn.functional.hardsigmoid(x)
```

### Part 3: 杀死该变异体的输入

**杀死输入代码 (Round 1)：**

```python
import torch
def generate_inputs(device):
    # Use a size NOT divisible by 4 to trigger scalar tail
    # Original was 262144 (divisible by 4). Use 262143 instead.
    size_not_divisible_by_4 = 16 * 16384 - 1  # = 262143
    # Create tensor with all 1.0 values to maximize difference
    # Original: __fmaf_rd(1.0, 1/6, 0.5) = 0.666667 -> clamp -> 0.666667
    # Mutant:   __fmaf_rd(1.0, 6, 0.5) = 6.5 -> clamp -> 1.0
    x = torch.full((1, size_not_divisible_by_4), 1.0, dtype=torch.float32, device=device)
    # Reshape to 2D but with columns = size_not_divisible_by_4
    x = x.view(1, -1)  # shape [1, 262143]
    return [x]
```

**输出差异**: `max_diff=3.333333e-01, mean_diff=3.874316e-06, ref_range=[6.6667e-01,6.6667e-01], mut_range=[6.6667e-01,1.0000e+00]`

### Part 4: 大模型诊断流程

共 1 轮迭代分析：

#### Round 1 (**KILLED**)

**LLM 存活原因分析：**

> The mutant survived because the mutation only affects the scalar tail handling code (lines 34-40), which executes only when the index `idx + 3 >= size`. The stress tests used input shape [16, 16384] = 262144 elements, which is divisible by 4 (262144/4 = 65536). With typical kernel launch configurations (blockDim.x * gridDim.x divisible by 4), all threads process full float4 vectors, and the scalar tail path is NEVER executed. The mutation changes `1.0f/6.0f` (0.166667) to `1.0f*6.0f` (6.0) in the __fmaf_rd operation, but this dead code is never reached.

**LLM 杀死策略：**

> Force execution of the scalar tail by using an input size NOT divisible by 4. The kernel processes elements in chunks of 4 (vectorized), with remaining elements (1-3) handled by the scalar tail loop. We need a size where at least one thread enters the `else` branch (line 32). The most effective approach is to use a size where the total number of threads launched covers the array but the last thread(s) have fewer than 4 elements remaining.

**LLM 建议的输入代码：**

```python
import torch
def generate_inputs(device):
    # Use a size NOT divisible by 4 to trigger scalar tail
    # Original was 262144 (divisible by 4). Use 262143 instead.
    size_not_divisible_by_4 = 16 * 16384 - 1  # = 262143
    # Create tensor with all 1.0 values to maximize difference
    # Original: __fmaf_rd(1.0, 1/6, 0.5) = 0.666667 -> clamp -> 0.666667
    # Mutant:   __fmaf_rd(1.0, 6, 0.5) = 6.5 -> clamp -> 1.0
    x = torch.full((1, size_not_divisible_by_4), 1.0, dtype=torch.float32, device=device)
    # Reshape to 2D but with columns = size_not_divisible_by_4
    x = x.view(1, -1)  # shape [1, 262143]
    return [x]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=3.333333e-01, mean_diff=3.874316e-06, ref_range=[6.6667e-01,6.6667e-01], mut_range=[6.6667e-01,1.0000e+00]`

#### 测试构造规则

- **规则名**: `non_divisible_by_vector_width`
- **描述**: Generates input tensors whose total element count is NOT divisible by the kernel's vector width (typically 4 for float4). This forces execution of scalar tail handling code paths that process remaining elements after vectorized chunks. This is crucial for testing mutations that only affect scalar fallback code.
- **适用算子**: arith_replace, relational_operator_replacement, abs_remove, neg_replace, math_replace, incdec_operator_replacement

```python
import torch
import random

def policy(shape, dtype, device, rng):
    # Choose a small adjustment to keep shape similar but break vector alignment
    adjustments = [1, 2, 3]
    adjustment = rng.choice(adjustments)
    
    # Calculate total elements and adjust to make non-divisible by 4
    total_elements = torch.prod(torch.tensor(shape)).item()
    
    # Create base tensor with random values
    x = torch.randn(shape, dtype=dtype, device=device, generator=rng)
    
    # For testing arithmetic mutations, values that produce noticeable differences work best
    # Use values that amplify arithmetic differences (like 1.0 for multiplicative ops)
    scaling_factor = 1.0
    x = x * 0.1 + scaling_factor  # Center around 1.0 but with some variation
    
    # Now adjust shape to break vector alignment
    # Strategy: reduce the last dimension slightly
    new_shape = list(shape)
    if len(new_shape) > 0:
        last_dim = new_shape[-1]
        if last_dim > adjustment:
            new_shape[-1] = last_dim - adjustment
        else:
            # If last dimension too small, adjust total by adding elements
            # Reshape to 1D, truncate, then reshape back
            flat_size = total_elements - adjustment
            x_flat = x.flatten()
            x = x_flat[:flat_size].view(*new_shape[:-1], -1)
    
    return x
```


---

## 7. `L1_P29__arith_replace__16`

- **Kernel**: `L1_P29`
- **算子**: `arith_replace` (Category A)
- **描述**: Replace binary arithmetic operators (+→-, -→+, *→/, /→*) @ L98
- **变异行**: Line 98, 原始片段 `/`, 节点类型 `cuda_Div`
- **杀死轮次**: Round 1

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized Softplus with vectorized loads
softplus_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename scalar_t, int VEC_SIZE>
struct VectorType;

template<>
struct VectorType<float, 4> {
    using type = float4;
};

template<>
struct VectorType<half, 8> {
    using type = uint4;  // For half, we use uint4 for 8 half values
};

template<typename scalar_t, int VEC_SIZE>
__device__ __forceinline__ scalar_t softplus_activation(scalar_t x, float beta) {
    float x_scaled = static_cast<float>(x) * beta;
    if (x_scaled > 20.0f) return static_cast<scalar_t>(x);
    if (x_scaled < -20.0f) return static_cast<scalar_t>(0.0f);
    return static_cast<scalar_t>(log1pf(expf(x_scaled)) / beta);
}

template<>
__device__ __forceinline__ half softplus_activation<half, 8>(half x, float beta) {
    float x_f = __half2float(x);
    float x_scaled = x_f * beta;
    if (x_scaled > 20.0f) return x;
    if (x_scaled < -20.0f) return __float2half(0.0f);
    return __float2half(log1pf(expf(x_scaled)) / beta);
}

template<typename scalar_t, int VEC_SIZE>
__global__ void softplus_vectorized_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t size,
    float beta) {
    
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;
    
    if (idx + VEC_SIZE <= size) {
        using VecType = typename VectorType<scalar_t, VEC_SIZE>::type;
        VecType vec_in = *reinterpret_cast<const VecType*>(&input[idx]);
        VecType vec_out;
        
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            scalar_t val;
            if constexpr (std::is_same<scalar_t, half>::value && VEC_SIZE == 8) {
                half* half_ptr = reinterpret_cast<half*>(&vec_in);
                val = half_ptr[i];
            } else {
                float* float_ptr = reinterpret_cast<float*>(&vec_in);
                val = static_cast<scalar_t>(float_ptr[i]);
            }
            
            scalar_t result = softplus_activation<scalar_t, VEC_SIZE>(val, beta);
            
            if constexpr (std::is_same<scalar_t, half>::value && VEC_SIZE == 8) {
                half* half_out = reinterpret_cast<half*>(&vec_out);
                half_out[i] = result;
            } else {
                float* float_out = reinterpret_cast<float*>(&vec_out);
                float_out[i] = static_cast<float>(result);
            }
        }
        
        *reinterpret_cast<VecType*>(&output[idx]) = vec_out;
    } else {
        // Handle remaining elements
        for (int i = 0; i < VEC_SIZE && idx + i < size; i++) {
            output[idx + i] = softplus_activation<scalar_t, VEC_SIZE>(input[idx + i], beta);
        }
    }
}

torch::Tensor softplus_cuda(torch::Tensor input, float beta) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    
    auto output = torch::empty_like(input);
    int64_t size = input.numel();
    
    // Choose optimal vector size based on dtype
    if (input.scalar_type() == torch::kFloat32) {
        const int VEC_SIZE = 4;
        const int block_size = 256;
        int grid_size = (size + block_size * VEC_SIZE - 1) / (block_size * VEC_SIZE);
        
        softplus_vectorized_kernel<float, VEC_SIZE><<<grid_size, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size,
            beta
        );
    } else if (input.scalar_type() == torch::kFloat16) {
        const int VEC_SIZE = 8;
        const int block_size = 256;
        int grid_size = (size + block_size * VEC_SIZE - 1) / (block_size * VEC_SIZE);
        
        softplus_vectorized_kernel<half, VEC_SIZE><<<grid_size, block_size>>>(
            reinterpret_cast<half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            size,
            beta
        );
    } else {
        AT_ERROR("Unsupported data type for softplus_cuda");
    }
    
    return output;
}
"""

softplus_cpp_source = """
torch::Tensor softplus_cuda(torch::Tensor input, float beta=1.0);
"""

# Compile the inline CUDA code
softplus_module = load_inline(
    name="softplus_module",
    cpp_sources=softplus_cpp_source,
    cuda_sources=softplus_source,
    functions=["softplus_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas=-v"],
    extra_cflags=["-O3"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Softplus activation using custom CUDA kernel.
    """
    def __init__(self, beta=1.0):
        super(ModelNew, self).__init__()
        self.beta = beta
        self.softplus_cuda = softplus_module.softplus_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
        if not x.is_contiguous():
            x = x.contiguous()
            
        return self.softplus_cuda(x, self.beta)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized Softplus with vectorized loads
softplus_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename scalar_t, int VEC_SIZE>
struct VectorType;

template<>
struct VectorType<float, 4> {
    using type = float4;
};

template<>
struct VectorType<half, 8> {
    using type = uint4;  // For half, we use uint4 for 8 half values
};

template<typename scalar_t, int VEC_SIZE>
__device__ __forceinline__ scalar_t softplus_activation(scalar_t x, float beta) {
    float x_scaled = static_cast<float>(x) * beta;
    if (x_scaled > 20.0f) return static_cast<scalar_t>(x);
    if (x_scaled < -20.0f) return static_cast<scalar_t>(0.0f);
    return static_cast<scalar_t>(log1pf(expf(x_scaled)) / beta);
}

template<>
__device__ __forceinline__ half softplus_activation<half, 8>(half x, float beta) {
    float x_f = __half2float(x);
    float x_scaled = x_f * beta;
    if (x_scaled > 20.0f) return x;
    if (x_scaled < -20.0f) return __float2half(0.0f);
    return __float2half(log1pf(expf(x_scaled)) / beta);
}

template<typename scalar_t, int VEC_SIZE>
__global__ void softplus_vectorized_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t size,
    float beta) {
    
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;
    
    if (idx + VEC_SIZE <= size) {
        using VecType = typename VectorType<scalar_t, VEC_SIZE>::type;
        VecType vec_in = *reinterpret_cast<const VecType*>(&input[idx]);
        VecType vec_out;
        
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            scalar_t val;
            if constexpr (std::is_same<scalar_t, half>::value && VEC_SIZE == 8) {
                half* half_ptr = reinterpret_cast<half*>(&vec_in);
                val = half_ptr[i];
            } else {
                float* float_ptr = reinterpret_cast<float*>(&vec_in);
                val = static_cast<scalar_t>(float_ptr[i]);
            }
            
            scalar_t result = softplus_activation<scalar_t, VEC_SIZE>(val, beta);
            
            if constexpr (std::is_same<scalar_t, half>::value && VEC_SIZE == 8) {
                half* half_out = reinterpret_cast<half*>(&vec_out);
                half_out[i] = result;
            } else {
                float* float_out = reinterpret_cast<float*>(&vec_out);
                float_out[i] = static_cast<float>(result);
            }
        }
        
        *reinterpret_cast<VecType*>(&output[idx]) = vec_out;
    } else {
        // Handle remaining elements
        for (int i = 0; i < VEC_SIZE && idx + i < size; i++) {
            output[idx + i] = softplus_activation<scalar_t, VEC_SIZE>(input[idx + i], beta);
        }
    }
}

torch::Tensor softplus_cuda(torch::Tensor input, float beta) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    
    auto output = torch::empty_like(input);
    int64_t size = input.numel();
    
    // Choose optimal vector size based on dtype
    if (input.scalar_type() == torch::kFloat32) {
        const int VEC_SIZE = 4;
        const int block_size = 256;
        int grid_size = (size + block_size * VEC_SIZE - 1) * (block_size * VEC_SIZE);
        
        softplus_vectorized_kernel<float, VEC_SIZE><<<grid_size, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size,
            beta
        );
    } else if (input.scalar_type() == torch::kFloat16) {
        const int VEC_SIZE = 8;
        const int block_size = 256;
        int grid_size = (size + block_size * VEC_SIZE - 1) / (block_size * VEC_SIZE);
        
        softplus_vectorized_kernel<half, VEC_SIZE><<<grid_size, block_size>>>(
            reinterpret_cast<half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            size,
            beta
        );
    } else {
        AT_ERROR("Unsupported data type for softplus_cuda");
    }
    
    return output;
}
"""

softplus_cpp_source = """
torch::Tensor softplus_cuda(torch::Tensor input, float beta=1.0);
"""

# Compile the inline CUDA code
softplus_module = load_inline(
    name="softplus_module",
    cpp_sources=softplus_cpp_source,
    cuda_sources=softplus_source,
    functions=["softplus_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas=-v"],
    extra_cflags=["-O3"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Softplus activation using custom CUDA kernel.
    """
    def __init__(self, beta=1.0):
        super(ModelNew, self).__init__()
        self.beta = beta
        self.softplus_cuda = softplus_module.softplus_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
        if not x.is_contiguous():
            x = x.contiguous()
            
        return self.softplus_cuda(x, self.beta)
```

### Part 3: 杀死该变异体的输入

**杀死输入代码 (Round 1)：**

```python
import torch
def generate_inputs(device):
    # Target just above overflow threshold: (2^31 // 1024) + 1 = 2,097,152 + 1
    overflow_size = 2097153  # Slightly above threshold
    # Create 1D tensor to exactly control size (total elements = overflow_size)
    # Kernel only cares about numel(), shape is irrelevant
    tensor = torch.randn(overflow_size, dtype=torch.float32, device=device)
    return [tensor]
```

**输出差异**: `max_diff=4.768372e-07, mean_diff=1.093743e-08, ref_range=[4.8026e-03,4.9415e+00], mut_range=[4.8026e-03,4.9415e+00]`

### Part 4: 大模型诊断流程

共 1 轮迭代分析：

#### Round 1 (**KILLED**)

**LLM 存活原因分析：**

> The mutation changes grid size calculation from division to multiplication, dramatically increasing grid_size (from ~size/1024 to size*1024). However, the kernel likely uses a grid-stride loop with bounds checking or modulo arithmetic, making it robust to oversized grid launches. The kernel's internal index calculation (like `idx = blockIdx.x * blockDim.x + threadIdx.x`) combined with size-based early returns ensures only valid threads process data. This means both original and mutant produce identical outputs regardless of input values because the kernel's internal logic compensates for incorrect grid sizing.

**LLM 杀死策略：**

> Use an input size that causes integer overflow in grid_size calculation. The mutant's multiplication `(size + block_size*VEC_SIZE - 1) * (block_size*VEC_SIZE)` with size=262144 produces ~269M, but if size is large enough to overflow 32-bit signed int (2^31-1), the grid_size becomes negative, causing CUDA launch failure or wrap-around to small positive number. The original's division cannot overflow. Target size > ~2M elements to trigger overflow.

**LLM 建议的输入代码：**

```python
import torch
def generate_inputs(device):
    # Target just above overflow threshold: (2^31 // 1024) + 1 = 2,097,152 + 1
    overflow_size = 2097153  # Slightly above threshold
    # Create 1D tensor to exactly control size (total elements = overflow_size)
    # Kernel only cares about numel(), shape is irrelevant
    tensor = torch.randn(overflow_size, dtype=torch.float32, device=device)
    return [tensor]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=4.768372e-07, mean_diff=1.093743e-08, ref_range=[4.8026e-03,4.9415e+00], mut_range=[4.8026e-03,4.9415e+00]`

#### 测试构造规则

- **规则名**: `int_overflow_grid_calculation`
- **描述**: Targets kernels where grid size calculations use integer arithmetic that could overflow when operators are mutated (e.g., division→multiplication). Generates inputs where total elements slightly exceed (2^31 / typical_block_size) to cause 32-bit signed integer overflow in the mutant's grid calculation while keeping the original kernel's grid size valid.
- **适用算子**: arith_replace, const_replace, expr_replace

```python
def policy(shape, dtype, device, rng):
    import torch
    import math
    
    # Calculate total elements needed to overflow grid calculation
    # Assumes typical block size of 1024, adjust if specific kernel info available
    typical_block_size = 1024
    overflow_threshold = (2**31) // typical_block_size
    
    # Add small margin above threshold to ensure overflow
    target_elements = overflow_threshold + 1
    
    # Create tensor with target number of elements
    # If requested shape already has more elements, keep it (already problematic)
    # Otherwise create 1D tensor of required size
    shape_elements = math.prod(shape) if shape else 1
    if shape_elements < target_elements:
        # Create minimal 1D tensor meeting overflow requirement
        tensor = torch.randn(target_elements, dtype=dtype, device=device)
    else:
        # Use requested shape if it already has enough elements
        tensor = torch.randn(shape, dtype=dtype, device=device)
    
    return tensor
```


---

## 8. `L1_P29__arith_replace__19`

- **Kernel**: `L1_P29`
- **算子**: `arith_replace` (Category A)
- **描述**: Replace binary arithmetic operators (+→-, -→+, *→/, /→*) @ L109
- **变异行**: Line 109, 原始片段 `*`, 节点类型 `cuda_Mult`
- **杀死轮次**: Round 2

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized Softplus with vectorized loads
softplus_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename scalar_t, int VEC_SIZE>
struct VectorType;

template<>
struct VectorType<float, 4> {
    using type = float4;
};

template<>
struct VectorType<half, 8> {
    using type = uint4;  // For half, we use uint4 for 8 half values
};

template<typename scalar_t, int VEC_SIZE>
__device__ __forceinline__ scalar_t softplus_activation(scalar_t x, float beta) {
    float x_scaled = static_cast<float>(x) * beta;
    if (x_scaled > 20.0f) return static_cast<scalar_t>(x);
    if (x_scaled < -20.0f) return static_cast<scalar_t>(0.0f);
    return static_cast<scalar_t>(log1pf(expf(x_scaled)) / beta);
}

template<>
__device__ __forceinline__ half softplus_activation<half, 8>(half x, float beta) {
    float x_f = __half2float(x);
    float x_scaled = x_f * beta;
    if (x_scaled > 20.0f) return x;
    if (x_scaled < -20.0f) return __float2half(0.0f);
    return __float2half(log1pf(expf(x_scaled)) / beta);
}

template<typename scalar_t, int VEC_SIZE>
__global__ void softplus_vectorized_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t size,
    float beta) {
    
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;
    
    if (idx + VEC_SIZE <= size) {
        using VecType = typename VectorType<scalar_t, VEC_SIZE>::type;
        VecType vec_in = *reinterpret_cast<const VecType*>(&input[idx]);
        VecType vec_out;
        
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            scalar_t val;
            if constexpr (std::is_same<scalar_t, half>::value && VEC_SIZE == 8) {
                half* half_ptr = reinterpret_cast<half*>(&vec_in);
                val = half_ptr[i];
            } else {
                float* float_ptr = reinterpret_cast<float*>(&vec_in);
                val = static_cast<scalar_t>(float_ptr[i]);
            }
            
            scalar_t result = softplus_activation<scalar_t, VEC_SIZE>(val, beta);
            
            if constexpr (std::is_same<scalar_t, half>::value && VEC_SIZE == 8) {
                half* half_out = reinterpret_cast<half*>(&vec_out);
                half_out[i] = result;
            } else {
                float* float_out = reinterpret_cast<float*>(&vec_out);
                float_out[i] = static_cast<float>(result);
            }
        }
        
        *reinterpret_cast<VecType*>(&output[idx]) = vec_out;
    } else {
        // Handle remaining elements
        for (int i = 0; i < VEC_SIZE && idx + i < size; i++) {
            output[idx + i] = softplus_activation<scalar_t, VEC_SIZE>(input[idx + i], beta);
        }
    }
}

torch::Tensor softplus_cuda(torch::Tensor input, float beta) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    
    auto output = torch::empty_like(input);
    int64_t size = input.numel();
    
    // Choose optimal vector size based on dtype
    if (input.scalar_type() == torch::kFloat32) {
        const int VEC_SIZE = 4;
        const int block_size = 256;
        int grid_size = (size + block_size * VEC_SIZE - 1) / (block_size * VEC_SIZE);
        
        softplus_vectorized_kernel<float, VEC_SIZE><<<grid_size, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size,
            beta
        );
    } else if (input.scalar_type() == torch::kFloat16) {
        const int VEC_SIZE = 8;
        const int block_size = 256;
        int grid_size = (size + block_size * VEC_SIZE - 1) / (block_size * VEC_SIZE);
        
        softplus_vectorized_kernel<half, VEC_SIZE><<<grid_size, block_size>>>(
            reinterpret_cast<half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            size,
            beta
        );
    } else {
        AT_ERROR("Unsupported data type for softplus_cuda");
    }
    
    return output;
}
"""

softplus_cpp_source = """
torch::Tensor softplus_cuda(torch::Tensor input, float beta=1.0);
"""

# Compile the inline CUDA code
softplus_module = load_inline(
    name="softplus_module",
    cpp_sources=softplus_cpp_source,
    cuda_sources=softplus_source,
    functions=["softplus_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas=-v"],
    extra_cflags=["-O3"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Softplus activation using custom CUDA kernel.
    """
    def __init__(self, beta=1.0):
        super(ModelNew, self).__init__()
        self.beta = beta
        self.softplus_cuda = softplus_module.softplus_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
        if not x.is_contiguous():
            x = x.contiguous()
            
        return self.softplus_cuda(x, self.beta)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized Softplus with vectorized loads
softplus_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename scalar_t, int VEC_SIZE>
struct VectorType;

template<>
struct VectorType<float, 4> {
    using type = float4;
};

template<>
struct VectorType<half, 8> {
    using type = uint4;  // For half, we use uint4 for 8 half values
};

template<typename scalar_t, int VEC_SIZE>
__device__ __forceinline__ scalar_t softplus_activation(scalar_t x, float beta) {
    float x_scaled = static_cast<float>(x) * beta;
    if (x_scaled > 20.0f) return static_cast<scalar_t>(x);
    if (x_scaled < -20.0f) return static_cast<scalar_t>(0.0f);
    return static_cast<scalar_t>(log1pf(expf(x_scaled)) / beta);
}

template<>
__device__ __forceinline__ half softplus_activation<half, 8>(half x, float beta) {
    float x_f = __half2float(x);
    float x_scaled = x_f * beta;
    if (x_scaled > 20.0f) return x;
    if (x_scaled < -20.0f) return __float2half(0.0f);
    return __float2half(log1pf(expf(x_scaled)) / beta);
}

template<typename scalar_t, int VEC_SIZE>
__global__ void softplus_vectorized_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t size,
    float beta) {
    
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;
    
    if (idx + VEC_SIZE <= size) {
        using VecType = typename VectorType<scalar_t, VEC_SIZE>::type;
        VecType vec_in = *reinterpret_cast<const VecType*>(&input[idx]);
        VecType vec_out;
        
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            scalar_t val;
            if constexpr (std::is_same<scalar_t, half>::value && VEC_SIZE == 8) {
                half* half_ptr = reinterpret_cast<half*>(&vec_in);
                val = half_ptr[i];
            } else {
                float* float_ptr = reinterpret_cast<float*>(&vec_in);
                val = static_cast<scalar_t>(float_ptr[i]);
            }
            
            scalar_t result = softplus_activation<scalar_t, VEC_SIZE>(val, beta);
            
            if constexpr (std::is_same<scalar_t, half>::value && VEC_SIZE == 8) {
                half* half_out = reinterpret_cast<half*>(&vec_out);
                half_out[i] = result;
            } else {
                float* float_out = reinterpret_cast<float*>(&vec_out);
                float_out[i] = static_cast<float>(result);
            }
        }
        
        *reinterpret_cast<VecType*>(&output[idx]) = vec_out;
    } else {
        // Handle remaining elements
        for (int i = 0; i < VEC_SIZE && idx + i < size; i++) {
            output[idx + i] = softplus_activation<scalar_t, VEC_SIZE>(input[idx + i], beta);
        }
    }
}

torch::Tensor softplus_cuda(torch::Tensor input, float beta) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    
    auto output = torch::empty_like(input);
    int64_t size = input.numel();
    
    // Choose optimal vector size based on dtype
    if (input.scalar_type() == torch::kFloat32) {
        const int VEC_SIZE = 4;
        const int block_size = 256;
        int grid_size = (size + block_size * VEC_SIZE - 1) / (block_size * VEC_SIZE);
        
        softplus_vectorized_kernel<float, VEC_SIZE><<<grid_size, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size,
            beta
        );
    } else if (input.scalar_type() == torch::kFloat16) {
        const int VEC_SIZE = 8;
        const int block_size = 256;
        int grid_size = (size + block_size / VEC_SIZE - 1) / (block_size * VEC_SIZE);
        
        softplus_vectorized_kernel<half, VEC_SIZE><<<grid_size, block_size>>>(
            reinterpret_cast<half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            size,
            beta
        );
    } else {
        AT_ERROR("Unsupported data type for softplus_cuda");
    }
    
    return output;
}
"""

softplus_cpp_source = """
torch::Tensor softplus_cuda(torch::Tensor input, float beta=1.0);
"""

# Compile the inline CUDA code
softplus_module = load_inline(
    name="softplus_module",
    cpp_sources=softplus_cpp_source,
    cuda_sources=softplus_source,
    functions=["softplus_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas=-v"],
    extra_cflags=["-O3"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Softplus activation using custom CUDA kernel.
    """
    def __init__(self, beta=1.0):
        super(ModelNew, self).__init__()
        self.beta = beta
        self.softplus_cuda = softplus_module.softplus_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
        if not x.is_contiguous():
            x = x.contiguous()
            
        return self.softplus_cuda(x, self.beta)
```

### Part 3: 杀死该变异体的输入

**杀死输入代码 (Round 2)：**

```python
import torch

def generate_inputs(device):
    # Single-element half tensor; mutant grid size = (1 + 32 - 1) / 2048 = 0
    x = torch.tensor([1000.0], device=device, dtype=torch.float16)
    return [x]
```

**输出差异**: `max_diff=1.040000e+02, mean_diff=1.040000e+02, ref_range=[1.0000e+03,1.0000e+03], mut_range=[8.9600e+02,8.9600e+02]`

### Part 4: 大模型诊断流程

共 2 轮迭代分析：

#### Round 1 (未杀死)

**LLM 存活原因分析：**

> The mutation changes the grid size calculation for float16 from `(size + block_size * VEC_SIZE - 1)` to `(size + block_size / VEC_SIZE - 1)`. For the original input shape [16, 16384] with size=262144, both expressions evaluate to the same grid size when VEC_SIZE=8 and block_size=256. Since the denominator is `block_size * VEC_SIZE = 2048`, the original numerator adds 2047 while the mutant adds 31. However, because 262144 is exactly divisible by 2048, both `(262144 + 2047) // 2048 = 129` and `(262144 + 31) // 2048 = 129` due to integer division truncation. Thus, the kernel launches identical grid dimensions, producing identical outputs.

**LLM 杀死策略：**

> Use a tensor size where `size % (block_size * VEC_SIZE)` is non-zero and specifically where the remainder causes the mutant's grid size to be one less than the original. This happens when `size = N*2048 + r` with `1 ≤ r ≤ 2016`. At r=2017, both would round up again. Choose r=1 (size=2049) to maximize the difference: original grid=2 blocks, mutant grid=1 block. The mutant will under-launch blocks, leaving the last 1 element unprocessed in output.

**LLM 建议的输入代码：**

```python
import torch
import math

def generate_inputs(device):
    # Create float16 tensor with size 2049 elements
    # Shape [1, 2049] gives total size 2049
    size = 2049
    x = torch.randn(size, device=device, dtype=torch.float16).reshape(1, -1)
    # Set last element to a distinctive value for easy detection
    x[0, -1] = 1000.0  # Large value for clear softplus difference
    return [x]

```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `True`

#### Round 2 (**KILLED**)

**LLM 存活原因分析：**

> The mutant survived previous tests because the kernel likely uses a grid-stride loop (common in CUDA for robustness and performance). In such a design, each thread processes multiple vectors in a loop, so even if the grid size is smaller (due to mutation), the kernel still processes all elements, making the mutation functionally harmless. The previous test with size=2049 failed because the grid-stride loop still covered the entire array. The mutation alters the grid size calculation but does not affect correctness because the kernel compensates by having each thread handle more work.

**LLM 杀死策略：**

> Use a half tensor with size so small (1 element) that the mutant's grid size becomes zero. This causes the kernel launch to be skipped (0 blocks), leaving output uninitialized, while the original launches one block and computes softplus. Unlike the previous attempt (size=2049), this size is below the threshold where the mutant grid size rounds to zero, exploiting a different boundary condition.

**LLM 建议的输入代码：**

```python
import torch

def generate_inputs(device):
    # Single-element half tensor; mutant grid size = (1 + 32 - 1) / 2048 = 0
    x = torch.tensor([1000.0], device=device, dtype=torch.float16)
    return [x]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=1.040000e+02, mean_diff=1.040000e+02, ref_range=[1.0000e+03,1.0000e+03], mut_range=[8.9600e+02,8.9600e+02]`

#### 测试构造规则

- **规则名**: `minimum_data_volume_grid_stride`
- **描述**: Tests kernels with the smallest possible input size (1 element) to expose mutations in grid/block calculations. Particularly effective for grid-stride loop implementations where larger inputs might mask mutations by having threads process multiple elements. The minimum input size ensures any miscalculation in thread/block counts becomes immediately visible.
- **适用算子**: arith_replace, constant_replacement, conditionals_boundary, negate_conditionals

```python
def policy(shape, dtype, device, rng):
    # Create tensor with exactly 1 element regardless of requested shape
    # but preserve the rank/dimensions if important for kernel validation
    single_element_shape = (1,) * len(shape) if shape else (1,)
    
    # Use a non-zero, non-trivial value to ensure computation occurs
    # and produce detectable output difference
    value = 1000.0 if torch.is_floating_point(torch.tensor(0, dtype=dtype)) else 42
    
    return torch.full(single_element_shape, value, device=device, dtype=dtype)
```


---

## 9. `L1_P31__arith_replace__3`

- **Kernel**: `L1_P31`
- **算子**: `arith_replace` (Category A)
- **描述**: Replace binary arithmetic operators (+→-, -→+, *→/, /→*) @ L23
- **变异行**: Line 23, 原始片段 `+`, 节点类型 `cuda_Add`
- **杀死轮次**: Round 1

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Optimized ELU kernel with minimal branching and vectorization
elu_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>

template<typename scalar_t>
__device__ __forceinline__ scalar_t elu_activation(scalar_t x, scalar_t alpha) {
    return x >= scalar_t(0) ? x : alpha * (expf(x) - scalar_t(1));
}

template<typename scalar_t>
__global__ void elu_kernel_fast(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t alpha,
    int64_t size) {
    
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop for better occupancy
    for (int64_t i = tid; i < size; i += stride) {
        scalar_t val = input[i];
        output[i] = elu_activation<scalar_t>(val, alpha);
    }
}

template<typename scalar_t>
__global__ void elu_kernel_vectorized4(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t alpha,
    int64_t size) {
    
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t idx = tid * 4;
    
    if (idx + 3 < size) {
        // Load 4 elements
        scalar_t vals[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            vals[j] = input[idx + j];
        }
        
        // Compute ELU for all 4 elements
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            output[idx + j] = elu_activation<scalar_t>(vals[j], alpha);
        }
    } else {
        // Handle remaining elements
        for (int j = 0; j < 4; ++j) {
            int64_t elem_idx = idx + j;
            if (elem_idx < size) {
                scalar_t val = input[elem_idx];
                output[elem_idx] = elu_activation<scalar_t>(val, alpha);
            }
        }
    }
}

torch::Tensor elu_cuda(torch::Tensor input, float alpha) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    
    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);
    
    int64_t size = input_contig.numel();
    
    if (size == 0) {
        return output;
    }
    
    // Optimal block size for modern GPUs
    const int block_size = 256;
    const int max_blocks = 256;
    int grid_size = (size + block_size - 1) / block_size;
    grid_size = grid_size > max_blocks ? max_blocks : grid_size;
    
    // Use vectorized kernel for large tensors
    if (size >= 4096 && (size % 4) == 0) {
        int vectorized_grid = (size / 4 + block_size - 1) / block_size;
        vectorized_grid = vectorized_grid > max_blocks ? max_blocks : vectorized_grid;
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "elu_cuda", [&] {
            elu_kernel_vectorized4<scalar_t><<<vectorized_grid, block_size>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                static_cast<scalar_t>(alpha),
                size
            );
        });
    } else {
        // Use simple grid-stride kernel for other cases
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "elu_cuda", [&] {
            elu_kernel_fast<scalar_t><<<grid_size, block_size>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                static_cast<scalar_t>(alpha),
                size
            );
        });
    }
    
    return output;
}
"""

elu_cpp_source = """
torch::Tensor elu_cuda(torch::Tensor input, float alpha);
"""

# Compile with optimizations
elu_module = load_inline(
    name="elu_module",
    cpp_sources=elu_cpp_source,
    cuda_sources=elu_source,
    functions=["elu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs ELU activation using custom CUDA kernel.
    """
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return elu_module.elu_cuda(x, self.alpha)
        else:
            return F.elu(x, alpha=self.alpha)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Optimized ELU kernel with minimal branching and vectorization
elu_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>

template<typename scalar_t>
__device__ __forceinline__ scalar_t elu_activation(scalar_t x, scalar_t alpha) {
    return x >= scalar_t(0) ? x : alpha * (expf(x) - scalar_t(1));
}

template<typename scalar_t>
__global__ void elu_kernel_fast(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t alpha,
    int64_t size) {
    
    const int64_t tid = blockIdx.x * blockDim.x - threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop for better occupancy
    for (int64_t i = tid; i < size; i += stride) {
        scalar_t val = input[i];
        output[i] = elu_activation<scalar_t>(val, alpha);
    }
}

template<typename scalar_t>
__global__ void elu_kernel_vectorized4(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t alpha,
    int64_t size) {
    
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t idx = tid * 4;
    
    if (idx + 3 < size) {
        // Load 4 elements
        scalar_t vals[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            vals[j] = input[idx + j];
        }
        
        // Compute ELU for all 4 elements
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            output[idx + j] = elu_activation<scalar_t>(vals[j], alpha);
        }
    } else {
        // Handle remaining elements
        for (int j = 0; j < 4; ++j) {
            int64_t elem_idx = idx + j;
            if (elem_idx < size) {
                scalar_t val = input[elem_idx];
                output[elem_idx] = elu_activation<scalar_t>(val, alpha);
            }
        }
    }
}

torch::Tensor elu_cuda(torch::Tensor input, float alpha) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    
    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);
    
    int64_t size = input_contig.numel();
    
    if (size == 0) {
        return output;
    }
    
    // Optimal block size for modern GPUs
    const int block_size = 256;
    const int max_blocks = 256;
    int grid_size = (size + block_size - 1) / block_size;
    grid_size = grid_size > max_blocks ? max_blocks : grid_size;
    
    // Use vectorized kernel for large tensors
    if (size >= 4096 && (size % 4) == 0) {
        int vectorized_grid = (size / 4 + block_size - 1) / block_size;
        vectorized_grid = vectorized_grid > max_blocks ? max_blocks : vectorized_grid;
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "elu_cuda", [&] {
            elu_kernel_vectorized4<scalar_t><<<vectorized_grid, block_size>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                static_cast<scalar_t>(alpha),
                size
            );
        });
    } else {
        // Use simple grid-stride kernel for other cases
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "elu_cuda", [&] {
            elu_kernel_fast<scalar_t><<<grid_size, block_size>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                static_cast<scalar_t>(alpha),
                size
            );
        });
    }
    
    return output;
}
"""

elu_cpp_source = """
torch::Tensor elu_cuda(torch::Tensor input, float alpha);
"""

# Compile with optimizations
elu_module = load_inline(
    name="elu_module",
    cpp_sources=elu_cpp_source,
    cuda_sources=elu_source,
    functions=["elu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs ELU activation using custom CUDA kernel.
    """
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return elu_module.elu_cuda(x, self.alpha)
        else:
            return F.elu(x, alpha=self.alpha)
```

### Part 3: 杀死该变异体的输入

**杀死输入代码 (Round 1)：**

```python
import torch
import math

def generate_inputs(device):
    # Small size that is prime and not a multiple of common block sizes (256, 512, etc.)
    # This ensures the grid-stride loop behavior differs between original and mutant
    size = 17  # Prime number, ensures non-uniform coverage
    
    # Create alternating positive and negative values to exercise both branches of ELU
    input_tensor = torch.zeros(size, device=device, dtype=torch.float32)
    for i in range(size):
        input_tensor[i] = 1.0 if (i % 2 == 0) else -1.0
    
    # Reshape to match original input specification [16, 16384] -> we use [1, 17]
    # The kernel only cares about total size, not dimensions
    input_tensor = input_tensor.view(1, 17)
    return [input_tensor]
```

**输出差异**: `max_diff=1.264241e+00, mean_diff=5.949369e-01, ref_range=[-6.3212e-01,1.0000e+00], mut_range=[6.3212e-01,1.0000e+00]`

### Part 4: 大模型诊断流程

共 1 轮迭代分析：

#### Round 1 (**KILLED**)

**LLM 存活原因分析：**

> The mutant survived because the grid-stride loop's initial offset (tid) and stride (blockDim.x * gridDim.x) work together to eventually cover the same set of indices, just in a different permutation. When using the original `tid = blockIdx.x * blockDim.x + threadIdx.x`, threads start at consecutive offsets within each block. With the mutant `tid = blockIdx.x * blockDim.x - threadIdx.x`, threads start at decreasing offsets within each block, but the grid-stride loop (i += stride) still ensures all threads collectively iterate over the entire [0, size) range. Since the ELU operation is elementwise and independent, the output is identical regardless of which thread processes which element, as long as every element is processed exactly once. The stress tests used fixed input shape [16, 16384] = 262144 elements, which is likely a multiple of the stride, ensuring complete coverage.

**LLM 杀死策略：**

> The mutant can be killed by using an input size that is NOT a multiple of the stride, causing some elements to be processed multiple times or missed entirely due to the inverted thread mapping. Specifically, we need a size where the grid-stride loop with the mutant's negative starting offsets leads to either: 1) duplicate processing of some indices, 2) missing some indices, or 3) out-of-bounds access. This requires choosing a size that is not a multiple of (blockDim.x * gridDim.x) and is smaller than the maximum initial offset.

**LLM 建议的输入代码：**

```python
import torch
import math

def generate_inputs(device):
    # Small size that is prime and not a multiple of common block sizes (256, 512, etc.)
    # This ensures the grid-stride loop behavior differs between original and mutant
    size = 17  # Prime number, ensures non-uniform coverage
    
    # Create alternating positive and negative values to exercise both branches of ELU
    input_tensor = torch.zeros(size, device=device, dtype=torch.float32)
    for i in range(size):
        input_tensor[i] = 1.0 if (i % 2 == 0) else -1.0
    
    # Reshape to match original input specification [16, 16384] -> we use [1, 17]
    # The kernel only cares about total size, not dimensions
    input_tensor = input_tensor.view(1, 17)
    return [input_tensor]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=1.264241e+00, mean_diff=5.949369e-01, ref_range=[-6.3212e-01,1.0000e+00], mut_range=[6.3212e-01,1.0000e+00]`

#### 测试构造规则

- **规则名**: `small_prime_size_alternating_sign`
- **描述**: Generates tensors with small prime total sizes (e.g., 17, 19, 23) and alternating positive/negative values. This breaks stride/coverage assumptions in grid-stride loops, exposing indexing arithmetic mutants when the stride exceeds the size. The alternating pattern ensures any duplicate/missing element processing becomes observable in elementwise operations with different branches.
- **适用算子**: arith_replace, const_replace, index_replace, boundary_replace

```python
def policy(shape, dtype, device, rng):
    import torch
    import math
    # List of small primes that are not multiples of common block sizes (256, 512, 1024)
    small_primes = [17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    # Randomly select a prime size
    p = small_primes[rng.randint(len(small_primes))]
    # Create a 1D tensor of alternating 1.0 and -1.0 values to exercise different branches
    data = [1.0 if i % 2 == 0 else -1.0 for i in range(p)]
    tensor = torch.tensor(data, device=device, dtype=dtype)
    # Reshape to match the original rank but with the prime size in the last dimension
    new_shape = [1] * len(shape)  # Preserve rank
    new_shape[-1] = p             # Set last dimension to prime size
    return tensor.view(new_shape)
```


---

## 10. `L1_P31__launch_config_mutate__0`

- **Kernel**: `L1_P31`
- **算子**: `launch_config_mutate` (Category B)
- **描述**: Perturb grid/block sizing expressions (// BLOCK, triton.cdiv) by ±1 @ L83
- **变异行**: Line 83, 原始片段 `(size + block_size - 1) / block_size`, 节点类型 `-1`
- **杀死轮次**: Round 1

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Optimized ELU kernel with minimal branching and vectorization
elu_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>

template<typename scalar_t>
__device__ __forceinline__ scalar_t elu_activation(scalar_t x, scalar_t alpha) {
    return x >= scalar_t(0) ? x : alpha * (expf(x) - scalar_t(1));
}

template<typename scalar_t>
__global__ void elu_kernel_fast(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t alpha,
    int64_t size) {
    
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop for better occupancy
    for (int64_t i = tid; i < size; i += stride) {
        scalar_t val = input[i];
        output[i] = elu_activation<scalar_t>(val, alpha);
    }
}

template<typename scalar_t>
__global__ void elu_kernel_vectorized4(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t alpha,
    int64_t size) {
    
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t idx = tid * 4;
    
    if (idx + 3 < size) {
        // Load 4 elements
        scalar_t vals[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            vals[j] = input[idx + j];
        }
        
        // Compute ELU for all 4 elements
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            output[idx + j] = elu_activation<scalar_t>(vals[j], alpha);
        }
    } else {
        // Handle remaining elements
        for (int j = 0; j < 4; ++j) {
            int64_t elem_idx = idx + j;
            if (elem_idx < size) {
                scalar_t val = input[elem_idx];
                output[elem_idx] = elu_activation<scalar_t>(val, alpha);
            }
        }
    }
}

torch::Tensor elu_cuda(torch::Tensor input, float alpha) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    
    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);
    
    int64_t size = input_contig.numel();
    
    if (size == 0) {
        return output;
    }
    
    // Optimal block size for modern GPUs
    const int block_size = 256;
    const int max_blocks = 256;
    int grid_size = (size + block_size - 1) / block_size;
    grid_size = grid_size > max_blocks ? max_blocks : grid_size;
    
    // Use vectorized kernel for large tensors
    if (size >= 4096 && (size % 4) == 0) {
        int vectorized_grid = (size / 4 + block_size - 1) / block_size;
        vectorized_grid = vectorized_grid > max_blocks ? max_blocks : vectorized_grid;
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "elu_cuda", [&] {
            elu_kernel_vectorized4<scalar_t><<<vectorized_grid, block_size>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                static_cast<scalar_t>(alpha),
                size
            );
        });
    } else {
        // Use simple grid-stride kernel for other cases
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "elu_cuda", [&] {
            elu_kernel_fast<scalar_t><<<grid_size, block_size>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                static_cast<scalar_t>(alpha),
                size
            );
        });
    }
    
    return output;
}
"""

elu_cpp_source = """
torch::Tensor elu_cuda(torch::Tensor input, float alpha);
"""

# Compile with optimizations
elu_module = load_inline(
    name="elu_module",
    cpp_sources=elu_cpp_source,
    cuda_sources=elu_source,
    functions=["elu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs ELU activation using custom CUDA kernel.
    """
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return elu_module.elu_cuda(x, self.alpha)
        else:
            return F.elu(x, alpha=self.alpha)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Optimized ELU kernel with minimal branching and vectorization
elu_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>

template<typename scalar_t>
__device__ __forceinline__ scalar_t elu_activation(scalar_t x, scalar_t alpha) {
    return x >= scalar_t(0) ? x : alpha * (expf(x) - scalar_t(1));
}

template<typename scalar_t>
__global__ void elu_kernel_fast(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t alpha,
    int64_t size) {
    
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop for better occupancy
    for (int64_t i = tid; i < size; i += stride) {
        scalar_t val = input[i];
        output[i] = elu_activation<scalar_t>(val, alpha);
    }
}

template<typename scalar_t>
__global__ void elu_kernel_vectorized4(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t alpha,
    int64_t size) {
    
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t idx = tid * 4;
    
    if (idx + 3 < size) {
        // Load 4 elements
        scalar_t vals[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            vals[j] = input[idx + j];
        }
        
        // Compute ELU for all 4 elements
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            output[idx + j] = elu_activation<scalar_t>(vals[j], alpha);
        }
    } else {
        // Handle remaining elements
        for (int j = 0; j < 4; ++j) {
            int64_t elem_idx = idx + j;
            if (elem_idx < size) {
                scalar_t val = input[elem_idx];
                output[elem_idx] = elu_activation<scalar_t>(val, alpha);
            }
        }
    }
}

torch::Tensor elu_cuda(torch::Tensor input, float alpha) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    
    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);
    
    int64_t size = input_contig.numel();
    
    if (size == 0) {
        return output;
    }
    
    // Optimal block size for modern GPUs
    const int block_size = 256;
    const int max_blocks = 256;
    int grid_size = (size + block_size - 1) / block_size - 1;
    grid_size = grid_size > max_blocks ? max_blocks : grid_size;
    
    // Use vectorized kernel for large tensors
    if (size >= 4096 && (size % 4) == 0) {
        int vectorized_grid = (size / 4 + block_size - 1) / block_size;
        vectorized_grid = vectorized_grid > max_blocks ? max_blocks : vectorized_grid;
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "elu_cuda", [&] {
            elu_kernel_vectorized4<scalar_t><<<vectorized_grid, block_size>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                static_cast<scalar_t>(alpha),
                size
            );
        });
    } else {
        // Use simple grid-stride kernel for other cases
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "elu_cuda", [&] {
            elu_kernel_fast<scalar_t><<<grid_size, block_size>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                static_cast<scalar_t>(alpha),
                size
            );
        });
    }
    
    return output;
}
"""

elu_cpp_source = """
torch::Tensor elu_cuda(torch::Tensor input, float alpha);
"""

# Compile with optimizations
elu_module = load_inline(
    name="elu_module",
    cpp_sources=elu_cpp_source,
    cuda_sources=elu_source,
    functions=["elu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs ELU activation using custom CUDA kernel.
    """
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return elu_module.elu_cuda(x, self.alpha)
        else:
            return F.elu(x, alpha=self.alpha)
```

### Part 3: 杀死该变异体的输入

**杀死输入代码 (Round 1)：**

```python
import torch

def generate_inputs(device):
    # One-element tensor with a negative value to produce a distinct ELU output
    # Shape [1] ensures size=1, which avoids vectorization and makes grid_size=1 (original) vs 0 (mutant)
    return [torch.tensor([-10.0], device=device, dtype=torch.float32)]
```

**输出差异**: `max_diff=1.999909e+00, mean_diff=1.999909e+00, ref_range=[-9.9995e-01,-9.9995e-01], mut_range=[9.9995e-01,9.9995e-01]`

### Part 4: 大模型诊断流程

共 1 轮迭代分析：

#### Round 1 (**KILLED**)

**LLM 存活原因分析：**

> The mutant survived all stress tests because the original input shape [16, 16384] (size=262144) triggers the vectorized kernel path (size >= 4096 and divisible by 4). The mutated grid_size variable is only used for the non-vectorized kernel, so its value is irrelevant for this input. Additionally, even if the non-vectorized path were taken, both original and mutated grid_size calculations exceed max_blocks=256 and are clamped to 256, making them equal. Thus, the mutation does not affect execution for the tested input shape.

**LLM 杀死策略：**

> Use an input tensor with a size that does NOT satisfy the vectorization condition (size < 4096 or size % 4 != 0) and where the original grid_size is not clamped to max_blocks, so the mutation reduces the active grid size. The most effective case is when original grid_size = 1 (size ≤ 256), causing mutated grid_size = 0, leading to an empty kernel launch and uninitialized output.

**LLM 建议的输入代码：**

```python
import torch

def generate_inputs(device):
    # One-element tensor with a negative value to produce a distinct ELU output
    # Shape [1] ensures size=1, which avoids vectorization and makes grid_size=1 (original) vs 0 (mutant)
    return [torch.tensor([-10.0], device=device, dtype=torch.float32)]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=1.999909e+00, mean_diff=1.999909e+00, ref_range=[-9.9995e-01,-9.9995e-01], mut_range=[9.9995e-01,9.9995e-01]`

#### 测试构造规则

- **规则名**: `small_non_vectorized_inputs`
- **描述**: Generates small tensor inputs (size ≤ 256) that force execution through non-vectorized kernel paths. These inputs are particularly effective against launch configuration mutations because they produce small grid/block values where ±1 perturbations change the actual launch dimensions (unlike larger inputs that get clamped to hardware limits).
- **适用算子**: launch_config_mutate, thread_mutate, block_mutate

```python
def policy(shape, dtype, device, rng):
    # Ignore requested shape to force small non-vectorized inputs
    # Generate random small size (1-256) to avoid vectorization thresholds
    import torch
    size = rng.randint(1, 257)
    
    # Create tensor with distinct values (negative for activation functions,
    # random otherwise) to make output differences detectable
    if rng.random() < 0.5:
        # Use negative values that produce non-linear outputs for common activations
        data = -rng.uniform(1.0, 10.0) * torch.ones(size, dtype=dtype, device=device)
    else:
        # Random values in a range that avoids saturation
        data = torch.rand(size, dtype=dtype, device=device) * 2 - 1  # [-1, 1]
    
    # Reshape to 1D unless kernel requires specific shape
    return data.view(-1) if data.dim() == 1 else data
```


---

## 11. `L1_P37__index_replace__11`

- **Kernel**: `L1_P37`
- **算子**: `index_replace` (Category B)
- **描述**: Swap Triton program_id axis or CUDA thread/block dimension index (e.g. program_id(0)→(1), threadIdx.x→threadIdx.y) @ L21
- **变异行**: Line 21, 原始片段 `gridDim.x`, 节点类型 `cuda_dim|z`
- **杀死轮次**: Round 3

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
    int stride = blockDim.x * gridDim.z;
    
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

**杀死输入代码 (Round 3)：**

```python
import torch
import math

def generate_inputs(device):
    shape = (1, 1, 1, 1025)
    data = torch.zeros(shape, dtype=torch.float32, device=device)
    data[0, 0, 0, 1024] = 1000000.0  # 1e6
    return [data]
```

**输出差异**: `max_diff=2.928932e-01, mean_diff=2.857494e-04, ref_range=[0.0000e+00,1.0000e+00], mut_range=[0.0000e+00,7.0711e-01]`

### Part 4: 大模型诊断流程

共 3 轮迭代分析：

#### Round 1 (未杀死)

**LLM 存活原因分析：**

> The mutant survived because the kernel launch configuration likely uses a 1D grid where `gridDim.z` = 1, but the input shape (16,64,256,256) yields `gridDim.x` = 65536 (assuming blockDim.x=1024). With this configuration, both the original and mutant kernels compute the same norm for the tested value distributions. The original kernel's stride (`blockDim.x * gridDim.x`) equals the total number of threads, causing each thread to process one element. The mutant's stride (`blockDim.x * gridDim.z` = `blockDim.x`) causes each thread to process many elements, but due to the large input size and specific value patterns (e.g., uniform constants), the overlapping accesses in the mutant might still yield the same sum of squares. Additionally, the original kernel itself may be buggy for certain distributions, masking the difference.

**LLM 杀死策略：**

> Change the input shape so that `total_elements` is small and not a multiple of `blockDim.x`, causing the mutant's overlapping accesses to produce a different norm than the original. Specifically, use a total number of elements where `gridDim.x` > 1 (to make strides differ) and the input values are non-zero, leading to repeated summations in the mutant. A constant input (all ones) makes the expected norm easy to compute: the original should sum each element once, while the mutant will sum some elements multiple times, resulting in a larger norm.

**LLM 建议的输入代码：**

```python
import torch
import math

def generate_inputs(device):
    # Total elements = 1025, small enough to cause overlapping in mutant
    # Shape is 4D to match the kernel's expectation (though it only cares about total_elements)
    total_elements = 1025
    shape = (1, 1, 1, total_elements)  # 4D tensor
    # All ones: makes the norm difference easy to compute
    data = torch.ones(shape, dtype=torch.float32, device=device)
    return [data]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `True`

#### Round 2 (未杀死)

**LLM 存活原因分析：**

> The mutant survives because the kernel launch configuration uses a 1D grid where gridDim.z = 1 by default. When gridDim.z = 1, the mutation 'gridDim.x → gridDim.z' changes stride from blockDim.x * gridDim.x to blockDim.x * 1. However, for small inputs where gridDim.x = 1 (as likely occurred in previous tests), both expressions evaluate to blockDim.x, making the mutant semantically identical to the original. Even when gridDim.x > 1, the mutant may still compute the same Frobenius norm because the parallel reduction's correctness depends only on covering all elements without duplication, not on the specific stride pattern, as long as the loop visits each element exactly once. The previous test with 1025 elements likely launched with gridDim.x = 1, so stride remained unchanged.

**LLM 杀死策略：**

> Force a 3D grid launch where gridDim.z > 1 and gridDim.x ≠ gridDim.z, using an input shape that the framework maps to a 3D grid. This ensures the mutant's stride differs from the original, potentially causing incorrect element coverage or duplication. Specifically, we need total_elements large enough that the framework chooses gridDim.z > 1, and values non-zero to detect norm differences.

**LLM 建议的输入代码：**

```python
import torch
import math

def generate_inputs(device):
    # Use the exact shape from specification; large enough to potentially use 3D grid
    shape = (16, 64, 256, 256)
    # Random values to avoid symmetry that could hide errors
    data = torch.rand(shape, dtype=torch.float32, device=device) * 2.0 - 1.0  # values in [-1, 1]
    return [data]

```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `True`

#### Round 3 (**KILLED**)

**LLM 存活原因分析：**

> The mutant likely survived because previous inputs either had uniform values (all ones) where double-counting differences were small relative to total norm, or the launch configuration inadvertently made gridDim.x equal to gridDim.z (e.g., 1D grid with gridDim.z=1 and gridDim.x=1). In the latter case, the mutant's stride becomes blockDim.x, which still partitions the input correctly for small total elements when only one block is used. The framework may also use block sizes that minimize overlap or the reduction across blocks may use atomic operations that sum partial results, potentially masking the double-counting effect when values are symmetric.

**LLM 杀死策略：**

> Force a launch configuration with multiple blocks and non-uniform input values, ensuring that overlapping indices (which occur at multiples of blockDim.x) contain large magnitudes. This amplifies the double-counting error in the mutant. Choose total elements = 1025 to guarantee at least two blocks for any block size ≤1024, and set only the last element (index 1024) to an extreme value (1e6). The original will sum this value once, while the mutant will sum it multiple times across blocks, producing a norm difference of several orders of magnitude.

**LLM 建议的输入代码：**

```python
import torch
import math

def generate_inputs(device):
    shape = (1, 1, 1, 1025)
    data = torch.zeros(shape, dtype=torch.float32, device=device)
    data[0, 0, 0, 1024] = 1000000.0  # 1e6
    return [data]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=2.928932e-01, mean_diff=2.857494e-04, ref_range=[0.0000e+00,1.0000e+00], mut_range=[0.0000e+00,7.0711e-01]`

#### 测试构造规则

- **规则名**: `non_uniform_block_boundary_amplification`
- **描述**: Generates inputs with a strategically placed large-magnitude value at an index that falls on a block boundary when using non-uniform grid dimensions. This amplifies indexing errors caused by thread/block dimension swaps, ensuring multiple blocks are launched and creating measurable discrepancies when overlapping or incorrect indices access the amplified value. The large value magnitude ensures detection even when reductions use atomic operations or symmetric values.
- **适用算子**: index_replace, grid_stride_loop_index_error, dimension_swap

```python
def policy(shape, dtype, device, rng):
    import torch
    import math
    
    # Create tensor with all zeros
    tensor = torch.zeros(shape, dtype=dtype, device=device)
    
    # Place a large value at a strategic index likely to be on a block boundary
    # Choose the last element along the innermost dimension for maximum boundary effect
    idx = [0] * len(shape)
    idx[-1] = shape[-1] - 1  # Last element of innermost dimension
    
    # Select amplification value based on dtype to avoid overflow
    if dtype in [torch.float32, torch.float64]:
        amp_value = 1000000.0  # 1e6
    elif dtype in [torch.float16, torch.bfloat16]:
        amp_value = 1000.0
    elif dtype.is_floating_point:
        amp_value = 1000000.0
    else:
        # For integer types, use maximum value that won't overflow in common operations
        amp_value = min(1000000, torch.iinfo(dtype).max // 2)
    
    tensor[tuple(idx)] = amp_value
    
    # Ensure tensor requires at least 2 blocks in typical configurations
    # by making innermost dimension > typical block size (1024)
    if shape[-1] <= 1024:
        # If shape too small, create new tensor with expanded innermost dimension
        new_shape = list(shape)
        new_shape[-1] = 1025  # Force >1024 to ensure multiple blocks
        tensor = torch.zeros(new_shape, dtype=dtype, device=device)
        tensor[tuple(idx)] = amp_value
    
    return tensor
```


---

## 12. `L1_P39__arith_replace__7`

- **Kernel**: `L1_P39`
- **算子**: `arith_replace` (Category A)
- **描述**: Replace binary arithmetic operators (+→-, -→+, *→/, /→*) @ L49
- **变异行**: Line 49, 原始片段 `+`, 节点类型 `cuda_Add`
- **杀死轮次**: Round 3

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized L2 normalization
l2_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__global__ void l2_norm_kernel(const T* __restrict__ input, T* __restrict__ output, 
                               int batch_size, int dim) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Use 256 threads per block (power of 2 for reduction)
    const int threads = 256;
    
    // Shared memory for reduction
    __shared__ float sdata[256];
    
    // Each thread processes multiple elements
    float thread_sum = 0.0f;
    
    // Process elements with stride equal to total threads
    for (int i = tid; i < dim; i += threads) {
        int idx = batch_idx * dim + i;
        float val = static_cast<float>(input[idx]);
        thread_sum += val * val;
    }
    
    // Parallel reduction
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Tree reduction
    for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Compute reciprocal of L2 norm (1 / sqrt(sum))
    float scale = 0.0f;
    if (tid == 0) {
        float norm = sqrtf(sdata[0] + 1e-8f);
        scale = 1.0f / norm;
        sdata[0] = scale;
    }
    __syncthreads();
    
    scale = sdata[0];
    
    // Normalize elements
    for (int i = tid; i < dim; i += threads) {
        int idx = batch_idx * dim + i;
        float val = static_cast<float>(input[idx]);
        output[idx] = static_cast<T>(val * scale);
    }
}

torch::Tensor l2_norm_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto dim = input.size(1);
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = batch_size;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_cuda", ([&] {
        l2_norm_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim
        );
    }));
    
    cudaDeviceSynchronize();
    return output;
}
"""

l2_norm_cpp_source = "torch::Tensor l2_norm_cuda(torch::Tensor input);"

# Compile the inline CUDA code for L2 normalization
l2_norm_module = load_inline(
    name="l2_norm",
    cpp_sources=l2_norm_cpp_source,
    cuda_sources=l2_norm_source,
    functions=["l2_norm_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized L2 normalization model with custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization using custom CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).
            
        Returns:
            torch.Tensor: L2 normalized tensor, same shape as input.
        """
        # Ensure tensor is on CUDA device
        if not x.is_cuda:
            x = x.cuda()
            
        return l2_norm_module.l2_norm_cuda(x)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized L2 normalization
l2_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__global__ void l2_norm_kernel(const T* __restrict__ input, T* __restrict__ output, 
                               int batch_size, int dim) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Use 256 threads per block (power of 2 for reduction)
    const int threads = 256;
    
    // Shared memory for reduction
    __shared__ float sdata[256];
    
    // Each thread processes multiple elements
    float thread_sum = 0.0f;
    
    // Process elements with stride equal to total threads
    for (int i = tid; i < dim; i += threads) {
        int idx = batch_idx * dim + i;
        float val = static_cast<float>(input[idx]);
        thread_sum += val * val;
    }
    
    // Parallel reduction
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Tree reduction
    for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Compute reciprocal of L2 norm (1 / sqrt(sum))
    float scale = 0.0f;
    if (tid == 0) {
        float norm = sqrtf(sdata[0] - 1e-8f);
        scale = 1.0f / norm;
        sdata[0] = scale;
    }
    __syncthreads();
    
    scale = sdata[0];
    
    // Normalize elements
    for (int i = tid; i < dim; i += threads) {
        int idx = batch_idx * dim + i;
        float val = static_cast<float>(input[idx]);
        output[idx] = static_cast<T>(val * scale);
    }
}

torch::Tensor l2_norm_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto dim = input.size(1);
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = batch_size;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_cuda", ([&] {
        l2_norm_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim
        );
    }));
    
    cudaDeviceSynchronize();
    return output;
}
"""

l2_norm_cpp_source = "torch::Tensor l2_norm_cuda(torch::Tensor input);"

# Compile the inline CUDA code for L2 normalization
l2_norm_module = load_inline(
    name="l2_norm",
    cpp_sources=l2_norm_cpp_source,
    cuda_sources=l2_norm_source,
    functions=["l2_norm_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized L2 normalization model with custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization using custom CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).
            
        Returns:
            torch.Tensor: L2 normalized tensor, same shape as input.
        """
        # Ensure tensor is on CUDA device
        if not x.is_cuda:
            x = x.cuda()
            
        return l2_norm_module.l2_norm_cuda(x)
```

### Part 3: 杀死该变异体的输入

**杀死输入代码 (Round 3)：**

```python
import torch
import math

def generate_inputs(device):
    batch_size = 16
    dim = 16384
    target_sum = 2e-8  # Exact value > 1e-8
    num_nonzero = 1024  # Spread across batch to keep each value small
    val_per_element = math.sqrt(target_sum / num_nonzero)  # ~1.41421356e-06
    x = torch.zeros(batch_size, dim, device=device, dtype=torch.float32)
    # Set first 1024 elements of each batch to the computed value
    x[:, :num_nonzero] = val_per_element
    return [x]
```

**输出差异**: `max_diff=1.294417e-02, mean_diff=8.090108e-04, ref_range=[0.0000e+00,3.1250e-02], mut_range=[0.0000e+00,4.4194e-02]`

### Part 4: 大模型诊断流程

共 3 轮迭代分析：

#### Round 1 (未杀死)

**LLM 存活原因分析：**

> The mutant survived because all stress tests that could potentially expose the difference (e.g., near-zero inputs where sum of squares < 1e-8) caused the original kernel to fail (REF_FAIL) with NaN/crashes, preventing comparison. The mutation changes sqrtf(sdata[0] + 1e-8f) to sqrtf(sdata[0] - 1e-8f), which only produces divergent behavior when sdata[0] < 1e-8. However, the original kernel itself appears numerically unstable for extreme inputs—likely due to underflow in the tree reduction or overflow during normalization—making it impossible to distinguish from the mutant in those regimes. For normal inputs where sdata[0] >> 1e-8, the epsilon perturbation is negligible, so both kernels produce identical outputs within floating-point tolerance.

**LLM 杀死策略：**

> Design an input where the sum of squares per batch is strictly between 0 and 1e-8 (e.g., ~5e-9), ensuring the original kernel computes a finite positive norm (since it adds epsilon) while the mutant attempts sqrt(negative) → NaN. This requires carefully controlling the magnitude so that sdata[0] is small but non-zero, avoiding underflow in the reduction and preventing the original kernel from failing.

**LLM 建议的输入代码：**

```python
import torch
import math

def generate_inputs(device):
    batch_size = 16
    dim = 16384
    val = math.sqrt(5e-9)  # ~7.071067811865475e-05
    x = torch.zeros(batch_size, dim, device=device, dtype=torch.float32)
    x[:, 0] = val  # First element of each batch set to val
    return [x]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `False`, mutant_ok: `False`
- diff: `max_diff=nan, mean_diff=nan, ref_range=[0.0000e+00,1.0000e+00], mut_range=[nan,nan]`

#### Round 2 (未杀死)

**LLM 存活原因分析：**

> The previous analysis correctly identified that the mutation can cause sqrtf(negative) → NaN when sdata[0] < 1e-8f. However, the implementation appears to have numerical issues when sdata[0] is very small but non-zero. The failed test shows both original and mutant kernels failing (original_ok=False, mutant_ok=False), suggesting there's additional validation or numerical instability in the kernel when handling extremely small norms. The mutant survives because the problematic regime (tiny norms) causes both implementations to fail, making them indistinguishable to the test harness. The mutation only changes behavior when sdata[0] is between 0 and 1e-8f, but if both implementations crash in that regime, the mutant remains undetected.

**LLM 杀死策略：**

> Instead of focusing on extremely small norms where both may fail, target the boundary where sdata[0] is exactly 1e-8f. The original computes sqrtf(2e-8f) ≈ 1.414e-4, while the mutant computes sqrtf(0) = 0, leading to division by zero (1/0 → inf). This should cause the mutant to produce inf values while the original produces finite values around 7071.07. Use exact 1e-8f sum by setting appropriate input values.

**LLM 建议的输入代码：**

```python
import torch
import math

def generate_inputs(device):
    batch_size = 16
    dim = 16384
    target_sum = 1e-8  # Exact boundary value
    val = math.sqrt(target_sum)  # sqrt(1e-8) ≈ 0.0001
    x = torch.zeros(batch_size, dim, device=device, dtype=torch.float32)
    # Set first element of each batch to achieve exact sum of squares = 1e-8
    x[:, 0] = val
    return [x]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `False`, mutant_ok: `False`
- diff: `max_diff=nan, mean_diff=nan, ref_range=[0.0000e+00,1.0000e+00], mut_range=[nan,nan]`

#### Round 3 (**KILLED**)

**LLM 存活原因分析：**

> The previous attempts failed because they targeted extremely small sum-of-squares values (≤1e-8), which caused numerical instability in both the original and mutant kernels, leading to both producing NaN/Inf outputs (original_ok=False, mutant_ok=False). The original kernel's +1e-8 epsilon should guarantee positive sqrt arguments, but small norms cause large scale factors (up to 1e4), which—when multiplied by non-zero inputs—can overflow to Inf if intermediate calculations exceed float32 range. The mutant's -1e-8 epsilon introduces NaN for sdata[0] < 1e-8, but the original also fails due to overflow, making differentiation impossible. The key is to avoid overflow in the original while exploiting the mutant's negative sqrt argument.

**LLM 杀死策略：**

> Use a moderately small sum-of-squares value (e.g., 2e-8) where sdata[0] > 1e-8, ensuring the mutant's sqrt argument remains positive but smaller than the original's. This produces different finite scale factors, leading to diverging normalized outputs. Choose an input distribution that avoids overflow in the original (scale ~1/sqrt(2e-8) ≈ 7071) and uses multiple non-zero elements to keep individual values small, preventing overflow when scaled.

**LLM 建议的输入代码：**

```python
import torch
import math

def generate_inputs(device):
    batch_size = 16
    dim = 16384
    target_sum = 2e-8  # Exact value > 1e-8
    num_nonzero = 1024  # Spread across batch to keep each value small
    val_per_element = math.sqrt(target_sum / num_nonzero)  # ~1.41421356e-06
    x = torch.zeros(batch_size, dim, device=device, dtype=torch.float32)
    # Set first 1024 elements of each batch to the computed value
    x[:, :num_nonzero] = val_per_element
    return [x]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=1.294417e-02, mean_diff=8.090108e-04, ref_range=[0.0000e+00,3.1250e-02], mut_range=[0.0000e+00,4.4194e-02]`

#### 测试构造规则

- **规则名**: `epsilon_sign_flip_norm_boundary`
- **描述**: This rule generates inputs where the sum of squares per batch is exactly 2*epsilon, where epsilon is the small constant added/subtracted in normalization kernels. This creates a boundary condition where the original kernel's sqrt(3*epsilon) and mutant's sqrt(epsilon) produce distinct, finite scale factors without overflow/underflow. Input values are small but non-zero to avoid numerical instability.
- **适用算子**: arith_replace, const_replace, expr_replace

```python
def policy(shape, dtype, device, rng):
    import torch
    import math
    
    # Configuration for epsilon boundary test
    EPS = 1e-8  # Common epsilon value in normalization kernels
    TARGET_SUM = 2 * EPS  # 2*epsilon boundary
    
    # Ensure we have at least batch dimension
    if len(shape) == 1:
        batch_size = 1
        feature_dim = shape[0]
    else:
        batch_size = shape[0]
        feature_dim = shape[1]
    
    # Use ~10% non-zero elements to keep values small but spread out
    num_nonzero = max(1, int(feature_dim * 0.1))
    
    # Compute value per element to achieve exact target sum
    val_per_element = math.sqrt(TARGET_SUM / num_nonzero)
    
    # Create tensor with controlled sum of squares
    x = torch.zeros(shape, device=device, dtype=dtype)
    for b in range(batch_size):
        # Place non-zero elements at deterministic positions
        indices = torch.arange(num_nonzero, device=device)
        if len(shape) == 1:
            x[indices] = val_per_element
        else:
            x[b, indices] = val_per_element
    
    # Verify the construction (optional, for debugging)
    if len(shape) == 1:
        actual_sum = (x ** 2).sum().item()
    else:
        actual_sum = (x[0] ** 2).sum().item()  # Check first batch
    
    return x
```


---

## 13. `L1_P39__epsilon_modify__0`

- **Kernel**: `L1_P39`
- **算子**: `epsilon_modify` (Category C)
- **描述**: Alter small epsilon literals (LayerNorm / safe_div / log stability) @ L49
- **变异行**: Line 49, 原始片段 `1e-8f`, 节点类型 `eps:to_1e-2`
- **杀死轮次**: Round 1

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized L2 normalization
l2_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__global__ void l2_norm_kernel(const T* __restrict__ input, T* __restrict__ output, 
                               int batch_size, int dim) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Use 256 threads per block (power of 2 for reduction)
    const int threads = 256;
    
    // Shared memory for reduction
    __shared__ float sdata[256];
    
    // Each thread processes multiple elements
    float thread_sum = 0.0f;
    
    // Process elements with stride equal to total threads
    for (int i = tid; i < dim; i += threads) {
        int idx = batch_idx * dim + i;
        float val = static_cast<float>(input[idx]);
        thread_sum += val * val;
    }
    
    // Parallel reduction
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Tree reduction
    for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Compute reciprocal of L2 norm (1 / sqrt(sum))
    float scale = 0.0f;
    if (tid == 0) {
        float norm = sqrtf(sdata[0] + 1e-8f);
        scale = 1.0f / norm;
        sdata[0] = scale;
    }
    __syncthreads();
    
    scale = sdata[0];
    
    // Normalize elements
    for (int i = tid; i < dim; i += threads) {
        int idx = batch_idx * dim + i;
        float val = static_cast<float>(input[idx]);
        output[idx] = static_cast<T>(val * scale);
    }
}

torch::Tensor l2_norm_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto dim = input.size(1);
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = batch_size;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_cuda", ([&] {
        l2_norm_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim
        );
    }));
    
    cudaDeviceSynchronize();
    return output;
}
"""

l2_norm_cpp_source = "torch::Tensor l2_norm_cuda(torch::Tensor input);"

# Compile the inline CUDA code for L2 normalization
l2_norm_module = load_inline(
    name="l2_norm",
    cpp_sources=l2_norm_cpp_source,
    cuda_sources=l2_norm_source,
    functions=["l2_norm_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized L2 normalization model with custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization using custom CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).
            
        Returns:
            torch.Tensor: L2 normalized tensor, same shape as input.
        """
        # Ensure tensor is on CUDA device
        if not x.is_cuda:
            x = x.cuda()
            
        return l2_norm_module.l2_norm_cuda(x)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized L2 normalization
l2_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__global__ void l2_norm_kernel(const T* __restrict__ input, T* __restrict__ output, 
                               int batch_size, int dim) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Use 256 threads per block (power of 2 for reduction)
    const int threads = 256;
    
    // Shared memory for reduction
    __shared__ float sdata[256];
    
    // Each thread processes multiple elements
    float thread_sum = 0.0f;
    
    // Process elements with stride equal to total threads
    for (int i = tid; i < dim; i += threads) {
        int idx = batch_idx * dim + i;
        float val = static_cast<float>(input[idx]);
        thread_sum += val * val;
    }
    
    // Parallel reduction
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Tree reduction
    for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Compute reciprocal of L2 norm (1 / sqrt(sum))
    float scale = 0.0f;
    if (tid == 0) {
        float norm = sqrtf(sdata[0] + 1e-2f);
        scale = 1.0f / norm;
        sdata[0] = scale;
    }
    __syncthreads();
    
    scale = sdata[0];
    
    // Normalize elements
    for (int i = tid; i < dim; i += threads) {
        int idx = batch_idx * dim + i;
        float val = static_cast<float>(input[idx]);
        output[idx] = static_cast<T>(val * scale);
    }
}

torch::Tensor l2_norm_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto dim = input.size(1);
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = batch_size;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_cuda", ([&] {
        l2_norm_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim
        );
    }));
    
    cudaDeviceSynchronize();
    return output;
}
"""

l2_norm_cpp_source = "torch::Tensor l2_norm_cuda(torch::Tensor input);"

# Compile the inline CUDA code for L2 normalization
l2_norm_module = load_inline(
    name="l2_norm",
    cpp_sources=l2_norm_cpp_source,
    cuda_sources=l2_norm_source,
    functions=["l2_norm_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized L2 normalization model with custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization using custom CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).
            
        Returns:
            torch.Tensor: L2 normalized tensor, same shape as input.
        """
        # Ensure tensor is on CUDA device
        if not x.is_cuda:
            x = x.cuda()
            
        return l2_norm_module.l2_norm_cuda(x)
```

### Part 3: 杀死该变异体的输入

**杀死输入代码 (Round 1)：**

```python
import torch
import math

def generate_inputs(device):
    batch_size = 16
    dim = 16384
    
    # Create zero tensor
    x = torch.zeros(batch_size, dim, device=device, dtype=torch.float32)
    
    # For batch 0, set exactly one element to sqrt(1e-5) = ~0.003162
    # This makes sum-of-squares = (0.003162)^2 = 1e-5
    target_sum_sq = 1e-5
    single_val = math.sqrt(target_sum_sq)  # ~0.003162
    x[0, 0] = single_val
    
    # For other batches, set random values that won't trigger epsilon sensitivity
    # to ensure other outputs remain identical
    for b in range(1, batch_size):
        x[b] = torch.randn(dim, device=device) * 0.1 + 1.0  # Norm ~sqrt(16384×1)≈128
    
    return [x]
```

**输出差异**: `max_diff=9.683930e-01, mean_diff=3.696234e-06, ref_range=[0.0000e+00,1.0000e+00], mut_range=[0.0000e+00,3.1607e-02]`

### Part 4: 大模型诊断流程

共 1 轮迭代分析：

#### Round 1 (**KILLED**)

**LLM 存活原因分析：**

> The mutation survives because the epsilon value (1e-2 vs 1e-8) only affects the normalization scale when the sum of squares (sdata[0]) is extremely small. For the fixed input shape [16, 16384], stress tests used random values but couldn't produce sum-of-squares in the critical regime where epsilon dominates. Specifically, with 16384 elements, even tiny values like 1e-7 produce sum-of-squares ~16384×(1e-7)² ≈ 1.6e-10, which is still much smaller than 1e-8, making both epsilons dominate. However, the output becomes effectively zero for all elements in this regime, masking differences. The mutation only matters when sum-of-squares is between ~1e-8 and ~1e-2, but random distributions rarely hit this narrow window precisely.

**LLM 杀死策略：**

> Create a vector where the sum of squares is precisely controlled to be between 1e-8 and 1e-2, making the epsilon term significant but not dominant. Use a single non-zero element with carefully chosen magnitude, so the scale factor differs substantially between original (1/sqrt(x+1e-8)) and mutant (1/sqrt(x+1e-2)).

**LLM 建议的输入代码：**

```python
import torch
import math

def generate_inputs(device):
    batch_size = 16
    dim = 16384
    
    # Create zero tensor
    x = torch.zeros(batch_size, dim, device=device, dtype=torch.float32)
    
    # For batch 0, set exactly one element to sqrt(1e-5) = ~0.003162
    # This makes sum-of-squares = (0.003162)^2 = 1e-5
    target_sum_sq = 1e-5
    single_val = math.sqrt(target_sum_sq)  # ~0.003162
    x[0, 0] = single_val
    
    # For other batches, set random values that won't trigger epsilon sensitivity
    # to ensure other outputs remain identical
    for b in range(1, batch_size):
        x[b] = torch.randn(dim, device=device) * 0.1 + 1.0  # Norm ~sqrt(16384×1)≈128
    
    return [x]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=9.683930e-01, mean_diff=3.696234e-06, ref_range=[0.0000e+00,1.0000e+00], mut_range=[0.0000e+00,3.1607e-02]`

#### 测试构造规则

- **规则名**: `epsilon_sensitivity_targeting`
- **描述**: Generates inputs where the sum of squares along the normalized dimension falls within epsilon-sensitive regimes. Specifically targets values comparable to the epsilon values used in numerical stability guards (e.g., between 1e-8 and 1e-2). Creates controlled scenarios where epsilon variations produce measurable output differences.
- **适用算子**: epsilon_modify, constant_change, arithmetic_replacement, div_mul_replacement

```python
def policy(shape, dtype, device, rng):
    import torch
    import math
    
    # Assume shape follows (batch, dim) pattern for normalization kernels
    if len(shape) != 2:
        # For other shapes, create a simple 2D case
        batch_size = 16
        dim = 16384
    else:
        batch_size, dim = shape
    
    x = torch.zeros(batch_size, dim, device=device, dtype=dtype)
    
    # Select target sum-of-squares values in epsilon-sensitive range
    # Cover multiple orders of magnitude around typical epsilon values
    target_ss_candidates = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    
    # Use rng to select one target value for the sensitive batch
    ss_target = target_ss_candidates[rng.randint(0, len(target_ss_candidates))]
    
    # For batch 0: create epsilon-sensitive pattern
    # Option 1: Single non-zero element (simplest)
    if rng.random() < 0.5:
        single_val = math.sqrt(ss_target)
        x[0, 0] = single_val
    else:
        # Option 2: Multiple small elements that sum to target
        # Distribute target sum across multiple elements
        n_elements = rng.randint(2, min(10, dim))
        val_per_element = math.sqrt(ss_target / n_elements)
        for i in range(n_elements):
            x[0, i] = val_per_element
    
    # For remaining batches: create non-sensitive patterns
    # with norms large enough to dominate epsilon
    for b in range(1, batch_size):
        # Create random vectors with norm >> epsilon
        # Norm ~ sqrt(dim) to ensure sum-squares ~ O(dim)
        random_vec = torch.randn(dim, device=device, dtype=dtype) * 0.1 + 1.0
        x[b] = random_vec
    
    return [x]
```


---

## 14. `L1_P47__index_replace__3`

- **Kernel**: `L1_P47`
- **算子**: `index_replace` (Category B)
- **描述**: Swap Triton program_id axis or CUDA thread/block dimension index (e.g. program_id(0)→(1), threadIdx.x→threadIdx.y) @ L22
- **变异行**: Line 22, 原始片段 `blockIdx.y`, 节点类型 `cuda_dim|z`
- **杀死轮次**: Round 2

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized sum reduction with keepdim=True
sum_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

template<typename scalar_t>
__global__ void sum_reduction_kernel(
    const scalar_t* input,
    scalar_t* output,
    const int outer_size,
    const int reduce_size,
    const int inner_size
) {
    // Each thread handles one element in the output
    const int outer_idx = blockIdx.x;
    const int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    const scalar_t* input_ptr = input + outer_idx * reduce_size * inner_size + inner_idx;
    scalar_t sum = 0;
    
    // Sequential reduction across reduce dimension
    for (int reduce_idx = 0; reduce_idx < reduce_size; reduce_idx++) {
        sum += input_ptr[reduce_idx * inner_size];
    }
    
    output[outer_idx * inner_size + inner_idx] = sum;
}

torch::Tensor sum_reduction_cuda(torch::Tensor input, int dim) {
    // Get tensor dimensions
    auto sizes = input.sizes();
    int ndim = input.dim();
    
    // Validate dimension
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("Invalid reduction dimension");
    }
    
    // Calculate sizes for the reduction
    int outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    
    int reduce_size = sizes[dim];
    
    int inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner_size *= sizes[i];
    }
    
    // Prepare output tensor with keepdim=True
    std::vector<int64_t> output_sizes;
    for (int i = 0; i < ndim; i++) {
        if (i == dim) {
            output_sizes.push_back(1);
        } else {
            output_sizes.push_back(sizes[i]);
        }
    }
    
    auto output = torch::empty(output_sizes, input.options());
    
    // Launch kernel configuration
    const int threads_per_block = 256;
    dim3 blocks(outer_size, (inner_size + threads_per_block - 1) / threads_per_block);
    
    // Dispatch based on data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "sum_reduction_cuda", [&] {
        sum_reduction_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            reduce_size,
            inner_size
        );
    });
    
    return output;
}
"""

sum_reduction_cpp_source = "torch::Tensor sum_reduction_cuda(torch::Tensor input, int dim);"

# Compile the inline CUDA code
sum_reduction = load_inline(
    name="sum_reduction",
    cpp_sources=sum_reduction_cpp_source,
    cuda_sources=sum_reduction_source,
    functions=["sum_reduction_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction over a specified dimension
    using custom CUDA kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.sum_reduction = sum_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized sum reduction over the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        if x.is_cuda:
            return self.sum_reduction.sum_reduction_cuda(x, self.dim)
        else:
            # Fallback to PyTorch for CPU tensors
            return torch.sum(x, dim=self.dim, keepdim=True)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized sum reduction with keepdim=True
sum_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

template<typename scalar_t>
__global__ void sum_reduction_kernel(
    const scalar_t* input,
    scalar_t* output,
    const int outer_size,
    const int reduce_size,
    const int inner_size
) {
    // Each thread handles one element in the output
    const int outer_idx = blockIdx.x;
    const int inner_idx = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    const scalar_t* input_ptr = input + outer_idx * reduce_size * inner_size + inner_idx;
    scalar_t sum = 0;
    
    // Sequential reduction across reduce dimension
    for (int reduce_idx = 0; reduce_idx < reduce_size; reduce_idx++) {
        sum += input_ptr[reduce_idx * inner_size];
    }
    
    output[outer_idx * inner_size + inner_idx] = sum;
}

torch::Tensor sum_reduction_cuda(torch::Tensor input, int dim) {
    // Get tensor dimensions
    auto sizes = input.sizes();
    int ndim = input.dim();
    
    // Validate dimension
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("Invalid reduction dimension");
    }
    
    // Calculate sizes for the reduction
    int outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    
    int reduce_size = sizes[dim];
    
    int inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner_size *= sizes[i];
    }
    
    // Prepare output tensor with keepdim=True
    std::vector<int64_t> output_sizes;
    for (int i = 0; i < ndim; i++) {
        if (i == dim) {
            output_sizes.push_back(1);
        } else {
            output_sizes.push_back(sizes[i]);
        }
    }
    
    auto output = torch::empty(output_sizes, input.options());
    
    // Launch kernel configuration
    const int threads_per_block = 256;
    dim3 blocks(outer_size, (inner_size + threads_per_block - 1) / threads_per_block);
    
    // Dispatch based on data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "sum_reduction_cuda", [&] {
        sum_reduction_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            reduce_size,
            inner_size
        );
    });
    
    return output;
}
"""

sum_reduction_cpp_source = "torch::Tensor sum_reduction_cuda(torch::Tensor input, int dim);"

# Compile the inline CUDA code
sum_reduction = load_inline(
    name="sum_reduction",
    cpp_sources=sum_reduction_cpp_source,
    cuda_sources=sum_reduction_source,
    functions=["sum_reduction_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction over a specified dimension
    using custom CUDA kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.sum_reduction = sum_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized sum reduction over the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        if x.is_cuda:
            return self.sum_reduction.sum_reduction_cuda(x, self.dim)
        else:
            # Fallback to PyTorch for CPU tensors
            return torch.sum(x, dim=self.dim, keepdim=True)
```

### Part 3: 杀死该变异体的输入

**杀死输入代码 (Round 2)：**

```python
import torch
def generate_inputs(device):
    outer_size = 2
    reduce_size = 3
    inner_size = 2000  # Ensures grid.y > 1 even if blockDim.x = 1024
    input_tensor = torch.randn(outer_size, reduce_size, inner_size, device=device, dtype=torch.float32)
    return [input_tensor]
```

**输出差异**: `max_diff=1.195624e+01, mean_diff=1.205811e+00, ref_range=[-5.9781e+00,6.0174e+00], mut_range=[-5.3571e+00,6.0174e+00]`

### Part 4: 大模型诊断流程

共 2 轮迭代分析：

#### Round 1 (未杀死)

**LLM 存活原因分析：**

> The mutation changed `blockIdx.y` to `blockIdx.z` for computing the inner index. Since the kernel launch configuration uses a 2D grid (x and y dimensions only), `blockIdx.z` is always 0. The original kernel's grid is configured such that `blockIdx.y` spans the inner dimension. With the fixed test input shape [16, 256, 256] and typical block size of 256, the grid's y-dimension size is ⌈256/256⌉ = 1. Thus, both `blockIdx.y` and `blockIdx.z` evaluate to 0, making the mutant equivalent to the original. The stress tests only varied values within the same shape, so this equivalence persisted.

**LLM 杀死策略：**

> Change the input shape to require multiple blocks in the y-dimension by increasing `inner_size` beyond the block's x-dimension (typically 256). The original kernel will use `blockIdx.y` to cover the full inner range, while the mutant's `blockIdx.z` remains 0, causing it to compute only the first block's worth of inner indices and leave the rest uninitialized (or zero if output is zero-initialized). Using an `inner_size` not divisible by the block size ensures the last block is partially filled, making the missing computations detectable.

**LLM 建议的输入代码：**

```python
import torch
def generate_inputs(device):
    outer_size = 16
    reduce_size = 256
    inner_size = 257  # > typical blockDim.x (256)
    input_tensor = torch.ones(outer_size, reduce_size, inner_size, device=device, dtype=torch.float32)
    return [input_tensor]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `True`

#### Round 2 (**KILLED**)

**LLM 存活原因分析：**

> The mutant likely survived because the previous test's inner_size (257) was still too small relative to the actual blockDim.x used by the test framework, causing grid.y to remain 1 (e.g., if blockDim.x=1024). With grid.y=1, both the original and mutant kernels use only blockIdx.y=0 (or blockIdx.z=0), making them equivalent. Additionally, even if grid.y>1, the mutant's use of blockIdx.z (always 0) leads to repeated writes to the same inner indices from multiple blocks, but the original kernel writes to distinct indices. However, if the output tensor is not zero-initialized, the missing writes in the mutant might be filled with uninitialized memory that coincidentally matches the expected values, though this is unlikely. The key issue is ensuring grid.y>1 regardless of blockDim.x.

**LLM 杀死策略：**

> Use a very large inner_size (2000) to guarantee grid.y>1 even if blockDim.x is at its maximum (1024). This forces the mutant to compute only the first blockDim.x inner indices and leave the rest uninitialized, while the original kernel computes all inner indices. Random input values ensure each inner index has a unique sum, making missing writes detectable. Also, small outer_size and reduce_size keep the test fast.

**LLM 建议的输入代码：**

```python
import torch
def generate_inputs(device):
    outer_size = 2
    reduce_size = 3
    inner_size = 2000  # Ensures grid.y > 1 even if blockDim.x = 1024
    input_tensor = torch.randn(outer_size, reduce_size, inner_size, device=device, dtype=torch.float32)
    return [input_tensor]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=1.195624e+01, mean_diff=1.205811e+00, ref_range=[-5.9781e+00,6.0174e+00], mut_range=[-5.3571e+00,6.0174e+00]`

#### 测试构造规则

- **规则名**: `force_multiple_grid_dimensions`
- **描述**: Generates inputs where at least one dimension significantly exceeds typical block sizes (e.g., >1024) to ensure multiple blocks are launched in multiple grid dimensions. This reveals index-swapping mutants that incorrectly map threads to data elements when grid dimensions differ.
- **适用算子**: index_replace, dimension_swap, thread_id_mutation

```python
def policy(shape, dtype, device, rng):
    # Determine the largest dimension in the original shape
    max_dim_idx = max(range(len(shape)), key=lambda i: shape[i])
    # Create a modified shape where the largest dimension is increased to 2000
    # (or at least >1024) to guarantee multiple blocks even with max blockDim.x
    modified_shape = list(shape)
    modified_shape[max_dim_idx] = max(2000, shape[max_dim_idx] + 1024)
    
    # Generate random non-zero data to ensure distinct outputs
    if dtype.is_floating_point:
        tensor = torch.rand(*modified_shape, device=device, dtype=dtype, generator=rng) * 2 - 1  # uniform [-1, 1]
        # Avoid exact zeros to prevent masking errors
        tensor[tensor == 0] = 0.001
    else:
        # For integer types, generate values in a non-zero range
        tensor = torch.randint(1, 100, modified_shape, device=device, dtype=dtype, generator=rng)
    return tensor
```


---

## 15. `L1_P47__relop_replace__1`

- **Kernel**: `L1_P47`
- **算子**: `relop_replace` (Category A)
- **描述**: Replace relational operators (<→<=, <=→<, >→>=, >=→>, ==→!=, !=→==) @ L24
- **变异行**: Line 24, 原始片段 `>=`, 节点类型 `cuda_GtE`
- **杀死轮次**: Round 1

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized sum reduction with keepdim=True
sum_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

template<typename scalar_t>
__global__ void sum_reduction_kernel(
    const scalar_t* input,
    scalar_t* output,
    const int outer_size,
    const int reduce_size,
    const int inner_size
) {
    // Each thread handles one element in the output
    const int outer_idx = blockIdx.x;
    const int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    const scalar_t* input_ptr = input + outer_idx * reduce_size * inner_size + inner_idx;
    scalar_t sum = 0;
    
    // Sequential reduction across reduce dimension
    for (int reduce_idx = 0; reduce_idx < reduce_size; reduce_idx++) {
        sum += input_ptr[reduce_idx * inner_size];
    }
    
    output[outer_idx * inner_size + inner_idx] = sum;
}

torch::Tensor sum_reduction_cuda(torch::Tensor input, int dim) {
    // Get tensor dimensions
    auto sizes = input.sizes();
    int ndim = input.dim();
    
    // Validate dimension
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("Invalid reduction dimension");
    }
    
    // Calculate sizes for the reduction
    int outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    
    int reduce_size = sizes[dim];
    
    int inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner_size *= sizes[i];
    }
    
    // Prepare output tensor with keepdim=True
    std::vector<int64_t> output_sizes;
    for (int i = 0; i < ndim; i++) {
        if (i == dim) {
            output_sizes.push_back(1);
        } else {
            output_sizes.push_back(sizes[i]);
        }
    }
    
    auto output = torch::empty(output_sizes, input.options());
    
    // Launch kernel configuration
    const int threads_per_block = 256;
    dim3 blocks(outer_size, (inner_size + threads_per_block - 1) / threads_per_block);
    
    // Dispatch based on data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "sum_reduction_cuda", [&] {
        sum_reduction_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            reduce_size,
            inner_size
        );
    });
    
    return output;
}
"""

sum_reduction_cpp_source = "torch::Tensor sum_reduction_cuda(torch::Tensor input, int dim);"

# Compile the inline CUDA code
sum_reduction = load_inline(
    name="sum_reduction",
    cpp_sources=sum_reduction_cpp_source,
    cuda_sources=sum_reduction_source,
    functions=["sum_reduction_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction over a specified dimension
    using custom CUDA kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.sum_reduction = sum_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized sum reduction over the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        if x.is_cuda:
            return self.sum_reduction.sum_reduction_cuda(x, self.dim)
        else:
            # Fallback to PyTorch for CPU tensors
            return torch.sum(x, dim=self.dim, keepdim=True)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized sum reduction with keepdim=True
sum_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

template<typename scalar_t>
__global__ void sum_reduction_kernel(
    const scalar_t* input,
    scalar_t* output,
    const int outer_size,
    const int reduce_size,
    const int inner_size
) {
    // Each thread handles one element in the output
    const int outer_idx = blockIdx.x;
    const int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (outer_idx >= outer_size || inner_idx > inner_size) return;
    
    const scalar_t* input_ptr = input + outer_idx * reduce_size * inner_size + inner_idx;
    scalar_t sum = 0;
    
    // Sequential reduction across reduce dimension
    for (int reduce_idx = 0; reduce_idx < reduce_size; reduce_idx++) {
        sum += input_ptr[reduce_idx * inner_size];
    }
    
    output[outer_idx * inner_size + inner_idx] = sum;
}

torch::Tensor sum_reduction_cuda(torch::Tensor input, int dim) {
    // Get tensor dimensions
    auto sizes = input.sizes();
    int ndim = input.dim();
    
    // Validate dimension
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("Invalid reduction dimension");
    }
    
    // Calculate sizes for the reduction
    int outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    
    int reduce_size = sizes[dim];
    
    int inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner_size *= sizes[i];
    }
    
    // Prepare output tensor with keepdim=True
    std::vector<int64_t> output_sizes;
    for (int i = 0; i < ndim; i++) {
        if (i == dim) {
            output_sizes.push_back(1);
        } else {
            output_sizes.push_back(sizes[i]);
        }
    }
    
    auto output = torch::empty(output_sizes, input.options());
    
    // Launch kernel configuration
    const int threads_per_block = 256;
    dim3 blocks(outer_size, (inner_size + threads_per_block - 1) / threads_per_block);
    
    // Dispatch based on data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "sum_reduction_cuda", [&] {
        sum_reduction_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            reduce_size,
            inner_size
        );
    });
    
    return output;
}
"""

sum_reduction_cpp_source = "torch::Tensor sum_reduction_cuda(torch::Tensor input, int dim);"

# Compile the inline CUDA code
sum_reduction = load_inline(
    name="sum_reduction",
    cpp_sources=sum_reduction_cpp_source,
    cuda_sources=sum_reduction_source,
    functions=["sum_reduction_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction over a specified dimension
    using custom CUDA kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.sum_reduction = sum_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized sum reduction over the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        if x.is_cuda:
            return self.sum_reduction.sum_reduction_cuda(x, self.dim)
        else:
            # Fallback to PyTorch for CPU tensors
            return torch.sum(x, dim=self.dim, keepdim=True)
```

### Part 3: 杀死该变异体的输入

**杀死输入代码 (Round 1)：**

```python
import torch

def generate_inputs(device):
    outer_size = 2
    reduce_size = 2
    inner_size = 1
    # Input tensor shaped [outer_size, reduce_size, inner_size]
    # Values: [[[1], [2]], [[3], [4]]] flattened to [1,2,3,4]
    input_tensor = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]], device=device, dtype=torch.float32)
    return [input_tensor]

```

**输出差异**: `max_diff=2.000000e+00, mean_diff=1.000000e+00, ref_range=[3.0000e+00,7.0000e+00], mut_range=[3.0000e+00,5.0000e+00]`

### Part 4: 大模型诊断流程

共 1 轮迭代分析：

#### Round 1 (**KILLED**)

**LLM 存活原因分析：**

> The mutant changes the boundary condition from 'inner_idx >= inner_size' to 'inner_idx > inner_size', allowing threads with inner_idx == inner_size to execute. In the original stress tests, the input shape [16, 256, 256] was combined with a block size of 256 (or a divisor), resulting in grid dimensions that never launch threads with inner_idx == inner_size. Specifically, with inner_size=256 and blockDim.x=256, gridDim.y = ceil(256/256)=1, so inner_idx ranges from 0 to 255. Thus, the mutant's altered condition never triggers, and both kernels behave identically for all value distributions.

**LLM 杀死策略：**

> To kill the mutant, we must change the input shape so that inner_size is not a multiple of the block size, causing extra threads with inner_idx == inner_size to be launched. A shape of [2, 2, 1] with blockDim.x=2 (implied by typical kernel launch heuristics) will launch threads with inner_idx=0 and 1. The thread with inner_idx=1 (equal to inner_size) proceeds in the mutant but not in the original. This thread performs out-of-bounds reads and writes, leading to a race condition that can alter the output numerically.

**LLM 建议的输入代码：**

```python
import torch

def generate_inputs(device):
    outer_size = 2
    reduce_size = 2
    inner_size = 1
    # Input tensor shaped [outer_size, reduce_size, inner_size]
    # Values: [[[1], [2]], [[3], [4]]] flattened to [1,2,3,4]
    input_tensor = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]], device=device, dtype=torch.float32)
    return [input_tensor]

```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=2.000000e+00, mean_diff=1.000000e+00, ref_range=[3.0000e+00,7.0000e+00], mut_range=[3.0000e+00,5.0000e+00]`

#### 测试构造规则

- **规则名**: `non_aligned_inner_dimension`
- **描述**: Generates inputs where the inner dimension size is not a multiple of common GPU block sizes (especially 256). This forces grid launches where the last block contains threads with indices that exceed valid boundaries when boundary conditions are mutated. The rule targets boundary condition mutants by ensuring at least one thread executes beyond the intended range.
- **适用算子**: relop_replace, boundary_mutation, relop_mutation

```python
def policy(shape, dtype, device, rng):
    # Target kernels with grid-stride loops where boundary conditions
    # protect against out-of-range access
    import torch
    import numpy as np
    
    # Choose inner_size that is not divisible by typical block sizes
    # to ensure extra threads in the last block
    problematic_sizes = [1, 3, 5, 7, 9, 17, 33, 65, 129, 257, 513, 1025]
    inner_size = int(rng.choice(problematic_sizes))
    
    # Keep outer dimensions small to create manageable test cases
    outer_size = rng.randint(1, 5)
    reduce_size = rng.randint(1, 5)
    
    # Construct tensor with shape that forces boundary condition exposure
    shape = (outer_size, reduce_size, inner_size)
    
    # Use random values in a reasonable range
    tensor = torch.randn(shape, dtype=dtype, device=device)
    
    # Scale to avoid extreme values that might mask boundary issues
    tensor = tensor * 2.0 - 1.0  # Values in [-1, 1]
    
    return tensor
```


---

## 16. `L1_P50__mask_boundary__2`

- **Kernel**: `L1_P50`
- **算子**: `mask_boundary` (Category B)
- **描述**: Weaken or tighten boundary checks in mask/tl.where (Triton) or thread/block guards (CUDA) @ L51
- **变异行**: Line 51, 原始片段 `idx < reduction_size`, 节点类型 `rhs-1`
- **杀死轮次**: Round 1

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
            if (idx < reduction_size - 1) {
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

### Part 3: 杀死该变异体的输入

**杀死输入代码 (Round 1)：**

```python
import torch
def generate_inputs(device):
    x = torch.ones([16, 256, 256], device=device, dtype=torch.float32)
    x[-1, :, :] = 0.5  # last element of first dimension
    x[:, -1, :] = 0.5  # last element of second dimension
    x[:, :, -1] = 0.5  # last element of third dimension
    return [x]
```

**输出差异**: `max_diff=5.000000e-01, mean_diff=1.831055e-03, ref_range=[0.0000e+00,5.0000e-01], mut_range=[0.0000e+00,1.0000e+00]`

### Part 4: 大模型诊断流程

共 1 轮迭代分析：

#### Round 1 (**KILLED**)

**LLM 存活原因分析：**

> The mutant survived because the original kernel's memory access pattern does not include the last element (index reduction_size-1) in its iteration space. The loop strides by elements_per_thread * blockDim.x, and the inner loop increments by blockDim.x. For the given input shape and launch configuration, the index reduction_size-1 is not accessed by any thread due to the combination of blockDim.x and reduction_size. Hence, tightening the boundary condition from idx < reduction_size to idx < reduction_size - 1 has no effect because the last element was never processed in the original kernel.

**LLM 杀死策略：**

> The mutant can be killed by ensuring the last element of the reduction dimension is actually accessed in the original kernel. This requires changing the input shape so that the reduction dimension size is not a multiple of the stride (elements_per_thread * blockDim.x), forcing the last element to be within the loop's iteration space. Alternatively, if the reduction dimension is one of the tensor dimensions, we can set the last element along that dimension to a distinctive non-1 value (e.g., 0.5) while setting all other elements to 1.0, making the product sensitive to the presence of that last element.

**LLM 建议的输入代码：**

```python
import torch
def generate_inputs(device):
    x = torch.ones([16, 256, 256], device=device, dtype=torch.float32)
    x[-1, :, :] = 0.5  # last element of first dimension
    x[:, -1, :] = 0.5  # last element of second dimension
    x[:, :, -1] = 0.5  # last element of third dimension
    return [x]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=5.000000e-01, mean_diff=1.831055e-03, ref_range=[0.0000e+00,5.0000e-01], mut_range=[0.0000e+00,1.0000e+00]`

#### 测试构造规则

- **规则名**: `last_element_non_identity_per_dim`
- **描述**: Creates inputs where the last element along each dimension is set to a non-identity value for common reduction operations (like 0.5 for product, non-zero for sum), while all other elements are set to the identity value. This exposes boundary condition mutations that skip the last element by making its inclusion/exclusion affect the computation result.
- **适用算子**: mask_boundary, guard_removal, loop_boundary_adjust

```python
def policy(shape, dtype, device, rng):
    import torch
    # Create tensor with identity values (1 for multiplicative reductions, 0 for additive)
    # Use 1.0 as default since it's identity for multiplication and doesn't affect addition
    tensor = torch.ones(shape, dtype=dtype, device=device)
    
    # Set last element along each dimension to non-identity value
    # For floating point: use 0.5 (affects product, mean, etc.)
    # For integer: use 2 (affects sum, product)
    non_identity = torch.tensor(0.5 if torch.is_floating_point(tensor) else 2,
                                 dtype=dtype, device=device)
    
    for dim in range(len(shape)):
        # Build index to select last element along current dimension
        idx = [slice(None)] * len(shape)
        idx[dim] = -1
        tensor[idx] = non_identity
    
    return tensor
```


---

## 17. `L1_P50__relop_replace__8`

- **Kernel**: `L1_P50`
- **算子**: `relop_replace` (Category A)
- **描述**: Replace relational operators (<→<=, <=→<, >→>=, >=→>, ==→!=, !=→==) @ L69
- **变异行**: Line 69, 原始片段 `<`, 节点类型 `cuda_Lt`
- **杀死轮次**: Round 3

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
        scalar_t val = (threadIdx.x <= (blockDim.x + 31) / 32) ? shared[threadIdx.x] : scalar_t(1.0);
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

**杀死输入代码 (Round 3)：**

```python
import torch

def generate_inputs(device):
    shape = [16, 256, 256]
    x = torch.ones(shape, dtype=torch.float32, device=device)
    return [x]
```

**输出差异**: `max_diff=1.000000e+00, mean_diff=3.906250e-03, ref_range=[1.0000e+00,1.0000e+00], mut_range=[0.0000e+00,1.0000e+00]`

### Part 4: 大模型诊断流程

共 3 轮迭代分析：

#### Round 1 (未杀死)

**LLM 存活原因分析：**

> The mutation changes the condition from `threadIdx.x < (blockDim.x + 31) / 32` to `threadIdx.x <= (blockDim.x + 31) / 32` for reading from shared memory. This only matters when threadIdx.x equals the number of warps (num_warps = ceil(blockDim.x/32)). For blockDim.x=256 (as implied by input shape), num_warps=8. The original uses scalar_t(1.0) for thread 8, while the mutant reads shared[8] (uninitialized). However, if the uninitialized value happens to be 1.0 (common in GPU memory), the product reduction remains unchanged. Additionally, even if it's not 1.0, the final product might be dominated by zeros or NaNs from other threads, masking the difference.

**LLM 杀死策略：**

> Use a block size that is NOT a multiple of 32, making num_warps = ceil(blockDim.x/32) not equal to blockDim.x/32. Then threadIdx.x = num_warps will be valid and read uninitialized shared memory. Set all input values to a small positive non-zero number (e.g., 1.000001) so the product is finite and sensitive to the extra factor. Ensure no zeros or extreme values that could dominate or cause NaN/overflow.

**LLM 建议的输入代码：**

```python
import torch
def generate_inputs(device):
    # Use a shape that forces blockDim.x = 33 (or any non-multiple of 32).
    # We need to adjust the kernel launch, but for the test we just provide input.
    # Since the kernel likely uses blockDim.x = last dimension (256 originally),
    # we change last dimension to 33.
    shape = [16, 256, 33]  # Changed last dim from 256 to 33
    # Fill with values slightly above 1.0, so product is sensitive to extra factor
    val = 1.000001
    x = torch.full(shape, val, dtype=torch.float32, device=device)
    return [x]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `False`, mutant_ok: `False`
- diff: `max_diff=1.000244e+00, mean_diff=3.031043e-02, ref_range=[1.0002e+00,1.0002e+00], mut_range=[0.0000e+00,1.0002e+00]`

#### Round 2 (未杀死)

**LLM 存活原因分析：**

> The mutant survives because the out-of-bounds access at shared[num_warps] (when threadIdx.x = num_warps) reads uninitialized memory, which can be any value. Previous attempts with small positive values (1.000001) caused the original kernel to compute approximately 1.000033^33 while the mutant multiplied by uninitialized memory (potentially 0.0, giving 0.0). However, the comparison between original and mutant outputs showed differences but the mutant wasn't 'killed' - likely because the testing framework only kills mutants when they produce DIFFERENT but VALID outputs (not NaN/inf). When uninitialized memory contains 0.0, the mutant produces 0.0, which differs from the original's ~1.000033, but 0.0 might be considered a 'valid' output (finite). The key is that the out-of-bounds read's value is non-deterministic across runs, making consistent detection tricky.

**LLM 杀死策略：**

> Force the out-of-bounds read to be a predictable, non-1.0 value by poisoning shared memory with sentinel values. Use a block size that's NOT a multiple of 32 (e.g., 33) so num_warps=2 and threadIdx.x=2 reads shared[2] (out-of-bounds). Before the kernel launch, we can't directly initialize shared memory, but we can influence what resides in that memory location by using input values that cause specific warp products to be written to adjacent shared memory slots. By making all valid warp products = 1.0 and ensuring the out-of-bounds location (which is actually the next slot in shared memory) contains a known non-1.0 value from a previous kernel launch or careful indexing.

**LLM 建议的输入代码：**

```python
import torch

def generate_inputs(device):
    shape = [16, 256, 33]  # last dim 33 forces blockDim.x=33
    # Fill with -1.0, so each warp product depends on number of elements
    # Warp0: 32 elements → (-1)^32 = 1.0
    # Warp1: 1 element  → (-1)^1 = -1.0
    x = torch.full(shape, -1.0, dtype=torch.float32, device=device)
    # Modify one element in warp1 to be 2.0, making shared[1] = -2.0
    # This ensures the final product is not ±1.0 and more sensitive
    x[:, :, 32] = 2.0  # last element (index 32) is in warp1
    return [x]

```

**GPU 验证结果：**

- ref_ok: `False`, original_ok: `False`, mutant_ok: `False`
- error: `ref output NaN/Inf on LLM input`

#### Round 3 (**KILLED**)

**LLM 存活原因分析：**

> The mutant survives because previous attempts changed the block size (via input shape), causing the original kernel to also fail (original_ok=False). The mutation only affects behavior when threadIdx.x equals num_warps (the number of warps), triggering an out-of-bounds read of shared memory. This out-of-bounds read is uncontrolled and may coincidentally read 1.0, making the mutant behave identically to the original. The original kernel is designed for blockDim.x=256 (last dimension 256), so altering the shape invalidates the kernel's assumptions, leading to incorrect outputs for both original and mutant.

**LLM 杀死策略：**

> Use the intended input shape [16, 256, 256] and fill all values with 1.0. This ensures the correct product is 1.0 and the original kernel should compute 1.0. The mutant will read an out-of-bounds shared memory location (index 8) due to the changed condition. If this location contains any value other than 1.0 (e.g., 0.0 from uninitialized memory), the mutant's product will differ, killing the mutant.

**LLM 建议的输入代码：**

```python
import torch

def generate_inputs(device):
    shape = [16, 256, 256]
    x = torch.ones(shape, dtype=torch.float32, device=device)
    return [x]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=1.000000e+00, mean_diff=3.906250e-03, ref_range=[1.0000e+00,1.0000e+00], mut_range=[0.0000e+00,1.0000e+00]`

#### 测试构造规则

- **规则名**: `design_shape_constant_one`
- **描述**: Generates inputs with the kernel's designed shape (typically the last dimension matching thread block size) and all values set to 1.0. This tests boundary condition mutations by ensuring out-of-bounds memory accesses (caused by operator changes) read uncontrolled values that differ from the constant, while the original kernel behaves correctly with valid indices.
- **适用算子**: relop_replace, array_index_replace, boundary_change

```python
def policy(shape, dtype, device, rng):
    # Preserve the original shape (assumed to be the kernel's designed shape)
    # Fill entire tensor with 1.0 to detect out-of-bounds reads from mutations
    return torch.ones(shape, dtype=dtype, device=device)
```


---

## 18. `L1_P90__arith_replace__1`

- **Kernel**: `L1_P90`
- **算子**: `arith_replace` (Category A)
- **描述**: Replace binary arithmetic operators (+→-, -→+, *→/, /→*) @ L23
- **变异行**: Line 23, 原始片段 `*`, 节点类型 `cuda_Mult`
- **杀死轮次**: Round 3

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for cumulative product
cumprod_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void cumprod_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size) {
    
    const int64_t outer_idx = blockIdx.x;
    const int64_t inner_idx = threadIdx.x;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
    const int64_t stride = inner_size;
    
    // Sequential cumulative product for each position
    scalar_t prod = static_cast<scalar_t>(1.0);
    for (int64_t d = 0; d < dim_size; ++d) {
        scalar_t val = input[base_offset + d * stride];
        prod *= val;
        output[base_offset + d * stride] = prod;
    }
}

torch::Tensor cumprod_cuda_forward(torch::Tensor input, int64_t dim) {
    // Validate input
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "Invalid dimension");
    
    // Get tensor dimensions
    auto sizes = input.sizes();
    int64_t dim_size = sizes[dim];
    
    // Calculate outer and inner dimensions
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= sizes[i];
    }
    
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < input.dim(); ++i) {
        inner_size *= sizes[i];
    }
    
    // Create output tensor
    auto output = torch::empty_like(input);
    
    // Launch kernel
    dim3 block(inner_size);
    dim3 grid(outer_size);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "cumprod_cuda_forward", ([&] {
        cumprod_forward_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size
        );
    }));
    
    return output;
}
"""

cumprod_cpp_source = """
torch::Tensor cumprod_cuda_forward(torch::Tensor input, int64_t dim);
"""

# Compile the inline CUDA code
cumprod_cuda = load_inline(
    name="cumprod_cuda",
    cpp_sources=cumprod_cpp_source,
    cuda_sources=cumprod_cuda_source,
    functions=["cumprod_cuda_forward"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    extra_cuda_cflags=["--use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model that performs cumulative product operation using custom CUDA kernel.
    
    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim):
        """
        Initialize the optimized CumulativeProductModel.
        
        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass using custom CUDA kernel for cumulative product.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            
        Returns:
            torch::Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        return cumprod_cuda.cumprod_cuda_forward(x, self.dim)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for cumulative product
cumprod_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void cumprod_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size) {
    
    const int64_t outer_idx = blockIdx.x;
    const int64_t inner_idx = threadIdx.x;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    const int64_t base_offset = outer_idx * dim_size / inner_size + inner_idx;
    const int64_t stride = inner_size;
    
    // Sequential cumulative product for each position
    scalar_t prod = static_cast<scalar_t>(1.0);
    for (int64_t d = 0; d < dim_size; ++d) {
        scalar_t val = input[base_offset + d * stride];
        prod *= val;
        output[base_offset + d * stride] = prod;
    }
}

torch::Tensor cumprod_cuda_forward(torch::Tensor input, int64_t dim) {
    // Validate input
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "Invalid dimension");
    
    // Get tensor dimensions
    auto sizes = input.sizes();
    int64_t dim_size = sizes[dim];
    
    // Calculate outer and inner dimensions
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= sizes[i];
    }
    
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < input.dim(); ++i) {
        inner_size *= sizes[i];
    }
    
    // Create output tensor
    auto output = torch::empty_like(input);
    
    // Launch kernel
    dim3 block(inner_size);
    dim3 grid(outer_size);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "cumprod_cuda_forward", ([&] {
        cumprod_forward_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size
        );
    }));
    
    return output;
}
"""

cumprod_cpp_source = """
torch::Tensor cumprod_cuda_forward(torch::Tensor input, int64_t dim);
"""

# Compile the inline CUDA code
cumprod_cuda = load_inline(
    name="cumprod_cuda",
    cpp_sources=cumprod_cpp_source,
    cuda_sources=cumprod_cuda_source,
    functions=["cumprod_cuda_forward"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    extra_cuda_cflags=["--use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model that performs cumulative product operation using custom CUDA kernel.
    
    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim):
        """
        Initialize the optimized CumulativeProductModel.
        
        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass using custom CUDA kernel for cumulative product.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            
        Returns:
            torch::Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        return cumprod_cuda.cumprod_cuda_forward(x, self.dim)
```

### Part 3: 杀死该变异体的输入

**杀死输入代码 (Round 3)：**

```python
import torch

def generate_inputs(device):
    # Shape: [outer_size, dim_size, inner_size] = [3, 5, 4]
    # This corresponds to cumprod along dim=1
    torch.manual_seed(123)
    input_tensor = torch.rand(3, 5, 4, device=device) + 0.5  # Values in (0.5, 1.5) to avoid zeros
    # Make values distinct for better error detection
    input_tensor = input_tensor * torch.arange(1, 3*5*4+1, device=device).view(3,5,4).float()
    return [input_tensor]

```

**输出差异**: `max_diff=3.669397e+05, mean_diff=7.378583e+03, ref_range=[5.6695e-01,8.0590e+08], mut_range=[5.6695e-01,8.0590e+08]`

### Part 4: 大模型诊断流程

共 3 轮迭代分析：

#### Round 1 (未杀死)

**LLM 存活原因分析：**

> The mutation survived because the stress tests only varied input values while keeping the input shape fixed at [128, 4000]. With this shape and dim=1 (implied by the kernel structure), inner_size = 1. When inner_size = 1, the mutated expression 'outer_idx * dim_size / inner_size + inner_idx' simplifies to 'outer_idx * dim_size + inner_idx', which is mathematically identical to the original 'outer_idx * dim_size * inner_size + inner_idx' when inner_size = 1 (since multiplying or dividing by 1 yields the same result). Thus, the mutant is functionally equivalent to the original for all value distributions when inner_size = 1.

**LLM 杀死策略：**

> Change the input shape so that inner_size ≠ 1. The mutant's division by inner_size (instead of multiplication) will cause incorrect base_offset calculation when inner_size > 1. Specifically, when inner_size does not evenly divide (outer_idx * dim_size), integer division truncation will produce wrong offsets, leading to memory access misalignment and incorrect cumulative products.

**LLM 建议的输入代码：**

```python
import torch

def generate_inputs(device):
    # Create a 3D tensor with shape [outer, dim, inner] = [2, 3, 4]
    # This corresponds to dim=1 cumulative product
    # Fill with small positive integers for predictable products
    input_tensor = torch.arange(1, 2*3*4+1, dtype=torch.float32, device=device).view(2, 3, 4)
    return [input_tensor]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `True`

#### Round 2 (未杀死)

**LLM 存活原因分析：**

> The mutant survives because the division by inner_size produces the same result as multiplication when inner_size=1. Previous attempts used inner_size>1, but the mutation testing tool appears to be applying the test incorrectly or the host code is always setting inner_size=1. Since forward() takes a 2D tensor [128, 4000], the kernel's 3D representation likely sets inner_size=1 when dim=1, making division by 1 identical to multiplication by 1. The division would only differ if inner_size doesn't divide (outer_idx * dim_size) evenly or inner_size≠1, but the host code seems to ensure inner_size=1 for this 2D case.

**LLM 杀死策略：**

> Test with the same shape but different dimension for cumulative product (dim=0) so that inner_size becomes the second dimension (4000). This makes the division by inner_size≠1, causing integer truncation and wrong memory access patterns.

**LLM 建议的输入代码：**

```python
import torch
def generate_inputs(device):
    # Same shape [128, 4000] but we'll compute cumprod along dim=0
    # Create random positive values to avoid zero products masking errors
    torch.manual_seed(42)
    input_tensor = torch.rand(128, 4000, device=device) + 0.1  # Values in (0.1, 1.1)
    return [input_tensor]

```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `True`

#### Round 3 (**KILLED**)

**LLM 存活原因分析：**

> The mutant survives because the previous tests either used inner_size=1 (making the division equivalent to multiplication) or used shapes where the integer division truncation coincidentally produced correct or undetected outputs due to race conditions or overlapping memory accesses. In Round 1, the 3D tensor with shape [2,3,4] likely caused overlapping writes and race conditions in the mutant, but the test may have only validated partial output or the non-deterministic results accidentally matched expected values. In Round 2, using dim=0 with shape [128,4000] set inner_size=4000 but outer_size=1, so the base_offset calculation was identical for the only outer_idx=0, making the mutant behave identically to the original. Thus, the mutant evaded detection because the tests didn't trigger a deterministic, observable difference in output.

**LLM 杀死策略：**

> Design a test where outer_size > 1, inner_size > 1, and the integer division in the mutant causes a non-overlapping but incorrect memory access pattern that leads to deterministic wrong outputs. Specifically, choose a shape where inner_size does not divide outer_idx * dim_size, causing base_offset truncation, and ensure each thread accesses distinct memory locations to avoid race conditions. Use distinct positive values so errors are easily detectable.

**LLM 建议的输入代码：**

```python
import torch

def generate_inputs(device):
    # Shape: [outer_size, dim_size, inner_size] = [3, 5, 4]
    # This corresponds to cumprod along dim=1
    torch.manual_seed(123)
    input_tensor = torch.rand(3, 5, 4, device=device) + 0.5  # Values in (0.5, 1.5) to avoid zeros
    # Make values distinct for better error detection
    input_tensor = input_tensor * torch.arange(1, 3*5*4+1, device=device).view(3,5,4).float()
    return [input_tensor]

```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=3.669397e+05, mean_diff=7.378583e+03, ref_range=[5.6695e-01,8.0590e+08], mut_range=[5.6695e-01,8.0590e+08]`

#### 测试构造规则

- **规则名**: `non_trivial_inner_outer_dimensions`
- **描述**: Generates inputs where both inner_size > 1 and outer_size > 1 to expose integer division truncation errors in index calculations. This is particularly effective for kernels that compute offsets using expressions like (outer_idx * dim_size) / inner_size, where replacing multiplication with division causes different truncation behavior when inner_size doesn't evenly divide the product.
- **适用算子**: arith_replace, binary_operator_replacement, arithmetic_mutation

```python
def policy(shape, dtype, device, rng):
    import torch
    # Ensure tensor has at least 3 dimensions for outer/dim/inner structure
    if len(shape) < 3:
        # For 1D/2D tensors, pad to 3D with size > 1 in all dimensions
        padded_shape = [3, 5, 4]
        # But respect original dimensionality if caller expects specific shape
        # Return original shape with values that would still trigger the issue
        # Use the killing example's approach: distinct positive values
        tensor = torch.rand(shape, device=device, generator=rng) + 0.5
        # Make values distinct by multiplying with indices
        indices = torch.arange(1, torch.prod(torch.tensor(shape)).item() + 1,
                              device=device).view(shape).to(dtype)
        return tensor * indices
    
    # For 3D+ tensors, ensure middle dimensions have size > 1
    # The cumulative dimension (dim) should be > 1
    # For a 3D tensor, shape = [outer, dim, inner]
    # We want outer > 1 and inner > 1
    if shape[0] == 1:
        # Increase outer dimension if it's 1
        new_shape = list(shape)
        new_shape[0] = 3
        tensor = torch.rand(new_shape, device=device, generator=rng) + 0.5
    elif shape[-1] == 1:
        # Increase inner dimension if it's 1
        new_shape = list(shape)
        new_shape[-1] = 4
        tensor = torch.rand(new_shape, device=device, generator=rng) + 0.5
    else:
        # Original shape already has outer>1 and inner>1
        tensor = torch.rand(shape, device=device, generator=rng) + 0.5
    
    # Make values distinct and positive to avoid zero masking and improve error detection
    indices = torch.arange(1, torch.prod(torch.tensor(tensor.shape)).item() + 1,
                          device=device).view(tensor.shape).to(dtype)
    return tensor * indices
```


---

## 19. `L1_P90__arith_replace__6`

- **Kernel**: `L1_P90`
- **算子**: `arith_replace` (Category A)
- **描述**: Replace binary arithmetic operators (+→-, -→+, *→/, /→*) @ L31
- **变异行**: Line 31, 原始片段 `*`, 节点类型 `cuda_Mult`
- **杀死轮次**: Round 1

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for cumulative product
cumprod_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void cumprod_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size) {
    
    const int64_t outer_idx = blockIdx.x;
    const int64_t inner_idx = threadIdx.x;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
    const int64_t stride = inner_size;
    
    // Sequential cumulative product for each position
    scalar_t prod = static_cast<scalar_t>(1.0);
    for (int64_t d = 0; d < dim_size; ++d) {
        scalar_t val = input[base_offset + d * stride];
        prod *= val;
        output[base_offset + d * stride] = prod;
    }
}

torch::Tensor cumprod_cuda_forward(torch::Tensor input, int64_t dim) {
    // Validate input
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "Invalid dimension");
    
    // Get tensor dimensions
    auto sizes = input.sizes();
    int64_t dim_size = sizes[dim];
    
    // Calculate outer and inner dimensions
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= sizes[i];
    }
    
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < input.dim(); ++i) {
        inner_size *= sizes[i];
    }
    
    // Create output tensor
    auto output = torch::empty_like(input);
    
    // Launch kernel
    dim3 block(inner_size);
    dim3 grid(outer_size);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "cumprod_cuda_forward", ([&] {
        cumprod_forward_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size
        );
    }));
    
    return output;
}
"""

cumprod_cpp_source = """
torch::Tensor cumprod_cuda_forward(torch::Tensor input, int64_t dim);
"""

# Compile the inline CUDA code
cumprod_cuda = load_inline(
    name="cumprod_cuda",
    cpp_sources=cumprod_cpp_source,
    cuda_sources=cumprod_cuda_source,
    functions=["cumprod_cuda_forward"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    extra_cuda_cflags=["--use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model that performs cumulative product operation using custom CUDA kernel.
    
    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim):
        """
        Initialize the optimized CumulativeProductModel.
        
        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass using custom CUDA kernel for cumulative product.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            
        Returns:
            torch::Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        return cumprod_cuda.cumprod_cuda_forward(x, self.dim)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for cumulative product
cumprod_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void cumprod_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size) {
    
    const int64_t outer_idx = blockIdx.x;
    const int64_t inner_idx = threadIdx.x;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
    const int64_t stride = inner_size;
    
    // Sequential cumulative product for each position
    scalar_t prod = static_cast<scalar_t>(1.0);
    for (int64_t d = 0; d < dim_size; ++d) {
        scalar_t val = input[base_offset + d * stride];
        prod *= val;
        output[base_offset + d / stride] = prod;
    }
}

torch::Tensor cumprod_cuda_forward(torch::Tensor input, int64_t dim) {
    // Validate input
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "Invalid dimension");
    
    // Get tensor dimensions
    auto sizes = input.sizes();
    int64_t dim_size = sizes[dim];
    
    // Calculate outer and inner dimensions
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= sizes[i];
    }
    
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < input.dim(); ++i) {
        inner_size *= sizes[i];
    }
    
    // Create output tensor
    auto output = torch::empty_like(input);
    
    // Launch kernel
    dim3 block(inner_size);
    dim3 grid(outer_size);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "cumprod_cuda_forward", ([&] {
        cumprod_forward_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size
        );
    }));
    
    return output;
}
"""

cumprod_cpp_source = """
torch::Tensor cumprod_cuda_forward(torch::Tensor input, int64_t dim);
"""

# Compile the inline CUDA code
cumprod_cuda = load_inline(
    name="cumprod_cuda",
    cpp_sources=cumprod_cpp_source,
    cuda_sources=cumprod_cuda_source,
    functions=["cumprod_cuda_forward"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    extra_cuda_cflags=["--use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model that performs cumulative product operation using custom CUDA kernel.
    
    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim):
        """
        Initialize the optimized CumulativeProductModel.
        
        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass using custom CUDA kernel for cumulative product.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            
        Returns:
            torch::Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        return cumprod_cuda.cumprod_cuda_forward(x, self.dim)
```

### Part 3: 杀死该变异体的输入

**杀死输入代码 (Round 1)：**

```python
import torch

def generate_inputs(device):
    # Create a 3D tensor where cumprod along dim=1 (middle dimension)
    # Original kernel likely flattens to [outer=2, dim=3, inner=4]
    # This makes inner_size=4, stride=4, so d*stride vs d/stride differ
    shape = [2, 3, 4]
    # Use simple increasing values to easily detect differences
    data = torch.arange(1, 25, dtype=torch.float32, device=device).view(shape)
    return [data]

```

**输出差异**: `max_diff=7.664000e+03, mean_diff=1.047167e+03, ref_range=[1.0000e+00,7.6800e+03], mut_range=[5.0000e+00,7.6800e+03]`

### Part 4: 大模型诊断流程

共 1 轮迭代分析：

#### Round 1 (**KILLED**)

**LLM 存活原因分析：**

> The mutation changes the output index calculation from `d * stride` to `d / stride`. With the fixed input shape [128, 4000] and assuming the kernel is processing dimension 1 (the 4000 dimension), we have `inner_size = 1` (since there are no dimensions after the target dimension when using typical CUDA kernel flattening). When `inner_size = 1`, then `stride = 1`, making both `d * 1 = d` and `d / 1 = d` equivalent. Therefore, the mutated index calculation produces identical memory accesses to the original, making the mutant behaviorally equivalent for this specific shape/dimension configuration.

**LLM 杀死策略：**

> To kill this mutant, we must change the input shape such that `inner_size > 1`. This would make `stride > 1`, causing `d / stride` to produce different indices than `d * stride`. Specifically, we need a tensor shape where the dimension being cumproduced is NOT the innermost dimension in the kernel's flattening scheme. A 3D tensor with cumprod on the middle dimension would achieve this.

**LLM 建议的输入代码：**

```python
import torch

def generate_inputs(device):
    # Create a 3D tensor where cumprod along dim=1 (middle dimension)
    # Original kernel likely flattens to [outer=2, dim=3, inner=4]
    # This makes inner_size=4, stride=4, so d*stride vs d/stride differ
    shape = [2, 3, 4]
    # Use simple increasing values to easily detect differences
    data = torch.arange(1, 25, dtype=torch.float32, device=device).view(shape)
    return [data]

```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=7.664000e+03, mean_diff=1.047167e+03, ref_range=[1.0000e+00,7.6800e+03], mut_range=[5.0000e+00,7.6800e+03]`

#### 测试构造规则

- **规则名**: `stride_amplification_for_index_calc`
- **描述**: Generates tensors where the stride (inner_size) in dimension-indexing calculations is deliberately set to a value >1, making index calculations sensitive to arithmetic mutations in stride multiplication/division. This exposes bugs where arithmetic operators in index computations are mutated, particularly when the mutated operation would cause repeated writes (division) versus strided writes (multiplication).
- **适用算子**: arith_replace, binary_operator_replace, constant_replace

```python
def policy(shape, dtype, device, rng):
    # Construct a multi-dimensional tensor where:
    # - The target dimension (for operations like cumprod) is not the last dimension
    # - The inner_size (product of dimensions after target dim) is >1
    # This ensures stride != 1 in index calculations
    
    # We'll create a 3D tensor by default, as it's sufficient to expose the issue
    # but can be generalized to higher dimensions
    outer = rng.randint(2, 5)          # dimensions before target
    target_dim_size = rng.randint(3, 6) # target dimension size
    inner = rng.randint(2, 5)          # dimensions after target (must be >1)
    
    # Optional: with 25% probability, create 4D tensor for additional coverage
    if rng.random() < 0.25:
        outer2 = rng.randint(2, 4)
        shape = (outer2, outer, target_dim_size, inner)
    else:
        shape = (outer, target_dim_size, inner)
    
    # Fill with distinct, easily traceable values
    total_elements = torch.prod(torch.tensor(shape)).item()
    data = torch.arange(1, total_elements + 1, dtype=dtype, device=device).view(shape)
    
    # Optional: add small random perturbations to avoid perfect regularity
    # while maintaining distinctness
    if dtype.is_floating_point:
        noise = torch.randn_like(data) * 0.01
        data = data + noise
    
    return data
```


---

## 20. `L1_P91__mask_boundary__2`

- **Kernel**: `L1_P91`
- **算子**: `mask_boundary` (Category B)
- **描述**: Weaken or tighten boundary checks in mask/tl.where (Triton) or thread/block guards (CUDA) @ L22
- **变异行**: Line 22, 原始片段 `inner_idx >= inner_size`, 节点类型 `ge>`
- **杀死轮次**: Round 1

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized reverse cumulative sum
reverse_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void reverse_cumsum_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    // Each thread handles one inner element across the entire dimension
    const int64_t outer_idx = blockIdx.y;
    const int64_t inner_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
    
    // Direct computation without shared memory overhead
    scalar_t sum = 0;
    
    // Process in reverse order for cumulative sum
    for (int64_t dim_idx = dim_size - 1; dim_idx >= 0; --dim_idx) {
        const int64_t input_offset = base_offset + dim_idx * inner_size;
        sum += input[input_offset];
        output[input_offset] = sum;
    }
}

template<typename scalar_t>
__global__ void reverse_cumsum_kernel_unrolled(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    const int64_t outer_idx = blockIdx.y;
    const int64_t inner_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
    
    scalar_t sum = 0;
    int64_t dim_idx = dim_size - 1;
    
    // Unroll by 4 for better instruction-level parallelism
    for (; dim_idx >= 3; dim_idx -= 4) {
        const int64_t offset0 = base_offset + dim_idx * inner_size;
        const int64_t offset1 = base_offset + (dim_idx - 1) * inner_size;
        const int64_t offset2 = base_offset + (dim_idx - 2) * inner_size;
        const int64_t offset3 = base_offset + (dim_idx - 3) * inner_size;
        
        const scalar_t val0 = input[offset0];
        const scalar_t val1 = input[offset1];
        const scalar_t val2 = input[offset2];
        const scalar_t val3 = input[offset3];
        
        sum += val0;
        output[offset0] = sum;
        
        sum += val1;
        output[offset1] = sum;
        
        sum += val2;
        output[offset2] = sum;
        
        sum += val3;
        output[offset3] = sum;
    }
    
    // Handle remaining elements
    for (; dim_idx >= 0; --dim_idx) {
        const int64_t offset = base_offset + dim_idx * inner_size;
        sum += input[offset];
        output[offset] = sum;
    }
}

torch::Tensor reverse_cumsum_cuda_optimized(torch::Tensor input, int64_t dim) {
    auto input_contiguous = input.contiguous();
    auto output = torch::empty_like(input_contiguous);
    
    const int64_t ndim = input_contiguous.dim();
    dim = dim < 0 ? ndim + dim : dim;
    
    // Calculate sizes
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= input_contiguous.size(i);
    }
    
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) {
        inner_size *= input_contiguous.size(i);
    }
    
    const int64_t dim_size = input_contiguous.size(dim);
    
    // Configure kernel launch
    const int block_size = 256;
    const int grid_x = (inner_size + block_size - 1) / block_size;
    const dim3 grid(grid_x, outer_size);
    
    AT_DISPATCH_FLOATING_TYPES(
        input_contiguous.scalar_type(),
        "reverse_cumsum_cuda_optimized",
        ([&] {
            if (dim_size >= 32) {
                // Use unrolled kernel for better performance
                reverse_cumsum_kernel_unrolled<scalar_t><<<grid, block_size>>>(
                    input_contiguous.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    outer_size,
                    dim_size,
                    inner_size
                );
            } else {
                // Use simple kernel for small dimensions
                reverse_cumsum_kernel<scalar_t><<<grid, block_size>>>(
                    input_contiguous.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    outer_size,
                    dim_size,
                    inner_size
                );
            }
        })
    );
    
    return output;
}
"""

reverse_cumsum_cpp_source = """
torch::Tensor reverse_cumsum_cuda_optimized(torch::Tensor input, int64_t dim);
"""

# Compile the inline CUDA code
reverse_cumsum_module = load_inline(
    name="reverse_cumsum_optimized",
    cpp_sources=reverse_cumsum_cpp_source,
    cuda_sources=reverse_cumsum_source,
    functions=["reverse_cumsum_cuda_optimized"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a reverse cumulative sum operation 
    using a streamlined custom CUDA kernel.
    
    Parameters:
        dim (int): The dimension along which to perform the reverse cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.reverse_cumsum = reverse_cumsum_module

    def forward(self, x):
        # Use optimized CUDA kernel
        return self.reverse_cumsum.reverse_cumsum_cuda_optimized(x, self.dim)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized reverse cumulative sum
reverse_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void reverse_cumsum_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    // Each thread handles one inner element across the entire dimension
    const int64_t outer_idx = blockIdx.y;
    const int64_t inner_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (outer_idx >= outer_size || inner_idx > inner_size) return;
    
    const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
    
    // Direct computation without shared memory overhead
    scalar_t sum = 0;
    
    // Process in reverse order for cumulative sum
    for (int64_t dim_idx = dim_size - 1; dim_idx >= 0; --dim_idx) {
        const int64_t input_offset = base_offset + dim_idx * inner_size;
        sum += input[input_offset];
        output[input_offset] = sum;
    }
}

template<typename scalar_t>
__global__ void reverse_cumsum_kernel_unrolled(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    const int64_t outer_idx = blockIdx.y;
    const int64_t inner_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
    
    scalar_t sum = 0;
    int64_t dim_idx = dim_size - 1;
    
    // Unroll by 4 for better instruction-level parallelism
    for (; dim_idx >= 3; dim_idx -= 4) {
        const int64_t offset0 = base_offset + dim_idx * inner_size;
        const int64_t offset1 = base_offset + (dim_idx - 1) * inner_size;
        const int64_t offset2 = base_offset + (dim_idx - 2) * inner_size;
        const int64_t offset3 = base_offset + (dim_idx - 3) * inner_size;
        
        const scalar_t val0 = input[offset0];
        const scalar_t val1 = input[offset1];
        const scalar_t val2 = input[offset2];
        const scalar_t val3 = input[offset3];
        
        sum += val0;
        output[offset0] = sum;
        
        sum += val1;
        output[offset1] = sum;
        
        sum += val2;
        output[offset2] = sum;
        
        sum += val3;
        output[offset3] = sum;
    }
    
    // Handle remaining elements
    for (; dim_idx >= 0; --dim_idx) {
        const int64_t offset = base_offset + dim_idx * inner_size;
        sum += input[offset];
        output[offset] = sum;
    }
}

torch::Tensor reverse_cumsum_cuda_optimized(torch::Tensor input, int64_t dim) {
    auto input_contiguous = input.contiguous();
    auto output = torch::empty_like(input_contiguous);
    
    const int64_t ndim = input_contiguous.dim();
    dim = dim < 0 ? ndim + dim : dim;
    
    // Calculate sizes
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= input_contiguous.size(i);
    }
    
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) {
        inner_size *= input_contiguous.size(i);
    }
    
    const int64_t dim_size = input_contiguous.size(dim);
    
    // Configure kernel launch
    const int block_size = 256;
    const int grid_x = (inner_size + block_size - 1) / block_size;
    const dim3 grid(grid_x, outer_size);
    
    AT_DISPATCH_FLOATING_TYPES(
        input_contiguous.scalar_type(),
        "reverse_cumsum_cuda_optimized",
        ([&] {
            if (dim_size >= 32) {
                // Use unrolled kernel for better performance
                reverse_cumsum_kernel_unrolled<scalar_t><<<grid, block_size>>>(
                    input_contiguous.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    outer_size,
                    dim_size,
                    inner_size
                );
            } else {
                // Use simple kernel for small dimensions
                reverse_cumsum_kernel<scalar_t><<<grid, block_size>>>(
                    input_contiguous.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    outer_size,
                    dim_size,
                    inner_size
                );
            }
        })
    );
    
    return output;
}
"""

reverse_cumsum_cpp_source = """
torch::Tensor reverse_cumsum_cuda_optimized(torch::Tensor input, int64_t dim);
"""

# Compile the inline CUDA code
reverse_cumsum_module = load_inline(
    name="reverse_cumsum_optimized",
    cpp_sources=reverse_cumsum_cpp_source,
    cuda_sources=reverse_cumsum_source,
    functions=["reverse_cumsum_cuda_optimized"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a reverse cumulative sum operation 
    using a streamlined custom CUDA kernel.
    
    Parameters:
        dim (int): The dimension along which to perform the reverse cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.reverse_cumsum = reverse_cumsum_module

    def forward(self, x):
        # Use optimized CUDA kernel
        return self.reverse_cumsum.reverse_cumsum_cuda_optimized(x, self.dim)
```

### Part 3: 杀死该变异体的输入

**杀死输入代码 (Round 1)：**

```python
import torch

def generate_inputs(device):
    # Shape: [outer_size, dim_size, inner_size] = [2, 2, 2]
    # Values chosen so that the extra thread (inner_idx=2) reads 5 and 3,
    # writes 5 to index 4 (which the valid thread would overwrite with 12).
    # The original kernel never runs the extra thread, so index 4 becomes 12.
    # The mutant may write 5 or 12 depending on race, exposing a difference.
    data = [
        [[1.0, 2.0], [3.0, 4.0]],  # outer_idx = 0
        [[5.0, 6.0], [7.0, 8.0]]   # outer_idx = 1
    ]
    input_tensor = torch.tensor(data, dtype=torch.float32, device=device)
    return [input_tensor]

```

**输出差异**: `max_diff=5.000000e+00, mean_diff=6.250000e-01, ref_range=[3.0000e+00,1.4000e+01], mut_range=[4.0000e+00,1.4000e+01]`

### Part 4: 大模型诊断流程

共 1 轮迭代分析：

#### Round 1 (**KILLED**)

**LLM 存活原因分析：**

> The mutant changes the boundary condition from `inner_idx >= inner_size` to `inner_idx > inner_size`. In the original test suite, the kernel is always called with `inner_size = 1` (since the input is 2D shape [128, 4000], interpreted as [outer_size=128, dim_size=4000, inner_size=1]). The launch configuration is set to exactly `inner_size` threads, so `inner_idx` only takes values 0..0, never reaching the boundary. Thus, the mutated condition behaves identically because no thread ever satisfies `inner_idx == inner_size` (which would be 1) to expose the difference. The stress tests only varied input values, not shapes, so the mutant survived.

**LLM 杀死策略：**

> To kill the mutant, we must use a 3D tensor where `inner_size > 1` and a launch configuration that creates extra threads (i.e., total threads > inner_size). This allows a thread with `inner_idx == inner_size` to execute. Such a thread will compute out-of-bounds memory accesses, potentially corrupting adjacent data in the tensor. By carefully choosing values, we can make the out-of-bounds writes affect visible outputs, causing a divergence from the original kernel.

**LLM 建议的输入代码：**

```python
import torch

def generate_inputs(device):
    # Shape: [outer_size, dim_size, inner_size] = [2, 2, 2]
    # Values chosen so that the extra thread (inner_idx=2) reads 5 and 3,
    # writes 5 to index 4 (which the valid thread would overwrite with 12).
    # The original kernel never runs the extra thread, so index 4 becomes 12.
    # The mutant may write 5 or 12 depending on race, exposing a difference.
    data = [
        [[1.0, 2.0], [3.0, 4.0]],  # outer_idx = 0
        [[5.0, 6.0], [7.0, 8.0]]   # outer_idx = 1
    ]
    input_tensor = torch.tensor(data, dtype=torch.float32, device=device)
    return [input_tensor]

```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=5.000000e+00, mean_diff=6.250000e-01, ref_range=[3.0000e+00,1.4000e+01], mut_range=[4.0000e+00,1.4000e+01]`

#### 测试构造规则

- **规则名**: `boundary_expansion_3d_input`
- **描述**: Generates 3D tensors where the innermost dimension is >1 and launch configuration may create extra threads. This exposes boundary condition mutations by ensuring threads with indices at or beyond the original boundary execute, revealing differences in masking logic. Values are set to be distinct to detect out-of-bounds writes or race conditions.
- **适用算子**: mask_boundary, mask_expansion, guard_removal

```python
def policy(shape, dtype, device, rng):
    # Ensure we have at least 3 dimensions for this rule
    if len(shape) < 3:
        # If original shape has fewer dims, expand to 3D with inner_size=2
        # Outer size = max(2, shape[0] if exists else 2)
        # Dim size = shape[1] if exists else 2
        # Inner size = 2 (critical for boundary exposure)
        outer = shape[0] if len(shape) > 0 else 2
        dim = shape[1] if len(shape) > 1 else 2
        inner = 2  # Force inner dimension >1
        shape = (max(2, outer), max(2, dim), inner)
    else:
        # Keep outer and middle dimensions but ensure inner dimension >= 2
        shape = (max(2, shape[0]), max(2, shape[1]), max(2, shape[2]))
    
    # Generate distinct values to detect incorrect writes
    total_elements = shape[0] * shape[1] * shape[2]
    # Create a sequence of distinct values
    if dtype.is_floating_point:
        # For floating point, use incrementing sequence
        data = torch.arange(total_elements, dtype=dtype, device=device).view(shape)
    else:
        # For integer types, use modulo to stay in type range
        max_val = min(torch.iinfo(dtype).max, 1000) if hasattr(torch, 'iinfo') else 100
        data = torch.randint(0, max_val, (total_elements,), dtype=dtype, device=device, generator=rng)
        data = data.view(shape)
    
    return data
```


---

## 21. `L1_P93__arith_replace__4`

- **Kernel**: `L1_P93`
- **算子**: `arith_replace` (Category A)
- **描述**: Replace binary arithmetic operators (+→-, -→+, *→/, /→*) @ L32
- **变异行**: Line 32, 原始片段 `+`, 节点类型 `cuda_Add`
- **杀死轮次**: Round 1

### Part 1: 未变异的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized masked cumulative sum
masked_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void masked_cumsum_kernel(
    const scalar_t* __restrict__ input,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    // Each block handles one outer position
    const int64_t outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;
    
    // Each thread handles multiple inner positions with vectorized loads
    const int64_t base_offset = outer_idx * dim_size * inner_size;
    
    // Process inner positions in parallel
    for (int64_t inner_idx = threadIdx.x; inner_idx < inner_size; inner_idx += blockDim.x) {
        scalar_t cum_sum = 0;
        
        // Sequential scan along the cumulative dimension
        for (int64_t dim_idx = 0; dim_idx < dim_size; dim_idx++) {
            const int64_t offset = base_offset + dim_idx * inner_size + inner_idx;
            if (mask[offset]) {
                cum_sum += input[offset];
            }
            output[offset] = cum_sum;
        }
    }
}

torch::Tensor masked_cumsum_cuda(torch::Tensor input, torch::Tensor mask, int64_t dim) {
    // Validate inputs
    TORCH_CHECK(input.sizes() == mask.sizes(), "Input and mask must have same shape");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "Mask must be boolean tensor");
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "Dimension out of range");
    
    auto output = torch::empty_like(input);
    
    // Get tensor dimensions
    auto sizes = input.sizes();
    
    // Calculate dimensions for kernel launch
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    
    int64_t dim_size = sizes[dim];
    
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < input.dim(); i++) {
        inner_size *= sizes[i];
    }
    
    // Launch kernel with optimized configuration
    // Use maximum threads per block for better occupancy
    const int block_size = 256;
    dim3 grid_size(outer_size);
    dim3 block_size_dim(min(block_size, (int)inner_size));
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "masked_cumsum_kernel",
        [&] {
            masked_cumsum_kernel<scalar_t><<<grid_size, block_size_dim>>>(
                input.data_ptr<scalar_t>(),
                mask.data_ptr<bool>(),
                output.data_ptr<scalar_t>(),
                outer_size,
                dim_size,
                inner_size
            );
        }
    );
    
    return output;
}
"""

masked_cumsum_cpp_source = """
torch::Tensor masked_cumsum_cuda(torch::Tensor input, torch::Tensor mask, int64_t dim);
"""

# Compile the inline CUDA code
masked_cumsum = load_inline(
    name="masked_cumsum",
    cpp_sources=masked_cumsum_cpp_source,
    cuda_sources=masked_cumsum_source,
    functions=["masked_cumsum_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a masked cumulative sum using custom CUDA kernel.
    
    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """
    
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.masked_cumsum = masked_cumsum
    
    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            mask (torch.Tensor): Boolean mask of the same shape as x.
            
        Returns:
            torch.Tensor: Cumulative sum of elements where mask is True.
        """
        return self.masked_cumsum.masked_cumsum_cuda(x, mask, self.dim)
```

### Part 2: 变异后的完整源代码

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized masked cumulative sum
masked_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void masked_cumsum_kernel(
    const scalar_t* __restrict__ input,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    // Each block handles one outer position
    const int64_t outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;
    
    // Each thread handles multiple inner positions with vectorized loads
    const int64_t base_offset = outer_idx * dim_size * inner_size;
    
    // Process inner positions in parallel
    for (int64_t inner_idx = threadIdx.x; inner_idx < inner_size; inner_idx += blockDim.x) {
        scalar_t cum_sum = 0;
        
        // Sequential scan along the cumulative dimension
        for (int64_t dim_idx = 0; dim_idx < dim_size; dim_idx++) {
            const int64_t offset = base_offset + dim_idx * inner_size - inner_idx;
            if (mask[offset]) {
                cum_sum += input[offset];
            }
            output[offset] = cum_sum;
        }
    }
}

torch::Tensor masked_cumsum_cuda(torch::Tensor input, torch::Tensor mask, int64_t dim) {
    // Validate inputs
    TORCH_CHECK(input.sizes() == mask.sizes(), "Input and mask must have same shape");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "Mask must be boolean tensor");
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "Dimension out of range");
    
    auto output = torch::empty_like(input);
    
    // Get tensor dimensions
    auto sizes = input.sizes();
    
    // Calculate dimensions for kernel launch
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    
    int64_t dim_size = sizes[dim];
    
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < input.dim(); i++) {
        inner_size *= sizes[i];
    }
    
    // Launch kernel with optimized configuration
    // Use maximum threads per block for better occupancy
    const int block_size = 256;
    dim3 grid_size(outer_size);
    dim3 block_size_dim(min(block_size, (int)inner_size));
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "masked_cumsum_kernel",
        [&] {
            masked_cumsum_kernel<scalar_t><<<grid_size, block_size_dim>>>(
                input.data_ptr<scalar_t>(),
                mask.data_ptr<bool>(),
                output.data_ptr<scalar_t>(),
                outer_size,
                dim_size,
                inner_size
            );
        }
    );
    
    return output;
}
"""

masked_cumsum_cpp_source = """
torch::Tensor masked_cumsum_cuda(torch::Tensor input, torch::Tensor mask, int64_t dim);
"""

# Compile the inline CUDA code
masked_cumsum = load_inline(
    name="masked_cumsum",
    cpp_sources=masked_cumsum_cpp_source,
    cuda_sources=masked_cumsum_source,
    functions=["masked_cumsum_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a masked cumulative sum using custom CUDA kernel.
    
    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """
    
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.masked_cumsum = masked_cumsum
    
    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            mask (torch.Tensor): Boolean mask of the same shape as x.
            
        Returns:
            torch.Tensor: Cumulative sum of elements where mask is True.
        """
        return self.masked_cumsum.masked_cumsum_cuda(x, mask, self.dim)
```

### Part 3: 杀死该变异体的输入

**杀死输入代码 (Round 1)：**

```python
import torch
import math

def generate_inputs(device):
    # Create 3D tensor: shape = [outer, dim, inner] = [4, 5, 3]
    # This ensures inner_size = 3 > 1, so inner_idx varies across threads
    outer, dim, inner = 4, 5, 3
    
    # Input tensor with structured pattern: values = 100*outer + 10*dim + inner
    # Makes each element unique and predictable
    input_tensor = torch.zeros(outer, dim, inner, device=device, dtype=torch.float32)
    for o in range(outer):
        for d in range(dim):
            for i in range(inner):
                input_tensor[o, d, i] = 100*o + 10*d + i
    
    # Mask with alternating pattern to test conditional accumulation
    mask = torch.ones(outer, dim, inner, device=device, dtype=torch.bool)
    # Set every other element along dim to False
    for d in range(dim):
        if d % 2 == 1:
            mask[:, d, :] = False
    
    return [input_tensor, mask]
```

**输出差异**: `max_diff=4.240000e+02, mean_diff=7.890000e+01, ref_range=[0.0000e+00,9.6600e+02], mut_range=[0.0000e+00,9.6600e+02]`

### Part 4: 大模型诊断流程

共 1 轮迭代分析：

#### Round 1 (**KILLED**)

**LLM 存活原因分析：**

> The mutant survives because the input tensor shape has inner_size=1 (from shape [128, 4000] with dim=1). When inner_size=1, inner_idx is always 0 for all threads (threadIdx.x=0 is the only thread accessing inner_idx=0 since threadIdx.x >= inner_size causes early exit). With inner_idx=0, the mutation `+ inner_idx` vs `- inner_idx` yields identical offsets (0 added vs 0 subtracted). The kernel's parallelization pattern makes inner_idx effectively constant at 0 for this specific shape, making the mutation behaviorally equivalent.

**LLM 杀死策略：**

> Use input tensors with inner_size > 1 to expose the mutation. This requires changing the tensor dimensions so that the cumulative dimension (dim) is not the innermost dimension. For a 2D tensor [outer, dim], inner_size=1 by definition. We need at least 3 dimensions: [outer, dim, inner] with inner > 1. The mutation will then cause threads to access wrong memory locations (potentially out-of-bounds or different elements), producing incorrect cumulative sums.

**LLM 建议的输入代码：**

```python
import torch
import math

def generate_inputs(device):
    # Create 3D tensor: shape = [outer, dim, inner] = [4, 5, 3]
    # This ensures inner_size = 3 > 1, so inner_idx varies across threads
    outer, dim, inner = 4, 5, 3
    
    # Input tensor with structured pattern: values = 100*outer + 10*dim + inner
    # Makes each element unique and predictable
    input_tensor = torch.zeros(outer, dim, inner, device=device, dtype=torch.float32)
    for o in range(outer):
        for d in range(dim):
            for i in range(inner):
                input_tensor[o, d, i] = 100*o + 10*d + i
    
    # Mask with alternating pattern to test conditional accumulation
    mask = torch.ones(outer, dim, inner, device=device, dtype=torch.bool)
    # Set every other element along dim to False
    for d in range(dim):
        if d % 2 == 1:
            mask[:, d, :] = False
    
    return [input_tensor, mask]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=4.240000e+02, mean_diff=7.890000e+01, ref_range=[0.0000e+00,9.6600e+02], mut_range=[0.0000e+00,9.6600e+02]`

#### 测试构造规则

- **规则名**: `inner_dimension_parallelization`
- **描述**: Targets kernels with inner-dimension parallelization patterns where indexing depends on threadIdx.x modulo inner_size. Ensures inner_size > 1 so that inner_idx varies across threads, exposing arithmetic mutations in index calculations that would be hidden when inner_size=1 (where inner_idx always equals 0). Uses structured values that depend on position to detect offset miscalculations.
- **适用算子**: arith_replace, const_replace, expr_replace, var_replace

```python
def policy(shape, dtype, device, rng):
    import torch
    import math
    
    # Determine if we should apply this rule based on shape characteristics
    # Only apply if we can create at least 3 dimensions with inner_size > 1
    if len(shape) < 3:
        # Reshape to add dimensions if needed, preserving total elements
        # Target shape: [outer, middle, inner] with inner > 1
        total_elements = math.prod(shape)
        # Find a factorization with inner > 1
        inner = 2  # Minimum inner dimension to expose the issue
        middle = 3  # Reasonable middle dimension
        outer = total_elements // (inner * middle)
        if outer == 0:
            # Fallback if tensor is too small
            outer = 1
            middle = total_elements // inner
            if middle == 0:
                # Very small tensor, use minimal 3D shape
                middle = 1
                inner = total_elements
        shape = (outer, middle, inner)
    else:
        # Already 3D+, ensure inner dimension > 1
        if shape[-1] == 1:
            # Reshape the last two dimensions to make inner > 1
            new_shape = list(shape)
            if len(shape) >= 2:
                # Combine last two dimensions then split with inner > 1
                combined = shape[-2] * shape[-1]
                # Find factors where inner > 1
                for inner in range(2, int(math.sqrt(combined)) + 1):
                    if combined % inner == 0:
                        new_shape[-2:] = [combined // inner, inner]
                        break
                else:
                    # Prime case, make inner dimension 2 by adjusting previous dim
                    if combined > 2:
                        new_shape[-2:] = [(combined + 1) // 2, 2]
                    else:
                        new_shape[-2:] = [1, 2]
            shape = tuple(new_shape)
    
    # Create tensor with structured pattern: value = 1000*outer + 100*middle + inner
    # This makes each position uniquely identifiable and detects offset errors
    tensor = torch.zeros(shape, device=device, dtype=dtype)
    
    if len(shape) == 3:
        outer, middle, inner = shape
        for o in range(outer):
            for m in range(middle):
                for i in range(inner):
                    # Scale factors ensure uniqueness across dimensions
                    tensor[o, m, i] = 1000 * o + 100 * m + i
    elif len(shape) == 4:
        # Handle 4D case similarly
        d1, d2, d3, d4 = shape
        for a in range(d1):
            for b in range(d2):
                for c in range(d3):
                    for d in range(d4):
                        tensor[a, b, c, d] = 10000 * a + 1000 * b + 100 * c + d
    else:
        # General n-dimensional case: linear index in row-major order
        idx = 0
        for coords in torch.ndindex(shape):
            tensor[coords] = idx
            idx += 1
    
    # For boolean masks (if dtype is bool), create alternating pattern
    if dtype == torch.bool:
        # Alternating pattern along the middle dimension
        tensor = torch.ones(shape, device=device, dtype=torch.bool)
        if len(shape) >= 2:
            middle_dim = -2 if len(shape) > 2 else -1
            for idx in range(shape[middle_dim]):
                if idx % 2 == 1:
                    if len(shape) == 3:
                        tensor[:, idx, :] = False
                    elif len(shape) == 2:
                        tensor[:, idx] = False
                    else:
                        # General case: set all elements where middle_dim = idx to False
                        slice_obj = [slice(None)] * len(shape)
                        slice_obj[middle_dim] = idx
                        tensor[tuple(slice_obj)] = False
    
    return tensor
```


---

## 22. `L1_P96__arith_replace__12`

- **Kernel**: `L1_P96`
- **算子**: `arith_replace` (Category A)
- **描述**: Replace binary arithmetic operators (+→-, -→+, *→/, /→*) @ L73
- **变异行**: Line 73, 原始片段 `+`, 节点类型 `cuda_Add`
- **杀死轮次**: Round 1

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
    const int num_blocks = (total_elements - block_size - 1) / block_size;
    
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

**杀死输入代码 (Round 1)：**

```python
import torch
import math

def generate_inputs(device):
    # Use total_elements = 256 (exactly block_size)
    # Original: (256 + 256 - 1)/256 = 511/256 = 1 (since integer division truncates)
    # Mutant: (256 - 256 - 1)/256 = (-1)/256 = 0 (truncates toward zero)
    shape = [1, 256]  # total_elements = 256
    predictions = torch.randn(shape, device=device, dtype=torch.float32)
    targets = torch.randn(shape, device=device, dtype=torch.float32)
    return [predictions, targets]
```

### Part 4: 大模型诊断流程

共 1 轮迭代分析：

#### Round 1 (**KILLED**)

**LLM 存活原因分析：**

> The mutant changes the grid dimension calculation from `(total_elements + block_size - 1) / block_size` to `(total_elements - block_size - 1) / block_size`. For the fixed input shape [128, 4096] with total_elements = 524,288 and block_size = 256, both formulas produce the same integer result: original = (524288 + 255)/256 = 2048, mutant = (524288 - 257)/256 = (524031)/256 = 2046 (integer division). Both evaluate to 2048 in integer division (524,543/256 = 2049 - epsilon, 524,031/256 = 2047 - epsilon; both truncate to 2048). Therefore, the kernel launches with identical grid dimensions (2048 blocks), making the mutant behaviorally equivalent to the original for this specific input shape. The mutation is effectively neutral due to integer division truncation.

**LLM 杀死策略：**

> Change the input tensor shape/dimensions so that `total_elements` is NOT a multiple of block_size, specifically making it small enough that the subtraction in the mutant causes underflow or a different integer division result. When `total_elements < block_size + 1`, the mutant's numerator becomes negative, causing `num_blocks` to be 0 (or negative, but CUDA may interpret negative as huge unsigned), while the original yields at least 1. This causes the kernel to launch with 0 blocks (no threads) vs. 1+ blocks, leaving output uninitialized vs. computed.

**LLM 建议的输入代码：**

```python
import torch
import math

def generate_inputs(device):
    # Use total_elements = 256 (exactly block_size)
    # Original: (256 + 256 - 1)/256 = 511/256 = 1 (since integer division truncates)
    # Mutant: (256 - 256 - 1)/256 = (-1)/256 = 0 (truncates toward zero)
    shape = [1, 256]  # total_elements = 256
    predictions = torch.randn(shape, device=device, dtype=torch.float32)
    targets = torch.randn(shape, device=device, dtype=torch.float32)
    return [predictions, targets]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`

#### 测试构造规则

- **规则名**: `grid_dimension_boundary_tests`
- **描述**: This rule generates inputs where the total number of elements equals or is slightly less than the thread block size, specifically targeting mutations in grid dimension calculations. It exploits integer division truncation to cause different block counts between original and mutated kernels, leading to uninitialized outputs in mutants.
- **适用算子**: arith_replace, const_replace, negate_condition

```python
def policy(shape, dtype, device, rng):
    import torch
    # Target boundary where total_elements equals or is less than typical block size (256)
    # Create tensor with exactly 256 elements regardless of requested shape
    # Use 2D shape [1, 256] to match typical kernel expectations
    total_elements = 256
    target_shape = [1, total_elements]
    
    # Generate random data with the target shape
    # Using normal distribution for typical loss function inputs
    tensor = torch.randn(target_shape, dtype=dtype, device=device)
    return tensor
```


---

## 23. `L1_P96__const_perturb__4`

- **Kernel**: `L1_P96`
- **算子**: `const_perturb` (Category A)
- **描述**: Perturb numeric literals (integer ±1; float ×1.01 or ×0.99) @ L89
- **变异行**: Line 89, 原始片段 `256`, 节点类型 `const:int+1`
- **杀死轮次**: Round 2

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
    const int reduction_block_size = 257;
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

**杀死输入代码 (Round 2)：**

```python
import torch
import math

def generate_inputs(device):
    shape = [128, 4096]
    total_elements = 128 * 4096
    predictions = torch.zeros(shape, device=device, dtype=torch.float32)
    targets = torch.zeros(shape, device=device, dtype=torch.float32)
    
    # Use values that create cancellation sensitive to summation order
    # Pattern: alternating signs with magnitudes that vary systematically
    # Base value carefully chosen to avoid overflow/underflow
    base_val = 1.33333333  # Repeating decimal for binary rounding effects
    scale = 1e-4
    
    for idx in range(total_elements):
        row = idx // 4096
        col = idx % 4096
        
        # Create pattern based on position within hypothetical 257-sized blocks
        pos_in_block = idx % 257
        
        # For positions that would be in the last (partial) warp in mutant:
        # Thread 256 is the only active thread in warp 8 (of 9 warps)
        # This changes reduction tree structure
        if pos_in_block == 256:
            # Make these values significant contributors
            val = base_val * 100.0  # Larger magnitude
        else:
            # Alternating signs for cancellation effects
            sign = 1.0 if (idx % 2 == 0) else -1.0
            # Vary magnitude slightly based on position
            val = sign * base_val * (1.0 + scale * (pos_in_block % 17))
        
        predictions[row, col] = val
        targets[row, col] = 0.0  # Simple target for controllable loss
    
    # Add small random perturbations to break symmetry
    predictions += torch.randn_like(predictions) * 1e-7
    
    return [predictions, targets]

```

**输出差异**: `max_diff=5.168686e-01, mean_diff=5.168686e-01, ref_range=[1.3480e+00,1.3480e+00], mut_range=[8.3113e-01,8.3113e-01]`

### Part 4: 大模型诊断流程

共 2 轮迭代分析：

#### Round 1 (未杀死)

**LLM 存活原因分析：**

> The mutant changes reduction block size from 256 to 257, which changes the number of reduction blocks (2049→2041) and elements per block. However, both values lead to correct ceil division for total_elements=524288. The reduction kernel likely uses a standard parallel reduction that works correctly for any valid block size, and the shared memory allocation scales accordingly. Floating-point non-associativity differences are negligible (order 1e-7 relative error) for most distributions, and the final CPU reduction of block sums masks any minor differences. The kernel appears robust to this constant perturbation.

**LLM 杀死策略：**

> The mutant could be killed by exploiting the different block partition boundaries. Create input where elements at the boundary positions (multiples of 256 and 257) have extreme values that cause floating-point overflow/underflow or cancellation errors that propagate differently due to the changed grouping. Specifically, arrange values so that partial sums in boundary blocks overflow in one configuration but not the other, or cause catastrophic cancellation that differs between the two partitionings.

**LLM 建议的输入代码：**

```python
import torch
import math

def generate_inputs(device):
    shape = [128, 4096]
    total_elements = 128 * 4096
    predictions = torch.zeros(shape, device=device, dtype=torch.float32)
    targets = torch.zeros(shape, device=device, dtype=torch.float32)
    
    # Values near float32 max that could cause overflow during reduction
    extreme_val = 3.0e38  # Close to float32 max ~3.4e38
    normal_val = 1.0
    
    # Set extreme values at positions that are multiples of 256
    # These positions will be at the START of blocks for block_size=256
    # but at different positions for block_size=257
    for idx in range(0, total_elements, 256):
        if idx < total_elements:
            row = idx // 4096
            col = idx % 4096
            predictions[row, col] = extreme_val
            targets[row, col] = 0.0  # Loss will be ~extreme_val
        
    # Set all other values to normal
    predictions += normal_val
    targets += normal_val
    
    return [predictions, targets]
```

**GPU 验证结果：**

- ref_ok: `False`, original_ok: `False`, mutant_ok: `False`
- error: `ref output NaN/Inf on LLM input`

#### Round 2 (**KILLED**)

**LLM 存活原因分析：**

> The mutant's survival suggests the reduction kernel implementation handles non-power-of-two block sizes correctly, making it resilient to boundary-based attacks. Previous attempts failed because: 1) The overflow scenario caused both versions to fail (NaN/inf), not differentiate; 2) The sum_reduction_kernel likely uses robust algorithms (e.g., sequential addressing, proper bounds checking) that work with 257; 3) The kernel may use atomic operations or warp-level reductions that are block-size invariant; 4) The 257 block size still evenly distributes work (total_elements=524288 is divisible by both 256 and 257 in terms of block counts). The key insight is that the mutation affects only performance characteristics, not numerical correctness, unless there's a subtle interaction with shared memory banking or warp divergence.

**LLM 杀死策略：**

> Exploit potential warp divergence and bank conflicts in shared memory reduction. With block size 257 (not a multiple of warp size 32), some warps will have threads with inactive elements during tree reduction, affecting floating-point addition order and associativity. Create inputs where accumulated floating-point error differs systematically based on thread inactivity patterns. Use carefully scaled values that cause different cancellation/rounding when threads are masked differently between 256 (all warps full) and 257 (one warp partially empty).

**LLM 建议的输入代码：**

```python
import torch
import math

def generate_inputs(device):
    shape = [128, 4096]
    total_elements = 128 * 4096
    predictions = torch.zeros(shape, device=device, dtype=torch.float32)
    targets = torch.zeros(shape, device=device, dtype=torch.float32)
    
    # Use values that create cancellation sensitive to summation order
    # Pattern: alternating signs with magnitudes that vary systematically
    # Base value carefully chosen to avoid overflow/underflow
    base_val = 1.33333333  # Repeating decimal for binary rounding effects
    scale = 1e-4
    
    for idx in range(total_elements):
        row = idx // 4096
        col = idx % 4096
        
        # Create pattern based on position within hypothetical 257-sized blocks
        pos_in_block = idx % 257
        
        # For positions that would be in the last (partial) warp in mutant:
        # Thread 256 is the only active thread in warp 8 (of 9 warps)
        # This changes reduction tree structure
        if pos_in_block == 256:
            # Make these values significant contributors
            val = base_val * 100.0  # Larger magnitude
        else:
            # Alternating signs for cancellation effects
            sign = 1.0 if (idx % 2 == 0) else -1.0
            # Vary magnitude slightly based on position
            val = sign * base_val * (1.0 + scale * (pos_in_block % 17))
        
        predictions[row, col] = val
        targets[row, col] = 0.0  # Simple target for controllable loss
    
    # Add small random perturbations to break symmetry
    predictions += torch.randn_like(predictions) * 1e-7
    
    return [predictions, targets]

```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=5.168686e-01, mean_diff=5.168686e-01, ref_range=[1.3480e+00,1.3480e+00], mut_range=[8.3113e-01,8.3113e-01]`

#### 测试构造规则

- **规则名**: `reduction_order_sensitive_pattern`
- **描述**: Generates inputs with alternating positive/negative values and strategically placed large-magnitude elements to exploit floating-point associativity differences in parallel reductions. The pattern is sensitive to how threads are grouped in reduction kernels, making it effective for detecting mutations that change block/warp sizes or reduction tree structures.
- **适用算子**: const_perturb, warp_size_change, block_size_change, reduction_tree_alter

```python
def policy(shape, dtype, device, rng):
    import torch
    import math
    
    total_elements = math.prod(shape)
    
    # Base pattern parameters
    base_val = 1.33333333  # Repeating decimal for binary rounding effects
    scale = 1e-4
    
    # Create tensor with pattern sensitive to reduction order
    flat_tensor = torch.zeros(total_elements, dtype=dtype, device=device)
    
    # Use common block/warp sizes as potential alignment points
    alignment_points = [32, 64, 128, 256, 512, 1024]  # Common GPU thread configurations
    
    # For each element, create pattern based on position relative to alignment points
    for idx in range(total_elements):
        # Determine if this index aligns with any common thread boundary
        is_boundary = any((idx + 1) % align == 0 for align in alignment_points)
        
        # Create alternating sign pattern
        sign = 1.0 if (idx % 2 == 0) else -1.0
        
        if is_boundary:
            # Make boundary elements significant contributors
            # These elements will be combined differently in reduction trees
            val = sign * base_val * 100.0
        else:
            # Vary magnitude slightly for cancellation effects
            val = sign * base_val * (1.0 + scale * (idx % 17))
        
        flat_tensor[idx] = val
    
    # Reshape to original shape
    result_tensor = flat_tensor.reshape(shape)
    
    # Add small random perturbations to break symmetry
    # Use provided RNG for reproducibility
    noise = torch.randn_like(result_tensor, generator=rng) * 1e-7
    result_tensor += noise
    
    return result_tensor
```


---

## 24. `L1_P96__launch_config_mutate__1`

- **Kernel**: `L1_P96`
- **算子**: `launch_config_mutate` (Category B)
- **描述**: Perturb grid/block sizing expressions (// BLOCK, triton.cdiv) by ±1 @ L90
- **变异行**: Line 90, 原始片段 `(total_elements + reduction_block_size - 1) / reduction_block_size`, 节点类型 `-1`
- **杀死轮次**: Round 1

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
    const int reduction_blocks = (total_elements + reduction_block_size - 1) / reduction_block_size - 1;
    
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

**杀死输入代码 (Round 1)：**

```python
import torch

def generate_inputs(device):
    # Choose a shape with total_elements = 100 (<=256) to force original reduction_blocks = 1, mutant = 0
    shape = [1, 100]
    predictions = torch.ones(shape, device=device, dtype=torch.float32) * 2.0  # Non-zero difference
    targets = torch.zeros(shape, device=device, dtype=torch.float32)
    return [predictions, targets]
```

### Part 4: 大模型诊断流程

共 1 轮迭代分析：

#### Round 1 (**KILLED**)

**LLM 存活原因分析：**

> The mutant reduces the number of reduction blocks by one (from ceil(N/256) to ceil(N/256)-1). With the original input shape (524288 elements), the reduction kernel likely uses a grid-stride loop to process elements. When one block is removed, the stride decreases from 524288 to 523264, causing the first 1024 threads to each process one extra element, thus still covering all elements. The partial sums are then aggregated differently but still sum to the same total loss, making the mutant numerically equivalent for large inputs that are multiples of the block size or have a large number of blocks.

**LLM 杀死策略：**

> Use a small input where the original reduction_blocks = 1 (i.e., total_elements ≤ 256). The mutant will compute reduction_blocks = 0, leading to an empty grid launch and a zero sum of losses, while the original correctly computes the non-zero sum.

**LLM 建议的输入代码：**

```python
import torch

def generate_inputs(device):
    # Choose a shape with total_elements = 100 (<=256) to force original reduction_blocks = 1, mutant = 0
    shape = [1, 100]
    predictions = torch.ones(shape, device=device, dtype=torch.float32) * 2.0  # Non-zero difference
    targets = torch.zeros(shape, device=device, dtype=torch.float32)
    return [predictions, targets]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`

#### 测试构造规则

- **规则名**: `minimal_reduction_workload`
- **描述**: Targets kernels with grid-stride reduction patterns by creating inputs with total elements ≤ typical block size (e.g., 256). This forces the original kernel to launch exactly one block while mutants that reduce block count by one may launch zero blocks, causing silent computation failures.
- **适用算子**: launch_config_mutate, grid_mutate, block_mutate

```python
def policy(shape, dtype, device, rng):
    # Force total elements to be ≤256 to target reduction kernels with block size 256
    # Use small non-zero values to ensure detectable output differences
    total_elements = rng.integers(1, 257)  # 1-256 elements
    # Create shape that produces exactly total_elements
    if len(shape) > 1:
        # Keep first dimension as 1, adjust last dimension to reach target
        new_shape = [1] * (len(shape)-1) + [total_elements]
    else:
        new_shape = [total_elements]
    
    # Generate non-zero values to ensure reduction produces detectable result
    # Use simple pattern: constant value between 1.0 and 3.0
    value = 1.0 + 2.0 * rng.random()
    return torch.full(new_shape, value, dtype=dtype, device=device)
```


---

## 25. `L1_P96__relop_replace__8`

- **Kernel**: `L1_P96`
- **算子**: `relop_replace` (Category A)
- **描述**: Replace relational operators (<→<=, <=→<, >→>=, >=→>, ==→!=, !=→==) @ L106
- **变异行**: Line 106, 原始片段 `<`, 节点类型 `cuda_Lt`
- **杀死轮次**: Round 2

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
    for (int i = 0; i <= reduction_blocks; i++) {
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

**杀死输入代码 (Round 2)：**

```python
import torch
import math

def generate_inputs(device):
    # Use the required shape
    shape = [128, 4096]
    total_elements = 128 * 4096
    
    # First, clear CUDA cache to make allocation order more predictable
    torch.cuda.empty_cache()
    
    # Create a large dummy tensor that will be allocated, then freed.
    # This helps fragment memory and makes subsequent allocations more predictable.
    dummy = torch.zeros(1000000, device=device, dtype=torch.float32)
    del dummy
    torch.cuda.empty_cache()
    
    # Create predictions and targets that yield non-zero losses
    # Use values that ensure smooth L1 loss is non-zero
    predictions = torch.randn(shape, dtype=torch.float32, device=device) * 10.0
    targets = torch.randn(shape, dtype=torch.float32, device=device) * 10.0
    
    # Now create a tensor that will likely be allocated immediately after
    # block_sums in memory. We compute the expected block_sums size.
    reduction_block_size = 256  # typical for reduction kernels
    reduction_blocks = (total_elements + reduction_block_size - 1) // reduction_block_size
    
    # Create a tensor with large values that might land in the out-of-bounds location
    # We'll create it with the exact size that would be at the out-of-bounds index
    poison = torch.full((reduction_blocks + 100,), 1e6, dtype=torch.float32, device=device)
    # Keep a reference to prevent immediate reuse
    poison_holder = poison[reduction_blocks]  # Access to keep tensor alive but fragmented
    
    return [predictions, targets]

```

**输出差异**: `max_diff=2.643071e+28, mean_diff=2.643071e+28, ref_range=[1.0786e+01,1.0786e+01], mut_range=[2.6431e+28,2.6431e+28]`

### Part 4: 大模型诊断流程

共 2 轮迭代分析：

#### Round 1 (未杀死)

**LLM 存活原因分析：**

> The mutant changes the loop condition from `i < reduction_blocks` to `i <= reduction_blocks`, causing an out-of-bounds read at index `reduction_blocks` in the `block_sums_cpu` array. However, this out-of-bounds memory location likely contains zero or a small value due to memory alignment/padding or adjacent zero-initialized memory. Since the stress tests used fixed tensor shapes, the memory layout for `block_sums_cpu` remains identical across runs, and the out-of-bounds read consistently yields a benign value (likely 0.0f). The reduction sum thus remains unchanged, making the mutant behave identically to the original.

**LLM 杀死策略：**

> Change the input tensor dimensions to alter `reduction_blocks` (grid size), which changes the allocation size of `block_sums`. This can cause the out-of-bounds index to land in non-zero memory (e.g., adjacent tensor metadata or uninitialized memory). Specifically, use input shapes that cause `reduction_blocks` to be 1, making the out-of-bounds index 1 read the first element of whatever follows `block_sums_cpu` in memory. By controlling subsequent allocations (via careful tensor creation), we can force this memory to contain a large, distinguishable value.

**LLM 建议的输入代码：**

```python
import torch
import math

def generate_inputs(device):
    # Use minimal shape to minimize reduction_blocks
    shape = [1, 1]  # total_elements = 1
    
    # Create predictions and targets with arbitrary values
    # The exact values don't matter much, but we use non-zero to ensure block_sums[0] != 0
    predictions = torch.full(shape, 100.0, dtype=torch.float32, device=device)
    targets = torch.full(shape, 50.0, dtype=torch.float32, device=device)
    
    # To influence the out-of-bounds memory, we cannot directly control allocation order.
    # However, we can attempt to create a scenario where subsequent memory contains large values
    # by forcing reallocation of the same memory pool.
    # We'll create a dummy tensor with large values right before the block_sums allocation
    # to increase chance of leftover non-zero values.
    # This is heuristic and may not work on all allocators, but it's the best shot.
    
    # Note: The test may need to be run multiple times or with specific CUDA context
    # to ensure memory layout. We'll also use a large beta to amplify any difference.
    
    return [predictions, targets]

```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `True`

#### Round 2 (**KILLED**)

**LLM 存活原因分析：**

> The mutant survives because the out-of-bounds access at index `reduction_blocks` consistently reads zero-valued memory. The GPU reduction kernel likely produces exactly `reduction_blocks` valid values, with the immediate subsequent memory being zero (either due to CUDA memory allocation patterns, block_sums tensor padding, or adjacent zero-initialized memory). Since adding zero doesn't change the sum, the mutant computes the same total_loss as the original. The previous attempt failed because manipulating tensor dimensions didn't change this fundamental zero-access behavior - even with shape [1,1], the out-of-bounds memory was zero.

**LLM 杀死策略：**

> Force the out-of-bounds memory location to contain a non-zero value by carefully controlling CUDA memory layout. Since block_sums is a tensor of floats, we can allocate another tensor immediately after it in GPU memory (by reusing the same storage with offset) to place controlled values at the out-of-bounds index. The key is to make the allocation order predictable.

**LLM 建议的输入代码：**

```python
import torch
import math

def generate_inputs(device):
    # Use the required shape
    shape = [128, 4096]
    total_elements = 128 * 4096
    
    # First, clear CUDA cache to make allocation order more predictable
    torch.cuda.empty_cache()
    
    # Create a large dummy tensor that will be allocated, then freed.
    # This helps fragment memory and makes subsequent allocations more predictable.
    dummy = torch.zeros(1000000, device=device, dtype=torch.float32)
    del dummy
    torch.cuda.empty_cache()
    
    # Create predictions and targets that yield non-zero losses
    # Use values that ensure smooth L1 loss is non-zero
    predictions = torch.randn(shape, dtype=torch.float32, device=device) * 10.0
    targets = torch.randn(shape, dtype=torch.float32, device=device) * 10.0
    
    # Now create a tensor that will likely be allocated immediately after
    # block_sums in memory. We compute the expected block_sums size.
    reduction_block_size = 256  # typical for reduction kernels
    reduction_blocks = (total_elements + reduction_block_size - 1) // reduction_block_size
    
    # Create a tensor with large values that might land in the out-of-bounds location
    # We'll create it with the exact size that would be at the out-of-bounds index
    poison = torch.full((reduction_blocks + 100,), 1e6, dtype=torch.float32, device=device)
    # Keep a reference to prevent immediate reuse
    poison_holder = poison[reduction_blocks]  # Access to keep tensor alive but fragmented
    
    return [predictions, targets]

```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=2.643071e+28, mean_diff=2.643071e+28, ref_range=[1.0786e+01,1.0786e+01], mut_range=[2.6431e+28,2.6431e+28]`

#### 测试构造规则

- **规则名**: `relop_oob_poisoned_adjacent_memory`
- **描述**: Targets mutations that cause out-of-bounds accesses by poisoning memory adjacent to intermediate tensors with non-zero values. Creates allocation fragmentation to make adjacent memory locations predictable and fills them with large values that produce observable differences if accessed out-of-bounds. This is particularly effective for reduction kernels where out-of-bounds accesses might normally read zero-valued memory.
- **适用算子**: relop_replace, array_index_replace, constant_replace, boundary_condition

```python
import torch
import math

def policy(shape, dtype, device, rng):
    # Clear cache to make allocation order predictable
    torch.cuda.empty_cache()
    
    # Create and free a large tensor to fragment memory
    fragmentation_size = 1000000  # Adjust based on available GPU memory
    dummy = torch.zeros(fragmentation_size, device=device, dtype=dtype)
    del dummy
    torch.cuda.empty_cache()
    
    # Generate main input tensors with non-zero values
    # Use the rng for reproducibility but ensure non-zero outputs
    if dtype in [torch.float32, torch.float64, torch.float16]:
        # Generate values that produce non-zero outputs for common operations
        input1 = torch.randn(shape, device=device, dtype=dtype, generator=rng) * 10.0
        input2 = torch.randn(shape, device=device, dtype=dtype, generator=rng) * 10.0
    else:
        # For integer types
        input1 = torch.randint(1, 100, shape, device=device, dtype=dtype, generator=rng)
        input2 = torch.randint(1, 100, shape, device=device, dtype=dtype, generator=rng)
    
    # Estimate potential intermediate tensor sizes that might be accessed out-of-bounds
    # Common patterns: reduction blocks, padding, alignment boundaries
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    
    # Try multiple common block sizes for reductions
    block_sizes = [128, 256, 512, 1024]
    for block_size in block_sizes:
        reduction_blocks = (total_elements + block_size - 1) // block_size
        
        # Create poisoned memory region that might be adjacent to intermediate tensors
        # Use a size that covers potential out-of-bounds accesses
        poison_size = max(reduction_blocks + 100, 1024)
        poison_value = 1e6  # Large value that creates observable differences
        
        if dtype in [torch.float32, torch.float64, torch.float16]:
            poison = torch.full((poison_size,), poison_value, dtype=dtype, device=device)
        else:
            poison = torch.full((poison_size,), int(poison_value), dtype=dtype, device=device)
        
        # Keep reference to prevent immediate reuse
        _ = poison[0]  # Simple reference to keep tensor alive
    
    return [input1, input2]
```


---

## 26. `L1_P97__epsilon_modify__0`

- **Kernel**: `L1_P97`
- **算子**: `epsilon_modify` (Category C)
- **描述**: Alter small epsilon literals (LayerNorm / safe_div / log stability) @ L95
- **变异行**: Line 95, 原始片段 `1e-8f`, 节点类型 `eps:to_1e-2`
- **杀死轮次**: Round 1

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
        const float epsilon = 1e-2f;
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

**杀死输入代码 (Round 1)：**

```python
import torch

def generate_inputs(device):
    batch_size = 128
    feature_size = 4096
    # Small constant chosen so that norm per sample = feature_size * (5e-5)**2 ≈ 1e-5
    val = 5e-5
    predictions = torch.full((batch_size, feature_size), val, dtype=torch.float32, device=device)
    targets = predictions.clone()  # identical tensors
    return [predictions, targets]
```

**输出差异**: `max_diff=9.994887e-01, mean_diff=9.994887e-01, ref_range=[0.0000e+00,0.0000e+00], mut_range=[9.9949e-01,9.9949e-01]`

### Part 4: 大模型诊断流程

共 1 轮迭代分析：

#### Round 1 (**KILLED**)

**LLM 存活原因分析：**

> The mutant survived because the stress tests did not include inputs where the norms of the vectors are in the critical regime between the two epsilon values (1e-8 and 1e-2). For typical random inputs, norms are either much larger than 1e-2 (making epsilon negligible) or exactly zero (making loss identical). The near-zero tests likely used random uncorrelated vectors, resulting in tiny dot products, so the loss difference was too small to detect or within floating-point tolerance. The uniform_constant test may have used constants that were either too large, zero, or different across tensors, missing the case where identical small vectors yield norms comparable to epsilon.

**LLM 杀死策略：**

> Use identical small-valued vectors such that the norm (sum of squares) is on the order of 1e-5, which lies between 1e-8 and 1e-2. This causes the original epsilon (1e-8) to have negligible effect (since norm >> epsilon), giving a cosine similarity near 1 and loss near 0, while the mutant epsilon (1e-2) dominates the sqrt and denominator, giving a cosine similarity near 0 and loss near 1. The large difference in loss will expose the mutation.

**LLM 建议的输入代码：**

```python
import torch

def generate_inputs(device):
    batch_size = 128
    feature_size = 4096
    # Small constant chosen so that norm per sample = feature_size * (5e-5)**2 ≈ 1e-5
    val = 5e-5
    predictions = torch.full((batch_size, feature_size), val, dtype=torch.float32, device=device)
    targets = predictions.clone()  # identical tensors
    return [predictions, targets]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=9.994887e-01, mean_diff=9.994887e-01, ref_range=[0.0000e+00,0.0000e+00], mut_range=[9.9949e-01,9.9949e-01]`

#### 测试构造规则

- **规则名**: `critical_norm_epsilon_regime`
- **描述**: Generates two identical tensors where the L2 norm per sample is in a critical regime between typical epsilon values (1e-9 to 1e-1). This targets mutations that modify numerical stability epsilon values, as the original and mutant will produce significantly different outputs when norms are comparable to the epsilon thresholds.
- **适用算子**: epsilon_modify, epsilon_insert, epsilon_delete

```python
import torch
import numpy as np

def policy(shape, dtype, device, rng):
    # Assume shape is (batch, features) for a single tensor
    batch, features = shape[-2], shape[-1]
    # Sample target norm per sample in log space between 1e-9 and 1e-1
    log_norm = rng.uniform(np.log(1e-9), np.log(1e-1))
    target_norm = np.exp(log_norm)
    # Compute constant value to achieve this norm per sample: norm = sqrt(features * c^2)
    c = float(target_norm / np.sqrt(features))
    # Create identical tensors with the computed constant
    tensor = torch.full(shape, c, dtype=dtype, device=device)
    return [tensor, tensor.clone()]
```


---

## 27. `L1_P97__mask_boundary__2`

- **Kernel**: `L1_P97`
- **算子**: `mask_boundary` (Category B)
- **描述**: Weaken or tighten boundary checks in mask/tl.where (Triton) or thread/block guards (CUDA) @ L43
- **变异行**: Line 43, 原始片段 `element_idx < feature_size`, 节点类型 `rhs-1`
- **杀死轮次**: Round 1

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
            if (element_idx < feature_size - 1) {
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

### Part 3: 杀死该变异体的输入

**杀死输入代码 (Round 1)：**

```python
import torch
import math

def generate_inputs(device):
    # Use feature_size = 4097 instead of 4096
    # This ensures the last vectorized chunk processes only 1 valid element
    batch_size = 128
    feature_size = 4097  # Changed from 4096 to expose boundary
    
    # Initialize all zeros
    predictions = torch.zeros((batch_size, feature_size), 
                              device=device, dtype=torch.float32)
    targets = torch.zeros((batch_size, feature_size),
                          device=device, dtype=torch.float32)
    
    # Set ONLY the last element (index 4096) to large non-zero values
    # This element will be processed in original but skipped in mutant
    predictions[:, -1] = 1e6
    targets[:, -1] = 1e6
    
    return [predictions, targets]

```

**输出差异**: `max_diff=1.000000e+00, mean_diff=1.000000e+00, ref_range=[0.0000e+00,0.0000e+00], mut_range=[1.0000e+00,1.0000e+00]`

### Part 4: 大模型诊断流程

共 1 轮迭代分析：

#### Round 1 (**KILLED**)

**LLM 存活原因分析：**

> The mutant survived because all stress tests kept the input shape fixed at [128, 4096], where feature_size=4096 is likely divisible by the vectorization stride (THREADS_PER_BLOCK * ELEMENTS_PER_THREAD). This means the boundary condition only affects the final iteration when element_idx equals feature_size-1 (the very last element). Since the tests only varied values and not shapes, the mutant skips processing just one element per vector out of 4096. The difference in outputs (dot_product, norm_pred, norm_target) was too small relative to the total magnitude across all elements to be detected by floating-point comparisons. The 'boundary_last_element' test only set the last tensor element to extreme values, but didn't ensure the skipped element (the last of each vector) was non-zero across all 128 vectors.

**LLM 杀死策略：**

> Change the input shape so that feature_size is NOT a multiple of the vectorization stride. This causes multiple threads to hit the boundary condition in the final loop iteration, skipping more than just the single last element. The exact number depends on the kernel's THREADS_PER_BLOCK and ELEMENTS_PER_THREAD parameters. For maximum effect, choose a shape where feature_size ≡ 1 mod (THREADS_PER_BLOCK * ELEMENTS_PER_THREAD), making ALL threads in the final iteration skip their elements except one. Fill only the skipped elements with large values (like 1e6) while others are zero.

**LLM 建议的输入代码：**

```python
import torch
import math

def generate_inputs(device):
    # Use feature_size = 4097 instead of 4096
    # This ensures the last vectorized chunk processes only 1 valid element
    batch_size = 128
    feature_size = 4097  # Changed from 4096 to expose boundary
    
    # Initialize all zeros
    predictions = torch.zeros((batch_size, feature_size), 
                              device=device, dtype=torch.float32)
    targets = torch.zeros((batch_size, feature_size),
                          device=device, dtype=torch.float32)
    
    # Set ONLY the last element (index 4096) to large non-zero values
    # This element will be processed in original but skipped in mutant
    predictions[:, -1] = 1e6
    targets[:, -1] = 1e6
    
    return [predictions, targets]

```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=1.000000e+00, mean_diff=1.000000e+00, ref_range=[0.0000e+00,0.0000e+00], mut_range=[1.0000e+00,1.0000e+00]`

#### 测试构造规则

- **规则名**: `non_divisible_shape_boundary_exposure`
- **描述**: Generates input shapes where dimension sizes are not multiples of common vectorization strides (e.g., 1024). Sets extreme values exclusively at boundary positions that would be processed in the original kernel but skipped in boundary-tightening mutants. This forces divergence in final outputs by making boundary elements dominate the computation.
- **适用算子**: mask_boundary, relax_boundary, shift_boundary, wrap_boundary

```python
def policy(shape, dtype, device, rng):
    import torch
    import math
    
    # Use original shape but modify the last dimension to be non-divisible by common strides
    # Common vectorization strides: 256*4=1024, 128*4=512, 256*2=512
    common_strides = [1024, 512, 256]
    
    modified_shape = list(shape)
    if len(modified_shape) >= 2:
        last_dim = modified_shape[-1]
        # Find a non-divisible size slightly larger than original
        for stride in common_strides:
            if last_dim % stride == 0:
                # Make non-divisible by adding 1 (or small prime)
                modified_shape[-1] = last_dim + 1
                break
        # If already non-divisible, keep as is
    
    # Create zero tensors
    predictions = torch.zeros(modified_shape, device=device, dtype=dtype)
    targets = torch.zeros(modified_shape, device=device, dtype=dtype)
    
    # Set only boundary elements to extreme values
    # For boundary-tightening mutants: set elements at idx = N-1 (last valid)
    predictions.view(-1, modified_shape[-1])[:, -1] = 1e6
    targets.view(-1, modified_shape[-1])[:, -1] = 1e6
    
    # For boundary-relaxing mutants: also set idx = N (first out-of-bounds if any)
    # This covers multiple boundary mutation types
    
    return [predictions, targets]
```


---

## 28. `L1_P99__cast_remove__0`

- **Kernel**: `L1_P99`
- **算子**: `cast_remove` (Category C)
- **描述**: Remove explicit dtype casts (.to, .float, tl.cast, static_cast) @ L171
- **变异行**: Line 171, 原始片段 `.float()`, 节点类型 `cast:remove_method`
- **杀死轮次**: Round 1

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
        anchor = anchor.contiguous().cuda()
        positive = positive.contiguous().cuda().float()
        negative = negative.contiguous().cuda().float()
        
        return self.triplet_loss_cuda(anchor, positive, negative, self.margin)
```

### Part 3: 杀死该变异体的输入

**杀死输入代码 (Round 1)：**

```python
import torch

def generate_inputs(device):
    # Anchor: float16 (will NOT be cast to float32 in mutant)
    anchor = torch.randn(128, 4096, dtype=torch.float16, device=device).contiguous()
    
    # Positive/Negative: float32 (will be cast to float32 in both)
    positive = torch.randn(128, 4096, dtype=torch.float32, device=device).contiguous()
    negative = torch.randn(128, 4096, dtype=torch.float32, device=device).contiguous()
    
    # Ensure values are large enough to cause precision differences in float16
    # Scale up to make float16 rounding/overflow more apparent
    anchor = anchor * 100.0
    positive = positive * 100.0
    negative = negative * 100.0
    
    return [anchor, positive, negative]
```

### Part 4: 大模型诊断流程

共 1 轮迭代分析：

#### Round 1 (**KILLED**)

**LLM 存活原因分析：**

> The mutant survives because the mutation removes the .float() cast on line 171, but all stress tests used inputs with dtype=torch.float32 as specified in get_inputs(). When the input is already float32, calling .float() is a no-op - it returns the same tensor unchanged. The kernel likely expects float32 inputs, and since the stress tests never varied the dtype of the input tensors, both original and mutant received identical float32 tensors, producing identical outputs. The mutation only affects behavior when anchor has a different dtype (like float16, float64, int, etc.), which was never tested.

**LLM 杀死策略：**

> The mutant can be killed by providing an anchor tensor with a dtype different from float32 (e.g., float16, float64, or int32). The original code explicitly casts to float32, ensuring type consistency for the CUDA kernel. The mutant passes the anchor's original dtype directly, which could cause: 1) kernel errors if it doesn't support that dtype, 2) numerical differences due to precision loss/overflow, or 3) type promotion inconsistencies with positive/negative tensors (which remain float32).

**LLM 建议的输入代码：**

```python
import torch

def generate_inputs(device):
    # Anchor: float16 (will NOT be cast to float32 in mutant)
    anchor = torch.randn(128, 4096, dtype=torch.float16, device=device).contiguous()
    
    # Positive/Negative: float32 (will be cast to float32 in both)
    positive = torch.randn(128, 4096, dtype=torch.float32, device=device).contiguous()
    negative = torch.randn(128, 4096, dtype=torch.float32, device=device).contiguous()
    
    # Ensure values are large enough to cause precision differences in float16
    # Scale up to make float16 rounding/overflow more apparent
    anchor = anchor * 100.0
    positive = positive * 100.0
    negative = negative * 100.0
    
    return [anchor, positive, negative]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`

#### 测试构造规则

- **规则名**: `dtype_mismatch_precision_edge`
- **描述**: Generates tensors with dtypes that differ from the expected computation type (typically float32) to expose mutations that remove explicit type casts. This creates precision differences and potential kernel failures when mixed dtypes are used in computations without proper promotion. The rule produces large magnitude values to amplify floating-point precision discrepancies.
- **适用算子**: cast_remove, cast_change, operator_replacement

```python
def policy(shape, dtype, device, rng):
    # Generate random tensor with potentially mismatched dtype
    # Use half precision when expected is float32, or float64 when expected is float16
    if dtype == torch.float32:
        target_dtype = torch.float16
    elif dtype == torch.float16:
        target_dtype = torch.float64
    elif dtype == torch.float64:
        target_dtype = torch.float32
    else:
        # For integer types, switch to floating point
        target_dtype = torch.float32
    
    # Generate random data with large magnitude to amplify precision differences
    tensor = torch.randn(shape, dtype=target_dtype, device=device, generator=rng)
    tensor = tensor * 100.0  # Scale to create noticeable precision differences
    
    return tensor.contiguous()
```


---

## 29. `L1_P99__cast_remove__1`

- **Kernel**: `L1_P99`
- **算子**: `cast_remove` (Category C)
- **描述**: Remove explicit dtype casts (.to, .float, tl.cast, static_cast) @ L172
- **变异行**: Line 172, 原始片段 `.float()`, 节点类型 `cast:remove_method`
- **杀死轮次**: Round 1

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
        positive = positive.contiguous().cuda()
        negative = negative.contiguous().cuda().float()
        
        return self.triplet_loss_cuda(anchor, positive, negative, self.margin)
```

### Part 3: 杀死该变异体的输入

**杀死输入代码 (Round 1)：**

```python
import torch

def generate_inputs(device):
    torch.manual_seed(999)
    # anchor: float32 (as in original spec)
    anchor = torch.randn([128, 4096], dtype=torch.float32, device=device)
    # positive: float64 (different dtype) – large magnitude to amplify precision differences
    positive = torch.randn([128, 4096], dtype=torch.float64, device=device) * 1e6
    # negative: float32 (as in original spec)
    negative = torch.randn([128, 4096], dtype=torch.float32, device=device)
    return [anchor, positive, negative]
```

### Part 4: 大模型诊断流程

共 1 轮迭代分析：

#### Round 1 (**KILLED**)

**LLM 存活原因分析：**

> The mutant survived because all stress tests used input tensors with dtype torch.float32 (as specified in get_inputs()). The mutation removed .float() on line 172, but since the input 'positive' tensor already had dtype torch.float32, the removal had no effect—both original and mutant produce identical tensors after the line executes. The CUDA kernel likely expects float32 inputs, so with float32 inputs the computation remains identical. No value distribution changes could expose a difference because the dtype remained unchanged.

**LLM 杀死策略：**

> The mutant can be killed by providing a 'positive' tensor with a different dtype (e.g., torch.float64, torch.int32). The original code would cast it to float32, while the mutant would leave it as the original dtype. This will cause a dtype mismatch when the CUDA kernel is called (if the kernel expects all inputs to be float32) or produce different numerical results due to precision differences.

**LLM 建议的输入代码：**

```python
import torch

def generate_inputs(device):
    torch.manual_seed(999)
    # anchor: float32 (as in original spec)
    anchor = torch.randn([128, 4096], dtype=torch.float32, device=device)
    # positive: float64 (different dtype) – large magnitude to amplify precision differences
    positive = torch.randn([128, 4096], dtype=torch.float64, device=device) * 1e6
    # negative: float32 (as in original spec)
    negative = torch.randn([128, 4096], dtype=torch.float32, device=device)
    return [anchor, positive, negative]
```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`

#### 测试构造规则

- **规则名**: `mixed_precision_large_magnitude`
- **描述**: Generates tensors with mixed floating-point precisions (e.g., float32/float64) and large magnitude values to expose precision-loss bugs. When a mutation removes type casts, this causes dtype mismatches in kernels expecting uniform precision, leading to runtime errors or significant numerical differences.
- **适用算子**: cast_remove

```python
def policy(shape, dtype, device, rng):
    import torch
    # If original dtype is float32, switch to float64 (or vice versa)
    if dtype == torch.float32:
        new_dtype = torch.float64
    elif dtype == torch.float64:
        new_dtype = torch.float32
    else:
        # For non-float types, keep original dtype
        new_dtype = dtype
    
    # Generate random values with large magnitude to amplify precision differences
    tensor = torch.randn(shape, dtype=new_dtype, device=device, generator=rng)
    # Scale by large factor (1e6) to make precision differences more pronounced
    return tensor * 1e6
```


---

## 30. `L1_P99__cast_remove__2`

- **Kernel**: `L1_P99`
- **算子**: `cast_remove` (Category C)
- **描述**: Remove explicit dtype casts (.to, .float, tl.cast, static_cast) @ L173
- **变异行**: Line 173, 原始片段 `.float()`, 节点类型 `cast:remove_method`
- **杀死轮次**: Round 1

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
        negative = negative.contiguous().cuda()
        
        return self.triplet_loss_cuda(anchor, positive, negative, self.margin)
```

### Part 3: 杀死该变异体的输入

**杀死输入代码 (Round 1)：**

```python
import torch
import math

def generate_inputs(device):
    # Generate random values in a range where float32 vs float64 differences matter
    # Using values that require more precision than float32 can provide
    shape = [128, 4096]
    
    # Anchor and positive: float32 as usual
    anchor = torch.randn(shape, dtype=torch.float32, device=device) * 1000.0
    positive = anchor + torch.randn(shape, dtype=torch.float32, device=device) * 0.01
    
    # Negative: float64 (double) with values that have precision beyond float32
    # Create values with subtle differences that float32 cannot represent accurately
    negative = torch.randn(shape, dtype=torch.float64, device='cpu')
    # Scale to large magnitude to amplify precision differences
    negative = negative * 1000000.0
    # Add tiny perturbations that get lost in float32 rounding
    negative = negative + torch.randn(shape, dtype=torch.float64, device='cpu') * 1e-10
    negative = negative.to(device)  # Move to device (still float64)
    
    return [anchor, positive, negative]

```

### Part 4: 大模型诊断流程

共 1 轮迭代分析：

#### Round 1 (**KILLED**)

**LLM 存活原因分析：**

> The mutant survives because the input tensors from get_inputs() are already torch.float32. The mutation removes the .float() cast on the negative tensor, but since it's already float32, the cast is a no-op. The stress tests only varied values within float32, not dtype. The CUDA kernel likely accepts float32 inputs, so both original and mutant pass identical float32 tensors to the kernel, producing identical outputs. The mutation is semantically equivalent for float32 inputs, making it unkillable by value variations alone.

**LLM 杀死策略：**

> The mutant can be killed by providing a negative tensor with a non-float32 dtype (e.g., float64, float16, int32). The original code would cast it to float32, while the mutant would pass the original dtype to the kernel. The CUDA kernel may produce incorrect results, crash, or trigger dtype mismatches when receiving unexpected dtypes.

**LLM 建议的输入代码：**

```python
import torch
import math

def generate_inputs(device):
    # Generate random values in a range where float32 vs float64 differences matter
    # Using values that require more precision than float32 can provide
    shape = [128, 4096]
    
    # Anchor and positive: float32 as usual
    anchor = torch.randn(shape, dtype=torch.float32, device=device) * 1000.0
    positive = anchor + torch.randn(shape, dtype=torch.float32, device=device) * 0.01
    
    # Negative: float64 (double) with values that have precision beyond float32
    # Create values with subtle differences that float32 cannot represent accurately
    negative = torch.randn(shape, dtype=torch.float64, device='cpu')
    # Scale to large magnitude to amplify precision differences
    negative = negative * 1000000.0
    # Add tiny perturbations that get lost in float32 rounding
    negative = negative + torch.randn(shape, dtype=torch.float64, device='cpu') * 1e-10
    negative = negative.to(device)  # Move to device (still float64)
    
    return [anchor, positive, negative]

```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`

#### 测试构造规则

- **规则名**: `higher_precision_inputs`
- **描述**: Generates tensors with dtypes of higher precision than typical (e.g., float64 for float32 contexts). This reveals mutations that remove explicit casts by exposing precision differences or kernel compatibility issues when higher-precision data flows through computations designed for lower precision.
- **适用算子**: cast_remove, cast_type_change, arithmetic_replacement

```python
def policy(shape, dtype, device, rng):
    # Map common target dtypes to higher-precision alternatives
    precision_map = {
        torch.float16: torch.float32,
        torch.float32: torch.float64,
        torch.bfloat16: torch.float32,
        torch.int32: torch.int64,
        torch.int16: torch.int32,
    }
    
    # Use higher precision if available, otherwise fall back to original
    target_dtype = precision_map.get(dtype, dtype)
    
    # Generate base tensor with random values in a large range to amplify precision differences
    base = torch.randn(shape, dtype=target_dtype, device='cpu', generator=rng) * 1000000.0
    
    # Add tiny perturbations that get lost when converting to lower precision
    if target_dtype in [torch.float32, torch.float64]:
        # For floating types: add noise below the precision threshold of original dtype
        eps = torch.finfo(dtype).eps if hasattr(torch.finfo(dtype), 'eps') else 1e-7
        perturbation = torch.randn(shape, dtype=target_dtype, device='cpu', generator=rng) * (eps / 1000.0)
        base = base + perturbation
    elif target_dtype in [torch.int32, torch.int64]:
        # For integer types: ensure values exceed typical range of original dtype
        max_val = torch.iinfo(dtype).max if hasattr(torch.iinfo(dtype), 'max') else 2**31-1
        base = base % (max_val * 2)  # Values potentially beyond original dtype's range
    
    # Move to target device, preserving higher precision dtype
    return base.to(device)
```


---

## 31. `L1_P99__mask_boundary__2`

- **Kernel**: `L1_P99`
- **算子**: `mask_boundary` (Category B)
- **描述**: Weaken or tighten boundary checks in mask/tl.where (Triton) or thread/block guards (CUDA) @ L54
- **变异行**: Line 54, 原始片段 `idx < feature_dim`, 节点类型 `rhs-1`
- **杀死轮次**: Round 3

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
            if (idx < feature_dim - 1) {
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

**杀死输入代码 (Round 3)：**

```python
import torch

def generate_inputs(device):
    batch_size, feature_dim = 128, 4096
    # All zeros except last feature
    anchor = torch.zeros(batch_size, feature_dim, dtype=torch.float32, device=device)
    positive = torch.zeros(batch_size, feature_dim, dtype=torch.float32, device=device)
    negative = torch.zeros(batch_size, feature_dim, dtype=torch.float32, device=device)
    
    # Last element values that create asymmetric distances
    # Anchor differs more from positive than from negative (by square effect)
    anchor[:, -1] = 1.0          # Large difference to positive
    positive[:, -1] = 0.0        # Zero -> squared diff = 1.0
    negative[:, -1] = 0.9        # Close to anchor -> squared diff = 0.01
    
    # Ensure other elements are zero (already set)
    return [anchor, positive, negative]

```

**输出差异**: `max_diff=8.999999e-01, mean_diff=8.999999e-01, ref_range=[1.9000e+00,1.9000e+00], mut_range=[1.0000e+00,1.0000e+00]`

### Part 4: 大模型诊断流程

共 3 轮迭代分析：

#### Round 1 (未杀死)

**LLM 存活原因分析：**

> The mutant survives because the stress tests only varied input values, not shapes. With feature_dim=4096, the mutation changes the condition from 'idx < 4096' to 'idx < 4095', causing the last element (index 4095) to be skipped. However, in the loop structure, 'i' increments by 'blockDim.x * ELEMENTS_PER_THREAD'. Since 4096 is likely a multiple of the thread block's coverage (typically 128 threads × 32 elements/thread = 4096), the last iteration's 'i' value equals 4096 - ELEMENTS_PER_THREAD, making 'i + j' reach exactly 4095 only in the last iteration. The mutant skips index 4095, but all other indices (0-4094) are processed identically. The stress tests missed this because they didn't test where the last element's exclusion causes a measurable difference in the final loss output.

**LLM 杀死策略：**

> Design inputs where the last feature element (index 4095) carries significant, asymmetric information compared to other elements. Specifically, set anchor[..., 4095] = 1, positive[..., 4095] = 0, negative[..., 4095] = 0, while keeping all other elements zero. This makes the positive distance entirely from the last element (which mutant skips), while the negative distance is zero. The original will compute non-zero distances; mutant will compute zero distances for both, leading to different loss values.

**LLM 建议的输入代码：**

```python
import torch

def generate_inputs(device):
    batch_size, feature_dim = 128, 4096
    anchor = torch.zeros(batch_size, feature_dim, dtype=torch.float32, device=device)
    positive = torch.zeros(batch_size, feature_dim, dtype=torch.float32, device=device)
    negative = torch.zeros(batch_size, feature_dim, dtype=torch.float32, device=device)
    
    # Only last element differs: anchor=1, positive=0, negative=0
    anchor[:, -1] = 1.0  # Last element = 1
    # Positive and negative remain 0 everywhere
    
    return [anchor, positive, negative]

```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `True`

#### Round 2 (未杀死)

**LLM 存活原因分析：**

> The mutant survives because previous tests didn't account for the GPU kernel's execution pattern. The mutation skips the last feature element (index 4095), but this is only executed by the last thread in the last warp. If feature_dim (4096) is perfectly divisible by warp_size (32) and ELEMENTS_PER_THREAD, the skipped element might not affect the final reduction if the warp reduction doesn't properly accumulate contributions from all threads. The mutant's behavior matches the original when the last thread's skipped contribution is numerically insignificant or when other threads' contributions mask the difference due to floating-point non-associativity in the reduction order.

**LLM 杀死策略：**

> Force the last feature element (index 4095) to carry a unique, large-magnitude value that cannot be masked by other elements, and ensure it's processed by a thread whose contributions cannot be compensated by other threads. Use extreme values to exploit floating-point sensitivity in the reduction, and also test with a non-power-of-two feature dimension to alter thread mapping.

**LLM 建议的输入代码：**

```python
import torch

def generate_inputs(device):
    # Test two scenarios in one batch
    batch_size = 128
    
    # First half of batch: original dimension 4096
    feature_dim1 = 4096
    anchor1 = torch.zeros(batch_size//2, feature_dim1, dtype=torch.float32, device=device)
    positive1 = torch.zeros(batch_size//2, feature_dim1, dtype=torch.float32, device=device)
    negative1 = torch.zeros(batch_size//2, feature_dim1, dtype=torch.float32, device=device)
    
    # Only last element differs dramatically
    anchor1[:, -1] = 1e20
    negative1[:, -1] = -1e20  # Creates huge squared difference for negative pair too
    
    # Second half of batch: non-power-of-two dimension 4095
    feature_dim2 = 4095
    anchor2 = torch.zeros(batch_size//2, feature_dim2, dtype=torch.float32, device=device)
    positive2 = torch.zeros(batch_size//2, feature_dim2, dtype=torch.float32, device=device)
    negative2 = torch.zeros(batch_size//2, feature_dim2, dtype=torch.float32, device=device)
    
    # Last element differs dramatically
    anchor2[:, -1] = 1e20
    negative2[:, -1] = -1e20
    
    # Concatenate both scenarios
    anchor = torch.cat([anchor1, anchor2], dim=0)
    positive = torch.cat([positive1, positive2], dim=0)
    negative = torch.cat([negative1, negative2], dim=0)
    
    return [anchor, positive, negative]

```

**GPU 验证结果：**

- ref_ok: `False`, original_ok: `False`, mutant_ok: `False`
- error: `llm_generate: Sizes of tensors must match except in dimension 0. Expected size 4096 but got size 4095 for tensor number 1 in the list.`

#### Round 3 (**KILLED**)

**LLM 存活原因分析：**

> The mutant survives because all previous attempts created inputs where the missing last element's contribution was either identical between positive/negative pairs or canceled out in the loss computation. The kernel computes squared Euclidean distances separately for anchor-positive and anchor-negative pairs, then likely uses these in a contrastive loss (e.g., Triplet Loss: max(0, margin + pos_dist - neg_dist)). For the mutant to be killed, the last element must create an asymmetry in the loss that the reduction and subsequent operations don't mask. Previous attempts failed because they either made the last element contribute equally to both distances (canceling in loss) or the values were extreme enough to cause NaN/inf in both original and mutant, making them equivalent in failure mode.

**LLM 杀死策略：**

> Exploit the non-linear nature of the contrastive loss by making the last element's omission flip the sign of (pos_dist - neg_dist) relative to the margin. Specifically, craft inputs where the last element alone makes pos_dist > neg_dist + margin in the original (loss > 0), but with it omitted, pos_dist < neg_dist + margin in the mutant (loss = 0). Use moderate values to avoid overflow and ensure the difference is detectable.

**LLM 建议的输入代码：**

```python
import torch

def generate_inputs(device):
    batch_size, feature_dim = 128, 4096
    # All zeros except last feature
    anchor = torch.zeros(batch_size, feature_dim, dtype=torch.float32, device=device)
    positive = torch.zeros(batch_size, feature_dim, dtype=torch.float32, device=device)
    negative = torch.zeros(batch_size, feature_dim, dtype=torch.float32, device=device)
    
    # Last element values that create asymmetric distances
    # Anchor differs more from positive than from negative (by square effect)
    anchor[:, -1] = 1.0          # Large difference to positive
    positive[:, -1] = 0.0        # Zero -> squared diff = 1.0
    negative[:, -1] = 0.9        # Close to anchor -> squared diff = 0.01
    
    # Ensure other elements are zero (already set)
    return [anchor, positive, negative]

```

**GPU 验证结果：**

- ref_ok: `True`, original_ok: `True`, mutant_ok: `False`
- diff: `max_diff=8.999999e-01, mean_diff=8.999999e-01, ref_range=[1.9000e+00,1.9000e+00], mut_range=[1.0000e+00,1.0000e+00]`

#### 测试构造规则

- **规则名**: `boundary_dominant_last_element`
- **描述**: This rule targets boundary-guard mutations by making the last element of the guarded dimension the sole contributor to the output. It sets all other elements to zero and the last element to a non‑zero value that dominates any subsequent reduction (e.g., sum, squared distance). For kernels processing multiple tensors (e.g., contrastive loss), apply the same pattern independently to each input, choosing values that create an asymmetry when the last element is skipped.
- **适用算子**: mask_boundary, relax_boundary, mask_wrap

```python
def policy(shape, dtype, device, rng):
    import torch
    tensor = torch.zeros(shape, dtype=dtype, device=device)
    # Set the last element of the last dimension to a dominating non‑zero value
    tensor[..., -1] = torch.tensor(1.0, dtype=dtype, device=device)
    return tensor
```


---