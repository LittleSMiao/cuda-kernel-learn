#pragma once
// ============================================================================
// custom_gemm_example.cuh —— 自定义 GEMM 实现模板
// ============================================================================
// 在这里实现你自己的 GEMM 算子。每个实现只需要写一个符合签名的函数：
//
//   void my_gemm(int M, int N, int K, float alpha,
//                const float *A, int lda,
//                const float *B, int ldb,
//                float beta,
//                float *C, int ldc);
//
// 然后在 main.cu 的 impls 数组中注册即可，框架会自动完成：
//   - 预热 + 计时
//   - 与 cuBLAS 的正确性对比
//   - GFLOPS 计算
//   - 相对 cuBLAS 的百分比评分
//
// **重要**：矩阵是列主序 (column-major)，和 cuBLAS / Fortran 一致。
//   C[ldc][N] = α * A[lda][K] * B[ldb][N] + β * C[ldc][N]
//   其中 A 实际是 M×K，B 是 K×N，C 是 M×N
//   lda >= M, ldb >= K, ldc >= M
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

// ---------------------------------------------------------------------------
// 示例 1：最朴素的 GEMM（每个线程算 C 的一个元素）
// 仅供教学参考，性能极差，但可以跑通正确性测试。
// ---------------------------------------------------------------------------
__global__ void naive_gemm_kernel(
    int M, int N, int K,
    float alpha, const float *A, int lda,
    const float *B, int ldb,
    float beta, float *C, int ldc)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            // 列主序：A[row + k * lda], B[k + col * ldb]
            sum += A[row + k * lda] * B[k + col * ldb];
        }
        C[row + col * ldc] = alpha * sum + beta * C[row + col * ldc];
    }
}

inline void naive_gemm(int M, int N, int K,
                       float alpha,
                       const float *A, int lda,
                       const float *B, int ldb,
                       float beta,
                       float *C, int ldc)
{
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    naive_gemm_kernel<<<grid, block>>>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

// ---------------------------------------------------------------------------
// 示例 2：带 shared memory tiling 的 GEMM
// 分块乘法，利用共享内存减少全局内存访问。
// ---------------------------------------------------------------------------
#define TILE_SIZE 16

__global__ void tiled_gemm_kernel(
    int M, int N, int K,
    float alpha, const float *A, int lda,
    const float *B, int ldb,
    float beta, float *C, int ldc)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // 协作加载 A 的 tile
        int a_row = row;
        int a_col = t * TILE_SIZE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (a_row < M && a_col < K)
            ? A[a_row + a_col * lda] : 0.0f;

        // 协作加载 B 的 tile
        int b_row = t * TILE_SIZE + threadIdx.y;
        int b_col = col;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && b_col < N)
            ? B[b_row + b_col * ldb] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row + col * ldc] = alpha * sum + beta * C[row + col * ldc];
    }
}

inline void tiled_gemm(int M, int N, int K,
                       float alpha,
                       const float *A, int lda,
                       const float *B, int ldb,
                       float beta,
                       float *C, int ldc)
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);
    tiled_gemm_kernel<<<grid, block>>>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

#undef TILE_SIZE

// ---------------------------------------------------------------------------
// 你可以在这里继续添加更多 GEMM 实现：
//   - float4 向量化加载
//   - double buffering (warp-level)
//   - Tensor Core (wmma / mma)
//   - 混合精度 (FP16 → FP32 accumulate)
//   ...
//
// 每个实现只需要：
//   1. 写一个 __global__ kernel
//   2. 写一个 inline launch 函数
//   3. 在 main.cu 中注册到 impls 数组
// ---------------------------------------------------------------------------

// 输入的blockDim是固定的数值 (256 / 8) ^ 2 = 32 ^ 2 = 1024
template <uint32_t tile_size = 128, uint32_t block_size_xy = 32, uint32_t step_size = 8>
__global__ void coarse_tiled_gemm_kernel(
    int M, int N, int K,
    float alpha, const float *A, int lda,
    const float *B, int ldb,
    float beta, float *C, int ldc) {
    constexpr uint32_t coarsing_factor = tile_size / block_size_xy;
    constexpr uint32_t block_size = block_size_xy * block_size_xy;
    float C_temp[coarsing_factor][coarsing_factor] = {0};

    const uint32_t steps = (K + step_size - 1) / step_size;

    __shared__ float Ads[tile_size][step_size];
    __shared__ float Bds[step_size][tile_size];

    for (uint32_t step = 0; step < steps; ++step) {
        // 加载 A
        // 使用 (4, 8) 的 warp 进行加载
        uint32_t A_start_y = blockIdx.y * tile_size;
   
#if 1
        // 直接使用threadIdx的思维进行加载
        // 这里的blockSize为 * 8
        constexpr uint32_t tile_total_size = tile_size * step_size;
        constexpr uint32_t block_size_y_tile_a = block_size / step_size;
#pragma unroll
        for (uint32_t b = 0; b < (tile_total_size + block_size - 1) / block_size; ++b) {
            // 这个地方是相对tile的索引
            uint32_t row = block_size_y_tile_a * b + threadIdx.x % block_size_y_tile_a;
            uint32_t col = threadIdx.x / block_size_y_tile_a;

            if (row < tile_size && col < step_size) {
                if (A_start_y + row < M && col + step * step_size < K) {
                    Ads[row][col] = A[(A_start_y + row) + (col + step * step_size) * lda];
                } else {
                    Ads[row][col] = 0;
                }
            }
        }
#else
        // 将warp进行reshape然后进行加载
        constexpr uint32_t warp_stride_x = step_size;
        constexpr uint32_t warp_stride_y = warpSize / step_size;
        constexpr uint32_t block_size_y = tile_size / warp_stride_y; // y维度的warp数量
#pragma unroll
        for (uint32_t i = 0; i < (tile_size * step_size + blockSize - 1) / blockSize; ++i) {
            uint32_t warpId = threadIdx.x / warpSize; // 先得到线性的 warpId
            uint32_t loaded_a_y = warpId * warp_stride_y + threadIdx.x / warp_stride_x;
            uint32_t loaded_a_x = threadIdx.x % warp_stride_x + step_size * step;
            Ads[loaded_a_y][loaded_a_x] = A[A_start_y + loaded_a_y + loaded_a_x * K];
        }
#endif
        // 加载 B
        // 使用(32, 1) 的 warp 进行加载
        uint32_t B_start_x = blockIdx.x * tile_size;
        constexpr uint32_t block_size_y_tile_b = step_size;
#pragma unroll
        for (uint32_t b = 0; b < (tile_total_size + block_size - 1) / block_size; ++b) {
            // 这边也是在block中的索引，也就是在Bds中的索引
            uint32_t row = threadIdx.x % block_size_y_tile_b;
            uint32_t col = threadIdx.x / block_size_y_tile_b + b * (block_size / block_size_y_tile_b);

            if (row < step_size && col < tile_size) {
                if (row + step * step_size < K && col + B_start_x < N) {
                    Bds[row][col] = B[(row + step * step_size) + (col + B_start_x) * ldb];
                } else {
                    Bds[row][col] = 0;
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (uint32_t p = 0; p < step_size; ++p) {
#pragma unroll
            for (uint32_t c_i = 0; c_i < coarsing_factor; ++c_i) {
                uint32_t c_row = c_i * block_size_xy + (threadIdx.x / block_size_xy);
#pragma unroll
                for (uint32_t c_j = 0; c_j < coarsing_factor; ++c_j) {
                    uint32_t c_col = c_j * block_size_xy + (threadIdx.x & (block_size_xy - 1));

                    C_temp[c_i][c_j] += Ads[c_row][p] * Bds[p][c_col];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (uint32_t c_i = 0; c_i < coarsing_factor; ++c_i) {
        uint32_t c_row = c_i * block_size_xy + (threadIdx.x / block_size_xy);
        uint32_t c_global_row = blockIdx.y * tile_size + c_row;
#pragma unroll
        for (uint32_t c_j = 0; c_j < coarsing_factor; ++c_j) {
            uint32_t c_col = c_j * block_size_xy + (threadIdx.x & (block_size_xy - 1));
            uint32_t c_global_col = blockIdx.x * tile_size + c_col;

            if (c_global_row < M && c_global_col < N) {
                uint32_t idx = c_global_row + c_global_col * ldc;
                C[idx] = alpha * C_temp[c_i][c_j] + beta * C[idx];
            }
        }
    }
}

// 使用宏来生成 launch 函数，避免 template-template parameter 在 nvcc 12.8 中
// Kernel<args...><<<grid, block>>> 的解析问题。
#define DEFINE_LAUNCH_COARSE_GEMM(launch_name, kernel_name)                   \
inline void launch_name(int M, int N, int K,                                 \
                       float alpha,                                          \
                       const float *A, int lda,                              \
                       const float *B, int ldb,                              \
                       float beta,                                           \
                       float *C, int ldc)                                    \
{                                                                             \
    const uint32_t max_dim = max(M, N);                                       \
    if (max_dim <= 256) {                                                     \
        constexpr uint32_t tile_size     = 16;                                \
        constexpr uint32_t block_size_xy = 16;                                \
        constexpr uint32_t step_size     = 16;                                \
        dim3 blk(block_size_xy * block_size_xy);                              \
        dim3 grd((N + tile_size - 1) / tile_size,                             \
                 (M + tile_size - 1) / tile_size);                            \
        kernel_name<tile_size, block_size_xy, step_size><<<grd, blk>>>(       \
            M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);                   \
    } else if (max_dim < 1024) {                                              \
        constexpr uint32_t tile_size     = 64;                                \
        constexpr uint32_t block_size_xy = 32;                                \
        dim3 blk(block_size_xy * block_size_xy);                              \
        dim3 grd((N + tile_size - 1) / tile_size,                             \
                 (M + tile_size - 1) / tile_size);                            \
        kernel_name<tile_size, block_size_xy, 8><<<grd, blk>>>(               \
            M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);                   \
    } else {                                                                  \
        constexpr uint32_t tile_size     = 128;                               \
        constexpr uint32_t block_size_xy = 32;                                \
        dim3 blk(block_size_xy * block_size_xy);                              \
        dim3 grd((N + tile_size - 1) / tile_size,                             \
                 (M + tile_size - 1) / tile_size);                            \
        kernel_name<tile_size, block_size_xy, 8><<<grd, blk>>>(               \
            M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);                   \
    }                                                                         \
}

template <uint32_t tile_size = 128, uint32_t block_size_xy = 32, uint32_t step_size = 8>
__global__ void coarse_padding_tiled_gemm_kernel(
    int M, int N, int K,
    float alpha, const float *A, int lda,
    const float *B, int ldb,
    float beta, float *C, int ldc) {
    constexpr uint32_t coarsing_factor = tile_size / block_size_xy;
    constexpr uint32_t block_size = block_size_xy * block_size_xy;
    float C_temp[coarsing_factor][coarsing_factor] = {0};

    const uint32_t steps = (K + step_size - 1) / step_size;

    __shared__ float Ads[tile_size][step_size + 1];
    __shared__ float Bds[step_size][tile_size + 1];

    for (uint32_t step = 0; step < steps; ++step) {
        // 加载 A
        // 使用 (4, 8) 的 warp 进行加载
        uint32_t A_start_y = blockIdx.y * tile_size;
   
        // 直接使用threadIdx的思维进行加载
        // 这里的blockSize为 * 8
        constexpr uint32_t tile_total_size = tile_size * step_size;
        constexpr uint32_t block_size_y_tile_a = block_size / step_size;
#pragma unroll
        for (uint32_t b = 0; b < (tile_total_size + block_size - 1) / block_size; ++b) {
            // 这个地方是相对tile的索引
            uint32_t row = block_size_y_tile_a * b + threadIdx.x % block_size_y_tile_a;
            uint32_t col = threadIdx.x / block_size_y_tile_a;

            // 这个地方使用swizzle的影响不是那么大，因为这个地方的bankconflict跟后面的 global memory是可以相互隐藏的
            if (row < tile_size && col < step_size) {
                if (A_start_y + row < M && col + step * step_size < K) {
                    Ads[row][col] = A[(A_start_y + row) + (col + step * step_size) * lda];
                } else {
                    Ads[row][col] = 0;
                }
            }
        }

        // 加载 B
        // 使用(32, 1) 的 warp 进行加载
        uint32_t B_start_x = blockIdx.x * tile_size;
        constexpr uint32_t block_size_y_tile_b = step_size;
#pragma unroll
        for (uint32_t b = 0; b < (tile_total_size + block_size - 1) / block_size; ++b) {
            // 这边也是在block中的索引，也就是在Bds中的索引
            uint32_t row = threadIdx.x % block_size_y_tile_b;
            uint32_t col = threadIdx.x / block_size_y_tile_b + b * (block_size / block_size_y_tile_b);

            if (row < step_size && col < tile_size) {
                if (row + step * step_size < K && col + B_start_x < N) {
                    Bds[row][col] = B[(row + step * step_size) + (col + B_start_x) * ldb];
                } else {
                    Bds[row][col] = 0;
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (uint32_t p = 0; p < step_size; ++p) {
#pragma unroll
            for (uint32_t c_i = 0; c_i < coarsing_factor; ++c_i) {
                uint32_t c_row = c_i * block_size_xy + (threadIdx.x / block_size_xy);
#pragma unroll
                for (uint32_t c_j = 0; c_j < coarsing_factor; ++c_j) {
                    uint32_t c_col = c_j * block_size_xy + (threadIdx.x & (block_size_xy - 1));

                    C_temp[c_i][c_j] += Ads[c_row][p] * Bds[p][c_col];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (uint32_t c_i = 0; c_i < coarsing_factor; ++c_i) {
        uint32_t c_row = c_i * block_size_xy + (threadIdx.x / block_size_xy);
        uint32_t c_global_row = blockIdx.y * tile_size + c_row;
#pragma unroll
        for (uint32_t c_j = 0; c_j < coarsing_factor; ++c_j) {
            uint32_t c_col = c_j * block_size_xy + (threadIdx.x & (block_size_xy - 1));
            uint32_t c_global_col = blockIdx.x * tile_size + c_col;

            if (c_global_row < M && c_global_col < N) {
                uint32_t idx = c_global_row + c_global_col * ldc;
                C[idx] = alpha * C_temp[c_i][c_j] + beta * C[idx];
            }
        }
    }
}

template <uint32_t tile_size = 128, uint32_t block_size_xy = 32, uint32_t step_size = 8>
__global__ void coarse_register_load_tiled_gemm_kernel(
    int M, int N, int K,
    float alpha, const float *A, int lda,
    const float *B, int ldb,
    float beta, float *C, int ldc) {
    constexpr uint32_t coarsing_factor = tile_size / block_size_xy;
    constexpr uint32_t block_size = block_size_xy * block_size_xy;
    float C_temp[coarsing_factor][coarsing_factor] = {0};
    float tempA[coarsing_factor];
    float tempB[coarsing_factor];

    const uint32_t steps = (K + step_size - 1) / step_size;

    __shared__ float Ads[tile_size][step_size];
    __shared__ float Bds[step_size][tile_size];

    for (uint32_t step = 0; step < steps; ++step) {
        // 加载 A
        // 使用 (4, 8) 的 warp 进行加载
        uint32_t A_start_y = blockIdx.y * tile_size;
   
        // 直接使用threadIdx的思维进行加载
        // 这里的blockSize为 * 8
        constexpr uint32_t tile_total_size = tile_size * step_size;
        constexpr uint32_t block_size_y_tile_a = block_size / step_size;
#pragma unroll
        for (uint32_t b = 0; b < (tile_total_size + block_size - 1) / block_size; ++b) {
            // 这个地方是相对tile的索引
            uint32_t row = block_size_y_tile_a * b + threadIdx.x % block_size_y_tile_a;
            uint32_t col = threadIdx.x / block_size_y_tile_a;

            // 这个地方使用swizzle的影响不是那么大，因为这个地方的bankconflict跟后面的 global memory是可以相互隐藏的
            if (row < tile_size && col < step_size) {
                if (A_start_y + row < M && col + step * step_size < K) {
                    Ads[row][col] = A[(A_start_y + row) + (col + step * step_size) * lda];
                } else {
                    Ads[row][col] = 0;
                }
            }
        }

        // 加载 B
        // 使用(32, 1) 的 warp 进行加载
        uint32_t B_start_x = blockIdx.x * tile_size;
        constexpr uint32_t block_size_y_tile_b = step_size;
#pragma unroll
        for (uint32_t b = 0; b < (tile_total_size + block_size - 1) / block_size; ++b) {
            // 这边也是在block中的索引，也就是在Bds中的索引
            uint32_t row = threadIdx.x % block_size_y_tile_b;
            uint32_t col = threadIdx.x / block_size_y_tile_b + b * (block_size / block_size_y_tile_b);

            if (row < step_size && col < tile_size) {
                if (row + step * step_size < K && col + B_start_x < N) {
                    Bds[row][col] = B[(row + step * step_size) + (col + B_start_x) * ldb];
                } else {
                    Bds[row][col] = 0;
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (uint32_t p = 0; p < step_size; ++p) {
            // 这一步先加载到 register 里面，同时为了后续的float4优化，需要将Ads也进行加载
            // 这一步骤可以减少从 shared memory 中的访问

            // 加载 A
#pragma unroll
            for (uint32_t i = 0; i < coarsing_factor; ++i) {
                uint32_t c_row = i * block_size_xy + (threadIdx.x / block_size_xy);
                tempA[i] = Ads[c_row][p];
            }

            // 加载 B
#pragma unroll
            for (uint32_t i = 0; i < coarsing_factor; ++i) {
                uint32_t c_col = i * block_size_xy + (threadIdx.x & (block_size_xy - 1));
                tempB[i] = Bds[p][c_col];
            }

#pragma unroll
            for (uint32_t i = 0; i < coarsing_factor; ++i) {
#pragma unroll
                for (uint32_t j = 0; j < coarsing_factor; ++j) {
                    C_temp[i][j] += tempA[i] * tempB[j];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (uint32_t c_i = 0; c_i < coarsing_factor; ++c_i) {
        uint32_t c_row = c_i * block_size_xy + (threadIdx.x / block_size_xy);
        uint32_t c_global_row = blockIdx.y * tile_size + c_row;
#pragma unroll
        for (uint32_t c_j = 0; c_j < coarsing_factor; ++c_j) {
            uint32_t c_col = c_j * block_size_xy + (threadIdx.x & (block_size_xy - 1));
            uint32_t c_global_col = blockIdx.x * tile_size + c_col;

            if (c_global_row < M && c_global_col < N) {
                uint32_t idx = c_global_row + c_global_col * ldc;
                C[idx] = alpha * C_temp[c_i][c_j] + beta * C[idx];
            }
        }
    }
}

// 这个kernel调整一下再block内的warp布局(4*8，使得整个warp整体的计算密度提高)
// 这里实际使用的blockSize为 block_size_xy * block_size_xy
// 思想就是 block 的形状尽量贴近应用
// warp的形状可以适当的贴近优化
template <uint32_t tile_size = 128, uint32_t block_size_xy = 32, uint32_t step_size = 8>
__global__ void coarse_warp_load_tiled_gemm_kernel(
    int M, int N, int K,
    float alpha, const float *A, int lda,
    const float *B, int ldb,
    float beta, float *C, int ldc) {
    constexpr uint32_t coarsing_factor = tile_size / block_size_xy;
    constexpr uint32_t block_size = block_size_xy * block_size_xy;
    float C_temp[coarsing_factor][coarsing_factor] = {0};
    float tempA[coarsing_factor];
    float tempB[coarsing_factor];

    const uint32_t steps = (K + step_size - 1) / step_size;

    __shared__ float Ads[tile_size][step_size];
    __shared__ float Bds[step_size][tile_size];

    // 这个地方我们最终的目的还是需要相对 block 的坐标
    // warp坐标在当前这里只是辅助我们做计算

    uint32_t warp_id = threadIdx.x / warpSize;
    constexpr uint32_t warp_size_x = block_size_xy / 8;
    uint32_t warp_id_y_c = warp_id / warp_size_x;
    uint32_t warp_id_x_c = warp_id % warp_size_x;
    uint32_t landId = threadIdx.x % warpSize;

    // 这个地方计算出来线程在 block 里的相对坐标
    uint32_t thread_id_y_c = warp_id_y_c * 4 + landId / 8;
    uint32_t thread_id_x_c = warp_id_x_c * 8 + landId % 8;

    for (uint32_t step = 0; step < steps; ++step) {
        // 加载 A
        // 使用 (4, 8) 的 warp 进行加载
        uint32_t A_start_y = blockIdx.y * tile_size;
   
        // 直接使用threadIdx的思维进行加载
        // 这里的blockSize为 * 8
        constexpr uint32_t tile_total_size = tile_size * step_size;
        constexpr uint32_t block_size_y_tile_a = block_size / step_size;
#pragma unroll
        for (uint32_t b = 0; b < (tile_total_size + block_size - 1) / block_size; ++b) {
            // 这个地方是相对tile的索引
            uint32_t row = block_size_y_tile_a * b + threadIdx.x % block_size_y_tile_a;
            uint32_t col = threadIdx.x / block_size_y_tile_a;

            // 这个地方使用swizzle的影响不是那么大，因为这个地方的bankconflict跟后面的 global memory是可以相互隐藏的
            if (row < tile_size && col < step_size) {
                if (A_start_y + row < M && col + step * step_size < K) {
                    Ads[row][col] = A[(A_start_y + row) + (col + step * step_size) * lda];
                } else {
                    Ads[row][col] = 0;
                }
            }
        }

        // 加载 B
        // 使用(32, 1) 的 warp 进行加载
        uint32_t B_start_x = blockIdx.x * tile_size;
        constexpr uint32_t block_size_y_tile_b = step_size;
#pragma unroll
        for (uint32_t b = 0; b < (tile_total_size + block_size - 1) / block_size; ++b) {
            // 这边也是在block中的索引，也就是在Bds中的索引
            uint32_t row = threadIdx.x % block_size_y_tile_b;
            uint32_t col = threadIdx.x / block_size_y_tile_b + b * (block_size / block_size_y_tile_b);

            if (row < step_size && col < tile_size) {
                if (row + step * step_size < K && col + B_start_x < N) {
                    Bds[row][col] = B[(row + step * step_size) + (col + B_start_x) * ldb];
                } else {
                    Bds[row][col] = 0;
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (uint32_t p = 0; p < step_size; ++p) {
            // 这一步先加载到 register 里面，同时为了后续的float4优化，需要将Ads也进行加载
            // 这一步骤可以减少从 shared memory 中的访问

            // 加载 A
#pragma unroll
            for (uint32_t i = 0; i < coarsing_factor; ++i) {
                // block坐标映射到tile坐标
                uint32_t c_row = i * block_size_xy + thread_id_y_c;
                tempA[i] = Ads[c_row][p];
            }

            // 加载 B
#pragma unroll
            for (uint32_t i = 0; i < coarsing_factor; ++i) {
                uint32_t c_col = i * block_size_xy + thread_id_x_c;
                tempB[i] = Bds[p][c_col];
            }

#pragma unroll
            for (uint32_t i = 0; i < coarsing_factor; ++i) {
#pragma unroll
                for (uint32_t j = 0; j < coarsing_factor; ++j) {
                    C_temp[i][j] += tempA[i] * tempB[j];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (uint32_t c_i = 0; c_i < coarsing_factor; ++c_i) {
        // 相对block坐标
        uint32_t c_row = c_i * block_size_xy + thread_id_y_c;
        // 相对 global 坐标
        uint32_t c_global_row = blockIdx.y * tile_size + c_row;
#pragma unroll
        for (uint32_t c_j = 0; c_j < coarsing_factor; ++c_j) {
            uint32_t c_col = c_j * block_size_xy + thread_id_x_c;
            uint32_t c_global_col = blockIdx.x * tile_size + c_col;

            if (c_global_row < M && c_global_col < N) {
                uint32_t idx = c_global_row + c_global_col * ldc;
                C[idx] = alpha * C_temp[c_i][c_j] + beta * C[idx];
            }
        }
    }
}

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

template <uint32_t tile_size = 128, uint32_t block_size_xy = 32, uint32_t step_size = 8>
__global__ void coarse_float4_load_tiled_gemm_kernel(
    int M, int N, int K,
    float alpha, const float *A, int lda,
    const float *B, int ldb,
    float beta, float *C, int ldc) {
    constexpr uint32_t coarsing_factor = tile_size / block_size_xy;
    constexpr uint32_t block_size = block_size_xy * block_size_xy;
    float C_temp[coarsing_factor][coarsing_factor] = {0};
    float tempA[coarsing_factor];
    float tempB[coarsing_factor];

    const uint32_t steps = (K + step_size - 1) / step_size;

    // 为了使用 float4 来加载 Ads，这里使用列主序储存 Ads
    __shared__ float Ads[step_size][tile_size];
    __shared__ float Bds[step_size][tile_size];

    uint32_t warp_id = threadIdx.x / warpSize;
    constexpr uint32_t warp_size_x = block_size_xy / 8;
    uint32_t warp_id_y_c = warp_id / warp_size_x;
    uint32_t warp_id_x_c = warp_id % warp_size_x;
    uint32_t landId = threadIdx.x % warpSize;

    // 在 mapping 之后，相对于block的坐标
    uint32_t thread_idx_y_c = warp_id_y_c * 4 + landId / 8;
    uint32_t thread_idx_x_c = warp_id_x_c * 8 + landId % 8;

    for (uint32_t step = 0; step < steps; ++step) {
        // 加载 A
        // 使用 (4, 8) 的 warp 进行加载
        uint32_t A_start_y = blockIdx.y * tile_size;
   
        // 直接使用threadIdx的思维进行加载
        // 这里的blockSize为 * 8
        constexpr uint32_t tile_total_size = tile_size * step_size;
        constexpr uint32_t block_size_y_tile_a = block_size / step_size;
#pragma unroll
        for (uint32_t b = 0; b < (tile_total_size + block_size - 1) / block_size; ++b) {
            // 这个地方是相对tile的索引
            uint32_t row = block_size_y_tile_a * b + threadIdx.x % block_size_y_tile_a;
            uint32_t col = threadIdx.x / block_size_y_tile_a;

            if (row < tile_size && col < step_size) {
                if (A_start_y + row < M && col + step * step_size < K) {
                    Ads[col][row] = A[(A_start_y + row) + (col + step * step_size) * lda];
                } else {
                    Ads[col][row] = 0;
                }
            }
        }

        // 加载 B
        // 使用(32, 1) 的 warp 进行加载
        uint32_t B_start_x = blockIdx.x * tile_size;
        constexpr uint32_t block_size_y_tile_b = step_size;
#pragma unroll
        for (uint32_t b = 0; b < (tile_total_size + block_size - 1) / block_size; ++b) {
            // 这边也是在block中的索引，也就是在Bds中的索引
            uint32_t row = threadIdx.x % block_size_y_tile_b;
            uint32_t col = threadIdx.x / block_size_y_tile_b + b * (block_size / block_size_y_tile_b);

            if (row < step_size && col < tile_size) {
                if (row + step * step_size < K && col + B_start_x < N) {
                    Bds[row][col] = B[(row + step * step_size) + (col + B_start_x) * ldb];
                } else {
                    Bds[row][col] = 0;
                }
            }
        }

        __syncthreads();

        // 这个地方使用float4 先加载到register中
        // 因此需要调整一下数据的mapping结构
        // 每个线程负责的单位就是一个 4 * 4 的小块了
        static_assert(coarsing_factor >= 4);

#pragma unroll
        for (uint32_t p = 0; p < step_size; ++p) {
            // 这一步先加载到 register 里面，同时为了后续的float4优化，需要将Ads也进行加载
            // 这一步骤可以减少从 shared memory 中的访问

            // 加载 A
#pragma unroll
            for (uint32_t i = 0; i < coarsing_factor / 4; ++i) {
                // 这里计算的是让当前线程加载的那一行数据，只影响线程加载函数的排布，相邻两个加载数据其实地址的间隔是相等的
                uint32_t c_row = (i * block_size_xy + thread_idx_y_c) * 4;
                FLOAT4(tempA[i * 4]) = FLOAT4(Ads[p][c_row]);
            }

            // 加载 B
#pragma unroll
            for (uint32_t i = 0; i < coarsing_factor / 4; ++i) {
                uint32_t c_col = (i * block_size_xy + thread_idx_x_c) * 4;
                FLOAT4(tempB[i * 4]) = FLOAT4(Bds[p][c_col]);
            }

#pragma unroll
            for (uint32_t i = 0; i < coarsing_factor; ++i) {
#pragma unroll
                for (uint32_t j = 0; j < coarsing_factor; ++j) {
                    C_temp[i][j] += tempA[i] * tempB[j];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (uint32_t c_i = 0; c_i < coarsing_factor; ++c_i) {
        // 这个地方计算的是将tempA的坐标mapping到 tile_c中的相对坐标
        uint32_t c_row = (c_i / 4) * (4 * block_size_xy) +  thread_idx_y_c * 4 + c_i % 4;
        uint32_t c_global_row = blockIdx.y * tile_size + c_row;
#pragma unroll
        for (uint32_t c_j = 0; c_j < coarsing_factor; ++c_j) {
            // 这个地方计算的是在 tile_c 中的相对坐标
            uint32_t c_col = (c_j / 4) * (4 * block_size_xy) + thread_idx_x_c * 4 + c_j % 4;
            uint32_t c_global_col = blockIdx.x * tile_size + c_col;

            if (c_global_row < M && c_global_col < N) {
                uint32_t idx = c_global_row + c_global_col * ldc;
                C[idx] = alpha * C_temp[c_i][c_j] + beta * C[idx];
            }
        }
    }
}

// 为每个 coarse gemm kernel 实例化 launch 函数
DEFINE_LAUNCH_COARSE_GEMM(launch_coarse_tiled_gemm, coarse_tiled_gemm_kernel)
DEFINE_LAUNCH_COARSE_GEMM(launch_coarse_padding_tiled_gemm, coarse_padding_tiled_gemm_kernel)
DEFINE_LAUNCH_COARSE_GEMM(launch_coarse_register_tiled_gemm, coarse_register_load_tiled_gemm_kernel)
DEFINE_LAUNCH_COARSE_GEMM(launch_coarse_warp_tiled_gemm, coarse_warp_load_tiled_gemm_kernel)

// float4 kernel 需要 coarsing_factor >= 4，只使用大 tile 配置
inline void launch_coarse_float4_tiled_gemm(int M, int N, int K,
                       float alpha,
                       const float *A, int lda,
                       const float *B, int ldb,
                       float beta,
                       float *C, int ldc)
{
    constexpr uint32_t tile_size     = 128;
    constexpr uint32_t block_size_xy = 32;
    constexpr uint32_t step_size     = 8;

    dim3 blk(block_size_xy * block_size_xy);
    dim3 grd((N + tile_size - 1) / tile_size,
             (M + tile_size - 1) / tile_size);
    coarse_float4_load_tiled_gemm_kernel<tile_size, block_size_xy, step_size><<<grd, blk>>>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <uint32_t tile_size = 128, uint32_t block_size_xy = 32, uint32_t step_size = 8>
__global__ void coarse_float4_sizzle_load_tiled_gemm_kernel(
    int M, int N, int K,
    float alpha, const float *A, int lda,
    const float *B, int ldb,
    float beta, float *C, int ldc) {
    constexpr uint32_t coarsing_factor = tile_size / block_size_xy;
    constexpr uint32_t block_size = block_size_xy * block_size_xy;
    float C_temp[coarsing_factor][coarsing_factor] = {0};
    float tempA[coarsing_factor];
    float tempB[coarsing_factor];

    const uint32_t steps = (K + step_size - 1) / step_size;

    __shared__ float Ads[step_size][tile_size];
    __shared__ float Bds[step_size][tile_size];

    uint32_t warp_id = threadIdx.x / warpSize;
    constexpr uint32_t warp_size_x = block_size_xy / 8;
    uint32_t warp_id_y_c = warp_id / warp_size_x;
    uint32_t warp_id_x_c = warp_id % warp_size_x;
    uint32_t landId = threadIdx.x % warpSize;

    uint32_t thread_idx_y_c = warp_id_y_c * 4 + landId / 8;
    uint32_t thread_idx_x_c = warp_id_x_c * 8 + landId % 8;

    for (uint32_t step = 0; step < steps; ++step) {
        uint32_t A_start_y = blockIdx.y * tile_size;
   
        constexpr uint32_t tile_total_size = tile_size * step_size;
        constexpr uint32_t block_size_y_tile_a = block_size / step_size;
#pragma unroll
        for (uint32_t b = 0; b < (tile_total_size + block_size - 1) / block_size; ++b) {
            uint32_t row = block_size_y_tile_a * b + threadIdx.x % block_size_y_tile_a;
            uint32_t col = threadIdx.x / block_size_y_tile_a;

            if (row < tile_size && col < step_size) {
                if (A_start_y + row < M && col + step * step_size < K) {
                    Ads[col][row] = A[(A_start_y + row) + (col + step * step_size) * lda];
                } else {
                    Ads[col][row] = 0;
                }
            }
        }

        uint32_t B_start_x = blockIdx.x * tile_size;
        constexpr uint32_t block_size_y_tile_b = step_size;
#pragma unroll
        for (uint32_t b = 0; b < (tile_total_size + block_size - 1) / block_size; ++b) {
            uint32_t row = threadIdx.x % block_size_y_tile_b;
            uint32_t col = threadIdx.x / block_size_y_tile_b + b * (block_size / block_size_y_tile_b);

            // 自家在B的时候有bank conflict，故这个地方使用swizzle的方法来解决
            if (row < step_size && col < tile_size) {
                if (row + step * step_size < K && col + B_start_x < N) {
                    Bds[row][col ^ (row * 8)] = B[(row + step * step_size) + (col + B_start_x) * ldb];
                } else {
                    Bds[row][col ^ (row * 8)] = 0;
                }
            }
        }

        __syncthreads();

        static_assert(coarsing_factor >= 4);

#pragma unroll
        for (uint32_t p = 0; p < step_size; ++p) {
#pragma unroll
            for (uint32_t i = 0; i < coarsing_factor / 4; ++i) {
                uint32_t c_row = (i * block_size_xy + thread_idx_y_c) * 4;
                FLOAT4(tempA[i * 4]) = FLOAT4(Ads[p][c_row]);
            }

#pragma unroll
            for (uint32_t i = 0; i < coarsing_factor / 4; ++i) {
                uint32_t c_col = (i * block_size_xy + thread_idx_x_c) * 4;
                FLOAT4(tempB[i * 4]) = FLOAT4(Bds[p][c_col ^ (p * 8)]);
            }

#pragma unroll
            for (uint32_t i = 0; i < coarsing_factor; ++i) {
#pragma unroll
                for (uint32_t j = 0; j < coarsing_factor; ++j) {
                    C_temp[i][j] += tempA[i] * tempB[j];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (uint32_t c_i = 0; c_i < coarsing_factor; ++c_i) {
        uint32_t c_row = (c_i / 4) * (4 * block_size_xy) +  thread_idx_y_c * 4 + c_i % 4;
        uint32_t c_global_row = blockIdx.y * tile_size + c_row;
#pragma unroll
        for (uint32_t c_j = 0; c_j < coarsing_factor; ++c_j) {
            uint32_t c_col = (c_j / 4) * (4 * block_size_xy) + thread_idx_x_c * 4 + c_j % 4;
            uint32_t c_global_col = blockIdx.x * tile_size + c_col;

            if (c_global_row < M && c_global_col < N) {
                uint32_t idx = c_global_row + c_global_col * ldc;
                C[idx] = alpha * C_temp[c_i][c_j] + beta * C[idx];
            }
        }
    }
}

inline void launch_coarse_float4_swizzle_tiled_gemm(int M, int N, int K,
                       float alpha,
                       const float *A, int lda,
                       const float *B, int ldb,
                       float beta,
                       float *C, int ldc)
{
    constexpr uint32_t tile_size     = 128;
    constexpr uint32_t block_size_xy = 32;
    constexpr uint32_t step_size     = 8;

    dim3 blk(block_size_xy * block_size_xy);
    dim3 grd((N + tile_size - 1) / tile_size,
             (M + tile_size - 1) / tile_size);
    coarse_float4_sizzle_load_tiled_gemm_kernel<tile_size, block_size_xy, step_size><<<grd, blk>>>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <uint32_t tile_size = 128, uint32_t block_size_xy = 32, uint32_t step_size = 8>
__global__ void coarse_zorder_tiled_gemm_kernel(
    int M, int N, int K,
    float alpha, const float *A, int lda,
    const float *B, int ldb,
    float beta, float *C, int ldc) {
    constexpr uint32_t coarsing_factor = tile_size / block_size_xy;
    constexpr uint32_t block_size = block_size_xy * block_size_xy;
    float C_temp[coarsing_factor][coarsing_factor] = {0};
    float tempA[coarsing_factor];
    float tempB[coarsing_factor];

    const uint32_t steps = (K + step_size - 1) / step_size;

    __shared__ float Ads[step_size][tile_size];
    __shared__ float Bds[step_size][tile_size];

    uint32_t warp_id = threadIdx.x / warpSize;
    constexpr uint32_t warp_size_x = block_size_xy / 8;
    uint32_t warp_id_y_c = warp_id / warp_size_x;
    uint32_t warp_id_x_c = warp_id % warp_size_x;
    uint32_t laneId = threadIdx.x % warpSize;

    // 这个地方需要改变一下映射到 block中的相对索引
    // laneid / 16 获取的室在上半个half warp 还是下半个 half warp，*2表示高度为2
    uint32_t thread_idx_y_c = warp_id_y_c * 4 + (laneId / 16) * 2 + (laneId % 2);
    // 保证了两个相邻的线程加载的室同一个数据，可以广播
    // laneId%16表示在half warp的索引，然后一列两个，/2就是在x方向的索引
    uint32_t thread_idx_x_c = warp_id_x_c * 8 + (laneId % 16) / 2;

    for (uint32_t step = 0; step < steps; ++step) {
        uint32_t A_start_y = blockIdx.y * tile_size;
   
        constexpr uint32_t tile_total_size = tile_size * step_size;
        constexpr uint32_t block_size_y_tile_a = block_size / step_size;
#pragma unroll
        for (uint32_t b = 0; b < (tile_total_size + block_size - 1) / block_size; ++b) {
            uint32_t row = block_size_y_tile_a * b + threadIdx.x % block_size_y_tile_a;
            uint32_t col = threadIdx.x / block_size_y_tile_a;

            if (row < tile_size && col < step_size) {
                if (A_start_y + row < M && col + step * step_size < K) {
                    Ads[col][row] = A[(A_start_y + row) + (col + step * step_size) * lda];
                } else {
                    Ads[col][row] = 0;
                }
            }
        }

        uint32_t B_start_x = blockIdx.x * tile_size;
        constexpr uint32_t block_size_y_tile_b = step_size;
#pragma unroll
        for (uint32_t b = 0; b < (tile_total_size + block_size - 1) / block_size; ++b) {
            uint32_t row = threadIdx.x % block_size_y_tile_b;
            uint32_t col = threadIdx.x / block_size_y_tile_b + b * (block_size / block_size_y_tile_b);

            if (row < step_size && col < tile_size) {
                if (row + step * step_size < K && col + B_start_x < N) {
                    Bds[row][col ^ (row * 8)] = B[(row + step * step_size) + (col + B_start_x) * ldb];
                } else {
                    Bds[row][col ^ (row * 8)] = 0;
                }
            }
        }

        __syncthreads();

        static_assert(coarsing_factor >= 4);

#pragma unroll
        for (uint32_t p = 0; p < step_size; ++p) {
#pragma unroll
            for (uint32_t i = 0; i < coarsing_factor / 4; ++i) {
                // 这里考虑的是相邻thread
                // 在加载A的时候，基于以上我们的分析，使用的warpsize是4*8，正好每一行代表的是一个quater warp
                // 对于每个quater warp，加载的都是同一个float4，触发了广播
                // 因此这里需要两个 transaction
                uint32_t c_row = (i * block_size_xy + thread_idx_y_c) * 4;
                FLOAT4(tempA[i * 4]) = FLOAT4(Ads[p][c_row]);
            }

#pragma unroll
            for (uint32_t i = 0; i < coarsing_factor / 4; ++i) {
                // 在改动前，每个相邻的thread访问的向量是并列的
                // 导致需要4个transaction
                uint32_t c_col = (i * block_size_xy + thread_idx_x_c) * 4;
                FLOAT4(tempB[i * 4]) = FLOAT4(Bds[p][c_col ^ (p * 8)]);
            }

#pragma unroll
            for (uint32_t i = 0; i < coarsing_factor; ++i) {
#pragma unroll
                for (uint32_t j = 0; j < coarsing_factor; ++j) {
                    C_temp[i][j] += tempA[i] * tempB[j];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (uint32_t c_i = 0; c_i < coarsing_factor; ++c_i) {
        uint32_t c_row = (c_i / 4) * (4 * block_size_xy) +  thread_idx_y_c * 4 + c_i % 4;
        uint32_t c_global_row = blockIdx.y * tile_size + c_row;
#pragma unroll
        for (uint32_t c_j = 0; c_j < coarsing_factor; ++c_j) {
            uint32_t c_col = (c_j / 4) * (4 * block_size_xy) + thread_idx_x_c * 4 + c_j % 4;
            uint32_t c_global_col = blockIdx.x * tile_size + c_col;

            if (c_global_row < M && c_global_col < N) {
                uint32_t idx = c_global_row + c_global_col * ldc;
                C[idx] = alpha * C_temp[c_i][c_j] + beta * C[idx];
            }
        }
    }
}

inline void launch_coarse_zorder_tiled_gemm(int M, int N, int K,
                       float alpha,
                       const float *A, int lda,
                       const float *B, int ldb,
                       float beta,
                       float *C, int ldc)
{
    constexpr uint32_t tile_size     = 128;
    constexpr uint32_t block_size_xy = 16;
    constexpr uint32_t step_size     = 8;

    dim3 blk(block_size_xy * block_size_xy);
    dim3 grd((N + tile_size - 1) / tile_size,
             (M + tile_size - 1) / tile_size);
    coarse_zorder_tiled_gemm_kernel<tile_size, block_size_xy, step_size><<<grd, blk>>>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <uint32_t tile_size = 128, uint32_t block_size_xy = 32, uint32_t step_size = 8>
__global__ void coarse_double_buffer_tiled_gemm_kernel(
    int M, int N, int K,
    float alpha, const float *A, int lda,
    const float *B, int ldb,
    float beta, float *C, int ldc) {
    constexpr uint32_t coarsing_factor = tile_size / block_size_xy;
    constexpr uint32_t block_size = block_size_xy * block_size_xy;
    float C_temp[coarsing_factor][coarsing_factor] = {0};
    float tempA[2][coarsing_factor];
    float tempB[2][coarsing_factor];

    const uint32_t steps = (K + step_size - 1) / step_size;

    __shared__ float Ads[2][step_size][tile_size];
    __shared__ float Bds[2][step_size][tile_size];

    uint32_t warp_id = threadIdx.x / warpSize;
    constexpr uint32_t warp_size_x = block_size_xy / 8;
    uint32_t warp_id_y_c = warp_id / warp_size_x;
    uint32_t warp_id_x_c = warp_id % warp_size_x;
    uint32_t laneId = threadIdx.x % warpSize;

    // 这个地方需要改变一下映射到 block中的相对索引
    // laneid / 16 获取的室在上半个half warp 还是下半个 half warp，*2表示高度为2
    uint32_t thread_idx_y_c = warp_id_y_c * 4 + (laneId / 16) * 2 + (laneId % 2);
    // 保证了两个相邻的线程加载的室同一个数据，可以广播
    // laneId%16表示在half warp的索引，然后一列两个，/2就是在x方向的索引
    uint32_t thread_idx_x_c = warp_id_x_c * 8 + (laneId % 16) / 2;

    for (uint32_t step = 0; step < steps + 1; ++step) {
        uint32_t A_start_y = blockIdx.y * tile_size;
   
        if (step < steps) {
            constexpr uint32_t tile_total_size = tile_size * step_size;
            constexpr uint32_t block_size_y_tile_a = block_size / step_size;
#pragma unroll
            for (uint32_t b = 0; b < (tile_total_size + block_size - 1) / block_size; ++b) {
                uint32_t row = block_size_y_tile_a * b + threadIdx.x % block_size_y_tile_a;
                uint32_t col = threadIdx.x / block_size_y_tile_a;

                if (row < tile_size && col < step_size) {
                    if (A_start_y + row < M && col + step * step_size < K) {
                        Ads[step & 1][col][row] = A[(A_start_y + row) + (col + step * step_size) * lda];
                    } else {
                        Ads[step & 1][col][row] = 0;
                    }
                }
            }

            uint32_t B_start_x = blockIdx.x * tile_size;
            constexpr uint32_t block_size_y_tile_b = step_size;
#pragma unroll
            for (uint32_t b = 0; b < (tile_total_size + block_size - 1) / block_size; ++b) {
                uint32_t row = threadIdx.x % block_size_y_tile_b;
                uint32_t col = threadIdx.x / block_size_y_tile_b + b * (block_size / block_size_y_tile_b);

                if (row < step_size && col < tile_size) {
                    if (row + step * step_size < K && col + B_start_x < N) {
                        Bds[step & 1][row][col ^ (row * 8)] = B[(row + step * step_size) + (col + B_start_x) * ldb];
                    } else {
                        Bds[step & 1][row][col ^ (row * 8)] = 0;
                    }
                }
            }
        }

        if (step > 0) {
#pragma unroll
            for (uint32_t p = 0; p < step_size + 1; ++p) {
                if (p < step_size) {
#pragma unroll
                    for (uint32_t i = 0; i < coarsing_factor / 4; ++i) {
                        // 这里考虑的是相邻thread
                        // 在加载A的时候，基于以上我们的分析，使用的warpsize是4*8，正好每一行代表的是一个quater warp
                        // 对于每个quater warp，加载的都是同一个float4，触发了广播
                        // 因此这里需要两个 transaction
                        uint32_t c_row = (i * block_size_xy + thread_idx_y_c) * 4;
                        FLOAT4(tempA[p & 1][i * 4]) = FLOAT4(Ads[(step + 1) & 1][p][c_row]);
                    }

#pragma unroll
                    for (uint32_t i = 0; i < coarsing_factor / 4; ++i) {
                        // 在改动前，每个相邻的thread访问的向量是并列的
                        // 导致需要4个transaction
                        uint32_t c_col = (i * block_size_xy + thread_idx_x_c) * 4;
                        FLOAT4(tempB[p & 1][i * 4]) = FLOAT4(Bds[(step + 1) & 1][p][c_col ^ (p * 8)]);
                    }
                }

                if (p > 0) {
#pragma unroll
                    for (uint32_t i = 0; i < coarsing_factor; ++i) {
#pragma unroll
                        for (uint32_t j = 0; j < coarsing_factor; ++j) {
                            C_temp[i][j] += tempA[(p + 1) & 1][i] * tempB[(p + 1) & 1][j];
                        }
                    }
                }
            }

        }

        __syncthreads();
    }

#pragma unroll
    for (uint32_t c_i = 0; c_i < coarsing_factor; ++c_i) {
        uint32_t c_row = (c_i / 4) * (4 * block_size_xy) +  thread_idx_y_c * 4 + c_i % 4;
        uint32_t c_global_row = blockIdx.y * tile_size + c_row;
#pragma unroll
        for (uint32_t c_j = 0; c_j < coarsing_factor; ++c_j) {
            uint32_t c_col = (c_j / 4) * (4 * block_size_xy) + thread_idx_x_c * 4 + c_j % 4;
            uint32_t c_global_col = blockIdx.x * tile_size + c_col;

            if (c_global_row < M && c_global_col < N) {
                uint32_t idx = c_global_row + c_global_col * ldc;
                C[idx] = alpha * C_temp[c_i][c_j] + beta * C[idx];
            }
        }
    }
}

inline void launch_coarse_double_buffer_tiled_gemm(int M, int N, int K,
                       float alpha,
                       const float *A, int lda,
                       const float *B, int ldb,
                       float beta,
                       float *C, int ldc)
{
    constexpr uint32_t tile_size     = 128;
    constexpr uint32_t block_size_xy = 16;
    constexpr uint32_t step_size     = 8;

    dim3 blk(block_size_xy * block_size_xy);
    dim3 grd((N + tile_size - 1) / tile_size,
             (M + tile_size - 1) / tile_size);
    coarse_double_buffer_tiled_gemm_kernel<tile_size, block_size_xy, step_size><<<grd, blk>>>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <uint32_t tile_size = 128, uint32_t block_size_xy = 32, uint32_t step_size = 8>
__global__ void final_tiled_gemm_kernel(
    int M, int N, int K,
    float alpha, const float * __restrict__ A, int lda,
    const float * __restrict__ B, int ldb,
    float beta, float * __restrict__ C, int ldc) {
    constexpr uint32_t coarsing_factor = tile_size / block_size_xy;
    constexpr uint32_t block_size = block_size_xy * block_size_xy;
    float C_temp[coarsing_factor][coarsing_factor] = {0};
    float tempA[2][coarsing_factor];
    float tempB[2][coarsing_factor];

    const uint32_t steps = (K + step_size - 1) / step_size;

    __shared__ float Ads[2][step_size][tile_size];
    __shared__ float Bds[2][step_size][tile_size];

    uint32_t warp_id = threadIdx.x / warpSize;
    constexpr uint32_t warp_size_x = block_size_xy / 8;
    uint32_t warp_id_y_c = warp_id / warp_size_x;
    uint32_t warp_id_x_c = warp_id % warp_size_x;
    uint32_t laneId = threadIdx.x % warpSize;

    // 这个地方需要改变一下映射到 block中的相对索引
    // laneid / 16 获取的室在上半个half warp 还是下半个 half warp，*2表示高度为2
    uint32_t thread_idx_y_c = warp_id_y_c * 4 + (laneId / 16) * 2 + (laneId % 2);
    // 保证了两个相邻的线程加载的室同一个数据，可以广播
    // laneId%16表示在half warp的索引，然后一列两个，/2就是在x方向的索引
    uint32_t thread_idx_x_c = warp_id_x_c * 8 + (laneId % 16) / 2;

    for (uint32_t step = 0; step < steps + 1; ++step) {
        uint32_t A_start_y = blockIdx.y * tile_size;
   
        if (step < steps) {
            constexpr uint32_t tile_total_size = tile_size * step_size;
            constexpr uint32_t block_size_y_tile_a = block_size / step_size;
#pragma unroll
            for (uint32_t b = 0; b < (tile_total_size + block_size - 1) / block_size; ++b) {
                uint32_t row = block_size_y_tile_a * b + threadIdx.x % block_size_y_tile_a;
                uint32_t col = threadIdx.x / block_size_y_tile_a;

                if (row < tile_size && col < step_size) {
                    if (A_start_y + row < M && col + step * step_size < K) {
                        Ads[step & 1][col][row] = A[(A_start_y + row) + (col + step * step_size) * lda];
                    } else {
                        Ads[step & 1][col][row] = 0;
                    }
                }
            }

            uint32_t B_start_x = blockIdx.x * tile_size;
            constexpr uint32_t block_size_y_tile_b = step_size;
#pragma unroll
            for (uint32_t b = 0; b < (tile_total_size + block_size - 1) / block_size; ++b) {
                uint32_t row = threadIdx.x % block_size_y_tile_b;
                uint32_t col = threadIdx.x / block_size_y_tile_b + b * (block_size / block_size_y_tile_b);

                if (row < step_size && col < tile_size) {
                    if (row + step * step_size < K && col + B_start_x < N) {
                        Bds[step & 1][row][col ^ (row * 8)] = B[(row + step * step_size) + (col + B_start_x) * ldb];
                    } else {
                        Bds[step & 1][row][col ^ (row * 8)] = 0;
                    }
                }
            }
        }

        if (step > 0) {
#pragma unroll
            for (uint32_t p = 0; p < step_size + 1; ++p) {
                if (p < step_size) {
#pragma unroll
                    for (uint32_t i = 0; i < coarsing_factor / 4; ++i) {
                        // 这里考虑的是相邻thread
                        // 在加载A的时候，基于以上我们的分析，使用的warpsize是4*8，正好每一行代表的是一个quater warp
                        // 对于每个quater warp，加载的都是同一个float4，触发了广播
                        // 因此这里需要两个 transaction
                        uint32_t c_row = (i * block_size_xy + thread_idx_y_c) * 4;
                        FLOAT4(tempA[p & 1][i * 4]) = FLOAT4(Ads[(step + 1) & 1][p][c_row]);
                    }

#pragma unroll
                    for (uint32_t i = 0; i < coarsing_factor / 4; ++i) {
                        // 在改动前，每个相邻的thread访问的向量是并列的
                        // 导致需要4个transaction
                        uint32_t c_col = (i * block_size_xy + thread_idx_x_c) * 4;
                        FLOAT4(tempB[p & 1][i * 4]) = FLOAT4(Bds[(step + 1) & 1][p][c_col ^ (p * 8)]);
                    }
                }

                if (p > 0) {
#pragma unroll
                    for (uint32_t i = 0; i < coarsing_factor; ++i) {
#pragma unroll
                        for (uint32_t j = 0; j < coarsing_factor; ++j) {
                            C_temp[i][j] += tempA[(p + 1) & 1][i] * tempB[(p + 1) & 1][j];
                        }
                    }
                }
            }

        }

        __syncthreads();
    }

    // 写回 C：列主序下同一列的相邻行地址连续，C_temp 的行索引每 4 个
    // (i%4 = 0,1,2,3) 恰好映射到内存里连续的 4 行，可打包成 float4 读改写。
    // 列方向 (c_j) 不连续，逐列处理。
    // 前提 ldc % 4 == 0，保证 float4 的 16B 对齐（与加载侧一致）。
#pragma unroll
    for (uint32_t c_j = 0; c_j < coarsing_factor; ++c_j) {
        uint32_t c_col = (c_j / 4) * (4 * block_size_xy) + thread_idx_x_c * 4 + c_j % 4;
        uint32_t c_global_col = blockIdx.x * tile_size + c_col;
        if (c_global_col >= N) continue;

#pragma unroll
        for (uint32_t ib = 0; ib < coarsing_factor / 4; ++ib) {
            // ib 对应内存中连续的 4 行 (c_i = ib*4 + 0..3)
            uint32_t c_row = ib * (4 * block_size_xy) + thread_idx_y_c * 4;
            uint32_t c_global_row = blockIdx.y * tile_size + c_row;
            uint32_t idx = c_global_row + c_global_col * ldc;

            if (c_global_row + 3 < M) {
                float4 c_old = FLOAT4(C[idx]);
                float4 c_new;
                c_new.x = alpha * C_temp[ib * 4 + 0][c_j] + beta * c_old.x;
                c_new.y = alpha * C_temp[ib * 4 + 1][c_j] + beta * c_old.y;
                c_new.z = alpha * C_temp[ib * 4 + 2][c_j] + beta * c_old.z;
                c_new.w = alpha * C_temp[ib * 4 + 3][c_j] + beta * c_old.w;
                FLOAT4(C[idx]) = c_new;
            } else {
                // 尾部不足 4 行，标量回退避免越界
#pragma unroll
                for (uint32_t r = 0; r < 4; ++r) {
                    if (c_global_row + r < M) {
                        C[idx + r] = alpha * C_temp[ib * 4 + r][c_j] + beta * C[idx + r];
                    }
                }
            }
        }
    }
}

inline void launch_final_tiled_gemm(int M, int N, int K,
                       float alpha,
                       const float *A, int lda,
                       const float *B, int ldb,
                       float beta,
                       float *C, int ldc)
{
    constexpr uint32_t tile_size     = 128;
    constexpr uint32_t block_size_xy = 16;
    constexpr uint32_t step_size     = 8;

    dim3 blk(block_size_xy * block_size_xy);
    dim3 grd((N + tile_size - 1) / tile_size,
             (M + tile_size - 1) / tile_size);
    final_tiled_gemm_kernel<tile_size, block_size_xy, step_size><<<grd, blk>>>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

// ============================================================================
// final_async: cp.async pipeline (sm_80+).
// global→shared 拷贝走异步复制引擎，计算与下一 tile 加载可硬件重叠。
// 需要 M % tile_size == 0, K % step_size == 0, N % tile_size == 0，
// launch 函数内对不满足的尺寸回退到 final_tiled_gemm_kernel。
// ============================================================================
template <uint32_t tile_size = 128, uint32_t block_size_xy = 32, uint32_t step_size = 8>
__global__ void final_async_tiled_gemm_kernel(
    int M, int N, int K,
    float alpha, const float * __restrict__ A, int lda,
    const float * __restrict__ B, int ldb,
    float beta, float * __restrict__ C, int ldc) {
    constexpr uint32_t coarsing_factor = tile_size / block_size_xy;
    constexpr uint32_t block_size = block_size_xy * block_size_xy;
    float C_temp[coarsing_factor][coarsing_factor] = {0};
    float tempA[2][coarsing_factor];
    float tempB[2][coarsing_factor];

    const uint32_t steps = (K + step_size - 1) / step_size;

    __shared__ float Ads[2][step_size][tile_size];
    __shared__ float Bds[2][step_size][tile_size];

    uint32_t warp_id = threadIdx.x / warpSize;
    constexpr uint32_t warp_size_x = block_size_xy / 8;
    uint32_t warp_id_y_c = warp_id / warp_size_x;
    uint32_t warp_id_x_c = warp_id % warp_size_x;
    uint32_t laneId = threadIdx.x % warpSize;

    uint32_t thread_idx_y_c = warp_id_y_c * 4 + (laneId / 16) * 2 + (laneId % 2);
    uint32_t thread_idx_x_c = warp_id_x_c * 8 + (laneId % 16) / 2;

    constexpr uint32_t tile_total_size = tile_size * step_size;
    constexpr uint32_t block_size_y_tile_a = block_size / step_size;
    constexpr uint32_t block_size_y_tile_b = step_size;
    constexpr uint32_t num_load_iters = (tile_total_size + block_size - 1) / block_size;

    // Prefetch tile 0 → buffer 0, block 级同步确保所有 warp 的拷贝完成
    {
        // A float4 load: 16 threads/col, 2×float4 covers 8 consecutive rows
        uint32_t k0 = 0;
        {
            uint32_t A_start_y = blockIdx.y * tile_size;
            uint32_t col_a = threadIdx.x / 16;
            uint32_t row_base = (threadIdx.x % 16) * 8;
            uint32_t g_row = A_start_y + row_base;
            if (g_row + 7 < M && col_a + k0 < K) {
                __pipeline_memcpy_async(&Ads[0][col_a][row_base],
                    &A[g_row + (col_a + k0) * lda], 16);
                __pipeline_memcpy_async(&Ads[0][col_a][row_base + 4],
                    &A[g_row + 4 + (col_a + k0) * lda], 16);
            } else {
#pragma unroll
                for (uint32_t r = 0; r < 8; ++r)
                    if (g_row + r < M && col_a + k0 < K)
                        __pipeline_memcpy_async(&Ads[0][col_a][row_base + r],
                            &A[g_row + r + (col_a + k0) * lda], 4);
                    else
                        __pipeline_memcpy_async(&Ads[0][col_a][row_base + r], nullptr, 4, 4);
            }
        }
        uint32_t B_start_x = blockIdx.x * tile_size;
#pragma unroll
        for (uint32_t b = 0; b < num_load_iters; ++b) {
            uint32_t row = threadIdx.x % block_size_y_tile_b;
            uint32_t col = threadIdx.x / block_size_y_tile_b + b * (block_size / block_size_y_tile_b);
            if (row < step_size && col < tile_size) {
                if (row + k0 < K && col + B_start_x < N)
                    __pipeline_memcpy_async(&Bds[0][row][col ^ (row * 8)], &B[(row + k0) + (col + B_start_x) * ldb], 4);
                else
                    __pipeline_memcpy_async(&Bds[0][row][col ^ (row * 8)], nullptr, 4, 4);
            }
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();  // block 级：确保所有 warp 对 buffer 0 的写入已完成
    }

    for (uint32_t step = 0; step < steps; ++step) {
        // 提前发出下一 tile 的异步拷贝 → 在 compute 期间后台传输
        if (step + 1 < steps) {
            uint32_t k_next = (step + 1) * step_size;
            uint32_t nb = (step + 1) & 1;

            // A float4 load (same mapping as prefetch)
            uint32_t A_start_y = blockIdx.y * tile_size;
            {
                uint32_t col_a = threadIdx.x / 16;
                uint32_t row_base = (threadIdx.x % 16) * 8;
                uint32_t g_row = A_start_y + row_base;
                if (g_row + 7 < M && col_a + k_next < K) {
                    __pipeline_memcpy_async(&Ads[nb][col_a][row_base],
                        &A[g_row + (col_a + k_next) * lda], 16);
                    __pipeline_memcpy_async(&Ads[nb][col_a][row_base + 4],
                        &A[g_row + 4 + (col_a + k_next) * lda], 16);
                } else {
#pragma unroll
                    for (uint32_t r = 0; r < 8; ++r)
                        if (g_row + r < M && col_a + k_next < K)
                            __pipeline_memcpy_async(&Ads[nb][col_a][row_base + r],
                                &A[g_row + r + (col_a + k_next) * lda], 4);
                        else
                            __pipeline_memcpy_async(&Ads[nb][col_a][row_base + r], nullptr, 4, 4);
                }
            }
            uint32_t B_start_x = blockIdx.x * tile_size;
#pragma unroll
            for (uint32_t b = 0; b < num_load_iters; ++b) {
                uint32_t row = threadIdx.x % block_size_y_tile_b;
                uint32_t col = threadIdx.x / block_size_y_tile_b + b * (block_size / block_size_y_tile_b);
                if (row < step_size && col < tile_size) {
                    if (row + k_next < K && col + B_start_x < N)
                        __pipeline_memcpy_async(&Bds[nb][row][col ^ (row * 8)], &B[(row + k_next) + (col + B_start_x) * ldb], 4);
                    else
                        __pipeline_memcpy_async(&Bds[nb][row][col ^ (row * 8)], nullptr, 4, 4);
                }
            }
            __pipeline_commit();
        }

        // Compute on buffer step & 1（其数据已在上次迭代结束时同步完毕）
#pragma unroll
        for (uint32_t p = 0; p < step_size + 1; ++p) {
            if (p < step_size) {
#pragma unroll
                for (uint32_t i = 0; i < coarsing_factor / 4; ++i) {
                    uint32_t c_row = (i * block_size_xy + thread_idx_y_c) * 4;
                    FLOAT4(tempA[p & 1][i * 4]) = FLOAT4(Ads[step & 1][p][c_row]);
                }
#pragma unroll
                for (uint32_t i = 0; i < coarsing_factor / 4; ++i) {
                    uint32_t c_col = (i * block_size_xy + thread_idx_x_c) * 4;
                    FLOAT4(tempB[p & 1][i * 4]) = FLOAT4(Bds[step & 1][p][c_col ^ (p * 8)]);
                }
            }
            if (p > 0) {
#pragma unroll
                for (uint32_t i = 0; i < coarsing_factor; ++i) {
#pragma unroll
                    for (uint32_t j = 0; j < coarsing_factor; ++j) {
                        C_temp[i][j] += tempA[(p + 1) & 1][i] * tempB[(p + 1) & 1][j];
                    }
                }
            }
        }

        // 等待所有 warp 的 async copy 完成 + compute 读完成，保证下一
        // 迭代开始时 buffer[(step+1)&1] 的数据对所有线程可见
        if (step + 1 < steps) {
            __pipeline_wait_prior(0);
        }
        __syncthreads();
    }

    // Write-back C (同 final)
#pragma unroll
    for (uint32_t c_j = 0; c_j < coarsing_factor; ++c_j) {
        uint32_t c_col = (c_j / 4) * (4 * block_size_xy) + thread_idx_x_c * 4 + c_j % 4;
        uint32_t c_global_col = blockIdx.x * tile_size + c_col;
        if (c_global_col >= N) continue;

#pragma unroll
        for (uint32_t ib = 0; ib < coarsing_factor / 4; ++ib) {
            uint32_t c_row = ib * (4 * block_size_xy) + thread_idx_y_c * 4;
            uint32_t c_global_row = blockIdx.y * tile_size + c_row;
            uint32_t idx = c_global_row + c_global_col * ldc;

            if (c_global_row + 3 < M) {
                float4 c_old = FLOAT4(C[idx]);
                float4 c_new;
                c_new.x = alpha * C_temp[ib * 4 + 0][c_j] + beta * c_old.x;
                c_new.y = alpha * C_temp[ib * 4 + 1][c_j] + beta * c_old.y;
                c_new.z = alpha * C_temp[ib * 4 + 2][c_j] + beta * c_old.z;
                c_new.w = alpha * C_temp[ib * 4 + 3][c_j] + beta * c_old.w;
                FLOAT4(C[idx]) = c_new;
            } else {
#pragma unroll
                for (uint32_t r = 0; r < 4; ++r) {
                    if (c_global_row + r < M)
                        C[idx + r] = alpha * C_temp[ib * 4 + r][c_j] + beta * C[idx + r];
                }
            }
        }
    }
}

inline void launch_final_async_tiled_gemm(int M, int N, int K,
                       float alpha,
                       const float *A, int lda,
                       const float *B, int ldb,
                       float beta,
                       float *C, int ldc)
{
    constexpr uint32_t tile_size     = 128;
    constexpr uint32_t block_size_xy = 16;
    constexpr uint32_t step_size     = 16;

    // cp.async 流水线假设所有维度都被 tile/step 整除，
    // 边界不完全 tile 回退到同步 final kernel。
    if (M % tile_size != 0 || N % tile_size != 0 || K % step_size != 0) {
        launch_final_tiled_gemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    dim3 blk(block_size_xy * block_size_xy);
    dim3 grd((N + tile_size - 1) / tile_size,
             (M + tile_size - 1) / tile_size);
    final_async_tiled_gemm_kernel<tile_size, block_size_xy, step_size><<<grd, blk>>>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}



