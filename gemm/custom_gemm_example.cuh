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
    constexpr uint32_t coarsing_fator = tile_size / block_size_xy;
    constexpr uint32_t block_size = block_size_xy * block_size_xy;
    float C_temp[coarsing_fator][coarsing_fator] = {0};

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
            for (uint32_t c_i = 0; c_i < coarsing_fator; ++c_i) {
                uint32_t c_row = c_i * block_size_xy + (threadIdx.x / block_size_xy);
#pragma unroll
                for (uint32_t c_j = 0; c_j < coarsing_fator; ++c_j) {
                    uint32_t c_col = c_j * block_size_xy + (threadIdx.x & (block_size_xy - 1));

                    C_temp[c_i][c_j] += Ads[c_row][p] * Bds[p][c_col];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (uint32_t c_i = 0; c_i < coarsing_fator; ++c_i) {
        uint32_t c_row = c_i * block_size_xy + (threadIdx.x / block_size_xy);
        uint32_t c_global_row = blockIdx.y * tile_size + c_row;
#pragma unroll
        for (uint32_t c_j = 0; c_j < coarsing_fator; ++c_j) {
            uint32_t c_col = c_j * block_size_xy + (threadIdx.x & (block_size_xy - 1));
            uint32_t c_global_col = blockIdx.x * tile_size + c_col;

            if (c_global_row < M && c_global_col < N) {
                uint32_t idx = c_global_row + c_global_col * ldc;
                C[idx] = alpha * C_temp[c_i][c_j] + beta * C[idx];
            }
        }
    }
}

inline void coarse_tiled_gemm(int M, int N, int K,
                       float alpha,
                       const float *A, int lda,
                       const float *B, int ldb,
                       float beta,
                       float *C, int ldc)
{
    
    const uint32_t max_dim = max(M, N);

    // 按问题规模分派：小尺寸用更小的 tile 产生更多 block，提高 SM 利用率
    if (max_dim <= 256) {
        // 中等尺寸：64×64 tile，1024 thread，coarsing=2（4 regs）
        // 256² → 16 blocks，128² → 4 blocks
        constexpr uint32_t tile_size     = 16;
        constexpr uint32_t block_size_xy = 16;
        constexpr uint32_t step_size     = 16;

        dim3 block(block_size_xy * block_size_xy);
        dim3 grid((N + tile_size - 1) / tile_size,
                  (M + tile_size - 1) / tile_size);
        coarse_tiled_gemm_kernel<tile_size, block_size_xy, step_size><<<grid, block>>>(
            M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else if (max_dim < 1024) {
        // 大尺寸：128×128 tile，1024 thread，coarsing=4（16 regs）
        constexpr uint32_t tile_size     = 64;
        constexpr uint32_t block_size_xy = 32;

        dim3 block(block_size_xy * block_size_xy);
        dim3 grid((N + tile_size - 1) / tile_size,
                  (M + tile_size - 1) / tile_size);
        coarse_tiled_gemm_kernel<tile_size, block_size_xy, 8><<<grid, block>>>(
            M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        // 大尺寸：128×128 tile，1024 thread，coarsing=4（16 regs）
        constexpr uint32_t tile_size     = 128;
        constexpr uint32_t block_size_xy = 32;

        dim3 block(block_size_xy * block_size_xy);
        dim3 grid((N + tile_size - 1) / tile_size,
                  (M + tile_size - 1) / tile_size);
        coarse_tiled_gemm_kernel<tile_size, block_size_xy, 8><<<grid, block>>>(
            M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

template <uint32_t tile_size = 128, uint32_t block_size_xy = 32, uint32_t step_size = 8>
__global__ void coarse_padding_tiled_gemm_kernel(
    int M, int N, int K,
    float alpha, const float *A, int lda,
    const float *B, int ldb,
    float beta, float *C, int ldc) {
    constexpr uint32_t coarsing_fator = tile_size / block_size_xy;
    constexpr uint32_t block_size = block_size_xy * block_size_xy;
    float C_temp[coarsing_fator][coarsing_fator] = {0};

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
            for (uint32_t c_i = 0; c_i < coarsing_fator; ++c_i) {
                uint32_t c_row = c_i * block_size_xy + (threadIdx.x / block_size_xy);
#pragma unroll
                for (uint32_t c_j = 0; c_j < coarsing_fator; ++c_j) {
                    uint32_t c_col = c_j * block_size_xy + (threadIdx.x & (block_size_xy - 1));

                    C_temp[c_i][c_j] += Ads[c_row][p] * Bds[p][c_col];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (uint32_t c_i = 0; c_i < coarsing_fator; ++c_i) {
        uint32_t c_row = c_i * block_size_xy + (threadIdx.x / block_size_xy);
        uint32_t c_global_row = blockIdx.y * tile_size + c_row;
#pragma unroll
        for (uint32_t c_j = 0; c_j < coarsing_fator; ++c_j) {
            uint32_t c_col = c_j * block_size_xy + (threadIdx.x & (block_size_xy - 1));
            uint32_t c_global_col = blockIdx.x * tile_size + c_col;

            if (c_global_row < M && c_global_col < N) {
                uint32_t idx = c_global_row + c_global_col * ldc;
                C[idx] = alpha * C_temp[c_i][c_j] + beta * C[idx];
            }
        }
    }
}

inline void coarse_padding_tiled_gemm(int M, int N, int K,
                       float alpha,
                       const float *A, int lda,
                       const float *B, int ldb,
                       float beta,
                       float *C, int ldc)
{
    
    const uint32_t max_dim = max(M, N);

    // 按问题规模分派：小尺寸用更小的 tile 产生更多 block，提高 SM 利用率
    if (max_dim <= 256) {
        // 中等尺寸：64×64 tile，1024 thread，coarsing=2（4 regs）
        // 256² → 16 blocks，128² → 4 blocks
        constexpr uint32_t tile_size     = 16;
        constexpr uint32_t block_size_xy = 16;
        constexpr uint32_t step_size     = 16;

        dim3 block(block_size_xy * block_size_xy);
        dim3 grid((N + tile_size - 1) / tile_size,
                  (M + tile_size - 1) / tile_size);
        coarse_padding_tiled_gemm_kernel<tile_size, block_size_xy, step_size><<<grid, block>>>(
            M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else if (max_dim < 1024) {
        // 大尺寸：128×128 tile，1024 thread，coarsing=4（16 regs）
        constexpr uint32_t tile_size     = 64;
        constexpr uint32_t block_size_xy = 32;

        dim3 block(block_size_xy * block_size_xy);
        dim3 grid((N + tile_size - 1) / tile_size,
                  (M + tile_size - 1) / tile_size);
        coarse_padding_tiled_gemm_kernel<tile_size, block_size_xy, 8><<<grid, block>>>(
            M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        // 大尺寸：128×128 tile，1024 thread，coarsing=4（16 regs）
        constexpr uint32_t tile_size     = 128;
        constexpr uint32_t block_size_xy = 32;

        dim3 block(block_size_xy * block_size_xy);
        dim3 grid((N + tile_size - 1) / tile_size,
                  (M + tile_size - 1) / tile_size);
        coarse_padding_tiled_gemm_kernel<tile_size, block_size_xy, 8><<<grid, block>>>(
            M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

