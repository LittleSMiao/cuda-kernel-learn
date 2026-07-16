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
