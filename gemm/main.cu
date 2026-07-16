// ============================================================================
// GEMM 基准测试 —— 入口
// ============================================================================
// 编译（以 SM 8.0 / A100 为例，根据你的 GPU 调整 arch）：
//   nvcc -arch=sm_80 -O3 -lcublas -o gemm_bench main.cu gemm_test.cu
//
// 或使用附带的 Makefile：
//   make
//   make run
// ============================================================================

#include "gemm_test.h"
#include "custom_gemm_example.cuh"

#include <cstdio>
#include <cstdlib>
#include <ctime>

int main() {
    srand(42);

    // ========================================================================
    // 1. 注册所有 GEMM 实现
    // ========================================================================
    // 把你自己写的 gemm 按 { "名字", 函数指针 } 的格式加入这个数组即可。
    // 框架会自动完成计时、正确性校验、评分。
    // ========================================================================
    std::vector<GemmDescriptor> impls = {
        // ---------- baseline（不要删除）---------- //
        { "cuBLAS", cublas_gemm },

        // ---------- 自定义实现 ---------- //
        { "NaiveGEMM",  naive_gemm },
        { "TiledGEMM",  tiled_gemm },

        // 在这里添加你自己的 GEMM，例如：
        // { "MySuperGEMM", my_super_gemm },
    };

    // ========================================================================
    // 2. 配置测试参数
    // ========================================================================
    TestConfig cfg;

    // 测试的矩阵尺寸 —— 可以根据需要自由增删
    cfg.sizes = {
        {  128,  128,  128 },
        {  256,  256,  256 },
        {  512,  512,  512 },
        { 1024, 1024, 1024 },
        { 2048, 2048, 2048 },
    };

    cfg.warmup_iters     = 5;
    cfg.bench_iters      = 20;
    cfg.error_tol        = 1e-3f;
    cfg.skip_correctness = false;
    cfg.cuda_device      = 0;

    // ========================================================================
    // 3. 检查 GPU
    // ========================================================================
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) {
        fprintf(stderr, "No CUDA device found!\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cfg.cuda_device);
    printf("\nGPU: %s (SM %d.%d, %.1f GB)\n\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    cudaSetDevice(cfg.cuda_device);

    // ========================================================================
    // 4. 跑 benchmark
    // ========================================================================
    printf("Running GEMM benchmark (%zu implementations, %zu sizes)...\n\n",
           impls.size(), cfg.sizes.size());

    auto report = run_benchmark(impls, cfg);

    // ========================================================================
    // 5. 输出报告
    // ========================================================================
    print_report(report);
    save_csv(report, "gemm_benchmark.csv");

    return 0;
}
