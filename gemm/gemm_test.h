#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <vector>
#include <functional>

// ---------------------------------------------------------------------------
// GEMM 算子签名：C = α·A·B + β·C
// ---------------------------------------------------------------------------
// M, N, K    — 矩阵维度：A[M×K], B[K×N], C[M×N]
// alpha, beta — 标量系数
// A, B, C    — 设备端指针（列主序 / column-major，与 cuBLAS 保持一致）
//
// 所有自定义 GEMM 实现都必须遵循这个签名。
// ---------------------------------------------------------------------------
using GemmFunc = std::function<void(
    int M, int N, int K,
    float alpha,
    const float *A, int lda,
    const float *B, int ldb,
    float beta,
    float *C, int ldc)>;

// ---------------------------------------------------------------------------
// 单个 GEMM 实现的描述信息
// ---------------------------------------------------------------------------
struct GemmDescriptor {
    std::string name;      // 展示名称，如 "cuBLAS", "MyTileGemm"
    GemmFunc    func;      // 可调用的 GEMM 函数
};

// ---------------------------------------------------------------------------
// 单次测试结果
// ---------------------------------------------------------------------------
struct GemmTestResult {
    std::string  impl_name;       // 算子名称
    int          M, N, K;         // 矩阵维度
    double       time_ms;          // 耗时（毫秒）
    double       gflops;           // 算力 (GFLOPS)
    double       gflops_effective; // 有效算力 = 2*M*N*K / time（通常二者接近）
    double       max_error;        // 与 cuBLAS 结果的最大绝对误差
    bool         correct;          // 是否通过正确性检查（max_error < tol）
};

// ---------------------------------------------------------------------------
// 综合评测结果
// ---------------------------------------------------------------------------
struct BenchmarkReport {
    struct Entry {
        std::string name;
        double      score;      // 相对 cuBLAS 的百分比得分（以 GFLOPS 为基准）
        double      avg_time_ms;
        double      avg_gflops;
        bool        all_correct;
    };

    std::vector<Entry>                 entries;
    std::vector<std::vector<GemmTestResult>> per_size_results; // [size_idx][impl_idx]
};

// ---------------------------------------------------------------------------
// 测试配置
// ---------------------------------------------------------------------------
struct TestConfig {
    // 测试的矩阵尺寸列表，每个元素为 {M, N, K}
    std::vector<std::tuple<int,int,int>> sizes;

    int    warmup_iters   = 5;     // 预热迭代次数
    int    bench_iters    = 20;    // 计时迭代次数
    float  error_tol      = 1e-3f; // 正确性容差（相对误差）
    bool   skip_correctness = false; // 跳过正确性检查（仅测性能）
    int    cuda_device    = 0;     // 使用的 GPU 编号
};

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

// cuBLAS baseline 实现（定义在 gemm_test.cu），可直接注册到 impls 数组。
void cublas_gemm(
    int M, int N, int K,
    float alpha,
    const float *A, int lda,
    const float *B, int ldb,
    float beta,
    float *C, int ldc);

// 运行完整的基准测试：以 cuBLAS 为 baseline，对所有注册的 GEMM 实现进行
// 性能和正确性评测，返回综合报告。
BenchmarkReport run_benchmark(
    const std::vector<GemmDescriptor> &impls,
    const TestConfig                  &cfg);

// 打印报告到 stdout
void print_report(const BenchmarkReport &report);

// 将报告保存为 CSV 文件
void save_csv(const BenchmarkReport &report, const std::string &path);

// 工具函数：初始化一个列主序矩阵，host 端
void init_matrix_col_major(float *h, int rows, int cols, int ld);
