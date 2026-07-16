#include "gemm_test.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

// ============================================================================
// cuBLAS baseline — GEMM 封装
// ============================================================================
static cublasHandle_t g_cublas_handle = nullptr;

void cublas_gemm(
    int M, int N, int K,
    float alpha,
    const float *A, int lda,
    const float *B, int ldb,
    float beta,
    float *C, int ldc)
{
    // cuBLAS 使用列主序 (column-major / Fortran order)。
    // 签名: cublasSgemm(handle, transa, transb, m, n, k, α, A, lda, B, ldb, β, C, ldc)
    // 语义: C[m×n] = α · op(A)[m×k] · op(B)[k×n] + β · C[m×n]
    //
    // 我们的数据就是列主序：A 是 M×K, B 是 K×N, C 是 M×N
    // 所以直接设 m=M, n=N, k=K, op=IDENTITY 即可。
    cublasSgemm(g_cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &alpha,
                A, lda,
                B, ldb,
                &beta,
                C, ldc);
}

// ============================================================================
// 工具函数
// ============================================================================
void init_matrix_col_major(float *h, int rows, int cols, int ld) {
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            h[j * ld + i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        }
    }
}

static double get_gflops(int M, int N, int K, double time_ms) {
    double ops = 2.0 * M * N * K;  // 乘加算两次运算
    return ops / (time_ms * 1e6);  // ms → GFLOPS
}

// ============================================================================
// 核心：单次 GEMM 计时 + 正确性
// ============================================================================
static GemmTestResult test_one(
    const GemmDescriptor &desc,
    int M, int N, int K,
    const float *d_A, int lda,
    const float *d_B, int ldb,
    float *d_C, int ldc,
    const float *d_C_ref,          // cuBLAS 参考结果
    const TestConfig &cfg)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float alpha = 1.0f, beta = 0.0f;

    // --- 预热 ---
    for (int i = 0; i < cfg.warmup_iters; ++i) {
        cudaMemset(d_C, 0, ldc * N * sizeof(float));
        desc.func(M, N, K, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
    }
    cudaDeviceSynchronize();

    // --- 正式计时 ---
    float total_ms = 0.0f;
    for (int i = 0; i < cfg.bench_iters; ++i) {
        cudaMemset(d_C, 0, ldc * N * sizeof(float));

        cudaEventRecord(start);
        desc.func(M, N, K, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    double avg_ms = total_ms / cfg.bench_iters;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // --- 正确性 ---
    double max_err = 0.0;
    bool   correct = true;

    if (!cfg.skip_correctness) {
        std::vector<float> h_C(ldc * N);
        std::vector<float> h_ref(ldc * N);
        cudaMemcpy(h_C.data(),   d_C,     ldc * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ref.data(), d_C_ref, ldc * N * sizeof(float), cudaMemcpyDeviceToHost);

        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < M; ++i) {
                float ref  = h_ref[j * ldc + i];
                float val  = h_C[j * ldc + i];
                float diff = std::fabs(val - ref);
                float denom = std::max(1.0f, std::fabs(ref));
                if (diff > 1.0f && diff / denom > cfg.error_tol) {
                    correct = false;
                }
                if (diff > max_err) max_err = diff;
            }
        }
    }

    return {
        desc.name, M, N, K,
        avg_ms,
        get_gflops(M, N, K, avg_ms),
        get_gflops(M, N, K, avg_ms),  // same as gflops for now
        max_err,
        correct
    };
}

// ============================================================================
// BenchmarkReport 构建
// ============================================================================
BenchmarkReport run_benchmark(
    const std::vector<GemmDescriptor> &impls,
    const TestConfig                  &cfg)
{
    BenchmarkReport report;
    int n_impls = (int)impls.size();

    // 确保 cuBLAS handle 已创建
    if (!g_cublas_handle) {
        cublasCreate(&g_cublas_handle);
    }

    // 找到 cuBLAS 的索引（一定是 basline）
    int cuBLAS_idx = -1;
    for (int i = 0; i < n_impls; ++i) {
        if (impls[i].name == "cuBLAS") { cuBLAS_idx = i; break; }
    }

    report.per_size_results.resize(cfg.sizes.size());

    for (int si = 0; si < (int)cfg.sizes.size(); ++si) {
        auto [M, N, K] = cfg.sizes[si];

        int lda = M, ldb = K, ldc = M;
        size_t bytes_A = (size_t)lda * K * sizeof(float);
        size_t bytes_B = (size_t)ldb * N * sizeof(float);
        size_t bytes_C = (size_t)ldc * N * sizeof(float);

        // 分配设备内存
        float *d_A, *d_B, *d_C, *d_C_ref;
        cudaMalloc(&d_A, bytes_A);
        cudaMalloc(&d_B, bytes_B);
        cudaMalloc(&d_C, bytes_C);
        cudaMalloc(&d_C_ref, bytes_C);

        // 初始化 host 数据
        std::vector<float> h_A(lda * K), h_B(ldb * N), h_C(ldc * N);
        init_matrix_col_major(h_A.data(), M, K, lda);
        init_matrix_col_major(h_B.data(), K, N, ldb);
        std::memset(h_C.data(), 0, ldc * N * sizeof(float));

        cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice);

        // 先跑 cuBLAS 得到参考结果
        if (cuBLAS_idx >= 0) {
            cudaMemset(d_C, 0, bytes_C);
            float alpha = 1.0f, beta = 0.0f;
            cublasSgemm(g_cublas_handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        M, N, K,
                        &alpha, d_A, lda, d_B, ldb,
                        &beta, d_C, ldc);
            cudaMemcpy(d_C_ref, d_C, bytes_C, cudaMemcpyDeviceToDevice);
        }

        // 逐个测试每个实现
        report.per_size_results[si].resize(n_impls);
        for (int ii = 0; ii < n_impls; ++ii) {
            printf("  [%d/%d] %-20s  M=%d N=%d K=%d ... ",
                   ii + 1, n_impls, impls[ii].name.c_str(), M, N, K);
            fflush(stdout);

            auto res = test_one(impls[ii], M, N, K,
                                d_A, lda, d_B, ldb, d_C, ldc,
                                d_C_ref, cfg);

            report.per_size_results[si][ii] = res;
            printf("%8.3f ms  %7.1f GFLOPS  %s\n",
                   res.time_ms, res.gflops,
                   res.correct ? "✓" : "✗ ERROR");
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
    }

    // 汇总得分
    report.entries.resize(n_impls);
    for (int ii = 0; ii < n_impls; ++ii) {
        double sum_score    = 0.0;
        double sum_time     = 0.0;
        double sum_gflops   = 0.0;
        bool   all_ok       = true;
        int    n_sizes      = (int)cfg.sizes.size();

        for (int si = 0; si < n_sizes; ++si) {
            auto &res = report.per_size_results[si][ii];
            sum_time   += res.time_ms;
            sum_gflops += res.gflops;
            if (!res.correct) all_ok = false;

            // 相对 cuBLAS 的 GFLOPS 得分
            if (cuBLAS_idx >= 0 && ii != cuBLAS_idx) {
                auto &ref = report.per_size_results[si][cuBLAS_idx];
                sum_score += (ref.gflops > 0) ? (res.gflops / ref.gflops * 100.0) : 0.0;
            }
        }

        // cuBLAS 自身得分 100%
        if (ii == cuBLAS_idx) {
            sum_score = 100.0 * n_sizes;
        }

        report.entries[ii] = {
            impls[ii].name,
            sum_score / n_sizes,           // 平均相对得分
            sum_time / n_sizes,
            sum_gflops / n_sizes,
            all_ok
        };
    }

    return report;
}

// ============================================================================
// 打印报告
// ============================================================================
void print_report(const BenchmarkReport &report) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                     GEMM BENCHMARK REPORT                            ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Score = (impl_gflops / cuBLAS_gflops) × 100                         ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n\n";

    // 表头
    printf("%-24s %8s %12s %12s %8s\n",
           "Implementation", "Score%", "Avg Time(ms)", "Avg GFLOPS", "Correct");
    printf("%s\n", std::string(72, '-').c_str());

    for (auto &e : report.entries) {
        printf("%-24s %7.1f%% %11.3f %11.1f %8s\n",
               e.name.c_str(),
               e.score,
               e.avg_time_ms,
               e.avg_gflops,
               e.all_correct ? "✓" : "✗");
    }

    // 按尺寸列详表
    if (!report.per_size_results.empty() && !report.per_size_results[0].empty()) {
        std::cout << "\n--- Per-Size Breakdown ---\n\n";
        int n_impls = (int)report.per_size_results[0].size();

        // 表头
        printf("%-20s", "Impl \\ Size");
        for (int si = 0; si < (int)report.per_size_results.size(); ++si) {
            auto &res = report.per_size_results[si][0];
            printf(" | %4dx%4dx%4d", res.M, res.N, res.K);
        }
        printf("\n%s\n", std::string(20 + 22 * report.per_size_results.size(), '-').c_str());

        for (int ii = 0; ii < n_impls; ++ii) {
            printf("%-20s", report.entries[ii].name.c_str());
            for (int si = 0; si < (int)report.per_size_results.size(); ++si) {
                auto &res = report.per_size_results[si][ii];
                printf(" | %6.1f GF %s",
                       res.gflops,
                       res.correct ? "✓" : "✗");
            }
            printf("\n");
        }
    }

    std::cout << std::endl;
}

// ============================================================================
// CSV 导出
// ============================================================================
void save_csv(const BenchmarkReport &report, const std::string &path) {
    std::ofstream f(path);
    if (!f) { std::cerr << "Failed to open " << path << "\n"; return; }

    f << "implementation,score_pct,avg_time_ms,avg_gflops,all_correct\n";
    for (auto &e : report.entries) {
        f << e.name << ","
          << e.score << ","
          << e.avg_time_ms << ","
          << e.avg_gflops << ","
          << (e.all_correct ? "true" : "false") << "\n";
    }

    // Per-size detail
    f << "\nsize,implementation,time_ms,gflops,max_error,correct\n";
    for (auto &row : report.per_size_results) {
        for (auto &r : row) {
            f << r.M << "x" << r.N << "x" << r.K << ","
              << r.impl_name << ","
              << r.time_ms << ","
              << r.gflops << ","
              << r.max_error << ","
              << (r.correct ? "true" : "false") << "\n";
        }
    }
    std::cout << "Report saved to " << path << "\n";
}
