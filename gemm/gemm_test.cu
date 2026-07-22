#include "gemm_test.h"

#include <algorithm>
#include <chrono>
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
    if (time_ms <= 0.0) return 0.0;
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
    // 清除之前可能残留的 sticky error
    cudaGetLastError();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float alpha = 1.0f, beta = 0.0f;
    bool kernel_ok = true;

    // --- 预热 ---
    for (int i = 0; i < cfg.warmup_iters && kernel_ok; ++i) {
        if (cudaMemset(d_C, 0, ldc * N * sizeof(float)) != cudaSuccess)
            kernel_ok = false;
        desc.func(M, N, K, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) kernel_ok = false;
    if (cudaGetLastError() != cudaSuccess) kernel_ok = false;

    // --- 正式计时 ---
    // 一次 memset + sync，然后多次 kernel launch 放在一对 event 之间，
    // 累积足够的 GPU 时间以突破 timer 精度限制
    using Clock = std::chrono::high_resolution_clock;
    double avg_ms = 0.0;

    if (kernel_ok) {
        if (cudaMemset(d_C, 0, ldc * N * sizeof(float)) != cudaSuccess)
            kernel_ok = false;
        if (cudaDeviceSynchronize() != cudaSuccess) kernel_ok = false;

        if (kernel_ok) {
            auto t0 = Clock::now();

            cudaEventRecord(start);
            for (int i = 0; i < cfg.bench_iters; ++i) {
                desc.func(M, N, K, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            auto t1 = Clock::now();
            double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            float gpu_ms = 0.0f;
            cudaEventElapsedTime(&gpu_ms, start, stop);

            if (cudaGetLastError() != cudaSuccess) kernel_ok = false;

            // 优先使用 GPU event 计时，若为 0 则回退到 CPU 计时
            avg_ms = (gpu_ms > 0.0f) ? (gpu_ms / cfg.bench_iters)
                                     : (cpu_ms / cfg.bench_iters);
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // --- 正确性 ---
    double max_err = 0.0;
    bool   correct = true;

    if (!cfg.skip_correctness && kernel_ok) {
        std::vector<float> h_C(ldc * N);
        std::vector<float> h_ref(ldc * N);
        if (cudaMemcpy(h_C.data(),   d_C,     ldc * N * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(h_ref.data(), d_C_ref, ldc * N * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            correct = false;
        } else {
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
    } else if (!kernel_ok) {
        correct = false;
    }

    return {
        desc.name, M, N, K,
        avg_ms,
        get_gflops(M, N, K, avg_ms),
        get_gflops(M, N, K, avg_ms),
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
        int n_sizes = (int)report.per_size_results.size();

        constexpr int CELL_W = 16;
        char cell[64];

        // 找到 cuBLAS 索引
        int blas_idx = -1;
        for (int ii = 0; ii < n_impls; ++ii) {
            if (report.entries[ii].name == "cuBLAS") { blas_idx = ii; break; }
        }

        // 表头
        printf("%-20s", "Impl \\ Size");
        for (int si = 0; si < n_sizes; ++si) {
            auto &res = report.per_size_results[si][0];
            snprintf(cell, sizeof(cell), "%dx%dx%d", res.M, res.N, res.K);
            printf(" | %-*s", CELL_W, cell);
        }
        printf("\n");
        int total_w = 20 + (3 + CELL_W) * n_sizes;
        printf("%s\n", std::string(total_w, '-').c_str());

        for (int ii = 0; ii < n_impls; ++ii) {
            // 第一行：GFLOPS + 正确性
            printf("%-20s", report.entries[ii].name.c_str());
            for (int si = 0; si < n_sizes; ++si) {
                auto &res = report.per_size_results[si][ii];
                snprintf(cell, sizeof(cell), "%.1f GF %s",
                         res.gflops, res.correct ? "✓" : "✗");
                printf(" | %-*s", CELL_W, cell);
            }
            // 第二行：相对 cuBLAS 百分比
            printf("\n%-20s", "");
            for (int si = 0; si < n_sizes; ++si) {
                auto &res = report.per_size_results[si][ii];
                double pct = 100.0;
                if (blas_idx >= 0 && ii != blas_idx) {
                    auto &ref = report.per_size_results[si][blas_idx];
                    pct = (ref.gflops > 0) ? (res.gflops / ref.gflops * 100.0) : 0.0;
                }
                snprintf(cell, sizeof(cell), "%.1f%%", pct);
                printf(" | %-*s", CELL_W, cell);
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
