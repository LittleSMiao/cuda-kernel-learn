# ============================================================================
# GEMM Benchmark —— CMake 构建
# ============================================================================
# 用法：
#   cmake -B build                       # 配置（默认 Release + sm_120）
#   cmake --build build -j               # 编译
#   ./build/gemm_bench                   # 运行
#   cmake --build build --target run     # 编译并运行
#
# 指定 GPU 架构（默认 120，即 RTX 5090 / Blackwell）：
#   H100/RTX 4090  → 90
#   A100/RTX 3090  → 80
#   V100/T4        → 70
#   RTX 2080 Ti    → 75
# 例如：cmake -B build -DCMAKE_CUDA_ARCHITECTURES=80
# ============================================================================

cmake_minimum_required(VERSION 3.18)

# 默认 GPU 架构：RTX 5090 (Blackwell)。可用 -DCMAKE_CUDA_ARCHITECTURES=XX 覆盖。
# 注意：必须在 project() 之前设置——project(... CUDA) 会在初始化 CUDA 语言时
# 把未设置的 CMAKE_CUDA_ARCHITECTURES 填成一个低版本默认值（如 52），
# 之后再设置就晚了。命令行 -D 传入时会作为 cache 变量先于此处定义，故不会被覆盖。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

project(gemm_bench LANGUAGES CXX CUDA)

# 默认 Release（-O3）。可用 -DCMAKE_BUILD_TYPE=Debug 覆盖。
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 提供 CUDA::cublas 等 imported target（CMake 3.17+）
find_package(CUDAToolkit REQUIRED)

add_executable(gemm_bench
    main.cu
    gemm_test.cu)

target_link_libraries(gemm_bench PRIVATE CUDA::cublas)

# 便捷目标：编译后直接运行
add_custom_target(run
    COMMAND gemm_bench
    DEPENDS gemm_bench
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running gemm_bench")
