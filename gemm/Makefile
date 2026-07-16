# ============================================================================
# GEMM Benchmark Makefile
# ============================================================================
# 根据你的 GPU 修改 ARCH 值：
#   H100/RTX 4090  → sm_90
#   A100/RTX 3090  → sm_80
#   V100/T4        → sm_70
#   RTX 2080 Ti    → sm_75
#   GTX 1080 Ti    → sm_61
# ============================================================================

ARCH ?= sm_80

NVCC       = nvcc
NVCC_FLAGS = -arch=$(ARCH) -O3 -std=c++17
LDFLAGS    = -lcublas

TARGET = gemm_bench
SRCS   = main.cu gemm_test.cu

.PHONY: all run clean profile help

all: $(TARGET)

$(TARGET): $(SRCS) gemm_test.h custom_gemm_example.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ $(SRCS) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

# 使用 nvprof / nsys 进行 profiling
profile: $(TARGET)
	nsys profile --stats=true ./$(TARGET)

clean:
	rm -f $(TARGET) gemm_benchmark.csv *.qdrep *.sqlite

help:
	@echo "Usage:"
	@echo "  make          - 编译"
	@echo "  make run      - 编译并运行"
	@echo "  make profile  - nsys profiling"
	@echo "  make clean    - 清理"
	@echo ""
	@echo "  可通过 ARCH 变量指定 GPU 架构，例如:"
	@echo "    make ARCH=sm_90    (H100 / RTX 4090)"
	@echo "    make ARCH=sm_80    (A100 / RTX 3090)"
	@echo "    make ARCH=sm_75    (RTX 2080 Ti)"
	@echo "    make ARCH=sm_70    (V100 / T4)"
