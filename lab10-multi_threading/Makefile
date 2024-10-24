# Makefile for building CUDA programs with OpenMP support and setting environment variables

# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -Xcompiler -fopenmp

# Define executable names and source files
OPENMP_EXEC = openmp
OPENMP_SRC = openmp.cu

MULTI_GPU_EXEC = multi_gpu
MULTI_GPU_SRC = multi_gpu.cu

# Set environment variable
export OMP_NUM_THREADS = 8

# Default target
all: $(OPENMP_EXEC) $(MULTI_GPU_EXEC)

# Build OpenMP CUDA program
$(OPENMP_EXEC): $(OPENMP_SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(OPENMP_EXEC) $(OPENMP_SRC) -DUSE_STREAMS

# Build Multi-GPU CUDA program
$(MULTI_GPU_EXEC): $(MULTI_GPU_SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(MULTI_GPU_EXEC) $(MULTI_GPU_SRC) -DUSE_STREAMS

# Clean target
clean:
	rm -f $(OPENMP_EXEC) $(MULTI_GPU_EXEC)

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make clean              - Clean up generated binaries"
	@echo "  make openmp             - Compile openmp.cu"
	@echo "  make multi_gpu          - Compile multi_gpu.cu"
	@echo "  make help               - Print this help"

.PHONY: all clean openmp multi_gpu
