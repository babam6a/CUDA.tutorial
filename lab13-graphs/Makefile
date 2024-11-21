# Makefile

# Default target to build both axpy_stream_capture and axpy_cublas
.PHONY: all
all: axpy_stream_capture axpy_cublas

# Rule for axpy_stream_capture
axpy_stream_capture: axpy_stream_capture.cu
	nvcc -arch=sm_70 axpy_stream_capture.cu -o axpy_stream_capture

# Rule for axpy_cublas
axpy_cublas: axpy_cublas.cu
	nvcc -arch=sm_70 -lcublas axpy_cublas.cu -o axpy_cublas

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make clean              - Clean up generated binaries"
	@echo "  make axpy_stream_capture - Compile axpy_stream_capture.cu"
	@echo "  make axpy_cublas        - Compile axpy_cublas.cu"
	@echo "  make help               - Print this help"

# Clean target
.PHONY: clean
clean:
	rm -f axpy_stream_capture axpy_cublas
