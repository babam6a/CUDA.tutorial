# Makefile

# Compiler
CC := nvcc
ARCH := -arch=sm_70

# Default target to build both mmul and ahs executables
.PHONY: all
all: mmul ahs

# Rule for mmul
mmul: mmul.cu
	$(CC) $(ARCH) mmul.cu -o mmul -lineinfo

# Rule for ahs
ahs: ahs.cu
	$(CC) $(ARCH) ahs.cu -o ahs -G -g -std=c++14

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make mmul      - Compile mmul.cu"
	@echo "  make ahs       - Compile ahs.cu"
	@echo "  make clean     - Clean up generated binaries"
	@echo "  make help      - Print this help"

# Clean target
.PHONY: clean
clean:
	rm -f mmul ahs
