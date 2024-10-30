# Makefile

# Default target to build the test executable
.PHONY: all
all: test

# Rule for test
# Compile the test.cu file using nvcc and mpicxx as the C++ compiler
CC := nvcc
CXX := mpicxx
test: test.cu
	$(CC) -o test -ccbin=$(CXX) test.cu

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make clean     - Clean up generated binaries"
	@echo "  make test      - Compile test.cu"
	@echo "  make help      - Print this help"

# Clean target
.PHONY: clean
clean:
	rm -f test
