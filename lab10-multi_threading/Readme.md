# Lab 10: Multi-Threading with CUDA

This lab explores two main tasks involving multi-threading in CUDA: single GPU multi-threading using OpenMP and multi-GPU multi-threading using CUDA streams and OpenMP. The goal is to leverage both single GPU and multi-GPU configurations to maximize parallel processing efficiency.

## How to Run

The `Makefile` provided compiles all the necessary programs. You can run and clean the programs using the commands below:

### Compilation
```
# Compile all CUDA programs
make all

# Compile the single GPU OpenMP version
make openmp

# Compile the multi-GPU version
make multi_gpu
```

### Setting Environment Variables

The `Makefile` sets an environment variable to specify the number of OpenMP threads used:

```
export OMP_NUM_THREADS=8
```

This value is set to 8 by default but can be adjusted as needed to control the level of parallelism.

### Cleaning Up

```
make clean
```

## Brief Explanation of the Assignment

### 1. Single GPU Multi-Threading (`openmp.cu`)

This task focuses on using CUDA in conjunction with OpenMP to achieve parallel execution on a single GPU. The goal is to improve the performance of Gaussian PDF computation by parallelizing the processing of data chunks using multiple threads. Use OpenMP to parallelize the data chunks across multiple threads.

### 2. Multi-GPU Multi-Threading (`multi_gpu.cu`)

This task expands the scope to multiple GPUs, with the goal of utilizing more computational power to improve the overall throughput. CUDA streams are used in combination with OpenMP to enable concurrent execution across multiple GPUs.

**Key Steps:**

1. Use OpenMP to distribute workload across multiple GPUs.

2. Allocate device memory on each GPU and create multiple CUDA streams to perform asynchronous data transfers and computations.

3. Perform Gaussian PDF computation using multiple GPUs to process different chunks of data in parallel.
