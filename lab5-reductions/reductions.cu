#include <stdio.h>
#include <cstdlib>  // for rand()
#include <ctime>    // for seeding rand()
#include <cmath>    // for abs

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

/* TODO: Experiment with the following values in turn to observe the change in duration difference between kernels */
const size_t N = 8ULL*1024ULL*1024ULL;  // 8M
//const size_t N = 256*640; // 163840
//const size_t N = 32ULL*1024ULL*1024ULL; // 32M

const int BLOCK_SIZE = 256;  // CUDA maximum is 1024

/**
 * @brief CUDA kernel for atomic reduction.
 * 
 * Each thread adds an element from the input array `gdata` to a shared output
 * using atomic operations to avoid race conditions.
 * 
 * @param gdata Pointer to the input data (in device memory).
 * @param out   Pointer to the output sum (in device memory).
 * 
 * Details:
 * - This kernel uses atomic operations to ensure that multiple threads can safely
 *   update the shared `out` variable simultaneously.
 */
__global__ void atomic_red(const int *gdata, int *out) { 
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        /* TODO: Use atomicAdd to safely add the value from gdata to the output sum */
        atomicAdd(out, gdata[idx]);
    }
}

/**
 * @brief CUDA kernel for parallel reduction using atomic operations.
 * 
 * This kernel performs reduction similar to the previous one, but uses atomicAdd
 * to accumulate the final result.
 * 
 * @param gdata Pointer to the input data (in device memory).
 * @param out   Pointer to the output sum (in device memory).
 */
__global__ void reduce_a(int *gdata, int *out) {
    /* TODO: Declare shared memory for this block */
    __shared__ int temp[BLOCK_SIZE];

    /* TODO: Initialize thread-specific local sum in shared memory */
    temp[threadIdx.x] = 0;

    /* TODO: Calculate global thread index */
    int gidx = threadIdx.x + blockDim.x * blockIdx.x;

    /* TODO: Load data in chunks using grid-stride loop, accumulating the sum */
    int stride = blockDim.x * gridDim.x;
    for (int i = gidx; i < N; i += stride) {
        temp[threadIdx.x] += gdata[i];
    }

    /* TODO: Perform parallel reduction to sum the elements in shared memory */
    for (int j = 2; j < BLOCK_SIZE; j *= 2) {
        if (threadIdx.x % j == 0)
            temp[threadIdx.x] += temp[threadIdx.x + (j / 2)];
        __syncthreads();
    }

    /* TODO: Use atomicAdd to accumulate the final result safely */
    if (threadIdx.x == 0)
        atomicAdd(out, temp[0]);
}

/**
 * @brief CUDA kernel for warp-shuffle based parallel reduction.
 * 
 * This kernel uses warp-shuffle instructions to perform a reduction within a warp.
 * The result of each warp is stored in shared memory, and then a final warp reduces
 * the results from all warps.
 * 
 * @param gdata Pointer to the input data (in device memory).
 * @param out   Pointer to the output sum (in device memory).
 */
__global__ void reduce_ws(int *gdata, int *out) {
    /* TODO: Declare shared memory to hold the results of each warp(32 threads) */
    __shared__ int temp[8];
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int val = 0;
    unsigned mask = 0xFFFFFFFFU;
    int lane = threadIdx.x % warpSize;  // Lane within a warp
    int warpID = threadIdx.x / warpSize;  // Warp ID

    /* TODO: Load data in grid-stride loop, accumulating in val */
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride)
        val += gdata[i];
    __syncthreads();

    /* TODO: Perform warp-level reduction using __shfl_down_sync (within each warp) */
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(mask, val, offset);
    
    /* TODO: Write the warp's result to shared memory */
    if (lane == 0)
        temp[warpID] = val;
    __syncthreads();

    /* TODO: If warp 0, perform final reduction on the values from each warp */
    if (warpID == 0) {
        int total_sum = temp[lane];
        total_sum += __shfl_down_sync(mask, total_sum, 4);
        total_sum += __shfl_down_sync(mask, total_sum, 2);
        total_sum += __shfl_down_sync(mask, total_sum, 1);
        /* TODO: Use atomicAdd to safely add the final result to the global sum */
        if (tid == 0)
            atomicAdd(out, total_sum);
    }
}


int main() {
    int *h_A, *h_sum, *d_A, *d_sum;
    h_A = new int[N];  // Allocate space for data in host memory
    h_sum = new int;

    
    srand(time(0));
    int cpu_sum = 0;
    for (size_t i = 0; i < N; i++) {
        h_A[i] = rand() % 100;  // Random int between 0 and 99
        cpu_sum += h_A[i];  // Calculate CPU sum for validation
    }

    cudaMalloc(&d_A, N * sizeof(int));  
    cudaMalloc(&d_sum, sizeof(int));  
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    cudaMemset(d_sum, 0, sizeof(int));
    cudaCheckErrors("cudaMemset failure");

    atomic_red<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_sum);
    cudaCheckErrors("atomic reduction kernel launch failure");

    cudaMemcpy(h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("atomic reduction kernel execution failure or cudaMemcpy H2D failure");

    if (abs(*h_sum - cpu_sum) > 0) {
        printf("atomic sum reduction incorrect! CPU: %d, GPU: %d\n", cpu_sum, *h_sum);
        return -1;
    }
    printf("atomic sum reduction correct! CPU: %d, GPU: %d\n", cpu_sum, *h_sum);

    const int blocks = 640;
    cudaMemset(d_sum, 0, sizeof(int));
    cudaCheckErrors("cudaMemset failure");

    reduce_a<<<blocks, BLOCK_SIZE>>>(d_A, d_sum);
    cudaCheckErrors("reduction w/atomic kernel launch failure");

    cudaMemcpy(h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("reduction w/atomic kernel execution failure or cudaMemcpy H2D failure");

    if (abs(*h_sum - cpu_sum) > 0) {
        printf("reduction w/atomic sum incorrect! CPU: %d, GPU: %d\n", cpu_sum, *h_sum);
    }
    printf("reduction w/atomic sum correct! CPU: %d, GPU: %d\n", cpu_sum, *h_sum);

    cudaMemset(d_sum, 0, sizeof(int));
    cudaCheckErrors("cudaMemset failure");

    reduce_ws<<<blocks, BLOCK_SIZE>>>(d_A, d_sum);
    cudaCheckErrors("reduction warp shuffle kernel launch failure");

    cudaMemcpy(h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("reduction warp shuffle kernel execution failure or cudaMemcpy H2D failure");

    if (abs(*h_sum - cpu_sum) > 0) {
        printf("reduction warp shuffle sum incorrect! CPU: %d, GPU: %d\n", cpu_sum, *h_sum);
        return -1;
    }
    printf("reduction warp shuffle sum correct! CPU: %d, GPU: %d\n", cpu_sum, *h_sum);

    delete[] h_A;
    delete h_sum;
    cudaFree(d_A);
    cudaFree(d_sum);

    return 0;
}
