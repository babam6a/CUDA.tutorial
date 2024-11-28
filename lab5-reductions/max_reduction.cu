#include <stdio.h>
#include <cstdlib>  // for rand()
#include <ctime>    // for seeding rand()

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

const size_t N = 8ULL * 1024ULL * 1024ULL;  // data size
const int BLOCK_SIZE = 256;  // CUDA maximum is 1024

/**
 * @brief CUDA kernel for maximum reduction across an array.
 * 
 * This kernel reduces an array to its maximum value using shared memory and a parallel
 * reduction technique. It uses a grid-stride loop to load data from global memory and performs
 * a sweep reduction within each block to find the maximum. The results are written back to 
 * the global memory.
 * 
 * @param gdata Pointer to the input array in device memory.
 * @param out   Pointer to the output array for reduced results (block-wise).
 * @param n     Total number of elements in the array.
 */
__global__ void reduce(int *gdata, int *out, size_t n) {
    /* TODO: Declare shared memory for storing partial results for this block */
    __shared__ int max_out[BLOCK_SIZE];

    /* TODO: Calculate thread ID within the block and initialize shared memory */
    if (threadIdx.x < BLOCK_SIZE)
        max_out[threadIdx.x] = 0;

    /* TODO: Calculate global thread index */
    int gidx = threadIdx.x + blockDim.x * blockIdx.x;

    /* TODO: Load data in grid-stride loop and store the maximum in shared memory */
    int stride = blockDim.x * gridDim.x;

    if (gidx < n) {
        int thread_max = 0;
        for (int i = gidx; i < n; i += stride) {
            thread_max = max(thread_max, gdata[i]);
        }
        max_out[threadIdx.x] = thread_max;
    }

    /* TODO: Synchronize threads before performing reduction on shared memory */
    __syncthreads();

    /* TODO: Perform parallel reduction using a sweep reduction method to find the max */
    if (threadIdx.x == 0){
        int total_max = 0;
        for (int j = 0; j < BLOCK_SIZE; j++) 
            total_max = max(total_max, max_out[j]);

    /* TODO: Write the result from thread 0 in each block to the global output array */
        out[blockIdx.x] = total_max;
    }
}

int main() {
    int *h_A, *h_max, *d_A, *d_sums;
    const int blocks = 640;  // Number of blocks for the first reduction stage
    h_A = new int[N];  // Allocate space for data in host memory
    h_max = new int;
    int max_val = 500;  // Set the expected maximum value


    srand(time(0));  // Seed random generator
    for (size_t i = 0; i < N; i++) {
        h_A[i] = rand() % 100;  // Random values between 0 and 99
    }
    h_A[100] = max_val;  // Insert known max value at index 100

    cudaMalloc(&d_A, N * sizeof(int));
    cudaMalloc(&d_sums, blocks * sizeof(int));  // Allocate space for block-wise max results
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    reduce<<<blocks, BLOCK_SIZE>>>(d_A, d_sums, N);
    cudaCheckErrors("reduction kernel stage 1 launch failure");

    reduce<<<1, BLOCK_SIZE>>>(d_sums, d_A, blocks);
    cudaCheckErrors("reduction kernel stage 2 launch failure");

    cudaMemcpy(h_max, d_A, sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");

    printf("Reduction output: %d, expected max reduction output: %d\n", *h_max, max_val);
    if (*h_max != max_val) {
        printf("Max reduction incorrect!\n");
        return -1;
    }
    printf("Max reduction correct!\n");

    delete[] h_A;
    delete h_max;
    cudaFree(d_A);
    cudaFree(d_sums);

    return 0;
}
