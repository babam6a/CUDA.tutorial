#include <stdio.h>
#include <cuda_runtime_api.h>
#include <ctime>
#include <ratio>
#include <chrono>
#include <iostream>
#include <cmath>

using namespace std::chrono;  

// Error checking macro
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

#define N 500000
#define EPSILON 1e-5

/**
 * @brief CUDA kernel to perform an element-wise operation on two arrays.
 * 
 * This kernel performs the operation `y[idx] = 2.0 * x[idx] + y[idx]` 
 * for each element in the array within the given index range.
 *
 * @param x Input array.
 * @param y Output array.
 */
__global__
void kernel_a(float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] = 2.0 * x[idx] + y[idx];
}

__global__
void kernel_b(float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] = 2.0 * x[idx] + y[idx];
}

__global__
void kernel_c(float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] = 2.0 * x[idx] + y[idx];
}

__global__
void kernel_d(float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] = 2.0 * x[idx] + y[idx];
}

int main() {
    // Set up and create events
    cudaEvent_t event1, event2;
    cudaEventCreateWithFlags(&event1, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&event2, cudaEventDisableTiming);

    // Set up and create streams
    const int num_streams = 2;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    // Allocate and initialize host data
    float* h_x = (float*)malloc(N * sizeof(float));
    float* h_y = (float*)malloc(N * sizeof(float));
    float* h_y_no_graph = (float*)malloc(N * sizeof(float));
    float* h_y_graph = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        h_x[i] = (float)i;
        h_y[i] = (float)i;
    }

    // Allocate device memory
    float* d_x;
    float* d_y;
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Graph execution setup
    bool graphCreated = false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    int threads = 512;
    int blocks = (N + threads - 1) / threads;

    // Timing without graph
    auto t1 = high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        kernel_a<<<blocks, threads, 0, streams[0]>>>(d_x, d_y);
        cudaEventRecord(event1, streams[0]);
        cudaStreamWaitEvent(streams[0], event1);
        kernel_b<<<blocks, threads, 0, streams[0]>>>(d_x, d_y);
        cudaStreamWaitEvent(streams[1], event1);
        kernel_c<<<blocks, threads, 0, streams[1]>>>(d_x, d_y);
        cudaEventRecord(event2, streams[1]);
        cudaStreamWaitEvent(streams[0], event2);
        kernel_d<<<blocks, threads, 0, streams[0]>>>(d_x, d_y);
    }
    cudaDeviceSynchronize();
    auto t2 = high_resolution_clock::now();
    duration<double> no_graph_time = duration_cast<duration<double>>(t2 - t1);
    cudaMemcpy(h_y_no_graph, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Reset device memory
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Graph creation and execution
    t1 = high_resolution_clock::now();
    if (!graphCreated) {
        /* TODO: Start capturing the operations into a CUDA graph.
        * - Use `cudaStreamBeginCapture` to start capturing operations from the specified stream.
        * - Ensure the capture mode is set to `cudaStreamCaptureModeGlobal`.
        */
        

        /* TODO: Record events and use streams to manage dependencies during graph capture.
        * - Launch `kernel_a` and record an event after its completion.
        * - Wait for this event in a different stream before launching dependent kernels.
        * - Ensure proper event recording and synchronization between `kernel_a`, `kernel_b`, and `kernel_c`.
        */
        

        /* TODO: End capturing the operations into a CUDA graph.
        * - Record a final event for `kernel_c` in the second stream.
        * - Use `cudaStreamEndCapture` to end the capture and store the resulting graph.
        */
        

        /* TODO: Instantiate the CUDA graph and prepare it for execution.
        * - Use `cudaGraphInstantiate` to create an executable graph instance from the captured graph.
        */
       
        graphCreated = true;
    }
    for (int i = 0; i < 1000; ++i) {
        /* TODO: Launch the instantiated CUDA graph.
        * - Use `cudaGraphLaunch` to execute the graph on the specified stream.
        * - Run the graph multiple times in a loop and ensure proper synchronization after each launch.
        */
        
    }
    cudaDeviceSynchronize();
    t2 = high_resolution_clock::now();
    duration<double> graph_time = duration_cast<duration<double>>(t2 - t1);
    cudaMemcpy(h_y_graph, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Validate results
    bool results_match = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_y_no_graph[i] - h_y_graph[i]) > EPSILON) {
            results_match = false;
            std::cout << "Mismatch at index " << i << ": no_graph=" 
                      << h_y_no_graph[i] << ", graph=" << h_y_graph[i] << std::endl;
            break;
        }
    }

    // Print results
    if (results_match && graph_time.count() < no_graph_time.count()) {
        std::cout << "PASS: Results match and graph execution is faster.\n";
    } else if (!results_match) {
        std::cout << "FAIL: Results do not match.\n";
    } else {
        std::cout << "FAIL: Graph execution is not faster.\n";
    }

    std::cout << "No Graph Time: " << no_graph_time.count() << " s\n";
    std::cout << "Graph Execution Time: " << graph_time.count() << " s\n";

    // Cleanup
    cudaGraphDestroy(graph);
    cudaFree(d_x);
    cudaFree(d_y);
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    free(h_x);
    free(h_y);
    free(h_y_no_graph);
    free(h_y_graph);

    return 0;
}
