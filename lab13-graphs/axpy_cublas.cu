#include <stdio.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <chrono>

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

using namespace std::chrono;

// Simple short kernels
__global__
void kernel_a(float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] += 1;
}

__global__
void kernel_c(float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] += 1;
}

int main() {
    /**
     * @brief Main function to build a CUDA graph with 3 nodes,
     * measure execution time, and validate the node count and performance.
     */

    cudaStream_t stream1;
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, stream1);

    // Set up host data and initialize
    float* h_x = (float*)malloc(N * sizeof(float));
    float* h_y = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        h_x[i] = float(i);
        h_y[i] = float(i);
    }

    // Set up device data
    float* d_x;
    float* d_y;
    float d_a = 5.0;
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));
    cudaCheckErrors("cudaMalloc failed");

    cublasSetVector(N, sizeof(h_x[0]), h_x, 1, d_x, 1); // similar to cudaMemcpyHtoD
    cublasSetVector(N, sizeof(h_y[0]), h_y, 1, d_y, 1); // similar to cudaMemcpyHtoD
    cudaCheckErrors("cublasSetVector failed");

    // Set up graph
    cudaGraph_t graph;
    cudaGraph_t libraryGraph;
    std::vector<cudaGraphNode_t> nodeDependencies;
    cudaGraphNode_t kernelNode1, kernelNode2, libraryNode;
    cudaKernelNodeParams kernelNode1Params{0};
    cudaKernelNodeParams kernelNode2Params{0};

    /* TODO: Create a CUDA Graph
    * - Use `cudaGraphCreate` to initialize an empty CUDA graph.
    * - Pass a pointer to the graph and set the flags to `0` (default behavior).
    * - This graph will later be populated with nodes representing kernels and library calls.
    */

    cudaCheckErrors("cudaGraphCreate failure");

    // Kernel arguments
    void* kernelArgs[2] = {(void*)&d_x, (void*)&d_y};
    int threads = 512;
    int blocks = (N + (threads - 1)) / threads;

    // Adding 1st node: kernel_a
    kernelNode1Params.func = (void*)kernel_a;
    kernelNode1Params.gridDim = dim3(blocks, 1, 1);
    kernelNode1Params.blockDim = dim3(threads, 1, 1);
    kernelNode1Params.sharedMemBytes = 0;
    kernelNode1Params.kernelParams = (void**)kernelArgs;
    kernelNode1Params.extra = NULL;

    /* TODO: Add a Kernel Node to the CUDA Graph
    * - Use `cudaGraphAddKernelNode` to add a kernel execution node to the graph.
    * - Specify the graph where the node should be added and the kernel parameters:
    *   - `kernelNode1`: Pointer to the kernel node being created.
    *   - `graph`: The CUDA graph where the node will be added.
    *   - `NULL`: No dependencies for this node (it's the first node in the graph).
    *   - `0`: Number of dependencies (none for this node).
    *   - `kernelNode1Params`: Specifies the kernel function and launch configuration.
    * - This node represents the execution of `kernel_a` in the graph.
    */

    cudaCheckErrors("Adding kernelNode1 failed");

    /* TODO: Add the kernel node to the dependency list
    * - Use `nodeDependencies.push_back` to add `kernelNode1` as a dependency.
    * - This ensures that any subsequent nodes in the graph will wait for 
    *   `kernelNode1` to complete before executing.
    * - Dependencies help define the execution order of the graph's nodes.
    */


    /* TODO: Begin capturing operations in a CUDA stream
    * - Use `cudaStreamBeginCapture` to start recording operations in `stream1`.
    * - Set the capture mode to `cudaStreamCaptureModeGlobal` to include all global dependencies.
    * - The recorded operations will later be encapsulated into a subgraph.
    */

    cudaCheckErrors("Stream capture begin failure");

    // Library call
    cublasSaxpy(cublas_handle, N, &d_a, d_x, 1, d_y, 1);
    cudaCheckErrors("cublasSaxpy failure");

    /* TODO: End the capture of operations in the CUDA stream
    * - Use `cudaStreamEndCapture` to finalize the recording of operations in `stream1`.
    * - Store the resulting subgraph in `libraryGraph`.
    * - This completes the capture of all operations recorded after `cudaStreamBeginCapture`.
    */

    cudaCheckErrors("Stream capture end failure");

    /* TODO: Add a child graph node to the main CUDA graph
    * - Use `cudaGraphAddChildGraphNode` to include the captured `libraryGraph` as a child node in `graph`.
    * - Specify dependencies to ensure this node executes after its dependent nodes.
    *   - `libraryNode`: Pointer to the child graph node being created.
    *   - `graph`: The main CUDA graph where the child node is added.
    *   - `nodeDependencies.data()`: List of dependencies for this node.
    *   - `nodeDependencies.size()`: Number of dependencies.
    */

    cudaCheckErrors("Adding libraryNode failed");

    /* TODO: Update node dependencies for subsequent graph nodes
    * - Clear the `nodeDependencies` vector to prepare for the next node.
    * - Add `libraryNode` to the dependency list to ensure future nodes wait for its completion.
    * - Dependencies define the execution order within the graph.
    */


    // Adding 3rd node: kernel_c
    kernelNode2Params.func = (void*)kernel_c;
    kernelNode2Params.gridDim = dim3(blocks, 1, 1);
    kernelNode2Params.blockDim = dim3(threads, 1, 1);
    kernelNode2Params.sharedMemBytes = 0;
    kernelNode2Params.kernelParams = (void**)kernelArgs;
    kernelNode2Params.extra = NULL;

    /* TODO: Add a second kernel node to the CUDA graph
    * - Use `cudaGraphAddKernelNode` to add `kernelNode2` to the main graph.
    * - Specify its dependencies to ensure it executes after all nodes in `nodeDependencies`.
    *   - `kernelNode2`: Pointer to the kernel node being created.
    *   - `graph`: The main CUDA graph where the node will be added.
    *   - `nodeDependencies.data()`: Array of dependencies for this node.
    *   - `nodeDependencies.size()`: Number of dependencies for this node.
    *   - `kernelNode2Params`: Launch parameters for the kernel, including:
    *       - Function pointer (`kernel_c`).
    *       - Grid and block dimensions.
    *       - Shared memory size.
    *       - Kernel arguments.
    * - This step defines the execution of `kernel_c` in the graph.
    */

    cudaCheckErrors("Adding kernelNode2 failed");

    // Validate the number of nodes in the graph
    cudaGraphNode_t* nodes = NULL;
    size_t numNodes = 0;
    cudaGraphGetNodes(graph, nodes, &numNodes);
    cudaCheckErrors("Graph get nodes failed");
    printf("Number of the nodes in the graph = %zu\n", numNodes);

    if (numNodes != 3) {
        printf("FAIL: Graph does not contain the expected number of nodes (3).\n");
        return 1;
    }

    /* TODO: Instantiate the CUDA Graph
    * - Use `cudaGraphInstantiate` to create an executable instance of the graph.
    *   - `instance`: A pointer to the executable graph instance being created.
    *   - `graph`: The CUDA graph to be instantiated.
    *   - `NULL, NULL`: Pointers for error and log information (not used here).
    *   - `0`: Flags for default behavior.
    * - This step finalizes the graph and prepares it for execution on the GPU.
    */

    cudaCheckErrors("Graph instantiation failed");

    // Measure execution time
    auto t1 = high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        /* TODO: Launch the instantiated CUDA Graph multiple times
        *   - `cudaGraphLaunch`: Launches the executable graph instance on `stream1`.
        *       - `instance`: The instantiated graph to execute.
        *       - `stream1`: The stream on which the graph execution will run.
        *   - `cudaStreamSynchronize`: Ensures all operations in `stream1` are completed before proceeding.
        *       - This step is critical to synchronize the host with the GPU to avoid overlapping executions.
        * - This loop simulates repeated execution of the graph
        */

    }
    auto t2 = high_resolution_clock::now();
    duration<double> exec_time = duration_cast<duration<double>>(t2 - t1);
    cudaCheckErrors("Graph launch failed");
    printf("Graph execution time: %f seconds\n", exec_time.count());

    // Validate graph execution

    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Memcpy failed");


    if (valid && numNodes == 3) {
        printf("PASS: Graph execution is correct and contains 3 nodes.\n");
    } else {
        printf("FAIL: Graph execution validation failed.\n");
    }

    // Cleanup
    cudaGraphDestroy(graph);
    cudaFree(d_x);
    cudaFree(d_y);
    cublasDestroy(cublas_handle);
    free(h_x);
    free(h_y);

    return 0;
}
