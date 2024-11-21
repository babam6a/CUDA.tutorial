# Lab 13: CUDA Graphs Lab

This lab focuses on exploring CUDA Graphs, including Stream Capture and Explicit Graph Creation with library calls. The tasks demonstrate how to use CUDA Graph APIs to optimize kernel launches and integrate third-party library calls like cuBLAS into graphs.

## How to Run
Use the provided Makefile to compile and manage the programs.

### Compilation
```bash
# Compile all programs
make all

# Compile Stream Capture example
make axpy_stream_capture

# Compile Explicit Graph Creation example
make axpy_cublas
```

### Cleaning Up
```bash
make clean
```

## Brief Explanation of the Assignment
### 1. Stream Capture (`axpy_stream_capture.cu`)
This task demonstrates how to use CUDA Stream Capture to define and execute a sequence of kernel launches across two streams. The captured sequence of operations is instantiated into a CUDA graph and executed efficiently.

![image](https://github.com/user-attachments/assets/be1e718c-8c82-4e50-bd1e-994a061a2acc)

**Key Concepts:**

- **Stream Capture:** Records operations into a cudaGraph_t from specified streams.
- **Graph Instantiation:** Converts the recorded graph into a cudaGraphExec_t for execution.
Performance Comparison: Measures execution time for both stream-based and graph-based approaches.

**Steps to Implement:**

1. Use `cudaStreamBeginCapture` to start recording operations.
2. Capture dependencies between kernels using CUDA events.
3. End the stream capture with `cudaStreamEndCapture`.
4. Instantiate the graph using `cudaGraphInstantiate`.
5. Execute the graph multiple times using `cudaGraphLaunch`.

### 2. Explicit Graph Creation with Library Call (`axpy_cublas.cu`)

This task focuses on explicitly creating a CUDA graph with multiple nodes, including a child graph encapsulating a cuBLAS axpy operation.

![image](https://github.com/user-attachments/assets/eacba280-3055-4e2f-a5e0-88e4de2e3848)


**Key Concepts:**

- **Kernel Nodes:** Add kernel execution nodes to the graph using `cudaGraphAddKernelNode`.
- **Child Graph Nodes:** Use `cudaGraphAddChildGraphNode` to add a subgraph for the `cuBLAS` operation.
- **Dependencies:** Define dependencies between nodes to ensure correct execution order.

**Steps to Implement:**

1. Create an empty CUDA graph using `cudaGraphCreate`.
2. Add a kernel node (`kernel_a`) to the graph.
3. Use stream capture to encapsulate a cuBLAS axpy operation into a child graph.
4. Add the child graph as a node in the main graph using `cudaGraphAddChildGraphNode`.
5. Add another kernel node (`kernel_c`) with dependencies on the previous nodes.
5. Instantiate and execute the graph multiple times using `cudaGraphInstantiate` and `cudaGraphLaunch`.
