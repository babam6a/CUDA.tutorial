# CASYS Kwonst CUDA Tutorial
This repository contains tutorials and labs for learning CUDA programming, including concepts like shared memory, parallelism, and kernel optimization. The repository is structured into different labs with corresponding Makefiles for easy compilation and execution.

## Lab
- **lab1-intro**: Introduction to basic CUDA programming with "hello world," vector addition, and matrix multiplication examples.

- **lab2-shared_memory**: Optimizing matrix multiplication and 1D stencil computations using shared memory in CUDA.

- **lab3-grid_stride_loop**: Using grid-stride loops for efficient vector addition and better GPU utilization.

- **lab4-matrix_sums**: Calculating row and column sums of a matrix in CUDA and analyzing performance with Nsight Compute.

- **lab5-reductions**: Advanced reduction techniques in CUDA, including atomic reduction, parallel reduction with atomic finish, and warp-shuffle reduction.

- **lab6-managed_memory**: Porting linked lists and arrays to GPU using manual memory management, Unified Memory, and prefetching, with a focus on profiling memory behavior.

- **lab7-concurrency**: Exploring concurrency using CUDA streams for overlapping computation and memory transfers, and distributing tasks across multiple GPUs.

- **lab8-optimizing**: Optimizing CUDA matrix transpose using global memory, shared memory, and mitigating bank conflicts.

- **lab9-cooperative-groups**: Using CUDA cooperative groups for reductions and stream compaction with thread-level and grid-level synchronization.

- **lab10-multi-threading**: Exploring single and multi-GPU configurations using OpenMP and CUDA streams to optimize tasks.

- **lab11-multi-process-service**: Profiling the impact of NVIDIA MPS on kernel execution times to understand GPU resource sharing.


## How to Start
To get started with this CUDA tutorial, follow these steps:
```
# Clone the repository
git clone https://github.com/SeungjaeLim/kwonst_CUDA_tutorial.git

# Navigate into the project directory
cd kwonst_CUDA_tutorial

# Build and run the Docker container with the tutorial environment
make up
```
### Docker Setup
This project uses Docker to create a containerized environment with CUDA support. You can build and run the container using the provided Makefile commands.

### Makefile Commands
The Makefile provides various management commands for setting up and running the project. Here's how to use the Makefile:

```
make build            # Build the cu_tutorial project.
make preprocess       # Preprocess step.
make run              # Boot up Docker container.
make up               # Build and run the project.
make rm               # Remove Docker container.
make stop             # Stop Docker container.
make reset            # Stop and remove Docker container.
make docker-setup     # Setup Docker permissions for the user.
```