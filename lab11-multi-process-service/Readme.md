# Lab 11: Multi-Process Service (MPS) with CUDA

This lab explores the impact of the NVIDIA CUDA Multi-Process Service (MPS) on kernel execution times under different rank and data size configurations. The assignment involves profiling a CUDA application without any modifications to its code, focusing solely on performance measurements with and without MPS.

## How to Run

The `Makefile` provided compiles all the necessary programs. You can run and clean the programs using the commands below:

### Compilation
```
make test
```

### Cleaning Up

```
make clean
```

## Brief Explanation of the Assignment


### 1. Profiling Without MPS (`test.cu`)

Profile the application without enabling MPS. Use different ranks (`1`, `4`) and data sizes (`1e7`, `1e9`). Below are the commands to run the profiling:

**1. Single Rank, Data Size = 1e7**
```
nsys profile --stats=true --show-output=true --gpu-metrics-device=all -t nvtx,cuda -s none -o 1_rank_no_MPS_N_1e7 -f true mpirun --allow-run-as-root -np 1 ./test 10000000 --speed-of-light=true
```
**2. Single Rank, Data Size = 1e9**
```
nsys profile --stats=true --show-output=true --gpu-metrics-device=all -t nvtx,cuda -s none -o 1_rank_no_MPS_N_1e9 -f true mpirun --allow-run-as-root -np 1 ./test 1073741824 --speed-of-light=true
```
**3. Four Ranks, Data Size = 1e7**
```
nsys profile --stats=true --show-output=true --gpu-metrics-device=all -t nvtx,cuda -s none -o 4_rank_no_MPS_N_1e7 -f true mpirun --allow-run-as-root -np 4 ./test 10000000 --speed-of-light=true
```
**4. Four Ranks, Data Size = 1e9**
```
nsys profile --stats=true --show-output=true --gpu-metrics-device=all -t nvtx,cuda -s none -o 4_rank_no_MPS_N_1e9 -f true mpirun --allow-run-as-root -np 4 ./test 1073741824 --speed-of-light=true
```

### 2. Profiling With MPS (`test.cu`)

NVIDIA's Multi-Process Service (MPS) allows multiple CUDA applications (or ranks in an MPI environment) to share the same GPU efficiently. This feature is especially useful in multi-GPU systems where maximizing GPU utilization is crucial. We expect to observe lower execution times for higher ranks with MPS enabled due to better GPU resource sharing and reduced contention. However, if the compute capacity is already saturated, enabling MPS may not lead to further performance improvements.

1. Start the MPS control daemon:
```
nvidia-cuda-mps-control -d
```

2. Repeat the profiling commands from Task 1 with MPS enabled.

    - **Single Rank, Data Size = 1e7**
        ```
        nsys profile --stats=true --show-output=true --gpu-metrics-device=all -t nvtx,cuda -s none -o 1_rank_with_MPS_N_1e7 -f true mpirun --allow-run-as-root -np 1 ./test 10000000 --speed-of-light=true
        ```

    - **Single Rank, Data Size = 1e9**
        ```
        nsys profile --stats=true --show-output=true --gpu-metrics-device=all -t nvtx,cuda -s none -o 1_rank_with_MPS_N_1e9 -f true mpirun --allow-run-as-root -np 1 ./test 1073741824 --speed-of-light=true
        ```
    - **Four Ranks, Data Size = 1e7**
        ```
        nsys profile --stats=true --show-output=true --gpu-metrics-device=all -t nvtx,cuda -s none -o 4_rank_with_MPS_N_1e7 -f true mpirun --allow-run-as-root -np 4 ./test 10000000 --speed-of-light=true
        ```
    - **Four Ranks, Data Size = 1e9**
        ```
        nsys profile --stats=true --show-output=true --gpu-metrics-device=all -t nvtx,cuda -s none -o 4_rank_with_MPS_N_1e9 -f true mpirun --allow-run-as-root -np 4 ./test 1073741824 --speed-of-light=true
        ```
3. Stop the MPS control daemon:
```
echo "quit" | nvidia-cuda-mps-control
```

### Results

After running all the profiling commands, fill in the following table with the time per kernel (ms) values obtained from the profiling outputs. 

| Data Size | Rank | MPS | Time per Kernel (ms) |
| --------- | ---- | --- | -------------------- |
| 1e9       | 1    | X   | TODO                 |
| 1e9       | 1    | O   | TODO                 |
| 1e9       | 4    | X   | TODO                 |
| 1e9       | 4    | O   | TODO                 |
| 1e7       | 1    | X   | TODO                 |
| 1e7       | 1    | O   | TODO                 |
| 1e7       | 4    | X   | TODO                 |
| 1e7       | 4    | O   | **TODO**                 |

