# Lab 12: Debugging with CUDA Tools

In this lab, we will focus on debugging CUDA applications using `compute-sanitizer` and `cuda-gdb`. This lab includes two tasks: one for detecting memory issues and race conditions in a matrix multiplication program (`mmul.cu`) using `compute-sanitizer`, and another for identifying the cause of incorrect results in an alternating harmonic series (AHS) calculation (`ahs.cu`) using `cuda-gdb`.

## How to Run
The Makefile provided compiles both programs required for debugging. Use the commands below for compilation and cleanup:

### Compilation
```
make mmul
make ahs
```
Cleaning Up
```
make clean
```
## Brief Explanation of the Assignment
### 1. Debugging Matrix Multiplication (`mmul.cu`)
The matrix multiplication code provided (`mmul.cu`) compiles without CUDA errors, but does not produce correct results. We suspect a memory issue is causing this incorrect behavior. The steps below guide you through identifying and fixing these issues using `compute-sanitizer`:

**1. Memory Error Detection**

First, use compute-sanitizer with the memcheck tool to identify any out-of-bound memory accesses that may be affecting the result:

```
compute-sanitizer ./mmul
```

This tool will help pinpoint the lines where out-of-bound accesses occur. Correct any indexing issues as indicated by memcheck.


**2. Race Condition Detection**

After fixing memory issues, run the program again. If mmul now reports "Success!" for correct results, proceed to check for any race conditions. Use the racecheck tool with compute-sanitizer to identify any synchronization issues:

```
compute-sanitizer --tool racecheck ./mmul
```
Any race conditions reported by racecheck can be fixed by ensuring proper synchronization (e.g., adding `__syncthreads()`in the kernel where needed). If fixed correctly, the program should output "Success!" without race conditions.

### 2. Debugging Alternating Harmonic Series (AHS) (`ahs.cu`)

The second debugging task focuses on finding the sum of an alternating harmonic series using `ahs.cu`. Currently, running the code results in an unexpected output:

```
Estimated value: -inf Expected value: 0.693147
```

This incorrect output (`-inf`) suggests that an overflow (`inf`) may have occurred during the calculation, either in the reduction step or while generating individual terms. Use `cuda-gdb` to identify and resolve this issue.

**Setting Up Debugging**

Start `cuda-gdb` and set a breakpoint at line 15 in `ahs.cu` where the issue is likely arising:

```
cuda-gdb ./ahs
(cuda-gdb) break ahs.cu:15
```

**Stepping Through the Code**

Use the step command to move through each line and observe the calculation process:

```
(cuda-gdb) step
```
Use `next` to skip function calls if needed.

**Inspecting Variables**

Use print to inspect the values of variables and identify where inf may be introduced. 

```
(cuda_gdb) print <variable_name> 
```

Investigate intermediate calculations in both the term generation and reduction steps to isolate the cause.

**Expected Output**

After resolving the issue, the program should output the `Success!` for the alternating harmonic series sum.