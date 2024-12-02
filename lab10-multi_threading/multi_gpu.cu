#include <math.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include <omp.h>

// modifiable
typedef float ft;
const int chunks = 64;
const size_t ds = 20*1024*1024*chunks;
const int count = 22;
const int num_streams = 8;

// not modifiable
const float sqrt_2PIf = 2.5066282747946493232942230134974f;
const double sqrt_2PI = 2.5066282747946493232942230134974;

/**
 * @brief Compute the Gaussian PDF for a given value.
 * 
 * This function calculates the Gaussian probability density function (PDF)
 * for a given value and standard deviation.
 * 
 * @param val   The value for which the PDF is computed.
 * @param sigma The standard deviation for the Gaussian distribution.
 * @return float The computed Gaussian PDF value.
 */
__device__ float gpdf(float val, float sigma) {
  return expf(-0.5f * val * val) / (sigma * sqrt_2PIf);
}

/**
 * @brief Compute the Gaussian PDF for a given double value.
 * 
 * Similar to the float version, but works with double precision.
 * 
 * @param val   The value for which the PDF is computed.
 * @param sigma The standard deviation for the Gaussian distribution.
 * @return double The computed Gaussian PDF value.
 */
__device__ double gpdf(double val, double sigma) {
  return exp(-0.5 * val * val) / (sigma * sqrt_2PI);
}

/**
 * @brief CUDA kernel to compute the average Gaussian PDF over a window.
 * 
 * This kernel computes the average Gaussian PDF value over a window of values
 * around each point. It processes the data in parallel using CUDA threads.
 * 
 * @param x     Pointer to the input data array.
 * @param y     Pointer to the output data array.
 * @param mean  The mean of the Gaussian distribution.
 * @param sigma The standard deviation of the Gaussian distribution.
 * @param n     The number of data points.
 */
__global__ void gaussian_pdf(const ft * __restrict__ x, ft * __restrict__ y, const ft mean, const ft sigma, const int n) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < n) {
    ft in = x[idx] - (count / 2) * 0.01f;
    ft out = 0;
    for (int i = 0; i < count; i++) {
      ft temp = (in - mean) / sigma;
      out += gpdf(temp, sigma);
      in += 0.01f;
    }
    y[idx] = out / count;
  }
}

// error check macro
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

// host-based timing
#define USECPSEC 1000000ULL

/**
 * @brief Measure elapsed time in microseconds.
 * 
 * This function calculates the elapsed time in microseconds since the provided
 * start time.
 * 
 * @param start  The start time in microseconds.
 * @return unsigned long long The elapsed time in microseconds.
 */
unsigned long long dtime_usec(unsigned long long start) {
  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec * USECPSEC) + tv.tv_usec) - start;
}

int main() {
	int num_devices;
	cudaGetDeviceCount(&num_devices);
	if (num_devices < 2) {
		std::cerr << "This code requires at least 2 GPUs." << std::endl;
		return 1;
	}

	ft *h_x, *h_y, *h_y1;

	cudaHostAlloc(&h_x,  ds * sizeof(ft), cudaHostAllocDefault);
	cudaHostAlloc(&h_y,  ds * sizeof(ft), cudaHostAllocDefault);
	cudaHostAlloc(&h_y1, ds * sizeof(ft), cudaHostAllocDefault);

	// Initialize input data
	for (size_t i = 0; i < ds; i++) {
		h_x[i] = rand() / (ft)RAND_MAX;
	}

	// Warm-up on GPU 0
	cudaSetDevice(0);
	ft *d_x, *d_y;

	cudaMalloc(&d_x, ds * sizeof(ft));
	cudaMalloc(&d_y, ds * sizeof(ft));
	cudaMemcpy(d_x, h_x, ds * sizeof(ft), cudaMemcpyHostToDevice);
	gaussian_pdf<<<(ds + 255) / 256, 256>>>(d_x, d_y, 0.0, 1.0, ds);
	cudaDeviceSynchronize();
	cudaFree(d_x);
	cudaFree(d_y);
	cudaCheckErrors("warm-up error");

	// Allocate memory on multiple GPUs
	ft *d_x_multi[num_devices], *d_y_multi[num_devices];
	cudaStream_t streams[num_devices][num_streams];

	
	for (int d = 0; d < num_devices; d++) {
		/**
		* TODO: Allocate device memory and create CUDA streams for each GPU.
		* Use cudaMalloc to allocate memory for d_x_multi and d_y_multi for each GPU.
		* Use cudaStreamCreate to create streams for asynchronous execution.
		*/
		cudaSetDevice(d);
		cudaMalloc(&d_x_multi[d], ds * sizeof(ft) / num_devices);
		cudaMalloc(&d_y_multi[d], ds * sizeof(ft) / num_devices);
		for (int s = 0; s < num_streams; s++)
			cudaStreamCreate(&streams[d][s]);

		cudaCheckErrors("memory/stream allocation error");
	}

	// Non-stream version for comparison
	unsigned long long et1 = dtime_usec(0);
	cudaSetDevice(0);
	cudaMalloc(&d_x, ds * sizeof(ft));
	cudaMalloc(&d_y, ds * sizeof(ft));
	cudaMemcpy(d_x, h_x, ds * sizeof(ft), cudaMemcpyHostToDevice);
	gaussian_pdf<<<(ds + 255) / 256, 256>>>(d_x, d_y, 0.0, 1.0, ds);
	cudaMemcpy(h_y1, d_y, ds * sizeof(ft), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	et1 = dtime_usec(et1);
	std::cout << "non-stream elapsed time: " << et1 / (float)USECPSEC << std::endl;
	cudaFree(d_x);
	cudaFree(d_y);
	cudaCheckErrors("non-stream execution error");

#ifdef USE_STREAMS
	// Multi-GPU with streams
	unsigned long long et = dtime_usec(0);

	/**
	* TODO: Use OpenMP to parallelize the loop for multi-GPU chunk processing.
	* This directive will distribute the loop iterations across multiple GPUs.
	* Ensure OpenMP is enabled during compilation for effective parallel execution.
	*/
#pragma omp parallel for
	for (int d = 0; d < num_devices; d++) {
		/**
		* TODO: Set the device for the current iteration. 
		*/
		cudaSetDevice(d);
		/* chunk index */
		for (int i = 0; i < chunks / num_devices; i++) {
			int chunk_idx = d * (chunks / num_devices) + i;

			/**
			* TODO: Asynchronously copy input data to the device.
			* Use cudaMemcpyAsync to transfer data for each chunk from h_x to d_x_multi.
			*/
			int chunk_size = ds / chunks;
			int stream_size = chunk_size / num_streams;
			for (int s = 0; s < num_streams; s++) {
				cudaMemcpyAsync(&d_x_multi[d][i * chunk_size + s * stream_size], &h_x[chunk_idx * chunk_size + s * stream_size], stream_size * sizeof(ft), cudaMemcpyHostToDevice, streams[d][s]);

			/**
			* TODO: Launch the Gaussian PDF kernel asynchronously.
			* Launch the kernel for each chunk using the respective CUDA stream.
			*/

				gaussian_pdf<<<(ds + 255) / 256, 256, 0, streams[d][s]>>>(&d_x_multi[d][i * chunk_size + s * stream_size], &d_y_multi[d][i * chunk_size + s * stream_size], 0.0, 1.0, stream_size);

			/**
			* TODO: Asynchronously copy output data back to the host.
			* Use cudaMemcpyAsync to transfer results from d_y_multi to h_y.
			*/
				cudaMemcpyAsync(&h_y[chunk_idx * chunk_size + s * stream_size], &d_y_multi[d][i * chunk_size + s * stream_size],  stream_size * sizeof(ft), cudaMemcpyDeviceToHost, streams[d][s]);
			}
		}
	}

  for (int d = 0; d < num_devices; d++) {
    cudaSetDevice(d);
    cudaDeviceSynchronize();
  }
  cudaCheckErrors("multi-GPU streams execution error");

  et = dtime_usec(et);

  // Verify results
  for (int i = 0; i < ds; i++) {
    if (h_y[i] != h_y1[i]) {
      std::cout << "mismatch at " << i << " was: " << h_y[i] << " should be: " << h_y1[i] << std::endl;
      return -1;
    }
  }

  std::cout << "multi-GPU streams elapsed time: " << et / (float)USECPSEC << std::endl;
#endif

  // Cleanup
  for (int d = 0; d < num_devices; d++) {
    cudaSetDevice(d);
    cudaFree(d_x_multi[d]);
    cudaFree(d_y_multi[d]);
    for (int i = 0; i < num_streams; i++) {
      cudaStreamDestroy(streams[d][i]);
    }
  }

  cudaFreeHost(h_x);
  cudaFreeHost(h_y);
  cudaFreeHost(h_y1);

  return 0;
}
