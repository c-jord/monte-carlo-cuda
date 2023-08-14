#ifndef KERNELS_H
#define KERNELS_H

#include <stdio.h>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace monte_carlo_cuda_kernels
{
    __global__ void kernel_genchars(unsigned char* char_buffer, int buffer_len);
    __global__ void kernel_monte_carlo_pi(unsigned char* char_buffer, int buffer_len, int* count);
    __declspec(dllexport) double run_kernels_pi(int p_num_samples, int p_num_threads);
} // namespace monte_carlo_cuda_kernels
#endif