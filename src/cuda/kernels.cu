#include "kernels.h"

__global__ void monte_carlo_cuda_kernels::kernel_genchars(unsigned char* char_buffer, int buffer_len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init(clock64(), idx, 0, &state);
    
    for (int i = idx; i < buffer_len; i += gridDim.x * blockDim.x) {
        char_buffer[i] = curand(&state) % 256;
    }
}

__global__ void monte_carlo_cuda_kernels::kernel_monte_carlo_pi(unsigned char* char_buffer, int buffer_len, int* count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned char x, y;
    int local_count = 0;

    for (int i = idx; i < buffer_len / 2; i += gridDim.x * blockDim.x) {
        x = char_buffer[2 * i];
        y = char_buffer[2 * i + 1];
        if ((x * x) + (y * y) <= 65025) {
            local_count++;
        }
    }
    atomicAdd(count, local_count);
}

double monte_carlo_cuda_kernels::run_kernels_pi(int p_num_samples, int p_num_threads) {
    int buffer_len = p_num_samples;
    int num_threads = p_num_threads;
    unsigned char* device_buffer;
    int* device_count;
    int host_count;
    cudaMalloc(&device_buffer, sizeof(unsigned char) * buffer_len);
    cudaMalloc(&device_count, sizeof(int));

    kernel_genchars<<<1, num_threads>>>(device_buffer, buffer_len);
    cudaDeviceSynchronize();
    kernel_monte_carlo_pi<<<1, num_threads>>>(device_buffer, buffer_len, device_count);
    cudaDeviceSynchronize();

    cudaMemcpy(&host_count, device_count, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_buffer);
    cudaFree(device_count);
    return 4.0 * host_count / (buffer_len / 2);
}