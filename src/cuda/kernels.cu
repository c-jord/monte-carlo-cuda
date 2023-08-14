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

void monte_carlo_cuda_kernels::run_kernels() {
    int buffer_len = 10;
    int num_threads = 1;
    unsigned char* host_buffer = (unsigned char*)malloc(sizeof(unsigned char) * buffer_len);
    unsigned char* device_buffer;
    cudaMallocManaged(&device_buffer, sizeof(unsigned char) * buffer_len);

    kernel_genchars<<<1, num_threads>>>(device_buffer, buffer_len);
    cudaMemcpy(host_buffer, device_buffer, sizeof(unsigned char) * buffer_len, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    free(host_buffer);
    cudaFree(device_buffer);
}