#include <stdio.h>
#include "cuda/kernels.h"

int main(int argc, char const *argv[]){
    double result = monte_carlo_cuda_kernels::run_kernels_pi(1000000000, 1024);
    printf("Result: %lf\n", result);
    return 0;
}