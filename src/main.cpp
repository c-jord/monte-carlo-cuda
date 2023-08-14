#include <stdio.h>
#include "cuda/kernels.h"

int main(int argc, char const *argv[]){
    monte_carlo_cuda_kernels::run_kernels();
    return 0;
}