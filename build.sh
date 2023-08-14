nvcc -shared -o build/kercppnels.dll ./src/cuda/kernels.cu
nvcc -o ./build/a.exe -L"build/" -lkernels ./src/main.cpp