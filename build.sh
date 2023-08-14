nvcc -shared -o build/kernels.dll .\\src\\cuda\\kernels.cu
nvcc -o .\\build\\a.exe -L"build/" -lkernels .\\src\\main.cpp
.\\build\\a.exe