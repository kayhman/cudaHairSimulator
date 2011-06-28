set(CUDA_NVCC_FLAGS --ptxas-options=-v -arch=sm_13)
cuda_add_library(hairLib hairLib.cu hairLib.h)
