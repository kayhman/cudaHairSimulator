set(CUDA_NVCC_FLAGS --ptxas-options=-v -arch=sm_13 )
# set(CUDA_NVCC_FLAGS -maxregcount=16 )
cuda_add_library(hairLib hairLib.cu hairLib.h)
