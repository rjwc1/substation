#!/bin/bash

echo "FP16"
for i in {0..23}
do
    echo $i
    nvcc -O3 -lcudadevrt -lcuda -arch sm_70 -lcublas cublas-gemm.cpp -DCUBLAS_ALGO=CUBLAS_GEMM_ALGO$i -o gemm$i
done
for i in {0..15}
do
    echo $i
    nvcc -O3 -lcudadevrt -lcuda -arch sm_70 -lcublas cublas-gemm.cpp -DCUBLAS_ALGO=CUBLAS_GEMM_ALGO${i}_TENSOR_OP -o gemmtc$i
done
echo "Default"
nvcc -O3 -lcudadevrt -lcuda -arch sm_70 -lcublas cublas-gemm.cpp -DCUBLAS_ALGO=CUBLAS_GEMM_DEFAULT -o gemm_default
echo "Default TC"
nvcc -O3 -lcudadevrt -lcuda -arch sm_70 -lcublas cublas-gemm.cpp -DCUBLAS_ALGO=CUBLAS_GEMM_DEFAULT_TENSOR_OP -o gemmtc_default
echo "FP32"
for i in {0..23}
do
    echo $i
    nvcc -O3 -lcudadevrt -lcuda -arch sm_70 -lcublas cublas-gemm.cpp -DCUBLAS_ALGO=CUBLAS_GEMM_ALGO$i -DC_TYPE=float -DCUBLAS_C_TYPE=CUDA_R_32F -o gemm32$i
done
echo "Default"
nvcc -O3 -lcudadevrt -lcuda -arch sm_70 -lcublas cublas-gemm.cpp -DCUBLAS_ALGO=CUBLAS_GEMM_DEFAULT -DC_TYPE=float -DCUBLAS_C_TYPE=CUDA_R_32F -o gemm32_default
