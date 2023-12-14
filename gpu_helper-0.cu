
            #include "blocks.cuh"
            #include <cuda.h>
            #include <stdio.h>

            extern "C" {
                void* gpu_allocate(size_t size) {
                    void* ptr = nullptr;
                    CHECK(cudaMalloc(&ptr, size));
                    CHECK(cudaMemset(ptr, 0, size));
                    return ptr;
                }

                void gpu_free(void* ptr) {
                    CHECK(cudaFree(ptr));
                }

                void host_to_gpu(void* gpu, void* host, size_t size) {
                    CHECK(cudaMemcpy(gpu, host, size, cudaMemcpyHostToDevice));
                }

                void gpu_to_host(void* host, void* gpu, size_t size) {
                    CHECK(cudaMemcpy(host, gpu, size, cudaMemcpyDeviceToHost));
                }

                void device_synchronize() {
                    CHECK(cudaDeviceSynchronize());
                }

                int fast_allclose(float* a, float* b, size_t size, float atol, float rtol) {
                    for (size_t i = 0; i < size; ++i) {
                        if (fabs(a[i] - b[i]) > atol + rtol*fabs(b[i])) {
                            return 0;
                        }
                    }
                    return 1;
                }
            }
        