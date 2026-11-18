/**
 * vectorAdd.cu
 * Suma de vectores C = A + B en CPU vs GPU (CUDA) para N = 1M, 4M y 16M elementos.
 * Mide tiempo de CPU, tiempo exclusivo de kernel GPU y tiempo total GPU (H2D + kernel + D2H).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "Error CUDA en %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel CUDA: cada hilo calcula un elemento de C = A + B
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// Implementacion secuencial de referencia en CPU
void vectorAddCPU(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

static void printDeviceInfo(void) {
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int runtimeVer = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVer));
    printf("Dispositivo GPU: %s (CC %d.%d) | CUDA Runtime: %d.%d\n",
           prop.name, prop.major, prop.minor,
           runtimeVer / 1000, (runtimeVer % 100) / 10);
}

static void runBenchmark(int n) {
    size_t bytes = (size_t)n * sizeof(float);

    // Reserva de memoria en host (prefijo h_)
    float *h_A     = (float *)malloc(bytes);
    float *h_B     = (float *)malloc(bytes);
    float *h_C_cpu = (float *)malloc(bytes);
    float *h_C_gpu = (float *)malloc(bytes);
    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
        fprintf(stderr, "Error al reservar memoria en host\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++) {
        h_A[i] = (float)i * 0.5f;
        h_B[i] = (float)i * 1.5f;
    }

    // --- Benchmark CPU ---
    clock_t t0 = clock();
    vectorAddCPU(h_A, h_B, h_C_cpu, n);
    double cpu_ms = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000.0;

    // --- Reserva de memoria en dispositivo (prefijo d_) ---
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    cudaEvent_t startTotal, stopTotal, startKernel, stopKernel;
    CUDA_CHECK(cudaEventCreate(&startTotal));
    CUDA_CHECK(cudaEventCreate(&stopTotal));
    CUDA_CHECK(cudaEventCreate(&startKernel));
    CUDA_CHECK(cudaEventCreate(&stopKernel));

    int blockSize = 256;
    int gridSize  = (n + blockSize - 1) / blockSize;

    // Medicion total GPU: transferencias H2D + kernel + transferencia D2H
    CUDA_CHECK(cudaEventRecord(startTotal));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(startKernel));
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stopKernel));

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stopTotal));
    CUDA_CHECK(cudaEventSynchronize(stopTotal));

    float kernel_ms = 0.0f, total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, startKernel, stopKernel));
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, startTotal, stopTotal));

    // Verificacion numerica contra CPU (tolerancia < 1e-4)
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (fabsf(h_C_gpu[i] - h_C_cpu[i]) > 1e-4f) {
            errors++;
        }
    }

    printf("N = %8d (%2dM) | CPU: %7.2f ms | GPU kernel: %6.2f ms | GPU total: %6.2f ms | Errores: %d\n",
           n, n >> 20, cpu_ms, kernel_ms, total_ms, errors);

    CUDA_CHECK(cudaEventDestroy(startTotal));
    CUDA_CHECK(cudaEventDestroy(stopTotal));
    CUDA_CHECK(cudaEventDestroy(startKernel));
    CUDA_CHECK(cudaEventDestroy(stopKernel));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
}

int main(void) {
    printDeviceInfo();

    // Warm-up del contexto CUDA antes de medir
    CUDA_CHECK(cudaFree(0));

    const int sizes[] = {1 << 20, 1 << 22, 1 << 24}; // 1M, 4M, 16M
    int numSizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
    for (int i = 0; i < numSizes; i++) {
        runBenchmark(sizes[i]);
    }

    return 0;
}
