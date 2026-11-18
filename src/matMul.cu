/**
 * matMul.cu
 * Multiplicacion de matrices C = A * B en CUDA: comparativa entre kernel naive
 * (memoria global) y kernel con tiling en shared memory (TILE = 16) para N = 512 y 1024.
 * Verifica la exactitud frente a la referencia en CPU (error < 1e-3).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define TILE 16

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "Error CUDA en %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel naive: cada hilo lee directamente filas de A y columnas de B desde memoria global
__global__ void matMulNaive(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Kernel con tiling en shared memory: reduce accesos a memoria global en un factor TILE
__global__ void matMulTiled(const float *A, const float *B, float *C, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    int numTiles = (N + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; t++) {
        // Cargar tile de A en shared memory
        if (row < N && (t * TILE + threadIdx.x) < N) {
            sA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Cargar tile de B en shared memory
        if (col < N && (t * TILE + threadIdx.y) < N) {
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads(); // Esperar a que ambos tiles esten cargados

        for (int k = 0; k < TILE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads(); // Evitar sobrescribir sA/sB antes de que todos terminen
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Referencia secuencial en CPU
void matMulCPU(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

static int verifyResult(const float *ref, const float *cand, int N, float tol) {
    int errors = 0;
    size_t total = (size_t)N * (size_t)N;
    for (size_t i = 0; i < total; i++) {
        if (fabsf(cand[i] - ref[i]) > tol) {
            errors++;
        }
    }
    return errors;
}

static void runMatMulBenchmark(int N) {
    size_t bytes = (size_t)N * (size_t)N * sizeof(float);

    float *h_A       = (float *)malloc(bytes);
    float *h_B       = (float *)malloc(bytes);
    float *h_C_cpu   = (float *)malloc(bytes);
    float *h_C_naive = (float *)malloc(bytes);
    float *h_C_tiled = (float *)malloc(bytes);
    if (!h_A || !h_B || !h_C_cpu || !h_C_naive || !h_C_tiled) {
        fprintf(stderr, "Error al reservar memoria en host para N=%d\n", N);
        exit(EXIT_FAILURE);
    }

    // Valores acotados para evitar desbordamiento de precision en acumulacion float
    for (int i = 0; i < N * N; i++) {
        h_A[i] = ((i % 100) * 0.01f);
        h_B[i] = (((i * 3) % 100) * 0.01f);
    }

    // --- Referencia CPU ---
    clock_t t0 = clock();
    matMulCPU(h_A, h_B, h_C_cpu, N);
    double cpu_ms = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000.0;

    // --- Memoria en dispositivo ---
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- Kernel Naive ---
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));
    CUDA_CHECK(cudaEventRecord(start));
    matMulNaive<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float naive_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, bytes, cudaMemcpyDeviceToHost));

    // --- Kernel Tiled (Shared Memory) ---
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));
    CUDA_CHECK(cudaEventRecord(start));
    matMulTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float tiled_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&tiled_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, bytes, cudaMemcpyDeviceToHost));

    int errNaive = verifyResult(h_C_cpu, h_C_naive, N, 1e-3f);
    int errTiled = verifyResult(h_C_cpu, h_C_tiled, N, 1e-3f);
    float speedup = naive_ms / tiled_ms;

    printf("N = %4d | CPU: %8.2f ms | Naive: %6.2f ms (err=%d) | Tiled: %6.2f ms (err=%d) | Speedup Tiled/Naive: %.2fx\n",
           N, cpu_ms, naive_ms, errNaive, tiled_ms, errTiled, speedup);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_naive);
    free(h_C_tiled);
}

int main(void) {
    // Warm-up del contexto CUDA
    CUDA_CHECK(cudaFree(0));

    runMatMulBenchmark(512);
    runMatMulBenchmark(1024);

    return 0;
}
