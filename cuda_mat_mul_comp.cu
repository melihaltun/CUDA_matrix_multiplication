#include <iostream>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"


// Matrix multiplication without CUDA
void matrixMultiplicationCPU(const float* A, const float* B, float* C, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < size; ++k) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

// Matrix multiplication with CUDA
__global__ void matrixMultiplicationGPU(const float* A, const float* B, float* C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        float sum = 0.0f;
        for (int k = 0; k < size; ++k) {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}

int main() {
    int N = 1000;

    // Allocate memory for matrices on the host (CPU)
    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C_CPU = new float[N * N];
    float* C_GPU = new float[N * N];

    // Initialize matrices A and B with random values
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    for (size_t i = 0; i < N * N; i++) {
        A[i] = distribution(generator);
        B[i] = distribution(generator);
    }

    std::cout << "1000 x 1000 Matrix Multiplication Using CPU vs GPU:" << std::endl;

    // Perform matrix multiplication without CUDA and measure time
    auto start_CPU = std::chrono::high_resolution_clock::now();
    matrixMultiplicationCPU(A, B, C_CPU, N);
    auto end_CPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_CPU = end_CPU - start_CPU;
    std::cout << "Elapsed CPU calc. time: " << duration_CPU.count() << " seconds." << std::endl;

    // Allocate memory for matrices A, B, C on the device (GPU)
    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    // Copy matrices A, B from host to device
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Set grid and block dimensions for CUDA kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    auto start_GPU = std::chrono::high_resolution_clock::now();
    // Launch CUDA kernel
    matrixMultiplicationGPU <<<gridSize, blockSize >>>(d_A, d_B, d_C, N);

    // Wait for kernel to complete
    cudaDeviceSynchronize();

    // Copy result matrix C from device to host
    cudaMemcpy(C_GPU, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    auto end_GPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_GPU = end_GPU - start_GPU;
    std::cout << "Elapsed GPU calc. time: " << duration_GPU.count() << " seconds." << std::endl;
}
