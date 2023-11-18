#include <stdio.h>
#include <cuda.h>
#include <curand.h>

#define BLOCK_SIZE 32

__global__ void columnSum(float* array, float* result, int x, int y, int padded_x) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int index = bid * blockDim.x + tid;

    if (index < x) {
        float sum = 0;
        for (int i = 0; i < y; ++i) {
            sum += array[i * padded_x + index];
        }
        result[index] = sum;
    }
}

void randomInit(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = rand() / (float)RAND_MAX;
    }
}

void printArray(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        printf("Element %d: %f\n", i, data[i]);
    }
}

int main() {
    int x = 128; // Number of columns
    int y = 128; // Number of rows

    // Calculate the padded size
    int padded_x = (x + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    float* h_array = (float*)malloc(padded_x * y * sizeof(float));
    float* h_result = (float*)malloc(x * sizeof(float));

    // Initialize array with random values
    randomInit(h_array, padded_x * y);

    float* d_array;
    float* d_result;

    // Allocate device memory
    cudaMalloc((void**)&d_array, padded_x * y * sizeof(float));
    cudaMalloc((void**)&d_result, x * sizeof(float));

    // Copy array from host to device
    cudaMemcpy(d_array, h_array, padded_x * y * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((x + blockDim.x - 1) / blockDim.x);

    // Launch kernel
    columnSum<<<gridDim, blockDim>>>(d_array, d_result, x, y, padded_x);

    // Copy result from device to host
    cudaMemcpy(h_result, d_result, x * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    printArray(h_result, x);

    // Free allocated memory
    free(h_array);
    free(h_result);
    cudaFree(d_array);
    cudaFree(d_result);

    return 0;
}