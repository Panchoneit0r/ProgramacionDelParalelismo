# Ejercicio Memoria compartida
Código en CUDA C que cree una matriz de X * Y de valores aleatorios les haga un padding del tamaño a elegir y que realize de manera paralela la suma de los valores de cada columna.

## Codigo Completo
```c
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
```

## Explicación del código
**column_sum.cu**
El archivo column_sum.cu contiene el columnSum del kernel CUDA que calcula la suma de cada columna de la matriz.
```c
__global__ void columnSum(int *a, int *b, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        atomicAdd(&b[col], a[row * n + col]);
    }
}
```
El núcleo toma tres parámetros: a (la matriz 2D de entrada), b (la matriz de salida de sumas de columnas) y n (el número de columnas en la matriz 2D).
Cada hilo calcula sus índices de fila y columna en función de los índices de bloque y hilo.
Si los índices de fila y columna están dentro de los límites de la matriz, el subproceso realiza una suma atómica del elemento correspondiente en la matriz de entrada al elemento correspondiente en la matriz de salida.
La suma atómica es necesaria para garantizar el cálculo correcto de las sumas de las columnas en caso de que varios subprocesos escriban en el mismo elemento de la matriz de salida simultáneamente.

**main.cpp**
El archivo main.cpp contiene el código de host que inicializa la matriz 2D, asigna memoria del dispositivo, copia la matriz del host al dispositivo, inicia el kernel CUDA, copia los resultados del dispositivo al host e imprime los resultados en el consola.
```c
// Initialize the 2D array with random values
srand(time(NULL));
for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
        a[i * n + j] = rand() % 100;
    }
}

// Allocate device memory and copy the array from the host to the device
int *d_a, *d_b;
cudaMalloc((void **)&d_a, n * n * sizeof(int));
cudaMalloc((void **)&d_b, n * sizeof(int));
cudaMemcpy(d_a, a, n * n * sizeof(int), cudaMemcpyHostToDevice);

// Launch the CUDA kernel
dim3 blockSize(16, 16);
dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);
columnSum<<<gridSize, blockSize>>>(d_a, d_b, n);

// Copy the results from the device to the host and print them to the console
cudaMemcpy(b, d_b, n * sizeof(int), cudaMemcpyDeviceToHost);

for (int i = 0; i < n; i++) {
    printf("Column %d: %d\n", i, b[i]);
}
```
El código inicializa la matriz 2D con valores aleatorios. Luego asigna memoria del dispositivo y copia la matriz del host al dispositivo.
El código calcula las dimensiones de los bloques de hilos y la cuadrícula en función de las dimensiones de la matriz 2D. Lanza el kernel CUDA con estas dimensiones.
Una vez que el kernel termina de ejecutarse, el código copia los resultados del dispositivo al host y los imprime en la consola.
Finalmente, el código libera la memoria del dispositivo antes de salir.

## Aclaraciones
Codigo hecho en pareja por Juan Carlos Guerrero Murillo Y Gilberto Guzman Villareal, con asesoria de Octavio Diaz 
