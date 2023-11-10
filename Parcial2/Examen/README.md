# simpleMathAoS
~~~
==1011== NVPROF is profiling process 1011, command: ./simpleMathAoS
==1011== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==1011== Profiling application: ./simpleMathAoS
==1011== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   80.19%  23.304ms         2  11.652ms  8.6804ms  14.623ms  [CUDA memcpy DtoH]
                   18.05%  5.2457ms         1  5.2457ms  5.2457ms  5.2457ms  [CUDA memcpy HtoD]
                    0.88%  256.10us         1  256.10us  256.10us  256.10us  warmup(innerStruct*, innerStruct*, int)
                    0.88%  255.85us         1  255.85us  255.85us  255.85us  testInnerStruct(innerStruct*, innerStruct*, int)
      API calls:   88.83%  566.38ms         2  283.19ms  376.90us  566.01ms  cudaMalloc
                    5.61%  35.742ms         1  35.742ms  35.742ms  35.742ms  cudaDeviceReset
                    4.90%  31.267ms         3  10.422ms  6.6946ms  15.576ms  cudaMemcpy
                    0.34%  2.1562ms         1  2.1562ms  2.1562ms  2.1562ms  cuDeviceGetPCIBusId
                    0.19%  1.2200ms         2  610.00us  446.90us  773.10us  cudaFree
                    0.11%  669.90us         2  334.95us  334.90us  335.00us  cudaDeviceSynchronize
                    0.03%  161.60us         2  80.800us  63.500us  98.100us  cudaLaunchKernel
                    0.00%  14.600us       101     144ns     100ns  1.3000us  cuDeviceGetAttribute
                    0.00%  6.1000us         1  6.1000us  6.1000us  6.1000us  cudaSetDevice
                    0.00%  5.8000us         2  2.9000us  2.6000us  3.2000us  cudaGetLastError
                    0.00%  4.7000us         1  4.7000us  4.7000us  4.7000us  cudaGetDeviceProperties
                    0.00%  1.4000us         3     466ns     200ns  1.0000us  cuDeviceGetCount
                    0.00%     800ns         2     400ns     100ns     700ns  cuDeviceGet
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
~~~
**[CUDA memcpy DtoH]:** Copia de memoria de Device (Dispositivo, es decir, la GPU) a Host (el sistema principal, es decir, la CPU). Esta actividad tomó el 80.19% del tiempo total de las actividades de la GPU.
**[CUDA memcpy HtoD]:** Copia de memoria de Host a Device. Esta actividad tomó el 18.05% del tiempo total de las actividades de la GPU.
**warmup y testInnerStruct:** Son funciones (kernels) que se ejecutan en la GPU. Cada una de estas funciones tomó aproximadamente el 0.88% del tiempo total de las actividades de la GPU.
**cudaMalloc:** Función para asignar memoria en la GPU. Esta función tomó el 88.83% del tiempo total de las llamadas a la API.
**cudaDeviceReset:** Función para reiniciar el dispositivo (la GPU). Esta función tomó el 5.61% del tiempo total de las llamadas a la API.
**cudaMemcpy:** Función para copiar datos entre el host y el device. Esta función tomó el 4.90% del tiempo total de las llamadas a la API.
**cuDeviceGetPCIBusId:** Esta función se utiliza para obtener el identificador del bus PCI del dispositivo. Esta función tomó el 0.34% del tiempo total de las llamadas a la API.
**cudaFree:** Esta función libera la memoria que previamente fue asignada con cudaMalloc. Esta función tomó el 0.19% del tiempo total de las llamadas a la API.
**cudaDeviceSynchronize:** Esta función bloquea la CPU hasta que todas las tareas previamente emitidas en el dispositivo hayan completado. Esta función tomó el 0.11% del tiempo total de las llamadas a la API.
**cudaLaunchKernel:** Esta función lanza un kernel CUDA en el dispositivo. Esta función tomó el 0.03% del tiempo total de las llamadas a la API.

Este código en CUDA ilustra cómo las lecturas desalineadas afectan el rendimiento al provocar dichas lecturas en un float*. También incluye núcleos que minimizan este impacto al desenrollar bucles. Aquí se presenta una explicación detallada:

1. Se configura el dispositivo CUDA con cudaSetDevice(dev).
2. Se define el tamaño de la memoria asignada en el host y en el dispositivo, determinado por nElem (configurado en LEN).
3. Se obtienen e imprimen las propiedades del dispositivo CUDA mediante cudaGetDeviceProperties().
4. Se asigna memoria en el host (CPU) usando malloc().
5. La memoria del host se inicializa con valores aleatorios mediante inicialInnerStruct().
6. Se realiza una operación de suma en el host con sumArraysOnHost().
7. Se asigna memoria en el dispositivo (GPU) mediante cudaMalloc().
8. Los datos se copian del host al dispositivo con cudaMemcpy().
9. Se ejecutan cuatro núcleos: warmup(), readOffset(), readOffsetUnroll2() y readOffsetUnroll4(). El kernel warmup() calienta la GPU, readOffset() mide el impacto de las lecturas desalineadas en el rendimiento, y los kernels readOffsetUnroll2() y readOffsetUnroll4() utilizan desenrollado de bucles para reducir dicho impacto.
10. Los resultados se copian del dispositivo al host con cudaMemcpy().
11. Se verifica si los resultados coinciden entre el dispositivo y el host mediante checkInnerStruct().
12. Finalmente, se libera la memoria asignada en el host y el dispositivo con free() y cudaFree(), y se restablece el dispositivo con cudaDeviceReset().

El parámetro offset, que puede pasarse como argumento de línea de comandos, fuerza lecturas desalineadas. Estas ocurren cuando la dirección inicial de los datos leídos no es divisible por el tamaño del tipo de datos, lo que puede degradar el rendimiento al requerir transacciones de memoria adicionales. El desenrollado de bucles es una técnica que mitiga este impacto al permitir que la GPU fusione múltiples lecturas desalineadas en una sola lectura alineada.

# simpleMathSoA
~~~
==1027== NVPROF is profiling process 1027, command: ./simpleMathSoA
==1027== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==1027== Profiling application: ./simpleMathSoA
==1027== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   73.35%  12.215ms         2  6.1076ms  3.7599ms  8.4554ms  [CUDA memcpy DtoH]
                   23.58%  3.9265ms         1  3.9265ms  3.9265ms  3.9265ms  [CUDA memcpy HtoD]
                    1.54%  256.42us         1  256.42us  256.42us  256.42us  warmup2(InnerArray*, InnerArray*, int)
                    1.54%  256.03us         1  256.03us  256.03us  256.03us  testInnerArray(InnerArray*, InnerArray*, int)
      API calls:   90.98%  584.89ms         2  292.45ms  380.00us  584.51ms  cudaMalloc
                    5.47%  35.165ms         1  35.165ms  35.165ms  35.165ms  cudaDeviceReset
                    2.89%  18.564ms         3  6.1881ms  3.9129ms  9.2690ms  cudaMemcpy
                    0.39%  2.4897ms         1  2.4897ms  2.4897ms  2.4897ms  cuDeviceGetPCIBusId
                    0.15%  981.80us         2  490.90us  359.80us  622.00us  cudaFree
                    0.11%  682.20us         2  341.10us  302.90us  379.30us  cudaDeviceSynchronize
                    0.01%  94.200us         2  47.100us  43.700us  50.500us  cudaLaunchKernel
                    0.00%  16.500us       101     163ns     100ns  1.4000us  cuDeviceGetAttribute
                    0.00%  5.9000us         1  5.9000us  5.9000us  5.9000us  cudaSetDevice
                    0.00%  5.1000us         1  5.1000us  5.1000us  5.1000us  cudaGetDeviceProperties
                    0.00%  4.7000us         2  2.3500us  2.3000us  2.4000us  cudaGetLastError
                    0.00%  1.4000us         3     466ns     200ns  1.0000us  cuDeviceGetCount
                    0.00%  1.2000us         1  1.2000us  1.2000us  1.2000us  cuDeviceGetName
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
~~~

**[CUDA memcpy DtoH]:** Copia de memoria de Device (Dispositivo, es decir, la GPU) a Host (el sistema principal, es decir, la CPU). Esta actividad tomó el 73.35% del tiempo total de las actividades de la GPU.
**[CUDA memcpy HtoD]:** Copia de memoria de Host a Device. Esta actividad tomó el 23.58% del tiempo total de las actividades de la GPU.
**warmup y testInnerStruct:** Son funciones (kernels) que se ejecutan en la GPU. Cada una de estas funciones tomó aproximadamente el 1.54% del tiempo total de las actividades de la GPU.
**cudaMalloc:** Función para asignar memoria en la GPU. Esta función tomó el 90.98% del tiempo total de las llamadas a la API.
**cudaDeviceReset:** Función para reiniciar el dispositivo (la GPU). Esta función tomó el 5.47% del tiempo total de las llamadas a la API.
**cudaMemcpy:** Función para copiar datos entre el host y el device. Esta función tomó el 2.89% del tiempo total de las llamadas a la API.

Este código en CUDA ilustra cómo emplear estructuras en la programación de GPU. Aquí se presenta una descripción detallada:

1. Se configura el dispositivo CUDA con cudaSetDevice(dev).
2. Se define el tamaño de la memoria que se asignará tanto en el host como en el dispositivo, utilizando nElem que se establece en LEN.
3. Se obtienen e imprimen las propiedades del dispositivo CUDA mediante cudaGetDeviceProperties().
4. Se asigna memoria en el host (CPU) mediante malloc().
5. La memoria del host se inicializa con valores aleatorios utilizando inicialInnerArray().
6. Se realiza una operación de suma en el host mediante testInnerArrayHost().
7. Se asigna memoria en el dispositivo (GPU) con cudaMalloc().
8. Los datos se copian del host al dispositivo mediante cudaMemcpy().
9. Se ejecutan dos núcleos: warmup2() y testInnerArray(). El kernel warmup2() calienta la GPU, y el kernel testInnerArray() mide el impacto en el rendimiento de las lecturas desalineadas al realizar la misma operación de suma en el dispositivo.
10. Los resultados se copian desde el dispositivo al host mediante cudaMemcpy().
11. Se verifica si los resultados obtenidos del dispositivo coinciden con los resultados obtenidos del host utilizando checkInnerArray().

# sumArrayZerocpy
~~~
==1049== NVPROF is profiling process 1049, command: ./sumArrayZerocpy
==1049== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==1049== Profiling application: ./sumArrayZerocpy
==1049== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   33.33%  3.5200us         1  3.5200us  3.5200us  3.5200us  sumArraysZeroCopy(float*, float*, float*, int)
                   22.73%  2.4000us         2  1.2000us  1.1840us  1.2160us  [CUDA memcpy DtoH]
                   22.12%  2.3360us         1  2.3360us  2.3360us  2.3360us  sumArrays(float*, float*, float*, int)
                   21.82%  2.3040us         2  1.1520us     864ns  1.4400us  [CUDA memcpy HtoD]
      API calls:   94.24%  583.14ms         3  194.38ms  1.8000us  583.14ms  cudaMalloc
                    5.09%  31.475ms         1  31.475ms  31.475ms  31.475ms  cudaDeviceReset
                    0.35%  2.1756ms         1  2.1756ms  2.1756ms  2.1756ms  cuDeviceGetPCIBusId
                    0.16%  988.60us         2  494.30us  3.8000us  984.80us  cudaHostAlloc
                    0.06%  368.90us         2  184.45us  4.5000us  364.40us  cudaFreeHost
                    0.06%  358.00us         4  89.500us  33.100us  129.40us  cudaMemcpy
                    0.04%  218.20us         3  72.733us  2.5000us  208.10us  cudaFree
                    0.01%  60.300us         2  30.150us  28.600us  31.700us  cudaLaunchKernel
                    0.00%  14.900us       101     147ns     100ns  1.0000us  cuDeviceGetAttribute
                    0.00%  6.2000us         1  6.2000us  6.2000us  6.2000us  cudaSetDevice
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cudaGetDeviceProperties
                    0.00%  2.1000us         2  1.0500us     600ns  1.5000us  cudaHostGetDevicePointer
                    0.00%  1.6000us         3     533ns     200ns  1.0000us  cuDeviceGetCount
                    0.00%  1.1000us         2     550ns     200ns     900ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
~~~
**sumArraysZeroCopy:** Esta es una función (kernel) que se ejecuta en la GPU, suma los arreglos ayuda de memoria cero copia. Esta actividad tomó el 33.33% del tiempo total de las actividades de la GPU.
**[CUDA memcpy DtoH]:** Copia de memoria de Device (Dispositivo, es decir, la GPU) a Host (el sistema principal, es decir, la CPU). Esta actividad tomó el 22.73% del tiempo total de las actividades de la GPU.
**sumArrays:** Esta es otra función (kernel) que se ejecuta en la GPU, suma los arreglos del host. Esta actividad tomó el 22.12% del tiempo total de las actividades de la GPU.
**[CUDA memcpy HtoD]:** Copia de memoria de Host a Device. Esta actividad tomó el 21.82% del tiempo total de las actividades de la GPU.
**cudaHostAlloc:** Esta función se utiliza para asignar memoria en el host que será accesible desde el dispositivo. Esta función tomó el 0.16% del tiempo total de las llamadas a la API.
**cudaFreeHost:** Esta función libera la memoria que previamente fue asignada con cudaHostAlloc. Esta función tomó el 0.06% del tiempo total de las llamadas a la API.


Este código en CUDA demuestra la utilización de memoria de copia cero en la programación de GPU. Aquí se presenta una explicación detallada:

1. Se configura el dispositivo CUDA con cudaSetDevice(dev).
2. Se define el tamaño de la memoria que se asignará tanto en el host como en el dispositivo, utilizando nElem que se establece en LEN.
3. Se obtienen e imprimen las propiedades del dispositivo CUDA mediante cudaGetDeviceProperties().
4. Se asigna memoria en el host (CPU) mediante malloc().
5. La memoria del host se inicializa con valores aleatorios mediante initialData().
6. Se realiza una operación de suma en el host mediante sumArraysOnHost().
7. Se asigna memoria en el dispositivo (GPU) con cudaMalloc().
8. Los datos se copian del host al dispositivo mediante cudaMemcpy().
9. Se ejecutan dos núcleos: warmup() y sumArrays(). Ambos realizan la misma operación de suma en el dispositivo.
10. Los resultados se copian desde el dispositivo al host mediante cudaMemcpy().
11. Se verifica si los resultados obtenidos del dispositivo coinciden con los resultados obtenidos del host mediante checkResult().
12. Se libera la memoria asignada tanto en el host como en el dispositivo utilizando free() y cudaFree().
13. Se asigna memoria de copia cero en el host utilizando cudaHostAlloc().
14. La memoria del host se inicializa con valores aleatorios mediante initialData().
15. Se obtienen punteros de dispositivo a la memoria del host mediante cudaHostGetDevicePointer().
16. Se ejecuta el kernel sumArraysZeroCopy(), que realiza la operación de suma en el dispositivo utilizando la memoria de copia cero.

~~~
==1049== NVPROF is profiling process 1049, command: ./sumArrayZerocpy
==1049== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==1049== Profiling application: ./sumArrayZerocpy
==1049== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   33.33%  3.5200us         1  3.5200us  3.5200us  3.5200us  sumArraysZeroCopy(float*, float*, float*, int)
                   22.73%  2.4000us         2  1.2000us  1.1840us  1.2160us  [CUDA memcpy DtoH]
                   22.12%  2.3360us         1  2.3360us  2.3360us  2.3360us  sumArrays(float*, float*, float*, int)
                   21.82%  2.3040us         2  1.1520us     864ns  1.4400us  [CUDA memcpy HtoD]
      API calls:   94.24%  583.14ms         3  194.38ms  1.8000us  583.14ms  cudaMalloc
                    5.09%  31.475ms         1  31.475ms  31.475ms  31.475ms  cudaDeviceReset
                    0.35%  2.1756ms         1  2.1756ms  2.1756ms  2.1756ms  cuDeviceGetPCIBusId
                    0.16%  988.60us         2  494.30us  3.8000us  984.80us  cudaHostAlloc
                    0.06%  368.90us         2  184.45us  4.5000us  364.40us  cudaFreeHost
                    0.06%  358.00us         4  89.500us  33.100us  129.40us  cudaMemcpy
                    0.04%  218.20us         3  72.733us  2.5000us  208.10us  cudaFree
                    0.01%  60.300us         2  30.150us  28.600us  31.700us  cudaLaunchKernel
                    0.00%  14.900us       101     147ns     100ns  1.0000us  cuDeviceGetAttribute
                    0.00%  6.2000us         1  6.2000us  6.2000us  6.2000us  cudaSetDevice
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cudaGetDeviceProperties
                    0.00%  2.1000us         2  1.0500us     600ns  1.5000us  cudaHostGetDevicePointer
                    0.00%  1.6000us         3     533ns     200ns  1.0000us  cuDeviceGetCount
                    0.00%  1.1000us         2     550ns     200ns     900ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
~~~
# sumMatrixGPUManaged
~~~
==1071== NVPROF is profiling process 1071, command: ./sumMatrixGPUManaged
==1071== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==1071== Profiling application: ./sumMatrixGPUManaged
==1071== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  12.948ms         2  6.4741ms  288.67us  12.660ms  sumMatrixGPU(float*, float*, float*, int, int)
      API calls:   91.39%  815.38ms         4  203.85ms  27.532ms  731.17ms  cudaMallocManaged
                    3.45%  30.801ms         1  30.801ms  30.801ms  30.801ms  cudaDeviceReset
                    3.31%  29.569ms         4  7.3922ms  7.2484ms  7.4490ms  cudaFree
                    1.52%  13.583ms         1  13.583ms  13.583ms  13.583ms  cudaDeviceSynchronize
                    0.24%  2.1681ms         1  2.1681ms  2.1681ms  2.1681ms  cuDeviceGetPCIBusId
                    0.07%  644.20us         2  322.10us  11.200us  633.00us  cudaLaunchKernel
                    0.00%  14.100us       101     139ns     100ns     900ns  cuDeviceGetAttribute
                    0.00%  5.8000us         1  5.8000us  5.8000us  5.8000us  cudaSetDevice
                    0.00%  4.0000us         1  4.0000us  4.0000us  4.0000us  cudaGetDeviceProperties
                    0.00%  1.2000us         3     400ns     100ns     900ns  cuDeviceGetCount
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cudaGetLastError
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
~~~
**sumMatrixGPU:** Esta es una función (kernel) que se ejecuta en la GPU, como su nombre lo dice se encarga de hacer la suma de matrices. Esta actividad tomó el 100% del tiempo total de las actividades de la GPU debido a que es la unica funcion que se utiliza siendo llamda 2 veces.
**cudaMallocManaged:** Esta función se utiliza para asignar memoria en la GPU y el host de manera que ambos pueden acceder a ella. Esta función tomó el 91.39% del tiempo total de las llamadas a la API.

Este código en CUDA demuestra el uso de la memoria unificada en la programación de GPU. A continuación, se presenta una descripción detallada:

1. Se configura el dispositivo CUDA con cudaSetDevice(dev).
2. Se define el tamaño de la memoria que se asignará tanto en el host como en el dispositivo, utilizando nElem que se establece en LEN.
3. Se obtienen e imprimen las propiedades del dispositivo CUDA mediante cudaGetDeviceProperties().
4. Se asigna memoria unificada en el host mediante cudaMallocManaged().
5. La memoria del host se inicializa con valores aleatorios mediante initialData().
6. Se realiza una operación de suma en el host mediante sumMatrixOnHost().
7. Se ejecutan dos núcleos: warmup() y sumMatrixGPU(). El kernel warmup() calienta la GPU, y el kernel sumMatrixGPU() realiza la misma operación de suma en el dispositivo para medir el impacto en el rendimiento de las lecturas desalineadas.
8. Se verifica si los resultados obtenidos del dispositivo coinciden con los resultados obtenidos del host mediante checkResult().
9. Se libera la memoria asignada tanto en el host como en el dispositivo utilizando cudaFree(), y se restablece el dispositivo mediante cudaDeviceReset().

# sumMatrixGPUManual
~~~
==1089== NVPROF is profiling process 1089, command: ./sumMatrixGPUManual
==1089== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==1089== Profiling application: ./sumMatrixGPUManual
==1089== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   65.52%  27.101ms         2  13.550ms  8.3698ms  18.731ms  [CUDA memcpy HtoD]
                   30.63%  12.669ms         1  12.669ms  12.669ms  12.669ms  [CUDA memcpy DtoH]
                    2.69%  1.1118ms         2  555.89us  288.73us  823.04us  sumMatrixGPU(float*, float*, float*, int, int)
                    1.16%  479.42us         2  239.71us  238.91us  240.51us  [CUDA memset]
      API calls:   87.57%  607.17ms         3  202.39ms  713.10us  605.72ms  cudaMalloc
                    6.50%  45.038ms         3  15.013ms  8.6183ms  23.545ms  cudaMemcpy
                    5.26%  36.474ms         1  36.474ms  36.474ms  36.474ms  cudaDeviceReset
                    0.33%  2.2576ms         1  2.2576ms  2.2576ms  2.2576ms  cuDeviceGetPCIBusId
                    0.19%  1.3256ms         3  441.87us  223.90us  799.30us  cudaFree
                    0.13%  929.30us         1  929.30us  929.30us  929.30us  cudaDeviceSynchronize
                    0.01%  62.700us         2  31.350us  24.300us  38.400us  cudaMemset
                    0.01%  62.500us         2  31.250us  28.200us  34.300us  cudaLaunchKernel
                    0.00%  15.600us       101     154ns     100ns  1.0000us  cuDeviceGetAttribute
                    0.00%  7.3000us         1  7.3000us  7.3000us  7.3000us  cudaSetDevice
                    0.00%  7.1000us         1  7.1000us  7.1000us  7.1000us  cudaGetDeviceProperties
                    0.00%  1.3000us         3     433ns     200ns     900ns  cuDeviceGetCount
                    0.00%  1.1000us         1  1.1000us  1.1000us  1.1000us  cuDeviceGetName
                    0.00%  1.0000us         2     500ns     200ns     800ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cudaGetLastError
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
~~~
**[CUDA memcpy HtoD]:** Copia de memoria de Host (CPU) a Device (GPU). Esta actividad tomó el 65.52% del tiempo total de las actividades de la GPU.
**[CUDA memcpy DtoH]:** Copia de memoria de Device (GPU) a Host (CPU). Esta actividad tomó el 30.63% del tiempo total de las actividades de la GPU.
**sumMatrixGPU:** Esta es una función (kernel) que se ejecuta en la GPU, en encarga de hacer la suma de matrices. Esta actividad tomó el 2.69% del tiempo total de las actividades de la GPU.
**[CUDA memset]:** Esta función se utiliza para establecer la memoria del dispositivo a un valor específico. Esta función tomó el 1.16% del tiempo total de las actividades de la GPU.
Las demas son apis explicadas anteoriomente 

Este programa CUDA utiliza la paralelización en GPU para sumar matrices de forma eficiente. Configura el dispositivo CUDA y asigna memoria en el host y dispositivo. Inicializa los datos en el host y realiza la suma en CPU. Copia los datos a la GPU, lanza un kernel que ejecuta la suma en paralelo en la GPU utilizando una cuadrícula de bloques 2D. Copia los resultados a CPU y verifica que coincidan. Al final, libera la memoria y restablece el dispositivo. La suma en paralelo en GPU con cuadrículas y bloques 2D permite un cálculo de matrices mucho más rápido que en CPU.

# transpose
~~~
==1111== NVPROF is profiling process 1111, command: ./transpose
==1111== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==1111== Profiling application: ./transpose
==1111== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   86.82%  1.9853ms         1  1.9853ms  1.9853ms  1.9853ms  [CUDA memcpy HtoD]
                    6.62%  151.49us         1  151.49us  151.49us  151.49us  copyRow(float*, float*, int, int)
                    6.56%  150.02us         1  150.02us  150.02us  150.02us  warmup(float*, float*, int, int)
      API calls:   86.44%  634.10ms         2  317.05ms  434.00us  633.66ms  cudaMalloc
                   12.79%  93.791ms         1  93.791ms  93.791ms  93.791ms  cudaDeviceReset
                    0.32%  2.3634ms         1  2.3634ms  2.3634ms  2.3634ms  cudaMemcpy
                    0.31%  2.2569ms         1  2.2569ms  2.2569ms  2.2569ms  cuDeviceGetPCIBusId
                    0.07%  549.80us         2  274.90us  222.60us  327.20us  cudaFree
                    0.06%  404.50us         2  202.25us  166.80us  237.70us  cudaDeviceSynchronize
                    0.01%  57.000us         2  28.500us  15.400us  41.600us  cudaLaunchKernel
                    0.00%  16.500us       101     163ns     100ns  1.2000us  cuDeviceGetAttribute
                    0.00%  5.4000us         1  5.4000us  5.4000us  5.4000us  cudaSetDevice
                    0.00%  5.0000us         1  5.0000us  5.0000us  5.0000us  cudaGetDeviceProperties
                    0.00%  1.4000us         3     466ns     100ns  1.1000us  cuDeviceGetCount
                    0.00%  1.3000us         2     650ns     600ns     700ns  cudaGetLastError
                    0.00%  1.0000us         2     500ns     200ns     800ns  cuDeviceGet
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
~~~
**[CUDA memcpy HtoD]:** Copia de memoria de Host (CPU) a Device (GPU). Esta actividad tomó el 86.82% del tiempo total de las actividades de la GPU.
**copyRow:** Esta es una función (kernel) que se ejecuta en la GPU, copia los datos de una fila en una matriz. Esta actividad tomó el 6.62% del tiempo total de las actividades de la GPU.
**warmup:** Esta es otra función (kernel) que se ejecuta en la GPU, eta funcion no la entendi. Esta actividad tomó el 6.56% del tiempo total de las actividades de la GPU.
**cudaMalloc:** Función para asignar memoria en la GPU. Esta función tomó el 86.44% del tiempo total de las llamadas a la API.

# writeSegment
~~~
==1127== NVPROF is profiling process 1127, command: ./writeSegment
==1127== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==1127== Profiling application: ./writeSegment
==1127== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   65.98%  2.1129ms         3  704.29us  518.98us  921.45us  [CUDA memcpy DtoH]
                   29.36%  940.23us         2  470.12us  465.19us  475.04us  [CUDA memcpy HtoD]
                    1.55%  49.504us         1  49.504us  49.504us  49.504us  writeOffset(float*, float*, float*, int, int)
                    1.49%  47.712us         1  47.712us  47.712us  47.712us  warmup(float*, float*, float*, int, int)
                    0.91%  29.120us         1  29.120us  29.120us  29.120us  writeOffsetUnroll2(float*, float*, float*, int, int)
                    0.72%  23.072us         1  23.072us  23.072us  23.072us  writeOffsetUnroll4(float*, float*, float*, int, int)
      API calls:   92.61%  579.23ms         3  193.08ms  301.40us  578.59ms  cudaMalloc
                    6.01%  37.576ms         1  37.576ms  37.576ms  37.576ms  cudaDeviceReset
                    0.83%  5.1802ms         5  1.0360ms  537.40us  2.0100ms  cudaMemcpy
                    0.34%  2.1550ms         1  2.1550ms  2.1550ms  2.1550ms  cuDeviceGetPCIBusId
                    0.11%  687.50us         3  229.17us  186.80us  276.10us  cudaFree
                    0.06%  399.00us         4  99.750us  72.600us  145.30us  cudaDeviceSynchronize
                    0.04%  225.60us         4  56.400us  20.100us  89.100us  cudaLaunchKernel
                    0.00%  14.400us       101     142ns     100ns  1.0000us  cuDeviceGetAttribute
                    0.00%  4.9000us         1  4.9000us  4.9000us  4.9000us  cudaGetDeviceProperties
                    0.00%  4.1000us         1  4.1000us  4.1000us  4.1000us  cudaSetDevice
                    0.00%  3.3000us         4     825ns     400ns  1.1000us  cudaGetLastError
                    0.00%  1.2000us         3     400ns     200ns     800ns  cuDeviceGetCount
                    0.00%  1.1000us         2     550ns     200ns     900ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceTotalMem
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid
~~~

**[CUDA memcpy DtoH]:** Copia de memoria de Device (GPU) a Host (CPU). Esta actividad tomó el 65.98% del tiempo total de las actividades de la GPU.
**[CUDA memcpy HtoD]:** Copia de memoria de Host (CPU) a Device (GPU). Esta actividad tomó el 29.36% del tiempo total de las actividades de la GPU.
**writeOffset:** Esta es una función (kernel) que se ejecuta en la GPU. Esta actividad tomó el 1.55% del tiempo total de las actividades de la GPU.
**warmup:** Esta es otra función (kernel) que se ejecuta en la GPU. Esta actividad tomó el 1.49% del tiempo total de las actividades de la GPU.
**writeOffsetUnroll2** y **writeOffsetUnroll4**: Estas son funciones (kernels) que se ejecutan en la GPU. Estas actividades tomaron el 0.91% y el 0.72% del tiempo total de las actividades de la GPU, respectivamente.
**cudaMalloc:** Función para asignar memoria en la GPU. Esta función tomó el 92.61% del tiempo total de las llamadas a la API.

Este programa CUDA muestra cómo utilizar el desplazamiento de memoria y el desenrollado de bucles para optimizar el rendimiento en GPU. 

1. Configura el dispositivo CUDA, asigna memoria en host y device, inicializa datos en host y realiza la suma en CPU. 
2. Copia datos a GPU, ejecuta kernels que realizan la suma en paralelo en GPU utilizando desplazamiento de memoria para acceder a datos con eficiencia.
3. Usa diferentes grados de desenrollado de bucles en los kernels para incrementar el paralelismo y mejorar rendimiento. 
4. Compara resultados de CPU y GPU. Al final, libera memoria y restablece dispositivo.
5. El desplazamiento de memoria y desenrollado de bucles permiten un mejor aprovechamiento de la arquitectura paralela de la GPU para acelerar cálculos como suma de matrices.

# memTransfer
~~~
==935== NVPROF is profiling process 935, command: ./memTransfer
==935== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==935== Profiling application: ./memTransfer
==935== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   52.15%  2.1117ms         1  2.1117ms  2.1117ms  2.1117ms  [CUDA memcpy HtoD]
                   47.85%  1.9374ms         1  1.9374ms  1.9374ms  1.9374ms  [CUDA memcpy DtoH]
      API calls:   93.74%  577.35ms         1  577.35ms  577.35ms  577.35ms  cudaMalloc
                    5.15%  31.729ms         1  31.729ms  31.729ms  31.729ms  cudaDeviceReset
                    0.71%  4.3856ms         2  2.1928ms  2.1784ms  2.2072ms  cudaMemcpy
                    0.34%  2.0994ms         1  2.0994ms  2.0994ms  2.0994ms  cuDeviceGetPCIBusId
                    0.05%  306.30us         1  306.30us  306.30us  306.30us  cudaFree
                    0.00%  14.700us       101     145ns     100ns  1.0000us  cuDeviceGetAttribute
                    0.00%  8.0000us         1  8.0000us  8.0000us  8.0000us  cudaSetDevice
                    0.00%  2.8000us         1  2.8000us  2.8000us  2.8000us  cudaGetDeviceProperties
                    0.00%  1.4000us         3     466ns     200ns  1.0000us  cuDeviceGetCount
                    0.00%     900ns         2     450ns     100ns     800ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
~~~
**[CUDA memcpy DtoH]:** Copia de memoria de Device (GPU) a Host (CPU). Esta actividad tomó el 52.15% del tiempo total de las actividades de la GPU.
**[CUDA memcpy HtoD]:** Copia de memoria de Host (CPU) a Device (GPU). Esta actividad tomó el 47.85% del tiempo total de las actividades de la GPU.
**cudaMalloc:** Función para asignar memoria en la GPU. Esta función tomó el 93.74% del tiempo total de las llamadas a la API.


# pinMemTransfer
~~~
==947== NVPROF is profiling process 947, command: ./pinMemTransfer
==947== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==947== Profiling application: ./pinMemTransfer
==947== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   50.57%  1.3036ms         1  1.3036ms  1.3036ms  1.3036ms  [CUDA memcpy HtoD]
                   49.43%  1.2743ms         1  1.2743ms  1.2743ms  1.2743ms  [CUDA memcpy DtoH]
      API calls:   93.65%  564.84ms         1  564.84ms  564.84ms  564.84ms  cudaHostAlloc
                    5.15%  31.051ms         1  31.051ms  31.051ms  31.051ms  cudaDeviceReset
                    0.45%  2.7319ms         2  1.3660ms  1.3368ms  1.3951ms  cudaMemcpy
                    0.34%  2.0604ms         1  2.0604ms  2.0604ms  2.0604ms  cuDeviceGetPCIBusId
                    0.30%  1.8091ms         1  1.8091ms  1.8091ms  1.8091ms  cudaFreeHost
                    0.06%  342.90us         1  342.90us  342.90us  342.90us  cudaMalloc
                    0.04%  261.00us         1  261.00us  261.00us  261.00us  cudaFree
                    0.00%  15.400us       101     152ns     100ns     900ns  cuDeviceGetAttribute
                    0.00%  7.2000us         1  7.2000us  7.2000us  7.2000us  cudaSetDevice
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cudaGetDeviceProperties
                    0.00%  1.0000us         3     333ns     100ns     700ns  cuDeviceGetCount
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
~~~
**[CUDA memcpy DtoH]:** Copia de memoria de Device (GPU) a Host (CPU). Esta actividad tomó el 50.57% del tiempo total de las actividades de la GPU.
**[CUDA memcpy HtoD]:** Copia de memoria de Host (CPU) a Device (GPU). Esta actividad tomó el 49.43% del tiempo total de las actividades de la GPU.
**cudaHostAlloc**: Esta  funcion se utiliza para asignar memoria en el host (CPU) que también será accesible desde el dispositivo (GPU). Esta función tomó el 93.65% del tiempo total de las llamadas a la API.

# readSegment
~~~
==963== NVPROF is profiling process 963, command: ./readSegment
==963== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==963== Profiling application: ./readSegment
==963== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   49.71%  992.10us         1  992.10us  992.10us  992.10us  [CUDA memcpy DtoH]
                   45.41%  906.47us         2  453.23us  447.23us  459.23us  [CUDA memcpy HtoD]
                    2.48%  49.408us         1  49.408us  49.408us  49.408us  readOffset(float*, float*, float*, int, int)
                    2.40%  48.001us         1  48.001us  48.001us  48.001us  warmup(float*, float*, float*, int, int)
      API calls:   93.88%  603.77ms         3  201.26ms  313.00us  603.14ms  cudaMalloc
                    5.02%  32.299ms         1  32.299ms  32.299ms  32.299ms  cudaDeviceReset
                    0.52%  3.3638ms         3  1.1213ms  585.30us  2.1168ms  cudaMemcpy
                    0.40%  2.5464ms         1  2.5464ms  2.5464ms  2.5464ms  cuDeviceGetPCIBusId
                    0.13%  833.20us         3  277.73us  167.00us  455.50us  cudaFree
                    0.03%  206.30us         2  103.15us  68.900us  137.40us  cudaDeviceSynchronize
                    0.01%  65.800us         2  32.900us  16.900us  48.900us  cudaLaunchKernel
                    0.00%  15.800us       101     156ns     100ns  1.4000us  cuDeviceGetAttribute
                    0.00%  5.5000us         1  5.5000us  5.5000us  5.5000us  cudaSetDevice
                    0.00%  4.9000us         1  4.9000us  4.9000us  4.9000us  cudaGetDeviceProperties
                    0.00%  1.2000us         2     600ns     600ns     600ns  cudaGetLastError
                    0.00%     900ns         3     300ns     100ns     600ns  cuDeviceGetCount
                    0.00%     800ns         2     400ns     100ns     700ns  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
~~~
**[CUDA memcpy DtoH]:** Copia de memoria de Device (GPU) a Host (CPU). Esta actividad tomó el 49.71% del tiempo total de las actividades de la GPU.
**[CUDA memcpy HtoD]:** Copia de memoria de Host (CPU) a Device (GPU). Esta actividad tomó el 45.41% del tiempo total de las actividades de la GPU.
**readOffset:** Esta es una función (kernel) que se ejecuta en la GPU. Esta actividad tomó el 2.48% del tiempo total de las actividades de la GPU.
**warmup:** Esta es otra función (kernel) que se ejecuta en la GPU. Esta actividad tomó el 2.40% del tiempo total de las actividades de la GPU.
**cudaMalloc:** Función para asignar memoria en la GPU. Esta función tomó el 93.88% del tiempo total de las llamadas a la API.

Este programa CUDA demuestra el impacto negativo en rendimiento de lecturas de memoria no alineadas en GPU. 

1. Configura dispositivo CUDA, asigna memoria en host y device, inicializa datos en host.
2. Copia datos a GPU, ejecuta kernels que suman vectores en paralelo. 
3. Uno de los kernels fuerza lecturas no alineadas. 
4. Compara tiempos de ejecución entre lecturas alineadas y no alineadas.
5. Las lecturas no alineadas requieren transacciones de memoria adicionales que degradan rendimiento.
6. La GPU funciona mejor con accesos alineados debido a su arquitectura paralela.
7. El programa muestra experimentalmente cómo alinear datos en memoria mejora rendimiento en GPU.

# readSegmentUnroll
~~~
==985== NVPROF is profiling process 985, command: ./readSegmentUnroll
==985== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==985== Profiling application: ./readSegmentUnroll
==985== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   64.13%  2.0672ms         3  689.07us  470.56us  864.49us  [CUDA memcpy DtoH]
                   27.79%  895.65us         2  447.83us  446.53us  449.12us  [CUDA memcpy HtoD]
                    1.94%  62.593us         4  15.648us  15.360us  16.320us  [CUDA memset]
                    1.56%  50.368us         1  50.368us  50.368us  50.368us  readOffsetUnroll4(float*, float*, float*, int, int)
                    1.55%  49.984us         1  49.984us  49.984us  49.984us  readOffset(float*, float*, float*, int, int)
                    1.54%  49.632us         1  49.632us  49.632us  49.632us  readOffsetUnroll2(float*, float*, float*, int, int)
                    1.49%  47.904us         1  47.904us  47.904us  47.904us  warmup(float*, float*, float*, int, int)
      API calls:   93.30%  592.46ms         3  197.49ms  309.10us  591.77ms  cudaMalloc
                    5.46%  34.676ms         1  34.676ms  34.676ms  34.676ms  cudaDeviceReset
                    0.69%  4.4052ms         5  881.04us  498.20us  1.8633ms  cudaMemcpy
                    0.32%  2.0617ms         1  2.0617ms  2.0617ms  2.0617ms  cuDeviceGetPCIBusId
                    0.12%  749.60us         3  249.87us  170.00us  390.60us  cudaFree
                    0.06%  357.30us         4  89.325us  71.700us  130.70us  cudaDeviceSynchronize
                    0.02%  144.90us         4  36.225us  22.500us  52.700us  cudaMemset
                    0.01%  91.300us         4  22.825us  9.4000us  47.700us  cudaLaunchKernel
                    0.00%  14.600us       101     144ns     100ns  1.3000us  cuDeviceGetAttribute
                    0.00%  6.9000us         1  6.9000us  6.9000us  6.9000us  cudaGetDeviceProperties
                    0.00%  5.7000us         1  5.7000us  5.7000us  5.7000us  cudaSetDevice
                    0.00%  2.4000us         4     600ns     500ns     700ns  cudaGetLastError
                    0.00%  1.5000us         2     750ns     300ns  1.2000us  cuDeviceGet
                    0.00%  1.2000us         3     400ns     100ns     900ns  cuDeviceGetCount
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
~~~
**[CUDA memcpy DtoH]:** Copia de memoria de Device (GPU) a Host (CPU). Esta actividad tomó el 64.13% del tiempo total de las actividades de la GPU.
**[CUDA memcpy HtoD]:** Copia de memoria de Host (CPU) a Device (GPU). Esta actividad tomó el 27.79% del tiempo total de las actividades de la GPU.
**[CUDA memset]:** establece la memoria del dispositivo (GPU) a un valor específico.Esta actividad tomó el  1.94% del tiempo total de las actividades de la GPU.
**readOffset:** Esta es una función (kernel) que se ejecuta en la GPU. Esta actividad tomó el 1.55% del tiempo total de las actividades de la GPU.
**warmup:** Esta es otra función (kernel) que se ejecuta en la GPU. Esta actividad tomó el 1.49% del tiempo total de las actividades de la GPU.
**readOffsetUnroll2** y **readOffsetUnroll4**: Estas son funciones (kernels) que se ejecutan en la GPU. Estas actividades tomaron el 1.54% y el  1.56% del tiempo total de las actividades de la GPU, respectivamente.
**cudaMalloc:** Función para asignar memoria en la GPU. Esta función tomó el 93.30% del tiempo total de las llamadas a la API.

Este programa CUDA muestra técnicas para mitigar el impacto de lecturas de memoria no alineadas en GPU.

1. Configura dispositivo CUDA, asigna memoria en host y device, inicializa datos en host.  
2. Copia datos a GPU, ejecuta kernels que suman vectores en paralelo.
3. Uno de los kernels fuerza lecturas no alineadas. 
4. Otros kernels usan desenrollado de bucles para fundir lecturas no alineadas.
5. Compara tiempos entre lecturas alineadas, no alineadas y mitigación con desenrollado.
6. Las lecturas no alineadas degradan rendimiento en GPU. 
7. El desenrollado de bucles permite fundir múltiples lecturas no alineadas en una sola lectura alineada.
8. Esto reduce el impacto negativo de lecturas no alineadas en GPU.
9. El programa muestra experimentalmente estos conceptos de optimización de memoria.
