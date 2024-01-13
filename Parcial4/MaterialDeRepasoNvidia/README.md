# Data parallel computing

Este documento analiza específicamente la computación paralela de datos utilizando CUDA C. Cubre la estructura de los programas CUDA C, incluidas las funciones del kernel y la transferencia de datos. El capítulo también menciona la disponibilidad de herramientas y soporte para diferentes lenguajes de programación y API dentro de la plataforma CUDA.

## 2.1 Data Parallelism 

En el capítulo se nos enseña el concepto de paralelismo de datos en el contexto de las aplicaciones de software modernas. Explica que cuando las aplicaciones se ejecutan lentamente, a menudo se debe a que tienen demasiados datos para procesar.

El capítulo nos muestra varios ejemplos de tipos de aplicaciones que manejan grandes cantidades de datos, como edición de imágenes y videos, modelado científico, dinámica molecular y programación de aerolíneas. Destaca que muchos de estos elementos de datos se pueden procesar de forma independiente, lo que permite el cálculo paralelo.

El concepto de paralelismo de datos se explica como la reorganización de la computación en torno a evaluaciones independientes. Por ejemplo, se puede convertir un píxel de color a escala de grises o difuminar una imagen operando en píxeles individuales o en pequeñas vecindades de píxeles de forma independiente. Incluso operaciones aparentemente globales, como encontrar el brillo promedio de todos los píxeles de una imagen, se pueden dividir en cálculos más pequeños que se pueden ejecutar de forma independiente.

El capítulo también menciona que el libro se centra principalmente en enseñar conceptos relacionados con la programación CUDA C, pero proporciona apéndices que demuestran cómo estos conceptos se pueden aplicar a otros lenguajes como C++, FORTRAN, Python y OpenCL.

## 2.2 CUDA C Program Structure

La estructura de un programa CUDA C refleja la coexistencia de un host (CPU) y uno o más dispositivos (GPU) en la computadora. Cada archivo fuente CUDA puede tener una combinación de código de host y de dispositivo.

Una vez que se agregan las funciones del dispositivo y las declaraciones de datos a un archivo fuente, un compilador de C tradicional no puede compilarlo. En su lugar, debe ser compilado por un compilador CUDA C llamado NVCC (NVIDIA C Compiler). El compilador NVCC procesa el programa CUDA C, utilizando palabras clave CUDA para separar el código del host y el código del dispositivo.

El código del host, que es código ANSI C directo, se compila con los compiladores C/C++ estándar del host y se ejecuta como un proceso de CPU tradicional. Por otro lado, el código del dispositivo está marcado con palabras clave CUDA para funciones paralelas de datos, llamadas núcleos, y sus funciones auxiliares y estructuras de datos asociadas. El código del dispositivo es compilado por un componente de tiempo de ejecución de NVCC y ejecutado en un dispositivo GPU.

La ejecución de un programa CUDA comienza con el código de host (código de serie de la CPU). Cuando se llama o inicia una función del kernel (código de dispositivo paralelo), es ejecutada por una gran cantidad de subprocesos en un dispositivo. Estos hilos forman colectivamente una cuadrícula. La cuadrícula está dividida en bloques y cada bloque contiene varios subprocesos. Los subprocesos dentro de un bloque pueden cooperar y comunicarse entre sí mediante la memoria compartida.

El capítulo también analiza la asignación y desasignación de memoria del dispositivo utilizando funciones como cudaMalloc y cudaFree. Explica cómo se pueden transferir datos entre el host y el dispositivo utilizando la función cudaMemcpy. El capítulo concluye mencionando la importancia de la verificación y el manejo de errores en los programas CUDA.

## 2.3 A Vector Addition Kernel 
Esta sección analiza la implementación de una función del kernel para realizar la suma de vectores en CUDA C. Explica la estructura y sintaxis de la función del Kernel y cómo se puede utilizar para realizar cálculos paralelos en la GPU. El capítulo también proporciona ejemplos y fragmentos de código para ilustrar los conceptos y técnicas involucrados en la escritura de un núcleo de suma de vectores.
## 2.4 Device Global Memory and Data Transfer

En este capítulo se nos explica cómo asignar y desasignar memoria en el dispositivo, así como también cómo transferir datos entre el host y el dispositivo.

Se nos presenta el concepto de memoria del dispositivo, que es el espacio de memoria disponible en la GPU para almacenar datos. Explica que no se debe eliminar la referencia a la memoria del dispositivo en el código host para el cálculo y debe usarse principalmente para llamar a funciones API y funciones del kernel.

Se analiza el proceso de asignación de memoria del dispositivo utilizando la función cudaMalloc. Proporciona un ejemplo de asignación de memoria para una matriz flotante y explica que la dirección de la memoria asignada se almacena en una variable de puntero.

A continuación, el capítulo explica el proceso de transferencia de datos desde el host al dispositivo utilizando la función cudaMemcpy. Describe los parámetros de la función, incluida la ubicación de destino, la ubicación de origen y la cantidad de bytes que se copiarán.

Por último este capítulo también cubre la designación de memoria del dispositivo utilizando la función cudaFree. Explica que una vez que se completa el procesamiento de datos en el dispositivo, se puede liberar la memoria para que esté disponible para otros cálculos.

## 2.5 Kernel Functions and Threading

En CUDA, una función del kernel se encarga de especificar el código que ejecutarán todos los subprocesos durante una fase paralela. Todos los subprocesos ejecutan el mismo código, lo que hace que la programación CUDA sea una instancia del modelo de datos múltiples de programa único (SPMD).

El capítulo proporciona una descripción general de cómo se inician las funciones del kernel y el efecto de iniciar estas funciones. Menciona que la función del kernel es invocada por el código del host y ejecutada por múltiples subprocesos en la GPU. Los autores también analizan el concepto de bloques de subprocesos y configuración de cuadrícula, que determinan la cantidad de subprocesos y bloques de subprocesos utilizados en la ejecución.

Además, el capítulo destaca la importancia de la verificación y el manejo de errores en la programación CUDA. Menciona que las funciones de la API CUDA devuelven indicadores que indican si se ha producido un error durante la solicitud. Los autores enfatizan la necesidad de que los programas verifiquen y manejen los errores de manera adecuada.

## 2.6 Kernel Launch

Se nos explica cómo iniciar una función del kernel, que es el código que ejecutarán todos los subprocesos durante una fase paralela. El capítulo proporciona un ejemplo de un kernel de adición de vectores y demuestra cómo asignar memoria del dispositivo, transferir datos del host al dispositivo, invocar el kernel y transferir el resultado nuevamente al host. También cubre la verificación y el manejo de errores en la programación CUDA, enfatizando la importancia de verificar y manejar los errores que pueden ocurrir durante la ejecución de las funciones de la API CUDA.

Estructura de las funciones del kernel:

Las funciones del kernel se definen utilizando la palabra clave calificadora "global" en CUDA C.

La palabra clave "global" indica que la función es una función del kernel CUDA y se ejecutará en la GPU.

Las funciones del kernel solo se pueden llamar desde el código host, excepto en sistemas CUDA que admiten paralelismo dinámico.

Las funciones del kernel también pueden llamar a otras funciones del dispositivo u otras funciones del kernel.

Ejecución de funciones del kernel:

Cuando se llama o inicia una función del kernel desde el código del host, es ejecutada por una gran cantidad de subprocesos en la GPU.

Todos los subprocesos generados por el lanzamiento de un kernel se denominan colectivamente grilla.

La cuadrícula se divide en unidades más pequeñas llamadas bloques de hilos.

Cada bloque de subprocesos consta de varios subprocesos que se pueden ejecutar en paralelo.

La cantidad de bloques de subprocesos y la cantidad de subprocesos por bloque se pueden especificar durante el lanzamiento del kernel utilizando los parámetros de configuración de ejecución.

La ejecución de bloques de subprocesos dentro de una cuadrícula se puede realizar en paralelo, según los recursos de ejecución disponibles en la GPU.

Una GPU pequeña con recursos limitados puede ejecutar solo uno o dos bloques de subprocesos en paralelo, mientras que una GPU más grande puede ejecutar varios bloques de subprocesos simultáneamente.

Esta escalabilidad en la velocidad de ejecución permite a los kernels CUDA aprovechar diferentes configuraciones de hardware.
## 2.7 Summary

Este capítulo analiza las extensiones esenciales del lenguaje C que admiten la computación paralela. El capítulo destaca las siguientes extensiones:

Declaraciones de funciones: el capítulo resume las declaraciones de funciones que se han analizado, que se utilizan para definir funciones del núcleo y especificar su configuración de ejecución.

Lanzamiento del kernel: el capítulo proporciona una descripción general del lanzamiento del kernel, que implica especificar la cantidad de bloques de procesos y subprocesos por bloque para ejecutar la función del kernel.

Variables integradas (predefinidas): el capítulo menciona la presencia de variables integradas en CUDA, que tienen un significado y propósito especiales. Estas variables suelen ser de solo lectura y no deben usarse para otros fines.


## Conclusión

En mi opinión este documento me gusto mas debido a que siento que la información está más concentrada, con una mejor redacción y de fácil entendimiento que los libros pasados, siendo de esta forma mucho más sencillo entender lo que se explica a lo largo del documento.
