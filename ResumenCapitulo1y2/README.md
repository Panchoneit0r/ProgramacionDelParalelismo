# 1.- Introduction to GPU Kernels and Hardware

##1.1 Background
En este capítulo se nos menciona la importancia de que nuestros códigos se puedan ejecutar de forma paralela en nuestra computadora y como con herramientas como los hilos C++ es posible realizar esto mediante la cpu, más sin embargo el CPU tiene sus límites siendo que para poder aumentar la velocidad de nuestro equipo un doscientos por ciento se requerirá de grandes costos. Aquí es donde entra la GPU con CUDA de NVIDIA un lenguaje de programación que nos permite gestionar la GPU para alcanzar así ese aumento de 200 por ciento con una sola tarjeta gráfica gracias  que su memoria interna es aproximadamente 10 veces más rápida que la de una PC típica, lo cual es extremadamente útil para problemas limitados por el ancho de banda de memoria en lugar de la potencia de la CPU. Por último se nos habla un poco de los ejemplos que veremos más adelante y como estos ejemplos son mejores en este libro que en otros debido a que son interesantes de problemas del mundo real.

##1.2 First CUDA Example
En este capítulo se nos habla sobre el primer ejemplo que se utilizará para comprobar la eficacia de CUDA. El ejemplo utiliza la regla del trapecio para evaluar la integral de sin(x) desde 0 hasta π, basándose en la suma de un gran número de evaluaciones equidistantes de la función en este rango. Este ejemplo está hecho en C++ directo para ejecutarse en un solo hilo en la PC anfitriona. 
###El ejemplo 1.1 
```c++
#include <stdio.h>
#include <stdlib.h>
#include "cxtimers.h"
inline float sinsum(float x, int terms)
{
// sin(x) = x - x^3/3! + x^5/5! …
float term = x; // first term of series
float sum = term; // sum of terms so far
float x2 = x*x;
for(int n = 1; n < terms; n++){
term *= -x2 / (float)(2*n*(2*n+1));
sum += term;
}
return sum;
}

int main(int argc, char *argv[])
{
int steps = (argc >1) ? atoi(argv[1]) : 10000000;
int terms = (argc >2) ? atoi(argv[2]) : 1000;
double pi = 3.14159265358979323;
double step_size = pi/(steps-1); // n-1 steps
cx::timer tim;
double cpu_sum = 0.0;
for(int step = 0; step < steps; step++){
float x = step_size*step;
cpu_sum += sinsum(x, terms); // sum of Taylor series
}
double cpu_time = tim.lap_ms(); // elapsed time
// Trapezoidal Rule correction
 cpu_sum -= 0.5*(sinsum(0.0,terms)+sinsum(pi, terms));
cpu_sum *= step_size;
printf("cpu sum = %.10f,steps %d terms %d time %.3f ms\n",
cpu_sum, steps, terms, cpu_time);

return 0;
 }
```
Es una lista completa del programa cpusum. Calcula la suma de una integral sinusoidal utilizando un único subproceso de CPU en la PC host. El programa toma dos argumentos de línea de comando: el número de pasos y el número de términos. Utiliza una aproximación de la serie de Taylor para calcular la suma. El código incluye una función llamada sinsum que calcula cada término de la serie. Luego, la función principal itera sobre los pasos y acumula la suma. Finalmente aplica la corrección de la Regla Trapezoidal e imprime el resultado.

###El ejemplo 1.2
Este ejemplo se basa en el Ejemplo 1.1 por lo que no veo necesario ponerlo ya que solo incluye 3 líneas nuevas de código y una de esas solo se encarga de incluir la librería de OpenMP, un modelo de programación paralela para utilizar los hilos.
Línea 19.5: Agrega la variable "threads" configurable por el usuario para establecer el número de subprocesos de CPU utilizados por OpenMP.
Línea 23.5: Llama a la función omp_set_num_threads(threads) para especificar el número de subprocesos paralelos que se utilizarán.
Luego, el código configura el cálculo paralelo utilizando la directiva #pragma omp paralelo for, que divide el siguiente bucle for en subbucles ejecutados en paralelo en diferentes subprocesos de la CPU. La variable omp_sum se utiliza para almacenar la suma calculada en paralelo. El código demuestra cómo se puede utilizar OpenMP para paralelizar cálculos y mejorar el rendimiento alcanzando una mejora de rendimiento de 3.8.

###El ejemplo 1.3
```c++
// call sinsum steps times using parallel threads on GPU
#include <stdio.h>
 #include <stdlib.h>
#include "cxtimers.h" // cx timers
#include "cuda_runtime.h" // cuda basic
#include "thrust/device_vector.h" // thrust device vectors
 __host__ __device__ inline float sinsum(float x, int terms) {
 	float x2 = x*x;
float term = x; // first term of series
float sum = term; // sum of terms so far
for(int n = 1; n < terms; n++){
term *= -x2 / (2*n*(2*n+1)); // build factorial
sum += term;
}
return sum;
}
__global__ void gpu_sin(float *sums, int steps, int terms, float step_size){
// unique thread ID
int step = blockIdx.x*blockDim.x+threadIdx.x;
if(step<steps){
float x = step_size*step;
sums[step] = sinsum(x, terms); // store sums
}
}
int main(int argc, char *argv[]) {
// get command line arguments
int steps = (argc >1) ? atoi(argv[1]) : 10000000;
int terms = (argc >2) ? atoi(argv[2]) : 1000;
int threads = 256;
int blocks = (steps+threads-1)/threads; // round up
double pi = 3.14159265358979323;
double step_size = pi / (steps-1); // NB n-1
// allocate GPU buffer and get pointer
thrust::device_vector<float> dsums(steps);
float *dptr = thrust::raw_pointer_cast(&dsums[0]);
cx::timer tim;
 gpu_sin<<<blocks,threads>>>(dptr,steps,terms, (float)step_size);
double gpu_sum = thrust::reduce(dsums.begin(),dsums.end());
double gpu_time = tim.lap_ms(); // get elapsed time
// Trapezoidal Rule Correction
gpu_sum -= 0.5*(sinsum(0.0f,terms)+sinsum(pi, terms));
gpu_sum *= step_size;
printf("gpusum %.10f steps %d terms %d time %.3f  ms\n",gpu_sum,steps,terms,gpu_time);
return 0;
}
```
Este ejemplo es una continuación de los ejemplos anteriores y se centra en la programación CUDA.. Los métodos CUDA utilizados se describen detalladamente más adelante en el documento. El código está escrito en C++ con algunas palabras clave adicionales específicas de CUDA. Los detalles del cálculo son visibles en el código y gran parte de ellos permanecen sin cambios con respecto a los ejemplos anteriores. El ejemplo también introduce el concepto de utilizar el mismo código tanto para el host (CPU) como para el dispositivo (GPU), eliminando posibles errores. 
Como comentario final, notamos que el resultado de la versión CUDA es un poco menos preciso que cualquiera de las versiones host. Esto se debe a que la versión CUDA utiliza flotantes de 4 bytes durante todo el cálculo, incluido el paso de reducción final, mientras que las versiones de host utilizan un doble de 8 bytes para acumular la suma del resultado final . Sin embargo, el resultado CUDA tiene una precisión de ocho cifras significativas, lo que es más que suficiente para la mayoría de aplicaciones científicas.

##1.3 CPU Architecture
El reloj maestro de una CPU actúa como conductor, enviando pulsos de reloj a una frecuencia fija a cada unidad, determinando la velocidad de procesamiento de la CPU.
La memoria principal de una CPU contiene tanto datos de programas como instrucciones de código de máquina.
La arquitectura de una CPU se puede clasificar como von Neumann o Harvard, dependiendo de si las instrucciones y los datos se almacenan en una memoria común o en un hardware separado.
El contador de programa (PC) realiza un seguimiento de la instrucción actual que está ejecutando la CPU.
Las GPU tienen una disposición jerárquica de núcleos de cómputo, con múltiples multiprocesadores (SM) de transmisión y bloques de subprocesos.
Las GPU tienen diferentes tipos de memoria, incluida la memoria global, la memoria compartida y la memoria constante.
La arquitectura de una GPU permite la ejecución paralela de subprocesos, lo que la hace adecuada para tareas altamente paralelas.
En general, este capítulo proporciona una base para comprender la arquitectura de hardware de las CPU y GPU, lo cual es esencial para escribir código eficiente y de alto rendimiento.


##1.4 CPU Compute Power
La potencia informática de las CPU ha aumentado drásticamente con el tiempo, y el número de transistores por chip sigue la Ley de Moore. Sin embargo, la frecuencia de la CPU dejó de aumentar en 2002. El reciente crecimiento en el rendimiento por chip se debe a innovaciones en el diseño, en particular la adopción de tecnología multinúcleo. Vale la pena señalar que las GPU no están incluidas en esta gráfica, pero los diseños recientes de Intel Xeon-phi con cientos de núcleos se están volviendo similares a las GPU. Sorprendentemente, la energía utilizada por un solo dispositivo no ha aumentado desde 2002. Los datos para este análisis se pueden encontrar en https://github.com/karlrupp/microprocessor-trend-data.

##1.5 CPU Memory Management: Latency Hiding Using Caches
Los datos y las instrucciones en una CPU no se mueven instantáneamente entre bloques, sino que avanzan paso a paso a través de los registros de hardware, lo que genera una latencia entre la emisión de una solicitud de datos y su llegada.
Para ocultar esta latencia, se utilizan técnicas de almacenamiento en caché y canalización. Los datos almacenados en ubicaciones de memoria secuenciales se procesan secuencialmente, de modo que cuando se solicita un elemento, el hardware envía este elemento y los elementos adyacentes en sucesivos tics de reloj.
Las PC emplean unidades de memoria caché para almacenar datos de la memoria principal. En los chips multinúcleo de CPU modernos, hay tres niveles de memorias caché integradas en el chip de la CPU, con cachés L1 independientes para datos e instrucciones.
Los datos de la caché se transfieren en paquetes llamados líneas de caché, normalmente de 64 o 128 bytes de tamaño. Los datos en una línea de caché corresponden a una región contigua de la memoria principal que comienza en una dirección que es un múltiplo del tamaño de la línea de caché.
Caché L1: El caché L1 es un tipo de memoria caché que está integrada en cada núcleo de la CPU. Consta de cachés separados para datos e instrucciones. La caché L1 es la caché más rápida y se utiliza para almacenar datos e instrucciones a
los que se accede con frecuencia para un acceso rápido por parte del núcleo de la CPU.
Caché L2: El caché L2 es otro nivel de memoria caché que también está integrado en cada núcleo de la CPU. Es más grande que el caché L1 y proporciona almacenamiento adicional para datos e instrucciones a los que se accede con frecuencia. La caché L2 es más lenta que la caché L1 pero aún más rápida que la memoria principal.
Caché L3: El caché L3 es el caché más grande entre los tres niveles de memoria caché. Es compartido por todos los núcleos de la CPU en un chip multinúcleo. La caché L3 proporciona una mayor capacidad de almacenamiento para datos e instrucciones a los que se accede con frecuencia y que se comparten entre los núcleos. Es más lento que el caché L2 pero más rápido que la memoria principal.

##1.6 CPU: Parallel Instruction Set
Las CPU Intel tienen capacidades paralelas en forma de instrucciones vectoriales. Estas instrucciones vectoriales se introdujeron por primera vez en 1999 con el conjunto de instrucciones Pentium III SSE. Estas instrucciones utilizan registros de 128 bits que pueden contener 4 flotantes de 4 bytes. Al almacenar conjuntos de 4 flotantes en ubicaciones de memoria secuenciales y alinearlos en límites de memoria de 128 bytes, se pueden tratar como vectores de 4 elementos. Estos vectores se pueden cargar y almacenar desde los registros SSE en un solo ciclo de reloj, lo que permite una aritmética vectorial más rápida. A lo largo de los años, SSE ha evolucionado y las CPU Intel más nuevas ahora admiten AVX2, que utiliza registros de 256 bits y puede manejar varios tipos de datos. La versión más reciente es AVX-512, que utiliza registros de 512 bytes capaces de contener vectores de hasta 16 flotantes u 8 dobles. El uso de AVX en CPU Intel se analiza en detalle en el Apéndice D.

##1.7 GPU Architecture
Las GPU se diseñaron inicialmente para gráficos de computadora de alto rendimiento, específicamente para juegos. Se utilizaron para calcular cada píxel de la pantalla del juego, lo que requirió una gran cantidad de cálculos por segundo. Las tarjetas de juego surgieron como hardware dedicado con múltiples procesadores simples para realizar estos cálculos de píxeles. La matriz de píxeles que representa la imagen se almacena en una matriz 2D en un búfer de cuadros digital, con cada píxel representado por 3 bytes para las intensidades de rojo, verde y azul. Pronto se dio cuenta de que estas tarjetas económicas podían usarse para potentes cálculos paralelos más allá de los juegos. Esto llevó al surgimiento de GPGPU (computación de propósito general en unidades de procesamiento de gráficos) y al lanzamiento del kit de herramientas de programación de GPU de NVIDIA en 2007, lo que hizo que la programación de GPU fuera más accesible y convencional.
NVIDIA produce tres clases de GPU:
###Modelos de las marcas GeForce GTX, GeForce RTX y Titan: estos modelos están dirigidos al mercado de los juegos y son los más económicos. Es posible que tengan menos soporte para los cálculos de FP64 en comparación con las versiones científicas y no utilicen memoria EEC. Sin embargo, su rendimiento para los cálculos del FP32 puede igualar o superar las tarjetas científicas. La RTX 3090 tiene el rendimiento FP32 más alto entre todas las tarjetas NVIDIA lanzadas hasta marzo de 2021.
###Modelos de la marca Tesla: estos modelos están dirigidos al mercado de la informática científica de alta gama. Tienen buen soporte para cálculos FP64 y utilizan memoria EEC. Las tarjetas Tesla no tienen puertos de salida de vídeo y no se pueden utilizar para jugar. Son adecuados para su implementación en granjas de servidores.
###GPU de la marca Quadro: estas GPU son esencialmente GPU modelo Tesla con capacidades gráficas adicionales. Están dirigidos al mercado de estaciones de trabajo de escritorio de alta gama. Dentro de cada generación de GPU Quadro, suele haber varios modelos que pueden diferir en las características del software. La capacidad específica de una GPU se conoce como capacidad de cómputo (CC), que se especifica mediante un número que aumenta monótonamente. La generación más reciente es Ampere con un valor CC de 8,0. Los ejemplos del libro se desarrollaron utilizando una GPU Turing RX 2070 con un CC de 7,5.

##1.8 Pascal Architecture
Las GPU NVIDIA de la generación Pascal se construyen jerárquicamente, comenzando con los núcleos de computación básicos. Estos núcleos pueden realizar operaciones de punto flotante y enteros de 32 bits y no tienen contadores de programas individuales. Grupos de 32 núcleos se agrupan para formar "motores warp", que son las unidades de ejecución básicas en los programas del kernel CUDA. Todos los subprocesos en una deformación se ejecutan al mismo tiempo, ejecutando la misma instrucción. Los motores Warp agregan recursos informáticos adicionales, incluidas unidades de funciones especiales (SFU) para una evaluación rápida de funciones trascendentales y unidades de coma flotante de doble precisión (FP64). Los motores Warp se agrupan para formar multiprocesadores (SM) simétricos, que normalmente tienen 128 núcleos de cómputo. Los SM también incluyen unidades de textura y varios recursos de memoria en el chip. Luego se agrupan varios SM para crear la GPU final. Por ejemplo, la GTX 1080 tiene 20 SM que contienen un total de 2560 núcleos de cálculo. Las tarjetas de juego pueden tener diferentes números de unidades SM, velocidades de reloj y tamaños de memoria.

##1.9 GPU Memory Types
La memoria global es el tipo de memoria más grande y más lento de la GPU. Se utiliza para almacenar datos a los que deben acceder todos los subprocesos de un kernel.
La memoria compartida es un tipo de memoria más pequeña y rápida que se comparte entre subprocesos dentro de un bloque de subprocesos. Se utiliza para almacenar datos a los que los subprocesos dentro de un bloque deben acceder con frecuencia y rapidez.
La memoria constante es un tipo de memoria de solo lectura que se utiliza para almacenar datos constantes a los que acceden todos los subprocesos de un núcleo.
La memoria de textura es un tipo especializado de memoria optimizada para el acceso a datos espaciales 2D y 3D. Se utiliza para almacenar y acceder a datos de textura en programas CUDA.

##1.10 Warps and Waves
Los warps son un concepto clave en la arquitectura de GPU de NVIDIA. Un warp es un grupo de 32 subprocesos que se ejecutan al mismo tiempo, lo que significa que todos ejecutan la misma instrucción al mismo tiempo. Los warps se ejecutan en un motor warp, que es un grupo de 32 núcleos que comparten recursos informáticos. El motor warp mantiene un único contador de programa que se utiliza para enviar una secuencia de instrucciones común a todos los núcleos del warp. Esto permite la ejecución eficiente de instrucciones paralelas en la GPU. Las Waves son el concepto equivalente en las GPU de AMD.

##1.11 Blocks and Grids
Los bloques(blocks) y las cuadrículas(grid) son conceptos clave en la programación CUDA.
Un bloque es un grupo de subprocesos que se agrupan y se ejecutan en el mismo multiprocesador de transmisión (SM) en la GPU. El tamaño del bloque debe ser múltiplo del tamaño warp, que actualmente es 32 para todas las GPU NVIDIA. Los subprocesos dentro del mismo bloque pueden comunicarse entre sí mediante la memoria compartida o global del dispositivo y pueden sincronizarse entre sí cuando sea necesario.
Una cuadrícula, por otro lado, es una colección de bloques. Al iniciar un kernel CUDA, especificamos la configuración de lanzamiento con dos valores: el tamaño del bloque de subprocesos y el número de bloques de subprocesos. El tamaño de la cuadrícula es simplemente el número de bloques de hilos. El número total de subprocesos se especifica implícitamente como el producto del tamaño del bloque de subprocesos y el número de bloques de subprocesos.
Es importante tener en cuenta que los subprocesos en diferentes bloques de subprocesos no pueden comunicarse durante la ejecución del kernel y el sistema no puede sincronizar subprocesos en diferentes bloques de subprocesos. Por lo tanto, es común que el tamaño del bloque de subprocesos sea un submúltiplo de 1024, a menudo 256. En tales casos, las deformaciones de diferentes bloques de subprocesos coexistieron en los SM durante la ejecución del kernel.

##1.12 Occupancy
La Occupancy en la programación de GPU se refiere a la relación entre la cantidad de subprocesos que realmente residen en las unidades de Streaming Multiprocessors (SM) en comparación con el valor máximo. Generalmente se expresa como porcentaje. Una ocupación total del 100 por ciento significa que se están ejecutando oleadas completas en los SM de la GPU. Sin embargo, es posible que no siempre sea posible lograr la ocupación total debido a las limitaciones en el tamaño de la memoria compartida y la cantidad de registros en cada SM. La cantidad de memoria compartida y registros utilizados por cada bloque de subprocesos pueden afectar la ocupación.

#2 Thinking and Coding in Parallel 

##2.1 Flynn’s Taxonomy 
La taxonomía de Flynn clasifica las arquitecturas informáticas en cinco tipos: SISD, SIMD, MIMD, MISD y SIMT.
SISD (datos únicos de instrucción única): esta arquitectura representa un solo procesador que ejecuta un solo subproceso. Puede dar la ilusión de realizar múltiples tareas al cambiar rápidamente entre tareas, pero en realidad ejecuta una operación en un elemento de datos a la vez.
SIMD (Instrucción única de datos múltiples): en esta arquitectura, el hardware puede ejecutar la misma instrucción en múltiples elementos de datos simultáneamente. Utiliza múltiples ALU alimentadas con diferentes elementos de datos pero un decodificador de instrucciones común. Estas arquitecturas suelen denominarse procesadores vectoriales.
MIMD (Instrucción múltiple, datos múltiples): MIMD consta de CPU independientes que realizan tareas independientes. Esto incluye PC multinúcleo modernas y grupos de PC conectados por una red. Se puede utilizar software adecuado como MPI u OpenMP para permitir que varios procesadores trabajen juntos en una única tarea informática.
MISD (Datos únicos de instrucción múltiple): MISD rara vez se usa y se encuentra principalmente en sistemas integrados especializados que requieren redundancia contra fallas, como los satélites.
SIMT (Subprocesos múltiples de instrucción única): SIMT es una variación de SIMD introducida por NVIDIA para su arquitectura GPU. Utiliza una gran cantidad de subprocesos para procesar elementos de datos individuales. Si bien el comportamiento de SIMD se replica cuando todos los subprocesos utilizan una instrucción común, SIMT también permite que los subprocesos realicen operaciones divergentes, lo que genera un código más versátil.
La arquitectura SIMD es de particular interés para la programación paralela, donde se realiza la misma operación en múltiples elementos de datos, que a menudo se encuentran en bucles for. Sin embargo, para compartir el cálculo de un bucle entre varios subprocesos, es importante asegurarse de que no haya dependencias entre los pasos del bucle, lo que significa que el orden de los recorridos del bucle no debería importar.

##2.2 Kernel Call Syntax 
La forma general de una llamada a un kernel CUDA implica el uso de hasta cuatro argumentos especiales encerrados entre corchetes <<< >>>. Estos argumentos definen varios aspectos de la ejecución del kernel.
El primer argumento especifica las dimensiones de la cuadrícula de bloques de subprocesos utilizados por el núcleo. Puede ser un número entero o un entero sin signo para direccionamiento de bloques lineales, o un tipo dim3 para definir una cuadrícula 2D o 3D de bloques de subprocesos.
El segundo argumento determina el número de subprocesos en un solo bloque de subprocesos. Puede ser un número entero o un entero sin signo para el direccionamiento lineal de subprocesos dentro de un bloque, o un tipo dim3 para definir una estructura de matriz 2D o 3D para los subprocesos dentro de un bloque de subprocesos.
El tercer argumento es opcional y de tipo size_t o int. Define el número de bytes de memoria compartida asignada dinámicamente utilizados por cada bloque de subprocesos del kernel. Si se omite o se establece en cero, no se reserva ninguna memoria compartida. Alternativamente, el propio núcleo puede declarar memoria compartida estática, pero su tamaño debe conocerse en el momento de la compilación.
El cuarto argumento también es opcional y de tipo cudaStream_t. Especifica la secuencia CUDA en la que ejecutar el kernel. Esta opción se utiliza normalmente en aplicaciones avanzadas que ejecutan varios núcleos simultáneos.
Estos cuatro argumentos proporcionan la información necesaria para iniciar un kernel CUDA y controlar su ejecución.

##2.3 3D Kernel Launches 
Un núcleo 3D se refiere a una función o algoritmo computacional que opera en una cuadrícula de datos tridimensional. En el contexto de la programación de GPU con CUDA, un kernel 3D es una función que se ejecuta mediante varios subprocesos en paralelo en una GPU. Cada hilo realiza cálculos en un elemento específico de una matriz o cuadrícula tridimensional. El uso de núcleos 3D permite un procesamiento eficiente de datos en aplicaciones como procesamiento de volúmenes y transformaciones de imágenes 3D. El documento proporcionado proporciona ejemplos de kernels 3D y su implementación utilizando CUDA.

##2.4 Latency Hiding and Occupancy 
En el contexto de la programación de GPU, el ocultamiento de latencia se refiere a la técnica de ocultar la latencia de los accesos a la memoria realizando cálculos independientes mientras se espera que lleguen los datos. Esto se logra ejecutando instrucciones que no dependen de los datos pendientes. La ocupación, por otro lado, se refiere a la relación entre la cantidad de subprocesos que realmente residen en los multiprocesadores de transmisión (SM) de la GPU en comparación con el valor máximo. Suele expresarse como porcentaje e indica el nivel de utilización de los recursos.
La ocultación de latencia se utiliza para mitigar el impacto de la latencia de acceso a la memoria en el rendimiento de la GPU. Al realizar cálculos independientes mientras espera datos, la GPU puede mantener ocupadas sus unidades de ejecución y maximizar el rendimiento.
La ocupación es una medida de la utilización de recursos en la programación de GPU. Indica el porcentaje de warps activos (grupos de 32 hilos) que residen en los SM de la GPU. Una mayor ocupación generalmente conduce a un mejor rendimiento.
Lograr la ocupación total puede requerir una cuidadosa consideración del hilo

##2.5 Parallel Patterns 
La programación paralela para GPU requiere un enfoque diferente en comparación con las CPU individuales o códigos con una pequeña cantidad de subprocesos independientes. Una diferencia importante es que se evitan declaraciones de rama en el código CUDA. Las divergencias entre ramas pueden provocar una pérdida de rendimiento, ya que el sistema serializa las llamadas a diferentes funciones. Se recomienda utilizar una ejecución condicional modesta dentro de deformaciones de 32 subprocesos para minimizar el impacto de las declaraciones if.
Las declaraciones de rama son problemáticas en el código CUDA.
Si todos los subprocesos en una deformación tienen el mismo valor de indicador, hay poca pérdida de rendimiento.
Si incluso un hilo tiene un valor de indicador diferente, se produce una divergencia de rama.
El sistema serializa las llamadas a diferentes funciones en el warp.
La ejecución condicional modesta dentro de deformaciones de 32 subprocesos es menos dañina.

##2.6 Parallel Reduce
Parallel Reduce es un patrón de codificación que se utiliza para realizar la misma operación en un gran conjunto de números en paralelo. Implica encontrar la suma aritmética de los números, pero también se pueden realizar otras operaciones como max o min. El objetivo es utilizar tantos subprocesos como sea posible para ocultar la latencia de la memoria y mejorar la eficiencia. La operación de reducción es un ejemplo de una primitiva paralela, donde cada sub bucle puede acumular su propia suma parcial, que luego se puede sumar para calcular el valor final.
Parallel Reduce se utiliza para realizar la misma operación en un gran conjunto de números en paralelo.
Implica encontrar la suma aritmética de los números, pero también se pueden realizar otras operaciones como max o min.
El objetivo es utilizar tantos subprocesos como sea posible para ocultar la latencia de la memoria y mejorar la eficiencia.
Cada subbucle puede acumular su propia suma parcial, que luego se puede sumar para calcular el valor final.

##2.7 Shared Memory 
La memoria compartida es una característica importante en la programación CUDA que permite que los subprocesos dentro de un bloque de subprocesos cooperen eficientemente entre sí. Proporciona una forma rápida para que los subprocesos se comuniquen y compartan datos. La memoria compartida se asigna por unidad SM y se divide entre todos los bloques de subprocesos residentes actualmente que la solicitan. La cantidad de memoria compartida disponible depende de la capacidad informática de la GPU. Es importante equilibrar la ganancia de rendimiento derivada del uso de memoria compartida con la ocupación reducida de SM cuando se necesitan grandes cantidades de memoria compartida.
La memoria compartida permite que los subprocesos dentro de un bloque de subprocesos cooperen de manera eficiente.
Cada unidad SM tiene un grupo de memoria compartida que se divide entre todos los bloques de subprocesos residentes actualmente.
La memoria compartida es más rápida que la memoria global y puede ser tan rápida como usar registros.
La cantidad de memoria compartida disponible depende de la capacidad informática de la GPU.
Es importante equilibrar la ganancia de rendimiento derivada del uso de memoria compartida con la ocupación reducida de SM cuando se necesitan grandes cantidades de memoria compartida.

##2.8 Matrix Multiplication 
La multiplicación de matrices es una operación fundamental en álgebra lineal computacional. Una matriz es una matriz rectangular 2D de números y el resultado de multiplicar dos matrices es otra matriz. Las dimensiones de la matriz resultante están determinadas por el número de filas y columnas de las matrices de entrada. La multiplicación se realiza tomando el producto escalar de cada fila de la primera matriz con cada columna de la segunda matriz. Los elementos resultantes se calculan mediante una fórmula que implica sumar los productos de los elementos correspondientes de las dos matrices.
La multiplicación de matrices implica multiplicar dos matrices para obtener una tercera matriz.
Las dimensiones de la matriz resultante están determinadas por las dimensiones de las matrices de entrada.
La multiplicación se realiza tomando el producto escalar de cada fila de la primera matriz con cada columna de la segunda matriz.
Los elementos de la matriz resultante se calculan mediante una fórmula que implica sumar los productos de los elementos correspondientes de las dos matrices.

##2.9 Tiled Matrix Multiplication 
La multiplicación de matrices en mosaico es una técnica utilizada para mejorar el rendimiento de la multiplicación de matrices en GPU. Implica dividir las matrices en mosaicos más pequeños y realizar la multiplicación en estos mosaicos. Esto permite una mejor utilización de la memoria compartida y reduce la cantidad de veces que es necesario leer elementos desde la memoria externa.
La multiplicación de matrices en mosaico se puede implementar en los núcleos CUDA mediante el uso de bloques de subprocesos para representar pares de mosaicos de las matrices de entrada.
Cada subproceso en un bloque de subprocesos copia elementos de sus mosaicos asignados en matrices de memoria compartida.
Una vez que se copian los elementos, los subprocesos pueden calcular los elementos de la multiplicación de la matriz en mosaico, lo que reduce la cantidad de veces que es necesario leer los elementos desde la memoria externa.
La multiplicación de matrices en mosaico mejora el rendimiento al explotar la memoria compartida y reducir la latencia de acceso a la memoria.

##2.10 BLAS
El BLAS (Subprogramas de álgebra lineal básica) es una colección de funciones que proporcionan implementaciones eficientes de operaciones comunes de álgebra lineal, como la multiplicación de matrices. Estas funciones se han desarrollado durante más de 50 años y se utilizan ampliamente en álgebra lineal computacional.
El BLAS es un conjunto de funciones para realizar operaciones de álgebra lineal de manera eficiente.
Incluye funciones para multiplicación de matrices y otras operaciones comunes.
Las funciones BLAS se han desarrollado durante muchos años y se utilizan ampliamente en álgebra lineal computacional.

#Conclusión 
En general, este libro proporciona una base para comprender la arquitectura de hardware de las CPU y GPU, así como los conceptos clave relacionados con la programación paralela en CUDA. Se discuten las diferencias entre las arquitecturas de CPU y GPU, incluyendo la clasificación de Flynn y la disposición jerárquica de núcleos de cómputo en las GPU.
También se aborda la importancia de la memoria compartida, la ejecución paralela de subprocesos y la optimización del rendimiento en CUDA. Se mencionan los desafíos asociados con las declaraciones de rama en el código CUDA y se introduce el concepto de Parallel Reduce para realizar operaciones en paralelo en conjuntos de datos.
Además, se destaca la importancia de la memoria caché y la latencia en la gestión de la memoria de la CPU, y se menciona cómo se utilizan las técnicas de almacenamiento en caché y canalización para ocultar la latencia.
Este libro resultó de mucha utilidad para mi pues menciona muchas cosas que desconocía me hizo darme cuenta del desconocimiento que tengo en cuanto hardware pues había muchos conceptos que desconocía por completo. Siento que libro es algo avanzado pues una persona con desconocimiento de tecnología se va sentir muy perdido siendo en mi opinión que debería abordar un poco algunos conceptos básicos para que sea más entendible, sin embargo es un muy buen libro para alguien con conocimientos previos en informática que aborda de una muy buena manera la programación paralela de forma muy extensa   
