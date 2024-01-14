# Scalable parallel execution

El contenido general del documento se centra en discutir los conceptos de recursos, capacidades y ejecución paralela en el contexto de los sistemas informáticos. Proporciona ejemplos de la vida cotidiana, como servicios de hoteles y escenarios de oficinas de correos, para explicar estos conceptos. El documento también incluye ejercicios relacionados con bloques de subprocesos, dimensiones de cuadrícula y tolerancia de latencia. El propósito del documento es educar a los lectores sobre la importancia de comprender los recursos y capacidades en diversos entornos y cómo se relacionan con la ejecución paralela en sistemas informáticos.

## 3.1 CUDA Thread Organization 

Este capítulo trata sobre el acceso a datos multidimensionales en CUDA. Explica la organización de los subprocesos dentro de un bloque y cómo cada threadIdx consta de tres campos: threadIdx.x, threadIdx.y y threadIdx.z. El capítulo también analiza las dimensiones de la cuadrícula y el bloque, y cómo se asignan los bloques a los multiprocesadores de transmisión (SM) en el dispositivo CUDA. Menciona que el sistema de ejecución de CUDA ajusta automáticamente la cantidad de bloques asignados a cada SM según la disponibilidad de recursos. El capítulo también proporciona un ejemplo de cómo asignar subprocesos a datos multidimensionales y acceder a bytes específicos dentro de los datos.

## 3.2 Mapping Threads to Multidimensional Data 

El capitulo explica cómo se pueden organizar los subprocesos en una cuadrícula CUDA en una jerarquía de dos niveles, con una cuadrícula que consta de uno o más bloques y cada bloque consta de uno o más subprocesos.

El capítulo comienza explicando la organización de los hilos dentro de un bloque. Cada hilo tiene tres coordenadas: threadIdx.x, threadIdx.y y threadIdx.z. Estas coordenadas permiten que los subprocesos se distinguen entre sí e identifiquen la porción de datos que necesitan procesar.

Luego, el capítulo analiza el mapeo de subprocesos a datos multidimensionales, particularmente en el contexto del procesamiento de imágenes. Explica que utilizar una cuadrícula 2D con bloques 2D suele ser conveniente para procesar píxeles en una imagen. El ejemplo dado es una imagen de 76x62, donde se usa un bloque de 16x16 con 16 hilos en la dirección x y 16 hilos en la dirección y. La cuadrícula está organizada en 5 bloques en la dirección x y 4 bloques en la dirección y.

El capítulo también aborda la cuestión del acceso a datos multidimensionales en CUDA. Explica que las matrices multidimensionales asignadas dinámicamente deben linealizarse o "aplanarse" en matrices unidimensionales equivalentes en CUDA C. Esto se debe a que el número de columnas en una matriz 2D asignada dinámicamente no se conoce en el momento de la compilación.

En general, el capítulo proporciona una comprensión de cómo se pueden asignar subprocesos a datos multidimensionales en la programación CUDA, centrándose en el procesamiento de imágenes y el acceso a matrices multidimensionales.

## 3.3 Image Blur: A More Complex Kernel 

Se analiza la organización de subprocesos dentro de un bloque en la programación CUDA. Explica que cada threadIdx consta de tres campos: threadIdx.x, threadIdx.y y threadIdx.z, que representan las coordenadas x, y, z del hilo. El capítulo también menciona que los bloques dentro de una cuadrícula tienen las mismas dimensiones y proporciona un ejemplo de un bloque organizado en una matriz de subprocesos de 4x2x2. Además, menciona que los dispositivos con una capacidad inferior a 2.0 permiten bloques de hasta 512 subprocesos.

## 3.4 Synchronization and Transparent Scalability 

Se analiza los parámetros de configuración de ejecución del kernel en la programación CUDA. Explica que las dimensiones de una cuadrícula y sus bloques están definidas por estos parámetros. Las coordenadas únicas en blockIdx y threadIdx permiten que los subprocesos de una cuadrícula se identifiquen a sí mismos y a sus dominios de datos. El programador es responsable de utilizar estas variables en las funciones del kernel para garantizar que los subprocesos puedan identificar correctamente la parte de los datos a procesar.

El capítulo también destaca que los subprocesos en diferentes bloques no pueden sincronizarse entre sí, lo que permite una escalabilidad transparente de las aplicaciones CUDA. Para superar esta limitación, el método simple es terminar el kernel e iniciar un nuevo kernel para las actividades después del punto de sincronización.

Además, el capítulo explica que los subprocesos se asignan a Streaming Multiprocessors (SM) para su ejecución bloque por bloque. Cada dispositivo CUDA tiene limitaciones en la cantidad de bloques y subprocesos que sus SM pueden acomodar. Las limitaciones de recursos de cada dispositivo pueden convertirse en el factor limitante de un kernel.

En general, el Capítulo 3.4 proporciona una comprensión de cómo funcionan los parámetros de configuración de ejecución del kernel en la programación CUDA y las consideraciones para la sincronización de subprocesos y las limitaciones de recursos.

## 3.5 Resource Assignment 

Se nos habla sobre lla consulta de propiedades del dispositivo en la programación CUDA. Explica que los dispositivos CUDA tienen recursos de ejecución organizados en Streaming Multiprocessors (SM) y se pueden asignar múltiples bloques de subprocesos a cada SM. El documento menciona que cada dispositivo CUDA establece un límite en la cantidad de bloques que se pueden asignar a cada SM. Si hay escasez de recursos necesarios para la ejecución simultánea de bloques, el tiempo de ejecución de CUDA reduce automáticamente la cantidad de bloques asignados a cada SM. El documento también menciona que la cantidad de bloques que se pueden ejecutar activamente en un dispositivo CUDA es limitada. El sistema de ejecución mantiene una lista de bloques que deben ejecutarse y asigna nuevos bloques a los SM a medida que los bloques previamente asignados completan la ejecución. Además, el documento analiza las limitaciones en la cantidad de subprocesos que los SM pueden rastrear y programar simultáneamente. Explica que se necesitan recursos de hardware para que los SM mantengan los índices de subprocesos y bloques y realicen un seguimiento de su estado de ejecución. El documento proporciona un ejemplo en el que se asignan tres bloques de subprocesos a cada SM. También menciona que la cantidad máxima de subprocesos permitidos en un bloque puede variar según el dispositivo, y la cantidad de SM y la frecuencia de reloj del dispositivo proporcionan una indicación de su capacidad de ejecución de hardware. Finalmente, el documento explica cómo consultar propiedades del dispositivo, como la cantidad máxima de subprocesos permitidos en un bloque, la cantidad de SM en el dispositivo y la frecuencia de reloj del dispositivo.

## 3.6 Querying Device Properties

Aquí se nos explica las  propiedades del dispositivo en la programación CUDA. Explica que los recursos de ejecución en los dispositivos CUDA están organizados en Streaming Multiprocessors (SM), y cada dispositivo establece un límite en la cantidad de bloques que se pueden asignar a cada SM. El capítulo destaca la importancia de consultar las propiedades del dispositivo, como la cantidad máxima de subprocesos permitidos en un bloque, la cantidad de SM en el dispositivo y la frecuencia de reloj del dispositivo. También menciona que el código host puede encontrar la cantidad máxima de subprocesos permitidos en cada dimensión de un bloque. El capítulo enfatiza la necesidad de considerar estas propiedades al determinar las dimensiones de bloque más apropiadas para un rendimiento óptimo. Además, menciona que el sistema de tiempo de ejecución CUDA mantiene una lista de bloques que deben ejecutarse y asigna nuevos bloques a los SM a medida que los bloques previamente asignados completan la ejecución.

## 3.7 Thread Scheduling and Latency Tolerance

En este capítulo se nos habla de la programación de subprocesos y la tolerancia a la latencia en la programación CUDA. Explica que la programación de subprocesos es un concepto de implementación y debe entenderse en el contexto de implementaciones de hardware específicas. En la mayoría de las implementaciones de CUDA, un bloque asignado a un Streaming Multiprocessor (SM) se divide en 32 unidades de subprocesos llamadas warps. El tamaño de las deformaciones es específico de la implementación y se puede obtener de la variable de consulta del dispositivo.

El capítulo destaca que las deformaciones son la unidad de programación de subprocesos en los SM. Describe cómo los bloques se dividen en urdimbres y cada urdimbre consta de 32 hilos. Se introduce el concepto de tolerancia a la latencia, que se refiere a la capacidad de un programa de ocultar la latencia de los accesos a la memoria u otras operaciones superponiéndo las con otros cálculos. El capítulo explica que al ejecutar múltiples warps simultáneamente, el SM puede cambiar entre warps para ocultar la latencia de los accesos a la memoria y mejorar el rendimiento general.

En general, el Capítulo 3.7 proporciona información sobre el mecanismo de programación de subprocesos en la programación CUDA y la importancia de la tolerancia a la latencia para una ejecución paralela eficiente.

## Conclusión

Me quedo claro que controlar la ejecución paralela en la programación CUDA implica comprender la organización de los subprocesos, coordinar las actividades de los subprocesos mediante la sincronización, considerar la asignación y ocupación de recursos, optimizar la programación de los subprocesos y tolerar la latencia de la memoria. Dominar estos conceptos y técnicas permite el desarrollo de aplicaciones paralelas de alto rendimiento en CUDA.
