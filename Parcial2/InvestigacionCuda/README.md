# Investigación CUDA Cores, Threads, Blocks and Grids

## CUDA Cores
CUDA Cores son procesadores paralelos que se encuentran dentro de las unidades de procesamiento gráfico (GPU) de Nvidia. Dichas unidades de procesamiento
esta hechas principalmente para realizar calculcos graficos y gracias a que esatn diseñados para programacion paralela son capaces de realizar varias operaciones al mismo tiempo.

## Threads 
Un Thread o hilo de ejecución es una secuencia de instrucciones que puede ser ejecutada por un procesador. Los threads permiten que un programa realice varias tareas al mismo tiempo, aprovechando los recursos del sistema. Cada thread tiene un contexto que incluye los registros y la pila del procesador que utiliza.

## Blocks
Los Blocks o bloques de código son una estructura de código fuente que se agrupa y se utilizan en la programación paralela para dividir un problema en partes más pequeñas que se pueden resolver simultáneamente, lo que permite un mayor rendimiento y una mayor velocidad de procesamiento. En resumida cuentas los bloques son conjutos de Threads

## Grids
Grid o red de computadoras, es un conjunto de recursos de computación distribuidos que trabajan juntos para resolver problemas complejos que requieren una gran cantidad de recursos computacionales. El Grid es una estructura tridimensional compuesta por muchos bloques siendo este el que define la cantidad de bloques utilizados 

## Conclusion
Considero que es importante entender estos conceptos pues todos estan muy relacionados entre si, siendo base en la programacion paralela 

### Bibliografia 

A. Aller, “Qué son los Nvidia CUDA Cores y cuál es su importancia,” Profesional Review, 9-oct-2018. [En línea]. Disponible en: https://www.profesionalrevie

Eduardo Ismael García Pérez, “Threads y procesos en Python,” CódigoFacilito, 22-may-2017. [En línea]. Disponible en: https://codigofacilito.com/articulos/threads-procesos. [Accedido el 6-oct-2023].

Saturn Cloud, “Understanding CUDA Grid Dimensions, Block Dimensions, and Thread Organization,” Saturn Cloud Blog, 10-jul-2023. [En línea]. Disponible en: https://saturncloud.io/blog/understanding-cuda-grid-dimensions-block-dimensions-and-thread-organization/. [Accedido el 6-oct-2023].
