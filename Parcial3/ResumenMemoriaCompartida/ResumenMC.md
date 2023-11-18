# Resumen Memoria compartida  
Resumen del libro de clase de la página 216 a 232

## CHECKING THE DATA LAYOUT OF SHARED MEMORY
En esta parte, el documento examina cómo está diseñada la memoria compartida y cómo se puede usar de manera eficiente. Explica que la memoria compartida se puede emplear para almacenar en caché datos globales con dimensiones cuadradas, lo cual facilita calcular compensaciones de memoria 1D a partir de índices de subprocesos 2D. El documento muestra una ilustración de un mosaico de memoria compartida con 32 elementos en cada dimensión, almacenados en orden de fila principal. Asimismo, menciona la importancia de considerar cómo se asignan los subprocesos a los bancos de memoria compartida.

### Square Shared Memory:
La memoria compartida cuadrada es una técnica que aprovecha la naturaleza bidimensional de los bloques de hilos en GPU para almacenar datos globales en una matriz cuadrada en memoria compartida.
Cada elemento de la matriz corresponde a un hilo del bloque. De esta forma, los hilos vecinos pueden acceder fácilmente a elementos adyacentes en la matriz, tanto en la dimensión X como en la Y.
Esto permite un acceso de memoria más eficiente, ya que la memoria compartida es mucho más rápida que la memoria global. Además, al mapear los datos globales a una matriz cuadrada, se simplifica el cálculo de desplazamientos de memoria a partir de los índices de hilo 2D.
La memoria compartida está dividida en 32 bancos del mismo tamaño. Los datos se almacenan por filas para minimizar conflictos entre hilos. También hay una correspondencia directa entre elementos de 4 bytes en los datos y bancos de memoria compartida.

**Acceder en orden de fila versus columna**
Cuando accedes a una matriz cuadrada en memoria compartida por orden de fila, la longitud de la dimensión más interna se fija al mismo que la dimensión interna del bloque de hilos 2D. Esto significa que los hilos cercanos en el mismo warp accederán a celdas vecinas a lo largo de la dimensión interna de la matriz, dando mejor rendimiento y menos conflictos entre bancos.
Por otro lado, acceder en orden de columna fija la longitud interna al mismo de la dimensión externa del bloque. Esto hace que los hilos cercanos accedan a lo largo de la dimensión externa, potencialmente causando más conflictos de banco y peor rendimiento.

**Escribir por filas y leer por columnas**
Al escribir por filas, cada hilo escribe su índice global en ubicaciones consecutivas a lo largo de la dimensión interna. Esto permite accesos secuenciales a memoria y evita conflictos.
Leer por columnas hace que cada hilo lea valores consecutivos a lo largo de la dimensión externa. Esto puede causar conflictos de banco ya que hilos cercanos acceden a celdas vecinas en la dimensión externa.

**Memoria Compartida Dinámica**
La memoria compartida dinámica es la que se declara en tiempo de ejecución. En memoria cuadrada compartida, se puede hacer un nuevo núcleo que declare la memoria de forma dinámica y haga operaciones como escribir por columnas y leer por filas. Esto da más flexibilidad en la asignación de memoria y puede ser útil en ciertos casos.

**Rellenado de memoria declarada estáticamente**
En memoria cuadrada compartida, se usa rellenado para evitar conflictos de bancos. Para rellenar memoria declarada estáticamente, simplemente se agrega una columna al mapeo 2D. Así los elementos de columna se distribuyen en diferentes bancos, resultando en lecturas y escrituras sin conflicto.

**Rellenado de memoria declarada dinámicamente**
Para rellenar memoria dinámica en cuadrada compartida, se omite un espacio de relleno para cada fila al convertir índices de hilos 2D a 1D. Esto ayuda a resolver conflictos de bancos y mejora rendimiento al reducir transacciones compartidas.

**Comparación de rendimiento de núcleos cuadrados compartidos**
Comparando núcleos de memoria cuadrada compartida, los que usan rellenado ganan rendimiento por la reducción de conflictos de bancos. Los núcleos con memoria dinámica tienen una pequeña sobrecarga extra. Analizando tiempos transcurridos y transacciones de memoria, se puede determinar el impacto de distintas técnicas de optimización en el rendimiento.

### Rectangular Shared Memory
La memoria compartida rectangular es un tipo de memoria rápida en la GPU que deja que los hilos dentro de un mismo bloque trabajen juntos y usen menos ancho de banda de memoria global. Es una caché que se puede controlar por programa y se puede usar para comunicación de hilos en el bloque, almacenar en caché datos de memoria global y mejorar patrones de acceso a memoria global. 
Las matrices rectangulares en memoria compartida se pueden declarar estática o dinámicamente y se puede acceder a ellas por orden de filas o columnas.
- La memoria compartida rectangular es una memoria rápida en chip en las GPUs.
- Permite a hilos en un bloque cooperar y reducir uso de ancho de banda de memoria global. 
- Se puede usar para comunicación de hilos, almacenar en caché datos globales y mejorar acceso a memoria global.
- Las matrices rectangulares en memoria compartida pueden declararse estática o dinámicamente.
- Se puede acceder a la memoria rectangular por orden de filas o columnas.

**Acceder por filas vs columnas**
En memoria rectangular compartida, donde filas y columnas de una matriz no son iguales, acceder por filas o columnas requiere consideraciones diferentes. 
Acceder por filas fija la dimensión interna al mismo de la dimensión interna del bloque 2D. Esto permite a hilos cercanos en un warp acceder a celdas vecinas a lo largo de la dimensión interna, similar al caso cuadrado.
Acceder por columnas fija la interna al mismo de la externa del bloque 2D. Esto hace que hilos cercanos accedan a lo largo de la dimensión externa, potencialmente causando más conflictos de banco y peor rendimiento.

**Escribir por filas y leer por columnas**
Escribir por filas aún significa que cada hilo escribe su índice global en ubicaciones consecutivas a lo largo de la dimensión interna, ayudando a evitar conflictos.
Similarmente, leer por columnas significa que cada hilo lee valores consecutivos a lo largo de la externa. Sin embargo, debido a dimensiones desiguales, aún puede haber conflictos de banco si hilos cercanos acceden a celdas vecinas en la externa.

**Memoria compartida declarada dinámicamente**
En rectangular compartida, como las dimensiones de la matriz no son iguales, calcular desplazamientos 1D de índices 2D es más complejo. Se deben reimplementar los núcleos y recalcular índices de acceso basado en las dimensiones. Simplemente cambiar coordenadas de hilos para una matriz rectangular resultaría en errores de acceso a memoria.

**Rellenado de memoria declarada estáticamente**
En rectangular, el rellenado también se usa para evitar conflictos de bancos. Se agrega una columna al mapeo 2D, distribuyendo elementos de columna en diferentes bancos para lecturas y escrituras sin conflicto.

**Rellenado de memoria declarada dinámicamente**
El rellenado dinámico en rectangular es más complejo. Similar a cuadrada, se omite espacio de relleno por fila en la conversión de índices 2D a 1D. Pero la cantidad de elementos de relleno necesarios por fila depende del tamaño 2D, requiriendo más pruebas para determinar la cantidad adecuada.

**Comparación de rendimiento de núcleos rectangulares**
Los núcleos con rellenado y desenrollado muestran mejor rendimiento por la reducción de conflictos de banco y mayor paralelismo.

# Conclusión
En este capítulo, el documento analiza el diseño de la memoria compartida y su impacto en el rendimiento. Explica cómo diseñar un kernel eficiente para evitar conflictos bancarios y utilizar plenamente los beneficios de la memoria compartida. El documento proporciona ejemplos de acceso a la memoria compartida de diferentes maneras y compara su rendimiento. También introduce el concepto de memoria dinámica compartida y su implementación. El documento concluye analizando el rendimiento de diferentes núcleos de memoria compartida y la importancia de considerar la memoria compartida, la memoria constante, el caché de solo lectura y las instrucciones aleatorias al optimizar la GPU.

# Referencia:
S. Cook, CUDA Programming: A Developer's Guide to Parallel Computing with GPUs, John Wiley & Sons, Inc., 2013.
