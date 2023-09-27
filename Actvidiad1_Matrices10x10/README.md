# Matrices 10x10 en C
El proyecto es una aplicacion de consola en C que se encarga de hacer una operacion de multiplicacion de 2 matrices de 10x10
El proyecto solo nesecita del codigo en C para funcionar. 
Este solo necista de ser ejecutado y procedera a imprimir en la cosola 3 matrices de 10x10
Las primeras 2 son matrices con numeros aletorios y la tercera es el resultado de la multiplicacion de las primeras 2 

### Marco teorico

Una matriz es una estructura de datos que organiza números, letras u otros valores en filas y columnas, formando una cuadrícula. Cada elemento dentro de la matriz tiene una ubicación específica, identificada por su fila y columna.
La multiplicación de matrices consiste en combinar linealmente dos o más matrices mediante la adición de sus elementos dependiendo de su situación dentro de la matriz origen respetando el orden de los factores. 
Ejemplo de una multiplicacion de matriz 

![image](https://github.com/Panchoneit0r/Programacion_Paralela_JCGM/assets/100960796/cb973138-1845-4cf2-b744-26590f6d319c)

## Codigo

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    srand(time(NULL));
    
    int matrizA[10][10];
    int matrizB[10][10];
    int matrizAB[10][10];
    //Declaramos 3 matrices, 2 que seran la base y la tercera que sera el resultado de la multiplicacion de las 2 primeras

    printf("Matriz A\n");
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {

            matrizA[i][j] = rand() % 10;

            printf( "%d ", matrizA[i][j] );
        }
        printf("\n");
    }
    //For que genera la matriz A con valores random del 1 al 9, tambien la imrpime en consola 

    printf("Matriz B\n");

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {

            matrizB[i][j] = rand() % 10;

            printf( "%d ", matrizB[i][j] );
        }
        printf("\n");
    }
     //For que genera la matriz B con valores random del 1 al 9, tambien la imrpime en consola 


    printf("Matriz AB resultado de la multiplicacion de las 2 anteriores\n");

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            for (int k = 0; k < 10; ++k) {
                matrizAB[i][j] += matrizA[i][k] * matrizB[k][j];
            }
            printf( "%d ", matrizAB[i][j] );
        }
        printf("\n");
    }
     //For que genera la operacion de multiplicacion de A y B. La imprime en consola

    return 0;
}
```
# Análisis de complejidad temporal

## Generación de la matriz A y B:

Hay dos bucles anidados (for) que recorren la matriz de tamaño 10x10.
En cada iteración, se genera un número aleatorio y se asigna a la matriz.
Cada bucle realiza 10x10 = 100 iteraciones.
La complejidad temporal de esta parte es O(100), que se simplifica a O(1) porque el número de iteraciones es constante y no depende del tamaño de la matriz.

## Multiplicación de matrices (matrizAB):

Hay tres bucles anidados (for) que recorren la matriz de tamaño 10x10.
El bucle más interno realiza multiplicaciones y sumas.
Cada bucle externo realiza 10 iteraciones.
La complejidad temporal de esta parte es O(10 * 10 * 10) = O(1000).

# Conclusion
En resumen, la complejidad temporal total del código es O(1) para la generación de las matrices A y B, y O(1000) para la multiplicación de las matrices. Dado que la multiplicación de matrices tiene una complejidad dominante, podemos decir que la complejidad temporal general del código es O(1000), que se puede simplificar como O(n^3), donde n es el tamaño de las matrices (en este caso, n = 10).

Entonces, la complejidad temporal de este código es cúbica en relación al tamaño de las matrices. Esto significa que el tiempo de ejecución aumentará significativamente a medida que aumente el tamaño de las matrices.
