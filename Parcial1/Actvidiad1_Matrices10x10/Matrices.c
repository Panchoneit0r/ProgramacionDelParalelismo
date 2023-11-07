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
