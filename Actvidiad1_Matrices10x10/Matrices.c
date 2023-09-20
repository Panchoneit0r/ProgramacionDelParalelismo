#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    srand(time(NULL));
    int matrizA[10][10];
    int matrizB[10][10];
    int matrizAB[10][10];

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {

            matrizA[i][j] = rand() % 10;

            printf( "%d ", matrizA[i][j] );
        }
        printf("\n");
    }

    printf("\n");

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {

            matrizB[i][j] = rand() % 10;

            printf( "%d ", matrizB[i][j] );
        }
        printf("\n");
    }

    printf("\n");

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            for (int k = 0; k < 10; ++k) {
                matrizAB[i][j] += matrizA[i][k] * matrizB[k][j];
            }
            printf( "%d ", matrizAB[i][j] );
        }
        printf("\n");
    }
    return 0;
}