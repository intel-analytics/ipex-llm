/* build with: gcc -fopenmp openmp-test.c -o openmp-test */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int v[10];

int main(void) {
    /* We need to limit the number of threads, otherwise OpenMP spawns as many threads as available
     * logical cores on the machine, which may exceed the TCS count specified in the manifest.
     */
#pragma omp parallel for num_threads(10)
    for (int i = 0; i < 10; i++) {
        v[i] = i;
    }

    printf("first: %d, last: %d\n", v[0], v[9]);
    return 0;
}
