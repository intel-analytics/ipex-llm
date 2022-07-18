#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define THREAD_NUM      64
#define CONC_THREAD_NUM 4

atomic_int counter = 0;

static void* inc(void* arg) {
    counter++;
    return NULL;
}

int main(int argc, char** argv) {
    int ret = 0;
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    for (int i = 0; i < THREAD_NUM; i++) {
        pthread_t thread[CONC_THREAD_NUM];

        /* create several threads running in parallel */
        for (int j = 0; j < CONC_THREAD_NUM; j++) {
            int x = pthread_create(&thread[j], NULL, inc, NULL);
            if (x != 0) {
                printf("pthread_create: %d\n", x);
                ret = 1;
                goto out;
            }
        }

        /* join threads and continue with the next batch */
        for (int j = 0; j < CONC_THREAD_NUM; j++) {
            int x = pthread_join(thread[j], NULL);
            if (x != 0) {
                printf("pthread_join: %d\n", x);
                ret = 1;
            }
        }
    }

out:
    printf("%d Threads Created\n", counter);
    return ret;
}
