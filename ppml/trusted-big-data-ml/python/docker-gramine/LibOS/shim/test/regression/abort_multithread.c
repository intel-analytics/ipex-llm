#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

/* Test if abort() in a child thread kills the whole process. This should report 134 (128 + 6 where
 * 6 is SIGABRT) as its return code. */

static void* thread_abort(void* arg) {
    abort();
    return NULL; /* not reached */
}

int main(int argc, char* arvg[]) {
    pthread_t thread;
    pthread_create(&thread, NULL, thread_abort, NULL);
    pthread_join(thread, NULL);

    printf("Main thread returns successfully (must not happen)\n");
    return 0;
}
