#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "futex.h"

#define THREADS 8
static int myfutex = 0;

static int futex(int* uaddr, int futex_op, int val, const struct timespec* timeout, int* uaddr2,
                 int val3) {
    return syscall(SYS_futex, uaddr, futex_op, val, timeout, uaddr2, val3);
}

static void* thread_function(void* argument) {
    int* ptr = (int*)argument;
    long rv;

    // Sleep on the futex
    rv = futex(&myfutex, FUTEX_WAIT_BITSET, 0, NULL, NULL, *ptr);

    return (void*)rv;
}

int main(int argc, const char** argv) {
    pthread_t thread[THREADS];
    int varx[THREADS];

    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    static_assert(THREADS < sizeof(int) * 8 - 1, "Left shift in the loop below would overflow!");

    for (int i = 0; i < THREADS; i++) {
        varx[i] = (1 << i);

        int ret = pthread_create(&thread[i], NULL, &thread_function, &varx[i]);
        if (ret) {
            errno = ret;
            perror("pthread_create");
            return 1;
        }
    }

    printf("Waking up kiddos\n");
    /* Wake in reverse order */
    for (int i = THREADS - 1; i >= 0; i--) {
        int rv;
        int var = (1 << i);

        // Wake up the thread
        do {
            rv = futex(&myfutex, FUTEX_WAKE_BITSET, 1, NULL, NULL, var);
            if (rv == 0) {
                // the thread of thread_function() may not reach
                // futex(FUTEX_WAIT_BITSET) yet.
                // Wait for the thread to sleep and try again.
                // Since synchronization primitive, futex, is being tested,
                // futex can't be used here. resort to use sleep.
                sleep(1);
            }
        } while (rv == 0);
        printf("FUTEX_WAKE_BITSET i = %d rv = %d (expected: 1)\n", i, rv);
        if (rv != 1) {
            return 1;
        }

        // Wait for the child thread to exit
        intptr_t retval = 0;
        int ret = pthread_join(thread[i], (void**)&retval);
        if (ret) {
            errno = ret;
            perror("pthread_join");
            return 1;
        }
        if (retval != 0) {
            printf("Thread %d returned %zd (%s)\n", i, retval, strerror(retval));
            return 1;
        }

        if (i != 0) {
            errno = 0;
            ret = pthread_tryjoin_np(thread[0], (void**)&retval);
            if (ret != EBUSY) {
                printf("Unexpectedly pthread_tryjoin_np returned: %d (%s)\n", ret, strerror(ret));
                return 1;
            }
        }
    }

    printf("Woke all kiddos\n");

    return 0;
}
