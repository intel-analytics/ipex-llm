#define _GNU_SOURCE
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "futex.h"

static int futex(int* uaddr, int futex_op, int val, const struct timespec* timeout, int* uaddr2,
                 int val3) {
    return syscall(SYS_futex, uaddr, futex_op, val, timeout, uaddr2, val3);
}

static int futex_wait(int* uaddr, int val, const struct timespec* timeout) {
    return futex(uaddr, FUTEX_WAIT | FUTEX_PRIVATE_FLAG, val, timeout, NULL, 0);
}

static int futex_wake(int* uaddr, int to_wake) {
    return futex(uaddr, FUTEX_WAKE | FUTEX_PRIVATE_FLAG, to_wake, NULL, NULL, 0);
}

static int futex_cmp_requeue(int* uaddr, int val, int to_wake, int* uaddr2,
                             unsigned int max_requeue) {
    return futex(uaddr, FUTEX_CMP_REQUEUE | FUTEX_PRIVATE_FLAG, to_wake,
                 (struct timespec*)(unsigned long)max_requeue, uaddr2, val);
}

static void fail(const char* msg, int x) {
    printf("%s failed with %d (%s)\n", msg, x, strerror(x));
    exit(1);
}

static void check(int x) {
    if (x) {
        fail("pthread", x);
    }
}

static void store(int* ptr, int val) {
    __atomic_store_n(ptr, val, __ATOMIC_SEQ_CST);
}
static int load(int* ptr) {
    return __atomic_load_n(ptr, __ATOMIC_SEQ_CST);
}

static int futex1 = 0;
static int futex2 = 0;

#define THREADS         9
#define THREADS_WAKE    2
#define THREADS_REQUEUE 3

static int thread_state[THREADS] = {0};

static void* thread_func(void* arg) {
    unsigned long i = (unsigned long)arg;

    store(&thread_state[i], 1);

    int ret = futex_wait(&futex1, futex1, NULL);
    if (ret) {
        printf("futex_wait in thread %lu returned %d (%s)\n", i, ret, strerror(ret));
        // skip setting state below
        return arg;
    }

    store(&thread_state[i], 2);
    return arg;
}

int main(void) {
    pthread_t th[THREADS];
    unsigned long i;
    int ret;

    for (i = 0; i < THREADS; i++) {
        check(pthread_create(&th[i], NULL, thread_func, (void*)i));
    }

    // wait for all threads
    for (i = 0; i < THREADS; i++) {
        while (load(&thread_state[i]) != 1) {
            usleep(1000u);
        }
    }
    // and let them sleep on futex
    usleep(100000u);

    ret = futex_cmp_requeue(&futex1, futex1, THREADS_WAKE, &futex2, THREADS_REQUEUE);
    if (ret < 0) {
        fail("futex_cmp_requeue", errno);
    }
    if (ret != THREADS_WAKE + THREADS_REQUEUE) {
        printf("futex_cmp_requeue returned %d instead of %d!\n", ret,
               THREADS_WAKE + THREADS_REQUEUE);
        return 1;
    }

    // let the woken thread(s) end
    usleep(100000u);

    ret = 0;
    for (i = 0; i < THREADS; i++) {
        if (load(&thread_state[i]) == 2) {
            ret++;
            check(pthread_join(th[i], NULL));
            store(&thread_state[i], 3);
        }
    }
    if (ret != THREADS_WAKE) {
        printf("futex_cmp_requeue woke-up %d threads instead of %d!\n", ret, THREADS_WAKE);
        return 1;
    }

    ret = futex_wake(&futex1, INT_MAX);
    if (ret < 0) {
        fail("futex_wake(&futex1)", errno);
    }
    if (ret != (THREADS - THREADS_WAKE - THREADS_REQUEUE)) {
        printf("futex_wake on futex1 woke-up %d threads instead of %d!\n", ret,
               THREADS - THREADS_WAKE - THREADS_REQUEUE);
        return 1;
    }

    ret = futex_wake(&futex2, INT_MAX);
    if (ret < 0) {
        fail("futex_wake(&futex2)", errno);
    }
    if (ret != THREADS_REQUEUE) {
        printf("futex_wake on futex2 woke-up %d threads instead of %d!\n", ret, THREADS_REQUEUE);
        return 1;
    }

    for (i = 0; i < THREADS; i++) {
        if (load(&thread_state[i]) != 3) {
            check(pthread_join(th[i], NULL));
        }
    }

    puts("Test successful!");
    return 0;
}
