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

static int futex_wake_op(int* uaddr1, int to_wake1, int* uaddr2, int to_wake2, int op) {
    return futex(uaddr1, FUTEX_WAKE_OP | FUTEX_PRIVATE_FLAG, to_wake1,
                 (struct timespec*)(unsigned long)to_wake2, uaddr2, op);
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

static int wakeop_arg_extend(int x) {
    if (x >= 0x800) {
        return 0xfffff000 | x;
    }
    return x;
}

static int futex1 = 0;
static int futex2 = 0;

#define THREADS1      4
#define THREADS2      5
#define THREADS_WAKE1 2
#define THREADS_WAKE2 3

#define FUTEX_OP_MAX  5

static int thread_state[THREADS1 + THREADS2] = {0};

static void* thread_func(void* arg) {
    unsigned long i = (unsigned long)arg;
    int ret = -1;

    store(&thread_state[i], 1);

    if (i < THREADS1) {
        ret = futex_wait(&futex1, futex1, NULL);
    } else {
        ret = futex_wait(&futex2, futex2, NULL);
    }
    if (ret != 0) {
        printf("futex_wait in thread %lu returned %d (%s)\n", i, ret, strerror(ret));
        // skip setting state below
        return arg;
    }

    store(&thread_state[i], 2);
    return arg;
}

static int run_futex_wake_op(int orig_val, int encoded_op, int op, int oparg) {
    int ret;
    int newval;
    unsigned long i;
    pthread_t th[THREADS1 + THREADS2];

    /* Clear thread state for each test case */
    memset(thread_state, 0, sizeof(thread_state));

    for (i = 0; i < THREADS1 + THREADS2; i++) {
        check(pthread_create(&th[i], NULL, thread_func, (void*)i));
    }

    // wait for all threads
    for (i = 0; i < THREADS1 + THREADS2; i++) {
        while (load(&thread_state[i]) != 1) {
            sched_yield();
        }
    }
    // and let them sleep on futex
    usleep(100000u);

    store(&futex2, orig_val);
    ret = futex_wake_op(&futex1, THREADS_WAKE1, &futex2, THREADS_WAKE2, encoded_op);
    if (ret < 0) {
        fail("futex_wake_op", errno);
    }

    if (ret != THREADS_WAKE1 + THREADS_WAKE2) {
        printf("futex_wake_op %d returned %d instead of %d!\n", op, ret,
                THREADS_WAKE1 + THREADS_WAKE2);
        return 1;
    }

    newval = load(&futex2);

    /* Sign extend 12-bit argument to 32-bit */
    oparg = wakeop_arg_extend(oparg);
    if ((encoded_op >> 28) & FUTEX_OP_OPARG_SHIFT)
        oparg = 1 << oparg;

    switch (op) {
        case FUTEX_OP_SET:
            if (newval != oparg) {
                printf("FUTEX_OP_SET operation failed: current value: 0x%x, expected: 0x%x\n",
                        newval, oparg);
                return 1;
            }
            break;
        case FUTEX_OP_ADD:
            if (newval != (orig_val + oparg)) {
                printf("FUTEX_OP_ADD operation failed: current value: 0x%x, expected: 0x%x\n",
                        newval, orig_val + oparg);
                return 1;
            }
            break;
        case FUTEX_OP_OR:
            if (newval != (orig_val | oparg)) {
                printf("FUTEX_OP_OR operation failed: current value: 0x%x, expected: 0x%x\n",
                        newval, orig_val | oparg);
                return 1;
            }
            break;
        case FUTEX_OP_ANDN:
            if (newval != (orig_val & ~oparg)) {
                printf("FUTEX_OP_ANDN operation failed: current value: 0x%x, expected: 0x%x\n",
                        newval, orig_val & ~oparg);
                return 1;
            }
            break;
        case FUTEX_OP_XOR:
            if (newval != (orig_val ^ oparg)) {
                printf("FUTEX_OP_XOR operation failed: current value: 0x%x, expected: 0x%x\n",
                        newval, orig_val ^ oparg);
                return 1;
            }
            break;
        default:
            /* Invalid operation */
            return 1;
    }

    // let the woken thread(s) end
    usleep(100000u);

    ret = 0;
    for (i = 0; i < THREADS1; i++) {
        if (load(&thread_state[i]) == 2) {
            ret++;
            check(pthread_join(th[i], NULL));
            store(&thread_state[i], 3);
        }
    }
    if (ret != THREADS_WAKE1) {
        printf("futex_wake_op %d woke-up %d threads on futex1 instead of %d!\n", op, ret,
                THREADS_WAKE1);
        return 1;
    }

    ret = 0;
    for (i = THREADS1; i < THREADS1 + THREADS2; i++) {
        if (load(&thread_state[i]) == 2) {
            ret++;
            check(pthread_join(th[i], NULL));
            store(&thread_state[i], 3);
        }
    }
    if (ret != THREADS_WAKE2) {
        printf("futex_wake_op %d woke-up %d threads on futex2 instead of %d!\n", op, ret,
                THREADS_WAKE2);
        return 1;
    }

    ret = futex_wake(&futex1, INT_MAX);
    if (ret < 0) {
        fail("futex_wake(&futex1)", errno);
    }
    if (ret != (THREADS1 - THREADS_WAKE1)) {
        printf("futex_wake on futex1 woke-up %d threads instead of %d!\n", ret,
               THREADS1 - THREADS_WAKE1);
        return 1;
    }

    ret = futex_wake(&futex2, INT_MAX);
    if (ret < 0) {
        fail("futex_wake(&futex2)", errno);
    }
    if (ret != (THREADS2 - THREADS_WAKE2)) {
        printf("futex_wake on futex2 woke-up %d threads instead of %d!\n", ret,
               THREADS2 - THREADS_WAKE2);
        return 1;
    }

    for (i = 0; i < THREADS1 + THREADS2; i++) {
        if (load(&thread_state[i]) != 3) {
            check(pthread_join(th[i], NULL));
        }
    }

    return 0;
}

int main(void) {
    int orig_val;
    int oparg;
    int cmparg;
    int encoded_op;

    for (int op = 0; op < FUTEX_OP_MAX; op++) {
        switch (op) {
            case FUTEX_OP_SET:
                orig_val = 0x123846;
                oparg = 0xc73;
                cmparg = 0x1; /* Any 12-bit wide value such that orig_val > cmparg. */
                encoded_op = FUTEX_OP(FUTEX_OP_SET, oparg, FUTEX_OP_CMP_GT, cmparg);
                break;
            case FUTEX_OP_ADD:
                orig_val = -1;
                oparg = 0x1f;
                cmparg = -1; /* Any 12-bit wide value such that orig_val >= cmparg. */
                encoded_op = FUTEX_OP((FUTEX_OP_OPARG_SHIFT | FUTEX_OP_ADD), oparg, FUTEX_OP_CMP_GE,
                                      cmparg);
                break;
            case FUTEX_OP_OR:
                orig_val = 0x123;
                oparg = 0xc73;
                cmparg = 0xabc; /* Any 12-bit wide value such that orig_val != cmparg. */
                encoded_op = FUTEX_OP(FUTEX_OP_OR, oparg, FUTEX_OP_CMP_NE, cmparg);
                break;
            case FUTEX_OP_ANDN:
                orig_val = 0x456;
                oparg = 0xc73;
                cmparg = 0x456; /* Any 12-bit wide value such that orig_val == cmparg. */
                encoded_op = FUTEX_OP(FUTEX_OP_ANDN, oparg, FUTEX_OP_CMP_EQ, cmparg);
                break;
            case FUTEX_OP_XOR:
                orig_val = -2;
                oparg = 0x4;
                cmparg = 0x1; /* Any 12-bit wide value such that orig_val <= cmparg. */
                encoded_op = FUTEX_OP((FUTEX_OP_OPARG_SHIFT | FUTEX_OP_XOR), oparg, FUTEX_OP_CMP_LE,
                                      cmparg);
                break;
            default:
                /* Incorrect case */
                return 1;
        }

        int ret = run_futex_wake_op(orig_val, encoded_op, op, oparg);
        if (ret != 0)
            return 1;
    }

    puts("Test successful!");
    return 0;
}
