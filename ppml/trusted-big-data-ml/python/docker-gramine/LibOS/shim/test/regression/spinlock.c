/* Poor man's spinlock test */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define USE_STDLIB
#include "spinlock.h"

#define TEST_TIMES 1000

static spinlock_t guard = INIT_SPINLOCK_UNLOCKED;
static spinlock_t go    = INIT_SPINLOCK_UNLOCKED;
static volatile int x = 0;

static int set_expect(int s, int e) {
    int ret = 0;

    spinlock_lock(&guard);
    spinlock_unlock(&go);
    if (x != e) {
        ret = 1;
        goto out;
    }
    x = s;
out:
    spinlock_unlock(&guard);
    return ret;
}

static void* thr(void* unused) {
    spinlock_lock(&go);
    if (set_expect(0, -1)) {
        return (void*)1;
    }
    return (void*)0;
}

static void do_test(void) {
    pthread_t th;
    int ret_val = 0;

    spinlock_lock(&go);

    pthread_create(&th, NULL, thr, NULL);

    if (set_expect(-1, 0)) {
        puts("Test failed!");
        exit(1);
    }

    pthread_join(th, (void**)&ret_val);
    if (ret_val) {
        puts("Test failed!");
        exit(1);
    }
}

int main(void) {
    setbuf(stdout, NULL);

    for (int i = 0; i < TEST_TIMES; ++i) {
        do_test();
    }

    puts("Test successful!");
    return 0;
}
