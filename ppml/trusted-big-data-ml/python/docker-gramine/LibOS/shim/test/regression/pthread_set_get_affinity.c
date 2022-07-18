/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Corporation */

/*
 * Test to set/get cpu affinity by parent process on behalf of its child threads.
 */

#define _GNU_SOURCE
#include <err.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>

#define min(a, b)               (((a) < (b)) ? (a) : (b))
#define MAIN_THREAD_CNT         1
#define INTERNAL_THREAD_CNT     2
#define MANIFEST_SGX_THREAD_CNT 8 /* corresponds to sgx.thread_num in the manifest template */

/* barrier to synchronize between parent and children */
pthread_barrier_t barrier;

/* Run a busy loop for some iterations, so that we can verify affinity with htop manually */
static void* dowork(void* args) {
    volatile uint64_t iterations = *(uint64_t*)args;

    while (iterations != 0)
        iterations--;

    int ret = pthread_barrier_wait(&barrier);
    if (ret != 0 && ret != PTHREAD_BARRIER_SERIAL_THREAD) {
        errx(EXIT_FAILURE, "Child did not wait on barrier!");
    }
    return NULL;
}

int main(int argc, const char** argv) {
    int ret;
    long numprocs = sysconf(_SC_NPROCESSORS_ONLN);
    if (numprocs < 0) {
        err(EXIT_FAILURE, "Failed to retrieve the number of logical processors!");
    }

    /* If you want to run on all cores then increase sgx.thread_num in the manifest.template and
     * also set MANIFEST_SGX_THREAD_CNT to the same value.
     */
    numprocs = min(numprocs, (MANIFEST_SGX_THREAD_CNT - (INTERNAL_THREAD_CNT + MAIN_THREAD_CNT)));

    /* Affinitize threads to alternate logical processors to do a quick check from htop manually */
    numprocs = (numprocs >= 2) ? numprocs/2 : 1;

    pthread_t* threads = (pthread_t*)malloc(numprocs * sizeof(pthread_t));
    if (!threads) {
         errx(EXIT_FAILURE, "memory allocation failed");
    }

    if (pthread_barrier_init(&barrier, NULL, numprocs + 1)) {
        free(threads);
        errx(EXIT_FAILURE, "pthread barrier init failed");
    }

    cpu_set_t cpus, get_cpus;
    uint64_t iterations = argc > 1 ? atol(argv[1]) : 10000000000;

    /* Validate parent set/get affinity for child */
    for (long i = 0; i < numprocs; i++) {
        CPU_ZERO(&cpus);
        CPU_ZERO(&get_cpus);
        CPU_SET(i*2, &cpus);

        ret = pthread_create(&threads[i], NULL, dowork, (void*)&iterations);
        if (ret != 0) {
            free(threads);
            errx(EXIT_FAILURE, "pthread_create failed!");
        }

        ret = pthread_setaffinity_np(threads[i], sizeof(cpus), &cpus);
        if (ret != 0) {
            free(threads);
            errx(EXIT_FAILURE, "pthread_setaffinity_np failed for child!");
        }

        ret = pthread_getaffinity_np(threads[i], sizeof(get_cpus), &get_cpus);
        if (ret != 0) {
            free(threads);
            errx(EXIT_FAILURE, "pthread_getaffinity_np failed for child!");
        }

        if (!CPU_EQUAL_S(sizeof(cpus), &cpus, &get_cpus)) {
            free(threads);
            errx(EXIT_FAILURE, "get cpuset is not equal to set cpuset on proc: %ld", i);
        }
    }

    /* unblock the child threads */
    ret = pthread_barrier_wait(&barrier);
    if (ret != 0 && ret != PTHREAD_BARRIER_SERIAL_THREAD) {
        free(threads);
        errx(EXIT_FAILURE, "Parent did not wait on barrier!");
    }

    for (int i = 0; i < numprocs; i++) {
        ret = pthread_join(threads[i], NULL);
        if (ret != 0) {
            free(threads);
            errx(EXIT_FAILURE, "pthread_join failed!");
        }
    }

    /* Validating parent set/get affinity for children done. Free resources */
    pthread_barrier_destroy(&barrier);
    free(threads);

    /* Validate parent set/get affinity for itself */
    CPU_ZERO(&cpus);
    CPU_SET(0, &cpus);
    ret = pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus);
    if (ret != 0) {
        errx(EXIT_FAILURE, "pthread_setaffinity_np failed for parent!");
    }

    CPU_ZERO(&get_cpus);
    ret = pthread_getaffinity_np(pthread_self(), sizeof(get_cpus), &get_cpus);
    if (ret != 0) {
        errx(EXIT_FAILURE, "pthread_getaffinity_np failed for parent!");
    }

    if (!CPU_EQUAL_S(sizeof(cpus), &cpus, &get_cpus)) {
        errx(EXIT_FAILURE, "get cpuset is not equal to set cpuset on proc 0");
    }

    /* Negative test case with empty cpumask */
    CPU_ZERO(&cpus);
    ret = pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus);
    if (ret != EINVAL) {
        errx(EXIT_FAILURE, "pthread_setaffinity_np with empty cpumask did not return EINVAL!");
    }

    printf("TEST OK\n");
    return 0;
}
