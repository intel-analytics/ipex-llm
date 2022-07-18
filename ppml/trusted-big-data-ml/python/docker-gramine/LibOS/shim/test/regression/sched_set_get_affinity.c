/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Corporation */

/*
 * Test setting/getting cpu affinity on a single processor or multiple processors.
 */

#define _GNU_SOURCE
#include <assert.h>
#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, const char** argv) {
    int ret;
    cpu_set_t cpus, get_cpus;
    long numprocs = sysconf(_SC_NPROCESSORS_ONLN);
    if (numprocs < 0) {
        err(EXIT_FAILURE, "Failed to retrieve the number of logical processors!");
    }

    for (long i = 0; i < numprocs; i++) {
        printf("Testing processor id: %ld\n", i);
        CPU_ZERO(&cpus);
        CPU_ZERO(&get_cpus);
        CPU_SET(i, &cpus);
        ret = sched_setaffinity(0, sizeof(cpus), &cpus);
        if (ret < 0) {
            errx(EXIT_FAILURE, "Failed to set affinity for current thread, core id: %ld", i);
        }
        ret = sched_getaffinity(0, sizeof(get_cpus), &get_cpus);
        if (ret < 0) {
            errx(EXIT_FAILURE, "Failed to get affinity for current thread, core id: %ld", i);
        }
        if (!CPU_EQUAL_S(sizeof(cpus), &cpus, &get_cpus)) {
            errx(EXIT_FAILURE, "The get cpu set is not equal to set on core id: %ld", i);
        }
    }

    if (numprocs >= 2) {
        /* test for multiple cpu affinity */
        CPU_ZERO(&cpus);
        CPU_ZERO(&get_cpus);
        CPU_SET(0, &cpus);
        CPU_SET(1, &cpus);
        ret = sched_setaffinity(0, sizeof(cpus), &cpus);
        if (ret < 0) {
            err(EXIT_FAILURE, "Failed to set multiple affinity for current thread");
        }
        ret = sched_getaffinity(0, sizeof(get_cpus), &get_cpus);
        if (ret < 0) {
            err(EXIT_FAILURE, "Failed to get multiple affinity for current thread");
        }
        if (!CPU_EQUAL_S(sizeof(cpus), &cpus, &get_cpus)) {
            errx(EXIT_FAILURE, "The get cpu set is not equal to set on core id: 0 & 1");
        }
    } else {
        printf("Multiple CPU affinity test skipped since only one core was identified\n");
    }

    printf("TEST OK\n");
    return 0;
}
