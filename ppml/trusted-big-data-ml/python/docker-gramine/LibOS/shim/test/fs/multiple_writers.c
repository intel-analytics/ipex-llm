/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * This is a test for concurrent writes to the same file descriptor. Depending on the command line,
 * it spawns multiple processes, multiple threads, or both.
 */

#define _XOPEN_SOURCE 700
#include <pthread.h>
#include <sys/wait.h>

#include "common.h"

pthread_barrier_t g_barrier;
const char* g_path;
int g_fd;
int g_proc_id;
int g_n_lines;

static void* writer(void* arg) {
    int thread_id = *(int*)arg;

    pthread_barrier_wait(&g_barrier);

    for (int i = 0; i < g_n_lines; i++) {
        char buf[100];
        size_t size = snprintf(buf, sizeof(buf), "%04d %04d %04d: proc %d thread %d line %d\n",
                               g_proc_id, thread_id, i, g_proc_id, thread_id, i);
        write_fd(g_path, g_fd, buf, size);
    }
    return NULL;
}

static void multiple_writers(const char* path, int n_lines, int n_processes, int n_threads) {
    g_fd = open_output_fd(path, /*rdwr=*/false);
    if (ftruncate(g_fd, 0))
        fatal_error("truncate(%s) failed: %d\n", path, path, errno);

    g_path = path;
    g_n_lines = n_lines;

    g_proc_id = 0;
    for (int i = 1; i < n_processes; i++) {
        int ret = fork();
        if (ret < 0) {
            fatal_error("error on fork: %d\n", errno);
        } else if (ret == 0) {
            g_proc_id = i;
            break;
        }
    }

    pthread_barrier_init(&g_barrier, NULL, n_threads);
    pthread_t threads[n_threads];
    int ids[n_threads];
    for (int i = 1; i < n_threads; i++) {
        ids[i] = i;
        pthread_create(&threads[i], NULL, writer, &ids[i]);
    }

    ids[0] = 0;
    writer(&ids[0]);

    for (int i = 1; i < n_threads; i++)
        pthread_join(threads[i], NULL);

    if (g_proc_id == 0) {
        for (int i = 1; i < n_processes; i++)
            wait(NULL);
    }
    close_fd(path, g_fd);
}

int main(int argc, char* argv[]) {
    if (argc != 5)
        fatal_error("Usage: %s <file_path> <n_lines> <n_processes> <n_threads>\n", argv[0]);

    int n_lines = atoi(argv[2]);
    int n_processes = atoi(argv[3]);
    int n_threads = atoi(argv[4]);

    if (n_lines <= 0)
        fatal_error("wrong number of lines\n");
    if (n_processes <= 0)
        fatal_error("wrong number of processes\n");
    if (n_threads <= 0)
        fatal_error("wrong number of threads\n");

    setup();

    multiple_writers(argv[1], n_lines, n_processes, n_threads);
    return 0;
}
