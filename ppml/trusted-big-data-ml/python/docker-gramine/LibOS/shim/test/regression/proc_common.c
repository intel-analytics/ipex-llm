/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

#define _GNU_SOURCE
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

#include "dump.h"

static void* fn(void* arg) {
    /* not to consume CPU, each thread simply sleeps */
    while (true)
        sleep(10);

    return NULL; /* not reached */
}

int main(int argc, char** argv) {
    int ret;

    /* create a pipe, so that we see it in `/proc/[pid]/fd` */
    int pipefd[2];
    ret = pipe(pipefd);
    if (ret < 0) {
        perror("pipe");
        return 1;
    }

    /* fork, so that child sees `/proc/[parent-pid]/` */
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return 1;
    }

    if (pid == 0) {
        /* child process: create three threads so we have some info in `/proc/[pid]/task/[tid]`,
         * then dump `/proc` */
        pthread_t thread[3];
        for (int j = 0; j < 3; j++) {
            ret = pthread_create(&thread[j], NULL, fn, NULL);
            if (ret < 0) {
                perror("pthread_create");
                return 1;
            }
        }

        if (dump_path("/proc") < 0)
            return 1;

        /* Currently remote processes are not listed under "/proc", but their metadata is still
         * accessible. */
        char buf[0x20];
        snprintf(buf, sizeof(buf), "/proc/%u", getppid());
        if (dump_path(buf) < 0) {
            return 1;
        }

        return 0;
    }

    /* parent process: wait for child to finish */
    int status;
    ret = waitpid(pid, &status, 0);
    if (ret < 0) {
        perror("waitpid");
        return 1;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "waitpid: got %d\n", status);
        return 1;
    }
    return 0;
}
