/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */
/*
 * The main idea here is to test Gramine's internal connections between two processes.
 * The main process forks a child, which in turns forks a grandchild (hence the name "double_fork").
 * Then the intermediate (child process) exits and when parent receives information about that (via
 * `wait` syscall), it notifies grandchild process (using a pipe) that the setup is done. Then
 * the grandchild sends a signal to the main process, which receival should indicate that we have
 * a working "main <-> grandchild" Gramine's internal connection.
 */
#define _GNU_SOURCE
#include <err.h>
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "futex.h"

static pid_t main_pid;

static void do_grandchild(int fd) {
    char c = 0;
    if (read(fd, &c, 1) != 1 || c != 'a') {
        err(1, "grandchild read");
    }

    if (kill(main_pid, SIGALRM) < 0) {
        err(1, "grandchild kill");
    }

    _exit(42);
}

static uint32_t last_sig = 0;
static void handler(int sig) {
    __atomic_store_n(&last_sig, sig, __ATOMIC_RELAXED);
}

int main(void) {
    int fds[2] = { -1, -1 };
    if (pipe(fds) < 0) {
        err(1, "pipe");
    }

    main_pid = getpid();

    pid_t p = fork();
    if (p < 0) {
        err(1, "fork");
    } else if (p == 0) {
        // child
        p = fork();
        if (p < 0) {
            err(1, "fork");
        } else if (p == 0) {
            if (close(fds[1]) < 0) {
                err(1, "close");
            }
            do_grandchild(fds[0]);
            return 1;
        }
        return 0;
    }

    if (close(fds[0]) < 0) {
        err(1, "close");
    }

    int status = 0;
    pid_t x = waitpid(p, &status, 0);
    if (x < 0) {
        err(1, "waitpid");
    } else if (x != p) {
        errx(1, "wrong child pid");
    }

    if (!WIFEXITED(status)) {
        errx(1, "child died in an unknown manner: %d\n", status);
    }
    if (WEXITSTATUS(status) != 0) {
        errx(1, "child returned wrong error code: %d\n", status);
    }

    struct sigaction sa = {
        .sa_handler = handler,
    };
    if (sigaction(SIGALRM, &sa, NULL) < 0) {
        err(1, "sigaction");
    }

    if (write(fds[1], "a", 1) != 1) {
        err(1, "write");
    }

    struct timespec ts = {
        .tv_sec = 10,
    };
    errno = 0;
    long ret = syscall(SYS_futex, &last_sig, FUTEX_WAIT, 0, &ts, NULL, 0);
    if (ret >= 0) {
        errx(1, "unexpected futex success");
    } else if (errno != EAGAIN && errno != EINTR) {
        err(1, "unexpected futex error");
    }

    if (last_sig != SIGALRM) {
        errx(1, "did not receive SIGALRM??");
    }

    puts("TEST OK");
    return 0;
}
