#define _GNU_SOURCE
#include <err.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdnoreturn.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define N_CHILDREN 3

static noreturn void do_child(void) {
    while (1) {
        struct timespec ts = {
            .tv_sec = 1,
        };
        /* Cannot use `sleep` here as we also use `alarm`. */
        if (nanosleep(&ts, NULL) < 0) {
            err(1, "nanosleep");
        }
    }
}

int main(void) {
    pid_t children[N_CHILDREN] = { 0 };

    for (size_t i = 0; i < N_CHILDREN; i++) {
        children[i] = fork();
        if (children[i] < 0) {
            err(1, "fork");
        } else if (children[i] == 0) {
            alarm(60 * N_CHILDREN);
            do_child();
        }
    }

    if (kill(-1, SIGTERM) < 0) {
        err(1, "kill");
    }

    for (size_t i = 0; i < N_CHILDREN; i++) {
        int status = 0;
        if (waitpid(children[i], &status, 0) < 0) {
            err(1, "waitpid");
        }
        if (!WIFSIGNALED(status)) {
            errx(1, "child %d not killed (%d)", children[i], status);
        }
        if (WTERMSIG(status) != SIGTERM) {
            errx(1, "child %d killed by a wrong signal (%d)", children[i], WTERMSIG(status));
        }
    }

    puts("TEST OK");
    return 0;
}
