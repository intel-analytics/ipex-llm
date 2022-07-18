#define _GNU_SOURCE
#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdnoreturn.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static noreturn void child(void) {
    /* Waiting for parent to kill us. */
    while (1) {
        sleep(1);
    }
}

int main(int argc, char* argv[]) {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    if (argc < 1) {
        errx(1, "Invalid argc: %d", argc);
    } else if (argc > 1) {
        child();
    }

    int p[2];

    if (pipe2(p, O_CLOEXEC) < 0) {
        err(1, "pipe2");
    }

    pid_t pid = fork();
    if (pid < 0) {
        err(1, "fork");
    } else if (pid == 0) {
        char* arg[] = {argv[0], (char*)"xxx", NULL};
        execv(argv[0], arg);
        err(1, "execve");
    }

    if (close(p[1]) < 0) {
        err(1, "close");
    }

    long a = 0;
    /* This read will return 0 only if the write end of the pipe is closed (in the child). */
    ssize_t x = read(p[0], &a, sizeof(a));

    if (x < 0) {
        err(1, "read");
    } else if (x > 0) {
        errx(1, "read got unexpected data");
    }

    if (kill(pid, SIGKILL) < 0) {
        err(1, "kill");
    }

    int status = 0;
    if (waitpid(pid, &status, 0) != pid) {
        err(1, "wait");
    }

    if (WIFEXITED(status)) {
        errx(1, "child exited with: %d", WEXITSTATUS(status));
    } else if (WIFSIGNALED(status)) {
        if (WTERMSIG(status) != SIGKILL) {
            errx(1, "child received an unexpected signal: %d", WTERMSIG(status));
        }
    } else {
        errx(1, "this cannot happen");
    }

    puts("TEST OK");
    return 0;
}
