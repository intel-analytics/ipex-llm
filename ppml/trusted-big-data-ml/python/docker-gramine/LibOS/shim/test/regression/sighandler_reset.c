#define _GNU_SOURCE
#include <err.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static unsigned count = 0;

static void handler(int signum) {
    __atomic_add_fetch(&count, 1, __ATOMIC_RELEASE);
    printf("Got signal %d\n", signum);
}

int main(void) {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    struct sigaction action = {
        .sa_handler = handler,
        .sa_flags = SA_RESETHAND,
    };

    int ret = sigaction(SIGCHLD, &action, NULL);
    if (ret < 0) {
        err(1, "sigaction failed");
    }

    pid_t pid = fork();
    if (pid < 0) {
        err(1, "fork failed");
    } else if (pid == 0) {
        ret = kill(getppid(), SIGCHLD);
        if (ret < 0) {
            err(1, "kill 1");
        }
        ret = kill(getppid(), SIGCHLD);
        if (ret < 0) {
            err(1, "kill 2");
        }
        return 0;
    }

    int status = 0;
    pid_t wait_ret;
    do {
        wait_ret = wait(&status);
    } while (wait_ret == -1 && errno == EINTR);
    if (wait_ret < 0) {
        err(1, "wait");
    }
    if (wait_ret != pid) {
        errx(1, "unknown child died: %d", wait_ret);
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status)) {
        errx(1, "unexpected exit status: %d", status);
    }

    unsigned tmp_count = __atomic_load_n(&count, __ATOMIC_ACQUIRE);
    printf("Handler was invoked %u time(s).\n", tmp_count);

    if (tmp_count != 1)
        return 1;

    return 0;
}
