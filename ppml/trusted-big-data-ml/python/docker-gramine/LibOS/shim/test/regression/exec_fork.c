#define _XOPEN_SOURCE 700
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static void child_handler(int sig) {
    (void)sig;
    /* should never be printed because child doesn't inherit this handler */
    printf("Handled SIGCHLD\n");
    fflush(stdout);
}

int main(int argc, const char** argv, const char** envp) {
    if (argc > 1) {
        /* execv'ed child: do dummy fork, wait, and exit */
        pid_t child_pid = fork();
        if (child_pid == 0) {
            /* child just exits */
            return 0;
        } else if (child_pid > 0) {
            /* parent waits for child termination */
            int status;
            pid_t pid = wait(&status);
            if (pid != child_pid) {
                perror("wait failed");
                return 1;
            }
            if (WIFEXITED(status))
                printf("child exited with status: %d\n", WEXITSTATUS(status));
        } else {
            /* error */
            perror("fork failed");
            return 1;
        }

        puts("test completed successfully");
        return 0;
    }

    /* set signal handler for SIGCHLD signal */
    struct sigaction sa = {0};
    sa.sa_handler = child_handler;
    int ret = sigaction(SIGCHLD, &sa, NULL);
    if (ret < 0) {
        perror("sigaction error");
        return 1;
    }

    printf("Set up handler for SIGCHLD\n");
    fflush(stdout);

    /* SIGCHLD signal handler must *not* be inherited by execv'ed child */
    char* const new_argv[] = {(char*)argv[0], (char*)"dummy", NULL};
    execv(new_argv[0], new_argv);

    perror("execv failed");
    return 1;
}
