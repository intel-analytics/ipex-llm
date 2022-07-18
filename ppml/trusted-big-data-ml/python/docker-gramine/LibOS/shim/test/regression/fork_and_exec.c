#define _XOPEN_SOURCE 700
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, const char** argv, const char** envp) {
    pid_t child_pid;

    /* duplicate STDOUT into newfd and pass it as exec_victim argument
     * (it will be inherited by exec_victim) */
    int newfd = dup(1);
    if (newfd < 0) {
        perror("dup failed");
        return 1;
    }

    char fd_argv[12];
    snprintf(fd_argv, 12, "%d", newfd);
    char* const new_argv[] = {(char*)"./exec_victim", fd_argv, NULL};

    /* set environment variable to test that it is inherited by exec_victim */
    int ret = setenv("IN_EXECVE", "1", 1);
    if (ret < 0) {
        perror("setenv failed");
        return 1;
    }

    child_pid = fork();

    if (child_pid == 0) {
        /* child performs execve(exec_victim) */
        execv(new_argv[0], new_argv);
        perror("execve failed");
        return 1;
    } else if (child_pid > 0) {
        /* parent waits for child termination */
        int status;
        pid_t pid = wait(&status);
        if (pid < 0) {
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
