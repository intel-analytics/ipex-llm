#define _XOPEN_SOURCE 700
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, const char** argv, const char** envp) {
    int newfd = dup(1);
    char fd_argv[4];
    snprintf(fd_argv, 4, "%d", newfd);
    char* const new_argv[] = {(char*)"./exec_victim", fd_argv, NULL};

    setenv("IN_EXECVE", "1", 1);

    execv(new_argv[0], new_argv);
    return 1;
}
