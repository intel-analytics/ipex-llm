#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>

int main(int argc, const char** argv, const char** envp) {
    int r;
    char** badptr_argv = (char**)-1;
    char* badptr       = (char*)-1;

    char* bad_argv[]  = {badptr, NULL};
    char* good_argv[] = {(char*)"DUMMY", NULL};

    char* bad_envp[]  = {badptr, NULL};
    char* good_envp[] = {(char*)"DUMMY", NULL};

    r = syscall(SYS_execve, badptr, good_argv, good_envp);
    if (r == -1 && errno == EFAULT)
        printf("execve(invalid-path) correctly returned error\n");

    r = syscall(SYS_execve, argv[0], badptr_argv, good_envp);
    if (r == -1 && errno == EFAULT)
        printf("execve(invalid-argv-ptr) correctly returned error\n");

    r = syscall(SYS_execve, argv[0], good_argv, badptr_argv);
    if (r == -1 && errno == EFAULT)
        printf("execve(invalid-envp-ptr) correctly returned error\n");

    r = syscall(SYS_execve, argv[0], bad_argv, good_envp);
    if (r == -1 && errno == EFAULT)
        printf("execve(invalid-argv) correctly returned error\n");

    r = syscall(SYS_execve, argv[0], good_argv, bad_envp);
    if (r == -1 && errno == EFAULT)
        printf("execve(invalid-envp) correctly returned error\n");

    return 0;
}
