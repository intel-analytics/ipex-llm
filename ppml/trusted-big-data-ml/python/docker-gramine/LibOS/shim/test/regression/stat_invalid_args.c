#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char** argv) {
    int r;
    struct stat buf;

    char* goodpath = argv[0];
    char* badpath  = (void*)-1;

    struct stat* goodbuf = &buf;
    struct stat* badbuf  = (void*)-1;

    /* check stat() */
    r = syscall(SYS_stat, badpath, goodbuf);
    if (r == -1 && errno == EFAULT)
        printf("stat(invalid-path-ptr) correctly returned error\n");

    r = syscall(SYS_stat, goodpath, badbuf);
    if (r == -1 && errno == EFAULT)
        printf("stat(invalid-buf-ptr) correctly returned error\n");

    /* check lstat() */
    r = syscall(SYS_lstat, badpath, goodbuf);
    if (r == -1 && errno == EFAULT)
        printf("lstat(invalid-path-ptr) correctly returned error\n");

    r = syscall(SYS_lstat, goodpath, badbuf);
    if (r == -1 && errno == EFAULT)
        printf("lstat(invalid-buf-ptr) correctly returned error\n");

    return 0;
}
