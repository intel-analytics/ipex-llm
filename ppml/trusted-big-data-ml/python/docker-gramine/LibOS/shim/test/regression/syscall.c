#define _GNU_SOURCE
#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>

int main(void) {
    long ret = 0;
#ifdef __x86_64__
    __asm__ volatile (
        "syscall"
        : "=a"(ret)
        : "0"(__NR_getpid)
        : "memory", "cc", "rcx", "r11"
    );
#endif
    if (ret != 1)
        errx(EXIT_FAILURE, "getpid syscall: %ld (expected 1)", ret);

    puts("TEST OK");
    return 0;
}
