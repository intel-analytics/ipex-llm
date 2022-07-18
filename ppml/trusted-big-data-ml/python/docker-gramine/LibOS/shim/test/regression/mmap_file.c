#define _GNU_SOURCE
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static const char* message;

static void SIGBUS_handler(int sig) {
    puts(message);
    exit(0);
}

int main(int argc, const char** argv) {
    int rv;

    FILE* fp = fopen("testfile", "w+");
    if (!fp) {
        perror("fopen");
        return 1;
    }

    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size < 0) {
        perror("sysconf");
        return 1;
    }
    long quarter_page = page_size / 4;

    rv = ftruncate(fileno(fp), quarter_page);
    if (rv) {
        perror("ftruncate");
        return 1;
    }

    volatile unsigned char* a =
        mmap(NULL, page_size * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE, fileno(fp), 0);
    if (a == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    a[quarter_page - 1] = 0xff;
    a[page_size - 1] = 0xff;

    __asm__ volatile("nop" ::: "memory");

    int pid = fork();
    if (pid == -1) {
        perror("fork");
        return 1;
    }
    if (pid != 0) {
        rv = waitpid(pid, NULL, 0);
        if (rv == -1) {
            perror("waitpid");
            return 1;
        }
    }

    __asm__ volatile("nop" ::: "memory");

    a[0] = 0xff;
    printf(pid == 0 ? "mmap test 1 passed\n" : "mmap test 6 passed\n");
    a[quarter_page] = 0xff;
    printf(pid == 0 ? "mmap test 2 passed\n" : "mmap test 7 passed\n");

    __asm__ volatile("nop" ::: "memory");

    if (pid == 0) {
        if (a[quarter_page - 1] == 0xff)
            printf("mmap test 3 passed\n");
        if (a[page_size - 1] == 0xff)
            printf("mmap test 4 passed\n");
    }

    __asm__ volatile("nop" ::: "memory");

    if (signal(SIGBUS, SIGBUS_handler) == SIG_ERR) {
        perror("signal");
        return 1;
    }

    message = pid == 0 ? "mmap test 5 passed\n" : "mmap test 8 passed\n";
    /* need a barrier to assign message before SIGBUS due to a[page_size] */
    __asm__ volatile("nop" ::: "memory");
    a[page_size] = 0xff;

    if (signal(SIGBUS, SIG_DFL) == SIG_ERR) {
        perror("signal");
        return 1;
    }

    return 0;
}
