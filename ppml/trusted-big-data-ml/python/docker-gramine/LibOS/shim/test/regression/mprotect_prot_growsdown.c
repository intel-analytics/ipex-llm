#define _GNU_SOURCE
#include <err.h>
#include <errno.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

int main(void) {
    errno = 0;
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size == -1 && errno) {
        err(1, "sysconf");
    }

    char* ptr = mmap(NULL, 3 * page_size, PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (ptr == MAP_FAILED) {
        err(1, "mmap");
    }

    int x = mprotect(ptr + page_size, page_size, PROT_READ | PROT_WRITE | PROT_GROWSDOWN);
    if (x >= 0) {
        printf("mprotect succeeded unexpectedly!\n");
        return 1;
    }
    if (errno != EINVAL) {
        printf("Wrong errno value: %d\n", errno);
        return 1;
    }

    if (munmap(ptr, 3 * page_size) < 0) {
        err(1, "munmap");
    }

    ptr = mmap(NULL, 3 * page_size, PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE | MAP_GROWSDOWN, -1, 0);
    if (ptr == MAP_FAILED) {
        err(1, "mmap");
    }

    if (mprotect(ptr + page_size, page_size, PROT_READ | PROT_WRITE | PROT_GROWSDOWN) < 0) {
        err(1, "mprotect");
    }

    *(volatile char*)ptr = 'a';

    if (*(volatile char*)ptr != 'a') {
        printf("Value was not written to memory!\n");
        return 1;
    }

    puts("TEST OK");
    return 0;
}
