#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define TEST_LENGTH  0x10000f000
#define TEST_LENGTH2 0x8000f000

int main(void) {
    FILE* fp = fopen("testfile", "a+");
    if (!fp) {
        perror("fopen");
        return 1;
    }
    int rv = ftruncate(fileno(fp), TEST_LENGTH);
    if (rv) {
        perror("ftruncate");
        return 1;
    } else {
        printf("large_mmap: ftruncate OK\n");
    }

    void* a = mmap(NULL, TEST_LENGTH2, PROT_READ | PROT_WRITE, MAP_PRIVATE, fileno(fp), 0);
    if (a == MAP_FAILED) {
        perror("mmap 1");
        return 1;
    }
    ((char*)a)[0x80000000] = 0xff;
    printf("large_mmap: mmap 1 completed OK\n");

    rv = munmap(a, TEST_LENGTH2);
    if (rv) {
        perror("mumap");
        return 1;
    }

    a = mmap(NULL, TEST_LENGTH, PROT_READ | PROT_WRITE, MAP_PRIVATE, fileno(fp), 0);
    if (a == MAP_FAILED) {
        perror("mmap 2");
        return 1;
    }
    ((char*)a)[0x100000000] = 0xff;
    printf("large_mmap: mmap 2 completed OK\n");

#if 0
    /* The below fork tests sending of large checkpoints: at this point, the process allocated >4GB
     * of memory and must send it to the child. Thus, this fork stresses 32-bit/64-bit logic in
     * Gramine (especially on SGX PAL). However, for SGX enclaves, this takes several minutes to
     * execute on wimpy machines (with 128MB of EPC), so it is commented out by default. */

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return 1;
    } else if (pid > 0) {
        int status;
        if (wait(&status) < 0) {
            perror("wait");
            return 1;
        }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "wrong wait() status: %d\n", status);
            return 1;
        }
    }
#endif

    return 0;
}
