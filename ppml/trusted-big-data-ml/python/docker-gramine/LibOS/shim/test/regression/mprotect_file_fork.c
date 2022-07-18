#define _XOPEN_SOURCE 700
#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define FNAME "/tmp/test"

#define VAL 0xff

int main(void) {
    int fd;
    void* ptr;

    if (mkdir("/tmp", S_IRWXU | S_IRWXG | S_IRWXO) < 0 && errno != EEXIST) {
        err(1, "mkdir");
    }
    if (unlink(FNAME) < 0 && errno != ENOENT) {
        err(1, "unlink");
    }

    fd = open(FNAME, O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        err(1, "open");
    }
    if (ftruncate(fd, 0x10) < 0) {
        err(1, "ftruncate");
    }

    ptr = mmap(NULL, 0x1000, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) {
        err(1, "mmap");
    }

    if (close(fd) < 0) {
        err(1, "close");
    }

    if (mprotect(ptr, 0x1000, PROT_READ | PROT_WRITE) < 0) {
        err(1, "mprotect");
    }

    *(int*)ptr = VAL;

    pid_t p = fork();
    if (p < 0) {
        err(1, "fork");
    }

    if (p == 0) {
        // child
        if (*(int*)ptr != VAL) {
            printf("EXPECTED: 0x%x\nGOT     : 0x%x\n", VAL, *(int*)ptr);
            return 1;
        }
        return 0;
    }

    // parent
    int st = 0;
    if (wait(&st) < 0) {
        err(1, "wait");
    }

    if (unlink(FNAME) < 0) {
        err(1, "unlink");
    }

    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
        printf("abnormal child termination: %d\n", st);
        return 1;
    }

    puts("Test successful!");
    return 0;
}
