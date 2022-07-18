#define _XOPEN_SOURCE 700
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char** argv) {
    int r, fd;
    struct stat buf;

    fd = open(".", O_DIRECTORY);
    if (fd == -1) {
        printf("Opening CWD returned error %d\n", errno);
        return -1;
    }

    r = fstat(fd, &buf);
    if (r == -1) {
        printf("fstat on directory fd returned error %d\n", errno);
        return -1;
    }

    close(fd);

    if (S_ISDIR(buf.st_mode))
        printf("fstat returned the fd type as S_IFDIR\n");

    return 0;
}
