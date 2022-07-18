#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* This is supposed to expose resource leaks where close()d files are not properly cleaned up. */

int main(int argc, char** argv) {
    for (int i = 0; i < 10000; i++) {
        int fd = open(argv[0], O_RDONLY);
        if (fd == -1)
            abort();
        char buf[1024];
        ssize_t read_ret = read(fd, buf, sizeof(buf));
        if (read_ret == -1)
            abort();
        int ret = close(fd);
        if (ret == -1)
            abort();
    }

    puts("Test succeeded.");

    return 0;
}
