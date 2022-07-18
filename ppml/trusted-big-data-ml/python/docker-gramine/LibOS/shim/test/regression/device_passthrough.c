#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char* arvg[]) {
    int devfd = open("/dev/host-zero", O_RDONLY);
    if (devfd < 0)
        err(1, "/dev/host-zero open");

    off_t offset;
#if 0
    /* FIXME: doesn't work in Gramine because lseek() is fully emulated in LibOS and therefore
     *        lseek() is not aware of device-specific semantics */
    offset = lseek(devfd, 0, SEEK_CUR);
    if (offset != -1 || errno != EINVAL) {
        errx(1, "/dev/host-zero lseek(0, SEEK_CUR) didn't return -EINVAL (returned: %ld, errno=%d)",
             offset, errno);
    }

    offset = lseek(devfd, 1, SEEK_CUR);
    if (offset != -1 || errno != ESPIPE) {
        errx(1, "/dev/host-zero lseek(1, SEEK_CUR) didn't return -ESPIPE (returned: %ld, errno=%d)",
             offset, errno);
    }
#endif

    offset = lseek(devfd, /*offset=*/0, SEEK_SET);
    if (offset < 0)
        err(1, "/dev/host-zero lseek(0, SEEK_SET)");
    if (offset > 0)
        errx(1, "/dev/host-zero lseek(0, SEEK_SET) didn't return 0 (returned: %ld)", offset);

    char buf = 'A';
    ssize_t bytes = read(devfd, &buf, sizeof(buf));
    if (bytes < 0)
        err(1, "/dev/host-zero read");

    if (buf != '\0')
        errx(1, "read from /dev/host-zero didn't return NUL byte");

    int ret = close(devfd);
    if (ret < 0)
        err(1, "/dev/host-zero close");

    puts("TEST OK");
    return 0;
}
