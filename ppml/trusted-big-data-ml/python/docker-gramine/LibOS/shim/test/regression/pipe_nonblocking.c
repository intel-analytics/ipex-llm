#define _GNU_SOURCE
#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

int main(void) {
    int p[2];

    if (pipe2(p, O_NONBLOCK | O_CLOEXEC) < 0) {
        err(1, "pipe2");
    }

    /* Verify both ends of the pipe provide same flags. */
    int flags_wr = fcntl(p[1], F_GETFL);
    if (flags_wr < 0)
        err(1, "fcntl(<write end of pipe>, F_GETFL)");

    int flags_rd = fcntl(p[0], F_GETFL);
    if (flags_rd < 0)
        err(1, "fcntl(<read end of pipe>, F_GETFL)");

    /* Ensure O_NONBLOCK flag is properly set on both ends of pipe. */
    if (!(flags_wr & O_NONBLOCK) || !(flags_rd & O_NONBLOCK))
        errx(1, "Expected O_NONBLOCK flag to be set on both ends of pipe but got flags_wr=0x%x, "
             "flags_rd=0x%x", flags_wr, flags_rd);

    flags_rd = flags_rd & ~O_ACCMODE;
    flags_wr = flags_wr & ~O_ACCMODE;

    if (flags_wr != flags_rd)
        errx(1, "`F_GETFL` flags mismatch: flags_wr=0x%x and flags_rd=0x%x", flags_wr, flags_rd);

    flags_wr = fcntl(p[1], F_GETFD);
    if (flags_wr < 0)
        err(1, "fcntl(<write end of pipe>, F_GETFD)");

    flags_rd = fcntl(p[0], F_GETFD);
    if (flags_rd < 0)
        err(1, "fcntl(<read end of pipe>, F_GETFD)");

    /* Ensure O_CLOEXEC flag is properly set on both ends of pipe. */
    if (!(flags_wr & FD_CLOEXEC) || !(flags_rd & FD_CLOEXEC))
        errx(1, "Expected O_CLOEXEC flag to be set on both ends of pipe but got flags_wr=0x%x, "
             "flags_rd=0x%x", flags_wr, flags_rd);

    if (flags_wr != flags_rd)
        errx(1, "`F_GETFD` flags mismatch: flags_wr=0x%x and flags_rd=0x%x", flags_wr, flags_rd);

    ssize_t ret = write(p[1], "a", 1);
    if (ret < 0) {
        err(1, "write");
    } else if (ret != 1) {
        errx(1, "invalid return value from write: %zd\n", ret);
    }

    char c;
    ret = read(p[0], &c, 1);
    if (ret < 0) {
        err(1, "read");
    } else if (ret != 1) {
        errx(1, "invalid return value from read: %zd\n", ret);
    }

    ret = read(p[0], &c, 1);
    if (ret > 0) {
        errx(1, "read returned unexpected data: %zd\n", ret);
    } else if (ret == 0) {
        errx(1, "read returned 0 instead of EAGAIN\n");
    } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
        err(1, "unexpected read failure");
    }

    puts("TEST OK");
    return 0;
}
