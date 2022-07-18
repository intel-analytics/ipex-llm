/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/* Simple sanity check for poll() on various file types. */

#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static void check_poll(int fd, short events, int revents) {
    struct pollfd fds[] = {
        {.fd = fd, .events = events, .revents = 0},
    };

    int ret = poll(fds, /*nfds=*/1, /*timeout=*/0);
    fprintf(stderr, "poll {events = 0x%x} = %d", events, ret);
    if (ret == 1)
        fprintf(stderr, "; {revents = 0x%x}", (int)fds[0].revents);
    fprintf(stderr, "\n");
    fflush(stderr);
    if (ret < 0)
        err(1, "poll");

    int expected = revents ? 1 : 0;
    if (ret != expected)
        errx(1, "expected poll to return %d\n", expected);

    if (ret == 1 && fds[0].revents != revents)
        errx(1, "expected revents to be 0x%x", revents);
}

static void write_byte(int fd, char c) {
    ssize_t n;
    do {
        char c = 0;
        n = write(fd, &c, sizeof(c));
    } while (n == -1 && errno == EINTR);
    if (n == -1)
        err(1, "write");
    if (n != 1)
        errx(1, "write returned %ld", n);
}

static void test_pipe(void) {
    printf("testing poll() on pipe...\n");
    int ret;

    int fds[2];
    ret = pipe(fds);
    if (ret < 0)
        err(1, "pipe");

    check_poll(fds[1], POLLOUT, POLLOUT);
    check_poll(fds[0], POLLIN, 0);

    write_byte(fds[1], 0);

    check_poll(fds[0], POLLIN, POLLIN);

    if (close(fds[0]) < 0 || close(fds[1]) < 0)
        err(1, "close");
}

static void test_file(const char* path, int flags, int events1, int revents1, int events2,
                      int revents2) {
    printf("testing poll() on %s...\n", path);

    int fd = open(path, flags, 0600);
    if (fd < 0)
        err(1, "open");

    if (events1)
        check_poll(fd, events1, revents1);

    if (events2)
        check_poll(fd, events2, revents2);

    if (close(fd) < 0)
        err(1, "close");

    if (flags & O_CREAT) {
        if (unlink(path) < 0)
            err(1, "unlink");
    }
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    test_pipe();

    /* Emulated device */
    test_file("/dev/null", O_RDWR, POLLIN, POLLIN, POLLOUT, POLLOUT);

    /* File in /proc/ */
    test_file("/proc/meminfo", O_RDONLY, POLLIN, POLLIN, 0, 0);

    /* Host file */
    test_file(argv[0], O_RDONLY, POLLIN, POLLIN, 0, 0);

    /* Host file (empty) */
    test_file("tmp/host_file", O_RDWR | O_CREAT | O_TRUNC, POLLIN, 0, POLLOUT, POLLOUT);

    printf("TEST OK\n");
    return 0;
}
