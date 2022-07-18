#define _GNU_SOURCE
#include <assert.h>
#include <err.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <unistd.h>

static void double_try_epoll_wait(int efd, int rfd, int wfd, uint32_t events) {
    assert(EPOLLIN & events);
    struct epoll_event event = {
        .events = events,
        .data.fd = rfd,
    };
    if (epoll_ctl(efd, EPOLL_CTL_ADD, rfd, &event) < 0) {
        err(1, "EPOLL_CTL_ADD");
    }

    uint64_t n = 1;
    if (write(wfd, &n, sizeof(n)) != sizeof(n)) {
        err(1, "write");
    }
    if (write(wfd, &n, sizeof(n)) != sizeof(n)) {
        err(1, "write");
    }

    memset(&event, '\0', sizeof(event));
    if (epoll_wait(efd, &event, 1, -1) != 1) {
        err(1, "epoll_wait");
    }

    if (event.data.fd != rfd) {
        errx(1, "epoll invalid data: %d", event.data.fd);
    }
    if (event.events != EPOLLIN) {
        errx(1, "epoll invalid events: 0x%x", event.events);
    }

    n = 0;
    if (read(rfd, &n, sizeof(n)) != sizeof(n)) {
        err(1, "read");
    }
    if (n != 1) {
        errx(1, "invalid value read: %zu", n);
    }

    memset(&event, '\0', sizeof(event));
    int ret = epoll_wait(efd, &event, 1, 10);

    if (ret < 0) {
        err(1, "epoll_wait");
    } else if (ret != 0) {
        errx(1, "EPOLLET reported 2 times: %d", ret);
    }
}

int main(void) {
    int efd = epoll_create1(EPOLL_CLOEXEC);
    if (efd < 0) {
        err(1, "epoll_create1");
    }

    int p[2];
    if (pipe2(p, O_NONBLOCK) < 0) {
        err(1, "pipe2");
    }

    double_try_epoll_wait(efd, p[0], p[1], EPOLLIN | EPOLLOUT | EPOLLPRI | EPOLLET);

    if (close(efd) < 0) {
        err(1, "close");
    }

    efd = epoll_create1(EPOLL_CLOEXEC);
    if (efd < 0) {
        err(1, "epoll_create1");
    }

    int fd = eventfd(0, EFD_SEMAPHORE);
    if (fd < 0) {
        err(1, "eventfd");
    }

    double_try_epoll_wait(efd, fd, fd, EPOLLIN | EPOLLPRI | EPOLLET);

    puts("TEST OK");
    return 0;
}
