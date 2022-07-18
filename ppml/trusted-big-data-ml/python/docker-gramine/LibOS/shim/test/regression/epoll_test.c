#define _GNU_SOURCE
#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(x) ({                             \
    __typeof__(x) _x = (x);                     \
    if (_x == -1) {                             \
        err(1, "error at line %d", __LINE__);   \
    }                                           \
    _x;                                         \
})

#define ERR(msg, args...) \
    errx(1, "%d: " msg, __LINE__, ##args)

#define ARRAY_LEN(arr) (sizeof(arr) / sizeof(arr[0]))

static uint64_t wait_event(int epfd, struct epoll_event* possible_events,
                           size_t possible_events_len) {
    struct epoll_event event = { 0 };
    int x = CHECK(epoll_wait(epfd, &event, 1, -1));
    if (x != 1) {
        ERR("epoll_wait returned: %d", x);
    }

    for (size_t i = 0; i < possible_events_len; ++i) {
        if (possible_events[i].data.u64 == event.data.u64) {
            if (possible_events[i].events != event.events) {
                ERR("wrong events returned: %#x", event.events);
            }
            return event.data.u64;
        }
    }

    ERR("unknown event: %zu %#x", event.data.u64, event.events);
}

static void test_epoll_migration(void) {
    int epfd = CHECK(epoll_create1(EPOLL_CLOEXEC));
    int pipe_fds[2];
    CHECK(pipe(pipe_fds));

    struct epoll_event events[2] = {
        { .events = EPOLLIN, .data.u64 = pipe_fds[0], },
        { .events = EPOLLOUT, .data.u64 = pipe_fds[1], },
    };
    CHECK(epoll_ctl(epfd, EPOLL_CTL_ADD, pipe_fds[0], &events[0]));

    CHECK(epoll_ctl(epfd, EPOLL_CTL_ADD, pipe_fds[1], &events[1]));

    pid_t p = CHECK(fork());
    if (p != 0) {
        int status = 0;
        pid_t w = CHECK(wait(&status));
        if (w != p) {
            ERR("wait returned wrong pid: %d (expected: %d)", w, p);
        }
        if (!WIFEXITED(status) || (WEXITSTATUS(status) != 0)) {
            ERR("child exited with: %#x", status);
        }

        CHECK(close(pipe_fds[0]));
        CHECK(close(pipe_fds[1]));
        CHECK(close(epfd));
        return;
    }

    // child

    if (wait_event(epfd, events, ARRAY_LEN(events)) != (uint64_t)pipe_fds[1]) {
        ERR("expected different event");
    }

    char c = 0;
    ssize_t y = CHECK(write(pipe_fds[1], &c, sizeof(c)));
    if (y != sizeof(c)) {
        ERR("write: %zd", y);
    }

    uint64_t e1 = wait_event(epfd, events, ARRAY_LEN(events));
    uint64_t e2 = wait_event(epfd, events, ARRAY_LEN(events));

    if (e1 == e2) {
        ERR("epoll_wait did not round robin");
    }

    exit(0);
}

static void test_epoll_oneshot(void) {
    int epfd = CHECK(epoll_create1(EPOLL_CLOEXEC));
    int pipe_fds[2];
    CHECK(pipe(pipe_fds));

    struct epoll_event event = {
        .events = EPOLLIN | EPOLLONESHOT,
        .data.u64 = pipe_fds[0],
    };
    CHECK(epoll_ctl(epfd, EPOLL_CTL_ADD, pipe_fds[0], &event));

    memset(&event, 0, sizeof(event));
    int x = CHECK(epoll_wait(epfd, &event, 1, 1));
    if (x != 0) {
        ERR("epoll_wait returned: %d, events: %#x, data: %lu", x, event.events, event.data.u64);
    }

    char c = 0;
    ssize_t y = CHECK(write(pipe_fds[1], &c, sizeof(c)));
    if (y != sizeof(c)) {
        ERR("write: %zd", y);
    }

    memset(&event, 0, sizeof(event));
    x = CHECK(epoll_wait(epfd, &event, 1, 1));
    if (x != 1 || event.events != EPOLLIN || event.data.u64 != (uint64_t)pipe_fds[0]) {
        ERR("epoll_wait returned: %d, events: %#x, data: %lu", x, event.events, event.data.u64);
    }

    memset(&event, 0, sizeof(event));
    x = CHECK(epoll_wait(epfd, &event, 1, 1));
    if (x != 0) {
        ERR("epoll_wait returned: %d, events: %#x, data: %lu", x, event.events, event.data.u64);
    }

    /* rearm */
    event.events = EPOLLIN | EPOLLONESHOT;
    event.data.u64 = pipe_fds[0];
    CHECK(epoll_ctl(epfd, EPOLL_CTL_MOD, pipe_fds[0], &event));

    memset(&event, 0, sizeof(event));
    x = CHECK(epoll_wait(epfd, &event, 1, 1));
    if (x != 1 || event.events != EPOLLIN || event.data.u64 != (uint64_t)pipe_fds[0]) {
        ERR("epoll_wait returned: %d, events: %#x, data: %lu", x, event.events, event.data.u64);
    }

    CHECK(close(pipe_fds[0]));
    CHECK(close(pipe_fds[1]));
    CHECK(close(epfd));
}

int main(void) {
    test_epoll_migration();

    test_epoll_oneshot();

    puts("TEST OK");
    return 0;
}
