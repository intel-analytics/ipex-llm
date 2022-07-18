#define _GNU_SOURCE
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

int main(void) {
    int ret;
    int fd[2];
    char string[] = "Hello, world!\n";
    struct timespec tv = {.tv_sec = 10, .tv_nsec = 0};

    ret = pipe(fd);
    if (ret < 0) {
        perror("pipe creation failed");
        return 1;
    }

    struct pollfd outfds[] = {
        {.fd = fd[1], .events = POLLOUT},
    };
    ret = ppoll(outfds, 1, &tv, NULL);
    if (ret <= 0) {
        perror("ppoll with POLLOUT failed");
        return 1;
    }
    printf("ppoll(POLLOUT) returned %d file descriptors\n", ret);

    struct pollfd infds[] = {
        {.fd = fd[0], .events = POLLIN},
    };

    size_t size = strlen(string) + 1;
    ssize_t write_ret = write(fd[1], string, size);
    if (write_ret < 0 || (size_t)write_ret != size) {
        perror("write error");
        return 1;
    }

    ret = ppoll(infds, 1, &tv, NULL);
    if (ret <= 0) {
        perror("ppoll with POLLIN failed");
        return 1;
    }
    printf("ppoll(POLLIN) returned %d file descriptors\n", ret);

    return 0;
}
