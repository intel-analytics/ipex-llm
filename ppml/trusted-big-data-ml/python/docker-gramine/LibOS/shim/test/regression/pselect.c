#define _XOPEN_SOURCE 700
#include <stdio.h>
#include <string.h>
#include <sys/select.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

int main(void) {
    fd_set rfds;
    fd_set wfds;

    int ret;
    int fd[2];
    char string[] = "Hello, world!\n";
    struct timespec tv = {.tv_sec = 10, .tv_nsec = 0};

    ret = pipe(fd);
    if (ret < 0) {
        perror("pipe creation failed");
        return 1;
    }

    FD_ZERO(&rfds);
    FD_ZERO(&wfds);
    FD_SET(fd[0], &rfds);
    FD_SET(fd[1], &wfds);

    ret = pselect(fd[1] + 1, NULL, &wfds, NULL, &tv, NULL);
    if (ret <= 0) {
        perror("pselect() on write event failed");
        return 1;
    }
    printf("pselect() on write event returned %d file descriptors\n", ret);

    size_t size = strlen(string) + 1;
    ssize_t write_ret = write(fd[1], string, size);
    if (write_ret < 0 || (size_t)write_ret != size) {
        perror("write error");
        return 1;
    }
    ret = pselect(fd[1] + 1, &rfds, NULL, NULL, &tv, NULL);
    if (ret <= 0) {
        perror("pselect() on read event failed");
        return 1;
    }
    printf("pselect() on read event returned %d file descriptors\n", ret);

    return 0;
}
