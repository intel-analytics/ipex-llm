/* Unit test for issues #92 and #644 */

#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>

int main(int argc, char** argv) {
    int ret;
    socklen_t optlen; /* Option length */

    int fd = socket(PF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        perror("socket failed");
        return 1;
    }

    int so_type;
    optlen = sizeof(so_type);
    ret = getsockopt(fd, SOL_SOCKET, SO_TYPE, &so_type, &optlen);
    if (ret < 0) {
        perror("getsockopt(SOL_SOCKET, SO_TYPE) failed");
        return 1;
    }

    if (optlen != sizeof(so_type) || so_type != SOCK_STREAM) {
        fprintf(stderr, "getsockopt(SOL_SOCKET, SO_TYPE) failed\n");
        return 1;
    }

    printf("getsockopt: Got socket type OK\n");

    int so_flags = 1;
    optlen = sizeof(so_flags);
    ret = getsockopt(fd, SOL_TCP, TCP_NODELAY, (void*)&so_flags, &optlen);
    if (ret < 0) {
        perror("getsockopt(SOL_TCP, TCP_NODELAY) failed");
        return 1;
    }

    if (optlen != sizeof(so_flags) || (so_flags != 0 && so_flags != 1)) {
        fprintf(stderr, "getsockopt(SOL_TCP, TCP_NODELAY) failed\n");
        return 1;
    }

    printf("getsockopt: Got TCP_NODELAY flag OK\n");
    return 0;
}
