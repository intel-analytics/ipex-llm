#include <arpa/inet.h>
#include <errno.h>
#include <ifaddrs.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define PORT 1088

int main(void) {
    struct sockaddr_in bind_addr, connected_addr;
    int sockfd;
    socklen_t len = sizeof(struct sockaddr);

    bind_addr.sin_family = AF_INET;
    bind_addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    /* test 1: bind to static port PORT */
    sockfd = socket(AF_INET, SOCK_STREAM, 0);

    if (sockfd < 0) {
        printf("getsockname: error in socket %d\n", errno);
        return 1;
    }

    bind_addr.sin_port = htons(PORT);

    if (bind(sockfd, (struct sockaddr*)&bind_addr, sizeof(bind_addr)) < 0) {
        printf("getsockname: error in bind %d\n", errno);
        goto fail;
    }

    if (listen(sockfd, 3) < 0) {
        printf("getsockname: error in listen %d\n", errno);
        goto fail;
    }

    len = sizeof(connected_addr);
    if (getsockname(sockfd, (struct sockaddr*)&connected_addr, &len) < 0) {
        printf("getsockname: error in getsockname %d\n", errno);
        goto fail;
    }

    if (len != sizeof(connected_addr)) {
        printf("getsockname: unexpected length of sockaddr\n");
        goto fail;
    }

    if (ntohs(connected_addr.sin_port) != PORT) {
        printf("getsockname: port returned from getsockname is %d != %d\n", ntohs(connected_addr.sin_port), PORT);
        goto fail;
    }

    printf("getsockname: Local IP address is: %s:%d\n", inet_ntoa(connected_addr.sin_addr), ntohs(connected_addr.sin_port));
    printf("getsockname: Got socket name with static port OK\n");

    close(sockfd);

    /* test 2: bind to arbitrary port */
    sockfd = socket(AF_INET, SOCK_STREAM, 0);

    if (sockfd < 0) {
        printf("getsockname: error in socket %d\n", errno);
        return 1;
    }

    bind_addr.sin_port = 0;

    if (bind(sockfd, (struct sockaddr*)&bind_addr, sizeof(bind_addr)) < 0) {
        printf("getsockname: error in bind %d\n", errno);
        goto fail;
    }

    if (listen(sockfd, 3) < 0) {
        printf("getsockname: error in listen %d\n", errno);
        goto fail;
    }

    len = sizeof(connected_addr);
    if (getsockname(sockfd, (struct sockaddr*)&connected_addr, &len) < 0) {
        printf("getsockname: error in getsockname %d\n", errno);
        goto fail;
    }

    if (len != sizeof(connected_addr)) {
        printf("getsockname: unexpected length of sockaddr\n");
        goto fail;
    }

    if (connected_addr.sin_port == 0) {
        printf("getsockname: port returned from getsockname is %d == 0\n", ntohs(connected_addr.sin_port));
        goto fail;
    }

    close(sockfd);

    printf("getsockname: Local IP address is: %s:%d\n", inet_ntoa(connected_addr.sin_addr), ntohs(connected_addr.sin_port));
    printf("getsockname: Got socket name with arbitrary port OK\n");
    return 0;

fail:
    close(sockfd);
    return 1;
}
