#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

/* use the same loopback address and port for both IPV6 and IPV4 */
#define SRV_IPV6 "::1/128"
#define SRV_IPV4 "127.0.0.1"
#define PORT     11112
#define TIMEOUT  5

int main(int argc, char** argv) {
    int socket_ipv4;
    int socket_ipv6;
    int ret;

    if ((socket_ipv6 = socket(AF_INET6, SOCK_STREAM, 0)) < 0) {
        perror("socket(ipv6)");
        return 1;
    }

    if ((socket_ipv4 = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("socket(ipv4)");
        return 1;
    }

    int enable = 1;
    if (setsockopt(socket_ipv6, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable)) < 0) {
        perror("setsockopt(ipv6, SO_REUSEADDR = 1)");
        return 1;
    }
    if (setsockopt(socket_ipv4, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable)) < 0) {
        perror("setsockopt(ipv4, SO_REUSEADDR = 1)");
        return 1;
    }

    /* this forces IPV6 to listen for both IPV4 and IPV6 connections; this in turn makes binding
     * another (IPV4) socket on the same port meaningless and results in -EADDRINUSE */
    int disable = 0;
    if (setsockopt(socket_ipv6, IPPROTO_IPV6, IPV6_V6ONLY, &disable, sizeof(disable)) < 0) {
        perror("setsockopt(IPV6_V6ONLY = 0)");
        return 1;
    }

    struct sockaddr_in6 address_ipv6;
    memset(&address_ipv6, 0, sizeof(address_ipv6));
    address_ipv6.sin6_family = AF_INET6;
    address_ipv6.sin6_port   = htons(PORT);

    if (inet_pton(AF_INET6, SRV_IPV6, &address_ipv6.sin6_addr) < 0) {
        perror("inet_pton(ipv6)");
        return 1;
    }

    if (bind(socket_ipv6, (struct sockaddr*)&address_ipv6, sizeof(address_ipv6)) < 0) {
        perror("bind(ipv6)");
        return 1;
    }

    /* we must start listening on IPV6 socket to make it active and kick in Linux rules for bind()
     */
    if (listen(socket_ipv6, 3) < 0) {
        perror("listen(ipv6)");
        return 1;
    }

    struct timeval tv = {.tv_sec = TIMEOUT};
    socklen_t optlen  = sizeof(tv);
    if (setsockopt(socket_ipv6, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
        perror("setsockopt(ipv6, SO_RCVTIMEO)");
        return 1;
    }
    if (setsockopt(socket_ipv6, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) < 0) {
        perror("setsockopt(ipv6, SO_SNDTIMEO)");
        return 1;
    }

    tv.tv_sec  = 0;
    tv.tv_usec = 0;
    if (getsockopt(socket_ipv6, SOL_SOCKET, SO_RCVTIMEO, &tv, &optlen) < 0) {
        perror("getsockopt(ipv6, SO_RCVTIMEO)");
        return 1;
    }
    if (tv.tv_sec != TIMEOUT || tv.tv_usec != 0 || optlen != sizeof(tv)) {
        fprintf(stderr, "getsockopt(ipv6, SO_RCVTIMEO) returned wrong value\n");
        return 1;
    }

    tv.tv_sec  = 0;
    tv.tv_usec = 0;
    optlen     = sizeof(tv);
    if (getsockopt(socket_ipv6, SOL_SOCKET, SO_SNDTIMEO, &tv, &optlen) < 0) {
        perror("getsockopt(ipv6, SO_SNDTIMEO)");
        return 1;
    }
    if (tv.tv_sec != TIMEOUT || tv.tv_usec != 0 || optlen != sizeof(tv)) {
        fprintf(stderr, "getsockopt(ipv6, SO_SNDTIMEO) returned wrong value\n");
        return 1;
    }

    struct sockaddr_in address_ipv4;
    memset(&address_ipv4, 0, sizeof(address_ipv4));
    address_ipv4.sin_family = AF_INET;
    address_ipv4.sin_port   = htons(PORT); /* note the same port! */

    if (inet_pton(AF_INET, SRV_IPV4, &address_ipv4.sin_addr) < 0) {
        perror("inet_pton(ipv4)");
        return 1;
    }

    ret = bind(socket_ipv4, (struct sockaddr*)&address_ipv4, sizeof(address_ipv4));
    if (ret != -1 || errno != EADDRINUSE) {
        fprintf(stderr,
                "bind(ipv4) was successful even though there is no IPV6_V6ONLY on same port\n");
        return 1;
    }

    puts("test completed successfully");
    return 0;
}
