#define _GNU_SOURCE
#include <arpa/inet.h>
#include <err.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define SRV_IP "127.0.0.1"
#define PORT   9930
#define BUFLEN 512
#define NPACK  10

enum { SINGLE, PARALLEL } mode = PARALLEL;

int pipefds[2];

static void server(void) {
    struct sockaddr_in si_me, si_other;
    socklen_t slen = sizeof(si_other);
    char buf[BUFLEN];

    int s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s < 0)
        err(EXIT_FAILURE, "server socket");

    memset((char*)&si_me, 0, sizeof(si_me));
    si_me.sin_family      = AF_INET;
    si_me.sin_port        = htons(PORT);
    si_me.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(s, (struct sockaddr*)&si_me, sizeof(si_me)) < 0)
        err(EXIT_FAILURE, "server bind");

    if (mode == PARALLEL) {
        if (close(pipefds[0]) < 0)
            err(EXIT_FAILURE, "server close of pipe");

        char byte = 0;
        ssize_t written = -1;
        while (written < 0) {
            if ((written = write(pipefds[1], &byte, sizeof(byte))) < 0) {
                if (errno == EINTR || errno == EAGAIN)
                    continue;
                err(EXIT_FAILURE, "server write on pipe");
            }
            if (!written) {
                /* technically impossible, but let's fail loudly if we ever hit this */
                errx(EXIT_FAILURE, "server write on pipe returned zero");
            }
        }
    }

    for (int i = 0; i < NPACK; i++) {
        size_t read = 0;
        while (read < BUFLEN) {
            ssize_t n = recvfrom(s, buf + read, BUFLEN - read, /*flags=*/0,
                                 (struct sockaddr*)&si_other, &slen);
            if (n < 0) {
                if (errno == EINTR || errno == EAGAIN)
                    continue;
                err(EXIT_FAILURE, "server recvfrom");
            }
            if (!n) {
                /* socket closed prematurely, considered an error in this test */
                err(EXIT_FAILURE, "server recvfrom (EOF)");
            }
            read += n;
        }

        printf("Received packet from %s:%d ('%s')\n", inet_ntoa(si_other.sin_addr),
               ntohs(si_other.sin_port), buf);
    }

    if (close(s) < 0)
        err(EXIT_FAILURE, "server close");
}

static void client(void) {
    struct sockaddr_in si_other;
    socklen_t slen   = sizeof(si_other);
    char buf[BUFLEN] = "hi";

    if (mode == PARALLEL) {
        if (close(pipefds[1]) < 0)
            err(EXIT_FAILURE, "client close of pipe");

        char byte = 0;
        ssize_t received = -1;
        while (received < 0) {
            if ((received = read(pipefds[0], &byte, sizeof(byte))) < 0) {
                if (errno == EINTR || errno == EAGAIN)
                    continue;
                err(EXIT_FAILURE, "client read on pipe");
            }
            if (!received)
                err(EXIT_FAILURE, "client read on pipe (EOF)");
        }
    }

    int s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s < 0)
        err(EXIT_FAILURE, "client socket");

    memset((char*)&si_other, 0, sizeof(si_other));
    si_other.sin_family = AF_INET;
    si_other.sin_port   = htons((PORT));
    if (inet_aton(SRV_IP, &si_other.sin_addr) < 0)
        err(EXIT_FAILURE, "client inet_aton");

    for (int i = 0; i < NPACK; i++) {
        printf("Sending packet %d\n", i);
        sprintf(buf, "This is packet %d", i);

        ssize_t written = 0;
        while (written < BUFLEN) {
            ssize_t n = sendto(s, buf + written, BUFLEN - written, /*flags=*/0,
                               (struct sockaddr*)&si_other, slen);
            if (n < 0) {
                if (errno == EINTR || errno == EAGAIN)
                    continue;
                err(EXIT_FAILURE, "client sendto");
            }
            if (!n) {
                /* technically impossible, but let's fail loudly if we ever hit this */
                errx(EXIT_FAILURE, "client sendto returned zero");
            }
            written += n;
        }
    }

    if (close(s) < 0)
        err(EXIT_FAILURE, "client close");
}

int main(int argc, char** argv) {
    if (argc > 1) {
        if (strcmp(argv[1], "client") == 0) {
            mode = SINGLE;
            client();
            return 0;
        } else if (strcmp(argv[1], "server") == 0) {
            mode = SINGLE;
            server();
            return 0;
        } else {
            errx(EXIT_FAILURE, "Invalid option!");
        }
    }

    if (pipe(pipefds) < 0)
        err(EXIT_FAILURE, "pipe");

    int pid = fork();
    if (pid < 0)
        err(EXIT_FAILURE, "fork");

    if (pid == 0) {
        client();
        return 0;
    }

    server();

    if (waitpid(pid, NULL, 0) < 0)
        err(EXIT_FAILURE, "waitpid");

    return 0;
}
