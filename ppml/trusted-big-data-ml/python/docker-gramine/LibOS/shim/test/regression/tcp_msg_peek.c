#define _GNU_SOURCE
#include <arpa/inet.h>
#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define SRV_IP "127.0.0.1"
#define PORT   11111
#define BUFLEN 512

enum { SINGLE, PARALLEL } mode = PARALLEL;
int pipefds[2];

static void server(void) {
    int listening_socket, client_socket;
    struct sockaddr_in address;
    socklen_t addrlen;

    if ((listening_socket = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("socket");
        exit(1);
    }

    int enable = 1;
    if (setsockopt(listening_socket, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable)) < 0) {
        perror("setsockopt");
        exit(1);
    }

    memset(&address, 0, sizeof(address));
    address.sin_family      = AF_INET;
    address.sin_port        = htons(PORT);
    address.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(listening_socket, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("bind");
        exit(1);
    }

    if (listen(listening_socket, 3) < 0) {
        perror("listen");
        exit(1);
    }

    if (mode == PARALLEL) {
        if (close(pipefds[0]) < 0) {
            perror("close of pipe");
            exit(1);
        }

        char byte = 0;

        ssize_t written = -1;
        while (written < 0) {
            if ((written = write(pipefds[1], &byte, sizeof(byte))) < 0) {
                if (errno == EINTR || errno == EAGAIN)
                    continue;
                perror("write on pipe");
                exit(1);
            }
            if (!written) {
                /* technically impossible, but let's fail loudly if we ever hit this */
                errx(EXIT_FAILURE, "write on pipe returned zero");
            }
        }
    }

    addrlen       = sizeof(address);
    client_socket = accept(listening_socket, (struct sockaddr*)&address, &addrlen);

    if (client_socket < 0) {
        perror("accept");
        exit(1);
    }

    if (close(listening_socket) < 0) {
        perror("close of listening socket");
        exit(1);
    }

    puts("[server] client is connected...");

    char buffer[] = "Hello from server!\n";

    size_t written = 0;
    while (written < sizeof(buffer)) {
        ssize_t n;
        /* we specify dummy MSG_DONTWAIT just to test this flag */
        if ((n = sendto(client_socket, buffer + written, sizeof(buffer) - written, MSG_DONTWAIT,
                        /*dest_addr=*/0, /*addrlen=*/0)) < 0) {
            if (errno == EINTR || errno == EAGAIN)
                continue;
            perror("sendto to client");
            exit(1);
        }
        if (!n) {
            /* technically impossible, but let's fail loudly if we ever hit this */
            errx(EXIT_FAILURE, "sendto to client returned zero");
        }
        written += n;
    }

    if (close(client_socket) < 0) {
        perror("close of client socket");
        exit(1);
    }

    puts("[server] done");
}

static ssize_t client_recv(int server_socket, char* buf, size_t len, int flags) {
    ssize_t read = 0;
    while (1) {
        ssize_t n;
        if ((n = recv(server_socket, buf + read, len - read, flags)) < 0) {
            if (errno == EINTR || errno == EAGAIN)
                continue;
            perror("client recv");
            exit(1);
        }

        read += n;

        if (!n || flags & MSG_PEEK) {
            /* recv with MSG_PEEK flag should be done only once */
            break;
        }
    }

    return read;
}

static void client(void) {
    int server_socket;
    struct sockaddr_in address;
    char buffer[BUFLEN];
    ssize_t count;

    if (mode == PARALLEL) {
        if (close(pipefds[1]) < 0) {
            perror("close of pipe");
            exit(1);
        }

        char byte = 0;

        ssize_t received = -1;
        while (received < 0) {
            if ((received = read(pipefds[0], &byte, sizeof(byte))) < 0) {
                if (errno == EINTR || errno == EAGAIN)
                    continue;
                perror("read on pipe");
                exit(1);
            }
            if (!received) {
                perror("read on pipe (EOF)");
                exit(1);
            }
        }
    }

    if ((server_socket = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("socket");
        exit(1);
    }

    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port   = htons((PORT));
    if (inet_aton(SRV_IP, &address.sin_addr) == 0) {
        perror("inet_aton");
        exit(1);
    }

    if (connect(server_socket, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("connect");
        exit(1);
    }

    /* we specify dummy MSG_DONTWAIT and MSG_WAITALL just to test these flags */
    printf("[client] receiving with MSG_PEEK: ");
    count = client_recv(server_socket, buffer, sizeof(buffer),
                        MSG_WAITALL | MSG_DONTWAIT | MSG_PEEK);
    fwrite(buffer, count, 1, stdout);

    printf("[client] receiving with MSG_PEEK again: ");
    count = client_recv(server_socket, buffer, sizeof(buffer),
                        MSG_WAITALL | MSG_DONTWAIT | MSG_PEEK);
    fwrite(buffer, count, 1, stdout);

    printf("[client] receiving without MSG_PEEK: ");
    count = client_recv(server_socket, buffer, sizeof(buffer), MSG_DONTWAIT);
    fwrite(buffer, count, 1, stdout);

    printf("[client] checking how many bytes are left unread: ");
    count = client_recv(server_socket, buffer, sizeof(buffer), MSG_DONTWAIT);
    printf("%zu\n", count);

    if (close(server_socket) < 0) {
        perror("close of server socket");
        exit(1);
    }

    puts("[client] done");
}

int main(int argc, char** argv) {
    /* FIXME: This test depends on output being buffered. */
    const size_t output_buf_size = 0x1000;
    char* output_buf = malloc(output_buf_size);
    if (!output_buf) {
        err(1, "OOM");
    }
    setbuffer(stdout, output_buf, output_buf_size);

    if (argc > 1) {
        if (strcmp(argv[1], "client") == 0) {
            mode = SINGLE;
            client();
            return 0;
        }

        if (strcmp(argv[1], "server") == 0) {
            mode = SINGLE;
            server();
            return 0;
        }
    } else {
        if (pipe(pipefds) < 0) {
            perror("pipe error");
            return 1;
        }

        int pid = fork();

        if (pid == 0) {
            client();
        } else {
            server();

            int status = 0;
            if (wait(&status) < 0) {
                err(1, "wait");
            }
            if (!WIFEXITED(status) || WEXITSTATUS(status)) {
                errx(1, "child wait status: %d", status);
            }
        }
    }

    return 0;
}
