#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char** argv) {
    int pipefds[2];
    char buffer[1024];
    size_t bufsize = sizeof(buffer);

    if (pipe(pipefds) < 0) {
        perror("pipe error");
        return 1;
    }

    int pid = fork();

    if (pid < 0) {
        perror("fork error");
        return 1;
    } else if (pid == 0) {
        /* client */
        close(pipefds[1]);

        if (read(pipefds[0], &buffer, bufsize) < 0) {
            perror("read error");
            return 1;
        }
        buffer[bufsize - 1] = '\0';

        printf("read on pipe: %s\n", buffer);
    } else {
        /* server */
        close(pipefds[0]);

        snprintf(buffer, bufsize, "Hello from write end of pipe!");
        if (write(pipefds[1], &buffer, strlen(buffer) + 1) < 0) {
            perror("write error");
            return 1;
        }

        wait(NULL); /* wait for child termination, just for sanity */
    }

    return 0;
}
