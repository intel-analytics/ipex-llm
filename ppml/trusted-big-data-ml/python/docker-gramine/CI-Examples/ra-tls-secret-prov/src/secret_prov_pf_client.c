/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Labs */

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define INPUT_FILENAME "files/input.txt"

static int print_pf_key_and_read_protected_file(char* who) {
    char* secret = getenv("SECRET_PROVISION_SECRET_STRING");
    if (!secret) {
        fprintf(stderr, "did not receive protected files master key!\n");
        return -1;
    }
    printf("--- [%s] Received protected files master key = '%s' ---\n", who, secret);

    int fd = open(INPUT_FILENAME, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[error] cannot open '" INPUT_FILENAME "'\n");
        return -1;
    }

    char buf[1024] = {0};
    ssize_t bytes_read = 0;
    while (1) {
        ssize_t ret = read(fd, buf + bytes_read, sizeof(buf) - bytes_read);
        if (ret > 0) {
            bytes_read += ret;
        } else if (ret == 0) {
            /* end of file */
            break;
        } else if (errno == EAGAIN || errno == EINTR) {
            continue;
        } else {
            fprintf(stderr, "[error] cannot read '" INPUT_FILENAME "'\n");
            close(fd);
            return -1;
        }
    }

    int ret = close(fd);
    if (ret < 0) {
        fprintf(stderr, "[error] cannot close '" INPUT_FILENAME "'\n");
        return -1;
    }

    printf("--- [%s] Read from protected file: '%s' ---\n", who, buf);
    return 0;
}

int main(int argc, char** argv) {
    /* execvp() is only for testing purposes: to validate that secret provisioning happens only once
     * (for the first process) and that exec logic preserves the provisioned key */
    if (argc < 2 || strcmp(argv[1], "skip-execve")) {
        puts("--- [main] Re-starting myself using execvp() ---");
        fflush(stdout);
        char* new_argv[] = {argv[0], "skip-execve", NULL};
        int ret = execvp(argv[0], new_argv);
        if (ret < 0) {
            perror("execvp error");
            return 1;
        }
    }

    /* fork() is only for testing purposes: to validate that secret provisioning doesn't happen
     * after fork (in the child enclave) and that fork logic preserves the provisioned key */
    int pid = fork();

    if (pid < 0) {
        perror("fork error");
        return 1;
    } else if (pid == 0) {
        /* child goes first */
        if (print_pf_key_and_read_protected_file("child") < 0) {
            fprintf(stderr, "child could not read the protected file\n");
            return 1;
        }
    } else {
        /* parent waits for child to finish and repeats the same logic */
        if (wait(NULL) < 0) {
            perror("wait error");
            return 1;
        }
        if (print_pf_key_and_read_protected_file("parent") < 0) {
            fprintf(stderr, "parent could not read the protected file\n");
            return 1;
        }
    }

    return 0;
}
