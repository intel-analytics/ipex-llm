/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * Check if a file handle is correctly being sent to a child process. Writes data to a file, then
 * lets the child read it.
 *
 * With "-d", the file is deleted from host, so that it's accessible only through the open handle.
 */

#include <assert.h>
#include <err.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define MESSAGE "hello world"
#define MESSAGE_LEN (sizeof(MESSAGE) - 1)

int main(int argc, char** argv) {
    bool delete = false;

    int i = 1;
    if (i < argc && strcmp(argv[i], "-d") == 0) {
        delete = true;
        i++;
    }

    if (i + 1 != argc)
        errx(1, "Usage: %s [-d] path", argv[0]);

    const char* path = argv[i];

    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (fd == -1)
        err(1, "open");

    if (delete) {
        if (unlink(path) == -1)
            err(1, "unlink");
    }

    const char* message = MESSAGE;
    size_t pos = 0;
    do {
        ssize_t ret = write(fd, &message[pos], MESSAGE_LEN - pos);
        if (ret < 0)
            err(1, "write");
        if (ret == 0)
            errx(1, "write unexpectedly returned 0");
        pos += ret;
    } while (pos < MESSAGE_LEN);

    /*
     * Make sure writes to a protected file are flushed.
     *
     * TODO: this should not be necessary, Gramine should ensure protected files are flushed during
     * fork.
     */
    if (fsync(fd) == -1)
        err(1, "fsync");

    pid_t pid = fork();
    if (pid < 0)
        err(1, "fork");

    if (pid == 0) {
        if (lseek(fd, 0, SEEK_SET) == -1)
            err(1, "child: seek");

        char buf[MESSAGE_LEN];
        pos = 0;
        do {
            ssize_t ret = read(fd, &buf[pos], MESSAGE_LEN - pos);
            if (ret < 0)
                err(1, "child: read");
            if (ret == 0)
                errx(1, "child: unexpected EOF");
            pos += ret;
        } while (pos < MESSAGE_LEN);

        if (memcmp(message, buf, MESSAGE_LEN) != 0)
            errx(1, "child: wrong file content");
    } else {
        int status;
        if (waitpid(pid, &status, 0) == -1)
            err(1, "waitpid");
        if (!WIFEXITED(status))
            errx(1, "child not exited");
        if (WEXITSTATUS(status) != 0)
            errx(1, "unexpected exit status: %d", WEXITSTATUS(status));

        if (!delete) {
            if (unlink(path) == -1)
                err(1, "unlink");
        }

        printf("TEST OK\n");
    }
    return 0;
}
