/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation */

#include "rw_file.h"

#include <assert.h>
#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static ssize_t posix_fd_rw(int fd, char* buf, size_t count, bool do_write) {
    size_t transferred = 0;
    while (transferred < count) {
        ssize_t ret = do_write ? write(fd, buf + transferred, count - transferred) :
                                 read(fd, buf + transferred, count - transferred);

        if (ret < 0) {
            int ret_errno = errno;
            if (ret_errno == EINTR)
                continue;
            warn("%s", do_write ? "write" : "read");
            errno = ret_errno;
            return -1;
        }

        if (ret == 0) {
            /* end of file */
            break;
        }

        transferred += ret;
    }

    return transferred;
}

static ssize_t stdio_fd_rw(FILE* f, char* buf, size_t count, bool do_write) {
    size_t transferred = 0;
    while (transferred < count) {
        size_t ret = do_write ? fwrite(buf + transferred, 1, count - transferred, f) :
                                fread(buf + transferred, 1, count - transferred, f);

        if (ret == 0) {
            /* end of file or error */
            if (ferror(f)) {
                int ret_errno = errno;
                if (ret_errno == EINTR)
                    continue;
                warn("%s", do_write ? "write" : "read");
                errno = ret_errno;
                return -1;
            }

            assert(feof(f));
            break;
        }

        transferred += ret;
    }

    return transferred;
}

static ssize_t posix_file_rw(const char* path, char* buf, size_t count, bool do_write) {
    int fd = open(path, do_write ? O_WRONLY : O_RDONLY);
    if (fd < 0) {
        int ret_errno = errno;
        warn("open");
        errno = ret_errno;
        return -1;
    }

    ssize_t transferred = posix_fd_rw(fd, buf, count, do_write);
    if (transferred < 0) {
        int ret_errno = errno;
        int close_ret = close(fd);
        if (close_ret < 0)
            warn("close (during error handling)");
        errno = ret_errno;
        return -1;
    }

    int close_ret = close(fd);
    if (close_ret < 0) {
        int ret_errno = errno;
        warn("close");
        errno = ret_errno;
        return -1;
    }

    return transferred;
}

static ssize_t stdio_file_rw(const char* path, char* buf, size_t count, bool do_write) {
    FILE* f = fopen(path, do_write ? "w" : "r");
    if (!f) {
        int ret_errno = errno;
        warn("open");
        errno = ret_errno;
        return -1;
    }

    ssize_t transferred = stdio_fd_rw(f, buf, count, do_write);
    if (transferred < 0) {
        int ret_errno = errno;
        int close_ret = fclose(f);
        if (close_ret < 0)
            warn("close (during error handling)");
        errno = ret_errno;
        return -1;
    }

    int close_ret = fclose(f);
    if (close_ret < 0) {
        int ret_errno = errno;
        warn("close");
        errno = ret_errno;
        return -1;
    }

    return transferred;
}


ssize_t posix_file_read(const char* path, char* buf, size_t count) {
    return posix_file_rw(path, buf, count, /*do_write=*/false);
}

ssize_t posix_file_write(const char* path, const char* buf, size_t count) {
    return posix_file_rw(path, (char*)buf, count, /*do_write=*/true);
}


ssize_t stdio_file_read(const char* path, char* buf, size_t count) {
    return stdio_file_rw(path, buf, count, /*do_write=*/false);
}

ssize_t stdio_file_write(const char* path, const char* buf, size_t count) {
    return stdio_file_rw(path, (char*)buf, count, /*do_write=*/true);
}

ssize_t posix_fd_read(int fd, char* buf, size_t count) {
    return posix_fd_rw(fd, buf, count, /*do_write=*/false);
}

ssize_t posix_fd_write(int fd, const char* buf, size_t count) {
    return posix_fd_rw(fd, (char*)buf, count, /*do_write=*/true);
}

ssize_t stdio_fd_read(FILE* f, char* buf, size_t count) {
    return stdio_fd_rw(f, buf, count, /*do_write=*/false);
}

ssize_t stdio_fd_write(FILE* f, const char* buf, size_t count) {
    return stdio_fd_rw(f, (char*)buf, count, /*do_write=*/true);
}
