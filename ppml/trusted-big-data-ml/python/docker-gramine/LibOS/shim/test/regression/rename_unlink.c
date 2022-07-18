/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * Tests for renaming and deleting files. Mostly focus on cases where a file is still open.
 */

#include <assert.h>
#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "common.h"

static const char message1[] = "first message\n";
static const size_t message1_len = sizeof(message1) - 1;

static const char message2[] = "second message\n";
static const size_t message2_len = sizeof(message2) - 1;

static_assert(sizeof(message1) != sizeof(message2), "the messages should have different lengths");

static int write_all(int fd, const char* str, size_t size) {
    while (size > 0) {
        ssize_t n = write(fd, str, size);
        /* Treat EINTR as error: we don't expect it because the test doesn't use any signal
         * handlers. */
        if (n == -1) {
            warn("write");
            return -1;
        }
        assert(n >= 0 && (size_t)n <= size);
        size -= n;
        str += n;
    }
    return 0;
}

static int read_all(int fd, char* str, size_t size) {
    while (size > 0) {
        ssize_t n = read(fd, str, size);
        /* Treat EINTR as error: we don't expect it because the test doesn't use any signal
         * handlers. */
        if (n == -1) {
            warn("read");
            return -1;
        }
        if (n == 0) {
            if (size > 0) {
                warnx("read less bytes than expected");
                return -1;
            }
            break;
        }
        assert(n >= 0 && (size_t)n <= size);
        size -= n;
        str += n;
    }
    return 0;
}

static void should_not_exist(const char* path) {
    struct stat statbuf;

    if (stat(path, &statbuf) == 0)
        errx(1, "%s unexpectedly exists", path);
    if (errno != ENOENT)
        err(1, "stat %s", path);
}

static void check_statbuf(const char* desc, struct stat* statbuf, size_t size) {
    assert(!OVERFLOWS(off_t, size));

    if (!S_ISREG(statbuf->st_mode))
        errx(1, "%s: wrong mode (0o%o)", desc, statbuf->st_mode);
    if (statbuf->st_size != (off_t)size)
        errx(1, "%s: wrong size (%lu)", desc, statbuf->st_size);
}

static void should_exist(const char* path, size_t size) {
    struct stat statbuf;

    if (stat(path, &statbuf) == -1)
        err(1, "stat %s", path);

    check_statbuf(path, &statbuf, size);
}

static void should_contain(const char* desc, int fd, const char* str, size_t len) {
    char* buffer = malloc(len);
    if (!buffer)
        err(1, "malloc");

    struct stat statbuf;
    if (fstat(fd, &statbuf) == -1)
        err(1, "%s: fstat", desc);
    check_statbuf(desc, &statbuf, len);

    if (lseek(fd, 0, SEEK_SET) == -1)
        err(1, "%s: lseek", desc);

    if (read_all(fd, buffer, len) == -1)
        errx(1, "%s: read_all failed", desc);
    if (memcmp(buffer, str, len) != 0)
        errx(1, "%s: wrong content", desc);

    free(buffer);
}

static int create_file(const char* path, const char* str, size_t len) {
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd == -1)
        err(1, "open %s", path);

    if (write_all(fd, str, len) == -1)
        errx(1, "write_all %s", path);

    return fd;
}

static void create_file_and_close(const char* path, const char* str, size_t len) {
    int fd = create_file(path, str, len);
    if (close(fd) == -1)
        err(1, "close %s", path);
}

static void test_simple_rename(const char* path1, const char* path2) {
    printf("%s...\n", __func__);

    create_file_and_close(path1, message1, message1_len);

    if (rename(path1, path2) == -1)
        err(1, "rename");

    should_not_exist(path1);
    should_exist(path2, message1_len);

    int fd = open(path2, O_RDONLY, 0);
    if (fd == -1)
        err(1, "open %s", path2);

    should_contain("file opened after it's renamed", fd, message1, message1_len);

    if (close(fd) == -1)
        err(1, "close %s", path2);

    if (unlink(path2) == -1)
        err(1, "unlink %s", path2);
}

static void test_rename_replace(const char* path1, const char* path2) {
    printf("%s...\n", __func__);

    create_file_and_close(path1, message1, message1_len);

    int fd = create_file(path2, message2, message2_len);

    if (fd == -1)
        err(1, "open %s", path2);

    if (rename(path1, path2) == -1)
        err(1, "rename");

    should_not_exist(path1);
    should_exist(path2, message1_len);

    /* We expect `fd` to still point to old data, even though we replaced the file under its path */
    should_contain("file opened before it's replaced", fd, message2, message2_len);

    if (close(fd) == -1)
        err(1, "close %s", path2);

    fd = open(path2, O_RDONLY, 0);
    if (fd == -1)
        err(1, "open %s", path2);

    should_contain("file opened after it's replaced", fd, message1, message1_len);

    if (close(fd) == -1)
        err(1, "close %s", path2);

    if (unlink(path2) == -1)
        err(1, "unlink %s", path2);
}

static void test_rename_open_file(const char* path1, const char* path2) {
    printf("%s...\n", __func__);

    int fd = create_file(path1, message1, message1_len);

    if (rename(path1, path2) == -1)
        err(1, "rename");

    should_contain("file opened before it's renamed", fd, message1, message1_len);

    if (close(fd) == -1)
        err(1, "close %s", path2);

    if (unlink(path2) == -1)
        err(1, "unlink %s", path2);
}

static void test_unlink_and_recreate(const char* path) {
    printf("%s...\n", __func__);

    int fd1 = create_file(path, message1, message1_len);

    if (unlink(path) == -1)
        err(1, "unlink");

    should_not_exist(path);

    int fd2 = create_file(path, message2, message2_len);

    should_exist(path, message2_len);
    should_contain("file opened before deleting", fd1, message1, message1_len);
    should_contain("file opened after the old one is deleted", fd2, message2, message2_len);

    if (close(fd1) == -1)
        err(1, "close old %s", path);
    if (close(fd2) == -1)
        err(1, "close new %s", path);
    if (unlink(path) == -1)
        err(1, "unlink %s", path);
}

static void test_unlink_and_write(const char* path) {
    printf("%s...\n", __func__);

    int fd = create_file(path, /*message=*/NULL, /*len=*/0);

    if (unlink(path) == -1)
        err(1, "unlink");

    should_not_exist(path);

    if (write_all(fd, message1, message1_len) == -1)
        errx(1, "write_all %s", path);

    should_contain("unlinked file", fd, message1, message1_len);
    should_not_exist(path);

    if (close(fd) == -1)
        err(1, "close unlinked %s", path);
}

int main(int argc, char* argv[]) {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    if (argc != 3)
        errx(1, "Usage: %s <path1> <path2>", argv[0]);

    const char* path1 = argv[1];
    const char* path2 = argv[2];

    test_simple_rename(path1, path2);
    test_rename_replace(path1, path2);
    test_rename_open_file(path1, path2);
    test_unlink_and_recreate(path1);
    test_unlink_and_write(path1);
    printf("TEST OK\n");
    return 0;
}
