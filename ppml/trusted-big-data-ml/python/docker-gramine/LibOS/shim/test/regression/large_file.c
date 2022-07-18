/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/* This test checks large file sizes and offsets that overflow 32-bit integers. */

#define _GNU_SOURCE /* ftruncate */
#include <assert.h>
#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define TEST_FILE "tmp/large_file"

static_assert(sizeof(off_t) == 8, "this test is for 64-bit off_t");

static off_t test_lengths[] = {
    // around 2 GB (limit of 32-bit signed int)
    0x7FFFFFFF,
    0x80000000,
    0x80000001,
    // around 4 GB (limit of 32-bit unsigned int)
    0xFFFFFFFF,
    0x100000000,
    0x100000001,
    0,
};

static void try_seek(int fd, off_t offset, int whence, off_t expected) {
    off_t result = lseek(fd, offset, whence);
    if (result == -1)
        err(1, "lseek(fd, %ld, %d) returned -1", offset, whence);
    if (result != expected)
        errx(1, "lseek(fd, %ld, %d) returned %ld (0x%lx), expected %ld (0x%lx)", offset, whence,
             result, result, expected, expected);
}

int main(void) {
    setbuf(stdout, NULL);

    int fd = open(TEST_FILE, O_CREAT | O_TRUNC | O_RDWR, 0600);
    if (fd < 0)
        err(1, "open");

    int ret;

    for (unsigned int i = 0; test_lengths[i] != 0; i++) {
        off_t length = test_lengths[i];
        printf("testing length 0x%lx\n", length);

        /* Resize the file */
        ret = ftruncate(fd, length);
        if (ret < 0)
            err(1, "ftruncate");

        /* Check file size */
        struct stat st;
        if (stat(TEST_FILE, &st) < 0)
            err(1, "stat");
        if (st.st_size != length)
            errx(1, "stat: got 0x%lx, expected 0x%lx", st.st_size, length);

        /* Seek to end - 1 */
        try_seek(fd, -1, SEEK_END, length - 1);

        /* Read a single byte, check position */
        char c;
        ssize_t n;
        do {
            n = read(fd, &c, 1);
        } while (n == -1 && errno == -EINTR);
        if (n == -1)
            err(1, "read");
        if (n != 1)
            errx(1, "read %ld bytes, expected %d", n, 1);
        if (c != 0)
            errx(1, "read byte %d, expected %d", (int)c, 0);
        try_seek(fd, 0, SEEK_CUR, length);

        /* Seek to 0 and then back to length by providing large offsets */
        try_seek(fd, -length, SEEK_END, 0);
        try_seek(fd, length, SEEK_SET, length);
    }

    if (close(fd) < 0)
        err(1, "close");

    printf("TEST OK\n");
    return 0;
}
