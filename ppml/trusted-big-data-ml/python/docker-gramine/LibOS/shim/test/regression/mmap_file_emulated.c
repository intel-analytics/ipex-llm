/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2022 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * Test file mapping emulated by Gramine (encrypted, tmpfs): try reading and writing a file through
 * a mapping.
 */

#define _POSIX_C_SOURCE 200112 /* for ftruncate */
#include <assert.h>
#include <err.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#include "rw_file.h"

/* NOTE: these two messages should have equal length */
#define MESSAGE1 "hello world\n  "
#define MESSAGE2 "goodbye world\n"
#define MESSAGE_LEN (sizeof(MESSAGE1) - 1)

int main(int argc, char** argv) {
    if (argc != 2)
        errx(1, "Usage: %s path", argv[0]);

    const char* path = argv[1];

    setbuf(stdout, NULL);

    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size < 0)
        err(1, "sysconf");

    assert(MESSAGE_LEN + 1 <= (size_t)page_size);
    size_t mmap_size = page_size;
    size_t file_size = page_size + MESSAGE_LEN;

    /* Create a new file */

    int fd = open(path, O_WRONLY | O_CREAT, 0666);
    if (fd < 0)
        err(1, "open");

    ssize_t ret = ftruncate(fd, file_size);
    if (ret < 0)
        err(1, "ftruncate");

    /* Write MESSAGE1 at position 0 */

    ret = posix_fd_write(fd, MESSAGE1, MESSAGE_LEN);
    if (ret < 0)
        err(1, "failed to write file");
    if ((size_t)ret < MESSAGE_LEN)
        errx(1, "not enough bytes written");

    /* Write MESSAGE2 at position `page_size` */

    ret = lseek(fd, page_size, SEEK_SET);
    if (ret < 0)
        err(1, "lseek");

    ret = posix_fd_write(fd, MESSAGE2, MESSAGE_LEN);
    if (ret < 0)
        err(1, "failed to write file");
    if ((size_t)ret < MESSAGE_LEN)
        errx(1, "not enough bytes written");

    ret = close(fd);
    if (ret < 0)
        err(1, "close");

    puts("CREATE OK");

    /* Open and map it: MAP_SHARED at offset 0, MAP_PRIVATE at offset `page_size` */

    fd = open(path, O_RDWR, 0);
    if (fd < 0)
        err(1, "open");

    char* addr_shared = mmap(NULL, mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr_shared == MAP_FAILED)
        err(1, "mmap");

    char* addr_private = mmap(NULL, mmap_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, page_size);
    if (addr_private == MAP_FAILED)
        err(1, "mmap");

    /* Close the FD early, so that we know `munmap(addr_shared)` will flush changes */
    ret = close(fd);
    if (ret == -1)
        err(1, "close");

    if (memcmp(addr_shared, MESSAGE1, MESSAGE_LEN))
        errx(1, "wrong mapping content at addr_shared (%s)", addr_shared);

    if (memcmp(addr_private, MESSAGE2, MESSAGE_LEN))
        errx(1, "wrong mapping content at addr_private (%s)", addr_private);

    for (size_t i = MESSAGE_LEN; i < mmap_size; i++) {
        if (addr_shared[i] != 0)
            errx(1, "unexpected non-zero byte at addr_shared[%zu]", i);
        if (addr_private[i] != 0)
            errx(1, "unexpected non-zero byte at addr_private[%zu]", i);
    }

    puts("MAP OK");

    /* Write new message through mmap, then close it */

    strcpy(addr_shared, MESSAGE2);
    strcpy(addr_private, MESSAGE1);

    ret = munmap(addr_shared, mmap_size);
    if (ret < 0)
        err(1, "munmap");

    ret = munmap(addr_private, mmap_size);
    if (ret < 0)
        err(1, "munmap");

    puts("WRITE OK");

    /* Verify the file: only the first write should be applied */

    char buf[file_size];

    ret = posix_file_read(path, buf, sizeof(buf));
    if (ret < 0)
        err(1, "failed to read file");
    if ((size_t)ret < file_size)
        errx(1, "not enough bytes read");

    if (memcmp(&buf[0], MESSAGE2, MESSAGE_LEN))
        errx(1, "wrong file content");

    if (memcmp(&buf[page_size], MESSAGE2, MESSAGE_LEN))
        errx(1, "wrong file content");

    puts("TEST OK");

    return 0;
}
