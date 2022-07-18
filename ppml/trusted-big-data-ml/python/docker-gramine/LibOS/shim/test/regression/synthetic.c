/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2022 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * Test for synthetic directories created by Gramine in the process of mounting. In the test
 * configuration (`manifest.template`), "/mnt" is such a directory, because we mount a filesystem at
 * "/mnt/tmpfs".
 */

#define _GNU_SOURCE /* O_DIRECTORY */
#include <dirent.h>
#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* Check if it's possible to list `path` and whether it contains `subpath` */
static void test_list(const char* path, const char* subpath) {
    struct dirent** namelist;
    int n = scandir(path, &namelist, /*filter=*/NULL, alphasort);
    if (n == -1)
        err(1, "scandir");

    bool subpath_found = false;
    for (int i = 0; i < n; i++) {
        const char* name = namelist[i]->d_name;
        printf("dirent: %s\n", name);
        if (strcmp(name, subpath) == 0)
            subpath_found = true;
        free(namelist[i]);
    }
    free(namelist);
    if (!subpath_found)
        errx(1, "%s not found in directory entries of %s", subpath, path);
}

int main(void) {
    const char* path = "/mnt";
    const char* subpath = "tmpfs";
    struct stat statbuf;

    test_list(path, subpath);

    if (rmdir(path) != -1 || errno != EACCES)
        err(1, "rmdir should return EACCES");

    if (stat(path, &statbuf) == -1)
        err(1, "stat");

    int fd = open(path, O_DIRECTORY);
    if (fd == -1)
        err(1, "open");

    if (fstat(fd, &statbuf) == -1)
        err(1, "fstat");

    /* fsync on the directory is only for testing that it doesn't fail (and is a no-op) */
    if (fsync(fd) == -1)
        err(1, "fsync");

    if (close(fd) == -1)
        err(1, "close");

    printf("TEST OK\n");

    return 0;
}
