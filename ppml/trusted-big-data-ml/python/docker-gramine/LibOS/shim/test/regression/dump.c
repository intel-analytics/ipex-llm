/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

#define _DEFAULT_SOURCE /* lstat */
#include <dirent.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "dump.h"

#define PATH_MAX 4096

static int dump_dir(const char* path) {
    printf("%s: directory\n", path);
    fflush(stdout);

    size_t buf_size = PATH_MAX;
    size_t path_len = strlen(path);
    if (path_len + 1 > buf_size) {
        fprintf(stderr, "path too long: %s\n", path);
        return -1;
    }

    char* buf = malloc(buf_size);
    if (!buf) {
        fprintf(stderr, "out of memory\n");
        return -1;
    }
    memcpy(buf, path, path_len);
    buf[path_len] = '/';

    int ret, close_ret;

    DIR* dir = opendir(path);
    if (!dir) {
        perror("opendir");
        ret = -1;
        goto out;
    }

    for (;;) {
        errno = 0;
        struct dirent* dirent = readdir(dir);
        if (!dirent) {
            if (errno) {
                perror("readdir");
                ret = -1;
                goto out;
            }
            break;
        }

        if (strcmp(dirent->d_name, ".") == 0 || strcmp(dirent->d_name, "..") == 0)
            continue;

        size_t name_len = strlen(dirent->d_name);
        if (path_len + 1 + name_len + 1 > buf_size) {
            fprintf(stderr, "path too long: %s/%s", path, dirent->d_name);
            ret = -1;
            goto out;
        }

        /* Copy file name and null terminator */
        memcpy(&buf[path_len + 1], dirent->d_name, name_len + 1);
        ret = dump_path(buf);
        if (ret < 0)
            goto out;
    }

    ret = 0;
out:
    free(buf);
    close_ret = closedir(dir);
    if (close_ret < 0) {
        perror("closedir");
        return -1;
    }
    return ret;
}

static int dump_regular(const char* path) {
    printf("%s: file\n", path);
    fflush(stdout);

    FILE* f = fopen(path, "r");
    if (!f) {
        perror("fopen");
        return -1;
    }

    fflush(stdout);

    char buf[4096];
    size_t n;
    int ret, close_ret;
    bool new_line = true;

    do {
        n = fread(buf, 1, sizeof(buf), f);
        if (ferror(f) != 0) {
            perror("fread");
            goto out;
        }

        for (size_t i = 0; i < n; i++) {
            if (new_line) {
                printf("[%s] ", path);
                fflush(stdout);
                new_line = false;
            }
            printf("%c", buf[i]);
            if (buf[i] == '\n')
                new_line = true;
        }
    } while (n > 0);

    if (!new_line) {
        printf("\n");
        fflush(stdout);
    }
    ret = 0;
out:
    close_ret = fclose(f);
    if (close_ret < 0) {
        perror("fclose");
        return -1;
    }
    return ret;
}

int dump_path(const char* path) {
    int ret;
    struct stat statbuf;

    ret = lstat(path, &statbuf);
    if (ret < 0) {
        perror("lstat");
        return -1;
    }

    switch (statbuf.st_mode & S_IFMT) {
        case S_IFBLK:
            printf("%s: block device\n", path);
            fflush(stdout);
            break;
        case S_IFCHR:
            printf("%s: character device\n", path);
            break;
        case S_IFDIR:
            ret = dump_dir(path);
            if (ret < 0)
                return -1;
            break;
        case S_IFLNK: {
            size_t buf_size = PATH_MAX;
            char* buf = malloc(buf_size);
            if (!buf) {
                fprintf(stderr, "out of memory\n");
                return -1;
            }
            ssize_t n = readlink(path, buf, buf_size);
            if (n < 0) {
                free(buf);
                perror("readlink");
                return -1;
            }
            printf("%s: link: %.*s\n", path, (int)n, buf);
            fflush(stdout);
            free(buf);
            break;
        }
        case S_IFREG: {
            ret = dump_regular(path);
            if (ret < 0)
                return -1;
            break;
        }
        case S_IFSOCK:
            printf("%s: socket\n", path);
            fflush(stdout);
            break;
        default:
            fprintf(stderr, "unknown file type: %s\n", path);
            return -1;
    }
    return 0;
}
