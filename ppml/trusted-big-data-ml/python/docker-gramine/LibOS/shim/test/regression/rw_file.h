/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation */

#ifndef RW_FILE_H_
#define RW_FILE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <sys/types.h>

/* All functions below return the number of bytes read/written on success or -1 (with set errno) on
 * failure. All functions restart read/write syscall in case of EINTR. */

/* Opens file `path`, reads/writes at most `count` bytes into/from buffer `buf` and closes the
 * file. Uses POSIX functions: open, read/write, close. */
ssize_t posix_file_read(const char* path, char* buf, size_t count);
ssize_t posix_file_write(const char* path, const char* buf, size_t count);

/* Opens file `path`, reads/writes at most `count` bytes into/from buffer `buf` and closes the
 * file. Uses stdio functions: fopen, fread/fwrite, fclose. */
ssize_t stdio_file_read(const char* path, char* buf, size_t count);
ssize_t stdio_file_write(const char* path, const char* buf, size_t count);

/* Reads/writes at most `count` bytes into/from buffer `buf`. Uses POSIX functions: read/write. */
ssize_t posix_fd_read(int fd, char* buf, size_t count);
ssize_t posix_fd_write(int fd, const char* buf, size_t count);

/* Reads/writes at most `count` bytes into/from buffer `buf`. Uses stdio functions: fread/fwrite. */
ssize_t stdio_fd_read(FILE* f, char* buf, size_t count);
ssize_t stdio_fd_write(FILE* f, const char* buf, size_t count);

#endif /* RW_FILE_H_ */
