#ifndef COMMON_H
#define COMMON_H

#define _GNU_SOURCE
#include <assert.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdnoreturn.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/sendfile.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define OVERFLOWS(type, val)                        \
    ({                                              \
        type __dummy;                               \
        __builtin_add_overflow((val), 0, &__dummy); \
    })

noreturn void fatal_error(const char* fmt, ...);
void setup(void);
int open_input_fd(const char* path);
void read_fd(const char* path, int fd, void* buffer, size_t size);
void seek_fd(const char* path, int fd, off_t offset, int mode);
off_t tell_fd(const char* path, int fd);
int open_output_fd(const char* path, bool rdwr);
void write_fd(const char* path, int fd, const void* buffer, size_t size);
void sendfile_fd(const char* input_path, const char* output_path, int fi, int fo, size_t size);
void close_fd(const char* path, int fd);
void* mmap_fd(const char* path, int fd, int protection, size_t offset, size_t size);
void munmap_fd(const char* path, void* address, size_t size);
FILE* open_input_stdio(const char* path);
void read_stdio(const char* path, FILE* f, void* buffer, size_t size);
void seek_stdio(const char* path, FILE* f, off_t offset, int mode);
off_t tell_stdio(const char* path, FILE* f);
void close_stdio(const char* path, FILE* f);
FILE* open_output_stdio(const char* path, bool rdwr);
void write_stdio(const char* path, FILE* f, const void* buffer, size_t size);
void* alloc_buffer(size_t size);
void fill_random(void* buffer, size_t size);

void copy_data(int fi, int fo, const char* input_path, const char* output_path, size_t size);

#endif
