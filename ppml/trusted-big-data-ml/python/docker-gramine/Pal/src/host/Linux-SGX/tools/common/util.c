/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Invisible Things Lab
 *                    Rafal Wojdyla <omeg@invisiblethingslab.com>
 */

#define _LARGEFILE64_SOURCE

#include "util.h"

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

/*! Console stdout fd */
int g_stdout_fd = 1;

/*! Console stderr fd */
int g_stderr_fd = 2;

/*! Verbosity level */
bool g_verbose = false;

/*! Endianness for hex strings */
endianness_t g_endianness = ENDIAN_LSB;

void set_verbose(bool verbose) {
    g_verbose = verbose;
    if (verbose)
        DBG("Verbose output enabled\n");
    else
        DBG("Verbose output disabled\n");
}

bool get_verbose(void) {
    return g_verbose;
}

void set_endianness(endianness_t endianness) {
    g_endianness = endianness;
    if (g_verbose) {
        if (endianness == ENDIAN_LSB)
            DBG("Endianness set to LSB\n");
        else
            DBG("Endianness set to MSB\n");
    }
}

endianness_t get_endianness(void) {
    return g_endianness;
}

/* return -1 on error */
uint64_t get_file_size(int fd) {
    struct stat64 st;

    if (fstat64(fd, &st) != 0)
        return (uint64_t)-1;

    return st.st_size;
}

void* read_file(const char* path, size_t* size, void* buffer) {
    FILE* f = NULL;
    uint64_t fs = 0;
    void* orig_buf = buffer;

    if (!size || !path)
        return NULL;

    assert(*size > 0 || !buffer);

    f = fopen(path, "rb");
    if (!f) {
        ERROR("Failed to open file '%s' for reading: %s\n", path, strerror(errno));
        goto out;
    }

    if (*size == 0) { // read whole file
        fs = get_file_size(fileno(f));
        if (fs == (uint64_t)-1) {
            ERROR("Failed to get size of file '%s': %s\n", path, strerror(errno));
            goto out;
        }
    } else {
        fs = *size;
    }

    if (!buffer) {
        buffer = malloc(fs);
        if (!buffer) {
            ERROR("No memory\n");
            goto out;
        }
    }

    if (fread(buffer, fs, 1, f) != 1) {
        ERROR("Failed to read file '%s'\n", path);
        if (!orig_buf) {
            free(buffer);
            buffer = NULL;
        }
    }

out:
    if (f)
        fclose(f);

    if (buffer)
        *size = fs;

    return buffer;
}

static int write_file_internal(const char* path, size_t size, const void* buffer, bool append) {
    FILE* f = NULL;
    int status;

    if (append)
        f = fopen(path, "ab");
    else
        f = fopen(path, "wb");

    if (!f) {
        ERROR("Failed to open file '%s' for writing: %s\n", path, strerror(errno));
        goto out;
    }

    if (size > 0 && buffer) {
        if (fwrite(buffer, size, 1, f) != 1) {
            ERROR("Failed to write file '%s': %s\n", path, strerror(errno));
            goto out;
        }
    }

    errno = 0;

out:
    status = errno;
    if (f)
        fclose(f);
    return status;
}

/* Write buffer to file */
int write_file(const char* path, size_t size, const void* buffer) {
    return write_file_internal(path, size, buffer, false);
}

/* Append buffer to file */
int append_file(const char* path, size_t size, const void* buffer) {
    return write_file_internal(path, size, buffer, true);
}

/* Set stdout/stderr descriptors */
void util_set_fd(int stdout_fd, int stderr_fd) {
    g_stdout_fd = stdout_fd;
    g_stderr_fd = stderr_fd;
}

/* Print memory as hex in buffer */
int hexdump_mem_to_buffer(const void* data, size_t size, char* buffer, size_t buffer_size) {
    uint8_t* ptr = (uint8_t*)data;

    if (buffer_size < size * 2 + 1) {
        ERROR("Insufficiently large buffer to dump data as hex string\n");
        return -1;
    }

    for (size_t i = 0; i < size; i++) {
        if (g_endianness == ENDIAN_LSB) {
            sprintf(buffer + i * 2, "%02x", ptr[i]);
        } else {
            sprintf(buffer + i * 2, "%02x", ptr[size - i - 1]);
        }
    }

    buffer[size * 2] = 0; /* end of string */
    return 0;
}

/* Print memory as hex */
void hexdump_mem(const void* data, size_t size) {
    uint8_t* ptr = (uint8_t*)data;

    for (size_t i = 0; i < size; i++) {
        if (g_endianness == ENDIAN_LSB) {
            INFO("%02x", ptr[i]);
        } else {
            INFO("%02x", ptr[size - i - 1]);
        }
    }

    INFO("\n");
}

/* Parse hex string to buffer */
int parse_hex(const char* hex, void* buffer, size_t buffer_size, const char* mask) {
    if (!hex || !buffer || buffer_size == 0)
        return -1;

    char sep_l, sep_r;
    if (mask) {
        sep_l = '<';
        sep_r = '>';
    } else {
        mask = hex;
        sep_l = sep_r = '\'';
    }

    if (strlen(hex) != buffer_size * 2) {
        ERROR("Invalid length of hex string %c%s%c\n", sep_l, mask, sep_r);
        return -1;
    }

    for (size_t i = 0; i < buffer_size; i++) {
        if (!isxdigit(hex[i * 2]) || !isxdigit(hex[i * 2 + 1])) {
            ERROR("Invalid hex string %c%s%c\n", sep_l, mask, sep_r);
            return -1;
        }

        if (g_endianness == ENDIAN_LSB)
            sscanf(hex + i * 2, "%02hhx", &((uint8_t*)buffer)[i]);
        else
            sscanf(hex + i * 2, "%02hhx", &((uint8_t*)buffer)[buffer_size - i - 1]);
    }
    return 0;
}

/* _log() and abort() are needed for our <assert.h>; see also: common/include/callbacks.h */
void _log(int level, const char* fmt, ...) {
    (void)level;
    va_list ap;
    va_start(ap, fmt);
    vdprintf(g_stderr_fd, fmt, ap);
    va_end(ap);
    dprintf(g_stderr_fd, "\n");
}

noreturn void abort(void) {
    _Exit(ENOTRECOVERABLE);
}
