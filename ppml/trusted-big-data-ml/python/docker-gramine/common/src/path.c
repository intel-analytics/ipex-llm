/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains functions to read app config (manifest) file and create a tree to
 * lookup / access config values.
 */

#include "api.h"
#include "pal_error.h"

/*
 * Finds next '/' in `path`.
 * Returns a pointer to it or to the nullbyte ending the string if no '/' has been found.
 */
static inline const char* find_next_slash(const char* path) {
    while (*path && *path != '/') {
        path++;
    }
    return path;
}

/*
 * Finds previous '/' in `path` (starting from `offset` - 1).
 * If the last character is '/', then it is skipped (as a token can end with '/').
 *
 * Returns whether '/' was found.
 * Updates `*offset` to the index of the found '/' (or 0 if none was found).
 */
static inline bool find_prev_slash_offset(const char* path, size_t* offset) {
    size_t off = *offset;

    if (!off) {
        return false;
    }

    off--; // get offset to last character

    /* Skip trailing '/' if there is one */
    if (off && path[off] == '/') {
        off--;
    }
    while (off && path[off] != '/') {
        off--;
    }

    *offset = off;
    return path[off] == '/';
}

/*
 * Before calling this function *inout_size should hold the size of buf.
 * After returning it holds number of bytes actually written to it (including the ending '\0'). This
 * number is never greater than the size of the input path.
 */
int get_norm_path(const char* path, char* buf, size_t* inout_size) {
    assert(path && buf && inout_size);
    size_t path_size = strlen(path) + 1;
    __UNUSED(path_size);  // used only for an assert at the end

    size_t size = *inout_size;
    if (!size) {
        return -PAL_ERROR_INVAL;
    }
    /* reserve 1 byte for ending '\0' */
    size--;

    size_t offset = 0, ret_size = 0; /* accounts for undiscardable bytes written to `buf`
                                      * i.e. `buf - ret_size` points to original `buf` */
    bool need_slash = false; // is '/' needed before next token
    bool is_absolute_path = *path == '/';

    /* handle an absolute path */
    if (is_absolute_path) {
        if (size < 1) {
            return -PAL_ERROR_TOOLONG;
        }
        *buf++ = '/';
        size--;
        ret_size++;
        path++;
    }

    while (1) {
        /* handle next token */
        const char* end = find_next_slash(path);
        if (end - path == 2 && path[0] == '.' && path[1] == '.') {
            /* ".." */
            if (offset) {
                /* eat up previously written token */
                need_slash = find_prev_slash_offset(buf, &offset);
            } else if (!is_absolute_path) {
                /* append undiscardable ".." since there is no previous token
                 * but only if the path is not absolute */
                if (need_slash + 2u > size) {
                    return -PAL_ERROR_TOOLONG;
                }
                if (need_slash) {
                    *buf++ = '/';
                }
                *buf++ = '.';
                *buf++ = '.';
                size -= need_slash + 2u;
                ret_size += need_slash + 2u;
                need_slash = true;
            } else {
                /* remaining case: offset == 0, path is absolute and ".." was just seen,
                 * i.e. "/..", which is collapsed to "/", hence nothing needs to be done
                 */
            }
        } else if ((end == path) || (end - path == 1 && path[0] == '.')) {
            /* ignore "//" and "." */
        } else {
            size_t len = (size_t)(end - path);
            if (need_slash + len > size - offset) {
                return -PAL_ERROR_TOOLONG;
            }
            if (need_slash) {
                buf[offset++] = '/';
            }
            memcpy(buf + offset, path, len);
            offset += len;
            need_slash = true;
        }
        if (!*end) {
            break;
        }
        path = end + 1;
    }

    buf[offset] = '\0';

    *inout_size = ret_size + offset + 1;
    assert(*inout_size <= path_size);

    return 0;
}

/*
 * Returns the part after the last '/' (so `path` should probably be normalized).
 * Before calling this function *size should hold the size of buf.
 * After returning it holds number of bytes actually written to it (including the trailing '\0').
 */
int get_base_name(const char* path, char* buf, size_t* inout_size) {
    if (!path || !buf || !inout_size) {
        return -PAL_ERROR_INVAL;
    }

    const char* end;
    while (*(end = find_next_slash(path))) {
        path = end + 1;
    }

    size_t result = (size_t)(end - path);
    if (result + 1 > *inout_size) {
        return -PAL_ERROR_TOOLONG;
    }

    memcpy(buf, path, result);
    buf[result] = '\0';

    *inout_size = result + 1;

    return 0;
}
