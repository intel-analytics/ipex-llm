/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

#include <asm/errno.h>
#include <linux/mman.h>

#include "api.h"
#include "linux_utils.h"

static int parse_proc_maps_line(const char* line, struct proc_maps_range* r) {
    const char* next = line;

#define PARSE_NUMBER(base)                               \
    ({                                                   \
        unsigned long val;                               \
        if (str_to_ulong(next, (base), &val, &next) < 0) \
            return -1;                                   \
        val;                                             \
    })

#define PARSE_CHAR(c1, c2)             \
    ({                                 \
        char c = *next++;              \
        if (!(c == (c1) || c == (c2))) \
            return -1;                 \
        c;                             \
    })

#define SKIP_CHAR(c)        \
    do {                    \
        if (*next != (c))   \
            return -1;      \
        next++;             \
    } while(0)

#define SKIP_FIELD()                        \
    do {                                    \
        next++;                             \
    } while (*next != '\0' && *next != ' ')

    /* address */
    r->start = PARSE_NUMBER(16);
    SKIP_CHAR('-');
    r->end = PARSE_NUMBER(16);
    SKIP_CHAR(' ');

    /* perms (`rwxp`) */
    r->prot = 0;
    if (PARSE_CHAR('r', '-') == 'r')
        r->prot |= PROT_READ;
    if (PARSE_CHAR('w', '-') == 'w')
        r->prot |= PROT_WRITE;
    if (PARSE_CHAR('x', '-') == 'x')
        r->prot |= PROT_EXEC;
    PARSE_CHAR('p', 's'); // unused for now
    SKIP_CHAR(' ');

    /* offset */
    r->offset = PARSE_NUMBER(16);
    SKIP_CHAR(' ');

    /* dev */
    SKIP_FIELD();
    SKIP_CHAR(' ');

    /* inode */
    SKIP_FIELD();
    SKIP_CHAR(' ');

    /* pathname */
    while (*next == ' ')
        next++;

    r->name = (*next != '\0' ? next : NULL);

#undef PARSE_NUMBER
#undef PARSE_CHAR
#undef SKIP_CHAR
#undef SKIP_FIELD

    return 0;
}

struct parse_proc_maps_data {
    int (*orig_callback)(struct proc_maps_range*, void*);
    void* arg;
};

static int parse_proc_maps_callback(const char* line, void* arg, bool* out_stop) {
    __UNUSED(out_stop);

    struct parse_proc_maps_data* data = arg;

    struct proc_maps_range r;
    if (parse_proc_maps_line(line, &r) < 0) {
        log_warning("%s: failed to parse /proc/[pid]/maps line \"%s\"", __func__, line);
        return -EINVAL;
    }
    return data->orig_callback(&r, data->arg);
}

int parse_proc_maps(const char* path, int (*callback)(struct proc_maps_range* r, void* arg),
                    void* arg) {
    struct parse_proc_maps_data data = { .orig_callback = callback, .arg = arg };
    return read_text_file_iter_lines(path, parse_proc_maps_callback, &data);
}
