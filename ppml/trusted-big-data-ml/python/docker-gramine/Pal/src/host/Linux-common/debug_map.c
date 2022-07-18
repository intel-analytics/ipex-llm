/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

#include <asm/errno.h>
#include <asm/mman.h>

#include "api.h"
#include "debug_map.h"
#include "linux_utils.h"
#include "spinlock.h"

struct debug_map* _Atomic g_debug_map = NULL;

/* Lock for modifying g_debug_map on our end. Even though the list can be read by GDB at any time,
 * we need to prevent concurrent modification. */
static spinlock_t g_debug_map_lock = INIT_SPINLOCK_UNLOCKED;

static struct debug_map* debug_map_new(const char* name, void* addr) {
    struct debug_map* map;

    if (!(map = malloc(sizeof(*map))))
        return NULL;

    if (!(map->name = strdup(name))) {
        free(map);
        return NULL;
    }

    map->addr = addr;
    map->next = NULL;
    return map;
}

/* This function is hooked by our gdb integration script and should be left as is. */
__attribute__((__noinline__)) void debug_map_update_debugger(void) {
    __asm__ volatile(""); // Required in addition to __noinline__ to prevent deleting this function.
                          // See GCC docs.
}

int debug_map_add(const char* name, void* addr) {
    spinlock_lock(&g_debug_map_lock);

    struct debug_map* map = g_debug_map;
    while (map) {
        if (map->addr == addr) {
            bool name_matches = !strcmp(name, map->name);
            spinlock_unlock(&g_debug_map_lock);
            /* If the exact same map is already there, skip adding it and report success: this can
             * happen when we encounter two executable ranges for the same file. */
            return name_matches ? 0 : -EEXIST;
        }
        map = map->next;
    }

    map = debug_map_new(name, addr);
    if (!map) {
        spinlock_unlock(&g_debug_map_lock);
        return -ENOMEM;
    }

    map->next = g_debug_map;
    g_debug_map = map;

    spinlock_unlock(&g_debug_map_lock);

    debug_map_update_debugger();

    return 0;
}

int debug_map_remove(void* addr) {
    spinlock_lock(&g_debug_map_lock);

    struct debug_map* prev = NULL;
    struct debug_map* map = g_debug_map;
    while (map) {
        if (map->addr == addr)
            break;
        prev = map;
        map = map->next;
    }
    if (!map) {
        spinlock_unlock(&g_debug_map_lock);
        return -EINVAL;
    }
    if (prev) {
        prev->next = map->next;
    } else {
        g_debug_map = map->next;
    }

    spinlock_unlock(&g_debug_map_lock);

    debug_map_update_debugger();

    free(map->name);
    free(map);

    return 0;
}

/* Find a range that (likely) corresponds to a mapped executable file, and add it to debug maps. */
static int debug_map_init_callback(struct proc_maps_range* r, void* arg) {
    __UNUSED(arg);

    /* not executable */
    if (!(r->prot & PROT_EXEC))
        return 0;

    /* no name */
    if (!r->name)
        return 0;

    /* [vvar] etc. */
    if (r->name[0] != '/' && strcmp(r->name, "[vdso]"))
        return 0;

    /* /dev/sgx etc. */
    if (strstartswith(r->name, "/dev/"))
        return 0;

    void* addr = (void*)(r->start - r->offset);
    return debug_map_add(r->name, addr);
}

int debug_map_init_from_proc_maps(void) {
    return parse_proc_maps("/proc/self/maps", debug_map_init_callback, /*arg=*/NULL);
}

/* Search for a debug map the address belongs to. We don't store map sizes, so this searches for the
 * closest one. */
static int debug_map_find(uintptr_t addr, char** out_name, uintptr_t* out_offset) {
    int ret;

    spinlock_lock(&g_debug_map_lock);

    const char* best_name = NULL;
    uintptr_t best_addr = 0;
    struct debug_map* map = g_debug_map;
    while (map) {
        if ((uintptr_t)map->addr <= addr && (uintptr_t)map->addr > best_addr) {
            best_name = map->name;
            best_addr = (uintptr_t)map->addr;
        }
        map = map->next;
    }

    if (!best_name) {
        ret = -ENOENT;
        goto out;
    }

    char* name = strdup(best_name);
    if (!name) {
        ret = -ENOMEM;
        goto out;
    }

    *out_name = name;
    *out_offset = addr - best_addr;
    ret = 0;

out:
    spinlock_unlock(&g_debug_map_lock);
    return ret;
}

struct symbol_map_data {
    uintptr_t offset;
    char* buf;
    size_t buf_size;
    bool found;
};

/*
 * Parse a single line of the symbol map. The symbol map is generated using the following command:
 *
 *     nm --numeric-sort --defined-only --print-size --line-numbers <file>
 *
 * We're interested in lines with the following format (and we skip other lines):
 *
 *     <start> <size> [tT] <symbol_name>\t<source_file>:<line_number>
 *
 * where the source information (`\t` and everything afterwards) is optional. Note that we don't
 * actually use the line number: it describes only where the function starts, so it would be too
 * confusing.
 */
static int symbol_map_callback(const char* line, void* arg, bool* out_stop) {
    struct symbol_map_data* data = arg;
    unsigned long val;
    const char* next = line;

    /* Start address */
    if (str_to_ulong(next, 16, &val, &next) < 0)
        return 0;
    uintptr_t start = val;

    if (*next != ' ')
        return 0;
    next++;

    /* Size */
    if (str_to_ulong(next, 16, &val, &next) < 0)
        return 0;
    size_t size = val;

    if (*next != ' ')
        return 0;
    next++;

    /* Skip if we're too early; stop iteration if we're too late */
    if (start + size <= data->offset)
        return 0;
    if (data->offset < start) {
        *out_stop = true;
        return 0;
    }

    /* `t` or `T` (symbol in a text section) */
    if (*next != 't' && *next != 'T')
        return 0;
    next++;

    if (*next != ' ')
        return 0;
    next++;

    /* Symbol name */
    const char* symbol_name = next;
    next = strchr(next, '\t');
    if (next) {
        size_t symbol_name_len = next - symbol_name;
        next++;

        /* File name */
        const char* file_name = next;
        while (*next != ':' && *next != '\0') {
            /* Begin `file_name` after the last '/' encountered */
            if (*next == '/')
                file_name = next + 1;
            next++;
        }
        size_t file_name_len = next - file_name;

        snprintf(data->buf, data->buf_size, "%.*s at %.*s", (int)symbol_name_len, symbol_name,
                 (int)file_name_len, file_name);
    } else {
        /* There's no file name, the symbol name ends with null terminator */
        snprintf(data->buf, data->buf_size, "%s", symbol_name);
    }

    data->found = true;
    *out_stop = true;
    return 0;
}

/* Example output: "func_name at source_file.c" */
static int find_in_symbol_map(const char* name, uintptr_t offset, char* buf, size_t buf_size) {
    char* symbol_map_name = alloc_concat(name, /*a_len=*/-1, ".map", /*b_len=*/-1);
    if (!symbol_map_name)
        return -ENOMEM;

    struct symbol_map_data data = {
        .offset = offset,
        .buf = buf,
        .buf_size = buf_size,
        .found = false,
    };

    int ret = read_text_file_iter_lines(symbol_map_name, &symbol_map_callback, &data);
    free(symbol_map_name);

    if (ret < 0)
        return ret;

    return data.found ? 0 : -ENOENT;
}

/* Example output: "func_name at source_file.c, libpal.so+0x456" */
int debug_describe_location(uintptr_t addr, char* buf, size_t buf_size) {
    int ret;

    char* name;
    uintptr_t offset;

    ret = debug_map_find(addr, &name, &offset);
    if (ret < 0)
        return ret;

    const char* basename = name;
    for (const char* s = name; *s != '\0'; s++) {
        if (*s == '/')
            basename = s + 1;
    }

    ret = find_in_symbol_map(name, offset, buf, buf_size);
    if (ret < 0) {
        /* parsing symbol map failed, display just name and offset */
        snprintf(buf, buf_size, "%s+0x%lx", basename, offset);
    } else {
        size_t len = strlen(buf);
        snprintf(&buf[len], buf_size - len, ", %s+0x%lx", basename, offset);
    }

    free(name);
    return 0;
}
