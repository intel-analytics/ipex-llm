/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains code for registering libraries for debugger integration. We report the
 * libraries to PAL using `DkDebugMapAdd`/`DkDebugMapRemove`.
 *
 * We also keep our own list of reported libraries, so that we can copy them to the child process,
 * or remove them all (in case of `exec`).
 */

#include "list.h"
#include "pal.h"
#include "shim_checkpoint.h"
#include "shim_lock.h"
#include "shim_types.h"
#include "shim_utils.h"

void shim_describe_location(uintptr_t addr, char* buf, size_t buf_size) {
    DkDebugDescribeLocation(addr, buf, buf_size);
}

#ifndef DEBUG

int init_r_debug(void) {
    return 0;
}

void clean_link_map_list(void) {
    /* do nothing */
}

void remove_r_debug(void* addr) {
    __UNUSED(addr);
    /* do nothing */
}

void append_r_debug(const char* uri, void* addr) {
    __UNUSED(uri);
    __UNUSED(addr);
    /* do nothing */
}

#else /* !DEBUG */

DEFINE_LISTP(gdb_link_map);
DEFINE_LIST(gdb_link_map);
struct gdb_link_map {
    void* l_addr;
    char* l_name;

    LIST_TYPE(gdb_link_map) list;
};

static LISTP_TYPE(gdb_link_map) g_link_map_list = LISTP_INIT;
static struct shim_lock g_link_map_list_lock;

int init_r_debug(void) {
    return create_lock(&g_link_map_list_lock);
}

void clean_link_map_list(void) {
    lock(&g_link_map_list_lock);

    struct gdb_link_map* m;
    struct gdb_link_map* tmp;
    LISTP_FOR_EACH_ENTRY_SAFE(m, tmp, &g_link_map_list, list) {
        LISTP_DEL(m, &g_link_map_list, list);
        DkDebugMapRemove(m->l_addr);
        free(m);
    }

    unlock(&g_link_map_list_lock);
}

void remove_r_debug(void* addr) {
    lock(&g_link_map_list_lock);

    struct gdb_link_map* m;
    bool found = false;
    LISTP_FOR_EACH_ENTRY(m, &g_link_map_list, list) {
        if (m->l_addr == addr) {
            found = true;
            break;
        }
    }

    if (!found) {
        /* No map found. This is normal because we call remove_r_debug() on every munmap
         * operation. */
        goto out;
    }

    log_debug("%s: removing %s at %p", __func__, m->l_name, addr);
    LISTP_DEL(m, &g_link_map_list, list);
    DkDebugMapRemove(addr);
    free(m);

out:
    unlock(&g_link_map_list_lock);
}

void append_r_debug(const char* uri, void* addr) {
    lock(&g_link_map_list_lock);

    struct gdb_link_map* new = malloc(sizeof(struct gdb_link_map));
    if (!new) {
        log_warning("%s: couldn't allocate map", __func__);
        goto out;
    }

    char* new_uri = strdup(uri);
    if (!new_uri) {
        log_warning("%s: couldn't allocate uri", __func__);
        goto out;
    }

    new->l_addr = addr;
    new->l_name = new_uri;

    log_debug("%s: adding %s at %p", __func__, uri, addr);
    LISTP_ADD_TAIL(new, &g_link_map_list, list);
    DkDebugMapAdd(uri, addr);

out:
    unlock(&g_link_map_list_lock);
}

BEGIN_CP_FUNC(gdb_map) {
    __UNUSED(obj);
    __UNUSED(size);
    __UNUSED(objp);

    lock(&g_link_map_list_lock);

    struct gdb_link_map* m;
    LISTP_FOR_EACH_ENTRY(m, &g_link_map_list, list) {
        size_t off = ADD_CP_OFFSET(sizeof(struct gdb_link_map));
        struct gdb_link_map* newm = (struct gdb_link_map*)(base + off);

        newm->l_addr = m->l_addr;

        size_t len   = strlen(m->l_name);
        newm->l_name = (char*)(base + ADD_CP_OFFSET(len + 1));
        memcpy(newm->l_name, m->l_name, len + 1);

        INIT_LIST_HEAD(newm, list);

        ADD_CP_FUNC_ENTRY(off);
    }

    unlock(&g_link_map_list_lock);
}
END_CP_FUNC(gdb_map)

BEGIN_RS_FUNC(gdb_map) {
    __UNUSED(offset);
    struct gdb_link_map* m = (void*)(base + GET_CP_FUNC_ENTRY());

    CP_REBASE(m->l_name);

    LISTP_ADD_TAIL(m, &g_link_map_list, list);
    DkDebugMapAdd(m->l_name, m->l_addr);
}
END_RS_FUNC(gdb_map)

#endif
