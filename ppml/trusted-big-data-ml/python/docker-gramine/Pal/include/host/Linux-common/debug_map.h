/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * Internal debug maps, used to communicate with GDB.
 *
 * This functionality is placed in Linux-common to support setups in which the debug maps are
 * maintained in an "outer" binary instead of the main PAL binary.
 */

#ifndef DEBUG_MAP_H
#define DEBUG_MAP_H

#include <stddef.h>
#include <stdint.h>

/* TODO: We don't really use the fact that `addr` is a pointer. It should be converted to
 * `uintptr_t` (here, and in all APIs that handle debug maps). */

struct debug_map {
    char* name;
    void* addr;

    struct debug_map* _Atomic next;
};

extern struct debug_map* _Atomic g_debug_map;

/* GDB will set a breakpoint on this function. */
void debug_map_update_debugger(void);

int debug_map_add(const char* name, void* addr);
int debug_map_remove(void* addr);

/* Initialize `g_debug_map` with executable files already loaded by the system (main program, and
 * dynamic libraries). Processes `/proc/self/maps`. */
int debug_map_init_from_proc_maps(void);

/* Try to describe code location. Looks up the right debug map, and runs `addr2line` in a
 * subprocess. */
int debug_describe_location(uintptr_t addr, char* buf, size_t buf_size);

#endif /* DEBUG_MAP_H */
