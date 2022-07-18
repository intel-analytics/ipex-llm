/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University
 * Copyright (C) 2021 Intel Labs
 */

/*
 * This file contains host-specific code related to linking and reporting ELFs to debugger.
 *
 * Overview of ELF files used in this host:
 * - libpal.so - used as main executable, so it doesn't need to be reported separately
 * - vDSO - virtual library loaded by host Linux, doesn't need to be reported
 * - LibOS, application, libc... - reported through DkDebugMap*
 */

#include "api.h"
#include "debug_map.h"
#include "elf/elf.h"
#include "pal_internal.h"
#include "pal_linux.h"
#include "pal_rtld.h"

uintptr_t g_vdso_start = 0;
uintptr_t g_vdso_end = 0;

bool is_in_vdso(uintptr_t addr) {
    return (g_vdso_start || g_vdso_end) && g_vdso_start <= addr && addr < g_vdso_end;
}

void _DkDebugMapAdd(const char* name, void* addr) {
    int ret = debug_map_add(name, addr);
    if (ret < 0)
        log_error("debug_map_add(%s, %p) failed: %d", name, addr, ret);
}

void _DkDebugMapRemove(void* addr) {
    int ret = debug_map_remove(addr);
    if (ret < 0)
        log_error("debug_map_remove(%p) failed: %d", addr, ret);
}

/* populate g_pal_linux_state.vdso_clock_gettime based on vDSO */
int setup_vdso(elf_addr_t base_addr) {
    int ret;

    const char* string_table  = NULL;
    elf_sym_t* symbol_table = NULL;
    uint32_t symbol_table_cnt = 0;

    ret = find_string_and_symbol_tables(base_addr, base_addr, &string_table, &symbol_table,
                                        &symbol_table_cnt);
    if (ret < 0) {
        log_warning("The VDSO unexpectedly doesn't have string table or symbol table.");
        return 0;
    }

    /* iterate through the symbol table and find where clock_gettime vDSO func is located */
    for (uint32_t i = 0; i < symbol_table_cnt; i++) {
        const char* symbol_name = string_table + symbol_table[i].st_name;
        if (!strcmp("__vdso_clock_gettime", symbol_name)) {
            g_pal_linux_state.vdso_clock_gettime = (void*)(base_addr + symbol_table[i].st_value);
            break;
        }
    }

    return 0;
}

int _DkDebugDescribeLocation(uintptr_t addr, char* buf, size_t buf_size) {
    return debug_describe_location(addr, buf, buf_size);
}
