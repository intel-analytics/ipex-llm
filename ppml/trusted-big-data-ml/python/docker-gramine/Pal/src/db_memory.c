/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains APIs that allocate, free or protect virtual memory.
 */

#include "api.h"
#include "pal.h"
#include "pal_error.h"
#include "pal_internal.h"

int DkVirtualMemoryAlloc(void** addr_ptr, PAL_NUM size, pal_alloc_flags_t alloc_type,
                         pal_prot_flags_t prot) {
    assert(addr_ptr);
    void* map_addr = *addr_ptr;

    if ((map_addr && !IS_ALLOC_ALIGNED_PTR(map_addr)) || !size || !IS_ALLOC_ALIGNED(size)) {
        return -PAL_ERROR_INVAL;
    }

    if (map_addr && _DkCheckMemoryMappable(map_addr, size)) {
        return -PAL_ERROR_DENIED;
    }

    if ((alloc_type & PAL_ALLOC_INTERNAL) && map_addr) {
        return -PAL_ERROR_INVAL;
    }

    if (!(alloc_type & PAL_ALLOC_INTERNAL) && !map_addr) {
        return -PAL_ERROR_INVAL;
    }

    return _DkVirtualMemoryAlloc(addr_ptr, size, alloc_type, prot);
}

int DkVirtualMemoryFree(void* addr, PAL_NUM size) {
    if (!addr || !size) {
        return -PAL_ERROR_INVAL;
    }

    if (!IS_ALLOC_ALIGNED_PTR(addr) || !IS_ALLOC_ALIGNED(size)) {
        return -PAL_ERROR_INVAL;
    }

    if (_DkCheckMemoryMappable(addr, size)) {
        return -PAL_ERROR_DENIED;
    }

    return _DkVirtualMemoryFree(addr, size);
}

int DkVirtualMemoryProtect(void* addr, PAL_NUM size, pal_prot_flags_t prot) {
    if (!addr || !size) {
        return -PAL_ERROR_INVAL;
    }

    if (!IS_ALLOC_ALIGNED_PTR(addr) || !IS_ALLOC_ALIGNED(size)) {
        return -PAL_ERROR_INVAL;
    }

    if (_DkCheckMemoryMappable(addr, size)) {
        return -PAL_ERROR_DENIED;
    }

    return _DkVirtualMemoryProtect(addr, size, prot);
}

int add_preloaded_range(uintptr_t start, uintptr_t end, const char* comment) {
    size_t new_cnt = g_pal_public_state.preloaded_ranges_cnt + 1;
    void* new_ranges = malloc(new_cnt * sizeof(*g_pal_public_state.preloaded_ranges));
    if (!new_ranges) {
        return -PAL_ERROR_NOMEM;
    }

    if (g_pal_public_state.preloaded_ranges_cnt) {
        memcpy(new_ranges, g_pal_public_state.preloaded_ranges,
               g_pal_public_state.preloaded_ranges_cnt * sizeof(*g_pal_public_state.preloaded_ranges));
    }

    free(g_pal_public_state.preloaded_ranges);
    g_pal_public_state.preloaded_ranges = new_ranges;

    g_pal_public_state.preloaded_ranges[g_pal_public_state.preloaded_ranges_cnt].start = start;
    g_pal_public_state.preloaded_ranges[g_pal_public_state.preloaded_ranges_cnt].end = end;
    g_pal_public_state.preloaded_ranges[g_pal_public_state.preloaded_ranges_cnt].comment = comment;
    g_pal_public_state.preloaded_ranges_cnt++;

    return 0;
}
