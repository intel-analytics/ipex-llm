/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains APIs that allocate, free or protect virtual memory.
 */

#include "api.h"
#include "pal.h"
#include "pal_error.h"
#include "pal_internal.h"

bool _DkCheckMemoryMappable(const void* addr, size_t size) {
    return true;
}

int _DkVirtualMemoryAlloc(void** addr_ptr, uint64_t size, pal_alloc_flags_t alloc_type,
                          pal_prot_flags_t prot) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

int _DkVirtualMemoryFree(void* addr, uint64_t size) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

int _DkVirtualMemoryProtect(void* addr, uint64_t size, pal_prot_flags_t prot) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

unsigned long _DkMemoryQuota(void) {
    return 0;
}

unsigned long _DkMemoryAvailableQuota(void) {
    return 0;
}
