/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains host-specific code related to linking and reporting ELFs to debugger.
 */

#include <asm/errno.h>

#include "pal_rtld.h"

void _DkDebugMapAdd(const char* name, void* addr) {}

void _DkDebugMapRemove(void* addr) {}

int _DkDebugDescribeLocation(uintptr_t addr, char* buf, size_t buf_size) {
    __UNUSED(addr);
    __UNUSED(buf);
    __UNUSED(buf_size);
    return -ENOSYS;
}
