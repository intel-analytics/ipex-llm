/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University
 * Copyright (C) 2021 Intel Labs
 */

/*
 * This file contains host-specific code related to linking and reporting ELFs to debugger.
 *
 * Overview of ELF files used in this host:
 * - pal-sgx and libraries it uses (outside enclave) - handled by ld.so and reported by it (through
 *   _r_debug mechanism)
 * - libpal.so (in enclave) - reported in sgx_main.c before enclave start
 * - LibOS, application, libc... (in enclave) - reported through DkDebugMap*
 *
 * In addition, we report executable memory mappings to the profiling subsystem.
 */

#include "pal_linux.h"
#include "pal_rtld.h"

void _DkDebugMapAdd(const char* name, void* addr) {
    ocall_debug_map_add(name, addr);
}

void _DkDebugMapRemove(void* addr) {
    ocall_debug_map_remove(addr);
}

int _DkDebugDescribeLocation(uintptr_t addr, char* buf, size_t buf_size) {
    return ocall_debug_describe_location(addr, buf, buf_size);
}
