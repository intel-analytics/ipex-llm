/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains x86_64-specific functions of the PAL loader.
 */

#include <asm/prctl.h>

#include "cpu.h"
#include "pal_linux.h"

int _DkSegmentBaseGet(enum pal_segment_reg reg, uintptr_t* addr) {
    switch (reg) {
        case PAL_SEGMENT_FS:
            return unix_to_pal_error(DO_SYSCALL(arch_prctl, ARCH_GET_FS, (unsigned long*)addr));
        case PAL_SEGMENT_GS:
            // The GS segment is used for the internal TCB of PAL
            return -PAL_ERROR_DENIED;
        default:
            return -PAL_ERROR_INVAL;
    }
}

int _DkSegmentBaseSet(enum pal_segment_reg reg, uintptr_t addr) {
    switch (reg) {
        case PAL_SEGMENT_FS:
            return unix_to_pal_error(DO_SYSCALL(arch_prctl, ARCH_SET_FS, (unsigned long)addr));
        case PAL_SEGMENT_GS:
            // The GS segment is used for the internal TCB of PAL
            return -PAL_ERROR_DENIED;
        default:
            return -PAL_ERROR_INVAL;
    }
}

int _DkCpuIdRetrieve(uint32_t leaf, uint32_t subleaf, uint32_t values[4]) {
    cpuid(leaf, subleaf, values);
    return 0;
}
