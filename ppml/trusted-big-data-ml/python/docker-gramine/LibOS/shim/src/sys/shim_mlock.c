/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2022 Integritee AG
 *                    Frieder Paape <frieder@integritee.network>
 */

/*
 * Implements mlock, mlock2, munlock, mlockall, munlockall (lock and unlock memory). These syscalls
 * are stubbed to always return success -- Gramine cannot guarantee that the host OS will perform
 * lock/unlock anyway, and a malicious OS can still swap pages anyway.
 *
 * This (dummy) functionality is required by .NET workloads.
 */

#include <asm/mman.h>

#include "api.h"
#include "shim_table.h"

long shim_do_mlock(unsigned long start, size_t len) {
    if (!access_ok((void*)start, len)) {
        return -EINVAL;
    }
    return 0;
}

long shim_do_munlock(unsigned long start, size_t len) {
    if (!access_ok((void*)start, len)) {
        return -EINVAL;
    }
    return 0;
}

long shim_do_mlockall(int flags) {
    int unknown = flags & ~(MCL_CURRENT | MCL_FUTURE | MCL_ONFAULT);
    if (unknown != 0) {
        log_warning("Syscall mlockall was called with unknown flag(s): %#x\n", unknown);
        return -EINVAL;
    }

    return 0;
}

long shim_do_munlockall(void) {
    return 0;
}

long shim_do_mlock2(unsigned long start, size_t len, int flags) {
    int unknown = flags & ~MLOCK_ONFAULT;
    if (unknown != 0) {
        log_warning("Syscall mlock2 was called with unknown flag(s): %#x\n", unknown);
        return -EINVAL;
    }

    if (!access_ok((void*)start, len)) {
        return -EINVAL;
    }

    return 0;
}
