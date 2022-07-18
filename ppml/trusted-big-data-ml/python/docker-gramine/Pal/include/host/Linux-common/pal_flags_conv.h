/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Invisible Things Lab
 *                    Michał Kowalczyk <mkow@invisiblethingslab.com>
 * Copyright (C) 2021 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

/*
 * This file defines conversions between various PAL API flags and the flags used by Linux syscalls.
 *
 * The counterpart of this file is shim_flag_conv.h in LibOS.
 */

#ifndef PAL_FLAGS_CONV_H
#define PAL_FLAGS_CONV_H

#include <asm/fcntl.h>
#include <linux/mman.h>
#include <sys/types.h>

#include "assert.h"
#include "pal.h"

static inline int PAL_MEM_FLAGS_TO_LINUX(pal_alloc_flags_t alloc_type, pal_prot_flags_t prot) {
    assert(WITHIN_MASK(alloc_type, PAL_ALLOC_MASK));
    assert(WITHIN_MASK(prot,       PAL_PROT_MASK));

    return (alloc_type & PAL_ALLOC_RESERVE ? MAP_NORESERVE | MAP_UNINITIALIZED : 0) |
           (prot & PAL_PROT_WRITECOPY      ? MAP_PRIVATE : MAP_SHARED);
}

static inline int PAL_PROT_TO_LINUX(pal_prot_flags_t prot) {
    assert(WITHIN_MASK(prot, PAL_PROT_MASK));
    return (prot & PAL_PROT_READ  ? PROT_READ  : 0) |
           (prot & PAL_PROT_WRITE ? PROT_WRITE : 0) |
           (prot & PAL_PROT_EXEC  ? PROT_EXEC  : 0);
}

static inline int PAL_ACCESS_TO_LINUX_OPEN(enum pal_access access) {
    switch (access) {
        case PAL_ACCESS_RDONLY:
            return O_RDONLY;
        case PAL_ACCESS_WRONLY:
            return O_WRONLY;
        case PAL_ACCESS_RDWR:
            return O_RDWR;
        default:
            BUG();
    }
}

static inline int PAL_CREATE_TO_LINUX_OPEN(enum pal_create_mode create) {
    switch (create) {
        case PAL_CREATE_NEVER:
            return 0;
        case PAL_CREATE_TRY:
            return O_CREAT;
        case PAL_CREATE_ALWAYS:
            return O_CREAT | O_EXCL;
        case PAL_CREATE_IGNORED:
        default:
            BUG();
    }
}

static inline int PAL_OPTION_TO_LINUX_OPEN(pal_stream_options_t options) {
    assert(WITHIN_MASK(options, PAL_OPTION_CLOEXEC | PAL_OPTION_NONBLOCK | PAL_OPTION_PASSTHROUGH));
    return (options & PAL_OPTION_CLOEXEC  ? O_CLOEXEC  : 0) |
           (options & PAL_OPTION_NONBLOCK ? O_NONBLOCK : 0);
}

#endif /* PAL_FLAGS_CONV_H */
