/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */
#ifndef VDSO_SYSCALL_H_
#define VDSO_SYSCALL_H_

#include "gramine_entry_api.h"

#ifdef ASAN
#error This code should be compiled without AddressSanitizer.
#endif

static inline long vdso_arch_syscall(long nr, long arg1, long arg2) {
    long ret;
    __asm__ volatile(
        "lea .Lret%=(%%rip), %%rcx\n"
        "jmp *%%gs:%c[syscalldb]\n"
        ".Lret%=:\n"
        : "=a" (ret)
        : "0" (nr), "D"(arg1), "S"(arg2), [syscalldb] "i"(GRAMINE_SYSCALL_OFFSET)
        : "memory", "rcx", "r11"
    );
    return ret;
}

#endif // VDSO_SYSCALL_H_
