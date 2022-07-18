/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2018 Intel Corporation
 *                    Isaku Yamahata <isaku.yamahata at gmail.com>
 *                                   <isaku.yamahata at intel.com>
 * Copyright (C) 2020 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

#include <asm/unistd.h>

#include "vdso.h"
#include "vdso_syscall.h"

#ifdef ASAN
#error This code should be compiled without AddressSanitizer.
#endif

/*
 * The symbol below needs to be exported for libsysdb to inject those values,
 * but relocation (.rela.dyn section) isn't wanted in the code generation.
 */
#define EXPORT_SYMBOL(name) extern __typeof__(name) __vdso_##name __attribute__((alias(#name)))

#define EXPORT_WEAK_SYMBOL(name) \
    __typeof__(__vdso_##name) name __attribute__((weak, alias("__vdso_" #name)))

int __vdso_clock_gettime(clockid_t clock, struct timespec* t) {
    return vdso_arch_syscall(__NR_clock_gettime, (long)clock, (long)t);
}
EXPORT_WEAK_SYMBOL(clock_gettime);

int __vdso_gettimeofday(struct timeval* tv, struct timezone* tz) {
    return vdso_arch_syscall(__NR_gettimeofday, (long)tv, (long)tz);
}
EXPORT_WEAK_SYMBOL(gettimeofday);

time_t __vdso_time(time_t* t) {
    return vdso_arch_syscall(__NR_time, (long)t, 0);
}
EXPORT_WEAK_SYMBOL(time);

long __vdso_getcpu(unsigned* cpu, struct getcpu_cache* unused) {
    return vdso_arch_syscall(__NR_getcpu, (long)cpu, (long)unused);
}
EXPORT_WEAK_SYMBOL(getcpu);
