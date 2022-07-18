/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * This file contains code for calling Gramine from userspace. It can be used in patched
 * applications and libraries (e.g. glibc).
 *
 * To use, include this file in patched code and replace SYSCALL instructions with invocations of
 * GRAMINE_SYSCALL assembly macro.
 */

#ifndef GRAMINE_ENTRY_API_H_
#define GRAMINE_ENTRY_API_H_

/* Offsets for GS register at which entry vectors can be found */
#define GRAMINE_SYSCALL_OFFSET 24

#ifdef __ASSEMBLER__

.macro GRAMINE_SYSCALL
leaq .Lafter_gramine_syscall\@(%rip), %rcx
jmpq *%gs:GRAMINE_SYSCALL_OFFSET
.Lafter_gramine_syscall\@:
.endm

#else /* !__ASSEMBLER__ */

#define GRAMINE_STR(x)  #x
#define GRAMINE_XSTR(x) GRAMINE_STR(x)

__asm__(
    ".macro GRAMINE_SYSCALL\n"
    "leaq .Lafter_gramine_syscall\\@(%rip), %rcx\n"
    "jmpq *%gs:" GRAMINE_XSTR(GRAMINE_SYSCALL_OFFSET) "\n"
    ".Lafter_gramine_syscall\\@:\n"
    ".endm\n"
);

#undef GRAMINE_XSTR
#undef GRAMINE_STR

/* Custom call numbers */
enum {
    GRAMINE_CALL_REGISTER_LIBRARY = 1,
    GRAMINE_CALL_RUN_TEST,
};

/* Magic syscall number for Gramine custom calls */
#define GRAMINE_CUSTOM_SYSCALL_NR 0x100000

static inline int gramine_call(int number, unsigned long arg1, unsigned long arg2) {
    int ret;
    __asm__ volatile(
        "GRAMINE_SYSCALL"
        : "=a"(ret)
        : "0"(GRAMINE_CUSTOM_SYSCALL_NR), "D"(number), "S"(arg1), "d"(arg2)
        : "rcx", "r11", "memory");
    return ret;
}

static inline int gramine_register_library(const char* name, unsigned long load_address) {
    return gramine_call(GRAMINE_CALL_REGISTER_LIBRARY, (unsigned long)name, load_address);
}

static inline int gramine_run_test(const char* test_name) {
    return gramine_call(GRAMINE_CALL_RUN_TEST, (unsigned long)test_name, 0);
}

#endif /* __ASSEMBLER__ */

#endif /* GRAMINE_ENTRY_API_H_ */
