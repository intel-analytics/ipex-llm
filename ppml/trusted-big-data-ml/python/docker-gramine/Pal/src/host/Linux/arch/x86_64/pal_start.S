/* SPDX-License-Identifier: LGPL-3.0-or-later */

/*
 * pal_start-x86_64.S
 *
 * This file contains architecture-specific implementation of pal_start.
 */

.text
.global pal_start
.type pal_start, @function
.align 0x10
pal_start:
    .cfi_startproc
    movq %rsp, %rdi /* 1st arg for pal_linux_main: initial RSP */
    movq %rdx, %rsi /* 2nd arg: fini callback */
    xorq %rbp, %rbp /* mark the last stack frame with RBP == 0 (for debuggers) */
    andq $~0xf, %rsp
    call pal_linux_main
    /* Unreachable. */
    .cfi_endproc
