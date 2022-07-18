/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

/*
 * This file contains the entry point of system call table in library OS (the function syscalldb()).
 *
 * The below entry point implementation first saves the CPU context of the current application
 * thread on the thread's LibOS stack, then calls the LibOS syscall-emulation function, which, upon
 * returning, calls context restoring function, which passes control back to the application.
 * The context consists of GPRs, FP control word (fpcw) and the SSE/AVX/... control word (mxcsr).
 *
 * Note that LibOS may clobber all FP/SSE/AVX/... (extended) state except the control words. We rely
 * on the fact that applications do *not* assume that this extended state is preserved across system
 * calls. Indeed, the extended state (bar control words) is explicitly described as *not* preserved
 * by the System V ABI, and though syscall ABI is not the same as System V ABI, we assume that no
 * sane application issues syscalls in a non-System-V compliant manner. See System V ABI docs
 * (https://uclibc.org/docs/psABI-x86_64.pdf), "Register Usage" for more information.
 */

#include "asm-offsets.h"

.extern shim_emulate_syscall
.extern shim_xstate_size
.extern shim_xstate_restore

.global syscalldb
.type syscalldb, @function
syscalldb:
    # On entry to this function rcx contains the return value (next instruction after syscall),
    # all other registers can have arbitrary values.
    # We have to be very careful with executed instructions not to change any flags until they
    # are saved!
    .cfi_startproc
    .cfi_def_cfa %rsp, 0
    .cfi_register %rip, %rcx

    # We can clobber r11 as it will be set to rflags later on.
    mov %rsp, %r11
    .cfi_undefined %r11
    .cfi_register %rsp, %r11
    .cfi_def_cfa_register %r11
    mov %gs:(SHIM_TCB_OFF + SHIM_TCB_LIBOS_STACK_OFF), %rsp

    # Create PAL_CONTEXT struct on the stack.

    # reserve space for mxcsr + fpcw + is_fpregs_used
    pushq $0

    # fpregs, but for now we use this to store rax - to get a scratch register
    push %rax

    # err + trapno + oldmask + cr2 are cleared for a syscall frame
    mov $0, %eax
    push %rax
    push %rax
    push %rax
    push %rax

    # csgsfsss - default value, as we do not support changing it
    mov $(0x2b << 48 | 0x33), %rax
    push %rax

    # after this we can use instructions changing flags
    pushfq
    # Debuggers use Trap Flag (TF) of EFLAGS to do single-stepping - otherwise it is unused by normal
    # applications. If the previous instruction was single-stepped, it stored TF, so reset it here.
    andq $~0x100, (%rsp)

    # Set default rflags value (just IF set).
    pushq $0x202
    popfq

    push %rcx # rip
    push %r11 # rsp
    .cfi_def_cfa %rsp, 0x50
    .cfi_rel_offset %rsp, 0
    .cfi_rel_offset %rip, 8

    # Set r11 to rflags
    mov 0x10(%rsp), %r11

    push %rcx
    .cfi_adjust_cfa_offset 8
    .cfi_rel_offset %rcx, 0

    # rax was saved in fpregs, save it in proper place now, fpregs will be populated later
    pushq 0x48(%rsp)
    .cfi_adjust_cfa_offset 8
    .cfi_rel_offset %rax, 0

    push %rdx
    .cfi_adjust_cfa_offset 8
    .cfi_rel_offset %rdx, 0
    push %rbx
    .cfi_adjust_cfa_offset 8
    .cfi_rel_offset %rbx, 0
    push %rbp
    .cfi_adjust_cfa_offset 8
    .cfi_rel_offset %rbp, 0
    push %rsi
    .cfi_adjust_cfa_offset 8
    .cfi_rel_offset %rsi, 0
    push %rdi
    .cfi_adjust_cfa_offset 8
    .cfi_rel_offset %rdi, 0
    push %r15
    .cfi_adjust_cfa_offset 8
    .cfi_rel_offset %r15, 0
    push %r14
    .cfi_adjust_cfa_offset 8
    .cfi_rel_offset %r14, 0
    push %r13
    .cfi_adjust_cfa_offset 8
    .cfi_rel_offset %r13, 0
    push %r12
    .cfi_adjust_cfa_offset 8
    .cfi_rel_offset %r12, 0
    push %r11
    .cfi_adjust_cfa_offset 8
    .cfi_rel_offset %r11, 0
    push %r10
    .cfi_adjust_cfa_offset 8
    .cfi_rel_offset %r10, 0
    push %r9
    .cfi_adjust_cfa_offset 8
    .cfi_rel_offset %r9, 0
    push %r8
    .cfi_adjust_cfa_offset 8
    .cfi_rel_offset %r8, 0
    # PAL_CONTEXT struct ends here.

    mov %rsp, %r15
    .cfi_def_cfa_register %r15

    and $~0xF, %rsp # Required by System V AMD64 ABI.

    # save FP Control Word & MXCSR into current thread's TCB
    stmxcsr PAL_CONTEXT_MXCSR_OFF(%r15)
    fnstcw PAL_CONTEXT_FPCW_OFF(%r15)

    # fpregs is not populated, so is_fpregs_used should be 0.
    movb $0, PAL_CONTEXT_FPREGS_USED_OFF(%r15)

    call shim_xstate_size
    sub %rax, %rsp                          # allocate space for xstate
    and $~(SHIM_XSTATE_ALIGN - 1), %rsp
    mov %rsp, PAL_CONTEXT_FPREGS_OFF(%r15)

    and $~0xF, %rsp # Required by System V AMD64 ABI.
    xor %ebp, %ebp

#ifdef DEBUG
    # Pretend that this function (`syscalldb`) is called from somewhere inside a function called
    # `__morestack`. This is the only way to have a backtrace in GDB spanning from LibOS/Pal,
    # through `syscalldb` to the user application code/libc, because GDB does not handle switching
    # stacks in the middle of backtrace, unless the function doing it is called `__morestack`.
    # Thanks GDB!
    # Technical details: we load an address somewhere inside `__morestack` (this cannot be the first
    # instruction in there) into r14 (a callee-saved register) and mark it as holding old rip. This
    # way GDB thinks `syscalldb` was called by `__morestack`. Inside `__morestack` we mark all
    # registers as having the same value as in the previous frame (basically a no-op frame). Now GDB
    # sees a backtrace: `user_function` -> `__morestack` -> `syscalldb`, with `__morestack` having
    # the same stack value as `user_function` and `syscalldb` having the new stack value. This makes
    # GDB happy and it prints correct backtrace across all these functions, which is what we are
    # after with all this madness.
    lea Lmorestack_for_gdb_bt(%rip), %r14
    .cfi_register %rip, %r14
#endif

    mov %r15, %rdi
    call shim_emulate_syscall # this does not return

    # Just to make return address point inside this function.
    ud2

    .cfi_endproc
.size syscalldb, .-syscalldb

#ifdef DEBUG
.global __morestack
.type __morestack, @function
__morestack:
    .cfi_startproc
    .cfi_register %rip, %rcx
    .cfi_same_value %r8
    .cfi_same_value %r9
    .cfi_same_value %r10
    .cfi_same_value %r11
    .cfi_same_value %r12
    .cfi_same_value %r13
    .cfi_same_value %r14
    .cfi_same_value %r15
    .cfi_same_value %rdi
    .cfi_same_value %rsi
    .cfi_same_value %rbp
    .cfi_same_value %rbx
    .cfi_same_value %rdx
    .cfi_same_value %rax
    .cfi_same_value %rcx
    .cfi_same_value %rsp

    nop
Lmorestack_for_gdb_bt:
    nop

    .cfi_endproc
.size __morestack, .-__morestack
#endif

.global _return_from_syscall
.type _return_from_syscall, @function
_return_from_syscall:
    # expects one argument (in `rdi`) - pointer to PAL_CONTEXT
    .cfi_startproc

    mov %rdi, %rbx

    movb PAL_CONTEXT_FPREGS_USED_OFF(%rbx), %al
    test %al, %al
    jne .Lrestore_xstate

    # restore FP Control Word & MXCSR from TCB
    fldcw PAL_CONTEXT_FPCW_OFF(%rbx)
    ldmxcsr PAL_CONTEXT_MXCSR_OFF(%rbx)

.Lrestore_context:
    # After this line cfi will be broken, but we don't care much since this does not call anything
    # and just restores the user context, so it will not be visible in any backtrace.
    # Note that fixing it is not trivial - we would need the trick with `__morestack`, but we do not
    # have neither a stack, nor a scratch register.
    mov %rbx, %rsp

    pop %r8
    pop %r9
    pop %r10
    pop %r11
    pop %r12
    pop %r13
    pop %r14
    pop %r15
    pop %rdi
    pop %rsi
    pop %rbp
    pop %rbx
    pop %rdx

    # exchange rcx with rip
    mov 0x8(%rsp), %rcx
    mov 0x18(%rsp), %rax
    mov %rcx, 0x18(%rsp)
    mov %rax, 0x8(%rsp)
    # exchange rsp with flags
    mov 0x10(%rsp), %rcx
    mov 0x20(%rsp), %rax
    mov %rcx, 0x20(%rsp)
    mov %rax, 0x10(%rsp)

    pop %rax
    pop %rcx # rip
    popfq
    mov %rcx, %gs:(SHIM_TCB_OFF + SHIM_TCB_SCRATCH_PC_OFF)
    pop %rcx
    pop %rsp
    jmp *%gs:(SHIM_TCB_OFF + SHIM_TCB_SCRATCH_PC_OFF)

.Lrestore_xstate:
    mov PAL_CONTEXT_FPREGS_OFF(%rbx), %rdi
    call shim_xstate_restore
    jmp .Lrestore_context

    .cfi_endproc
.size _return_from_syscall, .-_return_from_syscall
