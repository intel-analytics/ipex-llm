; SPDX-License-Identifier: LGPL-3.0-or-later
; Copyright (C) 2022 Intel Corporation
;                    Mariusz Zaborski <oshogbo@invisiblethingslab.com>

; The System V ABI (see section 3.4.1) expects us to set the following before jumping to the entry
; point:
;
; - RDX: function pointer to be registered with `atexit` (we pass 0)
; - RSP: the initial stack, contains program arguments and environment
; - FLAGS: should be zeroed out

global    call_elf_entry

; noreturn void call_elf_entry(elf_addr_t entry, void* argp)
call_elf_entry:
    xor     rdx, rdx ; set RDX to 0
    push    0x202
    popf             ; set lower part of rFLAGS to 0
    mov     rsp, rsi ; set stack pointer to second arg
    jmp     rdi      ; jmp to entry point (first arg)
