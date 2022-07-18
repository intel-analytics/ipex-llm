; SPDX-License-Identifier: LGPL-3.0-or-later
; Copyright (C) 2022 Intel Corporation
;                    Mariusz Zaborski <oshogbo@invisiblethingslab.com>

; Verify envp which should be in 'rsp + 8 + 8 * (argc + 1)'
; In this test we don't pass any additional arguments so argc is 1.

default   rel

extern    test_exit
extern    test_str_neq

global    _start

%define    argc     1

section   .text

_start:
    lea   rdi, [env0]
    mov   rsi, [rsp + 8 + 8 * (argc + 1)]
    call  test_str_neq
    mov   rdi, rax
    cmp   rdi, 1
    je    test_exit

    lea   rdi, [env1]
    mov   rsi, [rsp + 8 + 8 * (argc + 2)]
    call  test_str_neq
    mov   rdi, rax
    cmp   rdi, 1
    je    test_exit

    mov   rdi, [rsp + 8 + 8 * (argc + 3)]
    jmp   test_exit

section   .data

env0    db    "foo=bar", 0x00
env1    db    "env0=val0", 0x00
