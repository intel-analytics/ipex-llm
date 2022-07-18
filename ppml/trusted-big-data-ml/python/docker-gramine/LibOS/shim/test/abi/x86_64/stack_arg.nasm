; SPDX-License-Identifier: LGPL-3.0-or-later
; Copyright (C) 2022 Intel Corporation
;                    Mariusz Zaborski <oshogbo@invisiblethingslab.com>

; Verify argv which should be in rsp + 8.

default   rel

extern    test_exit
extern    test_str_neq

global    _start

section   .text

_start:
    mov   rdi, [rsp]        ; Verify argc
    cmp   rdi, 3
    mov   rdi, 1
    jne   test_exit

    lea   rdi, [argv0]      ; Verify argv[0]
    mov   rsi, [rsp + 8 * 1]
    call  test_str_neq
    mov   rdi, rax
    cmp   rdi, 1
    je    test_exit

    lea   rdi, [argv1]      ; Verify argv[1]
    mov   rsi, [rsp + 8 * 2]
    call  test_str_neq
    mov   rdi, rax
    cmp   rdi, 1
    je    test_exit

    lea   rdi, [argv2]      ; Verify argv[2]
    mov   rsi, [rsp + 8 * 3]
    call  test_str_neq
    mov   rdi, rax
    cmp   rdi, 1
    je    test_exit

    mov   rdi, [rsp + 8 * 4]
    jmp   test_exit

section   .data

argv0    db    "stack_arg", 0x00
argv1    db    "foo", 0x00
argv2    db    "bar", 0x00
