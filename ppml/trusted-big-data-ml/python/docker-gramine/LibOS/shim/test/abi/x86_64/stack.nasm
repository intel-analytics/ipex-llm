; SPDX-License-Identifier: LGPL-3.0-or-later
; Copyright (C) 2022 Intel Corporation
;                    Mariusz Zaborski <oshogbo@invisiblethingslab.com>

; "The end of the input argument area shall be aligned on a 16..."

extern    test_exit

global    _start

section   .text

_start:
    xor   rdi, rdi

    mov   rax, rsp
    and   rax, 0xF
    setne dil

    jmp   test_exit
