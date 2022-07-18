; SPDX-License-Identifier: LGPL-3.0-or-later
; Copyright (C) 2022 Intel Corporation
;                    Mariusz Zaborski <oshogbo@invisiblethingslab.com>

; The x87 FPU Control Word register:
; +--+--+--+--+--+--+--+--+--+--+--+--+--+
; | X|   RC|   PC| R| R|PM|UM|OM|ZM|DM|IM|
; +--+--+--+--+--+--+--+--+--+--+--+--+--+
; The x87 FPU Control Word should be set as follow:
; +--+--+--+--+--+--+--+--+--+--+--+--+--+
; | X| 0  0| 1  1| X| X| 1| 1| 1| 1| 1| 1|
; +--+--+--+--+--+--+--+--+--+--+--+--+--+

extern    test_exit
global    _start

section   .text

_start:
    fstcw [rsp]

    mov   ax, [rsp]
    mov   rdi, rax
    and   rdi, 0b0111100111111
    xor   rdi, 0b0001100111111

    jmp   test_exit
