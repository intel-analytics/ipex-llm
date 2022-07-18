; SPDX-License-Identifier: LGPL-3.0-or-later
; Copyright (C) 2022 Intel Corporation
;                    Mariusz Zaborski <oshogbo@invisiblethingslab.com>

; The MXCSR register:
; +--+--+--+--+--+--+--+--+--+---+--+--+--+--+--+--+
; |FZ|   RC|PM|UM|OM|ZM|DM|IM|DAZ|PE|UE|OE|ZE|DE|IE|
; +--+--+--+--+--+--+--+--+--+---+--+--+--+--+--+--+
; The MXCSR should be set as follow:
; +--+--+--+--+--+--+--+--+--+---+--+--+--+--+--+--+
; | 0| 0  0| 1| 1| 1| 1| 1| 1|  0| X| X| X| X| X| X|
; +--+--+--+--+--+--+--+--+--+---+--+--+--+--+--+--+

extern    test_exit

global    _start

section   .text

_start:
    stmxcsr [rsp]

    mov     edi, [rsp]
    and     rdi, 0b1111111111000000
    xor     rdi, 0b0001111110000000

    jmp     test_exit
