; SPDX-License-Identifier: LGPL-3.0-or-later
; Copyright (C) 2022 Intel Corporation
;                    Mariusz Zaborski <oshogbo@invisiblethingslab.com>

; The rFlags should be cleared.
; Verify OF, DF, SF, ZF, AF, PF, CF
; The EFLAGS:
; +--+--+--+--+--+--+--+--+--+--+--+--+
; |OF|DF|IF|TF|SF|ZF| 0|AF| 0|PF| 1|CF|
; +--+--+--+--+--+--+--+--+--+--+--+--+
; Mask to get interesting bits:
; +--+--+--+--+--+--+--+--+--+--+--+--+
; | 1| 1| 0| 0| 1| 1| 0| 1| 0| 1| 0| 1|
; +--+--+--+--+--+--+--+--+--+--+--+--+

extern    test_exit

global    _start

section   .text

_start:
    pushfq
    pop   rdi

    and   rdi, 0b110011010101

    jmp   test_exit
