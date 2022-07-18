; SPDX-License-Identifier: LGPL-3.0-or-later
; Copyright (C) 2022 Intel Corporation
;                    Mariusz Zaborski <oshogbo@invisiblethingslab.com>

; rdx contains a function pointer that the application should register with atexit.
; Gramine sets rdx to NULL - it does not use this feature at all.

extern    test_exit
global    _start

section   .text

_start:
    mov   rdi, rdx
    jmp   test_exit
