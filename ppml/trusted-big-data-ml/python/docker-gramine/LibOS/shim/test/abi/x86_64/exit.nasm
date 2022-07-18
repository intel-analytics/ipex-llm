; SPDX-License-Identifier: LGPL-3.0-or-later
; Copyright (C) 2022 Intel Corporation
;                    Mariusz Zaborski <oshogbo@invisiblethingslab.com>

global    test_exit

%define   __NR_exit   60

section   .text

test_exit:
    xor   rax, rax
    test  rdi, rdi
    setne al
    mov   rdi, rax
    mov   rax, __NR_exit
    syscall
