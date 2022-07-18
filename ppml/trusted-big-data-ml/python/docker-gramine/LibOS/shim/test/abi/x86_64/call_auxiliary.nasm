; SPDX-License-Identifier: LGPL-3.0-or-later
; Copyright (C) 2022 Intel Corporation
;                    Mariusz Zaborski <oshogbo@invisiblethingslab.com>

; Verify auxiliary which should be in:
; 'rsp + 8 + (8 * (argc + 1)) + (8 * (env_count + 1))'
; In this test we don't pass any additional arguments so argc is 1,
; we also should not have any environment variables so env_count is 0.

extern    test_exit
extern    verify_auxiliary
global    _start

%define    argc      1
%define    env_count 0

section   .text

_start:
    lea   rdi, [rsp + 8 + 8 * (argc + 1) + 8 * (env_count + 1)]
    call  verify_auxiliary

    mov   rdi, rax
    jmp   test_exit
