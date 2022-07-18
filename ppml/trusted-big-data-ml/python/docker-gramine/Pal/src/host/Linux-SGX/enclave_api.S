# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2022 Intel Corporation
#                    Paweł Marczewski <pawel@invisiblethingslab.com>

#include "sgx_arch.h"

    .global sgx_report
    .type sgx_report, @function
sgx_report:
    .cfi_startproc
    # int sgx_report(const sgx_target_info_t* targetinfo, const void* reportdata,
    #                sgx_report_t* report);

    movq %rbx, %r10
    .cfi_register %rbx, %r10

    # EAX = EREPORT
    # RBX = targetinfo
    # RCX = reportdata
    # RDX = report (already set)
    movl $EREPORT, %eax
    movq %rdi, %rbx
    movq %rsi, %rcx

    ENCLU

    # return 0
    xorq %rax, %rax

    movq %r10, %rbx
    .cfi_restore %rbx
    ret
    .cfi_endproc

    .global sgx_getkey
    .type sgx_getkey, @function
sgx_getkey:
    .cfi_startproc
    # int64_t sgx_getkey(sgx_key_request_t* keyrequest, sgx_key_128bit_t* key);

    movq %rbx, %r10
    .cfi_register %rbx, %r10

    # EAX = EGETKEY
    # RBX = keyrequest
    # RCX = key
    movl $EGETKEY, %eax
    movq %rdi, %rbx
    movq %rsi, %rcx

    ENCLU

    # EGETKEY reports status in RAX, we return it

    movq %r10, %rbx
    .cfi_restore %rbx
    ret
    .cfi_endproc
