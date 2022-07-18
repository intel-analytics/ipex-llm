/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * This file describes Gramine's entrypoints from userspace.
 *
 * The userspace wrappers for these functions are defined in `gramine_entry_api.h`.
 */

#ifndef SHIM_ENTRY_H_
#define SHIM_ENTRY_H_

/*!
 * \brief LibOS syscall emulation entrypoint
 *
 * Actual implementation and ABI are architecture-specific, but generally should dump the CPU
 * context and call `shim_emulate_syscall`.
 */
void syscalldb(void);

/*!
 * \brief LibOS custom call entrypoint
 *
 * Invoked through the syscall interface with GRAMINE_CALL_SYSCALL_NR. The call numbers are defined
 * in `gramine_entry_api.h`.
 */
long handle_libos_call(int number, unsigned long arg1, unsigned long arg2);

#endif /* SHIM_ENTRY_H_ */
