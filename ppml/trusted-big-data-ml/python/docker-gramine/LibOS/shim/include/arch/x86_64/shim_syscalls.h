/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Corporation
 *                    Michał Kowalczyk <mkow@invisiblethingslab.com>
 */

#ifndef SHIM_SYSCALLS_H_
#define SHIM_SYSCALLS_H_

/* TODO: This is required when building Gramine on systems with older headers. We should actually
 * have our own copy of headers of the kernel we emulate, not from the one which is used on the
 * build machine.
 */
#ifndef __NR_copy_file_range
#define __NR_copy_file_range 326
#endif
#ifndef __NR_preadv2
#define __NR_preadv2 327
#endif
#ifndef __NR_pwritev2
#define __NR_pwritev2 328
#endif
#ifndef __NR_pkey_mprotect
#define __NR_pkey_mprotect 329
#endif
#ifndef __NR_pkey_alloc
#define __NR_pkey_alloc 330
#endif
#ifndef __NR_pkey_free
#define __NR_pkey_free 331
#endif
#ifndef __NR_statx
#define __NR_statx 332
#endif
#ifndef __NR_io_pgetevents
#define __NR_io_pgetevents 333
#endif
#ifndef __NR_rseq
#define __NR_rseq 334
#endif
#ifndef __NR_pidfd_send_signal
#define __NR_pidfd_send_signal 424
#endif
#ifndef __NR_io_uring_setup
#define __NR_io_uring_setup 425
#endif
#ifndef __NR_io_uring_enter
#define __NR_io_uring_enter 426
#endif
#ifndef __NR_io_uring_register
#define __NR_io_uring_register 427
#endif
#ifndef __NR_syscalls
#define __NR_syscalls 428
#endif

#endif /* SHIM_SYSCALLS_H_ */
