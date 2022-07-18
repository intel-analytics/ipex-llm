/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Wojtek Porczyk <woju@invisiblethingslab.com>
 */

#ifndef SGX_PROCESS_H
#define SGX_PROCESS_H

#include <stddef.h>

int sgx_create_process(size_t nargs, const char** args, const char* manifest, int* out_stream_fd);

#endif /* SGX_PROCESS_H */
