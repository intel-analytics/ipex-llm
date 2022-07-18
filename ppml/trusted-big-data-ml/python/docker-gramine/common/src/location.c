/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

#include "api.h"
#include "callbacks.h"

void default_describe_location(uintptr_t addr, char* buf, size_t buf_size) {
    snprintf(buf, buf_size, "0x%lx", addr);
}

void describe_location(uintptr_t addr, char* buf, size_t buf_size)
    __attribute__((weak, alias("default_describe_location")));
