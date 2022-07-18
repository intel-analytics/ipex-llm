/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

#include <stdnoreturn.h>

#include "callbacks.h"
#include "log.h"

/* declare here to silence GCC's "error: no previous prototype" */
noreturn void __stack_chk_fail(void);

noreturn void __stack_chk_fail(void) {
    log_always("Stack protector: Gramine internal stack corruption detected");
    abort();
}
