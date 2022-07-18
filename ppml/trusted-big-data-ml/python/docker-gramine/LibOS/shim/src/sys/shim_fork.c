/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

#include <stddef.h> // without this header we are missing `size_t` definition in "signal.h" ...
#include <linux/sched.h>
#include <linux/signal.h>

#include "shim_table.h"

long shim_do_fork(void) {
    return shim_do_clone(SIGCHLD, 0, NULL, NULL, 0);
}

long shim_do_vfork(void) {
    return shim_do_clone(CLONE_VFORK | CLONE_VM | SIGCHLD, 0, NULL, NULL, 0);
}
