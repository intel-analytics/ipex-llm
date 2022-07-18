/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Corporation
 *                    Michał Kowalczyk <mkow@invisiblethingslab.com>
 */

#include <limits.h>

#include "shim_internal.h"
#include "shim_table.h"

long shim_do_getrandom(char* buf, size_t count, unsigned int flags) {
    if (flags & ~(GRND_NONBLOCK | GRND_RANDOM | GRND_INSECURE))
        return -EINVAL;

    if ((flags & (GRND_INSECURE | GRND_RANDOM)) == (GRND_INSECURE | GRND_RANDOM))
        return -EINVAL;

    /* Weird, but that's what kernel does */
    if (count > INT_MAX)
        count = INT_MAX;

    if (!is_user_memory_writable(buf, count))
        return -EFAULT;

    /* In theory, DkRandomBitsRead may block on some PALs (which conflicts with GRND_NONBLOCK flag),
     * but this shouldn't be possible in practice, so we don't care.
     */
    int ret = DkRandomBitsRead(buf, count);
    if (ret < 0) {
        ret = pal_to_unix_errno(ret);
        if (ret == -EINTR) {
            ret = -ERESTARTSYS;
        }
        return ret;
    }

    return count;
}
