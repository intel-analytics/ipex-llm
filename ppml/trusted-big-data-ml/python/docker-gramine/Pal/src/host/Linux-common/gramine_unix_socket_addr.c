/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

#include <asm/errno.h>

#include "api.h"
#include "linux_utils.h"

int get_gramine_unix_socket_addr(uint64_t instance_id, const char* name,
                                  struct sockaddr_un* out_addr) {
    /* Apparently there is no way to get this define without including whole "sys/socket.h". */
    out_addr->sun_family = /*AF_UNIX*/1;

    /* We use abstract UNIX sockets, which start with a nullbyte and actually look at the whole path
     * buffer (i.e. do *not* consider the rest of the buffer as null-terminated C string, but rather
     * a fixed length byte array). */
    memset(out_addr->sun_path, 0, sizeof(out_addr->sun_path));
    int ret = snprintf(out_addr->sun_path + 1, sizeof(out_addr->sun_path) - 1, "/gramine/%lu/%s",
                       instance_id, name);
    if (ret < 0) {
        return ret;
    }
    if ((size_t)ret >= sizeof(out_addr->sun_path) - 1) {
        return -ERANGE;
    }
    return 0;
}
