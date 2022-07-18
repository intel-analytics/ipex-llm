/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

#include "api.h"
#include "log.h"
#include "shim_ipc.h"
#include "shim_types.h"

static IDTYPE g_last_vmid = STARTING_VMID;

static IDTYPE get_next_vmid(void) {
    return __atomic_add_fetch(&g_last_vmid, 1, __ATOMIC_RELAXED);
}

int ipc_get_new_vmid(IDTYPE* vmid) {
    if (!g_process_ipc_ids.leader_vmid) {
        *vmid = get_next_vmid();
        return 0;
    }

    size_t msg_size = get_ipc_msg_size(0);
    struct shim_ipc_msg* msg = malloc(msg_size);
    if (!msg) {
        return -ENOMEM;
    }
    init_ipc_msg(msg, IPC_MSG_GET_NEW_VMID, msg_size);

    log_debug("%s: sending a request", __func__);

    void* resp = NULL;
    int ret = ipc_send_msg_and_get_response(g_process_ipc_ids.leader_vmid, msg, &resp);
    if (ret < 0) {
        goto out;
    }

    *vmid = *(IDTYPE*)resp;
    ret = 0;
    log_debug("%s: got a response: %u", __func__, *vmid);

out:
    free(resp);
    free(msg);
    return ret;
}

int ipc_get_new_vmid_callback(IDTYPE src, void* data, uint64_t seq) {
    __UNUSED(data);
    IDTYPE vmid = get_next_vmid();

    log_debug("%s: %u", __func__, vmid);

    size_t msg_size = get_ipc_msg_size(sizeof(vmid));
    struct shim_ipc_msg* msg = __alloca(msg_size);
    init_ipc_response(msg, seq, msg_size);
    memcpy(&msg->data, &vmid, sizeof(vmid));

    return ipc_send_message(src, msg);
}
