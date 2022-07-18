/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 * Copyright (C) 2021 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

#include "api.h"
#include "shim_ipc.h"
#include "shim_process.h"

void ipc_child_disconnect_callback(IDTYPE vmid) {
    /*
     * NOTE: IPC port may be closed by the host OS because the child process exited on the host OS
     * (and so the host OS closed all its sockets). This may happen before arrival of the expected
     * IPC_MSG_CHILDEXIT message from child process. In such case report that the child process was
     * killed by SIGPWR (we've picked this signal hoping that nothing really uses it, as this case
     * is not distinguishable from a genuine signal).
     */
    if (mark_child_exited_by_vmid(vmid, /*uid=*/0, /*exit_code=*/0, SIGPWR)) {
        log_error("Child process (vmid: 0x%x) got disconnected", vmid);
    } else {
        log_debug("Unknown process (vmid: 0x%x) disconnected", vmid);
    }
}

int ipc_cld_exit_send(unsigned int exitcode, unsigned int term_signal) {
    if (!g_process.ppid) {
        /* We have no parent inside Gramine, so no one to notify. */
        return 0;
    }

    struct shim_thread* self = get_cur_thread();
    lock(&self->lock);
    IDTYPE uid = self->uid;
    unlock(&self->lock);

    struct shim_ipc_cld_exit msgin = {
        .ppid        = g_process.ppid,
        .pid         = g_process.pid,
        .exitcode    = exitcode,
        .term_signal = term_signal,
        .uid         = uid,
    };

    size_t total_msg_size = get_ipc_msg_size(sizeof(msgin));
    struct shim_ipc_msg* msg = __alloca(total_msg_size);
    init_ipc_msg(msg, IPC_MSG_CHILDEXIT, total_msg_size);

    memcpy(msg->data, &msgin, sizeof(msgin));

    return ipc_send_message(g_process_ipc_ids.parent_vmid, msg);
}

int ipc_cld_exit_callback(IDTYPE src, void* data, uint64_t seq) {
    __UNUSED(seq);
    struct shim_ipc_cld_exit* msgin = (struct shim_ipc_cld_exit*)data;

    log_debug("IPC callback from %u: IPC_MSG_CHILDEXIT(%u, %u, %d, %u)", src, msgin->ppid,
              msgin->pid, msgin->exitcode, msgin->term_signal);

    if (mark_child_exited_by_pid(msgin->pid, msgin->uid, msgin->exitcode, msgin->term_signal)) {
        log_debug("Child process (pid: %u) died", msgin->pid);
    } else {
        log_error("Unknown process sent a child-death notification: pid: %d, vmid: %u",
                  msgin->pid, src);
        return -EINVAL;
    }

    return 0;
}
