/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

/*
 * Inter process signal handling using IPC.
 */

#include "shim_ipc.h"
#include "shim_process.h"
#include "shim_signal.h"
#include "shim_thread.h"

static int ipc_pid_kill_send(enum kill_type type, IDTYPE sender, IDTYPE dest_pid, IDTYPE target,
                             int sig) {
    int ret;

    IDTYPE dest = 0;
    if (type == KILL_ALL) {
        if (g_process_ipc_ids.leader_vmid) {
            dest = g_process_ipc_ids.leader_vmid;
        }
    } else {
        ret = ipc_get_id_owner(dest_pid, &dest);
        if (ret < 0) {
            return ret;
        }
        if (dest == 0) {
            /* No process owns `dest_pid`... */
            if (is_zombie_process(dest_pid)) {
                /* ... but it's a zombie! */
                return 0;
            } else {
                /* ... so it does not exist. */
                return -ESRCH;
            }
        }
    }

    struct shim_ipc_pid_kill msgin = {
        .sender = sender,
        .type = type,
        .pid = dest_pid,
        .id = target,
        .signum = sig,
    };

    size_t total_msg_size    = get_ipc_msg_size(sizeof(msgin));
    struct shim_ipc_msg* msg = __alloca(total_msg_size);
    init_ipc_msg(msg, IPC_MSG_PID_KILL, total_msg_size);
    memcpy(&msg->data, &msgin, sizeof(msgin));

    if (type == KILL_ALL && !g_process_ipc_ids.leader_vmid) {
        log_debug("IPC broadcast: IPC_MSG_PID_KILL(%u, %d, %u, %d)", sender, type, dest_pid, sig);
        ret = ipc_broadcast(msg, /*exclude_id=*/0);
    } else {
        log_debug("IPC send to %u: IPC_MSG_PID_KILL(%u, %d, %u, %d)", dest, sender, type,
                  dest_pid, sig);

        void* resp = NULL;
        ret = ipc_send_msg_and_get_response(dest, msg, &resp);
        if (ret < 0) {
            /* During sending the message to destination process, it may have terminated and became
             * a zombie; kill shouldn't fail in this case. The below logic checks if the destination
             * process is _our_ zombie child -- this doesn't work for a case when the destination
             * process is not our child (the logic will just loop couple times and return error).
             * We assume that the latter case doesn't happen in real applications. */
            int wait_iter = 3;
            while (wait_iter--) {
                if (is_zombie_process(dest_pid)) {
                    log_debug("IPC send to terminated child process %u is dropped", dest_pid);
                    ret = 0;
                    break;
                } else {
                    /* There may be a race between receiving a SIGCHLD notification in the IPC
                     * worker thread (thus marking the destination process as a zombie) and sending
                     * a KILL in this thread, so we sleep for a bit and check for zombie again. */
                    uint64_t timeout_us = 10000;
                    thread_wait(&timeout_us, /*ignore_pending_signals=*/true);
                }
            }
        } else {
            ret = *(int*)resp;
            free(resp);
        }
    }

    return ret;
}

int ipc_kill_process(IDTYPE sender, IDTYPE target, int sig) {
    return ipc_pid_kill_send(KILL_PROCESS, sender, target, target, sig);
}

int ipc_kill_thread(IDTYPE sender, IDTYPE dest_pid, IDTYPE target, int sig) {
    return ipc_pid_kill_send(KILL_THREAD, sender, dest_pid, target, sig);
}

int ipc_kill_pgroup(IDTYPE sender, IDTYPE pgid, int sig) {
    return ipc_pid_kill_send(KILL_PGROUP, sender, pgid, pgid, sig);
}

int ipc_kill_all(IDTYPE sender, int sig) {
    return ipc_pid_kill_send(KILL_ALL, sender, /*dest_pid=*/0, /*target=*/0, sig);
}

int ipc_pid_kill_callback(IDTYPE src, void* data, uint64_t seq) {
    struct shim_ipc_pid_kill* msgin = (struct shim_ipc_pid_kill*)data;

    log_debug("IPC callback from %u: IPC_MSG_PID_KILL(%u, %d, %u, %d)", src, msgin->sender,
              msgin->type, msgin->id, msgin->signum);

    int ret = 0;
    bool response_expected = true;

    switch (msgin->type) {
        case KILL_THREAD:
            if (msgin->pid != g_process.pid) {
                ret = -ESRCH;
            } else {
                ret = do_kill_thread(msgin->sender, g_process.pid, msgin->id, msgin->signum);
            }
            break;
        case KILL_PROCESS:
            assert(g_process.pid == msgin->pid);
            assert(g_process.pid == msgin->id);
            ret = do_kill_proc(msgin->sender, msgin->id, msgin->signum);
            break;
        case KILL_PGROUP:
            ret = do_kill_pgroup(msgin->sender, msgin->id, msgin->signum);
            break;
        case KILL_ALL:
            if (!g_process_ipc_ids.leader_vmid) {
                size_t total_msg_size = get_ipc_msg_size(sizeof(*msgin));
                struct shim_ipc_msg* msg = __alloca(total_msg_size);
                init_ipc_msg(msg, IPC_MSG_PID_KILL, total_msg_size);
                memcpy(&msg->data, msgin, sizeof(*msgin));
                ret = ipc_broadcast(msg, /*exclude_id=*/src);
                if (ret < 0) {
                    break;
                }
            } else {
                response_expected = false;
            }

            ret = do_kill_proc(msgin->sender, g_process.pid, msgin->signum);
            break;
        default:
            BUG();
    }

    if (response_expected) {
        static_assert(SAME_TYPE(ret, int), "receiver assumes int");
        size_t total_msg_size = get_ipc_msg_size(sizeof(ret));
        struct shim_ipc_msg* msg = __alloca(total_msg_size);
        init_ipc_response(msg, seq, total_msg_size);
        memcpy(&msg->data, &ret, sizeof(ret));

        ret = ipc_send_message(src, msg);
    } else {
        ret = 0;
    }
    return ret;
}
