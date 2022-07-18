/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * IPC glue code for filesystem locks.
 */

#include "shim_fs_lock.h"
#include "shim_ipc.h"

int ipc_posix_lock_set(const char* path, struct posix_lock* pl, bool wait) {
    assert(g_process_ipc_ids.leader_vmid);

    struct shim_ipc_posix_lock msgin = {
        .type = pl->type,
        .start = pl->start,
        .end = pl->end,
        .pid = pl->pid,

        .wait = wait,
    };

    size_t path_len = strlen(path);
    size_t total_msg_size = get_ipc_msg_size(sizeof(msgin) + path_len + 1);
    struct shim_ipc_msg* msg = __alloca(total_msg_size);
    init_ipc_msg(msg, IPC_MSG_POSIX_LOCK_SET, total_msg_size);
    memcpy(msg->data, &msgin, sizeof(msgin));

    /* Copy path after message (`msg->data` is unaligned, so we have to compute the offset
     * manually) */
    char* path_ptr = (char*)&msg->data + offsetof(struct shim_ipc_posix_lock, path);
    memcpy(path_ptr, path, path_len + 1);

    void* data;
    int ret = ipc_send_msg_and_get_response(g_process_ipc_ids.leader_vmid, msg, &data);
    if (ret < 0)
        return ret;
    int result = *(int*)data;
    free(data);
    return result;
}

int ipc_posix_lock_set_send_response(IDTYPE vmid, unsigned long seq, int result) {
    assert(!g_process_ipc_ids.leader_vmid);

    size_t total_msg_size = get_ipc_msg_size(sizeof(result));
    struct shim_ipc_msg* msg = __alloca(total_msg_size);
    init_ipc_response(msg, seq, total_msg_size);
    memcpy(msg->data, &result, sizeof(result));
    return ipc_send_message(vmid, msg);
}

int ipc_posix_lock_get(const char* path, struct posix_lock* pl, struct posix_lock* out_pl) {
    assert(g_process_ipc_ids.leader_vmid);

    struct shim_ipc_posix_lock msgin = {
        .type = pl->type,
        .start = pl->start,
        .end = pl->end,
        .pid = pl->pid,
    };

    size_t path_len = strlen(path);
    size_t total_msg_size = get_ipc_msg_size(sizeof(msgin) + path_len + 1);
    struct shim_ipc_msg* msg = __alloca(total_msg_size);
    init_ipc_msg(msg, IPC_MSG_POSIX_LOCK_GET, total_msg_size);
    memcpy(msg->data, &msgin, sizeof(msgin));

    /* Copy path after message (`msg->data` is unaligned, so we have to compute the offset
     * manually) */
    char* path_ptr = (char*)&msg->data + offsetof(struct shim_ipc_posix_lock, path);
    memcpy(path_ptr, path, path_len + 1);

    void* data;
    int ret = ipc_send_msg_and_get_response(g_process_ipc_ids.leader_vmid, msg, &data);
    if (ret < 0)
        return ret;

    struct shim_ipc_posix_lock_resp* resp = data;
    int result = resp->result;
    if (resp->result == 0) {
        out_pl->type = resp->type;
        out_pl->start = resp->start;
        out_pl->end = resp->end;
        out_pl->pid = resp->pid;
    }
    free(data);
    return result;
}

int ipc_posix_lock_clear_pid(IDTYPE pid) {
    assert(g_process_ipc_ids.leader_vmid);

    size_t total_msg_size = get_ipc_msg_size(sizeof(pid));
    struct shim_ipc_msg* msg = __alloca(total_msg_size);
    init_ipc_msg(msg, IPC_MSG_POSIX_LOCK_CLEAR_PID, total_msg_size);
    memcpy(msg->data, &pid, sizeof(pid));

    void* data;
    int ret = ipc_send_msg_and_get_response(g_process_ipc_ids.leader_vmid, msg, &data);
    if (ret < 0)
        return ret;
    int result = *(int*)data;
    free(data);
    return result;
}

int ipc_posix_lock_set_callback(IDTYPE src, void* data, unsigned long seq) {
    struct shim_ipc_posix_lock* msgin = data;
    struct posix_lock pl = {
        .type = msgin->type,
        .start = msgin->start,
        .end = msgin->end,
        .pid = msgin->pid,
    };

    return posix_lock_set_from_ipc(msgin->path, &pl, msgin->wait, src, seq);
}

int ipc_posix_lock_get_callback(IDTYPE src, void* data, unsigned long seq) {
    struct shim_ipc_posix_lock* msgin = data;
    struct posix_lock pl = {
        .type = msgin->type,
        .start = msgin->start,
        .end = msgin->end,
        .pid = msgin->pid,
    };

    struct posix_lock pl2 = {0};
    int result = posix_lock_get_from_ipc(msgin->path, &pl, &pl2);
    struct shim_ipc_posix_lock_resp msgout = {
        .result = result,
        .type = pl2.type,
        .start = pl2.start,
        .end = pl2.end,
        .pid = pl2.pid,
    };

    size_t total_msg_size = get_ipc_msg_size(sizeof(msgout));
    struct shim_ipc_msg* msg = __alloca(total_msg_size);
    init_ipc_response(msg, seq, total_msg_size);
    memcpy(msg->data, &msgout, sizeof(msgout));
    return ipc_send_message(src, msg);
}

int ipc_posix_lock_clear_pid_callback(IDTYPE src, void* data, unsigned long seq) {
    IDTYPE* pid = data;
    int result = posix_lock_clear_pid(*pid);

    size_t total_msg_size = get_ipc_msg_size(sizeof(result));
    struct shim_ipc_msg* msg = __alloca(total_msg_size);
    init_ipc_response(msg, seq, total_msg_size);
    memcpy(msg->data, &result, sizeof(result));
    return ipc_send_message(src, msg);
}
