/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * IPC glue code for the sync engine. These functions handle IPC_MSG_SYNC_* messages, but delegate
 * the actual logic to sync_server_* and sync_client_* functions.
 */

#include "shim_ipc.h"
#include "shim_sync.h"

static const char* sync_state_names[SYNC_STATE_NUM] = {
    [SYNC_STATE_INVALID]   = "invalid",
    [SYNC_STATE_SHARED]    = "shared",
    [SYNC_STATE_EXCLUSIVE] = "exclusive",
};

static const char* sync_message_names[IPC_MSG_CODE_BOUND] = {
    [IPC_MSG_SYNC_REQUEST_UPGRADE]   = "REQUEST_UPGRADE",
    [IPC_MSG_SYNC_REQUEST_DOWNGRADE] = "REQUEST_DOWNGRADE",
    [IPC_MSG_SYNC_REQUEST_CLOSE]     = "REQUEST_CLOSE",
    [IPC_MSG_SYNC_CONFIRM_UPGRADE]   = "CONFIRM_UPGRADE",
    [IPC_MSG_SYNC_CONFIRM_DOWNGRADE] = "CONFIRM_DOWNGRADE",
    [IPC_MSG_SYNC_CONFIRM_CLOSE]     = "CONFIRM_CLOSE",
};

static inline void sync_log(const char* prefix, int code, uint64_t id, int state) {
    log_trace("%s: %s(0x%lx, %s)", prefix, sync_message_names[code], id, sync_state_names[state]);
}

static int sync_msg_send(IDTYPE dest, int code, uint64_t id, int state, size_t data_size,
                         void* data) {
    struct shim_ipc_sync msgin = {
        .id = id,
        .state = state,
        .data_size = data_size,
    };

    size_t total_msg_size = get_ipc_msg_size(sizeof(struct shim_ipc_sync) + data_size);
    struct shim_ipc_msg* msg = __alloca(total_msg_size);
    init_ipc_msg(msg, code, total_msg_size);

    memcpy(&msg->data, &msgin, sizeof(msgin));
    memcpy(&((struct shim_ipc_sync*)&msg->data)->data, data, data_size);

    return ipc_send_message(dest, msg);
}

int ipc_sync_client_send(int code, uint64_t id, int state, size_t data_size, void* data) {
    sync_log("sync client", code, id, state);
    IDTYPE dest = g_process_ipc_ids.leader_vmid ?: g_process_ipc_ids.self_vmid;
    return sync_msg_send(dest, code, id, state, data_size, data);
}

int ipc_sync_server_send(IDTYPE dest, int code, uint64_t id, int state,
                         size_t data_size, void* data) {
    sync_log("sync server", code, id, state);
    return sync_msg_send(dest, code, id, state, data_size, data);
}

static int ipc_sync_client_callback(int code, IDTYPE src, void* data, unsigned long seq) {
    struct shim_ipc_sync* msgin = data;
    __UNUSED(src);
    __UNUSED(seq);

    sync_log("sync client callback", code, msgin->id, msgin->state);
    sync_client_message_callback(code, msgin->id, msgin->state, msgin->data_size, &msgin->data);
    return 0;
}

static int ipc_sync_server_callback(int code, IDTYPE src, void* data, unsigned long seq) {
    struct shim_ipc_sync* msgin = data;
    __UNUSED(seq);

    sync_log("sync server callback", code, msgin->id, msgin->state);
    sync_server_message_callback(src, code, msgin->id, msgin->state, msgin->data_size,
                                 &msgin->data);
    return 0;
}

int ipc_sync_request_upgrade_callback(IDTYPE src, void* data, unsigned long seq) {
    return ipc_sync_server_callback(IPC_MSG_SYNC_REQUEST_UPGRADE, src, data, seq);
}

int ipc_sync_request_downgrade_callback(IDTYPE src, void* data, unsigned long seq) {
    return ipc_sync_client_callback(IPC_MSG_SYNC_REQUEST_DOWNGRADE, src, data, seq);
}

int ipc_sync_request_close_callback(IDTYPE src, void* data, unsigned long seq) {
    return ipc_sync_server_callback(IPC_MSG_SYNC_REQUEST_CLOSE, src, data, seq);
}

int ipc_sync_confirm_upgrade_callback(IDTYPE src, void* data, unsigned long seq) {
    return ipc_sync_client_callback(IPC_MSG_SYNC_CONFIRM_UPGRADE, src, data, seq);
}

int ipc_sync_confirm_downgrade_callback(IDTYPE src, void* data, unsigned long seq) {
    return ipc_sync_server_callback(IPC_MSG_SYNC_CONFIRM_DOWNGRADE, src, data, seq);
}

int ipc_sync_confirm_close_callback(IDTYPE src, void* data, unsigned long seq) {
    return ipc_sync_client_callback(IPC_MSG_SYNC_CONFIRM_CLOSE, src, data, seq);
}
