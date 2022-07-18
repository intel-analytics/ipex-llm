/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University
 * Copyright (C) 2021 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

/*
 * This file contains functions and callbacks to handle IPC of general process information.
 */

#include "shim_fs.h"
#include "shim_ipc.h"
#include "shim_lock.h"
#include "shim_process.h"
#include "shim_thread.h"

static const char* pid_meta_code_str[4] = {
    "CRED",
    "EXEC",
    "CWD",
    "ROOT",
};

int ipc_pid_getmeta(IDTYPE pid, enum pid_meta_code code, struct shim_ipc_pid_retmeta** data) {
    IDTYPE dest;
    int ret;

    if ((ret = ipc_get_id_owner(pid, &dest)) < 0)
        return ret;

    if (dest == 0) {
        /* No process owns `pid` thus it does not exist. */
        return -ESRCH;
    }

    struct shim_ipc_pid_getmeta msgin = {
        .pid = pid,
        .code = code,
    };
    size_t total_msg_size = get_ipc_msg_size(sizeof(msgin));
    struct shim_ipc_msg* msg = __alloca(total_msg_size);
    init_ipc_msg(msg, IPC_MSG_PID_GETMETA, total_msg_size);

    memcpy(&msg->data, &msgin, sizeof(msgin));

    log_debug("ipc send to %u: IPC_MSG_PID_GETMETA(%u, %s)", dest, pid, pid_meta_code_str[code]);

    struct shim_ipc_pid_retmeta* resp = NULL;
    ret = ipc_send_msg_and_get_response(dest, msg, (void**)&resp);
    if (ret < 0) {
        return ret;
    }
    if (resp->ret_val != 0) {
        ret = resp->ret_val;
        free(resp);
        return ret;
    }

    *data = resp;
    return 0;
}

int ipc_pid_getmeta_callback(IDTYPE src, void* msg_data, uint64_t seq) {
    struct shim_ipc_pid_getmeta* msgin = (struct shim_ipc_pid_getmeta*)msg_data;
    int ret = 0;

    log_debug("ipc callback from %u: IPC_MSG_PID_GETMETA(%u, %s)", src, msgin->pid,
              pid_meta_code_str[msgin->code]);

    struct shim_thread* thread = lookup_thread(msgin->pid);
    void* data = NULL;
    size_t datasize = 0;
    size_t bufsize = 0;
    int resp_ret_val = 0;

    if (!thread) {
        resp_ret_val = -ESRCH;
        goto out_send;
    }

    switch (msgin->code) {
        case PID_META_CRED:
            lock(&thread->lock);
            bufsize = sizeof(IDTYPE) * 2;
            data = malloc(bufsize);
            if (!data) {
                unlock(&thread->lock);
                ret = -ENOMEM;
                goto out;
            }
            datasize = bufsize;
            ((IDTYPE*)data)[0] = thread->uid;
            ((IDTYPE*)data)[1] = thread->gid;
            unlock(&thread->lock);
            break;
        case PID_META_EXEC:
        case PID_META_CWD:
        case PID_META_ROOT: {
            lock(&g_process.fs_lock);

            struct shim_dentry* dent = NULL;
            switch (msgin->code) {
               case PID_META_EXEC:
                   if (g_process.exec)
                       dent = g_process.exec->dentry;
                   break;
               case PID_META_CWD:
                   dent = g_process.cwd;
                   break;
               case PID_META_ROOT:
                   dent = g_process.root;
                   break;
               default:
                   BUG();
            }
            if (!dent) {
                unlock(&g_process.fs_lock);
                ret = -ENOENT;
                goto out;
            }
            if ((ret = dentry_abs_path(dent, (char**)&data, &bufsize)) < 0) {
                unlock(&g_process.fs_lock);
                goto out;
            }
            datasize = bufsize;
            unlock(&g_process.fs_lock);
            break;
        }
        default:
            BUG();
    }

out_send:;
    struct shim_ipc_pid_retmeta retmeta = {
        .datasize = datasize,
        .ret_val = resp_ret_val,
    };
    size_t total_msg_size = get_ipc_msg_size(sizeof(retmeta) + datasize);
    struct shim_ipc_msg* msg = __alloca(total_msg_size);
    init_ipc_response(msg, seq, total_msg_size);

    memcpy(&msg->data, &retmeta, sizeof(retmeta));
    memcpy(&((struct shim_ipc_pid_retmeta*)&msg->data)->data, data, datasize);

    log_debug("IPC send to %u: shim_ipc_pid_retmeta{%lu, ...}", src, datasize);

    ret = ipc_send_message(src, msg);

out:
    if (thread) {
        put_thread(thread);
    }
    free(data);
    return ret;
}
