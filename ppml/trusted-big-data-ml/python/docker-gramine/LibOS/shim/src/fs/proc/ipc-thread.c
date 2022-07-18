/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * Implementation of `/proc/<remote-pid>`. Currently supports only `root`, `cwd` and `exe` symlinks,
 * does not support process listing (you need to know the pid in advance) and does not do any
 * caching.
 */

#include "shim_fs.h"
#include "shim_fs_pseudo.h"
#include "shim_ipc.h"
#include "shim_process.h"

bool proc_ipc_thread_pid_name_exists(struct shim_dentry* parent, const char* name) {
    __UNUSED(parent);

    unsigned long pid;
    if (pseudo_parse_ulong(name, IDTYPE_MAX, &pid) < 0)
        return false;

    if (pid == g_process.pid) {
        /* This function should report only remote processes. */
        return false;
    }

    /* Send a dummy request to check whether `pid` exists. */
    struct shim_ipc_pid_retmeta* retmeta = NULL;
    int ret = ipc_pid_getmeta(pid, PID_META_CRED, &retmeta);
    if (ret < 0) {
        /* FIXME: this silences all errors. */
        return false;
    }
    free(retmeta);
    return true;
}

int proc_ipc_thread_follow_link(struct shim_dentry* dent, char** out_target) {
    assert(dent->parent);
    const char* parent_name = dent->parent->name;
    const char* name = dent->name;

    unsigned long pid;
    if (pseudo_parse_ulong(parent_name, IDTYPE_MAX, &pid) < 0)
        return -ENOENT;

    enum pid_meta_code ipc_code;
    if (strcmp(name, "root") == 0) {
        ipc_code = PID_META_ROOT;
    } else if (strcmp(name, "cwd") == 0) {
        ipc_code = PID_META_CWD;
    } else if (strcmp(name, "exe") == 0) {
        ipc_code = PID_META_EXEC;
    } else {
        return -ENOENT;
    }

    struct shim_ipc_pid_retmeta* ipc_data;
    int ret = ipc_pid_getmeta(pid, ipc_code, &ipc_data);
    if (ret < 0)
        return ret;

    *out_target = strdup(ipc_data->data);
    if (!*out_target) {
        ret = -ENOMEM;
        goto out;
    }
    ret = 0;
out:
    free(ipc_data);
    return ret;
}
