/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2022 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * This file contains implementation of the "synthetic" filesystem. This filesystem handles
 * in-memory files (currently, only directories) created in the process of mounting a filesystem.
 *
 * For example, if the manifest specifies a mount at "/usr/bin", and "usr" does not exist, Gramine
 * will create synthetic directories for "/usr" and "/usr/bin". While "/usr/bin" will be immediately
 * shadowed by the mounted directory, "/usr" will remain visible for the user.
 *
 * Operations on synthetic files are handled by `synthetic_builtin_fs` defined below. It should be
 * possible to retrieve information about them (`stat`, `getdents` etc.), but not modify them in any
 * way.
 */

#include "shim_fs.h"

int synthetic_setup_dentry(struct shim_dentry* dent, mode_t type, mode_t perm) {
    assert(locked(&g_dcache_lock));
    assert(!dent->inode);

    struct shim_inode* inode = get_new_inode(dent->mount, type, perm);
    if (!inode)
        return -ENOMEM;
    dent->inode = inode;

    inode->fs = &synthetic_builtin_fs;

    return 0;
}

static int synthetic_open(struct shim_handle* hdl, struct shim_dentry* dent, int flags) {
    assert(locked(&g_dcache_lock));
    assert(dent->inode);
    __UNUSED(dent);
    __UNUSED(flags);

    hdl->type = TYPE_SYNTHETIC;
    return 0;
}

static struct shim_fs_ops synthetic_fs_ops = {
    .hstat = &generic_inode_hstat,
};

static struct shim_d_ops synthetic_d_ops = {
    .open = &synthetic_open,
    .readdir = &generic_readdir,
    .stat = &generic_inode_stat,
};

struct shim_fs synthetic_builtin_fs = {
    .name = "synth",
    .fs_ops = &synthetic_fs_ops,
    .d_ops = &synthetic_d_ops,
};
