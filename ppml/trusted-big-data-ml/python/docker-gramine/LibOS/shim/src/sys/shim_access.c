/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * Implementation of system calls "access" and "faccessat".
 */

#include <errno.h>
#include <linux/fcntl.h>

#include "shim_fs.h"
#include "shim_internal.h"
#include "shim_lock.h"
#include "shim_table.h"

long shim_do_access(const char* file, mode_t mode) {
    return shim_do_faccessat(AT_FDCWD, file, mode);
}

long shim_do_faccessat(int dfd, const char* filename, mode_t mode) {
    if (!filename)
        return -EINVAL;

    if (!is_user_string_readable(filename))
        return -EFAULT;

    struct shim_dentry* dir = NULL;
    struct shim_dentry* dent = NULL;
    int ret = 0;

    if (*filename != '/' && (ret = get_dirfd_dentry(dfd, &dir)) < 0)
        return ret;

    lock(&g_dcache_lock);

    ret = path_lookupat(dir, filename, LOOKUP_FOLLOW, &dent);
    if (ret < 0)
        goto out;

    ret = check_permissions(dent, mode);

out:
    unlock(&g_dcache_lock);

    if (dir)
        put_dentry(dir);
    if (dent) {
        put_dentry(dent);
    }
    return ret;
}
