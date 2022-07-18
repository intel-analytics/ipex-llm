/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * Implementation of system calls "getcwd", "chdir" and "fchdir".
 */

#include <errno.h>

#include "shim_fs.h"
#include "shim_handle.h"
#include "shim_internal.h"
#include "shim_lock.h"
#include "shim_process.h"
#include "shim_table.h"
#include "shim_thread.h"
#include "stat.h"

#ifndef ERANGE
#define ERANGE 34
#endif

long shim_do_getcwd(char* buf, size_t buf_size) {
    if (!buf || !buf_size)
        return -EINVAL;

    if (!is_user_memory_writable(buf, buf_size))
        return -EFAULT;

    lock(&g_process.fs_lock);
    struct shim_dentry* cwd = g_process.cwd;
    get_dentry(cwd);
    unlock(&g_process.fs_lock);

    char* path = NULL;
    size_t size;
    int ret = dentry_abs_path(cwd, &path, &size);
    if (ret < 0)
        goto out;

    if (size > PATH_MAX) {
        ret = -ENAMETOOLONG;
    } else if (size > buf_size) {
        ret = -ERANGE;
    } else {
        ret = size;
        memcpy(buf, path, size);
    }

    free(path);

out:
    put_dentry(cwd);
    return ret;
}

long shim_do_chdir(const char* filename) {
    struct shim_dentry* dent = NULL;
    int ret;

    if (!is_user_string_readable(filename))
        return -EFAULT;

    if (strnlen(filename, PATH_MAX + 1) == PATH_MAX + 1)
        return -ENAMETOOLONG;

    lock(&g_dcache_lock);
    ret = path_lookupat(/*start=*/NULL, filename, LOOKUP_FOLLOW | LOOKUP_DIRECTORY, &dent);
    unlock(&g_dcache_lock);
    if (ret < 0)
        return ret;

    if (!dent)
        return -ENOENT;

    lock(&g_process.fs_lock);
    put_dentry(g_process.cwd);
    g_process.cwd = dent;
    unlock(&g_process.fs_lock);
    return 0;
}

long shim_do_fchdir(int fd) {
    struct shim_thread* thread = get_cur_thread();
    struct shim_handle* hdl    = get_fd_handle(fd, NULL, thread->handle_map);
    if (!hdl)
        return -EBADF;

    struct shim_dentry* dent = hdl->dentry;

    if (!dent->inode || dent->inode->type != S_IFDIR) {
        char* path = NULL;
        dentry_abs_path(dent, &path, /*size=*/NULL);
        log_debug("%s is not a directory", path);
        free(path);
        put_handle(hdl);
        return -ENOTDIR;
    }

    lock(&g_process.fs_lock);
    get_dentry(dent);
    put_dentry(g_process.cwd);
    g_process.cwd = dent;
    unlock(&g_process.fs_lock);
    put_handle(hdl);
    return 0;
}
