/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * Implementation of system calls "stat", "lstat", "fstat" and "readlink".
 */

#include <errno.h>
#include <linux/fcntl.h>

#include "shim_fs.h"
#include "shim_handle.h"
#include "shim_internal.h"
#include "shim_process.h"
#include "shim_table.h"
#include "stat.h"

static int do_stat(struct shim_dentry* dent, struct stat* stat) {
    assert(locked(&g_dcache_lock));
    assert(dent->inode);

    struct shim_fs* fs = dent->inode->fs;

    if (!fs || !fs->d_ops || !fs->d_ops->stat)
        return -EACCES;

    int ret = fs->d_ops->stat(dent, stat);
    if (ret < 0)
        return ret;

    /* Update `st_ino` from dentry */
    stat->st_ino = dentry_ino(dent);
    return 0;
}

static int do_hstat(struct shim_handle* hdl, struct stat* stat) {
    struct shim_fs* fs = hdl->fs;

    if (!fs || !fs->fs_ops || !fs->fs_ops->hstat)
        return -EACCES;

    int ret = fs->fs_ops->hstat(hdl, stat);
    if (ret < 0)
        return ret;

    /* Update `st_ino` from dentry */
    if (hdl->dentry)
        stat->st_ino = dentry_ino(hdl->dentry);

    return 0;
}

long shim_do_stat(const char* file, struct stat* stat) {
    if (!is_user_string_readable(file))
        return -EFAULT;

    if (!is_user_memory_writable(stat, sizeof(*stat)))
        return -EFAULT;

    int ret;
    struct shim_dentry* dent = NULL;

    lock(&g_dcache_lock);
    ret = path_lookupat(/*start=*/NULL, file, LOOKUP_FOLLOW, &dent);
    if (ret < 0)
        goto out;

    ret = do_stat(dent, stat);
out:
    unlock(&g_dcache_lock);
    if (dent)
        put_dentry(dent);
    return ret;
}

long shim_do_lstat(const char* file, struct stat* stat) {
    if (!is_user_string_readable(file))
        return -EFAULT;

    if (!is_user_memory_writable(stat, sizeof(*stat)))
        return -EFAULT;

    int ret;
    struct shim_dentry* dent = NULL;

    lock(&g_dcache_lock);
    ret = path_lookupat(/*start=*/NULL, file, LOOKUP_NO_FOLLOW, &dent);
    if (ret < 0)
        goto out;

    ret = do_stat(dent, stat);
out:
    unlock(&g_dcache_lock);
    if (dent)
        put_dentry(dent);
    return ret;
}

long shim_do_fstat(int fd, struct stat* stat) {
    struct shim_handle* hdl = get_fd_handle(fd, NULL, NULL);
    if (!hdl)
        return -EBADF;

    if (!is_user_memory_writable(stat, sizeof(*stat)))
        return -EFAULT;

    int ret = do_hstat(hdl, stat);
    put_handle(hdl);
    return ret;
}

long shim_do_readlinkat(int dirfd, const char* file, char* buf, int bufsize) {
    int ret;
    char* target = NULL;

    if (!is_user_string_readable(file))
        return -EFAULT;

    if (bufsize <= 0)
        return -EINVAL;

    if (!is_user_memory_writable(buf, bufsize))
        return -EFAULT;

    struct shim_dentry* dent = NULL;
    struct shim_dentry* dir = NULL;

    if (*file != '/' && (ret = get_dirfd_dentry(dirfd, &dir)) < 0)
        goto out;

    lock(&g_dcache_lock);
    ret = path_lookupat(dir, file, LOOKUP_NO_FOLLOW, &dent);
    if (ret < 0)
        goto out;

    ret = -EINVAL;
    /* The correct behavior is to return -EINVAL if file is not a
       symbolic link */
    if (dent->inode->type != S_IFLNK)
        goto out;

    struct shim_fs* fs = dent->inode->fs;
    if (!fs->d_ops || !fs->d_ops->follow_link)
        goto out;

    ret = fs->d_ops->follow_link(dent, &target);
    if (ret < 0)
        goto out;

    size_t target_len = strlen(target);

    ret = bufsize;
    if (target_len < (size_t)bufsize)
        ret = target_len;

    memcpy(buf, target, ret);
out:
    unlock(&g_dcache_lock);
    if (dent) {
        put_dentry(dent);
    }
    if (dir) {
        put_dentry(dir);
    }
    free(target);
    return ret;
}

long shim_do_readlink(const char* file, char* buf, int bufsize) {
    return shim_do_readlinkat(AT_FDCWD, file, buf, bufsize);
}

static int __do_statfs(struct shim_mount* mount, struct statfs* buf) {
    __UNUSED(mount);
    if (!is_user_memory_writable(buf, sizeof(*buf)))
        return -EFAULT;

    memset(buf, 0, sizeof(*buf));

    buf->f_bsize  = 4096;
    buf->f_blocks = 20000000;
    buf->f_bfree  = 10000000;
    buf->f_bavail = 10000000;

    log_debug("statfs: %ld %ld %ld", buf->f_blocks, buf->f_bfree, buf->f_bavail);

    return 0;
}

long shim_do_statfs(const char* path, struct statfs* buf) {
    if (!is_user_string_readable(path))
        return -EFAULT;

    int ret;
    struct shim_dentry* dent = NULL;

    lock(&g_dcache_lock);
    ret = path_lookupat(/*start=*/NULL, path, LOOKUP_FOLLOW, &dent);
    unlock(&g_dcache_lock);
    if (ret < 0)
        return ret;

    struct shim_mount* mount = dent->mount;
    put_dentry(dent);
    return __do_statfs(mount, buf);
}

long shim_do_fstatfs(int fd, struct statfs* buf) {
    struct shim_handle* hdl = get_fd_handle(fd, NULL, NULL);
    if (!hdl)
        return -EBADF;

    struct shim_mount* mount = hdl->dentry ? hdl->dentry->mount : NULL;
    put_handle(hdl);
    return __do_statfs(mount, buf);
}

/*
 * Handle the special case of `fstatat` with empty path (permitted with AT_EMPTY_PATH). Note that in
 * this case `dirfd` can point to a non-directory file, so we cannot use `get_dirfd_dentry`.
 */
static int do_fstatat_empty_path(int dirfd, struct stat* statbuf) {
    if (dirfd != AT_FDCWD)
        return shim_do_fstat(dirfd, statbuf);

    lock(&g_process.fs_lock);
    struct shim_dentry* dent = g_process.cwd;
    get_dentry(dent);
    unlock(&g_process.fs_lock);

    lock(&g_dcache_lock);

    int ret;

    if (!dent->inode) {
        ret = -ENOENT;
        goto out;
    }

    ret = do_stat(dent, statbuf);
out:
    unlock(&g_dcache_lock);
    put_dentry(dent);
    return ret;
}

long shim_do_newfstatat(int dirfd, const char* pathname, struct stat* statbuf, int flags) {
    if (flags & ~(AT_EMPTY_PATH | AT_NO_AUTOMOUNT | AT_SYMLINK_NOFOLLOW))
        return -EINVAL;
    if (!is_user_string_readable(pathname))
        return -EFAULT;
    if (!is_user_memory_writable(statbuf, sizeof(*statbuf)))
        return -EFAULT;

    int lookup_flags = LOOKUP_FOLLOW;
    if (flags & AT_SYMLINK_NOFOLLOW)
        lookup_flags &= ~LOOKUP_FOLLOW;
    if (flags & AT_NO_AUTOMOUNT) {
        /* Do nothing as automount isn't supported */
        log_warning("newfstatat: ignoring AT_NO_AUTOMOUNT.");
    }

    if (!*pathname) {
        if (!(flags & AT_EMPTY_PATH))
            return -ENOENT;

        return do_fstatat_empty_path(dirfd, statbuf);
    }

    int ret;

    struct shim_dentry* dir = NULL;
    if (*pathname != '/') {
        ret = get_dirfd_dentry(dirfd, &dir);
        if (ret < 0)
            return ret;
    }

    lock(&g_dcache_lock);

    struct shim_dentry* dent = NULL;
    ret = path_lookupat(dir, pathname, lookup_flags, &dent);
    if (ret < 0)
        goto out;

    ret = do_stat(dent, statbuf);
out:
    unlock(&g_dcache_lock);
    if (dent)
        put_dentry(dent);
    if (dir)
        put_dentry(dir);
    return ret;
}
