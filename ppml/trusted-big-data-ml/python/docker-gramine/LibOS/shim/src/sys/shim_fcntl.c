/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University
 * Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * Implementation of system call "fcntl":
 *
 * - F_DUPFD, F_DUPFD_CLOEXEC (duplicate a file descriptor)
 * - F_GETFD, F_SETFD (file descriptor flags)
 * - F_GETFL, F_SETFL (file status flags)
 * - F_SETLK, F_SETLKW, F_GETLK (POSIX advisory locks)
 * - F_SETOWN (file descriptor owner): dummy implementation
 */

#include <errno.h>
#include <linux/fcntl.h>

#include "shim_fs.h"
#include "shim_fs_lock.h"
#include "shim_handle.h"
#include "shim_internal.h"
#include "shim_lock.h"
#include "shim_process.h"
#include "shim_table.h"
#include "shim_thread.h"

#define FCNTL_SETFL_MASK (O_APPEND | O_NONBLOCK)

static int _set_handle_flags(struct shim_handle* hdl, unsigned long arg) {
    if (hdl->fs && hdl->fs->fs_ops && hdl->fs->fs_ops->setflags) {
        int ret = hdl->fs->fs_ops->setflags(hdl, arg & FCNTL_SETFL_MASK);
        if (ret < 0) {
            return ret;
        }
    }
    hdl->flags = (hdl->flags & ~FCNTL_SETFL_MASK) | (arg & FCNTL_SETFL_MASK);
    return 0;
}

int set_handle_nonblocking(struct shim_handle* hdl, bool on) {
    lock(&hdl->lock);
    int ret = _set_handle_flags(hdl, on ? hdl->flags | O_NONBLOCK : hdl->flags & ~O_NONBLOCK);
    unlock(&hdl->lock);
    return ret;
}

/*
 * Convert user-mode `struct flock` into our `struct posix_lock`. This mostly means converting the
 * position parameters (l_whence, l_start, l_len) to an absolute inclusive range [start .. end]. See
 * `man fcntl` for details.
 *
 * We need to return -EINVAL for underflow (positions before start of file), and -EOVERFLOW for
 * positive overflow.
 */
static int flock_to_posix_lock(struct flock* fl, struct shim_handle* hdl, struct posix_lock* pl) {
    if (!(fl->l_type == F_RDLCK || fl->l_type == F_WRLCK || fl->l_type == F_UNLCK))
        return -EINVAL;

    int ret;

    struct shim_fs* fs = hdl->fs;
    assert(fs && fs->fs_ops);

    /* Compute the origin based on `l_start` and `l_whence`. Note that we cannot directly call
     * `seek(hdl, l_start, l_whence)`, because that would modify the handle position. Only
     * retrieving the current position (by calling `seek(hdl, 0, SEEK_CUR)`) is safe. */
    uint64_t origin;
    switch (fl->l_whence) {
        case SEEK_SET:
            origin = 0;
            break;
        case SEEK_CUR: {
            if (!fs->fs_ops->seek)
                return -EINVAL;

            file_off_t pos = fs->fs_ops->seek(hdl, 0, SEEK_CUR);
            if (pos < 0)
                return pos;
            origin = pos;
            break;
        }
        case SEEK_END: {
            if (!fs->fs_ops->hstat)
                return -EINVAL;

            struct stat stat;
            ret = fs->fs_ops->hstat(hdl, &stat);
            if (ret < 0)
                return ret;
            assert(stat.st_size >= 0);
            origin = stat.st_size;
            break;
        }
        default:
            return -EINVAL;
    }

    if (__builtin_add_overflow(origin, fl->l_start, &origin)) {
        return fl->l_start > 0 ? -EOVERFLOW : -EINVAL;
    }

    uint64_t start, end;
    if (fl->l_len > 0) {
        /* len > 0: the range is [origin .. origin + len - 1] */
        start = origin;
        if (__builtin_add_overflow(origin, fl->l_len - 1, &end))
            return -EOVERFLOW;
    } else if (fl->l_len < 0) {
        /* len < 0: the range is [origin + len .. origin - 1] */
        if (__builtin_add_overflow(origin, fl->l_len, &start))
            return -EINVAL;
        if (__builtin_add_overflow(origin, -1, &end))
            return -EINVAL;
    } else {
        /* len == 0: the range is [origin .. EOF] */
        start = origin;
        end = FS_LOCK_EOF;
    }

    pl->type = fl->l_type;
    pl->start = start;
    pl->end = end;
    pl->pid = g_process.pid;
    return 0;
}

long shim_do_fcntl(int fd, int cmd, unsigned long arg) {
    int ret;
    int flags;

    struct shim_handle_map* handle_map = get_thread_handle_map(NULL);
    assert(handle_map);

    struct shim_handle* hdl = get_fd_handle(fd, &flags, handle_map);
    if (!hdl)
        return -EBADF;

    switch (cmd) {
        /* See `man fcntl` for the expected semantics of these commands. */

        /* F_DUPFD (int) */
        case F_DUPFD: {
            ret = set_new_fd_handle_above_fd(arg, hdl, flags, handle_map);
            break;
        }

        /* F_DUPFD_CLOEXEC (int) */
        case F_DUPFD_CLOEXEC: {
            flags |= FD_CLOEXEC;
            ret = set_new_fd_handle_above_fd(arg, hdl, flags, handle_map);
            break;
        }

        /* F_GETFD (int) */
        case F_GETFD:
            ret = flags & FD_CLOEXEC;
            break;

        /* F_SETFD (int) */
        case F_SETFD:
            lock(&handle_map->lock);
            if (HANDLE_ALLOCATED(handle_map->map[fd]))
                handle_map->map[fd]->flags = arg & FD_CLOEXEC;
            unlock(&handle_map->lock);
            ret = 0;
            break;

        /* F_GETFL (void) */
        case F_GETFL:
            lock(&hdl->lock);
            flags = hdl->flags;
            unlock(&hdl->lock);
            ret = flags;
            break;

        /* F_SETFL (int) */
        case F_SETFL:
            lock(&hdl->lock);
            ret = _set_handle_flags(hdl, arg);
            unlock(&hdl->lock);
            break;

        /* F_SETLK, F_SETLKW (struct flock*): see `shim_fs_lock.h` for caveats */
        case F_SETLK:
        case F_SETLKW: {
            struct flock *fl = (struct flock*)arg;
            if (!is_user_memory_readable(fl, sizeof(*fl))) {
                ret = -EFAULT;
                break;
            }

            if (!hdl->dentry) {
                /* TODO: Linux allows locks on pipes etc. Our locks work only for "normal" files
                 * that have a dentry. */
                ret = -EINVAL;
                break;
            }

            if (fl->l_type == F_RDLCK && !(hdl->acc_mode & MAY_READ)) {
                ret = -EINVAL;
                break;
            }

            if (fl->l_type == F_WRLCK && !(hdl->acc_mode & MAY_WRITE)) {
                ret = -EINVAL;
                break;
            }

            struct posix_lock pl;
            ret = flock_to_posix_lock(fl, hdl, &pl);
            if (ret < 0)
                break;

            ret = posix_lock_set(hdl->dentry, &pl, /*wait=*/cmd == F_SETLKW);
            break;
        }

        /* F_GETLK (struct flock*): see `shim_fs_lock.h` for caveats */
        case F_GETLK: {
            struct flock *fl = (struct flock*)arg;
            if (!is_user_memory_readable(fl, sizeof(*fl))
                    || !is_user_memory_writable(fl, sizeof(*fl))) {
                ret = -EFAULT;
                break;
            }

            if (!hdl->dentry) {
                ret = -EINVAL;
                break;
            }

            struct posix_lock pl;
            ret = flock_to_posix_lock(fl, hdl, &pl);
            if (ret < 0)
                break;

            if (pl.type == F_UNLCK) {
                ret = -EINVAL;
                break;
            }

            struct posix_lock pl2;
            ret = posix_lock_get(hdl->dentry, &pl, &pl2);
            if (ret < 0)
                break;

            fl->l_type = pl2.type;
            if (pl2.type != F_UNLCK) {
                fl->l_whence = SEEK_SET;
                fl->l_start = pl2.start;
                if (pl2.end == FS_LOCK_EOF) {
                    /* range until EOF is encoded as len == 0 */
                    fl->l_len = 0;
                } else {
                    fl->l_len = pl2.end - pl2.start + 1;
                }
                fl->l_pid = pl2.pid;
            }
            ret = 0;
            break;
        }

        /* F_SETOWN (int): dummy implementation */
        case F_SETOWN:
            ret = 0;
            /* XXX: DUMMY for now */
            break;

        default:
            ret = -EINVAL;
            break;
    }

    put_handle(hdl);
    return ret;
}
