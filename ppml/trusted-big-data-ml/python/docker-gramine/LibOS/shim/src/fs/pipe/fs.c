/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains code for implementation of `pipe` and `fifo` filesystems.
 *
 * Named pipes (`fifo`) are implemented as special dentries/inodes in the filesystem, with `fs`
 * fields overriden to `fifo_builtin_fs`.
 *
 * - We create two temporary handles for read and write end of pipe, with corresponding PAL handles,
 *   and put them in process FD table (see `shim_pipe.c`)
 * - We create an inode for the named pipe, and store both FDs with it (`struct fifo_data`)
 * - When opening a file for reading or writing, we retrieve one of the temporary handles and
 *   transfer the PAL handle to a newly initialized handle.
 *
 * Note that each end of a named pipe can only be retrieved once.
 *
 * It would be better to store the temporary handles directly, without allocating FDs for them.
 * However, using FDs makes it easier to checkpoint a named pipe.
 */

#include <asm/fcntl.h>
#include <errno.h>

#include "pal.h"
#include "perm.h"
#include "shim_fs.h"
#include "shim_handle.h"
#include "shim_internal.h"
#include "shim_lock.h"
#include "shim_process.h"
#include "shim_signal.h"
#include "shim_thread.h"
#include "stat.h"

struct fifo_data {
    int fd_read;
    int fd_write;
};

static int fifo_icheckpoint(struct shim_inode* inode, void** out_data, size_t* out_size) {
    assert(locked(&inode->lock));

    struct fifo_data* fifo_data = inode->data;
    struct fifo_data* fifo_data_cp = malloc(sizeof(*fifo_data_cp));
    if (!fifo_data_cp)
        return -ENOMEM;

    fifo_data_cp->fd_read = fifo_data->fd_read;
    fifo_data_cp->fd_write = fifo_data->fd_write;

    *out_data = fifo_data_cp;
    *out_size = sizeof(*fifo_data_cp);
    return 0;
}

static int fifo_irestore(struct shim_inode* inode, void* data) {
    /* Use the data from checkpoint blob directly */
    inode->data = data;
    return 0;
}

int fifo_setup_dentry(struct shim_dentry* dent, mode_t perm, int fd_read, int fd_write) {
    assert(locked(&g_dcache_lock));
    assert(!dent->inode);

    struct shim_inode* inode = get_new_inode(dent->mount, S_IFIFO, perm);
    if (!inode)
        return -ENOMEM;

    struct fifo_data* fifo_data = malloc(sizeof(*fifo_data));
    if (!fifo_data) {
        put_inode(inode);
        return -ENOMEM;
    }
    fifo_data->fd_read = fd_read;
    fifo_data->fd_write = fd_write;

    inode->fs = &fifo_builtin_fs;
    inode->data = fifo_data;

    dent->inode = inode;

    return 0;
}

static ssize_t pipe_read(struct shim_handle* hdl, void* buf, size_t count, file_off_t* pos) {
    assert(hdl->type == TYPE_PIPE);
    __UNUSED(pos);

    if (!hdl->info.pipe.ready_for_ops)
        return -EACCES;

    size_t orig_count = count;
    int ret = DkStreamRead(hdl->pal_handle, 0, &count, buf, NULL, 0);
    ret = pal_to_unix_errno(ret);
    maybe_epoll_et_trigger(hdl, ret, /*in=*/true, ret == 0 ? count < orig_count : false);
    if (ret < 0) {
        return ret;
    }

    return (ssize_t)count;
}

static ssize_t pipe_write(struct shim_handle* hdl, const void* buf, size_t count, file_off_t* pos) {
    assert(hdl->type == TYPE_PIPE);
    __UNUSED(pos);

    if (!hdl->info.pipe.ready_for_ops)
        return -EACCES;

    size_t orig_count = count;
    int ret = DkStreamWrite(hdl->pal_handle, 0, &count, (void*)buf, NULL);
    ret = pal_to_unix_errno(ret);
    maybe_epoll_et_trigger(hdl, ret, /*in=*/false, ret == 0 ? count < orig_count : false);
    if (ret < 0) {
        if (ret == -EPIPE) {
            siginfo_t info = {
                .si_signo = SIGPIPE,
                .si_pid = g_process.pid,
                .si_code = SI_USER,
            };
            if (kill_current_proc(&info) < 0) {
                log_error("pipe_write: failed to deliver a signal");
            }
        }
        return ret;
    }

    return (ssize_t)count;
}

static int pipe_hstat(struct shim_handle* hdl, struct stat* stat) {
    /* XXX: Is any of this right?
     * Shouldn't we be using hdl to figure something out?
     * if stat is NULL, should we not return -EFAULT?
     */
    __UNUSED(hdl);
    if (!stat)
        return 0;

    struct shim_thread* thread = get_cur_thread();

    stat->st_dev     = (dev_t)0;           /* ID of device containing file */
    stat->st_ino     = (ino_t)0;           /* inode number */
    stat->st_nlink   = (nlink_t)0;         /* number of hard links */
    stat->st_uid     = (uid_t)thread->uid; /* user ID of owner */
    stat->st_gid     = (gid_t)thread->gid; /* group ID of owner */
    stat->st_rdev    = (dev_t)0;           /* device ID (if special file) */
    stat->st_size    = (off_t)0;           /* total size, in bytes */
    stat->st_blksize = 0;                  /* blocksize for file system I/O */
    stat->st_blocks  = 0;                  /* number of 512B blocks allocated */
    stat->st_atime   = (time_t)0;          /* access time */
    stat->st_mtime   = (time_t)0;          /* last modification */
    stat->st_ctime   = (time_t)0;          /* last status change */
    stat->st_mode    = PERM_rw_______ | S_IFIFO;

    return 0;
}

static int pipe_setflags(struct shim_handle* hdl, int flags) {
    if (!hdl->pal_handle)
        return 0;

    PAL_STREAM_ATTR attr;

    int ret = DkStreamAttributesQueryByHandle(hdl->pal_handle, &attr);
    if (ret < 0) {
        return pal_to_unix_errno(ret);
    }

    if (attr.nonblocking) {
        if (flags & O_NONBLOCK)
            return 0;

        attr.nonblocking = false;
    } else {
        if (!(flags & O_NONBLOCK))
            return 0;

        attr.nonblocking = true;
    }

    ret = DkStreamAttributesSetByHandle(hdl->pal_handle, &attr);
    if (ret < 0) {
        return pal_to_unix_errno(ret);
    }

    return 0;
}

static int fifo_open(struct shim_handle* hdl, struct shim_dentry* dent, int flags) {
    assert(locked(&g_dcache_lock));
    assert(dent->inode);

    struct fifo_data* fifo_data = dent->inode->data;

    /* FIXME: man 7 fifo says "[with non-blocking flag], opening for write-only fails with ENXIO
     *        unless the other end has already been opened". We cannot enforce this failure since
     *        Gramine doesn't know whether the other process already opened this FIFO. */

    if (flags & O_RDWR) {
        /* POSIX disallows FIFOs opened for read-write, but Linux allows it. We must choose only
         * one end (read or write) in our emulation, so we treat such FIFOs as read-only. This
         * covers most apps seen in the wild (in particular, LTP apps). */
        log_warning("FIFO (named pipe) '%s' cannot be opened in read-write mode in Gramine. "
                    "Treating it as read-only.", dent->mount->path);
        flags = O_RDONLY;
    }

    int fd;

    lock(&dent->inode->lock);
    if (flags & O_WRONLY) {
        fd = fifo_data->fd_write;
        fifo_data->fd_write = -1;
    } else {
        fd = fifo_data->fd_read;
        fifo_data->fd_read = -1;
    }
    unlock(&dent->inode->lock);

    if (fd == -1) {
        /* fd is invalid, happens if app tries to open the same FIFO end twice; this is ok in
         * normal Linux but Gramine uses TLS-encrypted pipes which are inherently point-to-point;
         * if this changes, should remove this error case (see GitHub issue #1417) */
        return -EOPNOTSUPP;
    }

    struct shim_handle* fifo_hdl = get_fd_handle(fd, /*fd_flags=*/NULL, /*map=*/NULL);
    if (!fifo_hdl) {
        return -ENOENT;
    }

    if (flags & O_NONBLOCK) {
        /* FIFOs were created in blocking mode (see shim_do_mknodat), change their attributes */
        int ret = pipe_setflags(fifo_hdl, flags);
        if (ret < 0) {
            put_handle(fifo_hdl);
            return ret;
        }
    }

    /* rewire new hdl to contents of intermediate FIFO hdl */
    assert(fifo_hdl->type == TYPE_PIPE);
    assert(fifo_hdl->uri);
    assert(fifo_hdl->acc_mode == ACC_MODE(flags & O_ACCMODE));

    hdl->type       = fifo_hdl->type;
    hdl->info       = fifo_hdl->info;
    hdl->pal_handle = fifo_hdl->pal_handle;

    hdl->uri = strdup(fifo_hdl->uri);
    if (!hdl->uri)
        return -ENOMEM;

    hdl->info.pipe.ready_for_ops = true;

    fifo_hdl->pal_handle = NULL; /* ownership of PAL handle is transferred to hdl */

    /* can remove intermediate FIFO hdl and its fd now */
    struct shim_handle* tmp = detach_fd_handle(fd, NULL, NULL);
    assert(tmp == fifo_hdl);
    put_handle(tmp);      /* matches detach_fd_handle() */
    put_handle(fifo_hdl); /* matches get_fd_handle() */

    return 0;
}

static struct shim_fs_ops pipe_fs_ops = {
    .read     = &pipe_read,
    .write    = &pipe_write,
    .hstat    = &pipe_hstat,
    .setflags = &pipe_setflags,
};

static struct shim_fs_ops fifo_fs_ops = {
    .read     = &pipe_read,
    .write    = &pipe_write,
    .setflags = &pipe_setflags,
};

static struct shim_d_ops fifo_d_ops = {
    .open        = &fifo_open,
    .icheckpoint = &fifo_icheckpoint,
    .irestore    = &fifo_irestore,
};

struct shim_fs pipe_builtin_fs = {
    .name   = "pipe",
    .fs_ops = &pipe_fs_ops,
};

struct shim_fs fifo_builtin_fs = {
    .name   = "fifo",
    .fs_ops = &fifo_fs_ops,
    .d_ops  = &fifo_d_ops,
};
