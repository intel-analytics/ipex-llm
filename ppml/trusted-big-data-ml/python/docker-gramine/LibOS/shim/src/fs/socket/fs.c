/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains code for implementation of the `socket` filesystem.
 */

#include <asm/fcntl.h>
#include <errno.h>

#include "pal.h"
#include "shim_fs.h"
#include "shim_internal.h"
#include "shim_lock.h"
#include "shim_process.h"
#include "shim_signal.h"
#include "stat.h"

int unix_socket_setup_dentry(struct shim_dentry* dent, mode_t perm) {
    assert(locked(&g_dcache_lock));
    assert(!dent->inode);

    struct shim_inode* inode = get_new_inode(dent->mount, S_IFSOCK, perm);
    if (!inode)
        return -ENOMEM;

    inode->fs = &socket_builtin_fs;

    dent->inode = inode;
    return 0;
}

static int socket_close(struct shim_handle* hdl) {
    /* XXX: Shouldn't this do something? */
    __UNUSED(hdl);
    return 0;
}

static ssize_t socket_read(struct shim_handle* hdl, void* buf, size_t count, file_off_t* pos) {
    assert(hdl->type == TYPE_SOCK);
    __UNUSED(pos);

    struct shim_sock_handle* sock = &hdl->info.sock;

    lock(&hdl->lock);

    if (sock->sock_type == SOCK_STREAM && sock->sock_state != SOCK_ACCEPTED &&
        sock->sock_state != SOCK_CONNECTED && sock->sock_state != SOCK_BOUNDCONNECTED) {
        sock->error = ENOTCONN;
        unlock(&hdl->lock);
        return -ENOTCONN;
    }

    if (sock->sock_type == SOCK_DGRAM && sock->sock_state != SOCK_CONNECTED &&
        sock->sock_state != SOCK_BOUNDCONNECTED) {
        sock->error = EDESTADDRREQ;
        unlock(&hdl->lock);
        return -EDESTADDRREQ;
    }

    unlock(&hdl->lock);

    size_t orig_count = count;
    int ret = DkStreamRead(hdl->pal_handle, 0, &count, buf, NULL, 0);
    ret = pal_to_unix_errno(ret);
    maybe_epoll_et_trigger(hdl, ret, /*in=*/true, ret == 0 ? count < orig_count : false);
    if (ret < 0) {
        lock(&hdl->lock);
        sock->error = -ret;
        unlock(&hdl->lock);
        return ret;
    }

    return (ssize_t)count;
}

static ssize_t socket_write(struct shim_handle* hdl, const void* buf, size_t count,
                            file_off_t* pos) {
    assert(hdl->type == TYPE_SOCK);
    __UNUSED(pos);

    struct shim_sock_handle* sock = &hdl->info.sock;

    lock(&hdl->lock);

    if (sock->sock_type == SOCK_STREAM && sock->sock_state != SOCK_ACCEPTED &&
        sock->sock_state != SOCK_CONNECTED && sock->sock_state != SOCK_BOUNDCONNECTED) {
        sock->error = ENOTCONN;
        unlock(&hdl->lock);
        return -ENOTCONN;
    }

    if (sock->sock_type == SOCK_DGRAM && sock->sock_state != SOCK_CONNECTED &&
        sock->sock_state != SOCK_BOUNDCONNECTED) {
        sock->error = EDESTADDRREQ;
        unlock(&hdl->lock);
        return -EDESTADDRREQ;
    }

    unlock(&hdl->lock);

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
                log_error("socket_write: failed to deliver a signal");
            }
        }

        lock(&hdl->lock);
        sock->error = -ret;
        unlock(&hdl->lock);
        return ret;
    }

    return (ssize_t)count;
}

static int socket_hstat(struct shim_handle* hdl, struct stat* stat) {
    if (!stat)
        return 0;

    PAL_STREAM_ATTR attr;

    int ret = DkStreamAttributesQueryByHandle(hdl->pal_handle, &attr);
    if (ret < 0) {
        return pal_to_unix_errno(ret);
    }

    memset(stat, 0, sizeof(struct stat));

    stat->st_ino  = 0;
    stat->st_size = (off_t)attr.pending_size;
    stat->st_mode = S_IFSOCK;

    return 0;
}

static int socket_setflags(struct shim_handle* hdl, int flags) {
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

struct shim_fs_ops socket_fs_ops = {
    .close    = &socket_close,
    .read     = &socket_read,
    .write    = &socket_write,
    .hstat    = &socket_hstat,
    .setflags = &socket_setflags,
};

struct shim_fs socket_builtin_fs = {
    .name   = "socket",
    .fs_ops = &socket_fs_ops,
};
