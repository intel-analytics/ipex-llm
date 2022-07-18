/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * Implementation of system calls "dup", "dup2" and "dup3".
 */

#include <errno.h>

#include "shim_handle.h"
#include "shim_internal.h"
#include "shim_table.h"
#include "shim_thread.h"

long shim_do_dup(unsigned int fd) {
    struct shim_handle_map* handle_map = get_thread_handle_map(NULL);
    assert(handle_map);

    struct shim_handle* hdl = get_fd_handle(fd, NULL, handle_map);
    if (!hdl)
        return -EBADF;

    // dup() always zeroes fd flags
    int vfd = set_new_fd_handle(hdl, /*fd_flags=*/0, handle_map);
    put_handle(hdl);
    return vfd == -ENOMEM ? -EMFILE : vfd;
}

long shim_do_dup2(unsigned int oldfd, unsigned int newfd) {
    if (newfd >= get_rlimit_cur(RLIMIT_NOFILE))
        return -EBADF;

    struct shim_handle_map* handle_map = get_thread_handle_map(NULL);
    assert(handle_map);

    struct shim_handle* hdl = get_fd_handle(oldfd, NULL, handle_map);
    if (!hdl)
        return -EBADF;

    if (oldfd == newfd) {
        put_handle(hdl);
        return newfd;
    }

    struct shim_handle* new_hdl = detach_fd_handle(newfd, NULL, handle_map);

    if (new_hdl)
        put_handle(new_hdl);

    // dup2() always zeroes fd flags
    int vfd = set_new_fd_handle_by_fd(newfd, hdl, /*fd_flags=*/0, handle_map);
    put_handle(hdl);
    return vfd == -ENOMEM ? -EMFILE : vfd;
}

long shim_do_dup3(unsigned int oldfd, unsigned int newfd, int flags) {
    if ((flags & ~O_CLOEXEC) || oldfd == newfd)
        return -EINVAL;

    struct shim_handle_map* handle_map = get_thread_handle_map(NULL);
    assert(handle_map);

    struct shim_handle* hdl = get_fd_handle(oldfd, NULL, handle_map);
    if (!hdl)
        return -EBADF;

    struct shim_handle* new_hdl = detach_fd_handle(newfd, NULL, handle_map);

    if (new_hdl)
        put_handle(new_hdl);

    int fd_flags = (flags & O_CLOEXEC) ? FD_CLOEXEC : 0;
    int vfd = set_new_fd_handle_by_fd(newfd, hdl, fd_flags, handle_map);
    put_handle(hdl);
    return vfd == -ENOMEM ? -EMFILE : vfd;
}
