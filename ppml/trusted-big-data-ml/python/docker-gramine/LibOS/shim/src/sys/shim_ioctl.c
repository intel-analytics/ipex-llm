/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * Implementation of system call "ioctl".
 */

#include <asm/ioctls.h>
#include <sys/ioctl.h>

#include "pal.h"
#include "shim_handle.h"
#include "shim_internal.h"
#include "shim_process.h"
#include "shim_signal.h"
#include "shim_table.h"

static void signal_io(IDTYPE caller, void* arg) {
    __UNUSED(caller);
    __UNUSED(arg);
    /* TODO: fill these values e.g. by getting the handle in arg; this is completely unusable now */
    siginfo_t info = {
        .si_signo = SIGIO,
        .si_code = SI_SIGIO,
        .si_band = 0,
        .si_fd = 0,
    };
    if (kill_current_proc(&info) < 0) {
        log_warning("signal_io: failed to deliver a signal");
    }
}

long shim_do_ioctl(unsigned int fd, unsigned int cmd, unsigned long arg) {
    struct shim_handle* hdl = get_fd_handle(fd, NULL, NULL);
    if (!hdl)
        return -EBADF;

    int ret = 0;
    switch (cmd) {
        case TIOCGPGRP:
            if (!hdl->uri || strcmp(hdl->uri, "dev:tty") != 0) {
                ret = -ENOTTY;
                break;
            }

            if (!is_user_memory_writable((void*)arg, sizeof(int))) {
                ret = -EFAULT;
                break;
            }
            *(int*)arg = __atomic_load_n(&g_process.pgid, __ATOMIC_ACQUIRE);
            ret = 0;
            break;
        case FIONBIO:
            if (!is_user_memory_readable((void*)arg, sizeof(int))) {
                ret = -EFAULT;
                break;
            }
            int nonblocking_on = *(int*)arg;
            ret = set_handle_nonblocking(hdl, !!nonblocking_on);
            break;
        case FIONCLEX:
            hdl->flags &= ~FD_CLOEXEC;
            ret = 0;
            break;
        case FIOCLEX:
            hdl->flags |= FD_CLOEXEC;
            ret = 0;
            break;
        case FIOASYNC:
            ret = install_async_event(hdl->pal_handle, 0, &signal_io, NULL);
            break;
        case FIONREAD: {
            if (!is_user_memory_writable((void*)arg, sizeof(int))) {
                ret = -EFAULT;
                break;
            }

            struct shim_fs* fs = hdl->fs;
            if (!fs || !fs->fs_ops) {
                ret = -EACCES;
                break;
            }

            int size = 0;
            if (fs->fs_ops->hstat) {
                struct stat stat;
                ret = fs->fs_ops->hstat(hdl, &stat);
                if (ret < 0)
                    break;
                size = stat.st_size;
            } else if (hdl->pal_handle) {
                PAL_STREAM_ATTR attr;
                ret = DkStreamAttributesQueryByHandle(hdl->pal_handle, &attr);
                if (ret < 0) {
                    ret = pal_to_unix_errno(ret);
                    break;
                }
                size = attr.pending_size;
            }

            int offset = 0;
            if (fs->fs_ops->seek) {
                ret = fs->fs_ops->seek(hdl, 0, SEEK_CUR);
                if (ret < 0)
                    break;
                offset = ret;
            }

            *(int*)arg = size - offset;
            ret = 0;
            break;
        }
        case SIOCGIFCONF:
        case SIOCGIFHWADDR:
            if (hdl->type == TYPE_SOCK) {
                /* LibOS doesn't know how to handle this IOCTL, forward it to the host */
                ret = DkDeviceIoControl(hdl->pal_handle, cmd, arg);
                if (ret < 0)
                    ret = pal_to_unix_errno(ret);
            }
            break;
        default:
            ret = -ENOSYS;
            break;
    }

    put_handle(hdl);
    if (ret == -EINTR) {
        ret = -ERESTARTSYS;
    }
    return ret;
}
