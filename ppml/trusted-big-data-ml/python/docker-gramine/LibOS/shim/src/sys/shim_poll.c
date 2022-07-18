/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * Implementation of system calls "poll", "ppoll", "select" and "pselect6".
 */

#include <errno.h>

#include "pal.h"
#include "shim_fs.h"
#include "shim_handle.h"
#include "shim_internal.h"
#include "shim_lock.h"
#include "shim_signal.h"
#include "shim_table.h"
#include "shim_thread.h"
#include "shim_utils.h"

typedef unsigned long __fd_mask;

#ifndef __NFDBITS
#define __NFDBITS (8 * (int)sizeof(__fd_mask))
#endif

#ifndef __FDS_BITS
#define __FDS_BITS(set) ((set)->fds_bits)
#endif

#define __FD_ZERO(set)                                           \
    do {                                                         \
        unsigned int i;                                          \
        fd_set* arr = (set);                                     \
        for (i = 0; i < sizeof(fd_set) / sizeof(__fd_mask); i++) \
            __FDS_BITS(arr)[i] = 0;                              \
    } while (0)

#define __FD_ELT(d)        ((d) / __NFDBITS)
#define __FD_MASK(d)       ((__fd_mask)1 << ((d) % __NFDBITS))
#define __FD_SET(d, set)   ((void)(__FDS_BITS(set)[__FD_ELT(d)] |= __FD_MASK(d)))
#define __FD_CLR(d, set)   ((void)(__FDS_BITS(set)[__FD_ELT(d)] &= ~__FD_MASK(d)))
#define __FD_ISSET(d, set) ((__FDS_BITS(set)[__FD_ELT(d)] & __FD_MASK(d)) != 0)

static long _shim_do_poll(struct pollfd* fds, nfds_t nfds, uint64_t* timeout_us) {
    if ((uint64_t)nfds > get_rlimit_cur(RLIMIT_NOFILE))
        return -EINVAL;

    struct shim_handle_map* map = get_cur_thread()->handle_map;

    /* nfds is the upper limit for actual number of handles */
    PAL_HANDLE* pals = malloc(nfds * sizeof(PAL_HANDLE));
    if (!pals)
        return -ENOMEM;

    /* for bookkeeping, need to have a mapping FD -> {shim handle, index-in-pals} */
    struct fds_mapping_t {
        struct shim_handle* hdl; /* NULL if no mapping (handle is not used in polling) */
        nfds_t idx;              /* index from fds array to pals array */
    };
    struct fds_mapping_t* fds_mapping = malloc(nfds * sizeof(struct fds_mapping_t));
    if (!fds_mapping) {
        free(pals);
        return -ENOMEM;
    }

    /* allocate one memory region to hold two pal_wait_flags_t arrays: events and revents */
    pal_wait_flags_t* pal_events = malloc(nfds * sizeof(*pal_events) * 2);
    if (!pal_events) {
        free(pals);
        free(fds_mapping);
        return -ENOMEM;
    }
    pal_wait_flags_t* ret_events = pal_events + nfds;

    nfds_t pal_cnt  = 0;
    nfds_t nrevents = 0;

    lock(&map->lock);

    /* collect PAL handles that correspond to user-supplied FDs (only those that can be polled) */
    for (nfds_t i = 0; i < nfds; i++) {
        fds[i].revents = 0;
        fds_mapping[i].hdl = NULL;

        if (fds[i].fd < 0) {
            /* FD is negative, must be ignored */
            continue;
        }

        struct shim_handle* hdl = __get_fd_handle(fds[i].fd, NULL, map);
        if (!hdl || !hdl->fs || !hdl->fs->fs_ops) {
            /* The corresponding handle doesn't exist or doesn't provide FS-like semantics; do not
             * include it in handles-to-poll array but notify user about invalid request. */
            fds[i].revents = POLLNVAL;
            nrevents++;
            continue;
        }

        if (hdl->type == TYPE_CHROOT || hdl->type == TYPE_DEV || hdl->type == TYPE_STR ||
                hdl->type == TYPE_TMPFS) {
            /* Files, devs and strings are special cases: their poll is emulated at LibOS level; do
             * not include them in handles-to-poll array but instead use handle-specific
             * callback.
             *
             * TODO: we probably should use the poll() callback in all cases. */
            int shim_events = 0;
            if ((fds[i].events & (POLLIN | POLLRDNORM)) && (hdl->acc_mode & MAY_READ))
                shim_events |= FS_POLL_RD;
            if ((fds[i].events & (POLLOUT | POLLWRNORM)) && (hdl->acc_mode & MAY_WRITE))
                shim_events |= FS_POLL_WR;

            int shim_revents = hdl->fs->fs_ops->poll(hdl, shim_events);

            fds[i].revents = 0;
            if (shim_revents & FS_POLL_RD)
                fds[i].revents |= fds[i].events & (POLLIN | POLLRDNORM);
            if (shim_revents & FS_POLL_WR)
                fds[i].revents |= fds[i].events & (POLLOUT | POLLWRNORM);

            if (fds[i].revents)
                nrevents++;
            continue;
        }

        if (!hdl->pal_handle) {
            fds[i].revents = POLLNVAL;
            nrevents++;
            continue;
        }

        pal_wait_flags_t allowed_events = 0;
        if ((fds[i].events & (POLLIN | POLLRDNORM)) && (hdl->acc_mode & MAY_READ))
            allowed_events |= PAL_WAIT_READ;
        if ((fds[i].events & (POLLOUT | POLLWRNORM)) && (hdl->acc_mode & MAY_WRITE))
            allowed_events |= PAL_WAIT_WRITE;

        if ((fds[i].events & (POLLIN | POLLRDNORM | POLLOUT | POLLWRNORM)) && !allowed_events) {
            /* If user requested read/write events but they are not allowed on this handle, ignore
             * this handle (but note that user may only be interested in errors, and this is a valid
             * request). */
            continue;
        }

        get_handle(hdl);
        fds_mapping[i].hdl = hdl;
        fds_mapping[i].idx = pal_cnt;
        pals[pal_cnt] = hdl->pal_handle;
        pal_events[pal_cnt] = allowed_events;
        ret_events[pal_cnt] = 0;
        pal_cnt++;
    }

    unlock(&map->lock);

    bool polled = false;
    long error = 0;
    if (pal_cnt) {
        error = DkStreamsWaitEvents(pal_cnt, pals, pal_events, ret_events, timeout_us);
        polled = error == 0;
        error = pal_to_unix_errno(error);
    }

    for (nfds_t i = 0; i < nfds; i++) {
        if (!fds_mapping[i].hdl)
            continue;

        /* update fds.revents, but only if something was actually polled */
        if (polled) {
            fds[i].revents = 0;
            if (ret_events[fds_mapping[i].idx] & PAL_WAIT_ERROR)
                fds[i].revents |= POLLERR | POLLHUP;
            if (ret_events[fds_mapping[i].idx] & PAL_WAIT_READ)
                fds[i].revents |= fds[i].events & (POLLIN | POLLRDNORM);
            if (ret_events[fds_mapping[i].idx] & PAL_WAIT_WRITE)
                fds[i].revents |= fds[i].events & (POLLOUT | POLLWRNORM);

            if (fds[i].revents)
                nrevents++;
        }

        put_handle(fds_mapping[i].hdl);
    }

    free(pals);
    free(pal_events);
    free(fds_mapping);

    if (error == -EAGAIN) {
        /* `poll` returns 0 on timeout. */
        error = 0;
    } else if (error == -EINTR) {
        /* `poll`, `ppoll`, `select` and `pselect` are not restarted after being interrupted by
         * a signal handler. */
        error = -ERESTARTNOHAND;
    }
    return nrevents ? (long)nrevents : error;
}

long shim_do_poll(struct pollfd* fds, unsigned int nfds, int timeout_ms) {
    if (!is_user_memory_writable(fds, nfds * sizeof(*fds)))
        return -EFAULT;

    uint64_t timeout_us = (unsigned int)timeout_ms * TIME_US_IN_MS;
    return _shim_do_poll(fds, nfds, timeout_ms < 0 ? NULL : &timeout_us);
}

long shim_do_ppoll(struct pollfd* fds, unsigned int nfds, struct timespec* tsp,
                   const __sigset_t* sigmask_ptr, size_t sigsetsize) {
    if (!is_user_memory_writable(fds, nfds * sizeof(*fds))) {
        return -EFAULT;
    }

    int ret = set_user_sigmask(sigmask_ptr, sigsetsize);
    if (ret < 0) {
        return ret;
    }

    uint64_t timeout_us = tsp ? tsp->tv_sec * TIME_US_IN_S + tsp->tv_nsec / TIME_NS_IN_US : 0;
    return _shim_do_poll(fds, nfds, tsp ? &timeout_us : NULL);
}

long shim_do_select(int nfds, fd_set* readfds, fd_set* writefds, fd_set* errorfds,
                    struct __kernel_timeval* tsv) {
    if (tsv && (tsv->tv_sec < 0 || tsv->tv_usec < 0))
        return -EINVAL;

    if (nfds < 0 || (uint64_t)nfds > get_rlimit_cur(RLIMIT_NOFILE))
        return -EINVAL;

    if (!nfds) {
        if (!tsv)
            return shim_do_pause();

        /* special case of select(0, ..., tsv) used for sleep */
        return do_nanosleep(tsv->tv_sec * TIME_US_IN_S + tsv->tv_usec, NULL);
    }

    if (nfds < __NFDBITS) {
        /* interesting corner case: Linux always checks at least 64 first FDs */
        nfds = __NFDBITS;
    }

    /* nfds is the upper limit for actual number of fds for poll */
    struct pollfd* fds_poll = malloc(nfds * sizeof(struct pollfd));
    if (!fds_poll)
        return -ENOMEM;

    /* populate array of pollfd's based on user-supplied readfds & writefds */
    nfds_t nfds_poll = 0;
    for (int fd = 0; fd < nfds; fd++) {
        short events = 0;
        if (readfds && __FD_ISSET(fd, readfds))
            events |= POLLIN;
        if (writefds && __FD_ISSET(fd, writefds))
            events |= POLLOUT;

        if (!events)
            continue;

        fds_poll[nfds_poll].fd      = fd;
        fds_poll[nfds_poll].events  = events;
        fds_poll[nfds_poll].revents = 0;
        nfds_poll++;
    }

    /* select()/pselect() return -EBADF if invalid FD was given by user in readfds/writefds;
     * note that poll()/ppoll() don't have this error code, so we return this code only here */
    struct shim_handle_map* map = get_cur_thread()->handle_map;
    lock(&map->lock);
    for (nfds_t i = 0; i < nfds_poll; i++) {
        struct shim_handle* hdl = __get_fd_handle(fds_poll[i].fd, NULL, map);
        if (!hdl || !hdl->fs || !hdl->fs->fs_ops) {
            /* the corresponding handle doesn't exist or doesn't provide FS-like semantics */
            free(fds_poll);
            unlock(&map->lock);
            return -EBADF;
        }
    }
    unlock(&map->lock);

    uint64_t timeout_us = tsv ? tsv->tv_sec * TIME_US_IN_S + tsv->tv_usec : 0;
    long ret = _shim_do_poll(fds_poll, nfds_poll, tsv ? &timeout_us : NULL);

    if (ret < 0) {
        free(fds_poll);
        return ret;
    }

    /* modify readfds, writefds, and errorfds in-place with returned events */
    if (readfds)
        __FD_ZERO(readfds);
    if (writefds)
        __FD_ZERO(writefds);
    if (errorfds)
        __FD_ZERO(errorfds);

    ret = 0;
    for (nfds_t i = 0; i < nfds_poll; i++) {
        if (readfds && (fds_poll[i].revents & POLLIN)) {
            __FD_SET(fds_poll[i].fd, readfds);
            ret++;
        }
        if (writefds && (fds_poll[i].revents & POLLOUT)) {
            __FD_SET(fds_poll[i].fd, writefds);
            ret++;
        }
        if (errorfds && (fds_poll[i].revents & POLLERR)) {
            __FD_SET(fds_poll[i].fd, errorfds);
            ret++;
        }
    }

    free(fds_poll);
    return ret;
}

struct sigset_argpack {
    __sigset_t* p;
    size_t size;
};

long shim_do_pselect6(int nfds, fd_set* readfds, fd_set* writefds, fd_set* errorfds,
                      const struct __kernel_timespec* tsp, void* _sigmask_argpack) {
    struct sigset_argpack* sigmask_argpack = _sigmask_argpack;
    if (sigmask_argpack) {
        if (!is_user_memory_readable(sigmask_argpack, sizeof(*sigmask_argpack))) {
            return -EFAULT;
        }
        int ret = set_user_sigmask(sigmask_argpack->p, sigmask_argpack->size);
        if (ret < 0) {
            return ret;
        }
    }

    if (tsp) {
        struct __kernel_timeval tsv;
        tsv.tv_sec  = tsp->tv_sec;
        tsv.tv_usec = tsp->tv_nsec / 1000;
        return shim_do_select(nfds, readfds, writefds, errorfds, &tsv);
    }

    return shim_do_select(nfds, readfds, writefds, errorfds, NULL);
}
