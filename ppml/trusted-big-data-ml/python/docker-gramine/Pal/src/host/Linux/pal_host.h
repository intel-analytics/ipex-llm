/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains definition of PAL host ABI.
 */

#ifndef PAL_HOST_H
#define PAL_HOST_H

#ifndef IN_PAL
#error "cannot be included outside PAL"
#endif

#include <stdbool.h>
#include <stdint.h>

#include "spinlock.h"

typedef struct {
    char str[PIPE_NAME_MAX];
} PAL_PIPE_NAME;

typedef struct {
    /* TSAI: Here we define the internal types of PAL_HANDLE
     * in PAL design, user has not to access the content inside the
     * handle, also there is no need to allocate the internal
     * handles, so we hide the type name of these handles on purpose.
     */
    PAL_HDR hdr;
    /* Bitmask of `PAL_HANDLE_FD_*` flags. */
    uint32_t flags;

    union {
        /* Common field for accessing underlying host fd. See also `PAL_HANDLE_FD_READABLE`. */
        struct {
            PAL_IDX fd;
        } generic;

        struct {
            PAL_IDX fd;
            const char* realpath;
            /*
             * map_start is to request this file should be mapped to this
             * address. When fork is emulated, the address is already
             * determined by parent process.
             */
            void* map_start;
            bool seekable; /* regular files are seekable, FIFO pipes are not */
        } file;

        struct {
            PAL_IDX fd;
            PAL_PIPE_NAME name;
            bool nonblocking;
        } pipe;

        struct {
            PAL_IDX fd;
            /* TODO: add other flags in future, if needed (e.g., semaphore) */
            bool nonblocking;
        } eventfd;

        struct {
            PAL_IDX fd;
            bool nonblocking;
        } dev;

        struct {
            PAL_IDX fd;
            const char* realpath;
            void* buf;
            void* ptr;
            void* end;
            bool endofstream;
        } dir;

        struct {
            PAL_IDX fd;
            struct sockaddr* bind;
            struct sockaddr* conn;
            bool nonblocking;
            bool reuseaddr;
            PAL_NUM linger;
            PAL_NUM receivebuf;
            PAL_NUM sendbuf;
            uint64_t receivetimeout_us;
            uint64_t sendtimeout_us;
            bool tcp_cork;
            bool tcp_keepalive;
            bool tcp_nodelay;
        } sock;

        struct {
            PAL_IDX stream;
            bool nonblocking;
        } process;

        struct {
            PAL_IDX tid;
            void* stack;
        } thread;

        struct {
            spinlock_t lock;
            uint32_t waiters_cnt;
            uint32_t signaled;
            bool auto_clear;
        } event;
    };
}* PAL_HANDLE;

#define HANDLE_TYPE(handle) ((handle)->hdr.type)

/* These two flags indicate whether the underlying host fd of `PAL_HANDLE` is readable and/or
 * writable respectively. If none of these is set, then the handle has no host-level fd. */
#define PAL_HANDLE_FD_READABLE  1
#define PAL_HANDLE_FD_WRITABLE  2
/* Set if an error was seen on this handle. Currently only set by `_DkStreamsWaitEvents`. */
#define PAL_HANDLE_FD_ERROR     4

int arch_do_rt_sigprocmask(int sig, int how);
int arch_do_rt_sigaction(int sig, void* handler,
                         const int* async_signals, size_t async_signals_cnt);

#endif /* PAL_HOST_H */
