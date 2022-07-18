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
#include <stddef.h>
#include <stdint.h>

#include "atomic.h"
#include "enclave_tf_structs.h"
#include "list.h"
#include "spinlock.h"

void* malloc_untrusted(size_t size);
void free_untrusted(void* mem);

DEFINE_LIST(pal_handle_thread);
struct pal_handle_thread {
    PAL_HDR reserved;
    PAL_IDX tid;
    void* tcs;
    LIST_TYPE(pal_handle_thread) list;
    void* param;
};

typedef struct {
    char str[PIPE_NAME_MAX];
} PAL_PIPE_NAME;

/* RPC streams are encrypted with 256-bit AES keys */
typedef uint8_t PAL_SESSION_KEY[32];

typedef struct {
    /*
     * Here we define the internal structure of PAL_HANDLE.
     * user has no access to the content inside these handles.
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
            PAL_NUM total;
            /* below fields are used only for trusted files */
            sgx_chunk_hash_t* chunk_hashes; /* array of hashes of file chunks */
            void* umem;                     /* valid only when chunk_hashes != NULL */
            bool seekable;                  /* regular files are seekable, FIFO pipes are not */
        } file;

        struct {
            PAL_IDX fd;
            PAL_PIPE_NAME name;
            bool nonblocking;
            bool is_server;
            PAL_SESSION_KEY session_key;
            PAL_NUM handshake_done;
            void* ssl_ctx;
            void* handshake_helper_thread_hdl;
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
            bool is_server;
            PAL_SESSION_KEY session_key;
            void* ssl_ctx;
        } process;

        struct pal_handle_thread thread;

        struct {
            /* Guards accesses to the rest of the fields.
             * We need to be able to set `signaled` and `signaled_untrusted` atomically, which is
             * impossible without a lock. They are essentialy the same field, but we need two
             * separate copies, because we need to guard against malicious host modifications yet
             * still be able to call futex on it. */
            spinlock_t lock;
            /* Current number of waiters - used solely as an optimization. `uint32_t` because futex
             * syscall does not allow for more than `INT_MAX` waiters anyway. */
            uint32_t waiters_cnt;
            bool signaled;
            bool auto_clear;
            /* Access to the *content* of this field should be atomic, because it's used as futex
             * word on the untrusted host. */
            uint32_t* signaled_untrusted;
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

#endif /* PAL_HOST_H */
