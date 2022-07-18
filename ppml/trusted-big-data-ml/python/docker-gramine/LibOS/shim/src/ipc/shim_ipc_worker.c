/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University
 * Copyright (C) 2021 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

#include <stdint.h>
#include <stdnoreturn.h>

#include "api.h"
#include "assert.h"
#include "cpu.h"
#include "list.h"
#include "pal.h"
#include "shim_internal.h"
#include "shim_ipc.h"
#include "shim_lock.h"
#include "shim_pollable_event.h"
#include "shim_thread.h"
#include "shim_types.h"
#include "shim_utils.h"

#define LOG_PREFIX "IPC worker: "

DEFINE_LIST(shim_ipc_connection);
DEFINE_LISTP(shim_ipc_connection);
struct shim_ipc_connection {
    LIST_TYPE(shim_ipc_connection) list;
    PAL_HANDLE handle;
    IDTYPE vmid;
};

/* List of incoming IPC connections, fully managed by this IPC worker thread (hence no locking
 * needed). */
static LISTP_TYPE(shim_ipc_connection) g_ipc_connections;
static size_t g_ipc_connections_cnt = 0;

static struct shim_thread* g_worker_thread = NULL;
/* Used by `DkThreadExit` to indicate that the thread really exited and is not using any resources
 * (e.g. stack) anymore. Awaited to be `0` (thread exited) in `terminate_ipc_worker()`. */
static int g_clear_on_worker_exit = 1;
static PAL_HANDLE g_self_ipc_handle = NULL;

typedef int (*ipc_callback)(IDTYPE src, void* data, uint64_t seq);
static ipc_callback ipc_callbacks[] = {
    [IPC_MSG_RESP]              = ipc_response_callback,
    [IPC_MSG_GET_NEW_VMID]      = ipc_get_new_vmid_callback,
    [IPC_MSG_CHILDEXIT]         = ipc_cld_exit_callback,
    [IPC_MSG_ALLOC_ID_RANGE]    = ipc_alloc_id_range_callback,
    [IPC_MSG_RELEASE_ID_RANGE]  = ipc_release_id_range_callback,
    [IPC_MSG_CHANGE_ID_OWNER]   = ipc_change_id_owner_callback,
    [IPC_MSG_GET_ID_OWNER]      = ipc_get_id_owner_callback,
    [IPC_MSG_PID_KILL]          = ipc_pid_kill_callback,
    [IPC_MSG_PID_GETMETA]       = ipc_pid_getmeta_callback,

    [IPC_MSG_SYNC_REQUEST_UPGRADE]   = ipc_sync_request_upgrade_callback,
    [IPC_MSG_SYNC_REQUEST_DOWNGRADE] = ipc_sync_request_downgrade_callback,
    [IPC_MSG_SYNC_REQUEST_CLOSE]     = ipc_sync_request_close_callback,
    [IPC_MSG_SYNC_CONFIRM_UPGRADE]   = ipc_sync_confirm_upgrade_callback,
    [IPC_MSG_SYNC_CONFIRM_DOWNGRADE] = ipc_sync_confirm_downgrade_callback,
    [IPC_MSG_SYNC_CONFIRM_CLOSE]     = ipc_sync_confirm_close_callback,

    [IPC_MSG_POSIX_LOCK_SET]       = ipc_posix_lock_set_callback,
    [IPC_MSG_POSIX_LOCK_GET]       = ipc_posix_lock_get_callback,
    [IPC_MSG_POSIX_LOCK_CLEAR_PID] = ipc_posix_lock_clear_pid_callback,
};

static void ipc_leader_died_callback(void) {
    /* This might happen legitimately e.g. if IPC leader is also our parent and does `wait` + `exit`
     * If this is an erroneous disconnect it will be noticed when trying to communicate with
     * the leader. */
    log_debug("IPC leader disconnected");
}

static void disconnect_callbacks(struct shim_ipc_connection* conn) {
    if (g_process_ipc_ids.leader_vmid == conn->vmid) {
        ipc_leader_died_callback();
    }
    ipc_child_disconnect_callback(conn->vmid);

    if (!g_process_ipc_ids.leader_vmid) {
        sync_server_disconnect_callback(conn->vmid);
    }

    /*
     * Currently outgoing IPC connections (handled in `shim_ipc.c`) are not cleaned up - there is
     * no place that can notice disconnection of an outgoing connection other than a failure to send
     * data via such connection. We try to remove an outgoing IPC connection to a process that just
     * disconnected here - usually we have connections set up in both ways.
     * This also wakes all message response waiters (if there are any).
    */
    remove_outgoing_ipc_connection(conn->vmid);
}

static int add_ipc_connection(PAL_HANDLE handle, IDTYPE id) {
    struct shim_ipc_connection* conn = malloc(sizeof(*conn));
    if (!conn) {
        return -ENOMEM;
    }

    conn->handle = handle;
    conn->vmid = id;

    LISTP_ADD(conn, &g_ipc_connections, list);
    g_ipc_connections_cnt++;
    return 0;
}

static void del_ipc_connection(struct shim_ipc_connection* conn) {
    LISTP_DEL(conn, &g_ipc_connections, list);
    g_ipc_connections_cnt--;

    DkObjectClose(conn->handle);

    free(conn);
}

/*
 * Receive and handle some (possibly many) messages from IPC connection `conn`.
 * Returns `0` on success, `1` on EOF (connection closed on a message boundary), negative error
 * code on failures.
 */
static int receive_ipc_messages(struct shim_ipc_connection* conn) {
    size_t size = 0;
    /* Try to get more bytes that strictly required in case there are more messages waiting.
     * `0x40` as a random estimation of "couple of ints" + message header size to get the next
     * message header if possible. */
#define READAHEAD_SIZE (0x40 + sizeof(struct ipc_msg_header))
    union {
        struct ipc_msg_header msg_header;
        char buf[sizeof(struct ipc_msg_header) + READAHEAD_SIZE];
    } buf;
#undef READAHEAD_SIZE

    do {
        /* Receive at least the message header. */
        while (size < sizeof(buf.msg_header)) {
            size_t tmp_size = sizeof(buf) - size;
            int ret = DkStreamRead(conn->handle, /*offset=*/0, &tmp_size, buf.buf + size, NULL, 0);
            if (ret < 0) {
                if (ret == -PAL_ERROR_INTERRUPTED || ret == -PAL_ERROR_TRYAGAIN) {
                    continue;
                }
                ret = pal_to_unix_errno(ret);
                log_error(LOG_PREFIX "receiving message header from %u failed: %d", conn->vmid,
                          ret);
                return ret;
            }
            if (tmp_size == 0) {
                if (size == 0) {
                    /* EOF on the handle, but exactly on the message boundary. */
                    return 1;
                }
                log_error(LOG_PREFIX "receiving message from %u failed: remote closed early",
                          conn->vmid);
                return -ENODATA;
            }
            size += tmp_size;
        }

        size_t msg_size = GET_UNALIGNED(buf.msg_header.size);
        assert(msg_size >= sizeof(struct ipc_msg_header));
        size_t data_size = msg_size - sizeof(struct ipc_msg_header);
        void* msg_data = malloc(data_size);
        if (!msg_data) {
            return -ENOMEM;
        }

        unsigned char msg_code = GET_UNALIGNED(buf.msg_header.code);
        unsigned long msg_seq = GET_UNALIGNED(buf.msg_header.seq);

        if (msg_size <= size) {
            /* Already got the whole message (and possibly part of the next one). */
            memcpy(msg_data, buf.buf + sizeof(struct ipc_msg_header), data_size);
            memmove(buf.buf, buf.buf + msg_size, size - msg_size);
            size -= msg_size;
        } else {
            /* Need to get rest of the message. */
            assert(size >= sizeof(struct ipc_msg_header));
            size_t current_size = size - sizeof(struct ipc_msg_header);
            memcpy(msg_data, buf.buf + sizeof(struct ipc_msg_header), current_size);

            int ret = read_exact(conn->handle, (char*)msg_data + current_size,
                                 data_size - current_size);
            if (ret < 0) {
                free(msg_data);
                log_error(LOG_PREFIX "receiving message from %u failed: %d", conn->vmid, ret);
                return ret;
            }
            size = 0;
        }

        log_debug(LOG_PREFIX "received IPC message from %u: code=%d size=%lu seq=%lu", conn->vmid,
                  msg_code, msg_size, msg_seq);

        int ret = 0;
        if (msg_code < ARRAY_SIZE(ipc_callbacks) && ipc_callbacks[msg_code]) {
            ret = ipc_callbacks[msg_code](conn->vmid, msg_data, msg_seq);
            if (ret < 0) {
                log_error(LOG_PREFIX "error running IPC callback %u: %d", msg_code, ret);
                DkProcessExit(1);
            }
        } else {
            log_error(LOG_PREFIX "received unknown IPC msg type: %u", msg_code);
        }

        if (msg_code != IPC_MSG_RESP) {
            free(msg_data);
        }
    } while (size > 0);

    return 0;
}

static noreturn void ipc_worker_main(void) {
    /* TODO: If we had a global array of connections (instead of a list) we wouldn't have to gather
     * them all here in every loop iteration, but then deletion would be slower (but deletion should
     * be rare). */
    struct shim_ipc_connection** connections = NULL;
    PAL_HANDLE* handles = NULL;
    pal_wait_flags_t* events = NULL;
    pal_wait_flags_t* ret_events = NULL;
    size_t prev_items_cnt = 0;

    while (1) {
        /* Reserve 2 slots for `g_worker_thread->pollable_event` and `g_self_ipc_handle`. */
        const size_t reserved_slots = 2;
        size_t items_cnt = g_ipc_connections_cnt + reserved_slots;
        if (items_cnt != prev_items_cnt) {
            free(connections);
            free(handles);
            free(events);
            free(ret_events);

            connections = malloc(items_cnt * sizeof(*connections));
            handles = malloc(items_cnt * sizeof(*handles));
            events = malloc(items_cnt * sizeof(*events));
            ret_events = malloc(items_cnt * sizeof(*ret_events));
            if (!connections || !handles || !events || !ret_events) {
                log_error(LOG_PREFIX "arrays allocation failed");
                goto out_die;
            }

            prev_items_cnt = items_cnt;
        }

        memset(ret_events, 0, items_cnt * sizeof(*ret_events));

        connections[0] = NULL;
        handles[0] = g_worker_thread->pollable_event.read_handle;
        events[0] = PAL_WAIT_READ;
        connections[1] = NULL;
        handles[1] = g_self_ipc_handle;
        events[1] = PAL_WAIT_READ;

        struct shim_ipc_connection* conn;
        size_t i = reserved_slots;
        LISTP_FOR_EACH_ENTRY(conn, &g_ipc_connections, list) {
            connections[i] = conn;
            handles[i] = conn->handle;
            events[i] = PAL_WAIT_READ;
            /* `ret_events[i]` already cleared. */
            i++;
        }

        int ret = DkStreamsWaitEvents(items_cnt, handles, events, ret_events, /*timeout_us=*/NULL);
        if (ret < 0) {
            if (ret == -PAL_ERROR_INTERRUPTED) {
                /* Generally speaking IPC worker should not be interrupted, but this happens with
                 * SGX exitless feature. */
                continue;
            }
            ret = pal_to_unix_errno(ret);
            log_error(LOG_PREFIX "DkStreamsWaitEvents failed: %d", ret);
            goto out_die;
        }

        if (ret_events[0]) {
            /* `g_worker_thread->pollable_event` */
            if (ret_events[0] & ~PAL_WAIT_READ) {
                log_error(LOG_PREFIX "unexpected event (%d) on exit handle", ret_events[0]);
                goto out_die;
            }
            log_debug(LOG_PREFIX "exiting worker thread");

            free(connections);
            free(handles);
            free(events);
            free(ret_events);

            struct shim_thread* cur_thread = get_cur_thread();
            assert(g_worker_thread == cur_thread);
            assert(cur_thread->shim_tcb->tp == cur_thread);
            cur_thread->shim_tcb->tp = NULL;
            put_thread(cur_thread);

            DkThreadExit(&g_clear_on_worker_exit);
            /* Unreachable. */
        }

        if (ret_events[1]) {
            /* New connection incoming. */
            if (ret_events[1] & ~PAL_WAIT_READ) {
                log_error(LOG_PREFIX "unexpected event (%d) on listening handle", ret_events[1]);
                goto out_die;
            }
            PAL_HANDLE new_handle = NULL;
            do {
                /* Although IPC worker thread does not handle any signals (hence it should never be
                 * interrupted), lets handle it for uniformity with the rest of the code. */
                ret = DkStreamWaitForClient(g_self_ipc_handle, &new_handle, /*options=*/0);
            } while (ret == -PAL_ERROR_INTERRUPTED);
            if (ret < 0) {
                ret = pal_to_unix_errno(ret);
                log_error(LOG_PREFIX "DkStreamWaitForClient failed: %d", ret);
                goto out_die;
            }
            IDTYPE new_id = 0;
            ret = read_exact(new_handle, &new_id, sizeof(new_id));
            if (ret < 0) {
                log_error(LOG_PREFIX "receiving id failed: %d", ret);
                DkObjectClose(new_handle);
            } else {
                ret = add_ipc_connection(new_handle, new_id);
                if (ret < 0) {
                    log_error(LOG_PREFIX "add_ipc_connection failed: %d", ret);
                    goto out_die;
                }
            }
        }

        for (i = reserved_slots; i < items_cnt; i++) {
            conn = connections[i];
            if (ret_events[i] & PAL_WAIT_READ) {
                ret = receive_ipc_messages(conn);
                if (ret == 1) {
                    /* Connection closed. */
                    disconnect_callbacks(conn);
                    del_ipc_connection(conn);
                    continue;
                }
                if (ret < 0) {
                    log_error(LOG_PREFIX "failed to receive an IPC message from %u: %d",
                              conn->vmid, ret);
                    /* Let the code below handle this error. */
                    ret_events[i] = PAL_WAIT_ERROR;
                }
            }
            /* If there was something else other than error reported, let the loop spin at least one
             * more time - in case there are messages left to be read. */
            if (ret_events[i] == PAL_WAIT_ERROR) {
                disconnect_callbacks(conn);
                del_ipc_connection(conn);
            }
        }
    }

out_die:
    DkProcessExit(1);
}

static int ipc_worker_wrapper(void* arg) {
    __UNUSED(arg);
    assert(g_worker_thread);

    shim_tcb_init();
    set_cur_thread(g_worker_thread);

    log_setprefix(shim_get_tcb());

    log_debug("IPC worker started");
    ipc_worker_main();
    /* Unreachable. */
}

static int init_self_ipc_handle(void) {
    char uri[PIPE_URI_SIZE];
    return create_pipe(/*name=*/NULL, uri, sizeof(uri), &g_self_ipc_handle,
                       /*use_vmid_for_name=*/true);
}

static int create_ipc_worker(void) {
    int ret = init_self_ipc_handle();
    if (ret < 0) {
        return ret;
    }

    g_worker_thread = get_new_internal_thread();
    if (!g_worker_thread) {
        return -ENOMEM;
    }

    PAL_HANDLE handle = NULL;
    ret = DkThreadCreate(ipc_worker_wrapper, NULL, &handle);
    if (ret < 0) {
        put_thread(g_worker_thread);
        g_worker_thread = NULL;
        return pal_to_unix_errno(ret);
    }

    g_worker_thread->pal_handle = handle;

    return 0;
}

int init_ipc_worker(void) {
    enable_locking();
    return create_ipc_worker();
}

void terminate_ipc_worker(void) {
    set_pollable_event(&g_worker_thread->pollable_event, 1);

    while (__atomic_load_n(&g_clear_on_worker_exit, __ATOMIC_ACQUIRE)) {
        CPU_RELAX();
    }

    put_thread(g_worker_thread);
    g_worker_thread = NULL;
    DkObjectClose(g_self_ipc_handle);
    g_self_ipc_handle = NULL;
}
