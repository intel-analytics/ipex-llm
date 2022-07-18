/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * Sync engine. The engine allows you to create *sync handles* with a global ID, and associated
 * data.
 *
 * A handle has to be locked before using, in one of two modes:
 *
 * - SYNC_STATE_SHARED: many processes can have a copy of the handle
 * - SYNC_STATE_EXCLUSIVE: only one process can have a copy of the handle, and that process is
 *   allowed to update the data stored with the handle
 *
 * When the handle is not locked, it can be downgraded to INVALID by the remote server. However,
 * this will only happen when another process needs to modify the same resource, and locks its own
 * handle in EXCLUSIVE mode (which means that data held by other processes are about to become
 * invalid). Therefore, as long as the handle is uncontested, there is no communication overhead for
 * using it repeatedly (only for first use and shutdown).
 *
 * Example usage (note that the "Lock" and "Unlock" parts should probably be extracted to helper
 * functions):
 *
 *     struct obj {
 *         int field_one;
 *         long field_two;
 *         struct sync_handle* sync;
 *     };
 *     struct obj obj = {0};
 *
 *     struct obj_sync_data {
 *         int field_one;
 *         long field_two;
 *     };
 *
 *     // Initialize the handle
 *     sync_create(&obj.sync, 0);
 *
 *     // Lock. Use SYNC_STATE_SHARED for reading data, SYNC_STATE_EXCLUSIVE if you need to update
 *     // it. After locking, you can read latest data (if it's there: a newly created handle will
 *     // not have any data associated).
 *     struct obj_sync_data data;
 *     bool updated = sync_lock(obj.sync, SYNC_STATE_EXCLUSIVE, &data, sizeof(data));
 *     if (updated) {
 *         obj.field_one = data.field_one;
 *         obj.field_two = data.field_two;
 *     }
 *
 *     // Use the object
 *     obj.field = ...;
 *
 *     // Unlock, writing the new data first
 *     data.field_one = obj.field_one;
 *     data.field_two = obj.field_two;
 *     sync_unlock(obj.sync, &data, sizeof(data));
 *
 *     // Destroy the handle before destroying the object
 *     sync_destroy(obj.sync);
 *
 * The sync engine is currently experimental. To enable it, set `libos.sync.enable = 1` in the
 * manifest. When it's not enabled, sync_lock() and sync_unlock() will function as regular, local
 * locks, and no remote communication will be performed.
 */

/*
 * Implementation overview:
 *
 * The sync engine uses a client/server architecture. The client code runs in all participating
 * processes, and the server code runs in the main process. The client and server communicate over
 * IPC.
 *
 * The protocol consists of the following interactions.
 *
 * 1. Upgrade:
 *    - client: REQUEST_UPGRADE(id, state)
 *    - server: CONFIRM_UPGRADE(id, state, data)
 *
 *    The client requests access to a resource with given ID, in either SHARED or EXCLUSIVE
 *    mode. The server downgrades the handles for other clients (if that's necessary for fulfilling
 *    the request), and replies with CONFIRM_UPGRADE once the resource can be used (sending latest
 *    data associated with it).
 *
 * 2. Downgrade:
 *    - server: REQUEST_DOWNGRADE(id, state)
 *    - client: CONFIRM_DOWNGRADE(id, state, data)
 *
 *    The server requests client to downgrade its handle. The client replies with CONFIRM_DOWNGRADE
 *    once it's not used anymore (sending latest data associated with it).
 *
 * 3. Close:
 *    - client: REQUEST_CLOSE(id, cur_state, data)
 *    - server: CONFIRM_CLOSE(id)
 *
 *    The client informs that it has stopped using a handle, and (if the state was EXCLUSIVE) sends
 *    latest data. After REQUEST_CLOSE, the client will send no further messages regarding the
 *    handle.
 *
 *    The server unregisters the handle for client, and replies with CONFIRM_CLOSE. After
 *    CONFIRM_CLOSE, the server will send no further messages related to the handle, and so the
 *    handle can be safely destroyed by client.
 *
 *    This is done by client for every handle it destroys, and should be done for all handles before
 *    process exit.
 *
 * Note that the request and confirmation aren't necessarily paired one-to-one. For example:
 *
 * - the client might send REQUEST_UPGRADE to SHARED state, then another one to EXCLUSIVE state,
 *   before the server replies with CONFIRM_UPGRADE to EXCLUSIVE
 * - the server might send REQUEST_DOWNGRADE to SHARED state, then another one to INVALID state,
 *   before the client replies with CONFIRM_DOWNGRADE to INVALID
 *
 * Here is an example interaction:
 *
 *                                                        client1 state    client2 state
 *                                                        INVALID          INVALID
 *  1. client1 -> server: REQUEST_UPGRADE(123, SHARED)    .                .
 *  2. server -> client1: CONFIRM_UPGRADE(123, SHARED)    SHARED           .
 *  3. client2 -> server: REQUEST_UPGRADE(123, SHARED)    |                .
 *  4. server -> client2: CONFIRM_UPGRADE(123, SHARED)    |                SHARED
 *  5. client1 -> server: REQUEST_UPGRADE(123, EXCLUSIVE) |                |
 *  6. server -> client2: REQUEST_DOWNGRADE(123, INVALID) |                |
 *  7. client2 -> server: CONFIRM_DOWNGRADE(123, INVALID) |                INVALID
 *  8. server -> client1: CONFIRM_UPGRADE(123, EXCLUSIVE) EXCLUSIVE        .
 *
 * In the above diagram, both clients request resource 123 in SHARED mode, and they hold SHARED
 * handles at the same time. Later, when the first client requests EXCLUSIVE access, the other
 * client is asked to downgrade to INVALID before the upgrade can be confirmed.
 */

/*
 * TODO (new features for additional use cases):
 * - mechanism for independently acquiring the same handle, without knowing the ID (e.g. for a given
 *   file path); will probably need REQUEST_OPEN/CONFIRM_OPEN messages
 * - conditional acquire of a handle, perhaps similar to FUTEX_WAIT_BITSET: would allow for more
 *   fine-grained locking
 */

#ifndef SHIM_SYNC_H_
#define SHIM_SYNC_H_

#include <stdint.h>

#include "list.h"
#include "pal.h"
#include "shim_types.h"

#define uthash_fatal(msg)                      \
    do {                                       \
        log_error("uthash error: %s", msg);    \
        DkProcessExit(ENOMEM);                 \
    } while (0)
#include "uthash.h"

/* Describes a state of a client handle, as recognized by client and server. */
enum {
    /* No state, used for {client,server}_req_state */
    SYNC_STATE_NONE,

    /* Invalid: client doesn't have latest data */
    SYNC_STATE_INVALID,

    /* Client has the latest data, but cannot modify it */
    SYNC_STATE_SHARED,

    /* Client has the latest data, and can modify it */
    SYNC_STATE_EXCLUSIVE,

    SYNC_STATE_NUM,
};

/* Lifecycle phase of a handle on client. */
enum {
    /* New handle, not registered on the server yet */
    SYNC_PHASE_NEW,

    /* Registered with server, reacts to messages normally */
    SYNC_PHASE_OPEN,

    /* After sending REQUEST_CLOSE: messages from server should be disregarded
     * (except for CONFIRM_CLOSE) */
    SYNC_PHASE_CLOSING,

    /* After receiving CONFIRM_CLOSE: can be safely destroyed */
    SYNC_PHASE_CLOSED,
};

struct sync_handle {
    uint64_t id;

    UT_hash_handle hh;

    /* Used by sync_lock .. sync_unlock. */
    struct shim_lock use_lock;

    /*
     * Internal properties lock. Protects all the following fields. If used together with use_lock,
     * then use_lock needs to be taken first.
     */
    struct shim_lock prop_lock;

    /* Size of synchronized data (0 if no data yet). */
    size_t data_size;

    /* Current version of synchronized data (NULL if no data yet). */
    void* data;

    /* Notification event for state changes. */
    PAL_HANDLE event;
    size_t n_waiters;

    /* Set to true if object is currently used (use_lock is locked). */
    bool used;

    /* Lifecycle phase (SYNC_PHASE_*) */
    int phase;

    /* Current state, lower or equal to server's cur_state */
    int cur_state;
    /* Requested by client; always higher than cur_state, or NONE */
    int client_req_state;
    /* Requested by server; always lower than cur_state, or NONE */
    int server_req_state;

    /* Reference count, used internally by sync client: the user of the handle just calls
     * `sync_create` / `sync_destroy`. */
    REFTYPE ref_count;
};

/*** User interface (sync_handle) ***/

/* Initialize the sync server. Should be called in the process leader. */
int init_sync_server(void);

/* Initialize the sync client. Should be done after the IPC subsystem (including sync server) is up
 * and running. */
int init_sync_client(void);

/* Close and destroy all the handles. Has to be called after other user threads finish, but before
 * we shut down the IPC helper. */
int shutdown_sync_client(void);

/* Create a new sync handle. If `id` is 0, allocate a fresh handle ID. */
int sync_create(struct sync_handle** handle, uint64_t id);

/* Destroy a sync handle, unregistering it from the server if necessary. */
void sync_destroy(struct sync_handle* handle);

/*
 * Acquire a handle in a given state (or higher). If the handle was updated, writes new data to
 * `data` and returns true, otherwise returns false.
 *
 * Provides the following guarantees:
 * - only one thread is holding a lock in a given process
 * - if state is SYNC_STATE_SHARED, no other process is holding a lock in SYNC_STATE_EXCLUSIVE state
 * - if state is SYNC_STATE_EXCLUSIVE, no other process is holding a lock in any state
 */
bool sync_lock(struct sync_handle* handle, int state, void* data, size_t data_size)
    __attribute__((warn_unused_result));

/* Release a handle, updating data associated with it. */
void sync_unlock(struct sync_handle* handle, void* data, size_t data_size);

/*** Message handlers (called from IPC, see ipc/shim_ipc_sync.c) ***/

struct shim_ipc_port;

void sync_client_message_callback(int code, uint64_t id, int state, size_t data_size, void* data);
void sync_server_message_callback(IDTYPE src, int code, uint64_t id, int state,
                                  size_t data_size, void* data);
void sync_server_disconnect_callback(IDTYPE src);

#endif /* SHIM_SYNC_H_ */
