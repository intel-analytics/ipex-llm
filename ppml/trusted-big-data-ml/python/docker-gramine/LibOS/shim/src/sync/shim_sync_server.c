/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * Server part of the sync engine.
 *
 * TODO (performance): All the server work is protected by a global lock, and happens in one thread
 * (IPC helper). With high volume of requests, this might be a performance bottleneck.
 */

#include "pal.h"
#include "shim_ipc.h"
#include "shim_lock.h"
#include "shim_sync.h"

#define FATAL(fmt...)                                   \
    do {                                                \
        log_error("Fatal error in sync server: " fmt);  \
        DkProcessExit(1);                               \
    } while(0)

DEFINE_LIST(server_lease);
DEFINE_LISTP(server_lease);

/* Handle (stored in a per-id hash table) */
struct server_handle {
    uint64_t id;
    size_t data_size;
    void* data;

    LISTP_TYPE(server_lease) leases;

    UT_hash_handle hh;
};

/* Client, stored in a per-vmid hash table. Used to track open leases by a client. */
struct server_client {
    IDTYPE vmid;
    LISTP_TYPE(server_lease) leases;

    UT_hash_handle hh;
};

/* Lease for a given handle and client */
struct server_lease {
    struct server_client* client;

    /* Current state (INVALID, SHARED or EXCLUSIVE); higher or equal to client's cur_state */
    int cur_state;
    /* Requested by client; always higher than cur_state, or NONE */
    int client_req_state;
    /* Requested by server; always lower than cur_state, or NONE */
    int server_req_state;

    LIST_TYPE(server_lease) list_client;
    LIST_TYPE(server_lease) list_handle;
};

static struct server_handle* g_server_handles = NULL;
static struct server_client* g_server_clients = NULL;
static struct shim_lock g_server_lock;

int init_sync_server(void) {
    if (!create_lock(&g_server_lock))
        return -ENOMEM;
    return 0;
}

static struct server_handle* find_handle(uint64_t id, bool create) {
    assert(locked(&g_server_lock));

    struct server_handle* handle = NULL;
    HASH_FIND(hh, g_server_handles, &id, sizeof(id), handle);
    if (handle)
        return handle;

    if (!create)
        return NULL;

    if (!(handle = malloc(sizeof(*handle))))
        return NULL;

    handle->id = id;
    handle->data_size = 0;
    handle->data = NULL;
    INIT_LISTP(&handle->leases);
    HASH_ADD(hh, g_server_handles, id, sizeof(id), handle);

    return handle;
}

static struct server_client* find_client(IDTYPE vmid, bool create) {
    assert(locked(&g_server_lock));

    struct server_client* client = NULL;
    HASH_FIND(hh, g_server_clients, &vmid, sizeof(vmid), client);
    if (client)
        return client;

    if (!create)
        return NULL;

    if (!(client = malloc(sizeof(*client))))
        return NULL;

    client->vmid = vmid;
    INIT_LISTP(&client->leases);
    HASH_ADD(hh, g_server_clients, vmid, sizeof(vmid), client);

    return client;
}

static struct server_lease* find_lease(struct server_handle* handle, IDTYPE vmid, bool create) {
    assert(locked(&g_server_lock));

    struct server_lease* lease;
    LISTP_FOR_EACH_ENTRY(lease, &handle->leases, list_handle) {
        if (lease->client->vmid == vmid)
            return lease;
    }

    if (!create)
        return NULL;

    if (!(lease = malloc(sizeof(*lease))))
        return NULL;

    struct server_client* client = find_client(vmid, /*create=*/true);
    if (!client) {
        free(lease);
        return NULL;
    }

    lease->client = client;
    lease->cur_state = SYNC_STATE_INVALID;
    lease->client_req_state = SYNC_STATE_NONE;
    lease->server_req_state = SYNC_STATE_NONE;

    LISTP_ADD_TAIL(lease, &handle->leases, list_handle);
    LISTP_ADD_TAIL(lease, &client->leases, list_client);

    return lease;
}

static int send_confirm_upgrade(struct server_handle* handle, IDTYPE vmid, int state) {
    assert(locked(&g_server_lock));

    return ipc_sync_server_send(vmid, IPC_MSG_SYNC_CONFIRM_UPGRADE, handle->id, state,
                                handle->data_size, handle->data);
}

static int send_request_downgrade(struct server_handle* handle, IDTYPE vmid, int state) {
    assert(locked(&g_server_lock));

    return ipc_sync_server_send(vmid, IPC_MSG_SYNC_REQUEST_DOWNGRADE, handle->id, state,
                                /*data_size=*/0, /*data=*/NULL);
}

static int send_confirm_close(struct server_handle* handle, IDTYPE vmid) {
    assert(locked(&g_server_lock));

    return ipc_sync_server_send(vmid, IPC_MSG_SYNC_CONFIRM_CLOSE, handle->id, SYNC_STATE_INVALID,
                                /*data_size=*/0, /*data=*/NULL);
}

/* Process handle information after state change */
static int process_handle(struct server_handle* handle) {
    assert(locked(&g_server_lock));

    unsigned int n_shared = 0, n_exclusive = 0;
    unsigned int want_shared = 0, want_exclusive = 0;
    struct server_lease* lease;
    int ret;

    LISTP_FOR_EACH_ENTRY(lease, &handle->leases, list_handle) {
        if (lease->cur_state == SYNC_STATE_SHARED)
            n_shared++;
        if (lease->cur_state == SYNC_STATE_EXCLUSIVE)
            n_exclusive++;
        if (lease->client_req_state == SYNC_STATE_SHARED)
            want_shared++;
        if (lease->client_req_state == SYNC_STATE_EXCLUSIVE)
            want_exclusive++;
    }

    /* Fulfill upgrade requests, if possible right now */

    LISTP_FOR_EACH_ENTRY(lease, &handle->leases, list_handle) {
        if (lease->client_req_state == SYNC_STATE_SHARED && n_exclusive == 0) {
            /* Upgrade from INVALID to SHARED */
            assert(lease->cur_state == SYNC_STATE_INVALID);
            if ((ret = send_confirm_upgrade(handle, lease->client->vmid, SYNC_STATE_SHARED)) < 0)
                return ret;

            lease->cur_state = SYNC_STATE_SHARED;
            lease->client_req_state = SYNC_STATE_NONE;
            want_shared--;
            n_shared++;
        } else if (lease->client_req_state == SYNC_STATE_EXCLUSIVE && n_exclusive == 0
                   && n_shared == 0) {
            /* Upgrade from INVALID to EXCLUSIVE */
            assert(lease->cur_state == SYNC_STATE_INVALID);
            if ((ret = send_confirm_upgrade(handle, lease->client->vmid, SYNC_STATE_EXCLUSIVE)) < 0)
                return ret;

            lease->cur_state = SYNC_STATE_EXCLUSIVE;
            lease->client_req_state = SYNC_STATE_NONE;
            want_exclusive--;
            n_exclusive++;
        } else if (lease->client_req_state == SYNC_STATE_EXCLUSIVE && n_exclusive == 0
                   && n_shared == 1 && lease->cur_state == SYNC_STATE_SHARED) {
            /* Upgrade from SHARED to EXCLUSIVE */
            if ((ret = send_confirm_upgrade(handle, lease->client->vmid, SYNC_STATE_EXCLUSIVE)) < 0)
                return ret;

            lease->cur_state = SYNC_STATE_EXCLUSIVE;
            lease->client_req_state = SYNC_STATE_NONE;
            want_exclusive--;
            n_exclusive++;
            n_shared--;
        }
    }

    /* Issue downgrade requests, if necessary */

    if (want_exclusive) {
        /* Some clients wait for EXCLUSIVE, try to downgrade SHARED/EXCLUSIVE to INVALID */
        LISTP_FOR_EACH_ENTRY(lease, &handle->leases, list_handle) {
            if ((lease->cur_state == SYNC_STATE_SHARED || lease->cur_state == SYNC_STATE_EXCLUSIVE)
                    && lease->server_req_state != SYNC_STATE_INVALID) {
                if ((ret = send_request_downgrade(handle, lease->client->vmid,
                                                  SYNC_STATE_INVALID)) < 0)
                    return ret;
                lease->server_req_state = SYNC_STATE_INVALID;
            }
        }
    } else if (want_shared) {
        /* Some clients wait for SHARED, try to downgrade EXCLUSIVE to SHARED */
        LISTP_FOR_EACH_ENTRY(lease, &handle->leases, list_handle) {
            if (lease->cur_state == SYNC_STATE_EXCLUSIVE
                    && lease->server_req_state != SYNC_STATE_SHARED
                    && lease->server_req_state != SYNC_STATE_INVALID) {
                if ((ret = send_request_downgrade(handle, lease->client->vmid,
                                                  SYNC_STATE_SHARED)) < 0)
                    return ret;
                lease->server_req_state = SYNC_STATE_SHARED;
            }
        }
    }

    return 0;
}

static void do_request_upgrade(IDTYPE vmid, uint64_t id, int state) {
    assert(state == SYNC_STATE_SHARED || state == SYNC_STATE_EXCLUSIVE);

    lock(&g_server_lock);

    struct server_handle* handle;
    if (!(handle = find_handle(id, /*create=*/true)))
        FATAL("Cannot create a new handle\n");

    struct server_lease* lease;
    if ((!(lease = find_lease(handle, vmid, /*create=*/true))))
        FATAL("Cannot create a new handle client\n");

    assert(lease->cur_state < state);
    lease->client_req_state = state;

    /* Move the client to the end of the list, so that new requests are handled in FIFO
     * order. */
    if (LISTP_NEXT_ENTRY(lease, &handle->leases, list_handle) != NULL) {
        LISTP_DEL(lease, &handle->leases, list_handle);
        LISTP_ADD_TAIL(lease, &handle->leases, list_handle);
    }

    int ret;
    if ((ret = process_handle(handle)) < 0)
        FATAL("Error messaging clients: %d\n", ret);

    unlock(&g_server_lock);
}

static void update_handle_data(struct server_handle* handle, size_t data_size, void* data) {
    assert(locked(&g_server_lock));

    if (data_size != handle->data_size) {
        free(handle->data);
        handle->data = NULL;
        handle->data_size = data_size;
        if (handle->data_size > 0) {
            if (!(handle->data = malloc(handle->data_size)))
                FATAL("Cannot allocate data for handle\n");
        }
    }
    if (data_size > 0)
        memcpy(handle->data, data, data_size);
}

static void do_confirm_downgrade(IDTYPE vmid, uint64_t id, int state, size_t data_size,
                                 void* data) {
    assert(state == SYNC_STATE_INVALID || state == SYNC_STATE_SHARED);

    lock(&g_server_lock);

    struct server_handle* handle;
    if (!(handle = find_handle(id, /*create=*/true)))
        FATAL("Cannot create a new handle\n");

    struct server_lease* lease;
    if ((!(lease = find_lease(handle, vmid, /*create=*/true))))
        FATAL("Cannot create a new handle client\n");

    assert(lease->cur_state > state);

    if (lease->cur_state == SYNC_STATE_EXCLUSIVE) {
        update_handle_data(handle, data_size, data);
    } else {
        assert(data_size == 0);
    }

    lease->cur_state = state;
    if (state <= lease->server_req_state)
        lease->server_req_state = SYNC_STATE_NONE;

    int ret;
    if ((ret = process_handle(handle)) < 0)
        FATAL("Error messaging clients: %d\n", ret);

    unlock(&g_server_lock);
}

static void do_request_close(IDTYPE vmid, uint64_t id, int cur_state, size_t data_size,
                             void* data) {
    assert(cur_state == SYNC_STATE_INVALID || cur_state == SYNC_STATE_SHARED
           || cur_state == SYNC_STATE_EXCLUSIVE);
    lock(&g_server_lock);

    struct server_handle* handle;
    if (!(handle = find_handle(id, /*create=*/false)))
        FATAL("REQUEST_CLOSE for unknown handle\n");

    struct server_lease* lease;
    if ((!(lease = find_lease(handle, vmid, /*create=*/false))))
        FATAL("REQUEST_CLOSE for unknown client\n");

    struct server_client* client = lease->client;

    if (cur_state == SYNC_STATE_EXCLUSIVE) {
        update_handle_data(handle, data_size, data);
    } else {
        assert(data_size == 0);
    }

    if (send_confirm_close(handle, client->vmid) < 0)
        FATAL("sending CONFIRM_CLOSE\n");

    LISTP_DEL(lease, &client->leases, list_client);
    LISTP_DEL(lease, &handle->leases, list_handle);
    free(lease);

    if (LISTP_EMPTY(&client->leases)) {
        log_trace("sync server: deleting unused client: %d", client->vmid);
        HASH_DELETE(hh, g_server_clients, client);
        free(client);
    }

    if (LISTP_EMPTY(&handle->leases)) {
        log_trace("sync server: deleting unused handle: 0x%lx", handle->id);
        HASH_DELETE(hh, g_server_handles, handle);
        free(handle->data);
        free(handle);
    } else {
        int ret;
        if ((ret = process_handle(handle)) < 0)
            FATAL("Error messaging clients: %d\n", ret);
    }

    unlock(&g_server_lock);
}


void sync_server_message_callback(IDTYPE src, int code, uint64_t id, int state,
                                  size_t data_size, void* data) {
    switch (code) {
        case IPC_MSG_SYNC_REQUEST_UPGRADE:
            assert(data_size == 0);
            do_request_upgrade(src, id, state);
            break;
        case IPC_MSG_SYNC_CONFIRM_DOWNGRADE:
            do_confirm_downgrade(src, id, state, data_size, data);
            break;
        case IPC_MSG_SYNC_REQUEST_CLOSE:
            do_request_close(src, id, state, data_size, data);
            break;
        default:
            FATAL("unknown message: %d\n", code);
    }
}

/*
 * Check on client disconnect if all handles of that client have been closed.
 *
 * In principle, we could try to clean up after a client exiting. However, a disconnect without
 * cleanup probably means unclean Gramine exit (host SIGKILL, or fatal error), and in the case of
 * EXCLUSIVE handles, continuing will result in data loss.
 */
void sync_server_disconnect_callback(IDTYPE src) {
    lock(&g_server_lock);

    struct server_client* client = find_client(src, /*create=*/false);
    if (client) {
        assert(!LISTP_EMPTY(&client->leases));
        FATAL("Client %d disconnected without closing handles\n", src);
    }

    unlock(&g_server_lock);
}
