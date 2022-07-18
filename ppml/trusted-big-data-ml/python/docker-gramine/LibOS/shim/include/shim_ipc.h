/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University
 * Copyright (C) 2021 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

#ifndef SHIM_IPC_H_
#define SHIM_IPC_H_

#include <stdint.h>

#include "avl_tree.h"
#include "pal.h"
#include "shim_defs.h"
#include "shim_handle.h"
#include "shim_internal.h"
#include "shim_thread.h"
#include "shim_types.h"

enum {
    IPC_MSG_RESP = 0,
    IPC_MSG_GET_NEW_VMID,       /*!< Request new VMID. */
    IPC_MSG_CHILDEXIT,          /*!< Child exit/death information. */
    IPC_MSG_ALLOC_ID_RANGE,     /*!< Request new IDs range. */
    IPC_MSG_RELEASE_ID_RANGE,   /*!< Release IDs range. */
    IPC_MSG_CHANGE_ID_OWNER,    /*!< Change the owner of an ID. */
    IPC_MSG_GET_ID_OWNER,       /*!< Find the owner of an ID. */
    IPC_MSG_PID_KILL,
    IPC_MSG_PID_GETMETA,
    IPC_MSG_SYNC_REQUEST_UPGRADE,
    IPC_MSG_SYNC_REQUEST_DOWNGRADE,
    IPC_MSG_SYNC_REQUEST_CLOSE,
    IPC_MSG_SYNC_CONFIRM_UPGRADE,
    IPC_MSG_SYNC_CONFIRM_DOWNGRADE,
    IPC_MSG_SYNC_CONFIRM_CLOSE,
    IPC_MSG_POSIX_LOCK_SET,
    IPC_MSG_POSIX_LOCK_GET,
    IPC_MSG_POSIX_LOCK_CLEAR_PID,
    IPC_MSG_CODE_BOUND,
};

enum kill_type { KILL_THREAD, KILL_PROCESS, KILL_PGROUP, KILL_ALL };

enum pid_meta_code { PID_META_CRED, PID_META_EXEC, PID_META_CWD, PID_META_ROOT };

#define STARTING_VMID 1

struct shim_ipc_ids {
    IDTYPE self_vmid;
    IDTYPE parent_vmid;
    IDTYPE leader_vmid;
};

extern struct shim_ipc_ids g_process_ipc_ids;

int init_ipc(void);
int init_ipc_ids(void);

/*!
 * \brief Initialize the IPC worker thread.
 */
int init_ipc_worker(void);

/*!
 * \brief Terminate the IPC worker thread.
 */
void terminate_ipc_worker(void);

/*!
 * \brief Establish a one-way IPC connection to another process.
 *
 * \param dest  VMID of the destination process to connect to.
 */
int connect_to_process(IDTYPE dest);

/*!
 * \brief Remove an outgoing IPC connection.
 *
 * \param dest  VMID of the destination process.
 *
 * If there is no outgoing connection to \p dest, does nothing. If any thread waits for a response
 * to a message sent to \p dest, it is woken up and notified about the disconnect.
 */
void remove_outgoing_ipc_connection(IDTYPE dest);

struct ipc_msg_header {
    size_t size;
    uint64_t seq;
    unsigned char code;
} __attribute__((packed));

struct shim_ipc_msg {
    struct ipc_msg_header header;
    char data[];
} __attribute__((packed));

static inline size_t get_ipc_msg_size(size_t payload) {
    return sizeof(struct shim_ipc_msg) + payload;
}

void init_ipc_msg(struct shim_ipc_msg* msg, unsigned char code, size_t size);
void init_ipc_response(struct shim_ipc_msg* msg, uint64_t seq, size_t size);

/*!
 * \brief Send an IPC message.
 *
 * \param dest  VMID of the destination process.
 * \param msg   Message to send.
 */
int ipc_send_message(IDTYPE dest, struct shim_ipc_msg* msg);

/*!
 * \brief Send an IPC message and wait for a response.
 *
 * \param      dest  VMID of the destination process.
 * \param      msg   Message to send.
 * \param[out] resp  Upon successful return contains a pointer to the response.
 *
 * Send an IPC message to the \p dest process and wait for a response. An unique number is assigned
 * before sending the message and this thread will wait for a response IPC message, which contains
 * the same sequence number. If this function succeeds, \p resp will contain pointer to the response
 * data, which should be freed using `free` function. If \p resp is NULL, the response will be
 * discarded, but still awaited for.
 */
int ipc_send_msg_and_get_response(IDTYPE dest, struct shim_ipc_msg* msg, void** resp);

/*!
 * \brief Broadcast an IPC message.
 *
 * \param msg           Message to send.
 * \param exclude_vmid  VMID of process to be excluded.
 *
 * Send an IPC message \p msg to all known (connected) processes except for \p exclude_vmid.
 */
int ipc_broadcast(struct shim_ipc_msg* msg, IDTYPE exclude_vmid);

/*!
 * \brief Handle a response to a previously sent message.
 *
 * \param src   ID of sender.
 * \param data  Body of the response.
 * \param seq   Sequence number of the original message.
 *
 * Searches for a thread waiting for a response to a message previously sent to \p src with
 * the sequence number \p seq. If such thread is found, it is woken up and \p data is passed to it
 * (returned in `resp` argument of #ipc_send_msg_and_get_response).
 * This function always takes the ownership of \p data, the caller of this function should never
 * free it!
 */
int ipc_response_callback(IDTYPE src, void* data, uint64_t seq);

/*!
 * \brief Get a new VMID.
 *
 * \param[out] vmid  Contains the new VMID.
 */
int ipc_get_new_vmid(IDTYPE* vmid);
int ipc_get_new_vmid_callback(IDTYPE src, void* data, uint64_t seq);

struct shim_ipc_cld_exit {
    IDTYPE ppid, pid;
    IDTYPE uid;
    unsigned int exitcode;
    unsigned int term_signal;
} __attribute__((packed));

int ipc_cld_exit_send(unsigned int exitcode, unsigned int term_signal);
int ipc_cld_exit_callback(IDTYPE src, void* data, uint64_t seq);
void ipc_child_disconnect_callback(IDTYPE vmid);

#define MAX_RANGE_SIZE 0x20

/*!
 * \brief Request a new ID range from the IPC leader.
 *
 * \param[out] out_start  Start of the new ID range.
 * \param[out] out_end    End of the new ID range.
 *
 * Sender becomes the owner of the returned ID range.
 */
int ipc_alloc_id_range(IDTYPE* out_start, IDTYPE* out_end);
int ipc_alloc_id_range_callback(IDTYPE src, void* data, uint64_t seq);

/*!
 * \brief Release a previously allocated ID range.
 *
 * \param start  Start of the ID range.
 * \param end    End of the ID range.
 *
 * \p start and \p end must denote a full range (for details check #ipc_change_id_owner).
 */
int ipc_release_id_range(IDTYPE start, IDTYPE end);
int ipc_release_id_range_callback(IDTYPE src, void* data, uint64_t seq);

/*!
 * \brief Change owner of an ID.
 *
 * \param id         ID to change the ownership of.
 * \param new_owner  New owner of \p id.
 *
 * This operation effectively splits an existing ID range. Each (if any) of the range parts must be
 * later on freed separately. Example:
 * - process1 owns range 1..10
 * - `ipc_change_id_owner(id=5, new_owner=process2)`
 * - now process1 owns ranges 1..4 and 6..10, process2 owns 5..5
 * - each of these ranges must be freed separately, e.g.
 *   `ipc_release_id_range(5, 5); ipc_release_id_range(1, 4); ipc_release_id_range(6, 10);`
 *   is ok to do, but `ipc_release_id_range(5, 10);` is not.
 * Theoretically speaking any process can free any range (as long as each range is freed only once),
 * but in the current implementation a process frees only ranges it owns.
 */
int ipc_change_id_owner(IDTYPE id, IDTYPE new_owner);
int ipc_change_id_owner_callback(IDTYPE src, void* data, uint64_t seq);

/*!
 * \brief Find the owner of a given id.
 *
 * \param      id         ID to find the owner of.
 * \param[out] out_owner  Contains VMID of the process owning \p id.
 *
 * If nobody owns \p id then `0` is returned in \p out_owner.
 */
int ipc_get_id_owner(IDTYPE id, IDTYPE* out_owner);
int ipc_get_id_owner_callback(IDTYPE src, void* data, uint64_t seq);

struct shim_ipc_pid_kill {
    IDTYPE sender;
    IDTYPE pid;
    IDTYPE id;
    int signum;
    enum kill_type type;
};

int ipc_kill_process(IDTYPE sender, IDTYPE target, int sig);
int ipc_kill_thread(IDTYPE sender, IDTYPE dest_pid, IDTYPE target, int sig);
int ipc_kill_pgroup(IDTYPE sender, IDTYPE pgid, int sig);
int ipc_kill_all(IDTYPE sender, int sig);
int ipc_pid_kill_callback(IDTYPE src, void* data, uint64_t seq);

/* PID_GETMETA: get metadata of certain pid */
struct shim_ipc_pid_getmeta {
    IDTYPE pid;
    enum pid_meta_code code;
} __attribute__((packed));

/* PID_RETMETA: return metadata of certain pid */
struct shim_ipc_pid_retmeta {
    size_t datasize;
    int ret_val;
    char data[];
} __attribute__((packed));

int ipc_pid_getmeta(IDTYPE pid, enum pid_meta_code code, struct shim_ipc_pid_retmeta** data);
int ipc_pid_getmeta_callback(IDTYPE src, void* data, uint64_t seq);

/* SYNC_REQUEST_*, SYNC_CONFIRM_ */
struct shim_ipc_sync {
    uint64_t id;
    size_t data_size;
    int state;
    unsigned char data[];
};

int ipc_sync_client_send(int code, uint64_t id, int state, size_t data_size, void* data);
int ipc_sync_server_send(IDTYPE dest, int code, uint64_t id, int state, size_t data_size,
                         void* data);
int ipc_sync_request_upgrade_callback(IDTYPE src, void* data, unsigned long seq);
int ipc_sync_request_downgrade_callback(IDTYPE src, void* data, unsigned long seq);
int ipc_sync_request_close_callback(IDTYPE src, void* data, unsigned long seq);
int ipc_sync_confirm_upgrade_callback(IDTYPE src, void* data, unsigned long seq);
int ipc_sync_confirm_downgrade_callback(IDTYPE src, void* data, unsigned long seq);
int ipc_sync_confirm_close_callback(IDTYPE src, void* data, unsigned long seq);

/*
 * POSIX_LOCK_SET: `struct shim_ipc_posix_lock` -> `int`
 * POSIX_LOCK_GET: `struct shim_ipc_posix_lock` -> `struct shim_ipc_posix_lock_resp`
 * POSIX_LOCK_CLEAR_PID: `IDTYPE` -> `int`
 */

struct shim_ipc_posix_lock {
    /* see `struct posix_lock` in `shim_fs_lock.h` */
    int type;
    uint64_t start;
    uint64_t end;
    IDTYPE pid;

    bool wait;
    char path[]; /* null-terminated */
};

struct shim_ipc_posix_lock_resp {
    int result;

    /* see `struct posix_lock` in `shim_fs_lock.h` */
    int type;
    uint64_t start;
    uint64_t end;
    IDTYPE pid;
};

struct posix_lock;

int ipc_posix_lock_set(const char* path, struct posix_lock* pl, bool wait);
int ipc_posix_lock_set_send_response(IDTYPE vmid, unsigned long seq, int result);
int ipc_posix_lock_get(const char* path, struct posix_lock* pl, struct posix_lock* out_pl);
int ipc_posix_lock_clear_pid(IDTYPE pid);
int ipc_posix_lock_set_callback(IDTYPE src, void* data, unsigned long seq);
int ipc_posix_lock_get_callback(IDTYPE src, void* data, unsigned long seq);
int ipc_posix_lock_clear_pid_callback(IDTYPE src, void* data, unsigned long seq);

#endif /* SHIM_IPC_H_ */
