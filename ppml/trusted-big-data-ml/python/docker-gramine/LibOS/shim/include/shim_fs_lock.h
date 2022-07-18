/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * File locks. Currently only POSIX locks are implemented.
 */

#ifndef SHIM_FS_LOCK_H_
#define SHIM_FS_LOCK_H_


#include <stdbool.h>

#include "list.h"
#include "shim_types.h"

#define FS_LOCK_EOF ((uint64_t)-1)

struct shim_dentry;

/* Initialize the file locking subsystem. */
int init_fs_lock(void);

/*
 * POSIX locks (also known as advisory record locks). See `man fcntl` for details.
 *
 * The current implementation works over IPC and handles all requests in the main process. It has
 * the following caveats:
 *
 * - Lock requests from other processes will always have the overhead of IPC round-trip, even if the
 *   lock is uncontested.
 * - The main process has to be able to look up the same file, so locking will not work for files in
 *   local-process-only filesystems (tmpfs).
 * - There is no deadlock detection (EDEADLK).
 * - The lock requests cannot be interrupted (EINTR).
 * - The locks work only on files that have a dentry (no pipes, sockets etc.)
 */

DEFINE_LISTP(posix_lock);
DEFINE_LIST(posix_lock);
struct posix_lock {
    /* Lock type: F_RDLCK, F_WRLCK, F_UNLCK */
    int type;

    /* First byte of range */
    uint64_t start;

    /* Last byte of range (use FS_LOCK_EOF for a range until end of file) */
    uint64_t end;

    /* PID of process taking the lock */
    IDTYPE pid;

    /* List node, used internally */
    LIST_TYPE(posix_lock) list;
};

/*!
 * \brief Set or remove a lock on a file.
 *
 * \param dent  The dentry for a file.
 * \param pl    Parameters of new lock.
 * \param wait  If true, will wait until a lock can be taken.
 *
 * This is the equivalent of `fnctl(F_SETLK/F_SETLKW)`.
 *
 * If `pl->type` is `F_UNLCK`, the function will remove any locks held by the given PID for the
 * given range. Removing a lock never waits.
 *
 * If `pl->type` is `F_RDLCK` or `F_WRLCK`, the function will create a new lock for the given PID
 * and range, replacing the existing locks held by the given PID for that range. If there are
 * conflicting locks, the function either waits (if `wait` is true), or fails with `-EAGAIN` (if
 * `wait` is false).
 */
int posix_lock_set(struct shim_dentry* dent, struct posix_lock* pl, bool wait);

/*!
 * \brief Check for conflicting locks on a file.
 *
 * \param      dent    The dentry for a file.
 * \param      pl      Parameters of new lock (type cannot be `F_UNLCK`).
 * \param[out] out_pl  On success, set to `F_UNLCK` or details of a conflicting lock.
 *
 * This is the equivalent of `fcntl(F_GETLK)`.
 *
 * The function checks if there are locks by other PIDs preventing the proposed lock from being
 * placed. If the lock could be placed, `out_pl->type` is set to `F_UNLCK`. Otherwise, `out_pl`
 * fields (`type`, `start, `end`, `pid`) are set to details of a conflicting lock.
 */
int posix_lock_get(struct shim_dentry* dent, struct posix_lock* pl, struct posix_lock* out_pl);

/* Removes all locks for a given PID. Should be called before process exit. */
int posix_lock_clear_pid(IDTYPE pid);

/*!
 * \brief Set or remove a lock on a file (IPC handler).
 *
 * \param path  Absolute path for a file.
 * \param pl    Parameters of new lock.
 * \param wait  If true, will postpone the response until a lock can be taken.
 * \param vmid  Target process for IPC response.
 * \param seq   Sequence number for IPC response.
 *
 * This is a version of `posix_lock_set` called from an IPC callback. This function is responsible
 * for either sending an IPC response immediately, or scheduling one for later (if `wait` is true
 * and the lock cannot be taken immediately).
 *
 * This function will only return a negative error code when failing to send a response. A failure
 * to add a lock (-EAGAIN, -ENOMEM etc.) will be sent in the response instead.
 */
int posix_lock_set_from_ipc(const char* path, struct posix_lock* pl, bool wait, IDTYPE vmid,
                            unsigned long seq);

/*!
 * \brief Check for conflicting locks on a file (IPC handler).
 *
 * \param      path    Absolute path for a file.
 * \param      pl      Parameters of new lock (type cannot be `F_UNLCK`).
 * \param[out] out_pl  On success, set to `F_UNLCK` or details of a conflicting lock.
 *
 * This is a version of `posix_lock_get` called from an IPC callback. The caller is responsible to
 * send the returned value and `out_pl` in an IPC response.
 */
int posix_lock_get_from_ipc(const char* path, struct posix_lock* pl, struct posix_lock* out_pl);

#endif /* SHIM_FS_LOCK_H_ */
