/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

#include <linux/fcntl.h>

#include "shim_fs.h"
#include "shim_fs_lock.h"
#include "shim_ipc.h"
#include "shim_lock.h"

/*
 * Global lock for the whole subsystem. Protects access to `g_fs_lock_list`, and also to dentry
 * fields (`fs_lock` and `maybe_has_fs_locks`).
 */
static struct shim_lock g_fs_lock_lock;

/*
 * Describes a pending request for a POSIX lock. After processing the request, the object is
 * removed, and a possible waiter is notified (see below).
 *
 * If the request is initiated by another process over IPC, `notify.vmid` and `notify.seq` should be
 * set to parameters of IPC message. After processing the request, IPC response will be sent.
 *
 * If the request is initiated by process leader, `notify.vmid` should be set to 0, and
 * `notify.event` should be set to an event handle. After processing the request, the event will be
 * triggered, and `*notify.result` will be set to the result.
 */
DEFINE_LISTP(posix_lock_request);
DEFINE_LIST(posix_lock_request);
struct posix_lock_request {
    struct posix_lock pl;

    struct {
        IDTYPE vmid;
        unsigned int seq;

        /* Note that `event` and `result` are owned by the side making the request, and outlive this
         * object (we delete it as soon as the request is processed). */
        PAL_HANDLE event;
        int* result;
    } notify;

    LIST_TYPE(posix_lock_request) list;
};

/* Describes file lock details for a given dentry. Currently holds only POSIX locks. */
DEFINE_LISTP(fs_lock);
DEFINE_LIST(fs_lock);
struct fs_lock {
    struct shim_dentry* dent;

    /* POSIX locks, sorted by PID and then by start position (so that we are able to merge and split
     * locks). The ranges do not overlap within a given PID. */
    LISTP_TYPE(posix_lock) posix_locks;

    /* Pending requests. */
    LISTP_TYPE(posix_lock_request) posix_lock_requests;

    /* List node, for `g_fs_lock_list`. */
    LIST_TYPE(fs_lock) list;
};

/* Global list of `fs_lock` objects. Used for cleanup. */
static LISTP_TYPE(fs_lock) g_fs_lock_list = LISTP_INIT;

int init_fs_lock(void) {
    if (g_process_ipc_ids.leader_vmid)
        return 0;

    return create_lock(&g_fs_lock_lock);
}

static int find_fs_lock(struct shim_dentry* dent, bool create, struct fs_lock** out_fs_lock) {
    assert(locked(&g_fs_lock_lock));
    if (!dent->fs_lock && create) {
        struct fs_lock* fs_lock = malloc(sizeof(*fs_lock));
        if (!fs_lock)
            return -ENOMEM;
        fs_lock->dent = dent;
        get_dentry(dent);
        INIT_LISTP(&fs_lock->posix_locks);
        INIT_LISTP(&fs_lock->posix_lock_requests);
        dent->fs_lock = fs_lock;

        LISTP_ADD(fs_lock, &g_fs_lock_list, list);
    }
    *out_fs_lock = dent->fs_lock;
    return 0;
}

static int posix_lock_dump_write_all(const char* str, size_t size, void* arg) {
    __UNUSED(arg);
    log_always("posix_lock: %.*s", (int)size, str);
    return 0;
}

/* Log current locks for a file, for debugging purposes. */
static void posix_lock_dump(struct fs_lock* fs_lock) {
    assert(locked(&g_fs_lock_lock));
    struct print_buf buf = INIT_PRINT_BUF(&posix_lock_dump_write_all);
    IDTYPE pid = 0;

    struct posix_lock* pl;
    LISTP_FOR_EACH_ENTRY(pl, &fs_lock->posix_locks, list) {
        if (pl->pid != pid) {
            if (pid != 0)
                buf_flush(&buf);
            pid = pl->pid;
            buf_printf(&buf, "%d:", pid);
        }

        char c;
        switch (pl->type) {
            case F_RDLCK: c = 'r'; break;
            case F_WRLCK: c = 'w'; break;
            default: c = '?'; break;
        }
        if (pl->end == FS_LOCK_EOF) {
            buf_printf(&buf, " %c[%lu..end]", c, pl->start);
        } else {
            buf_printf(&buf, " %c[%lu..%lu]", c, pl->start, pl->end);
        }
    }
    if (LISTP_EMPTY(&fs_lock->posix_locks)) {
        buf_printf(&buf, "no locks");
    }
    buf_flush(&buf);
}

/* Removes `fs_lock` if it's not necessary (i.e. no locks are held or requested for a file). */
static void fs_lock_gc(struct fs_lock* fs_lock) {
    assert(locked(&g_fs_lock_lock));
    if (g_log_level >= LOG_LEVEL_TRACE)
        posix_lock_dump(fs_lock);
    if (LISTP_EMPTY(&fs_lock->posix_locks) && LISTP_EMPTY(&fs_lock->posix_lock_requests)) {
        struct shim_dentry* dent = fs_lock->dent;
        dent->fs_lock = NULL;

        LISTP_DEL(fs_lock, &g_fs_lock_list, list);

        put_dentry(dent);
        free(fs_lock);
    }
}

/*
 * Find first lock that conflicts with `pl`. Two locks conflict if they have different PIDs, their
 * ranges overlap, and at least one of them is a write lock.
 */
static struct posix_lock* posix_lock_find_conflict(struct fs_lock* fs_lock, struct posix_lock* pl) {
    assert(locked(&g_fs_lock_lock));
    assert(pl->type != F_UNLCK);

    struct posix_lock* cur;
    LISTP_FOR_EACH_ENTRY(cur, &fs_lock->posix_locks, list) {
        if (cur->pid != pl->pid && pl->start <= cur->end && cur->start <= pl->end
               && (cur->type == F_WRLCK || pl->type == F_WRLCK))
            return cur;
    }
    return NULL;
}

/*
 * Add a new lock request. Before releasing `g_fs_lock_lock`, the caller has to initialize the
 * `notify` part of the request (see `struct posix_lock_request` above).
 */
static int posix_lock_add_request(struct fs_lock* fs_lock, struct posix_lock* pl,
                                  struct posix_lock_request** out_req) {
    assert(locked(&g_fs_lock_lock));
    assert(pl->type != F_UNLCK);

    struct posix_lock_request* req = malloc(sizeof(*req));
    if (!req)
        return -ENOMEM;
    req->pl = *pl;
    LISTP_ADD(req, &fs_lock->posix_lock_requests, list);
    *out_req = req;
    return 0;
}

/*
 * Main part of `posix_lock_set`. Adds/removes a lock (depending on `pl->type`), assumes we already
 * verified there are no conflicts. Replaces existing locks for a given PID, and merges adjacent
 * locks if possible.
 *
 * See also Linux sources (`fs/locks.c`) for a similar implementation.
 */
static int _posix_lock_set(struct fs_lock* fs_lock, struct posix_lock* pl) {
    assert(locked(&g_fs_lock_lock));

    /* Preallocate new locks first, so that we don't fail after modifying something. */

    /* Lock to be added. Not necessary for F_UNLCK, because we're only removing existing locks. */
    struct posix_lock* new = NULL;
    if (pl->type != F_UNLCK) {
        new = malloc(sizeof(*new));
        if (!new)
            return -ENOMEM;
    }

    /* Extra lock that we might need when splitting existing one. */
    struct posix_lock* extra = malloc(sizeof(*extra));
    if (!extra) {
        free(new);
        return -ENOMEM;
    }

    /* Target range: we will be changing it when merging existing locks. */
    uint64_t start = pl->start;
    uint64_t end   = pl->end;

    /* `prev` will be set to the last lock before target range, so that we add the new lock just
     * after `prev`. */
    struct posix_lock* prev = NULL;

    struct posix_lock* cur;
    struct posix_lock* tmp;
    LISTP_FOR_EACH_ENTRY_SAFE(cur, tmp, &fs_lock->posix_locks, list) {
        if (cur->pid < pl->pid) {
            prev = cur;
            continue;
        }
        if (pl->pid < cur->pid) {
            break;
        }

        if (cur->type == pl->type) {
            /* Same lock type: we can possibly merge the locks. */

            if (start > 0 && cur->end < start - 1) {
                /* `cur` ends before target range begins, and is not even adjacent */
                prev = cur;
            } else if (end < FS_LOCK_EOF && end + 1 < cur->start) {
                /* `cur` begins after target range ends, and is not even adjacent - we're
                 * done */
                break;
            } else {
                /* `cur` is either adjacent to target range, or overlaps with it. Delete it, and
                 * expand the target range. */
                start = MIN(start, cur->start);
                end = MAX(end, cur->end);
                LISTP_DEL(cur, &fs_lock->posix_locks, list);
                free(cur);
            }
        } else {
            /* Different lock types: if they overlap, we delete the target range. */

            if (cur->end < start) {
                /* `cur` ends before target range begins */
                prev = cur;
            } else if (end < cur->start) {
                /* `cur` begins after target range ends - we're done */
                break;
            } else if (cur->start < start && cur->end <= end) {
                /*
                 * `cur` overlaps with beginning of target range. Shorten `cur`.
                 *
                 * cur:  =======
                 * tgt:    -------
                 *
                 * cur:  ==
                 */
                assert(start > 0);
                cur->end = start - 1;
                prev = cur;
            } else if (cur->start < start && cur->end > end) {
                /*
                 * The target range is inside `cur`. Split `cur` and finish.
                 *
                 * cur:    ========
                 * tgt:      ----
                 *
                 * cur:    ==
                 * extra:        ==
                 */

                /* We'll need `extra` only once, because we exit the loop afterwards. */
                assert(extra);

                assert(start > 0);
                extra->type = cur->type;
                extra->start = end + 1;
                extra->end = cur->end;
                extra->pid = cur->pid;
                cur->end = start - 1;
                LISTP_ADD_AFTER(extra, cur, &fs_lock->posix_locks, list);
                extra = NULL;
                /* We're done: the new lock, if any, will be added after `cur`. */
                prev = cur;
                break;
            } else if (start <= cur->start && cur->end <= end) {
                /*
                 * `cur` is completely covered by target range. Delete `cur`.
                 *
                 * cur:    ====
                 * tgt:  --------
                 */
                LISTP_DEL(cur, &fs_lock->posix_locks, list);
                free(cur);
            } else {
                /*
                 * `cur` overlaps with end of target range. Shorten `cur` and finish.
                 *
                 * cur:    ====
                 * tgt: -----
                 *
                 * cur:      ==
                 */
                assert(start <= cur->start && end < cur->end);
                assert(end < FS_LOCK_EOF);
                cur->start = end + 1;
                break;
            }
        }
    }

    if (new) {
        assert(pl->type != F_UNLCK);


        new->type = pl->type;
        new->start = start;
        new->end = end;
        new->pid = pl->pid;

#ifdef DEBUG
        /* Assert that list order is preserved */
        struct posix_lock* next = prev ? LISTP_NEXT_ENTRY(prev, &fs_lock->posix_locks, list)
            : LISTP_FIRST_ENTRY(&fs_lock->posix_locks, struct posix_lock, list);
        if (prev)
            assert(prev->pid < pl->pid || (prev->pid == pl->pid && prev->end < start));
        if (next)
            assert(pl->pid < next->pid || (pl->pid == next->pid && end < next->start));
#endif

        if (prev) {
            LISTP_ADD_AFTER(new, prev, &fs_lock->posix_locks, list);
        } else {
            LISTP_ADD(new, &fs_lock->posix_locks, list);
        }
    }

    if (extra)
        free(extra);
    return 0;
}

/*
 * Process pending requests. This function should be called after any modification to the list of
 * locks, since we might have unblocked a request.
 *
 * TODO: This is pretty inefficient, but perhaps good enough for now...
 */
static void posix_lock_process_requests(struct fs_lock* fs_lock) {
    assert(locked(&g_fs_lock_lock));

    bool changed;
    do {
        changed = false;

        struct posix_lock_request* req;
        struct posix_lock_request* tmp;
        LISTP_FOR_EACH_ENTRY_SAFE(req, tmp, &fs_lock->posix_lock_requests, list) {
            struct posix_lock* conflict = posix_lock_find_conflict(fs_lock, &req->pl);
            if (!conflict) {
                int result = _posix_lock_set(fs_lock, &req->pl);
                LISTP_DEL(req, &fs_lock->posix_lock_requests, list);

                /* Notify the waiter that we processed their request. Note that the result might
                 * still be a failure (-ENOMEM). */
                if (req->notify.vmid == 0) {
                    assert(req->notify.event);
                    assert(req->notify.result);
                    *req->notify.result = result;
                    DkEventSet(req->notify.event);
                } else {
                    assert(!req->notify.event);
                    assert(!req->notify.result);

                    int ret = ipc_posix_lock_set_send_response(req->notify.vmid, req->notify.seq,
                                                               result);
                    if (ret < 0) {
                        log_warning("posix lock: error sending result over IPC: %d", ret);
                    }
                }
                free(req);
                changed = true;
            }
        }
    } while (changed);
}

/* Add/remove a lock if possible. On conflict, returns -EAGAIN (if `wait` is false) or adds a new
 * request (if `wait` is true). */
static int posix_lock_set_or_add_request(struct shim_dentry* dent, struct posix_lock* pl, bool wait,
                                         struct posix_lock_request** out_req) {
    assert(locked(&g_fs_lock_lock));

    struct fs_lock* fs_lock = NULL;
    int ret = find_fs_lock(dent, /*create=*/pl->type != F_UNLCK, &fs_lock);
    if (ret < 0)
        goto out;
    if (!fs_lock) {
        assert(pl->type == F_UNLCK);
        /* Nothing to unlock. */
        return 0;
    }

    struct posix_lock* conflict = NULL;
    if (pl->type != F_UNLCK)
        conflict = posix_lock_find_conflict(fs_lock, pl);
    if (conflict) {
        if (!wait) {
            ret = -EAGAIN;
            goto out;
        }

        struct posix_lock_request* req;
        ret = posix_lock_add_request(fs_lock, pl, &req);
        if (ret < 0)
            goto out;

        *out_req = req;
    } else {
        ret = _posix_lock_set(fs_lock, pl);
        if (ret < 0)
            goto out;
        posix_lock_process_requests(fs_lock);
        *out_req = NULL;
    }
    ret = 0;
out:
    if (fs_lock)
        fs_lock_gc(fs_lock);
    return ret;
}

int posix_lock_set(struct shim_dentry* dent, struct posix_lock* pl, bool wait) {
    int ret;
    if (g_process_ipc_ids.leader_vmid) {
        /* In the IPC version, we use `dent->maybe_has_fs_locks` to short-circuit unlocking files
         * that we never locked. This is to prevent unnecessary IPC calls on a handle. */
        lock(&g_fs_lock_lock);
        if (pl->type == F_RDLCK || pl->type == F_WRLCK) {
            dent->maybe_has_fs_locks = true;
        } else if (!dent->maybe_has_fs_locks) {
            /* We know we're not holding any locks for the file */
            unlock(&g_fs_lock_lock);
            return 0;
        }
        unlock(&g_fs_lock_lock);

        char* path;
        ret = dentry_abs_path(dent, &path, /*size=*/NULL);
        if (ret < 0)
            return ret;

        ret = ipc_posix_lock_set(path, pl, wait);
        free(path);
        return ret;
    }

    lock(&g_fs_lock_lock);

    PAL_HANDLE event = NULL;
    struct posix_lock_request* req = NULL;
    ret = posix_lock_set_or_add_request(dent, pl, wait, &req);
    if (ret < 0)
        goto out;
    if (req) {
        /* `posix_lock_set_or_add_request` is allowed to add a request only if `wait` is true */
        assert(wait);

        int result;
        ret = DkEventCreate(&event, /*init_signaled=*/false, /*auto_clear=*/false);
        if (ret < 0)
            goto out;
        req->notify.vmid = 0;
        req->notify.seq = 0;
        req->notify.event = event;
        req->notify.result = &result;

        unlock(&g_fs_lock_lock);
        ret = object_wait_with_retry(event);
        lock(&g_fs_lock_lock);
        if (ret < 0)
            goto out;

        ret = result;
    } else {
        ret = 0;
    }
out:
    unlock(&g_fs_lock_lock);
    if (event)
        DkObjectClose(event);
    return ret;
}

int posix_lock_set_from_ipc(const char* path, struct posix_lock* pl, bool wait, IDTYPE vmid,
                            unsigned long seq) {
    assert(!g_process_ipc_ids.leader_vmid);

    struct shim_dentry* dent = NULL;
    struct posix_lock_request* req = NULL;

    lock(&g_dcache_lock);
    int ret = path_lookupat(g_dentry_root, path, LOOKUP_NO_FOLLOW, &dent);
    unlock(&g_dcache_lock);
    if (ret < 0) {
        log_warning("posix_lock_set_from_ipc: error on dentry lookup for %s: %d", path, ret);
        goto out;
    }

    lock(&g_fs_lock_lock);
    ret = posix_lock_set_or_add_request(dent, pl, wait, &req);
    unlock(&g_fs_lock_lock);
    if (ret < 0)
        goto out;

    if (req) {
        /* `posix_lock_set_or_add_request` is allowed to add a request only if `wait` is true */
        assert(wait);

        req->notify.vmid = vmid;
        req->notify.seq = seq;
        req->notify.event = NULL;
        req->notify.result = NULL;
    }
    ret = 0;
out:
    if (dent)
        put_dentry(dent);
    if (req) {
        /* We added a request, so response will be sent later. */
        return 0;
    }
    return ipc_posix_lock_set_send_response(vmid, seq, ret);
}

int posix_lock_get(struct shim_dentry* dent, struct posix_lock* pl, struct posix_lock* out_pl) {
    assert(pl->type != F_UNLCK);

    int ret;
    if (g_process_ipc_ids.leader_vmid) {
        char* path;
        ret = dentry_abs_path(dent, &path, /*size=*/NULL);
        if (ret < 0)
            return ret;

        ret = ipc_posix_lock_get(path, pl, out_pl);
        free(path);
        return ret;
    }

    lock(&g_fs_lock_lock);

    struct fs_lock* fs_lock = NULL;
    ret = find_fs_lock(dent, /*create=*/false, &fs_lock);
    if (ret < 0)
        goto out;

    struct posix_lock* conflict = NULL;
    if (fs_lock)
        conflict = posix_lock_find_conflict(fs_lock, pl);
    if (conflict) {
        out_pl->type = conflict->type;
        out_pl->start = conflict->start;
        out_pl->end = conflict->end;
        out_pl->pid = conflict->pid;
    } else {
        out_pl->type = F_UNLCK;
    }
    ret = 0;

out:
    if (fs_lock)
        fs_lock_gc(fs_lock);

    unlock(&g_fs_lock_lock);
    return ret;
}

int posix_lock_get_from_ipc(const char* path, struct posix_lock* pl, struct posix_lock* out_pl) {
    assert(!g_process_ipc_ids.leader_vmid);

    struct shim_dentry* dent = NULL;
    lock(&g_dcache_lock);
    int ret = path_lookupat(g_dentry_root, path, LOOKUP_NO_FOLLOW, &dent);
    unlock(&g_dcache_lock);
    if (ret < 0) {
        log_warning("posix_lock_get_from_ipc: error on dentry lookup for %s: %d", path, ret);
        return ret;
    }

    ret = posix_lock_get(dent, pl, out_pl);
    put_dentry(dent);
    return ret;
}

/* Removes all locks and lock requests for a given PID and dentry. */
static int posix_lock_clear_pid_from_dentry(struct shim_dentry* dent, IDTYPE pid) {
    assert(locked(&g_fs_lock_lock));

    struct fs_lock* fs_lock;
    int ret = find_fs_lock(dent, /*create=*/false, &fs_lock);
    if (ret < 0)
        return ret;
    if (!fs_lock) {
        /* Nothing to process. */
        return 0;
    }

    bool changed = false;

    struct posix_lock* pl;
    struct posix_lock* pl_tmp;
    LISTP_FOR_EACH_ENTRY_SAFE(pl, pl_tmp, &fs_lock->posix_locks, list) {
        if (pl->pid == pid) {
            LISTP_DEL(pl, &fs_lock->posix_locks, list);
            free(pl);
            changed = true;
        }
    }

    struct posix_lock_request* req;
    struct posix_lock_request* req_tmp;
    LISTP_FOR_EACH_ENTRY_SAFE(req, req_tmp, &fs_lock->posix_lock_requests, list) {
        if (req->pl.pid == pid) {
            assert(!req->notify.event);
            LISTP_DEL(req, &fs_lock->posix_lock_requests, list);
            free(req);
        }
    }

    if (changed) {
        posix_lock_process_requests(fs_lock);
        fs_lock_gc(fs_lock);
    }

    return 0;
}

int posix_lock_clear_pid(IDTYPE pid) {
    if (g_process_ipc_ids.leader_vmid) {
        return ipc_posix_lock_clear_pid(pid);
    }

    log_debug("clearing POSIX locks for pid %d", pid);

    int ret;

    struct fs_lock* fs_lock;
    struct fs_lock* fs_lock_tmp;

    lock(&g_fs_lock_lock);
    LISTP_FOR_EACH_ENTRY_SAFE(fs_lock, fs_lock_tmp, &g_fs_lock_list, list) {
        /* Note that the below call might end up deleting `fs_lock` */
        ret = posix_lock_clear_pid_from_dentry(fs_lock->dent, pid);
        if (ret < 0)
            goto out;
    }

    ret = 0;
out:
    unlock(&g_fs_lock_lock);
    return ret;
}
