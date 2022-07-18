/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

/* This file contains code for management of ID ranges owned by the current process. */

#include "assert.h"
#include "avl_tree.h"
#include "log.h"
#include "shim_ipc.h"
#include "shim_lock.h"
#include "shim_types.h"
#include "shim_utils.h"

/* Represents a range of ids `[start; end]` (i.e. `end` is included). There is no representation of
 * an empty range, but it's not needed. */
struct id_range {
    struct avl_tree_node node;
    IDTYPE start;
    IDTYPE end;
    unsigned int taken_count;
    static_assert(MAX_RANGE_SIZE <= UINT_MAX, "`taken_count` is capped by `MAX_RANGE_SIZE`");
};

static bool id_range_cmp(struct avl_tree_node* _a, struct avl_tree_node* _b) {
    struct id_range* a = container_of(_a, struct id_range, node);
    struct id_range* b = container_of(_b, struct id_range, node);
    /*
     * This is equivalent to:
     * ```return a->start <= b->start;```
     * because overlapping ranges in one tree are disallowed, but it also enables easy lookups of
     * ranges overlapping any given point.
     */
    return a->start <= b->end;
}

/* These are IDs that are owned by this process. */
static struct id_range* g_last_range = NULL;
static struct avl_tree g_used_ranges_tree = { .cmp = id_range_cmp };
static IDTYPE g_last_used_id = 0;
static struct shim_lock g_ranges_lock;

int init_id_ranges(IDTYPE preload_tid) {
    if (!create_lock(&g_ranges_lock)) {
        return -ENOMEM;
    }

    if (!preload_tid) {
        return 0;
    }

    struct id_range* range = malloc(sizeof(*range));
    if (!range) {
        return -ENOMEM;
    }
    range->start = preload_tid;
    range->end = preload_tid;
    range->taken_count = 1;

    lock(&g_ranges_lock);
    avl_tree_insert(&g_used_ranges_tree, &range->node);
    unlock(&g_ranges_lock);
    return 0;
}

IDTYPE get_new_id(IDTYPE move_ownership_to) {
    IDTYPE ret_id = 0;
    lock(&g_ranges_lock);
    if (!g_last_range) {
        g_last_range = malloc(sizeof(*g_last_range));
        if (!g_last_range) {
            log_debug("OOM in %s:%d", __FILE__, __LINE__);
            goto out;
        }
        IDTYPE start;
        IDTYPE end;
        int ret = ipc_alloc_id_range(&start, &end);
        if (ret < 0) {
            log_debug("Failed to allocate new id range: %d", ret);
            free(g_last_range);
            g_last_range = NULL;
            goto out;
        }
        assert(start <= end);
        assert(end - start + 1 <= MAX_RANGE_SIZE);
        assert(start > 0);

        g_last_range->start = start;
        g_last_range->end = end;
        g_last_range->taken_count = 0;
        g_last_used_id = start - 1;
    }
    assert(g_last_used_id < g_last_range->end);
    assert(g_last_range->taken_count < g_last_range->end - g_last_range->start + 1);

    ret_id = ++g_last_used_id;
    g_last_range->taken_count++;

    if (move_ownership_to) {
        g_last_range->taken_count--;
        if (g_last_range->start == g_last_range->end) {
            assert(g_last_range->taken_count == 0);
            free(g_last_range);
            g_last_range = NULL;
        } else if (g_last_range->start == g_last_used_id) {
            g_last_range->start++;
            assert(g_last_range->taken_count == 0);
        } else if (g_last_range->end == g_last_used_id) {
            g_last_range->end--;
            avl_tree_insert(&g_used_ranges_tree, &g_last_range->node);
            g_last_range = NULL;
        } else {
            struct id_range* range = malloc(sizeof(*range));
            if (!range) {
                log_debug("OOM in %s:%d", __FILE__, __LINE__);
                g_last_used_id--;
                ret_id = 0;
                goto out;
            }
            assert(g_last_range->start < g_last_used_id && g_last_used_id < g_last_range->end);
            range->start = g_last_used_id + 1;
            range->end = g_last_range->end;
            range->taken_count = 0;
            g_last_range->end = g_last_used_id - 1;
            avl_tree_insert(&g_used_ranges_tree, &g_last_range->node);
            g_last_range = range;
        }
        if (ipc_change_id_owner(ret_id, move_ownership_to) < 0) {
            /* Good luck unwinding all of above operations. Better just kill everything. */
            log_error("Unrecoverable error in %s:%d", __FILE__, __LINE__);
            DkProcessExit(1);
        }
    } else {
        if (g_last_used_id == g_last_range->end) {
            avl_tree_insert(&g_used_ranges_tree, &g_last_range->node);
            g_last_range = NULL;
        }
    }

out:
    unlock(&g_ranges_lock);
    return ret_id;
}

void release_id(IDTYPE id) {
    lock(&g_ranges_lock);
    if (g_last_range && g_last_range->start <= id && id <= g_last_range->end) {
        assert(g_last_range->taken_count > 0);
        g_last_range->taken_count--;
    } else {
        struct id_range dummy = {
            .start = id,
            .end = id,
        };
        struct avl_tree_node* node = avl_tree_lower_bound(&g_used_ranges_tree, &dummy.node);
        if (!node) {
            log_error("Trying to release unknown ID!");
            BUG();
        }
        struct id_range* range = container_of(node, struct id_range, node);
        if (id < range->start || range->end < id) {
            log_error("Trying to release unknown ID!");
            BUG();
        }
        assert(range->taken_count > 0);
        range->taken_count--;
        if (range->taken_count == 0) {
            avl_tree_delete(&g_used_ranges_tree, &range->node);
            unlock(&g_ranges_lock);

            int ret = ipc_release_id_range(range->start, range->end);
            if (ret < 0) {
                /* TODO: this is a fatal error, unfortunately it can happen if the IPC leader exits
                 * without fully waiting for this process to end. For more information check
                 * "LibOS/shim/src/sys/shim_exit.c". Change to `log_error` + `die` after fixing. */
                log_warning("IPC pid release failed");
                DkProcessExit(1);
            }
            free(range);
            return;
        }
    }
    unlock(&g_ranges_lock);
}
