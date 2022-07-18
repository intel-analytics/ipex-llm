/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

/* This file contains code for management of global ID ranges. */

#include "api.h"
#include "assert.h"
#include "avl_tree.h"
#include "shim_ipc.h"
#include "shim_lock.h"
#include "shim_types.h"

/* Represents a range of ids `[start; end]` (i.e. `end` is included). There is no representation of
 * an empty range, but it's not needed. */
struct id_range {
    struct avl_tree_node node;
    IDTYPE start;
    IDTYPE end;
    IDTYPE owner;
};

struct ipc_id_range_msg {
    IDTYPE start;
    IDTYPE end;
};

struct ipc_id_owner_msg {
    IDTYPE id;
    IDTYPE owner;
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

/* These are ranges of all used IDs. This tree is only meaningful in IPC leader.
 * No two ranges in this tree shall overlap. */
static struct avl_tree g_id_owners_tree = { .cmp = id_range_cmp };
static struct shim_lock g_id_owners_tree_lock;
static IDTYPE g_last_id = 0;

int init_ipc_ids(void) {
    if (!create_lock(&g_id_owners_tree_lock)) {
        return -ENOMEM;
    }
    return 0;
}

/* If a free range was found, sets `*start` and `*end` and returns `true`, if nothing was found
 * returns `false`. If a range was returned, it is not larger than `MAX_RANGE_SIZE`. */
static bool _find_free_id_range(IDTYPE* start, IDTYPE* end) {
    assert(locked(&g_id_owners_tree_lock));
    static_assert(!IS_SIGNED(IDTYPE), "IDTYPE must be unsigned");
    IDTYPE next_id = g_last_id + 1 ?: 1;

    struct id_range dummy = {
        .start = next_id,
        .end = next_id,
    };
    struct avl_tree_node* node = avl_tree_lower_bound(&g_id_owners_tree, &dummy.node);
    while (node) {
        struct id_range* range = container_of(node, struct id_range, node);
        if (next_id < range->start) {
            /* `next_id` does not overlap any existing range. */
            *start = next_id;
            if (__builtin_add_overflow(next_id, MAX_RANGE_SIZE - 1, end)) {
                *end = IDTYPE_MAX;
            }
            *end = MIN(*end, range->start - 1);
            return true;
        }
        /* `next_id` overlaps `range`. */
        assert(next_id <= range->end);
        if (range->end == IDTYPE_MAX) {
            /* No ids available in range `[g_last_id + 1, IDTYPE_MAX]`. If wrapping is needed, set
             * `g_last_id` and call this function again. */
            return false;
        }
        next_id = range->end + 1;
        node = avl_tree_next(node);
    }
    /* There are no ids greater or equal to `next_id`. */
    *start = next_id;
    if (__builtin_add_overflow(next_id, MAX_RANGE_SIZE - 1, end)) {
        *end = IDTYPE_MAX;
    }
    return true;
}

static int alloc_id_range(IDTYPE owner, IDTYPE* start, IDTYPE* end) {
    assert(owner);
    struct id_range* new_range = malloc(sizeof(*new_range));
    if (!new_range) {
        return -ENOMEM;
    }

    lock(&g_id_owners_tree_lock);
    bool found = _find_free_id_range(start, end);
    if (!found) {
        /* No id found, try wrapping around. */
        g_last_id = 0;
        found = _find_free_id_range(start, end);
    }

    int ret = 0;
    if (found) {
        assert(*start && *end);
        new_range->start = *start;
        new_range->end = *end;
        new_range->owner = owner;
        avl_tree_insert(&g_id_owners_tree, &new_range->node);
        g_last_id = *end;
    } else {
        free(new_range);
        ret = -EAGAIN;
    }
    unlock(&g_id_owners_tree_lock);
    return ret;
}

static int change_id_owner(IDTYPE id, IDTYPE new_owner) {
    struct id_range* new_range1 = malloc(sizeof(*new_range1));
    if (!new_range1) {
        return -ENOMEM;
    }
    struct id_range* new_range2 = malloc(sizeof(*new_range1));
    if (!new_range2) {
        free(new_range1);
        return -ENOMEM;
    }
    new_range1->start = id;
    new_range1->end = id;
    new_range1->owner = new_owner;

    lock(&g_id_owners_tree_lock);
    struct id_range dummy = {
        .start = id,
        .end = id,
    };
    struct avl_tree_node* node = avl_tree_lower_bound(&g_id_owners_tree, &dummy.node);
    if (!node) {
        log_debug("ID %u unknown!", id);
        BUG();
    }
    struct id_range* range = container_of(node, struct id_range, node);
    if (id < range->start || range->end < id) {
        log_debug("ID %u unknown!", id);
        BUG();
    }

    /* These `range` modifications are in place since we know they won't change the position of
     * `range` inside `g_id_owners_tree`. Otherwise we would have to remove it, modify and then
     * re-add. */
    if (range->start == range->end) {
        assert(id == range->start);
        range->owner = new_owner;
    } else if (range->start == id) {
        range->start++;
        avl_tree_insert(&g_id_owners_tree, &new_range1->node);
        new_range1 = NULL;
    } else if (range->end == id) {
        range->end--;
        avl_tree_insert(&g_id_owners_tree, &new_range1->node);
        new_range1 = NULL;
    } else {
        assert(range->end - range->start + 1 >= 3);
        new_range2->start = id + 1;
        new_range2->end = range->end;
        new_range2->owner = range->owner;
        range->end = id - 1;
        avl_tree_insert(&g_id_owners_tree, &new_range1->node);
        avl_tree_insert(&g_id_owners_tree, &new_range2->node);
        new_range1 = NULL;
        new_range2 = NULL;
    }

    unlock(&g_id_owners_tree_lock);
    free(new_range1);
    free(new_range2);
    return 0;
}

static void release_id_range(IDTYPE start, IDTYPE end) {
    lock(&g_id_owners_tree_lock);
    struct id_range dummy = {
        .start = start,
        .end = end,
    };
    struct avl_tree_node* node = avl_tree_find(&g_id_owners_tree, &dummy.node);
    if (!node) {
        log_debug("Releasing invalid ID range!");
        BUG();
    }
    struct id_range* range = container_of(node, struct id_range, node);
    if (range->start != start || range->end != end) {
        BUG();
    }
    avl_tree_delete(&g_id_owners_tree, &range->node);

    unlock(&g_id_owners_tree_lock);
    free(range);
}

static IDTYPE find_id_owner(IDTYPE id) {
    IDTYPE owner = 0;

    struct id_range dummy = {
        .start = id,
        .end = id,
    };
    lock(&g_id_owners_tree_lock);
    struct avl_tree_node* node = avl_tree_lower_bound(&g_id_owners_tree, &dummy.node);
    if (!node) {
        goto out;
    }
    struct id_range* range = container_of(node, struct id_range, node);
    if (id < range->start || range->end < id) {
        goto out;
    }
    owner = range->owner;

out:
    unlock(&g_id_owners_tree_lock);
    return owner;
}

int ipc_alloc_id_range(IDTYPE* out_start, IDTYPE* out_end) {
    if (!g_process_ipc_ids.leader_vmid) {
        return alloc_id_range(g_process_ipc_ids.self_vmid, out_start, out_end);
    }

    size_t msg_size = get_ipc_msg_size(0);
    struct shim_ipc_msg* msg = malloc(msg_size);
    if (!msg) {
        return -ENOMEM;
    }
    init_ipc_msg(msg, IPC_MSG_ALLOC_ID_RANGE, msg_size);

    log_debug("%s: sending a request", __func__);

    void* resp = NULL;
    int ret = ipc_send_msg_and_get_response(g_process_ipc_ids.leader_vmid, msg, &resp);
    if (ret < 0) {
        goto out;
    }

    struct ipc_id_range_msg* range = resp;
    if (range->start && range->end) {
        *out_start = range->start;
        *out_end = range->end;
        ret = 0;
    } else {
        ret = -EAGAIN;
    }

    log_debug("%s: got a response: [%u..%u]", __func__, range->start, range->end);

out:
    free(resp);
    free(msg);
    return ret;
}

int ipc_alloc_id_range_callback(IDTYPE src, void* data, uint64_t seq) {
    __UNUSED(data);
    IDTYPE start = 0;
    IDTYPE end = 0;
    int ret = alloc_id_range(src, &start, &end);
    if (ret < 0) {
        start = 0;
        end = 0;
    }

    log_debug("%s: %d", __func__, ret);

    struct ipc_id_range_msg range = {
        .start = start,
        .end = end,
    };
    size_t msg_size = get_ipc_msg_size(sizeof(range));
    struct shim_ipc_msg* msg = __alloca(msg_size);
    init_ipc_response(msg, seq, msg_size);
    memcpy(&msg->data, &range, sizeof(range));

    return ipc_send_message(src, msg);
}

int ipc_release_id_range(IDTYPE start, IDTYPE end) {
    if (!g_process_ipc_ids.leader_vmid) {
        release_id_range(start, end);
        return 0;
    }

    struct ipc_id_range_msg range = {
        .start = start,
        .end = end,
    };
    size_t msg_size = get_ipc_msg_size(sizeof(range));
    struct shim_ipc_msg* msg = malloc(msg_size);
    if (!msg) {
        return -ENOMEM;
    }
    init_ipc_msg(msg, IPC_MSG_RELEASE_ID_RANGE, msg_size);
    memcpy(&msg->data, &range, sizeof(range));

    log_debug("%s: sending a request: [%u..%u]", __func__, start, end);

    int ret = ipc_send_message(g_process_ipc_ids.leader_vmid, msg);
    log_debug("%s: ipc_send_message: %d", __func__, ret);
    free(msg);
    return ret;
}

int ipc_release_id_range_callback(IDTYPE src, void* data, uint64_t seq) {
    __UNUSED(src);
    __UNUSED(seq);
    struct ipc_id_range_msg* range = data;
    release_id_range(range->start, range->end);
    log_debug("%s: release_id_range(%u..%u)", __func__, range->start, range->end);
    return 0;
}

int ipc_change_id_owner(IDTYPE id, IDTYPE new_owner) {
    if (!g_process_ipc_ids.leader_vmid) {
        return change_id_owner(id, new_owner);
    }

    struct ipc_id_owner_msg owner_msg = {
        .id = id,
        .owner = new_owner,
    };
    size_t msg_size = get_ipc_msg_size(sizeof(owner_msg));
    struct shim_ipc_msg* msg = malloc(msg_size);
    if (!msg) {
        return -ENOMEM;
    }
    init_ipc_msg(msg, IPC_MSG_CHANGE_ID_OWNER, msg_size);
    memcpy(&msg->data, &owner_msg, sizeof(owner_msg));

    log_debug("%s: sending a request (%u, %u)", __func__, id, new_owner);

    int ret = ipc_send_msg_and_get_response(g_process_ipc_ids.leader_vmid, msg, /*resp=*/NULL);
    log_debug("%s: ipc_send_msg_and_get_response: %d", __func__, ret);
    free(msg);
    return ret;
}

int ipc_change_id_owner_callback(IDTYPE src, void* data, uint64_t seq) {
    struct ipc_id_owner_msg* owner_msg = data;
    int ret = change_id_owner(owner_msg->id, owner_msg->owner);
    log_debug("%s: change_id_owner(%u, %u): %d", __func__, owner_msg->id, owner_msg->owner, ret);
    if (ret < 0) {
        return ret;
    }

    /* Respond with a dummy empty message. */
    size_t msg_size = get_ipc_msg_size(0);
    struct shim_ipc_msg* msg = __alloca(msg_size);
    init_ipc_response(msg, seq, msg_size);
    return ipc_send_message(src, msg);
}

int ipc_get_id_owner(IDTYPE id, IDTYPE* out_owner) {
    if (!g_process_ipc_ids.leader_vmid) {
        *out_owner = find_id_owner(id);
        return 0;
    }

    size_t msg_size = get_ipc_msg_size(sizeof(id));
    struct shim_ipc_msg* msg = malloc(msg_size);
    if (!msg) {
        return -ENOMEM;
    }
    init_ipc_msg(msg, IPC_MSG_GET_ID_OWNER, msg_size);
    memcpy(&msg->data, &id, sizeof(id));

    log_debug("%s: sending a request: %u", __func__, id);

    void* resp = NULL;
    int ret = ipc_send_msg_and_get_response(g_process_ipc_ids.leader_vmid, msg, &resp);
    if (ret < 0) {
        goto out;
    }

    *out_owner = *(IDTYPE*)resp;
    ret = 0;

    log_debug("%s: got a response: %u", __func__, *out_owner);

out:
    free(resp);
    free(msg);
    return ret;
}

int ipc_get_id_owner_callback(IDTYPE src, void* data, uint64_t seq) {
    IDTYPE* id = data;
    IDTYPE owner = find_id_owner(*id);
    log_debug("%s: find_id_owner(%u): %u", __func__, *id, owner);

    size_t msg_size = get_ipc_msg_size(sizeof(owner));
    struct shim_ipc_msg* msg = __alloca(msg_size);
    init_ipc_response(msg, seq, msg_size);
    memcpy(&msg->data, &owner, sizeof(owner));

    return ipc_send_message(src, msg);
}
