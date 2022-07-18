/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University
 * Copyright (C) 2019 Invisible Things Lab
 */

/*
 * "The futexes are also cursed."
 * "But they come in a choice of three flavours!"
 *                                  ~ the Linux kernel source
 */

/*
 * Current implementation is limited to one process i.e. threads calling futex syscall on the same
 * futex word must reside in the same process.
 * As a result we can distinguish futexes by their virtual address.
 */

#include <linux/futex.h>
#include <stdbool.h>
#include <stdint.h>

#include "api.h"
#include "assert.h"
#include "avl_tree.h"
#include "list.h"
#include "pal.h"
#include "shim_internal.h"
#include "shim_table.h"
#include "shim_thread.h"
#include "shim_types.h"
#include "shim_utils.h"
#include "spinlock.h"

struct shim_futex;
struct futex_waiter;

DEFINE_LIST(futex_waiter);
DEFINE_LISTP(futex_waiter);
struct futex_waiter {
    struct shim_thread* thread;
    uint32_t bitset;
    LIST_TYPE(futex_waiter) list;
    /* futex field is guarded by g_futex_tree_lock, do not use it without taking that lock first.
     * This is needed to ensure that a waiter knows what futex they were sleeping on, after they
     * wake-up (because they could have been requeued to another futex).*/
    struct shim_futex* futex;
};

struct shim_futex {
    uint32_t* uaddr;
    LISTP_TYPE(futex_waiter) waiters;
    struct avl_tree_node tree_node;
    bool in_tree;
    /* This lock guards every access to *uaddr (futex word value) and waiters (above).
     * Always take g_futex_tree_lock before taking this lock. */
    spinlock_t lock;
    REFTYPE _ref_count;
};

static bool futex_tree_cmp(struct avl_tree_node* node_a, struct avl_tree_node* node_b) {
    struct shim_futex* a = container_of(node_a, struct shim_futex, tree_node);
    struct shim_futex* b = container_of(node_b, struct shim_futex, tree_node);

    return (uintptr_t)a->uaddr <= (uintptr_t)b->uaddr;
}

static struct avl_tree g_futex_tree = { .cmp = futex_tree_cmp };

static spinlock_t g_futex_tree_lock = INIT_SPINLOCK_UNLOCKED;

static void get_futex(struct shim_futex* futex) {
    REF_INC(futex->_ref_count);
}

static void put_futex(struct shim_futex* futex) {
    if (!REF_DEC(futex->_ref_count)) {
        free(futex);
    }
}

/* Since we distinguish futexes by their virtual address, we can as well create a total ordering
 * based on it. */
static int cmp_futexes(struct shim_futex* futex1, struct shim_futex* futex2) {
    uintptr_t f1 = (uintptr_t)futex1->uaddr;
    uintptr_t f2 = (uintptr_t)futex2->uaddr;

    if (f1 < f2) {
        return -1;
    } else if (f1 == f2) {
        return 0;
    } else {
        return 1;
    }
}

/*
 * Locks two futexes in ascending order (defined by cmp_futexes).
 * If a futex is NULL, it is just skipped.
 */
static void lock_two_futexes(struct shim_futex* futex1, struct shim_futex* futex2) {
    if (!futex1 && !futex2) {
        return;
    } else if (futex1 && !futex2) {
        spinlock_lock(&futex1->lock);
        return;
    } else if (!futex1 && futex2) {
        spinlock_lock(&futex2->lock);
        return;
    }
    /* Both are not NULL. */

    /* To avoid deadlocks we always take the locks in ascending order of futexes.
     * If both futexes are equal, just take one lock. */
    int cmp = cmp_futexes(futex1, futex2);
    if (cmp < 0) {
        spinlock_lock(&futex1->lock);
        spinlock_lock(&futex2->lock);
    } else if (cmp == 0) {
        spinlock_lock(&futex1->lock);
    } else {
        spinlock_lock(&futex2->lock);
        spinlock_lock(&futex1->lock);
    }
}

static void unlock_two_futexes(struct shim_futex* futex1, struct shim_futex* futex2) {
    if (!futex1 && !futex2) {
        return;
    } else if (futex1 && !futex2) {
        spinlock_unlock(&futex1->lock);
        return;
    } else if (!futex1 && futex2) {
        spinlock_unlock(&futex2->lock);
        return;
    }
    /* Both are not NULL. */

    /* For unlocking order does not matter. */
    int cmp = cmp_futexes(futex1, futex2);
    if (cmp) {
        spinlock_unlock(&futex1->lock);
        spinlock_unlock(&futex2->lock);
    } else {
        spinlock_unlock(&futex1->lock);
    }
}

/*
 * Adds `futex` to `g_futex_tree`.
 *
 * `g_futex_tree_lock` should be held while calling this function and you must ensure that nobody
 * is using `futex` (e.g. you have just created it).
 */
static void enqueue_futex(struct shim_futex* futex) {
    assert(spinlock_is_locked(&g_futex_tree_lock));

    get_futex(futex);
    avl_tree_insert(&g_futex_tree, &futex->tree_node);
    futex->in_tree = true;
}

/*
 * Checks whether `futex` has no waiters and is on `g_futex_tree`.
 *
 * This requires only `futex->lock` to be held.
 */
static bool check_dequeue_futex(struct shim_futex* futex) {
    assert(spinlock_is_locked(&futex->lock));

    return LISTP_EMPTY(&futex->waiters) && futex->in_tree;
}

static void _maybe_dequeue_futex(struct shim_futex* futex) {
    assert(spinlock_is_locked(&futex->lock));
    assert(spinlock_is_locked(&g_futex_tree_lock));

    if (check_dequeue_futex(futex)) {
        avl_tree_delete(&g_futex_tree, &futex->tree_node);
        futex->in_tree = false;
        /* We still hold this futex reference (in the caller), so this won't call free. */
        put_futex(futex);
    }
}

/*
 * If `futex` has no waiters and is on `g_futex_tree`, takes it off that tree.
 *
 * Neither `g_futex_tree_lock` nor `futex->lock` should be held while calling this,
 * it acquires these locks itself.
 */
static void maybe_dequeue_futex(struct shim_futex* futex) {
    spinlock_lock(&g_futex_tree_lock);
    spinlock_lock(&futex->lock);
    _maybe_dequeue_futex(futex);
    spinlock_unlock(&futex->lock);
    spinlock_unlock(&g_futex_tree_lock);
}

/*
 * Same as `maybe_dequeue_futex`, but works for two futexes, any of which might be NULL.
 */
static void maybe_dequeue_two_futexes(struct shim_futex* futex1, struct shim_futex* futex2) {
    spinlock_lock(&g_futex_tree_lock);
    lock_two_futexes(futex1, futex2);
    if (futex1) {
        _maybe_dequeue_futex(futex1);
    }
    if (futex2) {
        _maybe_dequeue_futex(futex2);
    }
    unlock_two_futexes(futex1, futex2);
    spinlock_unlock(&g_futex_tree_lock);
}

/*
 * Adds `waiter` to `futex` waiters list.
 * You need to make sure that this futex is still on `g_futex_tree`, but in most cases it follows
 * from the program control flow.
 *
 * `futex->lock` needs to be held.
 */
static void add_futex_waiter(struct futex_waiter* waiter, struct shim_futex* futex,
                             uint32_t bitset) {
    assert(spinlock_is_locked(&futex->lock));

    waiter->thread = get_cur_thread();
    get_thread(waiter->thread);

    INIT_LIST_HEAD(waiter, list);
    waiter->bitset = bitset;
    get_futex(futex);
    waiter->futex = futex;
    LISTP_ADD_TAIL(waiter, &futex->waiters, list);
}

/*
 * Ownership of the `waiter->thread` is passed to the caller; we do not change its refcount because
 * we take it of `futex->waiters` list (-1) and give it to caller (+1).
 *
 * `futex->lock` needs to be held.
 */
static struct shim_thread* remove_futex_waiter(struct futex_waiter* waiter,
                                               struct shim_futex* futex) {
    assert(spinlock_is_locked(&futex->lock));

    LISTP_DEL_INIT(waiter, &futex->waiters, list);
    return waiter->thread;
}

/*
 * Moves waiter from `futex1` to `futex2`.
 * As in `add_futex_waiter`, `futex2` needs to be on `g_futex_tree`.
 *
 * `futex1->lock` and `futex2->lock` need to be held.
 */
static void move_futex_waiter(struct futex_waiter* waiter, struct shim_futex* futex1,
                              struct shim_futex* futex2) {
    assert(spinlock_is_locked(&g_futex_tree_lock));
    assert(spinlock_is_locked(&futex1->lock));
    assert(spinlock_is_locked(&futex2->lock));

    LISTP_DEL_INIT(waiter, &futex1->waiters, list);
    get_futex(futex2);
    put_futex(waiter->futex);
    waiter->futex = futex2;
    LISTP_ADD_TAIL(waiter, &futex2->waiters, list);
}

/*
 * Creates a new futex.
 * Sets the new futex refcount to 1.
 */
static struct shim_futex* create_new_futex(uint32_t* uaddr) {
    struct shim_futex* futex;

    futex = calloc(1, sizeof(*futex));
    if (!futex) {
        return NULL;
    }

    REF_SET(futex->_ref_count, 1);

    futex->uaddr = uaddr;
    futex->in_tree = false;
    INIT_LISTP(&futex->waiters);
    spinlock_init(&futex->lock);

    return futex;
}

/*
 * Finds a futex in `g_futex_tree`.
 * Must be called with `g_futex_tree_lock` held.
 * Increases refcount of futex by 1.
 */
static struct shim_futex* find_futex(uint32_t* uaddr) {
    assert(spinlock_is_locked(&g_futex_tree_lock));
    struct shim_futex* futex = NULL;
    struct shim_futex cmp_arg = {
        .uaddr = uaddr
    };
    struct avl_tree_node* node = avl_tree_find(&g_futex_tree, &cmp_arg.tree_node);
    if (!node) {
        return NULL;
    }

    futex = container_of(node, struct shim_futex, tree_node);
    get_futex(futex);
    return futex;
}

static int futex_wait(uint32_t* uaddr, uint32_t val, uint64_t timeout, uint32_t bitset) {
    int ret = 0;
    struct shim_futex* futex = NULL;
    struct shim_thread* thread = NULL;
    struct shim_futex* tmp = NULL;

    spinlock_lock(&g_futex_tree_lock);
    futex = find_futex(uaddr);
    if (!futex) {
        spinlock_unlock(&g_futex_tree_lock);
        tmp = create_new_futex(uaddr);
        if (!tmp) {
            return -ENOMEM;
        }
        spinlock_lock(&g_futex_tree_lock);
        futex = find_futex(uaddr);
        if (!futex) {
            enqueue_futex(tmp);
            futex = tmp;
            tmp = NULL;
        }
    }
    spinlock_lock(&futex->lock);
    spinlock_unlock(&g_futex_tree_lock);

    if (__atomic_load_n(uaddr, __ATOMIC_RELAXED) != val) {
        ret = -EAGAIN;
        goto out_with_futex_lock;
    }

    thread_prepare_wait();

    struct futex_waiter waiter = {0};
    add_futex_waiter(&waiter, futex, bitset);

    spinlock_unlock(&futex->lock);

    /* Give up this futex reference - we have no idea what futex we will be on once we wake up
     * (due to possible requeues). */
    put_futex(futex);
    futex = NULL;

    ret = thread_wait(timeout != NO_TIMEOUT ? &timeout : NULL, /*ignore_pending_signals=*/false);

    spinlock_lock(&g_futex_tree_lock);
    /* We might have been requeued. Grab the (possibly new) futex reference. */
    futex = waiter.futex;
    assert(futex);
    get_futex(futex);
    spinlock_lock(&futex->lock);
    spinlock_unlock(&g_futex_tree_lock);

    if (!LIST_EMPTY(&waiter, list)) {
        /* If we woke up due to time out or a signal, we were not removed from the waiters list
         * (opposite of when another thread calls FUTEX_WAKE, which would remove us from the list).
         */
        thread = remove_futex_waiter(&waiter, futex);

        if (ret == 0 || ret == -EINTR) {
            ret = -ERESTARTSYS;
        }
    } else if (ret == -EINTR) {
        ret = 0;
    }

    /* At this point we are done using the `waiter` struct and need to give up the futex reference
     * it was holding.
     * NB: actually `futex` and this point to the same futex, so this won't call free. */
    put_futex(waiter.futex);

out_with_futex_lock:; // C is awesome!
    /* Because dequeuing a futex requires `g_futex_tree_lock` which we do not hold at this moment,
     * we check if we actually need to do it now (locks acquisition and dequeuing). */
    bool needs_dequeue = check_dequeue_futex(futex);

    spinlock_unlock(&futex->lock);

    if (needs_dequeue) {
        maybe_dequeue_futex(futex);
    }

    if (thread) {
        put_thread(thread);
    }

    put_futex(futex);
    if (tmp) {
        put_futex(tmp);
    }
    return ret;
}

/*
 * Moves at most `to_wake` waiters from futex to wake queue;
 * In the Linux kernel the number of waiters to wake has type `int` and we follow that here.
 * Normally `bitset` has to be non-zero, here zero means: do not even check it.
 *
 * Must be called with `futex->lock` held.
 *
 * Returns number of threads woken.
 */
static int move_to_wake_queue(struct shim_futex* futex, uint32_t bitset, int to_wake,
                              struct wake_queue_head* queue) {
    assert(spinlock_is_locked(&futex->lock));

    struct futex_waiter* waiter;
    struct futex_waiter* wtmp;
    struct shim_thread* thread;
    int woken = 0;

    LISTP_FOR_EACH_ENTRY_SAFE(waiter, wtmp, &futex->waiters, list) {
        if (bitset && !(waiter->bitset & bitset)) {
            continue;
        }

        thread = remove_futex_waiter(waiter, futex);
        add_thread_to_queue(queue, thread);
        put_thread(thread);

        /* If to_wake (3rd argument of futex syscall) is 0, the Linux kernel still wakes up
         * one thread - so we do the same here. */
        if (++woken >= to_wake) {
            break;
        }
    }

    return woken;
}

static int futex_wake(uint32_t* uaddr, int to_wake, uint32_t bitset) {
    struct shim_futex* futex;
    struct wake_queue_head queue = {.first = WAKE_QUEUE_TAIL};
    int woken = 0;

    if (!bitset) {
        return -EINVAL;
    }

    spinlock_lock(&g_futex_tree_lock);
    futex = find_futex(uaddr);
    if (!futex) {
        spinlock_unlock(&g_futex_tree_lock);
        return 0;
    }
    spinlock_lock(&futex->lock);
    spinlock_unlock(&g_futex_tree_lock);

    woken = move_to_wake_queue(futex, bitset, to_wake, &queue);

    bool needs_dequeue = check_dequeue_futex(futex);

    spinlock_unlock(&futex->lock);

    if (needs_dequeue) {
        maybe_dequeue_futex(futex);
    }

    wake_queue(&queue);

    put_futex(futex);

    return woken;
}

/*
 * Sign-extends 12 bit argument to 32 bits.
 */
static int wakeop_arg_extend(int x) {
    if (x >= 0x800) {
        return 0xfffff000 | x;
    }
    return x;
}

static int futex_wake_op(uint32_t* uaddr1, uint32_t* uaddr2, int to_wake1, int to_wake2,
                         uint32_t val3) {
    struct shim_futex* futex1 = NULL;
    struct shim_futex* futex2 = NULL;
    struct wake_queue_head queue = {.first = WAKE_QUEUE_TAIL};
    int ret = 0;
    bool needs_dequeue1 = false;
    bool needs_dequeue2 = false;

    spinlock_lock(&g_futex_tree_lock);
    futex1 = find_futex(uaddr1);
    futex2 = find_futex(uaddr2);

    lock_two_futexes(futex1, futex2);
    spinlock_unlock(&g_futex_tree_lock);

    unsigned int op = (val3 >> 28) & 0x7; // highest bit is for FUTEX_OP_OPARG_SHIFT
    unsigned int cmp = (val3 >> 24) & 0xf;
    int oparg = wakeop_arg_extend((val3 >> 12) & 0xfff);
    int cmparg = wakeop_arg_extend(val3 & 0xfff);
    int oldval;
    bool cmpval;

    if ((val3 >> 28) & FUTEX_OP_OPARG_SHIFT) {
        if (oparg < 0 || oparg > 31) {
            /* In case of invalid argument to shift the Linux kernel just fixes the argument,
             * so we do the same. */
            oparg &= 0x1f;
        }
        if (oparg == 31) {
            // left shift by 31 would be UB here
            oparg = -2147483648;
        } else {
            oparg = 1 << oparg;
        }
    }

    switch (op) {
        case FUTEX_OP_SET:
            oldval = __atomic_exchange_n(uaddr2, oparg, __ATOMIC_RELAXED);
            break;
        case FUTEX_OP_ADD:
            oldval = __atomic_fetch_add(uaddr2, oparg, __ATOMIC_RELAXED);
            break;
        case FUTEX_OP_OR:
            oldval = __atomic_fetch_or(uaddr2, oparg, __ATOMIC_RELAXED);
            break;
        case FUTEX_OP_ANDN:
            oldval = __atomic_fetch_and(uaddr2, ~oparg, __ATOMIC_RELAXED);
            break;
        case FUTEX_OP_XOR:
            oldval = __atomic_fetch_xor(uaddr2, oparg, __ATOMIC_RELAXED);
            break;
        default:
            ret = -ENOSYS;
            goto out_unlock;
    }

    switch (cmp) {
        case FUTEX_OP_CMP_EQ:
            cmpval = oldval == cmparg;
            break;
        case FUTEX_OP_CMP_NE:
            cmpval = oldval != cmparg;
            break;
        case FUTEX_OP_CMP_LT:
            cmpval = oldval < cmparg;
            break;
        case FUTEX_OP_CMP_LE:
            cmpval = oldval <= cmparg;
            break;
        case FUTEX_OP_CMP_GT:
            cmpval = oldval > cmparg;
            break;
        case FUTEX_OP_CMP_GE:
            cmpval = oldval >= cmparg;
            break;
        default:
            ret = -ENOSYS;
            goto out_unlock;
    }

    if (futex1) {
        ret += move_to_wake_queue(futex1, 0, to_wake1, &queue);
        needs_dequeue1 = check_dequeue_futex(futex1);
    }
    if (futex2 && cmpval) {
        ret += move_to_wake_queue(futex2, 0, to_wake2, &queue);
        needs_dequeue2 = check_dequeue_futex(futex2);
    }

out_unlock:
    unlock_two_futexes(futex1, futex2);

    if (needs_dequeue1 || needs_dequeue2) {
        maybe_dequeue_two_futexes(futex1, futex2);
    }

    if (ret > 0) {
        wake_queue(&queue);
    }

    if (futex1) {
        put_futex(futex1);
    }
    if (futex2) {
        put_futex(futex2);
    }
    return ret;
}

static int futex_requeue(uint32_t* uaddr1, uint32_t* uaddr2, int to_wake, int to_requeue,
                         uint32_t* val) {
    struct shim_futex* futex1 = NULL;
    struct shim_futex* futex2 = NULL;
    struct shim_futex* tmp = NULL;
    struct wake_queue_head queue = {.first = WAKE_QUEUE_TAIL};
    int ret = 0;
    int woken = 0;
    int requeued = 0;
    struct futex_waiter* waiter;
    struct futex_waiter* wtmp;
    struct shim_thread* thread;
    bool needs_dequeue1 = false;
    bool needs_dequeue2 = false;

    if (to_wake < 0 || to_requeue < 0) {
        return -EINVAL;
    }

    spinlock_lock(&g_futex_tree_lock);
    futex2 = find_futex(uaddr2);
    if (!futex2) {
        spinlock_unlock(&g_futex_tree_lock);
        tmp = create_new_futex(uaddr2);
        if (!tmp) {
            return -ENOMEM;
        }
        needs_dequeue2 = true;

        spinlock_lock(&g_futex_tree_lock);
        futex2 = find_futex(uaddr2);
        if (!futex2) {
            enqueue_futex(tmp);
            futex2 = tmp;
            tmp = NULL;
        }
    }
    futex1 = find_futex(uaddr1);

    lock_two_futexes(futex1, futex2);

    if (val != NULL) {
        if (__atomic_load_n(uaddr1, __ATOMIC_RELAXED) != *val) {
            ret = -EAGAIN;
            goto out_unlock;
        }
    }

    if (futex1) {
        /* We cannot call move_to_wake_queue here, as this function wakes at least 1 thread,
         * (even if to_wake is 0) and here we want to wake-up exactly to_wake threads.
         * I guess it's better to be compatible and replicate these weird corner cases. */
        LISTP_FOR_EACH_ENTRY_SAFE(waiter, wtmp, &futex1->waiters, list) {
            if (woken < to_wake) {
                thread = remove_futex_waiter(waiter, futex1);
                add_thread_to_queue(&queue, thread);
                put_thread(thread);
                ++woken;
            } else if (requeued < to_requeue) {
                move_futex_waiter(waiter, futex1, futex2);
                ++requeued;
            } else {
                break;
            }
        }

        needs_dequeue1 = check_dequeue_futex(futex1);
        needs_dequeue2 = check_dequeue_futex(futex2);

        ret = woken + requeued;
    }

out_unlock:
    unlock_two_futexes(futex1, futex2);
    spinlock_unlock(&g_futex_tree_lock);

    if (needs_dequeue1 || needs_dequeue2) {
        maybe_dequeue_two_futexes(futex1, futex2);
    }

    if (woken > 0) {
        wake_queue(&queue);
    }

    if (futex1) {
        put_futex(futex1);
    }
    assert(futex2);
    put_futex(futex2);

    if (tmp) {
        put_futex(tmp);
    }

    return ret;
}

#define FUTEX_CHECK_READ  false
#define FUTEX_CHECK_WRITE true
static int is_valid_futex_ptr(uint32_t* ptr, bool check_write) {
    if (!IS_ALIGNED_PTR(ptr, alignof(*ptr))) {
        return -EINVAL;
    }
    if (check_write) {
        if (!is_user_memory_writable(ptr, sizeof(*ptr))) {
            return -EFAULT;
        }
    } else {
        if (!is_user_memory_readable(ptr, sizeof(*ptr))) {
            return -EFAULT;
        }
    }
    return 0;
}

static int _shim_do_futex(uint32_t* uaddr, int op, uint32_t val, void* utime, uint32_t* uaddr2,
                          uint32_t val3) {
    int cmd = op & FUTEX_CMD_MASK;
    uint64_t timeout = NO_TIMEOUT;
    uint32_t val2 = 0;

    if (utime && (cmd == FUTEX_WAIT || cmd == FUTEX_WAIT_BITSET || cmd == FUTEX_LOCK_PI ||
                  cmd == FUTEX_WAIT_REQUEUE_PI)) {
        struct __kernel_timespec* user_timeout = utime;
        if (!is_user_memory_readable(user_timeout, sizeof(*user_timeout))) {
            return -EFAULT;
        }
        timeout = timespec_to_us(user_timeout);
        if (cmd != FUTEX_WAIT) {
            /* For FUTEX_WAIT, timeout is interpreted as a relative value, which differs from other
             * futex operations, where timeout is interpreted as an absolute value. */
            uint64_t current_time = 0;
            int ret = DkSystemTimeQuery(&current_time);
            if (ret < 0) {
                return pal_to_unix_errno(ret);
            }
            if (timeout < current_time) {
                /* We timeouted even before reaching this point. */
                return -ETIMEDOUT;
            }
            timeout -= current_time;
        }
    }

    if (cmd == FUTEX_CMP_REQUEUE || cmd == FUTEX_REQUEUE || cmd == FUTEX_WAKE_OP ||
          cmd == FUTEX_CMP_REQUEUE_PI) {
        val2 = (uint32_t)(unsigned long)utime;
    }

    if (op & FUTEX_CLOCK_REALTIME) {
        if (cmd != FUTEX_WAIT && cmd != FUTEX_WAIT_BITSET && cmd != FUTEX_WAIT_REQUEUE_PI) {
            return -ENOSYS;
        }
        /* Gramine has only one clock for now. */
        log_warning("Ignoring FUTEX_CLOCK_REALTIME flag");
    }

    if (!(op & FUTEX_PRIVATE_FLAG)) {
        log_warning("Non-private futexes are not supported, assuming implicit FUTEX_PRIVATE_FLAG");
    }

    int ret = 0;

    /* `uaddr` should be valid pointer in all cases. */
    ret = is_valid_futex_ptr(uaddr, FUTEX_CHECK_READ);
    if (ret) {
        return ret;
    }

    switch (cmd) {
        case FUTEX_WAIT:
            val3 = FUTEX_BITSET_MATCH_ANY;
            /* fallthrough */
        case FUTEX_WAIT_BITSET:
            return futex_wait(uaddr, val, timeout, val3);
        case FUTEX_WAKE:
            val3 = FUTEX_BITSET_MATCH_ANY;
            /* fallthrough */
        case FUTEX_WAKE_BITSET:
            return futex_wake(uaddr, val, val3);
        case FUTEX_WAKE_OP:
            ret = is_valid_futex_ptr(uaddr2, FUTEX_CHECK_WRITE);
            if (ret) {
                return ret;
            }
            return futex_wake_op(uaddr, uaddr2, val, val2, val3);
        case FUTEX_REQUEUE:
            ret = is_valid_futex_ptr(uaddr2, FUTEX_CHECK_READ);
            if (ret) {
                return ret;
            }
            return futex_requeue(uaddr, uaddr2, val, val2, NULL);
        case FUTEX_CMP_REQUEUE:
            ret = is_valid_futex_ptr(uaddr2, FUTEX_CHECK_READ);
            if (ret) {
                return ret;
            }
            return futex_requeue(uaddr, uaddr2, val, val2, &val3);
        case FUTEX_LOCK_PI:
        case FUTEX_TRYLOCK_PI:
        case FUTEX_UNLOCK_PI:
        case FUTEX_CMP_REQUEUE_PI:
        case FUTEX_WAIT_REQUEUE_PI:
            log_warning("PI futexes are not yet supported!");
            return -ENOSYS;
        default:
            log_warning("Invalid futex op: %d", cmd);
            return -ENOSYS;
    }
}

long shim_do_futex(int* uaddr, int op, int val, void* utime, int* uaddr2, int val3) {
    static_assert(sizeof(int) == 4, "futexes are defined to be 32-bit");
    return _shim_do_futex((uint32_t*)uaddr, op, (uint32_t)val, utime, (uint32_t*)uaddr2,
                          (uint32_t)val3);
}

long shim_do_set_robust_list(struct robust_list_head* head, size_t len) {
    if (len != sizeof(struct robust_list_head)) {
        return -EINVAL;
    }

    get_cur_thread()->robust_list = head;
    return 0;
}

long shim_do_get_robust_list(pid_t pid, struct robust_list_head** head, size_t* len) {
    struct shim_thread* thread;
    int ret = 0;

    if (pid) {
        thread = lookup_thread(pid);
        if (!thread) {
            /* We only support get_robust_list on threads in the same thread group. */
            return -ESRCH;
        }
    } else {
        thread = get_cur_thread();
        get_thread(thread);
    }

    if (!is_user_memory_writable(head, sizeof(*head)) ||
            !is_user_memory_writable(len, sizeof(*len))) {
        ret = -EFAULT;
        goto out;
    }

    *head = thread->robust_list;
    *len = sizeof(**head);

out:
    put_thread(thread);
    return ret;
}

/*
 * Process one robust futex, waking a waiter if present.
 * Returns 0 on success, negative value otherwise.
 */
static bool handle_futex_death(uint32_t* uaddr) {
    uint32_t val;

    if (!IS_ALIGNED_PTR(uaddr, alignof(*uaddr))) {
        return -EINVAL;
    }
    if (!is_valid_futex_ptr(uaddr, FUTEX_CHECK_WRITE)) {
        return -EFAULT;
    }

    /* Loop until we successfully set the futex word or see someone else taking this futex. */
    while (1) {
        val = __atomic_load_n(uaddr, __ATOMIC_RELAXED);

        if ((val & FUTEX_TID_MASK) != get_cur_thread()->tid) {
            /* Someone else is holding this futex. */
            return 0;
        }

        /* Mark the FUTEX_OWNER_DIED bit, clear all tid bits. */
        uint32_t new_val = (val & FUTEX_WAITERS) | FUTEX_OWNER_DIED;

        if (__atomic_compare_exchange_n(uaddr, &val, new_val,
                                        /*weak=*/false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
            /* Successfully set the new value, end the loop. */
            break;
        }
    }

    if (val & FUTEX_WAITERS) {
        /* There are waiters present, wake one of them. */
        futex_wake(uaddr, 1, FUTEX_BITSET_MATCH_ANY);
    }

    return 0;
}

/*
 * Fetches robust list entry from user memory, checking invalid pointers.
 * Returns 0 on success, negative value on error.
 */
static bool fetch_robust_entry(struct robust_list** entry, struct robust_list** head) {
    if (!is_user_memory_readable(head, sizeof(*head))) {
        return -EFAULT;
    }

    *entry = *head;
    return 0;
}

static uint32_t* entry_to_futex(struct robust_list* entry, long futex_offset) {
    return (uint32_t*)((char*)entry + futex_offset);
}

/*
 * Release all robust futexes.
 * The list itself is in user provided memory - we need to check each pointer before dereferencing
 * it. If any check fails, we silently return and ignore the rest.
 */
void release_robust_list(struct robust_list_head* head) {
    struct robust_list* entry;
    struct robust_list* pending;
    long futex_offset;
    unsigned long limit = ROBUST_LIST_LIMIT;

    /* `&head->list.next` does not dereference head, hence is safe. */
    if (fetch_robust_entry(&entry, &head->list.next)) {
        return;
    }

    if (!is_user_memory_readable(&head->futex_offset, sizeof(head->futex_offset))) {
        return;
    }
    futex_offset = head->futex_offset;

    if (fetch_robust_entry(&pending, &head->list_op_pending)) {
        return;
    }

    /* Last entry (or first, if the list is empty) points to the list head. */
    while (entry != &head->list) {
        struct robust_list* next_entry;

        /* Fetch the next entry before waking the next thread. */
        bool ret = fetch_robust_entry(&next_entry, &entry->next);

        if (entry != pending) {
            if (handle_futex_death(entry_to_futex(entry, futex_offset))) {
                return;
            }
        }

        if (ret) {
            return;
        }

        entry = next_entry;

        /* This mostly guards from circular lists. */
        if (!--limit) {
            break;
        }
    }

    if (pending) {
        if (handle_futex_death(entry_to_futex(pending, futex_offset))) {
            return;
        }
    }
}

void release_clear_child_tid(int* clear_child_tid) {
    if (!clear_child_tid || !IS_ALIGNED_PTR(clear_child_tid, alignof(*clear_child_tid)) ||
        !is_user_memory_writable(clear_child_tid, sizeof(*clear_child_tid)))
        return;

    /* child thread exited, now parent can wake up */
    __atomic_store_n(clear_child_tid, 0, __ATOMIC_RELAXED);
    futex_wake((uint32_t*)clear_child_tid, 1, FUTEX_BITSET_MATCH_ANY);
}
