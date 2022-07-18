/* NOTE: This spinlock library must be implemented in a secure way even when a spinlock is stored
 *       in the untrusted memory and accessed by the enclave. Currently, the only sensitive field
 *       is `lock` but there is no way to prevent tampering with it in untrusted memory: callers
 *       must ensure that returning prematurely from such a spinlock leads only to DoS and not to
 *       data corruptions. */

#ifndef _SPINLOCK_H
#define _SPINLOCK_H

#include "api.h"
#include "cpu.h"
#include "log.h"

#ifdef DEBUG
#define DEBUG_SPINLOCKS
#endif // DEBUG

#ifdef IN_SHIM
/* Forward declare, so this header stays standalone. */
static inline unsigned int get_cur_tid(void);

#ifdef DEBUG_SPINLOCKS
#define DEBUG_SPINLOCKS_SHIM
#endif // DEBUG_SPINLOCKS

#endif // IN_SHIM

typedef struct {
    uint32_t lock;
#ifdef DEBUG_SPINLOCKS_SHIM
    unsigned int owner;
#endif // DEBUG_SPINLOCKS_SHIM
} spinlock_t;

/* The below macros are only needed for our own futex implementation (based on Futexes are Tricky)
 * used in the Exitless mechanism (in the RPC queue synchronization). Note that ordering is
 * important due to atomic-decrement in unlock logic. */
#define SPINLOCK_UNLOCKED            0
#define SPINLOCK_LOCKED              1
#define SPINLOCK_LOCKED_NO_WAITERS   1 /* used for futex implementation */
#define SPINLOCK_LOCKED_WITH_WAITERS 2 /* used for futex implementation */

/*!
 * \brief Initialize spinlock with *static* storage duration.
 *
 * According to C standard, the only guarantee we have is that this initialization will happen
 * before main, which by itself is not enough (such store might not be visible before first lock
 * acquire). Fortunately on gcc global zeroed variables will just end up in .bss - zeroed memory
 * mapped during process creation, hence we are fine.
 *
 * Rest of the struct is zeroed implicitly, hence no need for ifdef here.
 */
#define INIT_SPINLOCK_UNLOCKED { .lock = SPINLOCK_UNLOCKED }

#ifdef DEBUG_SPINLOCKS_SHIM
static inline void debug_spinlock_take_ownership(spinlock_t* lock) {
    __atomic_store_n(&lock->owner, get_cur_tid(), __ATOMIC_RELAXED);
}

static inline void debug_spinlock_giveup_ownership(spinlock_t* lock) {
    __atomic_store_n(&lock->owner, SPINLOCK_UNLOCKED, __ATOMIC_RELAXED);
}
#else
static inline void debug_spinlock_take_ownership(spinlock_t* lock) {
    __UNUSED(lock);
}

static inline void debug_spinlock_giveup_ownership(spinlock_t* lock) {
    __UNUSED(lock);
}
#endif // DEBUG_SPINLOCKS_SHIM


/*!
 * \brief Initialize spinlock with *dynamic* storage duration.
 */
static inline void spinlock_init(spinlock_t* lock) {
    debug_spinlock_giveup_ownership(lock);
    __atomic_store_n(&lock->lock, SPINLOCK_UNLOCKED, __ATOMIC_RELAXED);
}

/*!
 * \brief Acquire spinlock.
 */
static inline void spinlock_lock(spinlock_t* lock) {
    uint32_t val;

    /* First check if lock is already free. */
    if (__atomic_exchange_n(&lock->lock, SPINLOCK_LOCKED, __ATOMIC_ACQUIRE) == SPINLOCK_UNLOCKED) {
        goto out;
    }

    do {
        /* This check imposes no inter-thread ordering, thus does not slow other threads. */
        while (__atomic_load_n(&lock->lock, __ATOMIC_RELAXED) != SPINLOCK_UNLOCKED)
            CPU_RELAX();
        /* Seen lock as free, check if it still is, this time with acquire semantics (but only
         * if we really take it). */
        val = SPINLOCK_UNLOCKED;
    } while (!__atomic_compare_exchange_n(&lock->lock, &val, SPINLOCK_LOCKED, /*weak=*/false,
                                          __ATOMIC_ACQUIRE, __ATOMIC_RELAXED));

out:
    debug_spinlock_take_ownership(lock);
}

/*!
 * \brief Try to acquire a spinlock, with a timeout.
 *
 * \param lock        The lock.
 * \param iterations  Number of iterations (tries) after which this function times out.
 *
 * \returns true if acquiring the lock succeeded, false if timed out.
 */
static inline bool spinlock_lock_timeout(spinlock_t* lock, unsigned long iterations) {
    uint32_t val;

    /* First check if lock is already free. */
    if (__atomic_exchange_n(&lock->lock, SPINLOCK_LOCKED, __ATOMIC_ACQUIRE) == SPINLOCK_UNLOCKED) {
        goto out_success;
    }

    do {
        /* This check imposes no inter-thread ordering, thus does not slow other threads. */
        while (__atomic_load_n(&lock->lock, __ATOMIC_RELAXED) != SPINLOCK_UNLOCKED) {
            if (iterations == 0) {
                return false;
            }
            iterations--;
            CPU_RELAX();
        }
        /* Seen lock as free, check if it still is, this time with acquire semantics (but only
         * if we really take it). */
        val = SPINLOCK_UNLOCKED;
    } while (!__atomic_compare_exchange_n(&lock->lock, &val, SPINLOCK_LOCKED, /*weak=*/false,
                                          __ATOMIC_ACQUIRE, __ATOMIC_RELAXED));

out_success:
    debug_spinlock_take_ownership(lock);
    return true;
}

/*!
 * \brief Compare the contents of `*lock` with the contents of `*expected`
 *
 * Semantics are the same as gcc's `atomic_compare_exchange_n()`. If the contents of `*lock` and
 * `*expected` are equal, this function writes `desired` into `*lock`. Otherwise, current contents
 * of `*lock` are written into `*expected`. If `desired` is written into `*lock` then true is
 * returned.
 */
static inline int spinlock_cmpxchg(spinlock_t* lock, uint32_t* expected, uint32_t desired) {
    static_assert(SAME_TYPE(&lock->lock, expected), "spinlock is not implemented as int*");
    return __atomic_compare_exchange_n(&lock->lock, expected, desired, /*weak=*/false,
                                       __ATOMIC_ACQUIRE, __ATOMIC_RELAXED);
}

/*!
 * \brief Release spinlock.
 */
static inline void spinlock_unlock(spinlock_t* lock) {
    debug_spinlock_giveup_ownership(lock);
    __atomic_store_n(&lock->lock, SPINLOCK_UNLOCKED, __ATOMIC_RELEASE);
}

#ifdef DEBUG_SPINLOCKS
static inline bool _spinlock_is_locked(spinlock_t* lock) {
    return __atomic_load_n(&lock->lock, __ATOMIC_SEQ_CST) != SPINLOCK_UNLOCKED;
}

#ifdef DEBUG_SPINLOCKS_SHIM
static inline bool spinlock_is_locked(spinlock_t* lock) {
    if (!_spinlock_is_locked(lock)) {
        return false;
    }
    unsigned int owner = __atomic_load_n(&lock->owner, __ATOMIC_RELAXED);
    if (owner != get_cur_tid()) {
        log_error("Unexpected lock ownership: owned by: %d, checked in: %d", owner, get_cur_tid());
        return false;
    }
    return true;
}
#else
static inline bool spinlock_is_locked(spinlock_t* lock) {
    return _spinlock_is_locked(lock);
}
#endif // DEBUG_SPINLOCKS_SHIM

#endif // DEBUG_SPINLOCKS

#endif // _SPINLOCK_H
