/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Labs */

/* NOTE: This seqlock implementation follows the spinlock one; see spinlock.h for details.
 *       This seqlock implementation is based on the Linux one; see include/linux/seqlock.h.
 *
 * Reader/writer consistent mechanism with fast lock-free readers and without starving writers.
 * Seqlocks allow readers to race with writers, but readers detect when a race occurs and retry the
 * critical section (i.e., reads are speculative). Writers synchronize their accesses with other
 * writers via a spinlock, but never wait for readers.
 *
 * Seqlocks can be only used when readers do not mutate any shared state and do not use pointers.
 * For example, seqlocks are a perfect fit for time management routines. Typical usage for readers:
 *     do {
 *         seq = read_seqbegin(&foo);
 *         ...
 *     } while (read_seqretry(&foo, seq));
 */

#ifndef _SEQLOCK_H
#define _SEQLOCK_H

#include "api.h"
#include "cpu.h"
#include "spinlock.h"

typedef struct {
    uint32_t sequence;
    spinlock_t lock;
} seqlock_t;

/*!
 * \brief Initialize seqlock with *static* storage duration.
 */
#define INIT_SEQLOCK_UNLOCKED { .sequence = 0, .lock = INIT_SPINLOCK_UNLOCKED }


/*!
 * \brief Initialize seqlock with *dynamic* storage duration.
 */
static inline void seqlock_init(seqlock_t* sl) {
    __atomic_store_n(&sl->sequence, 0, __ATOMIC_RELAXED);
    spinlock_init(&sl->lock);
}

/*!
 * \brief Start a reader-side critical section (no locking).
 */
static inline uint32_t read_seqbegin(seqlock_t* sl) {
    uint32_t start;
    while (true) {
        start = __atomic_load_n(&sl->sequence, __ATOMIC_RELAXED);
        if ((start & 1) == 0) {
            /* there is no writer in the middle of an update, data is currently consistent */
            break;
        }
        CPU_RELAX();
    }
    RMB();
    return start;
}

/*!
 * \brief Returns true if reader must retry the critical section.
 */
static inline bool read_seqretry(seqlock_t* sl, uint32_t start) {
    RMB();
    return start != __atomic_load_n(&sl->sequence, __ATOMIC_RELAXED);
}

/*!
 * \brief Start a writer-side critical section (acquires spinlock).
 */
static inline void write_seqbegin(seqlock_t* sl) {
    spinlock_lock(&sl->lock);
    __atomic_add_fetch(&sl->sequence, 1, __ATOMIC_RELAXED);
    WMB();
}

/*!
 * \brief End a writer-side critical section (releases spinlock).
 */
static inline void write_seqend(seqlock_t* sl) {
    WMB();
    __atomic_add_fetch(&sl->sequence, 1, __ATOMIC_RELAXED);
    spinlock_unlock(&sl->lock);
}

#endif // _SEQLOCK_H
