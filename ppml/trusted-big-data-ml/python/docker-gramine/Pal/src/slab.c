/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains implementation of PAL's internal memory allocator.
 */

#include "api.h"
#include "asan.h"
#include "pal.h"
#include "pal_error.h"
#include "pal_internal.h"
#include "spinlock.h"

static spinlock_t g_slab_mgr_lock = INIT_SPINLOCK_UNLOCKED;

#define SYSTEM_LOCK()   spinlock_lock(&g_slab_mgr_lock)
#define SYSTEM_UNLOCK() spinlock_unlock(&g_slab_mgr_lock)
#define SYSTEM_LOCKED() spinlock_is_locked(&g_slab_mgr_lock)

/* Initial memory pool, optionally provided to `init_slab_mgr()` */
static char* g_mem_pool = NULL;
static bool g_alloc_from_low = true; /* allocate from low addresses if true, from high if false */
static void* g_mem_pool_end;

/* `g_low` and `g_high` are protected by `SYSTEM_LOCK()` */
static void* g_low;
static void* g_high;

static inline void* __malloc(size_t size);
static inline void __free(void* addr, size_t size);
#define system_malloc(size) __malloc(size)
#define system_free(addr, size) __free(addr, size)

#include "slabmgr.h"

static inline void* __malloc(size_t size) {
    void* addr = NULL;

    size = ALIGN_UP(size, MIN_MALLOC_ALIGNMENT);

    /* Use `g_mem_pool`, if available */
    if (g_mem_pool) {
        SYSTEM_LOCK();
        if (g_low + size <= g_high) {
            /* alternate allocating objects from low and high addresses of available memory pool;
             * this allows to free memory for patterns like "malloc1 - malloc2 - free1" (seen in
             * e.g. realloc) */
            if (g_alloc_from_low) {
                addr = g_low;
                g_low += size;
            } else {
                addr = g_high - size;
                g_high -= size;
            }
            g_alloc_from_low = !g_alloc_from_low; /* switch alloc direction for next malloc */
        }
        SYSTEM_UNLOCK();
        if (addr)
            return addr;
    }

    /* We could not use `g_mem_pool`, let's fall back to PAL-internal allocations. PAL allocator
     * must be careful though because LibOS doesn't know about PAL-internal memory, limited via
     * manifest option `loader.pal_internal_mem_size` and thus this malloc may return -ENOMEM. */
    int ret = _DkVirtualMemoryAlloc(&addr, ALLOC_ALIGN_UP(size), PAL_ALLOC_INTERNAL,
                                    PAL_PROT_READ | PAL_PROT_WRITE);
    if (ret < 0) {
        log_error("*** Out-of-memory in PAL (try increasing `loader.pal_internal_mem_size`) ***");
        _DkProcessExit(ENOMEM);
    }
#ifdef ASAN
    asan_poison_region((uintptr_t)addr, ALLOC_ALIGN_UP(size), ASAN_POISON_HEAP_LEFT_REDZONE);
#endif

    return addr;
}

static inline void __free(void* addr, size_t size) {
    if (!addr)
        return;

    size = ALIGN_UP(size, MIN_MALLOC_ALIGNMENT);

    if (g_mem_pool && addr >= (void*)g_mem_pool && addr < g_mem_pool_end) {
        SYSTEM_LOCK();
        if (addr == g_high) {
            /* reclaim space of last object allocated at high addresses */
            g_high = addr + size;
        } else if (addr + size == g_low) {
            /* reclaim space of last object allocated at low addresses */
            g_low = addr;
        }
        /* not a last object from low/high addresses, can't do anything about this case */
#ifdef ASAN
        /* Keep the now-unused part of `g_mem_pool` poisoned, because we know it won't be used by
         * anything other than our allocator */
        asan_poison_region((uintptr_t)addr, size, ASAN_POISON_HEAP_LEFT_REDZONE);
#endif
        SYSTEM_UNLOCK();
        return;
    }

#ifdef ASAN
    /* Unpoison the memory before unmapping it */
    asan_unpoison_region((uintptr_t)addr, ALLOC_ALIGN_UP(size));
#endif

    _DkVirtualMemoryFree(addr, ALLOC_ALIGN_UP(size));
}

static SLAB_MGR g_slab_mgr = NULL;

void init_slab_mgr(char* mem_pool, size_t mem_pool_size) {
    assert(!g_slab_mgr);

    if (mem_pool) {
        g_mem_pool = mem_pool;
        g_mem_pool_end = mem_pool + mem_pool_size;

        g_low = g_mem_pool;
        g_high = g_mem_pool_end;

#ifdef ASAN
        /* Poison all of `mem_pool` initially */
        asan_poison_region((uintptr_t)mem_pool, mem_pool_size, ASAN_POISON_HEAP_LEFT_REDZONE);
#endif
    }

    g_slab_mgr = create_slab_mgr();
    if (!g_slab_mgr)
        INIT_FAIL(PAL_ERROR_NOMEM, "cannot initialize slab manager");
}

void* malloc(size_t size) {
    void* ptr = slab_alloc(g_slab_mgr, size);

#ifdef DEBUG
    /* In debug builds, try to break code that uses uninitialized heap
     * memory by explicitly initializing to a non-zero value. */
    if (ptr)
        memset(ptr, 0xa5, size);
#endif

    if (!ptr) {
        /*
         * Normally, the PAL should not run out of memory.
         * If malloc() failed internally, we cannot handle the
         * condition and must terminate the current process.
         */
        log_error("******** Out-of-memory in PAL ********");
        _DkProcessExit(ENOMEM);
    }
    return ptr;
}

// Copies data from `mem` to a newly allocated buffer of a specified size.
void* malloc_copy(const void* mem, size_t size) {
    void* nmem = malloc(size);

    if (nmem)
        memcpy(nmem, mem, size);

    return nmem;
}

void* calloc(size_t num, size_t size) {
    size_t total;
    if (__builtin_mul_overflow(num, size, &total))
        return NULL;

    void* ptr = malloc(total);
    if (ptr)
        memset(ptr, 0, total);
    return ptr;
}

void free(void* ptr) {
    if (!ptr)
        return;
    slab_free(g_slab_mgr, ptr);
}
