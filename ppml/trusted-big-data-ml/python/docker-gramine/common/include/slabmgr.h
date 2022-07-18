/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains implementation of SLAB (variable-size) memory allocator.
 */

#ifndef SLABMGR_H
#define SLABMGR_H

#include <errno.h>
#include <sys/mman.h>

#include "api.h"
#include "asan.h"
#include "assert.h"
#include "list.h"
#include "log.h"

// Before calling any of `system_malloc` and `system_free` this library will
// acquire `SYSTEM_LOCK` (the system_* implementation must not do it).
#ifndef system_malloc
#error "macro \"void* system_malloc(size_t size)\" not declared"
#endif
#ifndef system_free
#error "macro \"void* system_free(void* ptr, size_t size)\" not declared"
#endif
#ifndef SYSTEM_LOCK
#define SYSTEM_LOCK() ({})
#endif
#ifndef SYSTEM_UNLOCK
#define SYSTEM_UNLOCK() ({})
#endif

/* malloc is supposed to provide some kind of alignment guarantees, but
 * I can't find a specific reference to what that should be for x86_64.
 * The first link here is a reference to a technical report from Mozilla,
 * which seems to indicate that 64-bit platforms align return values to
 * 16-bytes. calloc and malloc provide the same alignment guarantees.
 * calloc additionally sets the memory to 0, which malloc is not required
 * to do.
 *
 * http://www.erahm.org/2016/03/24/minimum-alignment-of-allocation-across-platforms/
 * http://pubs.opengroup.org/onlinepubs/9699919799/functions/malloc.html
 */
#define MIN_MALLOC_ALIGNMENT 16

/* Slab objects need to be a multiple of 16 bytes to ensure proper address
 * alignment for malloc and calloc. */
#define OBJ_PADDING 15

#define LARGE_OBJ_PADDING 8

DEFINE_LIST(slab_obj);

typedef struct __attribute__((packed)) slab_obj {
    unsigned char level;
    unsigned char padding[OBJ_PADDING];
    union {
        LIST_TYPE(slab_obj) __list;
        unsigned char* raw;
    };
} SLAB_OBJ_TYPE, *SLAB_OBJ;

/* In order for slab elements to be 16-byte aligned, struct slab_area must be a multiple of 16 B.
 */
#define AREA_PADDING 8

DEFINE_LIST(slab_area);

typedef struct __attribute__((packed)) slab_area {
    LIST_TYPE(slab_area) __list;
    size_t size;
    unsigned char pad[AREA_PADDING];
    unsigned char raw[];
} SLAB_AREA_TYPE, *SLAB_AREA;

static_assert(IS_ALIGNED(offsetof(struct slab_area, raw), 16),
              "slab_area::raw must be aligned to 16 B");

#ifdef SLAB_CANARY
#define SLAB_CANARY_STRING 0xDEADBEEF
#define SLAB_CANARY_SIZE   (sizeof(unsigned long))
#else
#define SLAB_CANARY_SIZE 0
#endif

#define SLAB_HDR_SIZE                                                                \
    ALIGN_UP(sizeof(SLAB_OBJ_TYPE) - sizeof(LIST_TYPE(slab_obj)) + SLAB_CANARY_SIZE, \
             MIN_MALLOC_ALIGNMENT)

#ifndef SLAB_LEVEL
#define SLAB_LEVEL 8
#endif

#ifndef SLAB_LEVEL_SIZES
#define SLAB_LEVEL_SIZES                                                       \
    16, 32, 64, 128 - SLAB_HDR_SIZE, 256 - SLAB_HDR_SIZE, 512 - SLAB_HDR_SIZE, \
        1024 - SLAB_HDR_SIZE, 2048 - SLAB_HDR_SIZE
#define SLAB_LEVELS_SUM (4080 - SLAB_HDR_SIZE * 5)
#else
#ifndef SLAB_LEVELS_SUM
#error "SALB_LEVELS_SUM not defined"
#endif
#endif

// User buffer sizes on each level (not counting mandatory header
// (SLAB_HDR_SIZE)).
static const size_t slab_levels[SLAB_LEVEL] = {SLAB_LEVEL_SIZES};

DEFINE_LISTP(slab_obj);
DEFINE_LISTP(slab_area);
typedef struct slab_mgr {
    LISTP_TYPE(slab_area) area_list[SLAB_LEVEL];
    LISTP_TYPE(slab_obj) free_list[SLAB_LEVEL];
    size_t size[SLAB_LEVEL];
    void* addr[SLAB_LEVEL];
    void* addr_top[SLAB_LEVEL];
    SLAB_AREA active_area[SLAB_LEVEL];
} SLAB_MGR_TYPE, *SLAB_MGR;

typedef struct __attribute__((packed)) large_mem_obj {
    // offset 0
    unsigned long size;  // User buffer size (i.e. excluding control structures)
    unsigned char large_padding[LARGE_OBJ_PADDING];
    // offset 16
    unsigned char level;
    unsigned char padding[OBJ_PADDING];
    // offset 32
    unsigned char raw[];
} LARGE_MEM_OBJ_TYPE, *LARGE_MEM_OBJ;
static_assert(sizeof(LARGE_MEM_OBJ_TYPE) % MIN_MALLOC_ALIGNMENT == 0,
              "LARGE_MEM_OBJ_TYPE is not properly aligned");

#define OBJ_LEVEL(obj) ((obj)->level)
#define OBJ_RAW(obj)   (&(obj)->raw)

#define RAW_TO_LEVEL(raw_ptr)     (*((const unsigned char*)(raw_ptr) - OBJ_PADDING - 1))
#define RAW_TO_OBJ(raw_ptr, type) container_of((raw_ptr), type, raw)

#define __SUM_OBJ_SIZE(slab_size, size) (((slab_size) + SLAB_HDR_SIZE) * (size))
#define __MIN_MEM_SIZE()                (sizeof(SLAB_AREA_TYPE))
#define __MAX_MEM_SIZE(slab_size, size) (__MIN_MEM_SIZE() + __SUM_OBJ_SIZE((slab_size), (size)))

#define __INIT_SUM_OBJ_SIZE(size) ((SLAB_LEVELS_SUM + SLAB_HDR_SIZE * SLAB_LEVEL) * (size))
#define __INIT_MIN_MEM_SIZE()     (sizeof(SLAB_MGR_TYPE) + sizeof(SLAB_AREA_TYPE) * SLAB_LEVEL)
#define __INIT_MAX_MEM_SIZE(size) (__INIT_MIN_MEM_SIZE() + __INIT_SUM_OBJ_SIZE(size))

#ifdef ALLOC_ALIGNMENT
static inline size_t size_align_down(size_t slab_size, size_t size) {
    assert(IS_POWER_OF_2(ALLOC_ALIGNMENT));
    size_t s = __MAX_MEM_SIZE(slab_size, size);
    size_t p = s - ALIGN_DOWN_POW2(s, ALLOC_ALIGNMENT);
    size_t o = __SUM_OBJ_SIZE(slab_size, 1);
    return size - p / o - (p % o ? 1 : 0);
}

static inline size_t size_align_up(size_t slab_size, size_t size) {
    assert(IS_POWER_OF_2(ALLOC_ALIGNMENT));
    size_t s = __MAX_MEM_SIZE(slab_size, size);
    size_t p = ALIGN_UP_POW2(s, ALLOC_ALIGNMENT) - s;
    size_t o = __SUM_OBJ_SIZE(slab_size, 1);
    return size + p / o;
}

static inline size_t init_align_down(size_t size) {
    assert(IS_POWER_OF_2(ALLOC_ALIGNMENT));
    size_t s = __INIT_MAX_MEM_SIZE(size);
    size_t p = s - ALIGN_DOWN_POW2(s, ALLOC_ALIGNMENT);
    size_t o = __INIT_SUM_OBJ_SIZE(1);
    return size - p / o - (p % o ? 1 : 0);
}

static inline size_t init_size_align_up(size_t size) {
    assert(IS_POWER_OF_2(ALLOC_ALIGNMENT));
    size_t s = __INIT_MAX_MEM_SIZE(size);
    size_t p = ALIGN_UP_POW2(s, ALLOC_ALIGNMENT) - s;
    size_t o = __INIT_SUM_OBJ_SIZE(1);
    return size + p / o;
}
#endif /* ALLOC_ALIGNMENT */

#ifndef STARTUP_SIZE
#define STARTUP_SIZE 16
#endif

__attribute_no_sanitize_address
static inline void __set_free_slab_area(SLAB_AREA area, SLAB_MGR mgr, int level) {
    size_t slab_size = slab_levels[level] + SLAB_HDR_SIZE;
    mgr->addr[level]        = (void*)area->raw;
    mgr->addr_top[level]    = (void*)area->raw + (area->size * slab_size);
    mgr->size[level]       += area->size;
    mgr->active_area[level] = area;
}

__attribute_no_sanitize_address
static inline SLAB_MGR create_slab_mgr(void) {
#ifdef ALLOC_ALIGNMENT
    size_t size = init_size_align_up(STARTUP_SIZE);
#else
    size_t size = STARTUP_SIZE;
#endif
    void* mem = NULL;
    SLAB_AREA area;
    SLAB_MGR mgr;

    /* If the allocation failed, always try smaller sizes */
    for (; size > 0; size >>= 1) {
        mem = system_malloc(__INIT_MAX_MEM_SIZE(size));
        if (mem)
            break;
    }

    if (!mem)
        return NULL;

    mgr = (SLAB_MGR)mem;

    void* addr = (void*)mgr + sizeof(SLAB_MGR_TYPE);
    for (size_t i = 0; i < SLAB_LEVEL; i++) {
        area       = (SLAB_AREA)addr;
        area->size = size;

        INIT_LIST_HEAD(area, __list);
        INIT_LISTP(&mgr->area_list[i]);
        LISTP_ADD_TAIL(area, &mgr->area_list[i], __list);

        INIT_LISTP(&mgr->free_list[i]);
        mgr->size[i] = 0;
        __set_free_slab_area(area, mgr, i);

        addr += __MAX_MEM_SIZE(slab_levels[i], size);
    }

    return mgr;
}

__attribute_no_sanitize_address
static inline void destroy_slab_mgr(SLAB_MGR mgr) {
    void* addr = (void*)mgr + sizeof(SLAB_MGR_TYPE);
    SLAB_AREA area, tmp, n;
    for (size_t i = 0; i < SLAB_LEVEL; i++) {
        area = (SLAB_AREA)addr;

        LISTP_FOR_EACH_ENTRY_SAFE(tmp, n, &mgr->area_list[i], __list) {
            /* very first area in the list (`area`) was allocated together with mgr, so will be
             * freed together with mgr in the system_free outside this loop */
            if (tmp != area)
                system_free(tmp, __MAX_MEM_SIZE(slab_levels[i], tmp->size));
        }

        addr += __MAX_MEM_SIZE(slab_levels[i], area->size);
    }

    system_free(mgr, addr - (void*)mgr);
}

// SYSTEM_LOCK needs to be held by the caller on entry.
__attribute_no_sanitize_address
static inline int maybe_enlarge_slab_mgr(SLAB_MGR mgr, int level) {
    assert(SYSTEM_LOCKED());
    assert(level < SLAB_LEVEL);

    while (mgr->addr[level] == mgr->addr_top[level] && LISTP_EMPTY(&mgr->free_list[level])) {
        size_t size = mgr->size[level];
        SLAB_AREA area;

        /* If there is a previously allocated area, just activate it. */
        area = LISTP_PREV_ENTRY(mgr->active_area[level], &mgr->area_list[level], __list);
        if (area) {
            __set_free_slab_area(area, mgr, level);
            return 0;
        }

        /* system_malloc() may be blocking, so we release the lock before allocating more memory */
        SYSTEM_UNLOCK();
        for (; size > 0; size >>= 1) {
            /* If the allocation failed, always try smaller sizes */
            area = (SLAB_AREA)system_malloc(__MAX_MEM_SIZE(slab_levels[level], size));
            if (area)
                break;
        }
        SYSTEM_LOCK();

        if (!area)
            return -ENOMEM;

        area->size = size;
        INIT_LIST_HEAD(area, __list);

        /* There can be concurrent operations to extend the SLAB manager. In case someone has
         * already enlarged the space, we just add the new area to the list for later use. */
        LISTP_ADD(area, &mgr->area_list[level], __list);
    }

    return 0;
}

__attribute_no_sanitize_address
static inline void* slab_alloc(SLAB_MGR mgr, size_t size) {
    SLAB_OBJ mobj;
    size_t level = -1;

    for (size_t i = 0; i < SLAB_LEVEL; i++)
        if (size <= slab_levels[i]) {
            level = i;
            break;
        }

    if (level == (size_t)-1) {
        size = ALIGN_UP_POW2(size, MIN_MALLOC_ALIGNMENT);

        LARGE_MEM_OBJ mem = (LARGE_MEM_OBJ)system_malloc(sizeof(LARGE_MEM_OBJ_TYPE) + size);
        if (!mem)
            return NULL;

        mem->size = size;
        OBJ_LEVEL(mem) = (unsigned char)-1;

#ifdef ASAN
        asan_unpoison_region((uintptr_t)OBJ_RAW(mem), size);
#endif
        return OBJ_RAW(mem);
    }

    SYSTEM_LOCK();
    assert(mgr->addr[level] <= mgr->addr_top[level]);

    int ret = maybe_enlarge_slab_mgr(mgr, level);
    if (ret < 0) {
        SYSTEM_UNLOCK();
        return NULL;
    }

    bool use_free_list;
#ifdef ASAN
    /* With ASan enabled, prefer using new memory instead of recycling already freed objects, so
     * that we have a higher chance of detecting use-after-free bugs */
    use_free_list = mgr->addr[level] == mgr->addr_top[level];
    if (use_free_list)
        assert(!LISTP_EMPTY(&mgr->free_list[level]));
#else
    use_free_list = !LISTP_EMPTY(&mgr->free_list[level]);
#endif

    if (use_free_list) {
        mobj = LISTP_FIRST_ENTRY(&mgr->free_list[level], SLAB_OBJ_TYPE, __list);
        LISTP_DEL(mobj, &mgr->free_list[level], __list);
    } else {
        mobj = (void*)mgr->addr[level];
        mgr->addr[level] += slab_levels[level] + SLAB_HDR_SIZE;
    }
    assert(mgr->addr[level] <= mgr->addr_top[level]);
    OBJ_LEVEL(mobj) = level;
    SYSTEM_UNLOCK();

#ifdef SLAB_CANARY
    unsigned long* m = (unsigned long*)((void*)OBJ_RAW(mobj) + slab_levels[level]);
    *m = SLAB_CANARY_STRING;
#endif
#ifdef ASAN
    asan_unpoison_region((uintptr_t)OBJ_RAW(mobj), size);
#endif

    return OBJ_RAW(mobj);
}

// Returns user buffer size (i.e. excluding size of control structures).
__attribute_no_sanitize_address
static inline size_t slab_get_buf_size(const void* ptr) {
    assert(ptr);

    unsigned char level = RAW_TO_LEVEL(ptr);

    if (level == (unsigned char)-1) {
        LARGE_MEM_OBJ mem = RAW_TO_OBJ(ptr, LARGE_MEM_OBJ_TYPE);
        return mem->size;
    }

    if (level >= SLAB_LEVEL) {
        log_always("Heap corruption detected: invalid heap level %u", level);
        abort();
    }

#ifdef SLAB_CANARY
    const unsigned long* m = (const unsigned long*)(ptr + slab_levels[level]);
    __UNUSED(m);
    assert(*m == SLAB_CANARY_STRING);
#endif

    return slab_levels[level];
}

__attribute_no_sanitize_address
static inline void slab_free(SLAB_MGR mgr, void* obj) {
    /* In a general purpose allocator, free of NULL is allowed (and is a
     * nop). We might want to enforce stricter rules for our allocator if
     * we're sure that no clients rely on being able to free NULL. */
    if (!obj)
        return;

    unsigned char level = RAW_TO_LEVEL(obj);

    if (level == (unsigned char)-1) {
        LARGE_MEM_OBJ mem = RAW_TO_OBJ(obj, LARGE_MEM_OBJ_TYPE);
#ifdef DEBUG
        _real_memset(obj, 0xCC, mem->size);
#endif
#ifdef ASAN
        asan_unpoison_region((uintptr_t)mem, mem->size + sizeof(LARGE_MEM_OBJ_TYPE));
#endif
        system_free(mem, mem->size + sizeof(LARGE_MEM_OBJ_TYPE));
        return;
    }

    /* If this happens, either the heap is already corrupted, or someone's
     * freeing something that's wrong, which will most likely lead to heap
     * corruption. Either way, panic if this happens. TODO: this doesn't allow
     * us to detect cases where the heap headers have been zeroed, which
     * is a common type of heap corruption. We could make this case slightly
     * more likely to be detected by adding a non-zero offset to the level,
     * so a level of 0 in the header would no longer be a valid level. */
    if (level >= SLAB_LEVEL) {
        log_always("Heap corruption detected: invalid heap level %d", level);
        abort();
    }

#ifdef SLAB_CANARY
    unsigned long* m = (unsigned long*)(obj + slab_levels[level]);
    __UNUSED(m);
    assert(*m == SLAB_CANARY_STRING);
#endif

    SLAB_OBJ mobj = RAW_TO_OBJ(obj, SLAB_OBJ_TYPE);
#ifdef DEBUG
    _real_memset(obj, 0xCC, slab_levels[level]);
#endif
#ifdef ASAN
    asan_poison_region((uintptr_t)obj, slab_levels[level], ASAN_POISON_HEAP_AFTER_FREE);
#endif

    SYSTEM_LOCK();
    INIT_LIST_HEAD(mobj, __list);
    LISTP_ADD_TAIL(mobj, &mgr->free_list[level], __list);
    SYSTEM_UNLOCK();
}

#endif /* SLABMGR_H */
