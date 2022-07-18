/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

#include "asan.h"
#include "enclave_ocalls.h"
#include "pal_error.h"
#include "pal_internal.h"
#include "pal_linux.h"
#include "spinlock.h"

static spinlock_t g_malloc_lock = INIT_SPINLOCK_UNLOCKED;

#define SYSTEM_LOCK()   spinlock_lock(&g_malloc_lock)
#define SYSTEM_UNLOCK() spinlock_unlock(&g_malloc_lock)
#define SYSTEM_LOCKED() spinlock_is_locked(&g_malloc_lock)

#define ALLOC_ALIGNMENT g_page_size

static inline void* __malloc(size_t size) {
    void* addr = NULL;
    int ret = ocall_mmap_untrusted(&addr, size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE,
                                   /*fd=*/-1, /*offset=*/0);
    if (ret < 0)
        return NULL;

#ifdef ASAN
    asan_poison_region((uintptr_t)addr, size, ASAN_POISON_HEAP_LEFT_REDZONE);
#endif
    return addr;
}

#define system_malloc(size) __malloc(size)

static inline void __free(void* addr, size_t size) {
#ifdef ASAN
    asan_unpoison_region((uintptr_t)addr, size);
#endif
    ocall_munmap_untrusted(addr, size);
}

#define system_free(addr, size) __free(addr, size)

#include "slabmgr.h"

static SLAB_MGR untrusted_slabmgr = NULL;

void init_untrusted_slab_mgr(void) {
    if (untrusted_slabmgr)
        return;

    untrusted_slabmgr = create_slab_mgr();
    if (!untrusted_slabmgr)
        INIT_FAIL(PAL_ERROR_NOMEM, "cannot initialize slab manager");
}

void* malloc_untrusted(size_t size) {
    return slab_alloc(untrusted_slabmgr, size);
}

void free_untrusted(void* ptr) {
    slab_free(untrusted_slabmgr, ptr);
}
