/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file implements page allocation for the library OS-internal SLAB memory allocator. The slab
 * allocator is in common/include/slabmgr.h.
 *
 * When existing slabs are not sufficient, or a large (4k or greater) allocation is requested, it
 * ends up here (__system_alloc and __system_free).
 */

#include <asm/mman.h>

#include "asan.h"
#include "pal.h"
#include "shim_internal.h"
#include "shim_lock.h"
#include "shim_utils.h"
#include "shim_vma.h"

static struct shim_lock slab_mgr_lock;

#define SYSTEM_LOCK()   lock(&slab_mgr_lock)
#define SYSTEM_UNLOCK() unlock(&slab_mgr_lock)
#define SYSTEM_LOCKED() locked(&slab_mgr_lock)

#define SLAB_CANARY
#define STARTUP_SIZE 16

#include "slabmgr.h"

static SLAB_MGR slab_mgr = NULL;

/* Returns NULL on failure */
void* __system_malloc(size_t size) {
    size_t alloc_size = ALLOC_ALIGN_UP(size);
    void* addr = NULL;

    int ret = bkeep_mmap_any(alloc_size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS | VMA_INTERNAL, NULL, 0, "slab", &addr);
    if (ret < 0) {
        return NULL;
    }

    ret = DkVirtualMemoryAlloc(&addr, alloc_size, 0, PAL_PROT_WRITE | PAL_PROT_READ);
    if (ret < 0) {
        log_error("failed to allocate memory (%ld)", pal_to_unix_errno(ret));
        void* tmp_vma = NULL;
        if (bkeep_munmap(addr, alloc_size, /*is_internal=*/true, &tmp_vma) < 0) {
            BUG();
        }
        bkeep_remove_tmp_vma(tmp_vma);
        return NULL;
    }

#ifdef ASAN
    asan_poison_region((uintptr_t)addr, alloc_size, ASAN_POISON_HEAP_LEFT_REDZONE);
#endif
    return addr;
}

void __system_free(void* addr, size_t size) {
    void* tmp_vma = NULL;
    if (bkeep_munmap(addr, ALLOC_ALIGN_UP(size), /*is_internal=*/true, &tmp_vma) < 0) {
        BUG();
    }
    if (DkVirtualMemoryFree(addr, ALLOC_ALIGN_UP(size)) < 0) {
        BUG();
    }
#ifdef ASAN
    asan_unpoison_region((uintptr_t)addr, ALLOC_ALIGN_UP(size));
#endif
    bkeep_remove_tmp_vma(tmp_vma);
}

int init_slab(void) {
    if (!create_lock(&slab_mgr_lock)) {
        return -ENOMEM;
    }
    slab_mgr = create_slab_mgr();
    if (!slab_mgr) {
        return -ENOMEM;
    }
    return 0;
}

void* malloc(size_t size) {
    void* mem = slab_alloc(slab_mgr, size);

    if (!mem) {
        /*
         * Normally, the library OS should not run out of memory.
         * If malloc() failed internally, we cannot handle the
         * condition and must terminate the current process.
         */
        log_error("Out-of-memory in library OS");
        DkProcessExit(1);
    }

    return mem;
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

#if 0
void* realloc(void* ptr, size_t new_size) {
    /* TODO: decide how to deal with migrated-memory case */
    assert(!memory_migrated(ptr));

    size_t old_size = slab_get_buf_size(slab_mgr, ptr);

    /* TODO: this realloc() implementation follows the Glibc design, which will avoid reallocation
     *       when the buffer is large enough. Potentially this design can cause memory draining if
     *       user resizes an extremely large object to much smaller. */
    if (old_size >= new_size)
        return ptr;

    void* new_buf = malloc(new_size);
    if (!new_buf)
        return NULL;

    /* realloc() does not zero the rest of the object */
    memcpy(new_buf, ptr, old_size);

    free(ptr);
    return new_buf;
}
#endif

// Copies data from `mem` to a newly allocated buffer of a specified size.
void* malloc_copy(const void* mem, size_t size) {
    void* buff = malloc(size);
    if (buff)
        memcpy(buff, mem, size);
    return buff;
}

void free(void* mem) {
    if (memory_migrated(mem)) {
        return;
    }

    slab_free(slab_mgr, mem);
}
