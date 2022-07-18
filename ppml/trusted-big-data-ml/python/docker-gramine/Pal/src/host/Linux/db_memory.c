/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains APIs that allocate, free or protect virtual memory.
 */

#include <asm/fcntl.h>
#include <asm/mman.h>

#include "api.h"
#include "linux_utils.h"
#include "pal.h"
#include "pal_error.h"
#include "pal_flags_conv.h"
#include "pal_internal.h"
#include "pal_linux.h"
#include "pal_linux_error.h"
#include "spinlock.h"

/* Internal-PAL memory is allocated in range [g_pal_internal_mem_addr, g_pal_internal_mem_size).
 * This range is "preloaded" (LibOS is notified that it cannot use this range), so there can be no
 * overlap between LibOS and internal-PAL allocations.
 *
 * Internal-PAL allocation is trivial: we simply increment a global pointer to the next available
 * memory region on allocations and do nothing on deallocations (and fail loudly if the limit
 * specified in the manifest is exceeded). This wastes memory, but we assume that internal-PAL
 * allocations are rare, and that PAL doesn't consume much memory anyway. In near future, we need to
 * rewrite Gramine allocation logic in PAL.
 */

static size_t g_pal_internal_mem_used = 0;
static spinlock_t g_pal_internal_mem_lock = INIT_SPINLOCK_UNLOCKED;

bool _DkCheckMemoryMappable(const void* addr, size_t size) {
    if (addr < DATA_END && addr + size > TEXT_START) {
        log_error("Address %p-%p is not mappable", addr, addr + size);
        return true;
    }
    return false;
}

int _DkVirtualMemoryAlloc(void** addr_ptr, size_t size, pal_alloc_flags_t alloc_type,
                          pal_prot_flags_t prot) {
    assert(WITHIN_MASK(alloc_type, PAL_ALLOC_MASK));
    assert(WITHIN_MASK(prot,       PAL_PROT_MASK));

    void* addr = *addr_ptr;

    if (alloc_type & PAL_ALLOC_INTERNAL) {
        size = ALIGN_UP(size, g_page_size);
        spinlock_lock(&g_pal_internal_mem_lock);
        if (size > g_pal_internal_mem_size - g_pal_internal_mem_used) {
            /* requested PAL-internal allocation would exceed the limit, fail */
            spinlock_unlock(&g_pal_internal_mem_lock);
            return -PAL_ERROR_NOMEM;
        }
        addr = g_pal_internal_mem_addr + g_pal_internal_mem_used;
        g_pal_internal_mem_used += size;
        assert(IS_ALIGNED(g_pal_internal_mem_used, g_page_size));
        spinlock_unlock(&g_pal_internal_mem_lock);
    }

    assert(addr);

    int flags = PAL_MEM_FLAGS_TO_LINUX(alloc_type, prot | PAL_PROT_WRITECOPY);
    int linux_prot = PAL_PROT_TO_LINUX(prot);

    flags |= MAP_ANONYMOUS | MAP_FIXED;
    addr = (void*)DO_SYSCALL(mmap, addr, size, linux_prot, flags, -1, 0);

    if (IS_PTR_ERR(addr)) {
        /* note that we don't undo operations on `g_pal_internal_mem_used` in case of internal-PAL
         * allocations: this could lead to data races, so we just waste some memory on errors */
        return unix_to_pal_error(PTR_TO_ERR(addr));
    }

    *addr_ptr = addr;
    return 0;
}

int _DkVirtualMemoryFree(void* addr, size_t size) {
    int ret = DO_SYSCALL(munmap, addr, size);
    return ret < 0 ? unix_to_pal_error(ret) : 0;
}

int _DkVirtualMemoryProtect(void* addr, size_t size, pal_prot_flags_t prot) {
    int ret = DO_SYSCALL(mprotect, addr, size, PAL_PROT_TO_LINUX(prot));
    return ret < 0 ? unix_to_pal_error(ret) : 0;
}

static int read_proc_meminfo(const char* key, unsigned long* val) {
    int fd = DO_SYSCALL(open, "/proc/meminfo", O_RDONLY, 0);

    if (fd < 0)
        return -PAL_ERROR_DENIED;

    char buffer[40];
    int ret = 0;
    size_t n;
    size_t r = 0;
    size_t len = strlen(key);

    ret = -PAL_ERROR_DENIED;
    while (1) {
        ret = DO_SYSCALL(read, fd, buffer + r, 40 - r);
        if (ret < 0) {
            ret = -PAL_ERROR_DENIED;
            break;
        }

        for (n = r; n < r + ret; n++)
            if (buffer[n] == '\n')
                break;

        r += ret;
        if (n == r + ret || n <= len) {
            ret = -PAL_ERROR_INVAL;
            break;
        }

        if (!memcmp(key, buffer, len) && buffer[len] == ':') {
            for (size_t i = len + 1; i < n; i++)
                if (buffer[i] != ' ') {
                    *val = atol(buffer + i);
                    break;
                }
            ret = 0;
            break;
        }

        memmove(buffer, buffer + n + 1, r - n - 1);
        r -= n + 1;
    }

    DO_SYSCALL(close, fd);
    return ret;
}

unsigned long _DkMemoryQuota(void) {
    if (g_pal_linux_state.memory_quota == (unsigned long)-1)
        return 0;

    if (g_pal_linux_state.memory_quota)
        return g_pal_linux_state.memory_quota;

    unsigned long quota = 0;
    if (read_proc_meminfo("MemTotal", &quota) < 0) {
        g_pal_linux_state.memory_quota = (unsigned long)-1;
        return 0;
    }

    return (g_pal_linux_state.memory_quota = quota * 1024);
}

unsigned long _DkMemoryAvailableQuota(void) {
    unsigned long quota = 0;
    if (read_proc_meminfo("MemFree", &quota) < 0)
        return 0;
    return quota * 1024;
}

struct parsed_ranges {
    uintptr_t vdso_start;
    uintptr_t vdso_end;
    uintptr_t vvar_start;
    uintptr_t vvar_end;
};

static int parsed_ranges_callback(struct proc_maps_range* r, void* arg) {
    struct parsed_ranges* ranges = arg;

    if (r->name) {
        if (!strcmp(r->name, "[vdso]")) {
            ranges->vdso_start = r->start;
            ranges->vdso_end = r->end;
        } else if (!strcmp(r->name, "[vvar]")) {
            ranges->vvar_start = r->start;
            ranges->vvar_end = r->end;
        }
    }

    return 0;
}

int get_vdso_and_vvar_ranges(uintptr_t* vdso_start, uintptr_t* vdso_end, uintptr_t* vvar_start,
                             uintptr_t* vvar_end) {

    struct parsed_ranges ranges = {0};
    int ret = parse_proc_maps("/proc/self/maps", &parsed_ranges_callback, &ranges);
    if (ret < 0)
        return unix_to_pal_error(ret);

    *vdso_start = ranges.vdso_start;
    *vdso_end = ranges.vdso_end;
    *vvar_start = ranges.vvar_start;
    *vvar_end = ranges.vvar_end;
    return 0;
}
