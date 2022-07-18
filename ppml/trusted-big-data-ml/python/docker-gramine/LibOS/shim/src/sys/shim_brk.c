/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University
 * Copyright (C) 2020 Invisible Things Lab
 */

/*
 * Implementation of system call "brk".
 */

#include <sys/mman.h>

#include "pal.h"
#include "shim_checkpoint.h"
#include "shim_internal.h"
#include "shim_lock.h"
#include "shim_table.h"
#include "shim_utils.h"
#include "shim_vma.h"
#include "toml_utils.h"

static struct {
    size_t data_segment_size;
    char* brk_start;
    char* brk_current;
    char* brk_end;
} brk_region;

static struct shim_lock brk_lock;

int init_brk_region(void* brk_start, size_t data_segment_size) {
    int ret;

    if (!create_lock(&brk_lock)) {
        log_error("Creating brk_lock failed!");
        return -ENOMEM;
    }

    /* TODO: this needs a better fix. Currently after fork, in the new child process, `shim_init`
     * is run, hence this function too - but forked process will get its brk from checkpoints. */
    if (brk_region.brk_start) {
        return 0;
    }

    data_segment_size = ALLOC_ALIGN_UP(data_segment_size);

    assert(g_manifest_root);
    size_t brk_max_size;
    ret = toml_sizestring_in(g_manifest_root, "sys.brk.max_size", DEFAULT_BRK_MAX_SIZE,
                             &brk_max_size);
    if (ret < 0) {
        log_error("Cannot parse 'sys.brk.max_size'");
        return -EINVAL;
    }

    if (brk_start && !IS_ALLOC_ALIGNED_PTR(brk_start)) {
        log_error("Starting brk address is not aligned!");
        return -EINVAL;
    }
    if (!IS_ALLOC_ALIGNED(brk_max_size)) {
        log_error("Max brk size is not aligned!");
        return -EINVAL;
    }

    if (brk_start && brk_start <= g_pal_public_state->user_address_end
            && brk_max_size <= (uintptr_t)g_pal_public_state->user_address_end
            && (uintptr_t)brk_start < (uintptr_t)g_pal_public_state->user_address_end - brk_max_size) {
        int ret;
        size_t offset = 0;

        if (!g_pal_public_state->disable_aslr) {
            ret = DkRandomBitsRead(&offset, sizeof(offset));
            if (ret < 0) {
                return pal_to_unix_errno(ret);
            }
            /* Linux randomizes brk at offset from 0 to 0x2000000 from main executable data section
             * https://elixir.bootlin.com/linux/v5.6.3/source/arch/x86/kernel/process.c#L914 */
            offset %= MIN((size_t)0x2000000, (size_t)((char*)g_pal_public_state->user_address_end -
                                                      brk_max_size - (char*)brk_start));
            offset = ALLOC_ALIGN_DOWN(offset);
        }

        brk_start = (char*)brk_start + offset;

        ret = bkeep_mmap_fixed(brk_start, brk_max_size, PROT_NONE,
                               MAP_FIXED_NOREPLACE | VMA_UNMAPPED, NULL, 0, "heap");
        if (ret == -EEXIST) {
            /* Let's try mapping brk anywhere. */
            brk_start = NULL;
            ret = 0;
        }
        if (ret < 0) {
            return ret;
        }
    } else {
        /* Let's try mapping brk anywhere. */
        brk_start = NULL;
    }

    if (!brk_start) {
        int ret;
        ret = bkeep_mmap_any_aslr(brk_max_size, PROT_NONE, VMA_UNMAPPED, NULL, 0, "heap",
                                  &brk_start);
        if (ret < 0) {
            return ret;
        }
    }

    brk_region.brk_start         = brk_start;
    brk_region.brk_current       = brk_region.brk_start;
    brk_region.brk_end           = (char*)brk_start + brk_max_size;
    brk_region.data_segment_size = data_segment_size;

    set_rlimit_cur(RLIMIT_DATA, brk_max_size + data_segment_size);

    return 0;
}

void reset_brk(void) {
    lock(&brk_lock);

    void* tmp_vma = NULL;
    size_t allocated_size = ALLOC_ALIGN_UP_PTR(brk_region.brk_current) - brk_region.brk_start;
    if (bkeep_munmap(brk_region.brk_start, brk_region.brk_end - brk_region.brk_start,
                     /*is_internal=*/false, &tmp_vma) < 0) {
        BUG();
    }

    if (allocated_size > 0) {
        if (DkVirtualMemoryFree(brk_region.brk_start, allocated_size) < 0) {
            BUG();
        }
    }
    bkeep_remove_tmp_vma(tmp_vma);

    brk_region.brk_start         = NULL;
    brk_region.brk_current       = NULL;
    brk_region.brk_end           = NULL;
    brk_region.data_segment_size = 0;
    unlock(&brk_lock);

    destroy_lock(&brk_lock);
}

void* shim_do_brk(void* _brk) {
    char* brk = _brk;
    size_t size = 0;
    char* brk_aligned = ALLOC_ALIGN_UP_PTR(brk);

    lock(&brk_lock);

    char* brk_current = ALLOC_ALIGN_UP_PTR(brk_region.brk_current);

    if (brk < brk_region.brk_start) {
        goto out;
    } else if (brk <= brk_current) {
        size = brk_current - brk_aligned;

        if (size) {
            if (bkeep_mmap_fixed(brk_aligned, brk_region.brk_end - brk_aligned, PROT_NONE,
                                 MAP_FIXED | VMA_UNMAPPED, NULL, 0, "heap")) {
                goto out;
            }

            if (DkVirtualMemoryFree(brk_aligned, size) < 0) {
                BUG();
            }
        }

        brk_region.brk_current = brk;
        goto out;
    } else if (brk > brk_region.brk_end) {
        goto out;
    }

    uint64_t rlim_data = get_rlimit_cur(RLIMIT_DATA);
    size = brk_aligned - brk_region.brk_start;

    if (rlim_data < brk_region.data_segment_size
            || rlim_data - brk_region.data_segment_size < size) {
        goto out;
    }

    size = brk_aligned - brk_current;
    /* brk_aligned >= brk > brk_current */
    assert(size);

    if (bkeep_mmap_fixed(brk_current, size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, NULL, 0, "heap") < 0) {
        goto out;
    }

    int ret = DkVirtualMemoryAlloc((void**)&brk_current, size, 0, PAL_PROT_READ | PAL_PROT_WRITE);
    if (ret < 0) {
        if (bkeep_mmap_fixed(brk_current, brk_region.brk_end - brk_current, PROT_NONE,
                             MAP_FIXED | VMA_UNMAPPED, NULL, 0, "heap") < 0) {
            BUG();
        }
        goto out;
    }

    brk_region.brk_current = brk;

out:
    brk = brk_region.brk_current;
    unlock(&brk_lock);
    return brk;
}

BEGIN_CP_FUNC(brk) {
    __UNUSED(obj);
    __UNUSED(size);
    __UNUSED(objp);
    ADD_CP_FUNC_ENTRY((uintptr_t)brk_region.brk_start);
    ADD_CP_ENTRY(SIZE, brk_region.brk_current - brk_region.brk_start);
    ADD_CP_ENTRY(SIZE, brk_region.brk_end - brk_region.brk_start);
    ADD_CP_ENTRY(SIZE, brk_region.data_segment_size);
}
END_CP_FUNC(brk)

BEGIN_RS_FUNC(brk) {
    __UNUSED(rebase);
    brk_region.brk_start         = (char*)GET_CP_FUNC_ENTRY();
    brk_region.brk_current       = brk_region.brk_start + GET_CP_ENTRY(SIZE);
    brk_region.brk_end           = brk_region.brk_start + GET_CP_ENTRY(SIZE);
    brk_region.data_segment_size = GET_CP_ENTRY(SIZE);
}
END_RS_FUNC(brk)
