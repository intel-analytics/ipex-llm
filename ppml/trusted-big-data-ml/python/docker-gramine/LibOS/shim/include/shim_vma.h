/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University
 * Copyright (C) 2020 Invisible Things Lab
 */

/*
 * Definitions of types and functions for VMA bookkeeping.
 */

#ifndef _SHIM_VMA_H_
#define _SHIM_VMA_H_

#include <linux/mman.h>
#include <stdbool.h>
#include <stdint.h>

#include "api.h"
#include "pal.h"
#include "shim_defs.h"
#include "shim_handle.h"
#include "shim_types.h"

#define VMA_COMMENT_LEN 16

/* Public version of shim_vma, used when we want to copy out the VMA and use it without holding
 * the VMA list lock. */
struct shim_vma_info {
    void* addr;
    size_t length;
    int prot;  // memory protection flags: PROT_*
    int flags; // MAP_* and VMA_*
    struct shim_handle* file;
    uint64_t file_offset;
    char comment[VMA_COMMENT_LEN];
};

/* MAP_FIXED_NOREPLACE and MAP_SHARED_VALIDATE are fairly new and might not be defined. */
#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif // MAP_FIXED_NOREPLACE
#ifndef MAP_SHARED_VALIDATE
#define MAP_SHARED_VALIDATE 0x03
#endif // MAP_SHARED_VALIDATE

/* vma is kept for bookkeeping, but the memory is not actually allocated */
#define VMA_UNMAPPED 0x10000000
/* vma is used internally */
#define VMA_INTERNAL 0x20000000
/* vma is backed by a file and has been protected as writable, so it has to be checkpointed during
 * migration */
#define VMA_TAINTED 0x40000000

int init_vma(void);

/*
 * Bookkeeping a removal of mapped memory. On success returns a temporary VMA pointer in
 * `tmp_vma_ptr`, which must be subsequently freed by calling `bkeep_remove_tmp_vma` - but this
 * should be done only *AFTER* the memory deallocation itself. For example:
 *
 * void* tmp_vma = NULL;
 * if (bkeep_munmap(ptr, len, is_internal, &tmp_vma) < 0) {
 *     handle_errors();
 * }
 * if (DkVirtualMemoryFree(ptr, len) < 0) {
 *     handle_errors();
 * }
 * bkeep_remove_tmp_vma(tmp_vma);
 *
 * Such a way of freeing is needed, so that no other thread will map the same memory in the window
 * between `bkeep_munmap` and `DkVirtualMemoryFree`.
 */
int bkeep_munmap(void* addr, size_t length, bool is_internal, void** tmp_vma_ptr);
void bkeep_remove_tmp_vma(void* vma);

/* Bookkeeping a change to memory protections. */
int bkeep_mprotect(void* addr, size_t length, int prot, bool is_internal);

/*
 * Bookkeeping an allocation of memory at a fixed address. `flags` must contain either MAP_FIXED or
 * MAP_FIXED_NOREPLACE - the former forces bookkeeping and removes any overlapping VMAs, the latter
 * atomically checks for overlaps and fails if one is found.
 */
int bkeep_mmap_fixed(void* addr, size_t length, int prot, int flags, struct shim_handle* file,
                     uint64_t offset, const char* comment);

/*
 * Bookkeeping an allocation of memory at any address in the range [`bottom_addr`, `top_addr`).
 * The search is top-down, starting from `top_addr` - `length` and returning the first unoccupied
 * area capable of fitting the requested size.
 * Start of bookkept range is returned in `*ret_val_ptr`.
 */
int bkeep_mmap_any_in_range(void* bottom_addr, void* top_addr, size_t length, int prot, int flags,
                            struct shim_handle* file, uint64_t offset, const char* comment,
                            void** ret_val_ptr);

/* Shorthand for `bkeep_mmap_any_in_range` with the range
 * [`g_pal_public_state->user_address_start`, `g_pal_public_state->user_address_end`). */
int bkeep_mmap_any(size_t length, int prot, int flags, struct shim_handle* file, uint64_t offset,
                   const char* comment, void** ret_val_ptr);

/* First tries to bookkeep in the range [`g_pal_public_state->user_address_start`, `aslr_addr_top`)
 * and if it fails calls `bkeep_mmap_any`. `aslr_addr_top` is a value randomized on each program
 * run. */
int bkeep_mmap_any_aslr(size_t length, int prot, int flags, struct shim_handle* file,
                        uint64_t offset, const char* comment, void** ret_val_ptr);

/* Looking up VMA that contains `addr`. If one is found, returns its description in `vma_info`.
 * This function increases ref-count of `vma_info->file` by one (if it is not NULL). */
int lookup_vma(void* addr, struct shim_vma_info* vma_info);

/* Returns true if the whole range [`addr`, `addr` + `length`) is mapped as user memory and allows
 * for `prot` type of access. */
bool is_in_adjacent_user_vmas(const void* addr, size_t length, int prot);

/*
 * Dumps all non-internal and mapped VMAs.
 * On success returns 0 and puts the pointer to result array into `*vma_infos` and its length into
 * `*count`. On error returns negated error code.
 * The returned array can be subsequently freed by `free_vma_info_array`.
 */
int dump_all_vmas(struct shim_vma_info** vma_infos, size_t* count, bool include_unmapped);
void free_vma_info_array(struct shim_vma_info* vma_infos, size_t count);

/* Implementation of madvise(MADV_DONTNEED) syscall */
int madvise_dontneed_range(uintptr_t begin, uintptr_t end);

/* Call `msync` for file mappings in given range (should be page-aligned) */
int msync_range(uintptr_t begin, uintptr_t end);

/* Call `msync` for file mappings of `hdl` */
int msync_handle(struct shim_handle* hdl);

void debug_print_all_vmas(void);

#endif /* _SHIM_VMA_H_ */
