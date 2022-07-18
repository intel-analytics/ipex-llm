/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * This file defines functions for address sanitization (ASan).
 *
 * Normally, code compiled with ASan is linked against a special library (libasan), but that library
 * is hard to adapt to a no-stdlib setting as well as all the custom memory handling that we
 * perform.
 *
 * See also `ubsan.c` for a similar (but much simpler) integration with UBSan.
 *
 * For more information, see:
 *
 * - ASan documentation: https://clang.llvm.org/docs/AddressSanitizer.html
 *
 * - libasan source code in LLVM repository: https://github.com/llvm/llvm-project/
 *   (compiler-rt/lib/asan/)
 *
 * - AddressSanitizer compiler code, also in LLVM repository, for flags that we use to configure
 *   (llvm/lib/Transforms/Instrumentation)
 */

/*
 * How to use ASan:
 *
 * - Make sure the program maps the shadow memory area at startup. This will be something like:
 *
 *       mmap((void*)ASAN_SHADOW_START, ASAN_SHADOW_LENGTH, PROT_READ | PROT_WRITE,
 *            MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED,
 *            -1, 0);
 *
 * - Annotate all functions that shouldn't perform sanitized memory access with
 *   `__attribute_no_sanitize_address`.
 *
 * - Instrument your implementation of `malloc`/`free`:
 *
 *   * Make sure there is some padding before each object, to help detect overflows.
 *
 *   * In `malloc`, unpoison exactly the region requested (without rounding up the size).
 *
 *   * When freeing the allocated memory (in `free`), poison the memory region with
 *     ASAN_POISON_HEAP_AFTER_FREE.
 *
 *   * Make sure to manage mapped/unmapped pages (`system_malloc`/`system_free`). Newly mapped
 *     memory should be poisoned with ASAN_POISON_HEAP_LEFT_REDZONE, and unmapped memory should be
 *     unpoisoned before unmapping (in case ASan-unaware code uses this part of address space
 *     later).
 *
 * - Make sure to call the constructors in the `.init_array` section (see `init.h` for a helper
 *   function). This is necessary for global variables sanitization (see `__asan_register_globals`
 *   below).
 *
 * - You should compile the program with:
 *
 *       -fsanitize=address
 *       -fno-sanitize-link-runtime
 *       -mllvm -asan-mapping-offset=0x18000000000
 *       -mllvm -asan-use-after-return=0  (`never` in LLVM 13 or higher)
 *       -mllvm -asan-stack=0
 *       -DASAN
 *
 * If you want to enable stack sanitization (`-mllvm -asan-stack=1`), you need to also handle
 * unpoisoning the stack. When functions return normally, they clean up after themselves, but we
 * have cases where a function never returns (thread exit, or jump from the middle of a `noreturn`
 * function), leaving the stack memory poisoned. If the same stack memory is used later in the
 * program, it's likely that ASan will generate a false positive.
 *
 * - When a thread exits, make sure that all stack memory dedicated to this thread is unpoisoned
 *   before it's unmapped or recycled.
 *
 * - When a thread abandons the current stack, it should unpoison that stack. Typically, there will
 *   be a chain of `noreturn` functions: one of them (preferably the last) should call
 *   `asan_unpoison_current_stack`. This function (and any code it handles control to, before
 *   leaving the stack) cannot be ASan-instrumented: otherwise, it could poison the stack again.
 *
 *   This is necessary only if the jump is from the middle of a call stack. If all C functions have
 *   returned, and the jump is from a top-level assembly wrapper, there's no need for cleanup:
 *   ASan-instrumented functions poison parts of stack on entry/allocation, and unpoison these parts
 *   on return.
 *
 *   The cleanup can also be skipped if we will not be using that stack again (e.g. LibOS code
 *   abandoning the initial PAL stack).
 *
 * Note that we rely on the fact that no Gramine code runs on the user stack.
 */

#ifndef ASAN_H_
#define ASAN_H_

/* All ASan code should be guarded by `#ifdef ASAN`. */
#ifdef ASAN

#ifndef __x86_64__
/* Other systems will probably require different ASAN_SHADOW_* parameters */
#error ASan is currently supported only for x86_64
#endif

#ifndef __clang__
#error ASan is currently supported only for Clang
#endif

#include <stddef.h>
#include <stdint.h>

/*
 * Parameters of the shadow memory area. Each byte of shadow memory corresponds to ASAN_SHADOW_ALIGN
 * (by default 8) bytes of user memory. The shadow map length is 1 << 45 to cover the whole 48-bit
 * user address space: Linux sometimes maps `vvar` and `vdso` just above 0x800000000000 (= 1 << 47).
 *
 * Note that we override the address of shadow memory area (ASAN_SHADOW_START). We want the shadow
 * memory to begin at a high address, because the default for x86_64 (0x7fff8000, just before 2 GB)
 * doesn't work well with SGX: an enclave of size 2 GB or higher will be mapped over the shadow
 * memory. The same address has to be provided to the compiler using `-mllvm
 * -asan-mapping-offset=0x...`.
 *
 * (BEWARE when changing ASAN_SHADOW_START: the value should not be a power of two. For powers of
 * two, LLVM tries to optimize the generated code by emitting bitwise OR instead of addition in the
 * mem-to-shadow conversion. As a result, low values (such as 1 TB) will not work correctly. A value
 * at least as high as the shadow map length (1 << 45) should work, but it's probably better to stay
 * closer to the default configuration and not use a power of two.)
 *
 * The shadow memory bytes have the following meaning:
 *
 * - A value of 0 means all bytes are accessible.
 *
 * - A low value (01..07) means only the first N bytes are accessible.
 *
 * - A value with highest bit set (80..FF) means the memory is forbidden to use, and the exact value
 *   is used to diagnose the problem.
 */
#define ASAN_SHADOW_START 0x18000000000ULL /* 1.5 TB */
#define ASAN_SHADOW_SHIFT 3
#define ASAN_SHADOW_LENGTH (1ULL << 45)
#define ASAN_SHADOW_ALIGN (1 << ASAN_SHADOW_SHIFT)
#define ASAN_SHADOW_MASK ((1 << ASAN_SHADOW_SHIFT) - 1)

/* Conversion between user and shadow addresses */
#define ASAN_MEM_TO_SHADOW(addr) (((addr) >> ASAN_SHADOW_SHIFT) + ASAN_SHADOW_START)
#define ASAN_SHADOW_TO_MEM(addr) (((addr) - ASAN_SHADOW_START) << ASAN_SHADOW_SHIFT)

/* Magic values to mark different kinds of inaccessible memory. These are the same as in LLVM's
 * AddressSanitizer (`asan_internal.h`), because LLVM's instrumentation often writes them
 * directly instead of calling our callbacks. */
#define ASAN_POISON_HEAP_LEFT_REDZONE     0xfa
#define ASAN_POISON_HEAP_AFTER_FREE       0xfd
#define ASAN_POISON_STACK_LEFT            0xf1
#define ASAN_POISON_STACK_MID             0xf2
#define ASAN_POISON_STACK_RIGHT           0xf3
#define ASAN_POISON_STACK_AFTER_SCOPE     0xf8
#define ASAN_POISON_ALLOCA_LEFT           0xca
#define ASAN_POISON_ALLOCA_RIGHT          0xcb
#define ASAN_POISON_GLOBAL                0xf9
#define ASAN_POISON_USER                  0xf7  /* currently used for unallocated SGX memory */

/* Size of `alloca` redzone (hardcoded in LLVM: `kAllocaRzSize`); see `asan_alloca_poison` below. */
#define ASAN_ALLOCA_REDZONE_SIZE 32

/* Poison a memory region. `addr` must be aligned to ASAN_SHADOW_ALIGN, and `size` is rounded up to
 * ASAN_SHADOW_ALIGN. */
void asan_poison_region(uintptr_t addr, size_t size, uint8_t value);

/* Unpoison a memory region. `addr` must be aligned to ASAN_SHADOW_ALIGN, but `size` is treated
 * exactly. */
void asan_unpoison_region(uintptr_t addr, size_t size);

/* Unpoison current stack. The only difference between this function and `asan_unpoison_region` is
 * that it checks whether the current stack frame is within `addr .. addr + size`. If it's not, it
 * prints a warning instead. */
void asan_unpoison_current_stack(uintptr_t addr, size_t size);

/* Initialization callbacks, called in `.init_array`. In Gramine, these do nothing, and we
 * initialize ASan separately. */
void __asan_init(void);
void __asan_version_mismatch_check_v8(void);

/*
 * Load/store callbacks:
 *
 * - `load` / `store`: check if memory under given address is accessible; if not, report the error
 *   and abort
 *
 * - `report_load` / `report_store`: directly report an illegal access and abort
 *
 * For small areas, instead of generating `load` and `store` callbacks, LLVM can generate inline
 * checks for the shadow memory (and calls to `report_load` / `report_store`). This is controlled by
 * `-mllvm -asan-instrumentation-with-call-threshold=N`.
 */

#define DECLARE_ASAN_LOAD_STORE_CALLBACKS(n)                \
    void __asan_load##n(uintptr_t p);                       \
    void __asan_store##n(uintptr_t p);                      \
    void __asan_report_load##n(uintptr_t p);                \
    void __asan_report_store##n(uintptr_t p);

DECLARE_ASAN_LOAD_STORE_CALLBACKS(1)
DECLARE_ASAN_LOAD_STORE_CALLBACKS(2)
DECLARE_ASAN_LOAD_STORE_CALLBACKS(4)
DECLARE_ASAN_LOAD_STORE_CALLBACKS(8)
DECLARE_ASAN_LOAD_STORE_CALLBACKS(16)

/* Variable-size version of load/store callbacks, used for large accesses. */
void __asan_loadN(uintptr_t p, size_t size);
void __asan_storeN(uintptr_t p, size_t size);
void __asan_report_load_n(uintptr_t p, size_t size);
void __asan_report_store_n(uintptr_t p, size_t size);

/*
 * Called when exiting a function marked as no-return. In the original AddressSanitizer, unpoisons
 * the region starting from the current stack frame, until stack top.
 *
 * Our implementation is a no-op, because it's inconvenient to determine the top of the current
 * stack in a universal fashion. Instead, we rely on ASan-specific logic in Gramine to clean up the
 * right stack.
 */
void __asan_handle_no_return(void);

/*
 * Poison an area around a buffer allocated with `alloca`, using ASAN_POISON_ALLOCA_{LEFT,RIGHT}:
 *
 * - left redzone: ASAN_ALLOCA_REDZONE_SIZE bytes before `addr`
 * - right redzone: from `addr + size` to `ALIGN_UP(addr + size, ASAN_ALLOCA_REDZONE_SIZE) +
 *     ASAN_ALLOCA_REDZONE_SIZE`
 *
 * `addr` must be aligned to ASAN_ALLOCA_REDZONE_SIZE.
 */
void __asan_alloca_poison(uintptr_t addr, size_t size);

/* Unpoison the stack area from `start` to `end`. Do nothing if `start` is zero. */
void __asan_allocas_unpoison(uintptr_t start, uintptr_t end);

/*
 * Description of an instrumented global variable. See LLVM (`asan_interface_internal.h`) for
 * original definition.
 *
 * Currently, we read only `addr`, `size`, and `size_with_redzone` fields.
 */
struct __asan_global {
    uintptr_t addr; /* in LLVM: `beg` */
    size_t size;
    size_t size_with_redzone;
    const char* name;
    const char* module_name;
    size_t has_dynamic_init;
    void* location; /* in LLVM: `__asan_global_source_location* location` */
    uintptr_t odr_indicator;
};

/* Register global variables. Called in an `.init_array` constructor generated by ASan. Our
 * implementation poisons the right redzone (`size .. size_with_redzone`). */
void __asan_register_globals(struct __asan_global* globals, size_t n);

/* Unregister global variables. Called in a `.fini_array` destructor generated by ASan. Our
 * implementation does nothing (and we don't even handle `.fini_array`). */
void __asan_unregister_globals(struct __asan_global* globals, size_t n);

/* Callbacks for setting the shadow memory to specific values. As with load/store callbacks, LLVM
 * normally generates inline stores and calls these functions only for bigger areas. This is
 * controlled by `-mllvm -asan-max-inline-poisoning-size=N`. */
void __asan_set_shadow_00(uintptr_t addr, size_t size);
void __asan_set_shadow_f1(uintptr_t addr, size_t size);
void __asan_set_shadow_f2(uintptr_t addr, size_t size);
void __asan_set_shadow_f3(uintptr_t addr, size_t size);
void __asan_set_shadow_f5(uintptr_t addr, size_t size);
void __asan_set_shadow_f8(uintptr_t addr, size_t size);

/* Sanitized versions of builtin functions. Note that ASan also overrides the normal versions
 * (`memcpy` etc.) */
void* __asan_memcpy(void *dst, const void *src, size_t size);
void* __asan_memset(void *s, int c, size_t n);
void* __asan_memmove(void* dest, const void* src, size_t n);

#endif /* ASAN */

#endif /* ASAN_H */
