/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * SGX profiling. This code takes samples of running code and writes them out to a perf.data file
 * (see also sgx_perf_data.c).
 */

#ifdef DEBUG

#define USE_STDLIB

#include <assert.h>
#include <errno.h>
#include <linux/fs.h>
#include <linux/limits.h>
#include <stddef.h>

#include "cpu.h"
#include "elf/elf.h"
#include "linux_utils.h"
#include "sgx_internal.h"
#include "sgx_log.h"
#include "sgx_tls.h"
#include "spinlock.h"

#ifdef SGX_VTUNE_PROFILE
#include "ittnotify.h" // for __itt_module_load(...) which is defined as a macro
#endif

extern bool g_vtune_profile_enabled;

// FIXME: this is glibc realpath, declared here because the headers will conflict with PAL
char* realpath(const char* path, char* resolved_path);

#define NSEC_IN_SEC 1000000000

static spinlock_t g_perf_data_lock = INIT_SPINLOCK_UNLOCKED;
static struct perf_data* g_perf_data = NULL;

static bool g_profile_enabled = false;
static int g_profile_mode;
static uint64_t g_profile_period;
static int g_mem_fd = -1;

/* Read memory from inside enclave (using /proc/self/mem). */
static ssize_t debug_read(void* dest, void* addr, size_t size) {
    ssize_t ret;
    size_t total = 0;

    while (total < size) {
        ret = DO_SYSCALL(pread64, g_mem_fd, (uint8_t*)dest + total, size - total,
                         (off_t)addr + total);

        if (ret == -EINTR)
            continue;

        if (ret < 0)
            return ret;

        if (ret == 0)
            break;

        assert(ret > 0);
        assert((size_t)ret + total <= size);
        total += ret;
    }
    return total;
}

static int debug_read_all(void* dest, void* addr, size_t size) {
    ssize_t ret = debug_read(dest, addr, size);
    if (ret < 0)
        return ret;
    if ((size_t)ret < size)
        return -EINVAL;
    return 0;
}

static int get_sgx_gpr(sgx_pal_gpr_t* gpr, void* tcs) {
    int ret;
    uint64_t ossa;
    uint32_t cssa;
    ret = debug_read_all(&ossa, tcs + 16, sizeof(ossa));
    if (ret < 0)
        return ret;
    ret = debug_read_all(&cssa, tcs + 24, sizeof(cssa));
    if (ret < 0)
        return ret;

    void* gpr_addr = (void*)(
        g_pal_enclave.baseaddr
        + ossa + cssa * g_pal_enclave.ssa_frame_size
        - sizeof(*gpr));

    ret = debug_read_all(gpr, gpr_addr, sizeof(*gpr));
    if (ret < 0)
        return ret;

    return 0;
}

int sgx_profile_init(void) {
    int ret;

    assert(!g_profile_enabled);
    assert(g_mem_fd == -1);
    assert(!g_perf_data);

    g_profile_period = NSEC_IN_SEC / g_pal_enclave.profile_frequency;
    g_profile_mode = g_pal_enclave.profile_mode;

    ret = DO_SYSCALL(open, "/proc/self/mem", O_RDONLY | O_LARGEFILE, 0);
    if (ret < 0) {
        log_error("sgx_profile_init: opening /proc/self/mem failed: %d", ret);
        goto out;
    }
    g_mem_fd = ret;

    struct perf_data* pd = pd_open(g_pal_enclave.profile_filename, g_pal_enclave.profile_with_stack);
    if (!pd) {
        log_error("sgx_profile_init: pd_open failed");
        ret = -EINVAL;
        goto out;
    }
    g_perf_data = pd;

    ret = pd_event_command(pd, "pal-sgx", g_host_pid, /*tid=*/g_host_pid);
    if (!pd) {
        log_error("sgx_profile_init: reporting command failed: %d", ret);
        goto out;
    }

    g_profile_enabled = true;
    return 0;

out:
    if (g_mem_fd > 0) {
        int close_ret = DO_SYSCALL(close, g_mem_fd);
        if (close_ret < 0)
            log_error("sgx_profile_init: closing /proc/self/mem failed: %d", close_ret);
        g_mem_fd = -1;
    }

    if (g_perf_data) {
        ssize_t close_ret = pd_close(g_perf_data);
        if (close_ret < 0)
            log_error("sgx_profile_init: pd_close failed: %ld", close_ret);
        g_perf_data = NULL;
    }
    return ret;
}

void sgx_profile_finish(void) {
    int ret;
    ssize_t size;

    if (!g_profile_enabled)
        return;

    spinlock_lock(&g_perf_data_lock);

    size = pd_close(g_perf_data);
    if (size < 0)
        log_error("sgx_profile_finish: pd_close failed: %ld", size);
    g_perf_data = NULL;

    spinlock_unlock(&g_perf_data_lock);

    ret = DO_SYSCALL(close, g_mem_fd);
    if (ret < 0)
        log_error("sgx_profile_finish: closing /proc/self/mem failed: %d", ret);
    g_mem_fd = -1;

    log_debug("Profile data written to %s (%lu bytes)", g_pal_enclave.profile_filename, size);

    g_profile_enabled = false;
}

static void sample_simple(uint64_t rip) {
    int ret;

    spinlock_lock(&g_perf_data_lock);
    // Report all events as the same PID so that they are grouped in report.
    ret = pd_event_sample_simple(g_perf_data, rip, g_host_pid, /*tid=*/g_host_pid,
                                 g_profile_period);
    spinlock_unlock(&g_perf_data_lock);

    if (ret < 0) {
        log_error("error recording sample: %d", ret);
    }
}

static void sample_stack(sgx_pal_gpr_t* gpr) {
    int ret;

    uint8_t stack[PD_STACK_SIZE];
    size_t stack_size;
    ret = debug_read(stack, (void*)gpr->rsp, sizeof(stack));
    if (ret < 0) {
        log_error("error reading stack: %d", ret);
        return;
    }
    stack_size = ret;

    spinlock_lock(&g_perf_data_lock);
    // Report all events as the same PID so that they are grouped in report.
    ret = pd_event_sample_stack(g_perf_data, gpr->rip, g_host_pid, /*tid=*/g_host_pid,
                                g_profile_period, gpr, stack, stack_size);
    spinlock_unlock(&g_perf_data_lock);

    if (ret < 0) {
        log_error("error recording sample: %d", ret);
    }
}

/*
 * Update sample time, and return true if we should record a new sample.
 *
 * In case of AEX, we can use it to record a sample approximately every 'g_profile_period'
 * nanoseconds. Note that we rely on Linux scheduler to generate an AEX event 250 times per second
 * (although other events may cause an AEX to happen more often), so sampling frequency greater than
 * 250 cannot be reliably achieved.
 */
static bool update_time(void) {
    // Check current CPU time
    struct timespec ts;
    int ret = DO_SYSCALL(clock_gettime, CLOCK_THREAD_CPUTIME_ID, &ts);
    if (ret < 0) {
        log_error("sgx_profile_sample: clock_gettime failed: %d", ret);
        return false;
    }
    uint64_t sample_time = ts.tv_sec * NSEC_IN_SEC + ts.tv_nsec;

    // Compare and update last recorded time per thread
    PAL_TCB_URTS* tcb = get_tcb_urts();
    if (tcb->profile_sample_time == 0) {
        tcb->profile_sample_time = sample_time;
        return false;
    }

    if (sample_time - tcb->profile_sample_time >= g_profile_period) {
        tcb->profile_sample_time = sample_time;
        assert(sample_time >= tcb->profile_sample_time);
        return true;
    }
    return false;
}

void sgx_profile_sample_aex(void* tcs) {
    int ret;

    if (!(g_profile_enabled && g_profile_mode == SGX_PROFILE_MODE_AEX))
        return;

    if (!update_time())
        return;

    sgx_pal_gpr_t gpr;
    ret = get_sgx_gpr(&gpr, tcs);
    if (ret < 0) {
        log_error("sgx_profile_sample_aex: error reading GPR: %d", ret);
        return;
    }

    if (g_pal_enclave.profile_with_stack) {
        sample_stack(&gpr);
    } else {
        sample_simple(gpr.rip);
    }
}

void sgx_profile_sample_ocall_inner(void* enclave_gpr) {
    int ret;

    if (!(g_profile_enabled && g_profile_mode == SGX_PROFILE_MODE_OCALL_INNER))
        return;

    if (!enclave_gpr)
        return;

    sgx_pal_gpr_t gpr;
    ret = debug_read_all(&gpr, enclave_gpr, sizeof(gpr));
    if (ret < 0) {
        log_error("sgx_profile_sample_ocall_inner: error reading GPR: %d", ret);
        return;
    }

    if (g_pal_enclave.profile_with_stack) {
        sample_stack(&gpr);
    } else {
        sample_simple(gpr.rip);
    }
}

void sgx_profile_sample_ocall_outer(void* ocall_func) {
    if (!(g_profile_enabled && g_profile_mode == SGX_PROFILE_MODE_OCALL_OUTER))
        return;

    assert(ocall_func);
    assert(!g_pal_enclave.profile_with_stack);
    sample_simple((uint64_t)ocall_func);
}

void sgx_profile_report_elf(const char* filename, void* addr) {
    int ret;

    if (!g_profile_enabled && !g_vtune_profile_enabled)
        return;

    if (!strcmp(filename, ""))
        filename = get_main_exec_path();

    if (!strcmp(filename, "linux-vdso.so.1"))
        return;

    // Convert filename to absolute path - some tools (e.g. libunwind in 'perf report') refuse to
    // process relative paths.
    char buf[PATH_MAX];
    char* path = realpath(filename, buf);
    if (!path) {
        log_error("sgx_profile_report_elf(%s): realpath failed", filename);
        return;
    }

    // Open the file and mmap it.

    int fd = DO_SYSCALL(open, path, O_RDONLY, 0);
    if (fd < 0) {
        log_error("sgx_profile_report_elf(%s): open failed: %d", filename, fd);
        return;
    }

    off_t elf_length = DO_SYSCALL(lseek, fd, 0, SEEK_END);
    if (elf_length < 0) {
        log_error("sgx_profile_report_elf(%s): lseek failed: %ld", filename, elf_length);
        goto out_close;
    }

    void* elf_addr = (void*)DO_SYSCALL(mmap, NULL, elf_length, PROT_READ, MAP_PRIVATE, fd, 0);
    if (IS_PTR_ERR(elf_addr)) {
        log_error("sgx_profile_report_elf(%s): mmap failed: %ld", filename, PTR_TO_ERR(addr));
        goto out_close;
    }

    // Perform a simple sanity check to verify if this looks like ELF (see TODO for DkDebugMapAdd in
    // Pal/src/db_rtld.c).

    const elf_ehdr_t* ehdr = elf_addr;

    if (elf_length < (off_t)sizeof(*ehdr) || memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0) {
        log_error("sgx_profile_report_elf(%s): invalid ELF binary", filename);
        goto out_unmap;
    }

    // Read the program headers and record mmap events for the segments that should be mapped as
    // executable.

    const elf_phdr_t* phdr = (const elf_phdr_t*)((uintptr_t)elf_addr + ehdr->e_phoff);
    ret = 0;

    spinlock_lock(&g_perf_data_lock);

    for (unsigned int i = 0; i < ehdr->e_phnum; i++) {
        if (phdr[i].p_type == PT_LOAD && phdr[i].p_flags & PF_X) {
            uint64_t mapstart = ALLOC_ALIGN_DOWN(phdr[i].p_vaddr);
            uint64_t mapend = ALLOC_ALIGN_UP(phdr[i].p_vaddr + phdr[i].p_filesz);
            uint64_t offset = ALLOC_ALIGN_DOWN(phdr[i].p_offset);
            if (g_profile_enabled) {
                ret = pd_event_mmap(g_perf_data, path, g_host_pid,
                        (uint64_t)addr + mapstart, mapend - mapstart, offset);
                if (ret < 0)
                    break;
            }
#ifdef SGX_VTUNE_PROFILE
            if (g_vtune_profile_enabled)
                __itt_module_load((void*)addr + mapstart, (void*)addr + mapend - 1, path);
#endif
        }
    }

    spinlock_unlock(&g_perf_data_lock);

    if (ret < 0)
        log_error("sgx_profile_report_elf(%s): pd_event_mmap failed: %d", filename, ret);

    // Clean up.

out_unmap:
    ret = DO_SYSCALL(munmap, elf_addr, elf_length);
    if (ret < 0)
        log_error("sgx_profile_report_elf(%s): munmap failed: %d", filename, ret);

out_close:
    ret = DO_SYSCALL(close, fd);
    if (ret < 0)
        log_error("sgx_profile_report_elf(%s): close failed: %d", filename, ret);
}

#endif /* DEBUG */
