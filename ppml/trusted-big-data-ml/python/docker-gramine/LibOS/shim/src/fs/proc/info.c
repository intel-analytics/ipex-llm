/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*!
 * \file
 *
 * This file contains the implementation of `/proc/meminfo` and `/proc/cpuinfo`.
 */

#include "shim_fs_pseudo.h"

int proc_meminfo_load(struct shim_dentry* dent, char** out_data, size_t* out_size) {
    __UNUSED(dent);

    size_t size, max = 128;
    char* str = NULL;

    assert(g_pal_public_state->mem_total >= DkMemoryAvailableQuota());

    /*
     * Enumerate minimal set of meminfo stats; as reference workloads that use these stats, we use:
     *
     *   - Python's psutil library
     *     https://github.com/giampaolo/psutil/blob/f716afc8/psutil/_pslinux.py#L339-L568
     *
     *   - Rust's procfs crate
     *     https://github.com/eminence/procfs/blob/fc91e469/src/meminfo.rs#L321-L383
    */

    struct {
        const char* fmt;
        unsigned long val;
    } meminfo[] = {
        { "MemTotal:      %8lu kB\n", g_pal_public_state->mem_total / 1024 },
        { "MemFree:       %8lu kB\n", DkMemoryAvailableQuota() / 1024 },
        { "MemAvailable:  %8lu kB\n", DkMemoryAvailableQuota() / 1024 },
        { "Buffers:       %8lu kB\n", /*dummy value=*/0 },
        { "Cached:        %8lu kB\n", /*dummy value=*/0 },
        { "SwapCached:    %8lu kB\n", /*dummy value=*/0 },
        { "Active:        %8lu kB\n", /*dummy value=*/0 },
        { "Inactive:      %8lu kB\n", /*dummy value=*/0 },
        { "SwapTotal:     %8lu kB\n", /*dummy value=*/0 },
        { "SwapFree:      %8lu kB\n", /*dummy value=*/0 },
        { "Dirty:         %8lu kB\n", /*dummy value=*/0 },
        { "Writeback:     %8lu kB\n", /*dummy value=*/0 },
        { "Mapped:        %8lu kB\n", /*dummy value=*/0 },
        { "Shmem:         %8lu kB\n", /*dummy value=*/0 },
        { "Slab:          %8lu kB\n", /*dummy value=*/0 },
        { "Committed_AS:  %8lu kB\n", (g_pal_public_state->mem_total - DkMemoryAvailableQuota())
                                          / 1024 },
        { "VmallocTotal:  %8lu kB\n", g_pal_public_state->mem_total / 1024 },
        { "VmallocUsed:   %8lu kB\n", /*dummy value=*/0 },
        { "VmallocChunk:  %8lu kB\n", /*dummy value=*/0 },
    };

retry:
    max *= 2;
    size = 0;
    free(str);
    str = malloc(max);
    if (!str)
        return -ENOMEM;

    for (size_t i = 0; i < ARRAY_SIZE(meminfo); i++) {
        int ret = snprintf(str + size, max - size, meminfo[i].fmt, meminfo[i].val);
        if (ret < 0) {
            free(str);
            return ret;
        }

        if (size + ret >= max)
            goto retry;

        size += ret;
    }

    *out_data = str;
    *out_size = size;
    return 0;
}

// FIXME: remove once global realloc is enabled
static void* realloc_size(void* ptr, size_t old_size, size_t new_size) {
    void* tmp = malloc(new_size);
    if (!tmp) {
        return NULL;
    }

    memcpy(tmp, ptr, old_size);

    free(ptr);

    return tmp;
}

static int print_to_str(char** str, size_t off, size_t* size, const char* fmt, ...) {
    int ret;
    va_list ap;

retry:
    va_start(ap, fmt);
    ret = vsnprintf(*str + off, *size - off, fmt, ap);
    va_end(ap);

    if (ret < 0) {
        return -EINVAL;
    }

    if ((size_t)ret >= *size - off) {
        char* tmp = realloc_size(*str, *size, *size + 128);
        if (!tmp) {
            return -ENOMEM;
        }
        *size += 128;
        *str = tmp;
        goto retry;
    }

    return ret;
}

#define ADD_INFO(fmt, ...)                                            \
    do {                                                              \
        int ret = print_to_str(&str, size, &max, fmt, ##__VA_ARGS__); \
        if (ret < 0) {                                                \
            free(str);                                                \
            return ret;                                               \
        }                                                             \
        size += ret;                                                  \
    } while (0)

int proc_cpuinfo_load(struct shim_dentry* dent, char** out_data, size_t* out_size) {
    __UNUSED(dent);

    size_t size = 0;
    size_t max = 128;
    char* str = malloc(max);
    if (!str)
        return -ENOMEM;

    const struct pal_topo_info* topo = &g_pal_public_state->topo_info;
    const struct pal_cpu_info* cpu = &g_pal_public_state->cpu_info;
    for (size_t i = 0; i < topo->threads_cnt; i++) {
        struct pal_cpu_thread_info* thread = &topo->threads[i];
        if (!thread->is_online) {
            /* Offline cores are skipped in cpuinfo, with gaps in numbering. */
            continue;
        }
        struct pal_cpu_core_info* core = &topo->cores[thread->core_id];
        /* Below strings must match exactly the strings retrieved from /proc/cpuinfo
         * (see Linux's arch/x86/kernel/cpu/proc.c) */
        ADD_INFO("processor\t: %lu\n",   i);
        ADD_INFO("vendor_id\t: %s\n",    cpu->cpu_vendor);
        ADD_INFO("cpu family\t: %lu\n",  cpu->cpu_family);
        ADD_INFO("model\t\t: %lu\n",     cpu->cpu_model);
        ADD_INFO("model name\t: %s\n",   cpu->cpu_brand);
        ADD_INFO("stepping\t: %lu\n",    cpu->cpu_stepping);
        ADD_INFO("physical id\t: %zu\n", core->socket_id);

        /* Linux keeps this numbering socket-local, but we can use a different one, and it's
         * documented as "hardware platform's identifier (rather than the kernel's)" anyways. */
        ADD_INFO("core id\t\t: %lu\n",   thread->core_id);

        size_t cores_in_socket = 0;
        for (size_t j = 0; j < topo->cores_cnt; j++) { // slow, but shouldn't matter
            if (topo->cores[j].socket_id == core->socket_id)
                cores_in_socket++;
        }
        ADD_INFO("cpu cores\t: %zu\n", cores_in_socket);
        double bogomips = cpu->cpu_bogomips;
        // Apparently Gramine snprintf cannot into floats.
        ADD_INFO("bogomips\t: %lu.%02lu\n", (unsigned long)bogomips,
                 (unsigned long)(bogomips * 100.0 + 0.5) % 100);
        ADD_INFO("\n");
    }

    *out_data = str;
    *out_size = size;
    return 0;
}

int proc_stat_load(struct shim_dentry* dent, char** out_data, size_t* out_size) {
    __UNUSED(dent);

    size_t size = 0;
    size_t max = 128;
    char* str = malloc(max);
    if (!str)
        return -ENOMEM;

    /* 10 dummy time stats: currently all zeros */
    uint64_t user       = 0;
    uint64_t nice       = 0;
    uint64_t system     = 0;
    uint64_t idle       = 0;
    uint64_t iowait     = 0;
    uint64_t irq        = 0;
    uint64_t softirq    = 0;
    uint64_t steal      = 0;
    uint64_t guest      = 0;
    uint64_t guest_nice = 0;

    /* below strings must match exactly the strings retrieved from /proc/stat
     * (see Linux's fs/proc/stat.c) */
    ADD_INFO("cpu  %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu\n", user, nice, system, idle, iowait,
             irq, softirq, steal, guest, guest_nice);
    for (size_t i = 0; i < g_pal_public_state->topo_info.threads_cnt; i++) {
        if (!g_pal_public_state->topo_info.threads[i].is_online)
            continue;
        ADD_INFO("cpu%lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu\n", i, user, nice, system, idle,
                 iowait, irq, softirq, steal, guest, guest_nice);
    }

    /* no "intr" and "softirq" lines: no known workloads use them, and they are hard to emulate */
    ADD_INFO("ctxt %llu\n", 0);
    ADD_INFO("btime %llu\n", 0);
    ADD_INFO("processes %lu\n", 1);    /* at least this process was created */
    ADD_INFO("procs_running %u\n", 1); /* at least this process was created */
    ADD_INFO("procs_blocked %u\n", 0);

    *out_data = str;
    *out_size = size;
    return 0;
}

#undef ADD_INFO
