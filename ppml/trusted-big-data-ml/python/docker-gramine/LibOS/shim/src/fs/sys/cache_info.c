/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Vijay Dhanraj <vijay.dhanraj@intel.com>
 */

/*
 * This file contains the implementation of `/sys/devices/system/cpu/cpuX/cache` and its
 * sub-directories.
 */

#include <stdbool.h>

#include "api.h"
#include "shim_fs.h"
#include "shim_fs_pseudo.h"

struct callback_arg {
    size_t cache_id_to_match; // index in global caches list (topo_info->caches[])
    size_t cache_class; // index in ids_of_caches[]
};

static bool is_same_cache(size_t idx, const void* _arg) {
    const struct callback_arg* arg = _arg;
    const struct pal_cpu_thread_info* thread = &g_pal_public_state->topo_info.threads[idx];
    return thread->is_online
           && thread->ids_of_caches[arg->cache_class] == arg->cache_id_to_match;
}

int sys_cache_load(struct shim_dentry* dent, char** out_data, size_t* out_size) {
    int ret;

    unsigned int cache_class;
    ret = sys_resource_find(dent, "cache", &cache_class);
    if (ret < 0)
        return ret;

    unsigned int thread_id;
    ret = sys_resource_find(dent, "cpu", &thread_id);
    if (ret < 0)
        return ret;

    const char* name = dent->name;

    const struct pal_topo_info* topo = &g_pal_public_state->topo_info;
    size_t cache_idx = topo->threads[thread_id].ids_of_caches[cache_class];
    const struct pal_cache_info* cache = &topo->caches[cache_idx];
    char str[PAL_SYSFS_MAP_FILESZ] = {'\0'};
    if (strcmp(name, "shared_cpu_map") == 0) {
        struct callback_arg callback_arg = {
            .cache_id_to_match = cache_idx,
            .cache_class = cache_class,
        };
        ret = sys_print_as_bitmask(str, sizeof(str), topo->threads_cnt,
                                   is_same_cache, &callback_arg);
    } else if (strcmp(name, "level") == 0) {
        ret = snprintf(str, sizeof(str), "%zu\n", cache->level);
    } else if (strcmp(name, "type") == 0) {
        switch (cache->type) {
            case CACHE_TYPE_DATA:
                ret = snprintf(str, sizeof(str), "Data\n");
                break;
            case CACHE_TYPE_INSTRUCTION:
                ret = snprintf(str, sizeof(str), "Instruction\n");
                break;
            case CACHE_TYPE_UNIFIED:
                ret = snprintf(str, sizeof(str), "Unified\n");
                break;
            default:
                __builtin_unreachable();
        }
    } else if (strcmp(name, "size") == 0) {
        ret = snprintf(str, sizeof(str), "%zuK\n", cache->size >> 10);
    } else if (strcmp(name, "coherency_line_size") == 0) {
        ret = snprintf(str, sizeof(str), "%zu\n", cache->coherency_line_size);
    } else if (strcmp(name, "number_of_sets") == 0) {
        ret = snprintf(str, sizeof(str), "%zu\n", cache->number_of_sets);
    } else if (strcmp(name, "physical_line_partition") == 0) {
        snprintf(str, sizeof(str), "%zu\n", cache->physical_line_partition);
    } else {
        log_debug("unrecognized file: %s", name);
        ret = -ENOENT;
    }

    if (ret < 0)
        return ret;

    return sys_load(str, out_data, out_size);
}
