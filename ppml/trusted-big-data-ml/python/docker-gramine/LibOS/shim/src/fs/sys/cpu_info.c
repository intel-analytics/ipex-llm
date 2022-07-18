/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Vijay Dhanraj <vijay.dhanraj@intel.com>
 */

/*
 * This file contains the implementation of `/sys/devices/system/cpu` and its sub-directories
 * (except for `cache`, which is implemented in cache_info.c).
 */

#include "api.h"
#include "shim_fs.h"
#include "shim_fs_pseudo.h"

static bool is_online(size_t ind, const void* arg) {
    __UNUSED(arg);
    return g_pal_public_state->topo_info.threads[ind].is_online;
}

static bool return_true(size_t ind, const void* arg) {
    __UNUSED(ind);
    __UNUSED(arg);
    return true;
}

int sys_cpu_general_load(struct shim_dentry* dent, char** out_data, size_t* out_size) {
    int ret;
    const struct pal_topo_info* topo = &g_pal_public_state->topo_info;
    const char* name = dent->name;
    char str[PAL_SYSFS_BUF_FILESZ];

    if (strcmp(name, "online") == 0) {
        ret = sys_print_as_ranges(str, sizeof(str), topo->threads_cnt, is_online, NULL);
    } else if (strcmp(name, "possible") == 0) {
        ret = sys_print_as_ranges(str, sizeof(str), topo->threads_cnt, return_true, NULL);
    } else {
        log_debug("unrecognized file: %s", name);
        ret = -ENOENT;
    }

    if (ret < 0)
        return ret;

    return sys_load(str, out_data, out_size);
}

static bool is_in_same_core(size_t thread_id, const void* _arg) {
    size_t arg_id = *(const size_t*)_arg;
    struct pal_cpu_thread_info* thread = &g_pal_public_state->topo_info.threads[thread_id];
    return thread->is_online && thread->core_id == arg_id;
}

static bool is_in_same_socket(size_t thread_id, const void* _arg) {
    size_t arg_id = *(const size_t*)_arg;
    struct pal_cpu_thread_info* thread = &g_pal_public_state->topo_info.threads[thread_id];
    return thread->is_online
        && g_pal_public_state->topo_info.cores[thread->core_id].socket_id == arg_id;
}

int sys_cpu_load_online(struct shim_dentry* dent, char** out_data, size_t* out_size) {
    int ret;
    unsigned int thread_id;

    assert(!strcmp(dent->name, "online"));

    ret = sys_resource_find(dent, "cpu", &thread_id);
    if (ret < 0)
        return ret;

    /* `cpu/cpuX/online` is not present for cpu0 */
    if (thread_id == 0)
        return -ENOENT;

    struct pal_cpu_thread_info* thread = &g_pal_public_state->topo_info.threads[thread_id];
    return sys_load(thread->is_online ? "1\n" : "0\n", out_data, out_size);
}

int sys_cpu_load_topology(struct shim_dentry* dent, char** out_data, size_t* out_size) {
    int ret;
    unsigned int thread_id;
    ret = sys_resource_find(dent, "cpu", &thread_id);
    if (ret < 0)
        return ret;

    const char* name = dent->name;
    const struct pal_topo_info* topo = &g_pal_public_state->topo_info;
    struct pal_cpu_thread_info* thread = &topo->threads[thread_id];
    assert(thread->is_online); // `cpuX/topology/` should not exist for offline threads.
    struct pal_cpu_core_info* core = &topo->cores[thread->core_id];
    char str[PAL_SYSFS_MAP_FILESZ] = {'\0'};
    if (strcmp(name, "core_id") == 0) {
        ret = snprintf(str, sizeof(str), "%zu\n", thread->core_id);
    } else if (strcmp(name, "physical_package_id") == 0) {
        ret = snprintf(str, sizeof(str), "%zu\n", core->socket_id);
    } else if (strcmp(name, "thread_siblings") == 0) {
        ret = sys_print_as_bitmask(str, sizeof(str), topo->threads_cnt, is_in_same_core,
                                   &thread->core_id);
    } else if (strcmp(name, "core_siblings") == 0) {
        ret = sys_print_as_bitmask(str, sizeof(str), topo->threads_cnt, is_in_same_socket,
                                   &core->socket_id);
    } else {
        log_debug("unrecognized file: %s", name);
        ret = -ENOENT;
    }

    if (ret < 0)
        return ret;

    return sys_load(str, out_data, out_size);
}

bool sys_cpu_online_name_exists(struct shim_dentry* parent, const char* name) {
    if (strcmp(name, "online") != 0)
        return false;

    int ret;
    unsigned int thread_id;
    ret = sys_resource_find(parent, "cpu", &thread_id);
    if (ret < 0)
        return false;

    return thread_id != 0;
}

bool sys_cpu_exists_only_if_online(struct shim_dentry* parent, const char* name) {
    __UNUSED(name);
    int ret;
    unsigned int thread_id;
    ret = sys_resource_find(parent, "cpu", &thread_id);
    if (ret < 0)
        return false;

    const struct pal_topo_info* topo = &g_pal_public_state->topo_info;
    return topo->threads[thread_id].is_online;
}
