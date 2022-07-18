/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2022 Intel Corporation
 *                    Vijay Dhanraj <vijay.dhanraj@intel.com>
 *                    Michał Kowalczyk <mkow@invisiblethingslab.com>
 */

/*
 * This file contains the implementation of `/sys/devices/system/{cpu,node}/` pseudo-filesystem and
 * its sub-directories.
 */

#include "shim_checkpoint.h"
#include "shim_fs.h"
#include "shim_fs_pseudo.h"

int sys_print_as_ranges(char* buf, size_t buf_size, size_t count,
                        bool (*is_present)(size_t ind, const void* arg), const void* callback_arg) {
    size_t buf_pos = 0;
    const char* sep = "";
    for (size_t i = 0; i < count;) {
        while (i < count && !is_present(i, callback_arg))
            i++;
        size_t range_start = i;
        while (i < count && is_present(i, callback_arg))
            i++;
        size_t range_end = i; // exclusive

        if (range_start == range_end)
            break;
        int ret;
        if (range_start + 1 == range_end) {
            ret = snprintf(buf + buf_pos, buf_size - buf_pos, "%s%zu", sep, range_start);
        } else {
            ret = snprintf(buf + buf_pos, buf_size - buf_pos, "%s%zu-%zu", sep, range_start,
                           range_end - 1);
        }
        sep = ",";
        if (ret < 0)
            return ret;
        if ((size_t)ret >= buf_size - buf_pos)
            return -EOVERFLOW;
        buf_pos += ret;
    }
    if (buf_pos + 2 > buf_size)
        return -EOVERFLOW;
    buf[buf_pos]   = '\n';
    buf[buf_pos+1] = '\0';
    return 0;
}

int sys_print_as_bitmask(char* buf, size_t buf_size, size_t count,
                         bool (*is_present)(size_t ind, const void* arg),
                         const void* callback_arg) {
    if (count == 0)
        return strcpy_static(buf, "0\n", buf_size) ? 0 : -EOVERFLOW;

    size_t buf_pos = 0;
    int ret;
    size_t pos = count - 1;
    uint32_t word = 0;
    while (1) {
        if (is_present(pos, callback_arg))
            word |= 1 << pos % 32;
        if (pos % 32 == 0) {
            if (count <= 32) {
                /* Linux sysfs quirk: small bitmasks are printed without leading zeroes. */
                ret = snprintf(buf, buf_size, "%x\n", word); // pos == 0, loop exits afterwards
            } else {
                ret = snprintf(buf + buf_pos, buf_size - buf_pos,
                               "%08x%c", word, pos != 0 ? ',' : '\n');
            }
            if (ret < 0)
                return ret;
            if ((size_t)ret >= buf_size - buf_pos)
                return -EOVERFLOW;
            buf_pos += ret;
            word = 0;
        }

        if (pos == 0)
            break;
        pos--;
    }
    return 0;
}

static int sys_resource_info(const char* parent_name, size_t* out_total, const char** out_prefix) {
    const struct pal_topo_info* topo = &g_pal_public_state->topo_info;
    if (strcmp(parent_name, "node") == 0) {
        *out_total = topo->numa_nodes_cnt;
        *out_prefix = "node";
        return 0;
    } else if (strcmp(parent_name, "cpu") == 0) {
        *out_total = topo->threads_cnt;
        *out_prefix = "cpu";
        return 0;
    } else if (strcmp(parent_name, "cache") == 0) {
        size_t max = 0;
        /* Find the largest cache index used. */
        for (size_t i = 0; i < topo->threads_cnt; i++) {
            if (topo->threads[i].is_online) {
                for (size_t j = 0; j < MAX_CACHES; j++) {
                    if (topo->threads[i].ids_of_caches[j] != (size_t)-1) {
                        max = MAX(max, j + 1); // +1 to convert max index to elements count
                    }
                }
            }
        }
        *out_total = max;
        *out_prefix = "index";
        return 0;
    } else {
        log_debug("unrecognized resource: %s", parent_name);
        return -ENOENT;
    }
}

int sys_resource_find(struct shim_dentry* dent, const char* parent_name, unsigned int* out_num) {
    size_t total;
    const char* prefix;
    int ret = sys_resource_info(parent_name, &total, &prefix);
    if (ret < 0)
        return ret;

    if (total == 0)
        return -ENOENT;

    /* Search for "{parent_name}/{prefix}N", parse N (must be less than total) */

    struct shim_dentry* parent = dent->parent;
    while (parent) {
        if (strcmp(parent->name, parent_name) == 0) {
            if (!strstartswith(dent->name, prefix))
                return -ENOENT;

            size_t prefix_len = strlen(prefix);
            unsigned long n;
            if (pseudo_parse_ulong(&dent->name[prefix_len], total - 1, &n) < 0)
                return -ENOENT;

            *out_num = n;
            return 0;
        }

        dent = parent;
        parent = parent->parent;
    }
    return -ENOENT;
}

bool sys_resource_name_exists(struct shim_dentry* parent, const char* name) {
    size_t total;
    const char* prefix;
    int ret = sys_resource_info(parent->name, &total, &prefix);
    if (ret < 0)
        return false;

    if (total == 0)
        return false;

    /* Recognize "{prefix}N", check if N is less than total */

    if (!strstartswith(name, prefix))
        return false;

    size_t prefix_len = strlen(prefix);
    unsigned long n;
    if (pseudo_parse_ulong(&name[prefix_len], total - 1, &n) < 0)
        return false;

    return true;
}

int sys_resource_list_names(struct shim_dentry* parent, readdir_callback_t callback, void* arg) {
    size_t total;
    const char* prefix;
    int ret = sys_resource_info(parent->name, &total, &prefix);
    if (ret < 0)
        return -ENOENT;

    /* Generate "{prefix}N" names for all N less than total */

    for (size_t i = 0; i < total; i++) {
        char ent_name[strlen(prefix) + 22];
        snprintf(ent_name, sizeof(ent_name), "%s%zu", prefix, i);
        int ret = callback(ent_name, arg);
        if (ret < 0)
            return ret;
    }

    return 0;
}

int sys_load(const char* str, char** out_data, size_t* out_size) {
    assert(str);

    /* Use the string (without null terminator) as file data */
    size_t size = strlen(str);
    char* data = malloc(size);
    if (!data)
        return -ENOMEM;
    memcpy(data, str, size);
    *out_data = data;
    *out_size = size;
    return 0;
}

static void init_cpu_dir(struct pseudo_node* cpu) {
    pseudo_add_str(cpu, "online", &sys_cpu_general_load);
    pseudo_add_str(cpu, "possible", &sys_cpu_general_load);

    struct pseudo_node* cpuX = pseudo_add_dir(cpu, NULL);
    cpuX->name_exists = &sys_resource_name_exists;
    cpuX->list_names = &sys_resource_list_names;

    /* `cpu/cpuX/online` exists for all CPUs *except* `cpu0`. */
    struct pseudo_node* online = pseudo_add_str(cpuX, "online", &sys_cpu_load_online);
    online->name_exists = &sys_cpu_online_name_exists;

    /* `cpu/cpuX/topology` exists only for online CPUs. */
    struct pseudo_node* topology = pseudo_add_dir(cpuX, "topology");
    topology->name_exists = &sys_cpu_exists_only_if_online;
    pseudo_add_str(topology, "core_id", &sys_cpu_load_topology);
    pseudo_add_str(topology, "physical_package_id", &sys_cpu_load_topology);
    pseudo_add_str(topology, "core_siblings", &sys_cpu_load_topology);
    pseudo_add_str(topology, "thread_siblings", &sys_cpu_load_topology);

    /* `cpu/cpuX/cache` exists only for online CPUs. */
    struct pseudo_node* cache = pseudo_add_dir(cpuX, "cache");
    cache->name_exists = &sys_cpu_exists_only_if_online;
    struct pseudo_node* indexX = pseudo_add_dir(cache, NULL);
    indexX->name_exists = &sys_resource_name_exists;
    indexX->list_names = &sys_resource_list_names;

    pseudo_add_str(indexX, "shared_cpu_map", &sys_cache_load);
    pseudo_add_str(indexX, "level", &sys_cache_load);
    pseudo_add_str(indexX, "type", &sys_cache_load);
    pseudo_add_str(indexX, "size", &sys_cache_load);
    pseudo_add_str(indexX, "coherency_line_size", &sys_cache_load);
    pseudo_add_str(indexX, "number_of_sets", &sys_cache_load);
    pseudo_add_str(indexX, "physical_line_partition", &sys_cache_load);
}

static void init_node_dir(struct pseudo_node* node) {
    pseudo_add_str(node, "online", &sys_node_general_load);
    pseudo_add_str(node, "possible", &sys_node_general_load);

    struct pseudo_node* nodeX = pseudo_add_dir(node, NULL);
    nodeX->name_exists = &sys_resource_name_exists;
    nodeX->list_names = &sys_resource_list_names;

    pseudo_add_str(nodeX, "cpumap", &sys_node_load);
    pseudo_add_str(nodeX, "distance", &sys_node_load);

    // TODO(mkow): Does this show up for offline nodes? I never succeeded in shutting down one, even
    // after shutting down all CPUs inside the node it shows up as online on `node/online` list.
    struct pseudo_node* hugepages = pseudo_add_dir(nodeX, "hugepages");
    struct pseudo_node* hugepages_2m = pseudo_add_dir(hugepages, "hugepages-2048kB");
    pseudo_add_str(hugepages_2m, "nr_hugepages", &sys_node_load);
    struct pseudo_node* hugepages_1g = pseudo_add_dir(hugepages, "hugepages-1048576kB");
    pseudo_add_str(hugepages_1g, "nr_hugepages", &sys_node_load);
}

int init_sysfs(void) {
    struct pseudo_node* root = pseudo_add_root_dir("sys");
    struct pseudo_node* devices = pseudo_add_dir(root, "devices");
    struct pseudo_node* system = pseudo_add_dir(devices, "system");

    struct pseudo_node* cpu = pseudo_add_dir(system, "cpu");
    init_cpu_dir(cpu);

    struct pseudo_node* node = pseudo_add_dir(system, "node");
    init_node_dir(node);

    return 0;
}

BEGIN_CP_FUNC(topo_info) {
    __UNUSED(size);
    __UNUSED(obj);
    __UNUSED(objp);

    struct pal_topo_info* topo_info = &g_pal_public_state->topo_info;
    size_t off = ADD_CP_OFFSET(sizeof(*topo_info));
    struct pal_topo_info* new_topo_info = (void*)(base + off);
    memset(new_topo_info, 0, sizeof(*new_topo_info));

    new_topo_info->caches_cnt = topo_info->caches_cnt;
    size_t caches_size = topo_info->caches_cnt * sizeof(*topo_info->caches);
    new_topo_info->caches = (void*)(base + ADD_CP_OFFSET(caches_size));
    memcpy(new_topo_info->caches, topo_info->caches, caches_size);

    new_topo_info->threads_cnt = topo_info->threads_cnt;
    size_t threads_size = topo_info->threads_cnt * sizeof(*topo_info->threads);
    new_topo_info->threads = (void*)(base + ADD_CP_OFFSET(threads_size));
    memcpy(new_topo_info->threads, topo_info->threads, threads_size);

    new_topo_info->cores_cnt = topo_info->cores_cnt;
    size_t cores_size = topo_info->cores_cnt * sizeof(*topo_info->cores);
    new_topo_info->cores = (void*)(base + ADD_CP_OFFSET(cores_size));
    memcpy(new_topo_info->cores, topo_info->cores, cores_size);

    new_topo_info->sockets_cnt = topo_info->sockets_cnt;
    size_t sockets_size = topo_info->sockets_cnt * sizeof(*topo_info->sockets);
    new_topo_info->sockets = (void*)(base + ADD_CP_OFFSET(sockets_size));
    memcpy(new_topo_info->sockets, topo_info->sockets, sockets_size);

    new_topo_info->numa_nodes_cnt = topo_info->numa_nodes_cnt;
    size_t numa_nodes_size = topo_info->numa_nodes_cnt * sizeof(*topo_info->numa_nodes);
    new_topo_info->numa_nodes = (void*)(base + ADD_CP_OFFSET(numa_nodes_size));
    memcpy(new_topo_info->numa_nodes, topo_info->numa_nodes, numa_nodes_size);

    size_t numa_dm_size = topo_info->numa_nodes_cnt * topo_info->numa_nodes_cnt
                          * sizeof(*topo_info->numa_distance_matrix);
    new_topo_info->numa_distance_matrix = (void*)(base + ADD_CP_OFFSET(numa_dm_size));
    memcpy(new_topo_info->numa_distance_matrix, topo_info->numa_distance_matrix, numa_dm_size);

    ADD_CP_FUNC_ENTRY(off);
}
END_CP_FUNC(topo_info)

BEGIN_RS_FUNC(topo_info) {
    __UNUSED(offset);
    struct pal_topo_info* topo_info = (void*)(base + GET_CP_FUNC_ENTRY());

    CP_REBASE(topo_info->caches);
    CP_REBASE(topo_info->threads);
    CP_REBASE(topo_info->cores);
    CP_REBASE(topo_info->sockets);
    CP_REBASE(topo_info->numa_nodes);
    CP_REBASE(topo_info->numa_distance_matrix);

    memcpy(&g_pal_public_state->topo_info, topo_info, sizeof(*topo_info));
}
END_RS_FUNC(topo_info)
