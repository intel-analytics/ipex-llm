/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2022 Intel Corporation
 *                    Vijay Dhanraj <vijay.dhanraj@intel.com>
 *                    Michał Kowalczyk <mkow@invisiblethingslab.com>
 */

/*
 * This file contains the APIs to expose host topology information.
 * All of them are not suitable for untrusted inputs! (due to overflows and liberal parsing)
 */

#include <asm/errno.h>
#include <asm/fcntl.h>

#include "api.h"
#include "pal_linux.h"
#include "syscall.h"
#include "topo_info.h"

ssize_t read_file_buffer(const char* filename, char* buf, size_t count) {
    int fd = DO_SYSCALL(open, filename, O_RDONLY);
    if (fd < 0)
        return fd;

    ssize_t ret = DO_SYSCALL(read, fd, buf, count);
    long close_ret = DO_SYSCALL(close, fd);
    if (ret == 0 && close_ret < 0)
        ret = close_ret;

    return ret;
}

/* Opens a pseudo-file describing HW resources and reads the value stored in the file, optionally
 * honoring a 'K' unit suffix. */
static int get_hw_resource_value(const char* filename, size_t* out_value) {
    char buf[PAL_SYSFS_BUF_FILESZ];
    int ret = read_file_buffer(filename, buf, sizeof(buf) - 1);
    if (ret < 0)
        return ret;

    buf[ret] = '\0';

    const char* end;
    unsigned long val;
    ret = str_to_ulong(buf, 10, &val, &end);
    if (ret < 0)
        return -EINVAL;

    if (*end != '\n' && *end != '\0' && *end != 'K') {
        /* Illegal character found */
        return -EINVAL;
    }

    if (*end == 'K') {
        if (__builtin_mul_overflow(val, 1024, &val))
            return -EOVERFLOW;
    }

    *out_value = val;
    return 0;
}

/* Read a space-separated list of numbers.
 * The file has to contain at least `count` numbers, otherwise this function returns `-EINVAL`. */
static int read_numbers_from_file(const char* path, size_t* out_arr, size_t count) {
    char buf[PAL_SYSFS_BUF_FILESZ];
    int ret = read_file_buffer(path, buf, sizeof(buf) - 1);
    if (ret < 0)
        return ret;
    buf[ret] = '\0';

    const char* buf_it = buf;
    const char* end;
    for (size_t i = 0; i < count; i++) {
        unsigned long val;
        ret = str_to_ulong(buf_it, 10, &val, &end);
        if (ret < 0)
            return -EINVAL;
        char expected_separator = (i != count - 1) ? ' ' : '\n';
        if (*end != expected_separator)
            return -EINVAL;
        buf_it = end + 1;
        out_arr[i] = (size_t)val;
    }
    return 0;
}

static int iterate_ranges_from_file(const char* path, int (*callback)(size_t index, void* arg),
                                    void* callback_arg) {
    char buf[PAL_SYSFS_BUF_FILESZ];
    int ret = read_file_buffer(path, buf, sizeof(buf) - 1);
    if (ret < 0)
        return ret;
    buf[ret] = '\0';

    const char* buf_it = buf;
    long prev = -1;
    while (*buf_it && *buf_it != '\n') {
        const char* parse_end;
        unsigned long val;
        ret = str_to_ulong(buf_it, 10, &val, &parse_end);
        if (ret < 0)
            return -EINVAL;
        buf_it = parse_end;

        if (*buf_it == ',' || *buf_it == '\n') {
            if (prev == -1) {
                // single index
                ret = callback(val, callback_arg);
                if (ret < 0)
                    return ret;
            } else {
                // range
                for (size_t i = prev; i <= (size_t)val; i++) {
                    ret = callback(i, callback_arg);
                    if (ret < 0)
                        return ret;
                }
            }
            prev = -1;
        } else if (*buf_it == '-' && prev == -1) {
            // range start
            prev = val;
        } else {
            log_error("Invalid range format when parsing %s", path);
            return -EINVAL;
        }
        buf_it++;
    }
    if (prev != -1) {
        log_error("Invalid range format when parsing %s", path);
        return -EINVAL;
    }
    return 0;
}

static int read_cache_info(struct pal_cache_info* ci, size_t thread_idx, size_t cache_idx) {
    int ret;

    char path[128];

    ret = snprintf(path, sizeof(path),
                   "/sys/devices/system/cpu/cpu%zu/cache/index%zu/level", thread_idx, cache_idx);
    if (ret < 0)
        return ret;
    ret = get_hw_resource_value(path, &ci->level);
    if (ret < 0)
        return ret;

    char type[PAL_SYSFS_BUF_FILESZ] = {'\0'};
    ret = snprintf(path, sizeof(path),
                   "/sys/devices/system/cpu/cpu%zu/cache/index%zu/type", thread_idx, cache_idx);
    if (ret < 0)
        return ret;
    ret = read_file_buffer(path, type, sizeof(type) - 1);
    if (ret < 0)
        return ret;
    type[ret] = '\0';

    if (!strcmp(type, "Unified\n")) {
       ci->type = CACHE_TYPE_UNIFIED;
    } else if (!strcmp(type, "Instruction\n")) {
       ci->type = CACHE_TYPE_INSTRUCTION;
    } else if (!strcmp(type, "Data\n")) {
       ci->type = CACHE_TYPE_DATA;
    } else {
        return -EINVAL;
    }

    ret = snprintf(path, sizeof(path),
                   "/sys/devices/system/cpu/cpu%zu/cache/index%zu/size", thread_idx, cache_idx);
    if (ret < 0)
        return ret;
    ret = get_hw_resource_value(path, &ci->size);
    if (ret < 0)
        return ret;

    ret = snprintf(path, sizeof(path),
                   "/sys/devices/system/cpu/cpu%zu/cache/index%zu/coherency_line_size",
                   thread_idx, cache_idx);
    if (ret < 0)
        return ret;
    ret = get_hw_resource_value(path, &ci->coherency_line_size);
    if (ret < 0)
        return ret;

    ret = snprintf(path, sizeof(path),
                   "/sys/devices/system/cpu/cpu%zu/cache/index%zu/number_of_sets",
                   thread_idx, cache_idx);
    if (ret < 0)
        return ret;
    ret = get_hw_resource_value(path, &ci->number_of_sets);
    if (ret < 0)
        return ret;

    ret = snprintf(path, sizeof(path),
                   "/sys/devices/system/cpu/cpu%zu/cache/index%zu/physical_line_partition",
                   thread_idx, cache_idx);
    if (ret < 0)
        return ret;
    ret = get_hw_resource_value(path, &ci->physical_line_partition);
    if (ret < 0)
        return ret;

    return 0;
}

static int get_ranges_end(size_t ind, void* _arg) {
    *(size_t*)_arg = ind + 1; // can overflow, but this function is used only on trusted data
    return 0;
}

static int set_thread_online(size_t ind, void* _threads) {
    struct pal_cpu_thread_info* threads = (struct pal_cpu_thread_info*)_threads;
    threads[ind].is_online = true;
    return 0;
}

static int set_numa_node_online(size_t ind, void* _numa_nodes) {
    struct pal_numa_node_info* numa_nodes = (struct pal_numa_node_info*)_numa_nodes;
    numa_nodes[ind].is_online = true;
    return 0;
}

struct set_core_id_args {
    struct pal_cpu_thread_info* threads;
    size_t id_to_set;
};

static int set_core_id(size_t ind, void* _args) {
    struct set_core_id_args* args = _args;
    args->threads[ind].core_id = args->id_to_set;
    return 0;
}

struct set_socket_id_args {
    struct pal_cpu_thread_info* threads;
    struct pal_cpu_core_info* cores;
    size_t id_to_set;
};

static int set_socket_id(size_t ind, void* _args) {
    struct set_socket_id_args* args = _args;
    if (!args->threads[ind].is_online)
        return 0;
    args->cores[args->threads[ind].core_id].socket_id = args->id_to_set;
    return 0;
}

struct set_cache_id_args {
    struct pal_cpu_thread_info* threads;
    size_t cache_ind;
    size_t id_to_set;
};

static int set_cache_id(size_t ind, void* _args) {
    struct set_cache_id_args* args = _args;
    args->threads[ind].ids_of_caches[args->cache_ind] = args->id_to_set;
    return 0;
}

struct set_node_id_args {
    struct pal_cpu_thread_info* threads;
    struct pal_cpu_core_info* cores;
    size_t id_to_set;
};

static int set_node_id(size_t thr_ind, void* _args) {
    struct set_node_id_args* args = _args;
    if (!args->threads[thr_ind].is_online)
        return 0;
    args->cores[args->threads[thr_ind].core_id].node_id = args->id_to_set;
    return 0;
}

int get_topology_info(struct pal_topo_info* topo_info) {
    size_t threads_cnt = 0;
    int ret = iterate_ranges_from_file("/sys/devices/system/cpu/possible", get_ranges_end, &threads_cnt);
    if (ret < 0)
        return ret;
    size_t nodes_cnt = 0;
    ret = iterate_ranges_from_file("/sys/devices/system/node/possible", get_ranges_end, &nodes_cnt);
    if (ret < 0)
        return ret;

    struct pal_cpu_thread_info* threads = malloc(threads_cnt * sizeof(*threads));
    size_t caches_cnt = 0;
    struct pal_cache_info* caches = malloc(threads_cnt * sizeof(*caches) * MAX_CACHES); // overapproximate the count
    size_t cores_cnt = 0;
    struct pal_cpu_core_info* cores = malloc(threads_cnt * sizeof(*cores)); // overapproximate the count
    size_t sockets_cnt = 0;
    struct pal_socket_info* sockets = malloc(threads_cnt * sizeof(*sockets)); // overapproximate the count
    struct pal_numa_node_info* numa_nodes = malloc(nodes_cnt * sizeof(*numa_nodes));
    size_t* distances = malloc(nodes_cnt * nodes_cnt * sizeof(*distances));
    if (!threads || !caches || !cores || !sockets || !numa_nodes || !distances) {
        ret = -ENOMEM;
        goto fail;
    }

    for (size_t i = 0; i < threads_cnt; i++) {
        cores[i].node_id = -1;
        cores[i].socket_id = -1;
        threads[i].is_online = false;
        threads[i].core_id = -1;
        for (size_t j = 0; j < MAX_CACHES; j++) {
            threads[i].ids_of_caches[j] = (size_t)-1;
        }
    }
    for (size_t i = 0; i < nodes_cnt; i++)
        numa_nodes[i].is_online = false;

    ret = iterate_ranges_from_file("/sys/devices/system/cpu/online", set_thread_online, threads);
    if (ret < 0)
        goto fail;
    ret = iterate_ranges_from_file("/sys/devices/system/node/online", set_numa_node_online,
                                   numa_nodes);
    if (ret < 0)
        goto fail;

    char path[128];
    for (size_t i = 0; i < threads_cnt; i++) {
        if (!threads[i].is_online)
            /* No information is available for offline threads. */
            continue;

        if (threads[i].core_id == (size_t)-1) {
            /* Insert new core to the list. */
            snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu%zu/topology/thread_siblings_list", i); // includes ourselves
            ret = iterate_ranges_from_file(path, set_core_id, &(struct set_core_id_args) {
                .threads = threads,
                .id_to_set = cores_cnt,
            });
            if (ret < 0)
                goto fail;
            cores_cnt++;
        }
    }

    for (size_t i = 0; i < threads_cnt; i++) {
        if (!threads[i].is_online)
            continue;

        size_t core_id = threads[i].core_id;
        if (cores[core_id].socket_id == (size_t)-1) {
            /* Insert new socket to the list. */
            snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu%zu/topology/core_siblings_list", i);
            ret = iterate_ranges_from_file(path, set_socket_id, &(struct set_socket_id_args) {
                .threads = threads,
                .cores = cores,
                .id_to_set = sockets_cnt,
            });
            if (ret < 0)
                goto fail;
            sockets_cnt++;
        }
    }

    for (size_t i = 0; i < nodes_cnt; i++) {
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%zu/cpulist", i);
        ret = iterate_ranges_from_file(path, set_node_id, &(struct set_node_id_args){
            .threads = threads,
            .cores = cores,
            .id_to_set = i,
        });
        if (ret < 0)
            goto fail;

        ret = snprintf(path, sizeof(path), "/sys/devices/system/node/node%zu/distance", i);
        if (ret < 0)
            goto fail;
        ret = read_numbers_from_file(path, distances + i * nodes_cnt, nodes_cnt);
        if (ret < 0)
            goto fail;

        /* Since our sysfs doesn't support writes, set persistent hugepages to their default value
         * of zero */
        numa_nodes[i].nr_hugepages[HUGEPAGES_2M] = 0;
        numa_nodes[i].nr_hugepages[HUGEPAGES_1G] = 0;
    }

    for (size_t i = 0; i < threads_cnt; i++) {
        if (!threads[i].is_online)
            continue;

        for (size_t j = 0; j < MAX_CACHES; j++) {
            if (threads[i].ids_of_caches[j] == (size_t)-1) {
                /* Insert new cache to the list.
                 * `shared_cpu_map` lists threads sharing this very cache. All sharing is
                 * between caches on the same cache level. */
                snprintf(path, sizeof(path),
                         "/sys/devices/system/cpu/cpu%zu/cache/index%zu/shared_cpu_list", i, j);
                ret = iterate_ranges_from_file(path, set_cache_id, &(struct set_cache_id_args) {
                    .threads = threads,
                    .cache_ind = j,
                    .id_to_set = caches_cnt,
                });
                if (ret == -ENOENT) {
                    // No more caches.
                    break;
                }
                if (ret < 0)
                    goto fail;
                ret = read_cache_info(&caches[caches_cnt], i, j);
                if (ret < 0)
                    goto fail;
                caches_cnt++;
            }
        }
    }
    /* Note: We could add realloc here to save memory after we know the final sizes of all the
     * buffers (after we implement realloc()). But the savings would rather be negligible. */

    topo_info->caches_cnt     = caches_cnt;
    topo_info->threads_cnt    = threads_cnt;
    topo_info->cores_cnt      = cores_cnt;
    topo_info->sockets_cnt    = sockets_cnt;
    topo_info->numa_nodes_cnt = nodes_cnt;
    topo_info->caches               = caches;
    topo_info->threads              = threads;
    topo_info->cores                = cores;
    topo_info->sockets              = sockets;
    topo_info->numa_nodes           = numa_nodes;
    topo_info->numa_distance_matrix = distances;
    return 0;

fail:
    free(caches);
    free(threads);
    free(cores);
    free(sockets);
    free(numa_nodes);
    free(distances);
    return ret;
}
