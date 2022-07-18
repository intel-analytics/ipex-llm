/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University
 * Copyright (C) 2022 Intel Corporation
 *                    Michał Kowalczyk <mkow@invisiblethingslab.com>
 *                    Vijay Dhanraj <vijay.dhanraj@intel.com>
 */

/*
 * This file contains the main function of the PAL loader, which loads and processes environment,
 * arguments and manifest.
 */

#include <stdint.h>
#include <stdnoreturn.h>

#include "api.h"
#include "enclave_tf.h"
#include "init.h"
#include "pal.h"
#include "pal_internal.h"
#include "pal_linux.h"
#include "pal_linux_defs.h"
#include "pal_rtld.h"
#include "pal_topology.h"
#include "toml.h"
#include "toml_utils.h"

struct pal_linuxsgx_state g_pal_linuxsgx_state;

PAL_SESSION_KEY g_master_key = {0};

/* Limit of PAL memory available for _DkVirtualMemoryAlloc(PAL_ALLOC_INTERNAL) */
size_t g_pal_internal_mem_size = PAL_INITIAL_MEM_SIZE;

const size_t g_page_size = PRESET_PAGESIZE;

void _DkGetAvailableUserAddressRange(void** out_start, void** out_end) {
    *out_start = g_pal_linuxsgx_state.heap_min;
    *out_end   = g_pal_linuxsgx_state.heap_max;

    /* Keep some heap for internal PAL objects allocated at runtime (recall that LibOS does not keep
     * track of PAL memory, so without this limit it could overwrite internal PAL memory). See also
     * `enclave_pages.c`. */
    *out_end = SATURATED_P_SUB(*out_end, g_pal_internal_mem_size, *out_start);

    if (*out_end <= *out_start) {
        log_error("Not enough enclave memory, please increase enclave size!");
        ocall_exit(1, /*is_exitgroup=*/true);
    }
}

/*
 * Takes a pointer+size to an untrusted memory region containing a
 * NUL-separated list of strings. It builds an argv-style list in trusted memory
 * with those strings.
 *
 * It is responsible for handling the access to untrusted memory safely
 * (returns NULL on error) and ensures that all strings are properly
 * terminated. The content of the strings is NOT further sanitized.
 *
 * The argv-style list is allocated on the heap and the caller is responsible
 * to free it (For argv and envp we rely on auto free on termination in
 * practice).
 */
/* This function doesn't clean up resources on failure as we terminate the process anyway. */
static const char** make_argv_list(void* uptr_src, size_t src_size) {
    const char** argv;

    if (src_size == 0) {
        argv = malloc(sizeof(char*));
        if (argv)
            argv[0] = NULL;
        return argv;
    }

    char* data = sgx_import_to_enclave(uptr_src, src_size);
    if (!data) {
        return NULL;
    }
    data[src_size - 1] = '\0';

    size_t argc = 0;
    for (size_t i = 0; i < src_size; i++) {
        if (data[i] == '\0') {
            argc++;
        }
    }

    size_t argv_size;
    if (__builtin_mul_overflow(argc + 1, sizeof(char*), &argv_size)) {
        goto fail;
    }
    argv = malloc(argv_size);
    if (!argv) {
        goto fail;
    }
    argv[argc] = NULL;

    size_t data_i = 0;
    for (size_t arg_i = 0; arg_i < argc; arg_i++) {
        argv[arg_i] = &data[data_i];
        while (data[data_i] != '\0') {
            data_i++;
        }
        data_i++;
    }

    return argv;

fail:
    free(data);
    return NULL;
}

/* Without this, `(int)imported_bool` may actually return something outside of {0, 1}. */
static void coerce_untrusted_bool(bool* ptr) {
    static_assert(sizeof(bool) == sizeof(unsigned char), "Unsupported compiler");
    *ptr = !!READ_ONCE(*(unsigned char*)ptr);
}

/* `*topo_info` should already be deep-copied into the enclave. */
static int sanitize_topo_info(struct pal_topo_info* topo_info) {
    for (size_t i = 0; i < topo_info->caches_cnt; i++) {
        struct pal_cache_info* cache = &topo_info->caches[i];
        if (cache->type != CACHE_TYPE_DATA &&
            cache->type != CACHE_TYPE_INSTRUCTION &&
            cache->type != CACHE_TYPE_UNIFIED) {
            return -PAL_ERROR_INVAL;
        }

        if (   !IS_IN_RANGE_INCL(cache->level, 1, 3)
            || !IS_IN_RANGE_INCL(cache->size, 1, 1 << 30)
            || !IS_IN_RANGE_INCL(cache->coherency_line_size, 1, 1 << 16)
            || !IS_IN_RANGE_INCL(cache->number_of_sets, 1, 1 << 30)
            || !IS_IN_RANGE_INCL(cache->physical_line_partition, 1, 1 << 16))
            return -PAL_ERROR_INVAL;
    }

    if (topo_info->threads_cnt == 0 || !topo_info->threads[0].is_online) {
        // Linux requires this
        return -PAL_ERROR_INVAL;
    }

    for (size_t i = 0; i < topo_info->threads_cnt; i++) {
        struct pal_cpu_thread_info* thread = &topo_info->threads[i];
        coerce_untrusted_bool(&thread->is_online);
        if (thread->is_online) {
            if (thread->core_id >= topo_info->cores_cnt)
                return -PAL_ERROR_INVAL;
            /* Verify that the cache array has no holes... */
            for (size_t j = 0; j < MAX_CACHES - 1; j++)
                if (thread->ids_of_caches[j] == (size_t)-1
                        && thread->ids_of_caches[j + 1] != (size_t)-1)
                    return -PAL_ERROR_INVAL;
            /* ...and valid indices. */
            for (size_t j = 0; j < MAX_CACHES; j++) {
                if (thread->ids_of_caches[j] != (size_t)-1
                    && thread->ids_of_caches[j] >= topo_info->caches_cnt)
                    return -PAL_ERROR_INVAL;
            }
        } else {
            // Not required, just a hardening in case we accidentally accessed offline CPU's fields.
            thread->core_id = 0;
            for (size_t j = 0; j < MAX_CACHES; j++)
                thread->ids_of_caches[j] = 0;
        }
    }

    for (size_t i = 0; i < topo_info->cores_cnt; i++) {
        if (topo_info->cores[i].socket_id >= topo_info->sockets_cnt)
            return -PAL_ERROR_INVAL;
        if (topo_info->cores[i].node_id >= topo_info->numa_nodes_cnt)
            return -PAL_ERROR_INVAL;
    }

    if (!topo_info->numa_nodes[0].is_online) {
        // Linux requires this
        return -PAL_ERROR_INVAL;
    }

    for (size_t i = 0; i < topo_info->numa_nodes_cnt; i++) {
        struct pal_numa_node_info* node = &topo_info->numa_nodes[i];
        coerce_untrusted_bool(&node->is_online);
        if (node->is_online) {
            for (size_t j = 0; j < HUGEPAGES_MAX; j++) {
                size_t unused; // can't use __builtin_mul_overflow_p because clang doesn't have it.
                if (__builtin_mul_overflow(node->nr_hugepages[j], hugepage_size[j], &unused))
                    return -PAL_ERROR_INVAL;
            }
        } else {
            /* Not required, just a hardening in case we accidentally accessed offline node's
             * fields. */
            for (size_t j = 0; j < HUGEPAGES_MAX; j++)
                node->nr_hugepages[j] = 0;
        }
    }

    for (size_t i = 0; i < topo_info->numa_nodes_cnt; i++) {
        /* Note: Linux doesn't guarantee that distance i -> i is 0, so we aren't checking this (it's
         * actually non-zero on all machines we have). */
        for (size_t j = 0; j < topo_info->numa_nodes_cnt; j++) {
            if (   topo_info->numa_distance_matrix[i*topo_info->numa_nodes_cnt + j]
                != topo_info->numa_distance_matrix[j*topo_info->numa_nodes_cnt + i])
                return -PAL_ERROR_INVAL;
        }
    }
    return 0;
}

static int import_and_sanitize_topo_info(struct pal_topo_info* uptr_topo_info) {
    /* Import topology information via an untrusted pointer. This is only a shallow copy and we use
     * this temp variable to do deep copy into `g_pal_public_state.topo_info` */
    struct pal_topo_info shallow_topo_info;
    if (!sgx_copy_to_enclave(&shallow_topo_info, sizeof(shallow_topo_info),
                             uptr_topo_info, sizeof(*uptr_topo_info))) {
        return -PAL_ERROR_DENIED;
    }

    struct pal_topo_info* topo_info = &g_pal_public_state.topo_info;

    size_t caches_cnt     = shallow_topo_info.caches_cnt;
    size_t threads_cnt    = shallow_topo_info.threads_cnt;
    size_t cores_cnt      = shallow_topo_info.cores_cnt;
    size_t sockets_cnt    = shallow_topo_info.sockets_cnt;
    size_t numa_nodes_cnt = shallow_topo_info.numa_nodes_cnt;

    struct pal_cache_info*      caches     = sgx_import_array_to_enclave(shallow_topo_info.caches,     sizeof(*caches),     caches_cnt);
    struct pal_cpu_thread_info* threads    = sgx_import_array_to_enclave(shallow_topo_info.threads,    sizeof(*threads),    threads_cnt);
    struct pal_cpu_core_info*   cores      = sgx_import_array_to_enclave(shallow_topo_info.cores,      sizeof(*cores),      cores_cnt);
    struct pal_socket_info*     sockets    = sgx_import_array_to_enclave(shallow_topo_info.sockets,    sizeof(*sockets),    sockets_cnt);
    struct pal_numa_node_info*  numa_nodes = sgx_import_array_to_enclave(shallow_topo_info.numa_nodes, sizeof(*numa_nodes), numa_nodes_cnt);

    size_t* distances = sgx_import_array2d_to_enclave(shallow_topo_info.numa_distance_matrix,
                                                      sizeof(*distances),
                                                      numa_nodes_cnt,
                                                      numa_nodes_cnt);
    if (!caches || !threads || !cores || !sockets || !numa_nodes || !distances) {
        return -PAL_ERROR_NOMEM;
    }

    topo_info->caches = caches;
    topo_info->threads = threads;
    topo_info->cores = cores;
    topo_info->sockets = sockets;
    topo_info->numa_nodes = numa_nodes;
    topo_info->numa_distance_matrix = distances;

    topo_info->caches_cnt = caches_cnt;
    topo_info->threads_cnt = threads_cnt;
    topo_info->cores_cnt = cores_cnt;
    topo_info->sockets_cnt = sockets_cnt;
    topo_info->numa_nodes_cnt = numa_nodes_cnt;

    return sanitize_topo_info(topo_info);
}

extern void* g_enclave_base;
extern void* g_enclave_top;
extern bool g_allowed_files_warn;

static int print_warnings_on_insecure_configs(PAL_HANDLE parent_process) {
    int ret;

    if (parent_process) {
        /* Warn only in the first process. */
        return 0;
    }

    /* TODO: `sgx.insecure__protected_files_key` is deprecated in v1.2, remove two versions
     * later. */

    bool verbose_log_level    = false;
    bool sgx_debug            = false;
    bool use_cmdline_argv     = false;
    bool use_host_env         = false;
    bool disable_aslr         = false;
    bool allow_eventfd        = false;
    bool allow_all_files      = false;
    bool use_allowed_files    = g_allowed_files_warn;
    bool protected_files_key  = false;
    bool encrypted_files_keys = false;

    char* log_level_str = NULL;
    char* protected_files_key_str = NULL;

    ret = toml_string_in(g_pal_public_state.manifest_root, "loader.log_level", &log_level_str);
    if (ret < 0)
        goto out;
    if (log_level_str && strcmp(log_level_str, "none") && strcmp(log_level_str, "error"))
        verbose_log_level = true;

    ret = toml_bool_in(g_pal_public_state.manifest_root, "sgx.debug",
                       /*defaultval=*/false, &sgx_debug);
    if (ret < 0)
        goto out;

    ret = toml_bool_in(g_pal_public_state.manifest_root, "loader.insecure__use_cmdline_argv",
                       /*defaultval=*/false, &use_cmdline_argv);
    if (ret < 0)
        goto out;

    ret = toml_bool_in(g_pal_public_state.manifest_root, "loader.insecure__use_host_env",
                       /*defaultval=*/false, &use_host_env);
    if (ret < 0)
        goto out;

    ret = toml_bool_in(g_pal_public_state.manifest_root, "loader.insecure__disable_aslr",
                       /*defaultval=*/false, &disable_aslr);
    if (ret < 0)
        goto out;

    ret = toml_bool_in(g_pal_public_state.manifest_root, "sys.insecure__allow_eventfd",
                       /*defaultval=*/false, &allow_eventfd);
    if (ret < 0)
        goto out;

    if (get_file_check_policy() == FILE_CHECK_POLICY_ALLOW_ALL_BUT_LOG)
        allow_all_files = true;

    ret = toml_string_in(g_pal_public_state.manifest_root, "sgx.insecure__protected_files_key",
                         &protected_files_key_str);
    if (ret < 0)
        goto out;
    if (protected_files_key_str)
        protected_files_key = true;

    toml_table_t* manifest_fs = toml_table_in(g_pal_public_state.manifest_root, "fs");
    if (manifest_fs) {
        toml_table_t* manifest_fs_keys = toml_table_in(manifest_fs, "insecure__keys");
        if (manifest_fs_keys) {
            ret = toml_table_nkval(manifest_fs_keys);
            if (ret < 0)
                goto out;

            if (ret > 0)
                encrypted_files_keys = true;
        }
    }

    if (!verbose_log_level && !sgx_debug && !use_cmdline_argv && !use_host_env && !disable_aslr &&
            !allow_eventfd && !allow_all_files && !use_allowed_files && !protected_files_key &&
            !encrypted_files_keys) {
        /* there are no insecure configurations, skip printing */
        ret = 0;
        goto out;
    }

    log_always("-------------------------------------------------------------------------------"
               "----------------------------------------");
    log_always("Gramine detected the following insecure configurations:\n");

    if (sgx_debug)
        log_always("  - sgx.debug = true                           "
                   "(this is a debug enclave)");

    if (verbose_log_level)
        log_always("  - loader.log_level = warning|debug|trace|all "
                   "(verbose log level, may leak information)");

    if (use_cmdline_argv)
        log_always("  - loader.insecure__use_cmdline_argv = true   "
                   "(forwarding command-line args from untrusted host to the app)");

    if (use_host_env)
        log_always("  - loader.insecure__use_host_env = true       "
                   "(forwarding environment vars from untrusted host to the app)");

    if (disable_aslr)
        log_always("  - loader.insecure__disable_aslr = true       "
                   "(Address Space Layout Randomization is disabled)");

    if (allow_eventfd)
        log_always("  - sys.insecure__allow_eventfd = true         "
                   "(host-based eventfd is enabled)");

    if (allow_all_files)
        log_always("  - sgx.file_check_policy = allow_all_but_log  "
                   "(all files are passed through from untrusted host without verification)");

    if (use_allowed_files)
        log_always("  - sgx.allowed_files = [ ... ]                "
                   "(some files are passed through from untrusted host without verification)");

    if (protected_files_key)
        log_always("  - sgx.insecure__protected_files_key = \"...\"  "
                   "(key hardcoded in manifest)");

    if (encrypted_files_keys)
        log_always("  - fs.insecure__keys.* = \"...\"                "
                   "(keys hardcoded in manifest)");

    log_always("\nGramine will continue application execution, but this configuration must not be "
               "used in production!");
    log_always("-------------------------------------------------------------------------------"
               "----------------------------------------\n");

    ret = 0;
out:
    free(log_level_str);
    free(protected_files_key_str);
    return ret;
}

__attribute_no_sanitize_address
static void do_preheat_enclave(void) {
    for (uint8_t* i = g_pal_linuxsgx_state.heap_min; i < (uint8_t*)g_pal_linuxsgx_state.heap_max;
             i += g_page_size) {
        READ_ONCE(*(size_t*)i);
    }
}

/* Gramine uses GCC's stack protector that looks for a canary at gs:[0x8], but this function starts
 * with a default canary and then updates it to a random one, so we disable stack protector here */
__attribute_no_stack_protector
noreturn void pal_linux_main(char* uptr_libpal_uri, size_t libpal_uri_len, char* uptr_args,
                             size_t args_size, char* uptr_env, size_t env_size,
                             int parent_stream_fd, sgx_target_info_t* uptr_qe_targetinfo,
                             struct pal_topo_info* uptr_topo_info) {
    /* All our arguments are coming directly from the urts. We are responsible to check them. */
    int ret;

    /* Relocate PAL */
    ret = setup_pal_binary();
    if (ret < 0) {
        log_error("Relocation of the PAL binary failed: %d", ret);
        ocall_exit(1, /*is_exitgroup=*/true);
    }

    uint64_t start_time;
    ret = _DkSystemTimeQuery(&start_time);
    if (ret < 0) {
        log_error("_DkSystemTimeQuery() failed: %d", ret);
        ocall_exit(1, /*is_exitgroup=*/true);
    }

    call_init_array();

    /* Initialize alloc_align as early as possible, a lot of PAL APIs depend on this being set. */
    g_pal_public_state.alloc_align = g_page_size;
    assert(IS_POWER_OF_2(g_pal_public_state.alloc_align));

    g_pal_linuxsgx_state.heap_min = GET_ENCLAVE_TLS(heap_min);
    g_pal_linuxsgx_state.heap_max = GET_ENCLAVE_TLS(heap_max);

    /* Skip URI_PREFIX_FILE. */
    if (libpal_uri_len < URI_PREFIX_FILE_LEN) {
        log_error("Invalid libpal_uri length (missing \"%s\" prefix?)", URI_PREFIX_FILE);
        ocall_exit(1, /*is_exitgroup=*/true);
    }
    libpal_uri_len -= URI_PREFIX_FILE_LEN;
    uptr_libpal_uri += URI_PREFIX_FILE_LEN;

    /* At this point we don't yet have memory manager, so we cannot allocate memory dynamically. */
    static char libpal_path[1024 + 1];
    if (libpal_uri_len >= sizeof(libpal_path)
            || !sgx_copy_to_enclave(libpal_path, sizeof(libpal_path) - 1, uptr_libpal_uri,
                                    libpal_uri_len)) {
        log_error("Copying libpal_path into the enclave failed");
        ocall_exit(1, /*is_exitgroup=*/true);
    }
    libpal_path[libpal_uri_len] = '\0';

    /* Now that we have `libpal_path`, set name for PAL map */
    set_pal_binary_name(libpal_path);

    /* We can't verify the following arguments from the urts. So we copy them directly but need to
     * be careful when we use them. */
    if (!sgx_copy_to_enclave(&g_pal_linuxsgx_state.qe_targetinfo,
                             sizeof(g_pal_linuxsgx_state.qe_targetinfo),
                             uptr_qe_targetinfo,
                             sizeof(*uptr_qe_targetinfo))) {
        log_error("Copying qe_targetinfo into the enclave failed");
        ocall_exit(1, /*is_exitgroup=*/true);
    }

    /* Set up page allocator and slab manager. There is no need to provide any initial memory pool,
     * because the slab manager can use normal allocations (`_DkVirtualMemoryAlloc`) right away. */
    init_slab_mgr(/*mem_pool=*/NULL, /*mem_pool_size=*/0);
    init_untrusted_slab_mgr();

    /* initialize enclave properties */
    ret = init_enclave();
    if (ret) {
        log_error("Failed to initialize enclave properties: %d", ret);
        ocall_exit(1, /*is_exitgroup=*/true);
    }

    if (args_size > MAX_ARGS_SIZE || env_size > MAX_ENV_SIZE) {
        log_error("Invalid args_size (%lu) or env_size (%lu)", args_size, env_size);
        ocall_exit(1, /*is_exitgroup=*/true);
    }
    const char** arguments = make_argv_list(uptr_args, args_size);
    if (!arguments) {
        log_error("Creating arguments failed");
        ocall_exit(1, /*is_exitgroup=*/true);
    }
    const char** environments = make_argv_list(uptr_env, env_size);
    if (!environments) {
        log_error("Creating environments failed");
        ocall_exit(1, /*is_exitgroup=*/true);
    }

    SET_ENCLAVE_TLS(ready_for_exceptions, 1UL);

    /* initialize "Invariant TSC" HW feature for fast and accurate gettime and immediately probe
     * RDTSC instruction inside SGX enclave (via dummy get_tsc) -- it is possible that
     * the CPU supports invariant TSC but doesn't support executing RDTSC inside SGX enclave, in
     * this case the SIGILL exception is generated and leads to emulate_rdtsc_and_print_warning()
     * which unsets invariant TSC, and we end up falling back to the slower ocall_gettime() */
    init_tsc();
    (void)get_tsc(); /* must be after `ready_for_exceptions=1` since it may generate SIGILL */

    /* initialize master key (used for pipes' encryption for all enclaves of an application); it
     * will be overwritten below in init_child_process() with inherited-from-parent master key if
     * this enclave is child */
    ret = _DkRandomBitsRead(&g_master_key, sizeof(g_master_key));
    if (ret < 0) {
        log_error("_DkRandomBitsRead failed: %d", ret);
        ocall_exit(1, /*is_exitgroup=*/true);
    }

    /* if there is a parent, create parent handle */
    PAL_HANDLE parent = NULL;
    uint64_t instance_id = 0;
    if (parent_stream_fd != -1) {
        if ((ret = init_child_process(parent_stream_fd, &parent, &instance_id)) < 0) {
            log_error("Failed to initialize child process: %d", ret);
            ocall_exit(1, /*is_exitgroup=*/true);
        }
    }

    uint64_t manifest_size = GET_ENCLAVE_TLS(manifest_size);
    void* manifest_addr = g_enclave_top - ALIGN_UP_PTR_POW2(manifest_size, g_page_size);

    ret = add_preloaded_range((uintptr_t)manifest_addr, (uintptr_t)manifest_addr + manifest_size,
                              "manifest");
    if (ret < 0) {
        log_error("Failed to initialize manifest preload range: %d", ret);
        ocall_exit(1, /*is_exitgroup=*/true);
    }

    /* TOML parser (for whatever reason) allocates a lot of memory when parsing the manifest into an
     * in-memory struct. We heuristically pre-allocate additional PAL internal memory if the
     * manifest file looks large enough. Hopefully below sizes are sufficient for any manifest.
     *
     * FIXME: this is a quick hack, we need proper memory allocation in PAL. */
    if (manifest_size > 10 * 1024 * 1024) {
        log_always("Detected a huge manifest, preallocating 128MB of internal memory.");
        g_pal_internal_mem_size += 128 * 1024 * 1024; /* 10MB manifest -> 64 + 128 MB PAL mem */
    } else if (manifest_size > 5 * 1024 * 1024) {
        log_always("Detected a huge manifest, preallocating 64MB of internal memory.");
        g_pal_internal_mem_size += 64 * 1024 * 1024; /* 5MB manifest -> 64 + 64 MB PAL mem */
    }

    /* parse manifest */
    char errbuf[256];
    toml_table_t* manifest_root = toml_parse(manifest_addr, errbuf, sizeof(errbuf));
    if (!manifest_root) {
        log_error("PAL failed at parsing the manifest: %s", errbuf);
        ocall_exit(1, /*is_exitgroup=*/true);
    }
    g_pal_common_state.raw_manifest_data = manifest_addr;
    g_pal_public_state.manifest_root = manifest_root;

    /* Get host topology information only for the first process. This information will be
     * checkpointed and restored during forking of the child process(es). */
    if (parent_stream_fd < 0) {
        /* parse and store host topology info into g_pal_public_state struct */
        ret = import_and_sanitize_topo_info(uptr_topo_info);
        if (ret < 0) {
            log_error("Failed to copy and sanitize topology information");
            ocall_exit(1, /*is_exitgroup=*/true);
        }
    }

    bool preheat_enclave;
    ret = toml_bool_in(g_pal_public_state.manifest_root, "sgx.preheat_enclave",
                       /*defaultval=*/false, &preheat_enclave);
    if (ret < 0) {
        log_error("Cannot parse 'sgx.preheat_enclave' (the value must be `true` or `false`)");
        ocall_exit(1, /*is_exitgroup=*/true);
    }
    if (preheat_enclave)
        do_preheat_enclave();

    /* For backward compatibility, `loader.pal_internal_mem_size` does not include
     * PAL_INITIAL_MEM_SIZE */
    size_t extra_mem_size;
    ret = toml_sizestring_in(g_pal_public_state.manifest_root, "loader.pal_internal_mem_size",
                             /*defaultval=*/0, &extra_mem_size);
    if (ret < 0) {
        log_error("Cannot parse 'loader.pal_internal_mem_size'");
        ocall_exit(1, /*is_exitgroup=*/true);
    }

    if (extra_mem_size + PAL_INITIAL_MEM_SIZE < g_pal_internal_mem_size) {
        log_error("Too small `loader.pal_internal_mem_size`, need at least %luMB because the "
                  "manifest is large",
                  (g_pal_internal_mem_size - PAL_INITIAL_MEM_SIZE) / 1024 / 1024);
        ocall_exit(1, /*is_exitgroup=*/true);
    }
    g_pal_internal_mem_size = extra_mem_size + PAL_INITIAL_MEM_SIZE;

    if ((ret = init_seal_key_material()) < 0) {
        log_error("Failed to initialize SGX sealing key material: %d", ret);
        ocall_exit(1, /*is_exitgroup=*/true);
    }

    if ((ret = init_file_check_policy()) < 0) {
        log_error("Failed to load the file check policy: %d", ret);
        ocall_exit(1, /*is_exitgroup=*/true);
    }

    if ((ret = init_allowed_files()) < 0) {
        log_error("Failed to initialize allowed files: %d", ret);
        ocall_exit(1, /*is_exitgroup=*/true);
    }

    if ((ret = init_trusted_files()) < 0) {
        log_error("Failed to initialize trusted files: %d", ret);
        ocall_exit(1, /*is_exitgroup=*/true);
    }

    /* this should be placed *after all* initialize-from-manifest routines */
    if ((ret = print_warnings_on_insecure_configs(parent)) < 0) {
        log_error("Cannot parse the manifest (while checking for insecure configurations)");
        ocall_exit(1, /*is_exitgroup=*/true);
    }

    /* set up thread handle */
    PAL_HANDLE first_thread = calloc(1, HANDLE_SIZE(thread));
    if (!first_thread) {
        log_error("Out of memory");
        ocall_exit(1, /*is_exitgroup=*/true);
    }

    init_handle_hdr(first_thread, PAL_TYPE_THREAD);
    first_thread->thread.tcs = g_enclave_base + GET_ENCLAVE_TLS(tcs_offset);
    /* child threads are assigned TIDs 2,3,...; see pal_start_thread() */
    first_thread->thread.tid = 1;
    g_pal_public_state.first_thread = first_thread;
    SET_ENCLAVE_TLS(thread, &first_thread->thread);

    uint64_t stack_protector_canary;
    ret = _DkRandomBitsRead(&stack_protector_canary, sizeof(stack_protector_canary));
    if (ret < 0) {
        log_error("_DkRandomBitsRead failed: %d", ret);
        ocall_exit(1, /*is_exitgroup=*/true);
    }
    pal_set_tcb_stack_canary(stack_protector_canary);

    assert(!g_pal_linuxsgx_state.enclave_initialized);
    g_pal_linuxsgx_state.enclave_initialized = true;

    /* call main function */
    pal_main(instance_id, parent, first_thread, arguments, environments);
}
