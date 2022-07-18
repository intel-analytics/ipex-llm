/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * This file defines the "pseudo" filesystem: a building block for implementing pseudo-FSes such as
 * `/proc`, `/sys` or `/dev`.
 *
 * A pseudo-filesystem is defined by creating a tree of `pseudo_node` nodes:
 *
 *     struct pseudo_node* root = pseudo_add_root_dir("proc");
 *     pseudo_add_link(root, "self", &proc_self_follow_link);
 *     pseudo_add_str(root, "cpuinfo", &proc_cpuinfo_load);
 *
 * It can then be mounted by providing the root name as the `uri` parameter:
 *
 *     struct shim_mount_params params = { .type = "pseudo", .path = "/proc", .uri = "proc" };
 *     ret = mount_fs(&params);
 *
 * See the documentation of `pseudo_node` structure for details.
 *
 * TODO: The pseudo filesystem currently does not implement proper invalidation for
 * dynamically-generated directory listings (such as `/proc/<pid>`), so stale dentries might remain
 * for files that shouldn't exist anymore.
 *
 * TODO: The string-backed files are kept per handle, not per dentry. This doesn't matter so much
 * for read-only files, but if a writable file is opened twice, changes will not propagate between
 * handles.
 */

#ifndef SHIM_FS_PSEUDO_H_
#define SHIM_FS_PSEUDO_H_

#include "list.h"
#include "perm.h"
#include "shim_fs.h"

/* Node types */
enum pseudo_type {
    /* Directory */
    PSEUDO_DIR = 1,

    /* Symbolic link */
    PSEUDO_LINK = 2,

    /* String-backed file */
    PSEUDO_STR = 3,

    /* Character device */
    PSEUDO_DEV = 4,
};

/* Callbacks for PSEUDO_DEV nodes */
struct shim_dev_ops {
    int (*open)(struct shim_handle* hdl, struct shim_dentry* dent, int flags);
    int (*close)(struct shim_handle* hdl);
    ssize_t (*read)(struct shim_handle* hdl, void* buf, size_t count);
    ssize_t (*write)(struct shim_handle* hdl, const void* buf, size_t count);
    int (*flush)(struct shim_handle* hdl);
    int64_t (*seek)(struct shim_handle* hdl, int64_t offset, int whence);
    int (*truncate)(struct shim_handle* hdl, uint64_t len);
};

#define PSEUDO_PERM_DIR     PERM_r_xr_xr_x  /* default for directories */
#define PSEUDO_PERM_LINK    PERM_rwxrwxrwx  /* default for links */
#define PSEUDO_PERM_FILE_R  PERM_r__r__r__  /* default for all other files */
#define PSEUDO_PERM_FILE_RW PERM_rw_rw_rw_

/* Maximum count of created nodes (`pseudo_add_*` calls) */
#define PSEUDO_MAX_NODES 128

/* Used to represent cpumaps like "00000000,ffffffff,00000000,ffffffff".
 * Note: Used to allocate on stack; increase with caution or use malloc instead. */
#define PAL_SYSFS_MAP_FILESZ 256

/*
 * A node of the pseudo filesystem. A single node can describe either a single file, or a family of
 * files:
 *
 * - For a single file with a pre-determined name, set `name`
 * - For a single file with a pre-determined name that, depending on circumstances, might not exist,
 *   set `name` and provide the `name_exists` callback
 * - For any other case (e.g. a dynamically-generated list of files), keep `name` set to NULL, and
 *   provide both `name_exists` and `list_names` callbacks
 *
 * The constructors for `pseudo_node` (`pseudo_add_*`) take arguments for most commonly used fields.
 * The node can then be further customized by directly modifying other fields.
 *
 * NOTE: We assume that all Gramine processes within a single instance create exactly the same set
 * of nodes, in the same order, during initialization. This is because we use the node number (`id`)
 * for checkpointing.
 */
DEFINE_LIST(pseudo_node);
DEFINE_LISTP(pseudo_node);
struct pseudo_node {
    enum pseudo_type type;

    unsigned int id;

    LIST_TYPE(pseudo_node) siblings;

    struct pseudo_node* parent;

    /* File name. Can be NULL if the node provides the below callbacks. */
    const char* name;

    /* Returns true if a file with a given name exists. If both `name` and `name_exists` are
     * provided, both are used to determine if the file exists. */
    bool (*name_exists)(struct shim_dentry* parent, const char* name);

    /* Retrieves all file names for this node. Works the same as `readdir`. */
    int (*list_names)(struct shim_dentry* parent, readdir_callback_t callback, void* arg);

    /* File permissions. See `PSEUDO_PERM_*` above for defaults. */
    mode_t perm;

    /* Type-specific fields: */
    union {
        /* PSEUDO_DIR */
        struct {
            LISTP_TYPE(pseudo_node) children;
        } dir;

        /* PSEUDO_LINK */
        struct {
            /*
             * Reads link target. Should allocate a new (null-terminated) string: it will be freed
             * using `free`.
             *
             * If the callback is not provided, the `target` field will be used instead.
             */
            int (*follow_link)(struct shim_dentry* dent, char** out_target);

            const char* target;
        } link;

        /* PSEUDO_STR */
        struct {
            /*
             * Provides data for newly-opened file. Should allocate a new buffer: on file close, the
             * buffer will be freed using `free`.
             *
             * If the callback is not provided, the opened file will start out empty.
             */
            int (*load)(struct shim_dentry* dent, char** out_data, size_t* out_size);

            /*
             * Invoked on every write operation.
             *
             * This is used to implement writable files similar to the ones in Linux sysfs. The
             * callback is triggered on write, not on close, because otherwise there is no good way
             * to report errors to the application (which might even exit without closing the file
             * explicitly).
             *
             * As a consequence, pseudo-files that implement `save` are meant to be written using a
             * single `write` operation. For consistency with later reads, for such files we replace
             * existing content on every write.
             */
            int (*save)(struct shim_dentry* dent, const char* data, size_t size);
        } str;

        /* PSEUDO_DEV */
        struct {
            /* Callbacks to use with this device */
            struct shim_dev_ops dev_ops;

            /* Device major and minor numbers, used with `stat` */
            unsigned int major;
            unsigned int minor;
        } dev;
    };
};

/*
 * \brief Convert a string to number (for use in paths).
 *
 * \param      str        The string.
 * \param      max_value  Maximum value.
 * \param[out] out_value  On success, set to the parsed number.
 *
 * \returns 0 on success, -1 on failure.
 *
 * Recognizes a string that is a unique representation of a number (0 <= value <= max_value):
 * the string should be non-empty, consist only of digits, and have no leading zeroes.
 *
 * Provided for use in `name_exists` callback for recognizing pseudo-filesystem file names.
 */
int pseudo_parse_ulong(const char* str, unsigned long max_value, unsigned long* out_value);

struct pseudo_node* pseudo_add_root_dir(const char* name);

struct pseudo_node* pseudo_add_dir(struct pseudo_node* parent_ent, const char* name);

struct pseudo_node* pseudo_add_link(struct pseudo_node* parent_ent, const char* name,
                                    int (*follow_link)(struct shim_dentry*, char**));

struct pseudo_node* pseudo_add_str(struct pseudo_node* parent_ent, const char* name,
                                   int (*load)(struct shim_dentry*, char**, size_t*));

struct pseudo_node* pseudo_add_dev(struct pseudo_node* parent_ent, const char* name);

extern struct shim_fs pseudo_builtin_fs;

/* procfs */

int init_procfs(void);
int proc_meminfo_load(struct shim_dentry* dent, char** out_data, size_t* out_size);
int proc_cpuinfo_load(struct shim_dentry* dent, char** out_data, size_t* out_size);
int proc_stat_load(struct shim_dentry* dent, char** out_data, size_t* out_size);
int proc_self_follow_link(struct shim_dentry* dent, char** out_target);
bool proc_thread_pid_name_exists(struct shim_dentry* parent, const char* name);
int proc_thread_pid_list_names(struct shim_dentry* parent, readdir_callback_t callback, void* arg);
bool proc_thread_tid_name_exists(struct shim_dentry* parent, const char* name);
int proc_thread_tid_list_names(struct shim_dentry* parent, readdir_callback_t callback, void* arg);
int proc_thread_follow_link(struct shim_dentry* dent, char** out_target);
int proc_thread_maps_load(struct shim_dentry* dent, char** out_data, size_t* out_size);
int proc_thread_cmdline_load(struct shim_dentry* dent, char** out_data, size_t* out_size);
bool proc_thread_fd_name_exists(struct shim_dentry* parent, const char* name);
int proc_thread_fd_list_names(struct shim_dentry* parent, readdir_callback_t callback, void* arg);
int proc_thread_fd_follow_link(struct shim_dentry* dent, char** out_target);
bool proc_ipc_thread_pid_name_exists(struct shim_dentry* parent, const char* name);
int proc_ipc_thread_follow_link(struct shim_dentry* dent, char** out_target);

/* devfs */

int init_devfs(void);
int init_attestation(struct pseudo_node* dev);

/* sysfs */

int init_sysfs(void);
int sys_load(const char* str, char** out_data, size_t* out_size);
bool sys_resource_name_exists(struct shim_dentry* parent, const char* name);
int sys_resource_list_names(struct shim_dentry* parent, readdir_callback_t callback, void* arg);
int sys_node_general_load(struct shim_dentry* dent, char** out_data, size_t* out_size);
int sys_node_load(struct shim_dentry* dent, char** out_data, size_t* out_size);
int sys_cpu_general_load(struct shim_dentry* dent, char** out_data, size_t* out_size);
int sys_cpu_load_online(struct shim_dentry* dent, char** out_data, size_t* out_size);
int sys_cpu_load_topology(struct shim_dentry* dent, char** out_data, size_t* out_size);
int sys_cache_load(struct shim_dentry* dent, char** out_data, size_t* out_size);
bool sys_cpu_online_name_exists(struct shim_dentry* parent, const char* name);
bool sys_cpu_exists_only_if_online(struct shim_dentry* parent, const char* name);

/* Find resource number for a given name; e.g. `sys_resource_find(dent, "cpu", &n)` will search for
 * "cpu/cpuN" in the path of `dent`, and set `n` to the number N */
int sys_resource_find(struct shim_dentry* parent, const char* parent_name, unsigned int* out_num);

/* Print into the sysfs ranges format (e.g. "10-63,68,70-127\n"). */
int sys_print_as_ranges(char* buf, size_t buf_size, size_t count,
                        bool (*is_present)(size_t ind, const void* arg), const void* callback_arg);

/* Print into the sysfs bitmask format, with bitmask size based on `count`.
 * Example: count=64, indices 0-15,48-55 present: "00ff0000,0000ffff\n".
 */
int sys_print_as_bitmask(char* buf, size_t buf_size, size_t count,
                         bool (*is_present)(size_t ind, const void* arg),
                         const void* callback_arg);

#endif /* SHIM_FS_PSEUDO_H_ */
