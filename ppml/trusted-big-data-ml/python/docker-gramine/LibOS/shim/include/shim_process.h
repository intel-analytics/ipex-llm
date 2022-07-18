/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */
#ifndef _SHIM_PROCESS_H
#define _SHIM_PROCESS_H

#include <stdbool.h>

#include "list.h"
#include "pal.h"
#include "shim_fs.h"
#include "shim_handle.h"
#include "shim_lock.h"
#include "shim_types.h"

#define CMDLINE_SIZE 4096

DEFINE_LIST(shim_child_process);
DEFINE_LISTP(shim_child_process);
struct shim_child_process {
    IDTYPE pid;
    IDTYPE vmid;

    /* Signal sent to the parent process (us) on this child termination. */
    int child_termination_signal;

    /* These 3 fields are set when the child terminates. */
    int exit_code;
    int term_signal;
    IDTYPE uid;

    LIST_TYPE(shim_child_process) list;
};

struct shim_process {
    /* These 2 fields are constant and can safely be read without any locks held. */
    IDTYPE pid;
    IDTYPE ppid;

    /* This field should be accessed atomically, so no lock needed. */
    IDTYPE pgid;

    /* Currently all threads share filesystem information. For more info check `CLONE_FS` flag in
     * `clone.c`. Protected by `fs_lock`. */
    struct shim_dentry* root;
    struct shim_dentry* cwd;
    mode_t umask;

    /* Handle to the executable file. Protected by `fs_lock`. */
    struct shim_handle* exec;

    /* Threads waiting for some child to exit. Protected by `children_lock`. */
    struct shim_thread_queue* wait_queue;

    /* List of child processes that are still running. Protected by `children_lock`. */
    LISTP_TYPE(shim_child_process) children;
    /* List of already exited children. Protected by `children_lock`. */
    LISTP_TYPE(shim_child_process) zombies;

    struct shim_lock children_lock;
    struct shim_lock fs_lock;

    /* Complete command line for the process, as reported by /proc/[pid]/cmdline; currently filled
     * once during initialization, using static buffer and restricted to CMDLINE_SIZE. This is
     * enough for current workloads but see https://github.com/gramineproject/gramine/issues/79. */
    char cmdline[CMDLINE_SIZE];
    size_t cmdline_size;
};

extern struct shim_process g_process;

int init_process(int argc, const char** argv);

/* Allocates a new child process structure, initializing all fields. */
struct shim_child_process* create_child_process(void);
/* Frees `child` with all accompanying resources. */
void destroy_child_process(struct shim_child_process* child);
/* Adds `child` to `g_process` children list. */
void add_child_process(struct shim_child_process* child);

/*
 * These 2 functions mark a child as exited, moving it from `children` list to `zombies` list
 * and generate a child-termination signal (if needed).
 * Return `true` if the child was found, `false` otherwise.
 */
bool mark_child_exited_by_vmid(IDTYPE vmid, IDTYPE child_uid, int exit_code, int signal);
bool mark_child_exited_by_pid(IDTYPE pid, IDTYPE child_uid, int exit_code, int signal);

/*!
 * \brief Check whether the process is a zombie process (terminated but not yet waited for).
 *
 * \param pid  PID of the process to check.
 *
 * Returns `true` if the process \p pid is found in the zombie list of `g_process`.
 */
bool is_zombie_process(IDTYPE pid);

#endif // _SHIM_PROCESS_H
