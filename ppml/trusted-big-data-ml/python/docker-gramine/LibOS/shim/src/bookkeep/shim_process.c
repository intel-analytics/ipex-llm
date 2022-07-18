/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

#include "api.h"
#include "list.h"
#include "shim_checkpoint.h"
#include "shim_fs.h"
#include "shim_handle.h"
#include "shim_lock.h"
#include "shim_process.h"
#include "shim_signal.h"
#include "shim_thread.h"

typedef bool (*child_cmp_t)(const struct shim_child_process*, unsigned long);

struct shim_process g_process = { .pid = 0 };

int init_process(int argc, const char** argv) {
    if (g_process.pid) {
        /* `g_process` is already initialized, e.g. via checkpointing code. */
        return 0;
    }

    /* If init_* function fails, then the whole process should die, so we do not need to clean-up
     * on errors. */
    if (!create_lock(&g_process.children_lock)) {
        return -ENOMEM;
    }
    if (!create_lock(&g_process.fs_lock)) {
        return -ENOMEM;
    }

    /* `pid` and `pgid` are initialized together with the first thread. */
    g_process.ppid = 0;

    INIT_LISTP(&g_process.children);
    INIT_LISTP(&g_process.zombies);

    g_process.wait_queue = NULL;

    /* default Linux umask */
    g_process.umask = 0022;

    struct shim_dentry* dent = NULL;
    lock(&g_dcache_lock);
    int ret = path_lookupat(/*start=*/NULL, "/", LOOKUP_FOLLOW | LOOKUP_DIRECTORY, &dent);
    unlock(&g_dcache_lock);
    if (ret < 0) {
        log_error("Could not set up dentry for \"/\", something is seriously broken.");
        return ret;
    }
    g_process.root = dent;

    /* Temporarily set `cwd` to `root`. It will be updated if necessary in `init_mount`. */
    get_dentry(g_process.root);
    g_process.cwd = g_process.root;

    /* `g_process.exec` will be initialized later on (in `init_important_handles`). */
    g_process.exec = NULL;

    /* The command line arguments passed are stored in /proc/self/cmdline as part of the proc fs.
     * They are not separated by space, but by NUL instead. So, it is essential to maintain the
     * cmdline_size also as a member here. */

    g_process.cmdline_size = 0;
    memset(g_process.cmdline, '\0', ARRAY_SIZE(g_process.cmdline));
    size_t tmp_size = 0;

    for (int i = 0; i < argc; i++) {
        if (tmp_size + strlen(argv[i]) + 1 > ARRAY_SIZE(g_process.cmdline))
            return -ENOMEM;

        memcpy(g_process.cmdline + tmp_size, argv[i], strlen(argv[i]));
        tmp_size += strlen(argv[i]) + 1;
    }

    g_process.cmdline_size = tmp_size;

    return 0;
}

struct shim_child_process* create_child_process(void) {
    struct shim_child_process* child = calloc(1, sizeof(*child));
    if (!child) {
        return NULL;
    }

    INIT_LIST_HEAD(child, list);

    return child;
}

void destroy_child_process(struct shim_child_process* child) {
    assert(LIST_EMPTY(child, list));

    free(child);
}

void add_child_process(struct shim_child_process* child) {
    assert(LIST_EMPTY(child, list));
    lock(&g_process.children_lock);
    LISTP_ADD(child, &g_process.children, list);
    unlock(&g_process.children_lock);
}

static bool cmp_child_by_vmid(const struct shim_child_process* child, unsigned long arg) {
    IDTYPE vmid = (IDTYPE)arg;
    return child->vmid == vmid;
}

static bool cmp_child_by_pid(const struct shim_child_process* child, unsigned long arg) {
    IDTYPE pid = (IDTYPE)arg;
    return child->pid == pid;
}

static bool mark_child_exited(child_cmp_t child_cmp, unsigned long arg, IDTYPE child_uid,
                              int exit_code, int signal) {
    bool ret = false;
    int parent_signal = 0;
    IDTYPE child_pid = 0;
    struct shim_thread_queue* wait_queue = NULL;

    lock(&g_process.children_lock);

    struct shim_child_process* child = NULL;
    struct shim_child_process* tmp = NULL;
    LISTP_FOR_EACH_ENTRY_SAFE(child, tmp, &g_process.children, list) {
        if (child_cmp(child, arg)) {
            child->exit_code = exit_code;
            child->term_signal = signal;
            child->uid = child_uid;

            LISTP_DEL(child, &g_process.children, list);
            /* TODO: if SIGCHLD is ignored or has SA_NOCLDWAIT flag set, then the child should not
             * become a zombie. */
            LISTP_ADD(child, &g_process.zombies, list);

            parent_signal = child->child_termination_signal;
            child_pid = child->pid;

            wait_queue = g_process.wait_queue;
            g_process.wait_queue = NULL;

            ret = true;
            break;
        }
    }

    /* We send signal to our process while still holding the lock, so that no thread is able to
     * see 0 pending signals but still get an exited child info. */
    if (parent_signal) {
        siginfo_t info = {0};
        info.si_signo = parent_signal;
        info.si_pid = child_pid;
        info.si_uid = child_uid;
        fill_siginfo_code_and_status(&info, signal, exit_code);
        int x = kill_current_proc(&info);
        if (x < 0) {
            log_error("Sending child death signal failed: %d!", x);
        }
    }

    unlock(&g_process.children_lock);

    while (wait_queue) {
        struct shim_thread_queue* next = wait_queue->next;
        struct shim_thread* thread = wait_queue->thread;
        __atomic_store_n(&wait_queue->in_use, false, __ATOMIC_RELEASE);
        /* In theory the atomic release store above does not prevent hoisting of code to before it.
         * Here we rely on the order: first store to `in_use`, then wake the thread. Let's add
         * a compiler barrier to prevent compiler from messing with this (e.g. if `thread_wakeup`
         * gets inlined). */
        COMPILER_BARRIER();
        thread_wakeup(thread);
        put_thread(thread);
        wait_queue = next;
    }

    return ret;
}

bool mark_child_exited_by_vmid(IDTYPE vmid, IDTYPE child_uid, int exit_code, int signal) {
    return mark_child_exited(cmp_child_by_vmid, (unsigned long)vmid, child_uid, exit_code, signal);
}

bool mark_child_exited_by_pid(IDTYPE pid, IDTYPE child_uid, int exit_code, int signal) {
    return mark_child_exited(cmp_child_by_pid, (unsigned long)pid, child_uid, exit_code, signal);
}

bool is_zombie_process(IDTYPE pid) {
    bool found = false;
    lock(&g_process.children_lock);

    struct shim_child_process* zombie;
    LISTP_FOR_EACH_ENTRY(zombie, &g_process.zombies, list) {
        if (zombie->pid == pid) {
            found = true;
            goto out;
        }
    }

out:
    unlock(&g_process.children_lock);
    return found;
}

BEGIN_CP_FUNC(process_description) {
    __UNUSED(size);
    __UNUSED(objp);
    assert(size == sizeof(struct shim_process));

    struct shim_process* process = (struct shim_process*)obj;

    size_t children_count = 0;
    size_t zombies_count = 0;
    struct shim_child_process* child = NULL;
    LISTP_FOR_EACH_ENTRY(child, &process->children, list) {
        ++children_count;
    }
    struct shim_child_process* zombie = NULL;
    LISTP_FOR_EACH_ENTRY(zombie, &process->zombies, list) {
        ++zombies_count;
    }

    size_t off = ADD_CP_OFFSET(sizeof(struct shim_process) + sizeof(children_count)
                               + children_count * sizeof(*child)
                               + sizeof(zombies_count)
                               + zombies_count * sizeof(*zombie));
    char* data_ptr = (char*)(base + off);
    struct shim_process* new_process = (struct shim_process*)data_ptr;
    data_ptr += sizeof(*new_process);

    memset(new_process, '\0', sizeof(*new_process));

    new_process->pid = process->pid;
    new_process->ppid = process->ppid;
    new_process->pgid = process->pgid;

    /* copy cmdline (used by /proc/[pid]/cmdline) from the current process */
    memcpy(new_process->cmdline, g_process.cmdline, ARRAY_SIZE(g_process.cmdline));
    new_process->cmdline_size = g_process.cmdline_size;

    DO_CP_MEMBER(dentry, process, new_process, root);
    DO_CP_MEMBER(dentry, process, new_process, cwd);
    new_process->umask = process->umask;

    DO_CP_MEMBER(handle, process, new_process, exec);

    INIT_LISTP(&new_process->children);
    INIT_LISTP(&new_process->zombies);

    clear_lock(&new_process->fs_lock);
    clear_lock(&new_process->children_lock);

    *(size_t*)data_ptr = children_count;
    data_ptr += sizeof(size_t);
    struct shim_child_process* children = (struct shim_child_process*)data_ptr;
    data_ptr += children_count * sizeof(*children);
    size_t i = 0;
    LISTP_FOR_EACH_ENTRY(child, &process->children, list) {
        children[i] = *child;
        INIT_LIST_HEAD(&children[i], list);
        i++;
    }

    assert(i == children_count);

    *(size_t*)data_ptr = zombies_count;
    data_ptr += sizeof(size_t);
    struct shim_child_process* zombies = (struct shim_child_process*)data_ptr;
    data_ptr += zombies_count * sizeof(*zombies);
    i = 0;
    LISTP_FOR_EACH_ENTRY(zombie, &process->zombies, list) {
        zombies[i] = *zombie;
        INIT_LIST_HEAD(&zombies[i], list);
        i++;
    }

    assert(i == zombies_count);

    ADD_CP_FUNC_ENTRY(off);
}
END_CP_FUNC(process_description)

BEGIN_RS_FUNC(process_description) {
    struct shim_process* process = (void*)(base + GET_CP_FUNC_ENTRY());
    __UNUSED(offset);

    CP_REBASE(process->root);
    CP_REBASE(process->cwd);
    CP_REBASE(process->exec);

    if (process->exec) {
        get_handle(process->exec);
    }
    if (process->root) {
        get_dentry(process->root);
    }
    if (process->cwd) {
        get_dentry(process->cwd);
    }

    if (!create_lock(&process->fs_lock)) {
        return -ENOMEM;
    }
    if (!create_lock(&process->children_lock)) {
        destroy_lock(&process->fs_lock);
        return -ENOMEM;
    }

    /* We never checkpoint any thread wait queues, since after clone/execve there is only one thread
     * and by definition it is not waiting. */
    process->wait_queue = NULL;

    INIT_LISTP(&process->children);
    INIT_LISTP(&process->zombies);

    char* data_ptr = (char*)process + sizeof(*process);
    size_t children_count = *(size_t*)data_ptr;
    data_ptr += sizeof(children_count);
    struct shim_child_process* children = (struct shim_child_process*)data_ptr;
    data_ptr += children_count * sizeof(*children);
    for (size_t i = 0; i < children_count; i++) {
        LISTP_ADD_TAIL(&children[i], &process->children, list);
    }

    size_t zombies_count = *(size_t*)data_ptr;
    data_ptr += sizeof(zombies_count);
    struct shim_child_process* zombies = (struct shim_child_process*)data_ptr;
    data_ptr += zombies_count * sizeof(*zombies);
    for (size_t i = 0; i < zombies_count; i++) {
        LISTP_ADD_TAIL(&zombies[i], &process->zombies, list);
    }

    g_process = *process;
#ifdef DEBUG
    memset(process, '\xcc', sizeof(*process));
#endif
}
END_RS_FUNC(process_description)
