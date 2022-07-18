/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * Implementation of `/proc/<pid>` and `/proc/<pid>/task/<tid>`, for the local process.
 */

#include "shim_fs.h"
#include "shim_fs_pseudo.h"
#include "shim_lock.h"
#include "shim_process.h"
#include "shim_thread.h"
#include "shim_types.h"
#include "shim_vma.h"

int proc_thread_follow_link(struct shim_dentry* dent, char** out_target) {
    __UNUSED(dent);

    lock(&g_process.fs_lock);

    const char* name = dent->name;
    if (strcmp(name, "root") == 0) {
        dent = g_process.root;
        get_dentry(dent);
    } else if (strcmp(name, "cwd") == 0) {
        dent = g_process.cwd;
        get_dentry(dent);
    } else if (strcmp(name, "exe") == 0) {
        dent = g_process.exec->dentry;
        if (dent)
            get_dentry(dent);
    }

    unlock(&g_process.fs_lock);

    if (!dent)
        return -ENOENT;

    int ret = dentry_abs_path(dent, out_target, /*size=*/NULL);
    if (ret < 0)
        goto out;

    ret = 0;
out:
    put_dentry(dent);
    return ret;
}

int proc_thread_maps_load(struct shim_dentry* dent, char** out_data, size_t* out_size) {
    __UNUSED(dent);

    int ret;
    size_t vma_count;
    struct shim_vma_info* vmas = NULL;
    ret = dump_all_vmas(&vmas, &vma_count, /*include_unmapped=*/false);
    if (ret < 0) {
        return ret;
    }

#define DEFAULT_VMA_BUFFER_SIZE 256

    char* buffer;
    size_t buffer_size = DEFAULT_VMA_BUFFER_SIZE, offset = 0;
    buffer = malloc(buffer_size);
    if (!buffer) {
        ret = -ENOMEM;
        goto err;
    }

    for (struct shim_vma_info* vma = vmas; vma < vmas + vma_count; vma++) {
        size_t old_offset = offset;
        uintptr_t start   = (uintptr_t)vma->addr;
        uintptr_t end     = (uintptr_t)vma->addr + vma->length;
        char pt[3]        = {
            (vma->prot & PROT_READ) ? 'r' : '-',
            (vma->prot & PROT_WRITE) ? 'w' : '-',
            (vma->prot & PROT_EXEC) ? 'x' : '-',
        };
        char pr = (vma->flags & MAP_PRIVATE) ? 'p' : 's';

#define ADDR_FMT(addr) ((addr) > 0xffffffff ? "%lx" : "%08lx")
#define EMIT(fmt...)                                                        \
    do {                                                                    \
        if (offset < buffer_size)                                           \
            offset += snprintf(buffer + offset, buffer_size - offset, fmt); \
    } while (0)

    retry_emit_vma:
        if (vma->file) {
            int dev_major = 0, dev_minor = 0;
            unsigned long ino = vma->file->dentry ? dentry_ino(vma->file->dentry) : 0;
            char* path = NULL;

            if (vma->file->dentry)
                dentry_abs_path(vma->file->dentry, &path, /*size=*/NULL);

            EMIT(ADDR_FMT(start), start);
            EMIT("-");
            EMIT(ADDR_FMT(end), end);
            EMIT(" %c%c%c%c %08lx %02d:%02d %lu %s\n", pt[0], pt[1], pt[2], pr, vma->file_offset,
                 dev_major, dev_minor, ino, path ? path : "[unknown]");

            free(path);
        } else {
            EMIT(ADDR_FMT(start), start);
            EMIT("-");
            EMIT(ADDR_FMT(end), end);
            if (vma->comment[0])
                EMIT(" %c%c%c%c 00000000 00:00 0 %s\n", pt[0], pt[1], pt[2], pr, vma->comment);
            else
                EMIT(" %c%c%c%c 00000000 00:00 0\n", pt[0], pt[1], pt[2], pr);
        }

        if (offset >= buffer_size) {
            char* new_buffer = malloc(buffer_size * 2);
            if (!new_buffer) {
                ret = -ENOMEM;
                goto err;
            }

            offset = old_offset;
            memcpy(new_buffer, buffer, old_offset);
            free(buffer);
            buffer = new_buffer;
            buffer_size *= 2;
            goto retry_emit_vma;
        }
    }

    *out_data = buffer;
    *out_size = offset;
    ret = 0;

err:
    if (ret < 0) {
        free(buffer);
    }
    if (vmas) {
        free_vma_info_array(vmas, vma_count);
    }
    return ret;
}

int proc_thread_cmdline_load(struct shim_dentry* dent, char** out_data, size_t* out_size) {
    __UNUSED(dent);

    size_t buffer_size = g_process.cmdline_size;
    char* buffer = malloc(buffer_size);
    if (!buffer) {
        return -ENOMEM;
    }

    memcpy(buffer, g_process.cmdline, buffer_size);
    *out_data = buffer;
    *out_size = buffer_size;
    return 0;
}

bool proc_thread_pid_name_exists(struct shim_dentry* parent, const char* name) {
    __UNUSED(parent);

    unsigned long pid;
    if (pseudo_parse_ulong(name, IDTYPE_MAX, &pid) < 0)
        return false;

    return pid == g_process.pid;
}

int proc_thread_pid_list_names(struct shim_dentry* parent, readdir_callback_t callback, void* arg) {
    __UNUSED(parent);
    IDTYPE pid = g_process.pid;
    char name[11];
    snprintf(name, sizeof(name), "%u", pid);
    int ret = callback(name, arg);
    if (ret < 0)
        return ret;

    return 0;
}

bool proc_thread_tid_name_exists(struct shim_dentry* parent, const char* name) {
    __UNUSED(parent);

    unsigned long tid;
    if (pseudo_parse_ulong(name, IDTYPE_MAX, &tid) < 0)
        return false;

    struct shim_thread* thread = lookup_thread(tid);
    if (!thread)
        return false;

    put_thread(thread);
    return true;
}

struct walk_thread_arg {
    readdir_callback_t callback;
    void* arg;
};

static int walk_cb(struct shim_thread* thread, void* arg) {
    struct walk_thread_arg* args = arg;

    IDTYPE pid = thread->tid;
    char name[11];
    snprintf(name, sizeof(name), "%u", pid);
    int ret = args->callback(name, args->arg);
    if (ret < 0)
        return ret;
    return 1;
}

int proc_thread_tid_list_names(struct shim_dentry* parent, readdir_callback_t callback, void* arg) {
    __UNUSED(parent);
    struct walk_thread_arg args = {
        .callback = callback,
        .arg = arg,
    };

    int ret = walk_thread_list(&walk_cb, &args, /*one_shot=*/false);
    if (ret < 0)
        return ret;

    return 0;
}

bool proc_thread_fd_name_exists(struct shim_dentry* parent, const char* name) {
    __UNUSED(parent);
    unsigned long fd;
    if (pseudo_parse_ulong(name, UINT32_MAX, &fd) < 0)
        return false;

    struct shim_handle_map* handle_map = get_thread_handle_map(NULL);
    assert(handle_map);
    lock(&handle_map->lock);

    if (fd > handle_map->fd_top || handle_map->map[fd] == NULL ||
            handle_map->map[fd]->handle == NULL) {
        unlock(&handle_map->lock);
        return false;
    }

    unlock(&handle_map->lock);
    return true;
}

int proc_thread_fd_list_names(struct shim_dentry* parent, readdir_callback_t callback, void* arg) {
    __UNUSED(parent);

    struct shim_handle_map* handle_map = get_thread_handle_map(NULL);
    assert(handle_map);
    lock(&handle_map->lock);

    int ret = 0;
    for (uint32_t i = 0; i <= handle_map->fd_top; i++)
        if (handle_map->map[i] && handle_map->map[i]->handle) {
            char name[11];
            snprintf(name, sizeof(name), "%u", i);
            if ((ret = callback(name, arg)) < 0)
                break;
        }

    unlock(&handle_map->lock);
    return ret;
}

/*
 * Human-readable string for a handle without attached dentry.
 *
 * TODO: Linux uses names like `pipe:[INODE]`, we could at least include more information whenever
 * we can (e.g. socket address).
 */
static char* describe_handle(struct shim_handle* hdl) {
    const char* str;
    switch (hdl->type) {
        case TYPE_CHROOT:  str = "chroot:[?]";  break;
        case TYPE_DEV:     str = "dev:[?]";     break;
        case TYPE_STR:     str = "str:[?]";     break;
        case TYPE_PSEUDO:  str = "pseudo:[?]";  break;
        case TYPE_PIPE:    str = "pipe:[?]";    break;
        case TYPE_SOCK:    str = "sock:[?]";    break;
        case TYPE_EPOLL:   str = "epoll:[?]";   break;
        case TYPE_EVENTFD: str = "eventfd:[?]"; break;
        default:           str = "unknown:[?]"; break;
    }
    return strdup(str);
}

int proc_thread_fd_follow_link(struct shim_dentry* dent, char** out_target) {
    unsigned long fd;
    if (pseudo_parse_ulong(dent->name, UINT32_MAX, &fd) < 0)
        return -ENOENT;

    struct shim_handle_map* handle_map = get_thread_handle_map(NULL);
    assert(handle_map);
    lock(&handle_map->lock);

    if (fd > handle_map->fd_top || handle_map->map[fd] == NULL ||
            handle_map->map[fd]->handle == NULL) {
        unlock(&handle_map->lock);
        return -ENOENT;
    }

    int ret;
    struct shim_handle* hdl = handle_map->map[fd]->handle;

    if (hdl->dentry) {
        ret = dentry_abs_path(hdl->dentry, out_target, /*size=*/NULL);
    } else {
        /* The handle does not correspond to a dentry. Do our best to provide a human-readable link
         * target. */
        *out_target = describe_handle(hdl);
        ret = *out_target ? 0 : -ENOMEM;
    }

    unlock(&handle_map->lock);

    return ret;
}
