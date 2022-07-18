/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains code to maintain bookkeeping for handles in library OS.
 */

#include "pal.h"
#include "shim_checkpoint.h"
#include "shim_fs.h"
#include "shim_fs_lock.h"
#include "shim_handle.h"
#include "shim_internal.h"
#include "shim_lock.h"
#include "shim_thread.h"
#include "stat.h"
#include "toml_utils.h"

static struct shim_lock handle_mgr_lock;

#define HANDLE_MGR_ALLOC 32

#define SYSTEM_LOCK()   lock(&handle_mgr_lock)
#define SYSTEM_UNLOCK() unlock(&handle_mgr_lock)
#define SYSTEM_LOCKED() locked(&handle_mgr_lock)

#define OBJ_TYPE struct shim_handle
#include "memmgr.h"

static MEM_MGR handle_mgr = NULL;

#define INIT_HANDLE_MAP_SIZE 32

//#define DEBUG_REF

static int init_tty_handle(struct shim_handle* hdl, bool write) {
    int flags = write ? (O_WRONLY | O_APPEND) : O_RDONLY;
    return open_namei(hdl, /*start=*/NULL, "/dev/tty", flags, LOOKUP_FOLLOW, /*found=*/NULL);
}

int open_executable(struct shim_handle* hdl, const char* path) {
    struct shim_dentry* dent = NULL;

    lock(&g_dcache_lock);
    int ret = path_lookupat(/*start=*/NULL, path, LOOKUP_FOLLOW, &dent);
    if (ret < 0) {
        goto out;
    }

    if (dent->inode->type != S_IFREG) {
        ret = -EACCES;
        goto out;
    }

    if (!(dent->inode->perm & S_IXUSR)) {
        ret = -EACCES;
        goto out;
    }

    ret = dentry_open(hdl, dent, O_RDONLY);
    if (ret < 0) {
        goto out;
    }

    ret = 0;
out:
    unlock(&g_dcache_lock);
    if (dent)
        put_dentry(dent);

    return ret;
}

static int init_exec_handle(void) {
    lock(&g_process.fs_lock);
    if (g_process.exec) {
        /* `g_process.exec` handle is already initialized if we did execve. See
         * `shim_do_execve_rtld`. */
        unlock(&g_process.fs_lock);
        return 0;
    }
    unlock(&g_process.fs_lock);

    /* Initialize `g_process.exec` based on `libos.entrypoint` manifest key. */
    char* entrypoint = NULL;
    const char* exec_path;
    struct shim_handle* hdl = NULL;
    int ret;

    /* Initialize `g_process.exec` based on `libos.entrypoint` manifest key. */
    assert(g_manifest_root);
    ret = toml_string_in(g_manifest_root, "libos.entrypoint", &entrypoint);
    if (ret < 0) {
        log_error("Cannot parse 'libos.entrypoint'");
        ret = -EINVAL;
        goto out;
    }
    if (!entrypoint) {
        log_error("'libos.entrypoint' must be specified in the manifest");
        ret = -EINVAL;
        goto out;
    }

    exec_path = entrypoint;

    if (strstartswith(exec_path, URI_PREFIX_FILE)) {
        /* Technically we could skip this check, but it's something easy to confuse with
         * loader.entrypoint, so better to have this handled nicely for the users. */
        log_error("'libos.entrypoint' should be an in-Gramine path, not URI.");
        ret = -EINVAL;
        goto out;
    }

    hdl = get_new_handle();
    if (!hdl) {
        ret = -ENOMEM;
        goto out;
    }

    ret = open_executable(hdl, exec_path);
    if (ret < 0) {
        log_error("Error opening executable %s: %d", exec_path, ret);
        goto out;
    }

    lock(&g_process.fs_lock);
    g_process.exec = hdl;
    get_handle(hdl);
    unlock(&g_process.fs_lock);

    ret = 0;
out:
    free(entrypoint);
    if (hdl)
        put_handle(hdl);
    return ret;
}

static struct shim_handle_map* get_new_handle_map(uint32_t size);

static int __init_handle(struct shim_fd_handle** fdhdl, uint32_t fd, struct shim_handle* hdl,
                         int fd_flags);

static int __enlarge_handle_map(struct shim_handle_map* map, uint32_t size);

int init_handle(void) {
    if (!create_lock(&handle_mgr_lock)) {
        return -ENOMEM;
    }
    handle_mgr = create_mem_mgr(init_align_up(HANDLE_MGR_ALLOC));
    if (!handle_mgr) {
        return -ENOMEM;
    }
    return 0;
}

int init_important_handles(void) {
    int ret;
    struct shim_thread* thread = get_cur_thread();

    if (thread->handle_map)
        goto done;

    struct shim_handle_map* handle_map = get_thread_handle_map(thread);

    if (!handle_map) {
        handle_map = get_new_handle_map(INIT_HANDLE_MAP_SIZE);
        if (!handle_map)
            return -ENOMEM;

        set_handle_map(thread, handle_map);
        put_handle_map(handle_map);
    }

    /* `handle_map` is set in current thread, no need to increase ref-count. */

    lock(&handle_map->lock);

    if (handle_map->fd_size < 3) {
        ret = __enlarge_handle_map(handle_map, INIT_HANDLE_MAP_SIZE);
        if (ret < 0) {
            unlock(&handle_map->lock);
            return ret;
        }
    }

    /* initialize stdin */
    if (!HANDLE_ALLOCATED(handle_map->map[0])) {
        struct shim_handle* stdin_hdl = get_new_handle();
        if (!stdin_hdl) {
            unlock(&handle_map->lock);
            return -ENOMEM;
        }

        if ((ret = init_tty_handle(stdin_hdl, /*write=*/false)) < 0) {
            unlock(&handle_map->lock);
            put_handle(stdin_hdl);
            return ret;
        }

        __init_handle(&handle_map->map[0], /*fd=*/0, stdin_hdl, /*flags=*/0);
        put_handle(stdin_hdl);
    }

    /* initialize stdout */
    if (!HANDLE_ALLOCATED(handle_map->map[1])) {
        struct shim_handle* stdout_hdl = get_new_handle();
        if (!stdout_hdl) {
            unlock(&handle_map->lock);
            return -ENOMEM;
        }

        if ((ret = init_tty_handle(stdout_hdl, /*write=*/true)) < 0) {
            unlock(&handle_map->lock);
            put_handle(stdout_hdl);
            return ret;
        }

        __init_handle(&handle_map->map[1], /*fd=*/1, stdout_hdl, /*flags=*/0);
        put_handle(stdout_hdl);
    }

    /* initialize stderr as duplicate of stdout */
    if (!HANDLE_ALLOCATED(handle_map->map[2])) {
        struct shim_handle* stdout_hdl = handle_map->map[1]->handle;
        __init_handle(&handle_map->map[2], /*fd=*/2, stdout_hdl, /*flags=*/0);
    }

    if (handle_map->fd_top == FD_NULL || handle_map->fd_top < 2)
        handle_map->fd_top = 2;

    unlock(&handle_map->lock);

done:
    return init_exec_handle();
}

struct shim_handle* __get_fd_handle(uint32_t fd, int* fd_flags, struct shim_handle_map* map) {
    assert(map);
    assert(locked(&map->lock));

    struct shim_fd_handle* fd_handle = NULL;

    if (map->fd_top != FD_NULL && fd <= map->fd_top) {
        fd_handle = map->map[fd];
        if (!HANDLE_ALLOCATED(fd_handle))
            return NULL;

        if (fd_flags)
            *fd_flags = fd_handle->flags;

        return fd_handle->handle;
    }
    return NULL;
}

struct shim_handle* get_fd_handle(uint32_t fd, int* fd_flags, struct shim_handle_map* map) {
    map = map ?: get_thread_handle_map(NULL);
    assert(map);

    struct shim_handle* hdl = NULL;
    lock(&map->lock);
    if ((hdl = __get_fd_handle(fd, fd_flags, map)))
        get_handle(hdl);
    unlock(&map->lock);
    return hdl;
}

struct shim_handle* __detach_fd_handle(struct shim_fd_handle* fd, int* flags,
                                       struct shim_handle_map* map) {
    assert(locked(&map->lock));

    struct shim_handle* handle = NULL;

    if (HANDLE_ALLOCATED(fd)) {
        uint32_t vfd = fd->vfd;
        handle = fd->handle;
        int handle_fd = vfd;
        if (flags)
            *flags = fd->flags;

        fd->vfd    = FD_NULL;
        fd->handle = NULL;
        fd->flags  = 0;

        if (vfd == map->fd_top)
            do {
                if (vfd == 0) {
                    map->fd_top = FD_NULL;
                    break;
                }
                map->fd_top = vfd - 1;
                vfd--;
            } while (!HANDLE_ALLOCATED(map->map[vfd]));

        delete_epoll_items_for_fd(handle_fd, handle);
    }

    return handle;
}

struct shim_handle* detach_fd_handle(uint32_t fd, int* flags, struct shim_handle_map* handle_map) {
    struct shim_handle* handle = NULL;

    if (!handle_map && !(handle_map = get_thread_handle_map(NULL)))
        return NULL;

    lock(&handle_map->lock);

    if (fd < handle_map->fd_size)
        handle = __detach_fd_handle(handle_map->map[fd], flags, handle_map);

    unlock(&handle_map->lock);

    if (handle && handle->dentry) {
        /* Clear POSIX locks for a file. We are required to do that every time a FD is closed, even
         * if the process holds other handles for that file, or duplicated FDs for the same
         * handle. */
        struct posix_lock pl = {
            .type = F_UNLCK,
            .start = 0,
            .end = FS_LOCK_EOF,
            .pid = g_process.pid,
        };
        int ret = posix_lock_set(handle->dentry, &pl, /*block=*/false);
        if (ret < 0)
            log_warning("error releasing locks: %d", ret);
    }

    return handle;
}

struct shim_handle* get_new_handle(void) {
    struct shim_handle* new_handle =
        get_mem_obj_from_mgr_enlarge(handle_mgr, size_align_up(HANDLE_MGR_ALLOC));
    if (!new_handle)
        return NULL;

    memset(new_handle, 0, sizeof(struct shim_handle));
    REF_SET(new_handle->ref_count, 1);
    if (!create_lock(&new_handle->lock)) {
        free_mem_obj_to_mgr(handle_mgr, new_handle);
        return NULL;
    }
    if (!create_lock(&new_handle->pos_lock)) {
        destroy_lock(&new_handle->lock);
        free_mem_obj_to_mgr(handle_mgr, new_handle);
        return NULL;
    }
    INIT_LISTP(&new_handle->epoll_items);
    new_handle->epoll_items_count = 0;
    return new_handle;
}

static int __init_handle(struct shim_fd_handle** fdhdl, uint32_t fd, struct shim_handle* hdl,
                         int fd_flags) {
    struct shim_fd_handle* new_handle = *fdhdl;
    assert((fd_flags & ~FD_CLOEXEC) == 0);  // The only supported flag right now

    if (!new_handle) {
        new_handle = malloc(sizeof(struct shim_fd_handle));
        if (!new_handle)
            return -ENOMEM;
        *fdhdl = new_handle;
    }

    new_handle->vfd   = fd;
    new_handle->flags = fd_flags;
    get_handle(hdl);
    new_handle->handle = hdl;
    return 0;
}

/*
 * Helper function for set_new_fd_handle*(). If find_free is true, tries to find the first free fd
 * (starting from the provided one), otherwise, tries to use fd as-is.
 */
static int __set_new_fd_handle(uint32_t fd, struct shim_handle* hdl, int fd_flags,
                               struct shim_handle_map* handle_map, bool find_free) {
    int ret;

    if (!handle_map && !(handle_map = get_thread_handle_map(NULL)))
        return -EBADF;

    lock(&handle_map->lock);

    if (handle_map->fd_top != FD_NULL) {
        assert(handle_map->map);
        if (find_free) {
            // find first free fd
            while (fd <= handle_map->fd_top && HANDLE_ALLOCATED(handle_map->map[fd])) {
                fd++;
            }
        } else {
            // check if requested fd is occupied
            if (fd <= handle_map->fd_top && HANDLE_ALLOCATED(handle_map->map[fd])) {
                ret = -EBADF;
                goto out;
            }
        }
    }

    if (fd >= get_rlimit_cur(RLIMIT_NOFILE)) {
        ret = -EMFILE;
        goto out;
    }

    // Enlarge handle_map->map (or allocate if necessary)
    if (fd >= handle_map->fd_size) {
        uint32_t new_size = handle_map->fd_size;
        if (new_size == 0)
            new_size = INIT_HANDLE_MAP_SIZE;

        while (new_size <= fd) {
            if (__builtin_mul_overflow(new_size, 2, &new_size)) {
                ret = -ENFILE;
                goto out;
            }
        }

        ret = __enlarge_handle_map(handle_map, new_size);
        if (ret < 0)
            goto out;
    }

    assert(handle_map->map);
    assert(fd < handle_map->fd_size);
    ret = __init_handle(&handle_map->map[fd], fd, hdl, fd_flags);
    if (ret < 0)
        goto out;

    if (handle_map->fd_top == FD_NULL || fd > handle_map->fd_top)
        handle_map->fd_top = fd;

    ret = fd;

out:
    unlock(&handle_map->lock);
    return ret;
}

int set_new_fd_handle(struct shim_handle* hdl, int fd_flags, struct shim_handle_map* handle_map) {
    return __set_new_fd_handle(0, hdl, fd_flags, handle_map, /*find_first=*/true);
}

int set_new_fd_handle_by_fd(uint32_t fd, struct shim_handle* hdl, int fd_flags,
                            struct shim_handle_map* handle_map) {
    return __set_new_fd_handle(fd, hdl, fd_flags, handle_map, /*find_first=*/false);
}

int set_new_fd_handle_above_fd(uint32_t fd, struct shim_handle* hdl, int fd_flags,
                               struct shim_handle_map* handle_map) {
    return __set_new_fd_handle(fd, hdl, fd_flags, handle_map, /*find_first=*/true);
}

static inline __attribute__((unused)) const char* __handle_name(struct shim_handle* hdl) {
    if (hdl->uri)
        return hdl->uri;
    if (hdl->dentry && hdl->dentry->name[0] != '\0')
        return hdl->dentry->name;
    if (hdl->fs)
        return hdl->fs->name;
    return "(unknown)";
}

void get_handle(struct shim_handle* hdl) {
#ifdef DEBUG_REF
    int ref_count = REF_INC(hdl->ref_count);

    log_debug("get handle %p(%s) (ref_count = %d)", hdl, __handle_name(hdl), ref_count);
#else
    REF_INC(hdl->ref_count);
#endif
}

static void destroy_handle(struct shim_handle* hdl) {
    destroy_lock(&hdl->lock);
    destroy_lock(&hdl->pos_lock);

    free_mem_obj_to_mgr(handle_mgr, hdl);
}

void put_handle(struct shim_handle* hdl) {
    int ref_count = REF_DEC(hdl->ref_count);

#ifdef DEBUG_REF
    log_debug("put handle %p(%s) (ref_count = %d)", hdl, __handle_name(hdl), ref_count);
#endif

    if (!ref_count) {
        assert(hdl->epoll_items_count == 0);
        assert(LISTP_EMPTY(&hdl->epoll_items));

        if (hdl->is_dir) {
            clear_directory_handle(hdl);
        }

        if (hdl->fs && hdl->fs->fs_ops && hdl->fs->fs_ops->close)
            hdl->fs->fs_ops->close(hdl);

        if (hdl->type == TYPE_SOCK && hdl->info.sock.peek_buffer) {
            free(hdl->info.sock.peek_buffer);
            hdl->info.sock.peek_buffer = NULL;
        }

        if (hdl->fs && hdl->fs->fs_ops && hdl->fs->fs_ops->hdrop)
            hdl->fs->fs_ops->hdrop(hdl);

        free(hdl->uri);

        if (hdl->pal_handle) {
#ifdef DEBUG_REF
            log_debug("handle %p closes PAL handle %p", hdl, hdl->pal_handle);
#endif
            DkObjectClose(hdl->pal_handle); // TODO: handle errors
            hdl->pal_handle = NULL;
        }

        if (hdl->dentry)
            put_dentry(hdl->dentry);

        if (hdl->inode)
            put_inode(hdl->inode);

        destroy_handle(hdl);
    }
}

int get_file_size(struct shim_handle* hdl, uint64_t* size) {
    if (!hdl->fs || !hdl->fs->fs_ops)
        return -EINVAL;

    if (hdl->fs->fs_ops->hstat) {
        struct stat stat;
        int ret = hdl->fs->fs_ops->hstat(hdl, &stat);
        if (ret < 0) {
            return ret;
        }
        if (stat.st_size < 0) {
            return -EINVAL;
        }
        *size = (uint64_t)stat.st_size;
        return 0;
    }

    *size = 0;
    return 0;
}

static struct shim_handle_map* get_new_handle_map(uint32_t size) {
    struct shim_handle_map* handle_map = calloc(1, sizeof(struct shim_handle_map));

    if (!handle_map)
        return NULL;

    handle_map->map = calloc(size, sizeof(*handle_map->map));

    if (!handle_map->map) {
        free(handle_map);
        return NULL;
    }

    handle_map->fd_top  = FD_NULL;
    handle_map->fd_size = size;
    if (!create_lock(&handle_map->lock)) {
        free(handle_map->map);
        free(handle_map);
        return NULL;
    }

    REF_SET(handle_map->ref_count, 1);

    return handle_map;
}

static int __enlarge_handle_map(struct shim_handle_map* map, uint32_t size) {
    assert(locked(&map->lock));

    if (size <= map->fd_size)
        return 0;

    struct shim_fd_handle** new_map = calloc(size, sizeof(new_map[0]));
    if (!new_map)
        return -ENOMEM;

    memcpy(new_map, map->map, map->fd_size * sizeof(new_map[0]));
    free(map->map);
    map->map     = new_map;
    map->fd_size = size;
    return 0;
}

int dup_handle_map(struct shim_handle_map** new, struct shim_handle_map* old_map) {
    lock(&old_map->lock);

    /* allocate a new handle mapping with the same size as
       the old one */
    struct shim_handle_map* new_map = get_new_handle_map(old_map->fd_size);

    if (!new_map)
        return -ENOMEM;

    new_map->fd_top = old_map->fd_top;

    if (old_map->fd_top == FD_NULL)
        goto done;

    for (uint32_t i = 0; i <= old_map->fd_top; i++) {
        struct shim_fd_handle* fd_old = old_map->map[i];
        struct shim_fd_handle* fd_new;

        /* now we go through the handle map and reassign each
           of them being allocated */
        if (HANDLE_ALLOCATED(fd_old)) {
            /* first, get the handle to prevent it from being deleted */
            struct shim_handle* hdl = fd_old->handle;
            get_handle(hdl);

            fd_new = malloc(sizeof(struct shim_fd_handle));
            if (!fd_new) {
                for (uint32_t j = 0; j < i; j++) {
                    put_handle(new_map->map[j]->handle);
                    free(new_map->map[j]);
                }
                unlock(&old_map->lock);
                *new = NULL;
                free(new_map);
                return -ENOMEM;
            }

            /* DP: I assume we really need a deep copy of the handle map? */
            new_map->map[i] = fd_new;
            fd_new->vfd     = fd_old->vfd;
            fd_new->handle  = hdl;
            fd_new->flags   = fd_old->flags;
        }
    }

done:
    unlock(&old_map->lock);
    *new = new_map;
    return 0;
}

void get_handle_map(struct shim_handle_map* map) {
    REF_INC(map->ref_count);
}

void put_handle_map(struct shim_handle_map* map) {
    int ref_count = REF_DEC(map->ref_count);

    if (!ref_count) {
        if (map->fd_top == FD_NULL)
            goto done;

        for (uint32_t i = 0; i <= map->fd_top; i++) {
            if (!map->map[i])
                continue;

            if (map->map[i]->vfd != FD_NULL) {
                struct shim_handle* handle = map->map[i]->handle;

                if (handle)
                    put_handle(handle);
            }

            free(map->map[i]);
        }

    done:
        destroy_lock(&map->lock);
        free(map->map);
        free(map);
    }
}

int walk_handle_map(int (*callback)(struct shim_fd_handle*, struct shim_handle_map*),
                    struct shim_handle_map* map) {
    int ret = 0;
    lock(&map->lock);

    if (map->fd_top == FD_NULL)
        goto done;

    for (uint32_t i = 0; i <= map->fd_top; i++) {
        if (!HANDLE_ALLOCATED(map->map[i]))
            continue;

        if ((ret = (*callback)(map->map[i], map)) < 0)
            break;
    }

done:
    unlock(&map->lock);
    return ret;
}

BEGIN_CP_FUNC(handle) {
    __UNUSED(size);
    assert(size == sizeof(struct shim_handle));

    struct shim_handle* hdl     = (struct shim_handle*)obj;
    struct shim_handle* new_hdl = NULL;

    size_t off = GET_FROM_CP_MAP(obj);

    if (!off) {
        off = ADD_CP_OFFSET(sizeof(struct shim_handle));
        ADD_TO_CP_MAP(obj, off);
        new_hdl = (struct shim_handle*)(base + off);

        lock(&hdl->lock);
        *new_hdl = *hdl;

        if (hdl->fs && hdl->fs->fs_ops && hdl->fs->fs_ops->checkout)
            hdl->fs->fs_ops->checkout(new_hdl);

        new_hdl->dentry = NULL;
        REF_SET(new_hdl->ref_count, 0);
        clear_lock(&new_hdl->lock);
        clear_lock(&new_hdl->pos_lock);

        DO_CP(fs, hdl->fs, &new_hdl->fs);

        if (hdl->uri)
            DO_CP_MEMBER(str, hdl, new_hdl, uri);

        if (hdl->is_dir) {
            /*
             * We don't checkpoint children dentries of a directory dentry, so the child process
             * will need to list the directory again. However, we keep `dir_info.pos` unchanged
             * so that `getdents/getdents64` will resume from the same place.
             */
            new_hdl->dir_info.dents = NULL;
            new_hdl->dir_info.count = 0;
        }

        if (hdl->dentry) {
            DO_CP_MEMBER(dentry, hdl, new_hdl, dentry);
        }

        if (new_hdl->pal_handle) {
            struct shim_palhdl_entry* entry;
            DO_CP(palhdl_ptr, &hdl->pal_handle, &entry);
            entry->phandle = &new_hdl->pal_handle;
        }

        /* This list is created empty and all necessary references are added when checkpointing
         * items lists from specific epoll handles. See `epoll_items_list` checkpointing in
         * `shim_epoll.c` for more details. */
        INIT_LISTP(&new_hdl->epoll_items);
        new_hdl->epoll_items_count = 0;

        switch (hdl->type) {
            case TYPE_EPOLL:;
                struct shim_epoll_handle* epoll = &new_hdl->info.epoll;
                clear_lock(&epoll->lock);
                INIT_LISTP(&epoll->waiters);
                INIT_LISTP(&epoll->items);
                epoll->items_count = 0;
                DO_CP(epoll_items_list, hdl, new_hdl);
                break;
            case TYPE_SOCK:
                /* no support for multiple processes sharing options/peek buffer of the socket */
                new_hdl->info.sock.pending_options = NULL;
                new_hdl->info.sock.peek_buffer     = NULL;
                break;
            default:
                break;
        }

        unlock(&hdl->lock);
        if (hdl->inode) {
            /* NOTE: Checkpointing `inode` will take `inode->lock`, so we need to do it after
             * `hdl->lock` is released. */
            DO_CP_MEMBER(inode, hdl, new_hdl, inode);
        }

        ADD_CP_FUNC_ENTRY(off);
    } else {
        new_hdl = (struct shim_handle*)(base + off);
    }

    if (objp)
        *objp = (void*)new_hdl;
}
END_CP_FUNC(handle)

BEGIN_RS_FUNC(handle) {
    struct shim_handle* hdl = (void*)(base + GET_CP_FUNC_ENTRY());
    __UNUSED(offset);

    CP_REBASE(hdl->fs);
    CP_REBASE(hdl->dentry);
    CP_REBASE(hdl->inode);
    CP_REBASE(hdl->epoll_items);

    if (!create_lock(&hdl->lock)) {
        return -ENOMEM;
    }

    if (!create_lock(&hdl->pos_lock)) {
        return -ENOMEM;
    }

    if (hdl->dentry) {
        get_dentry(hdl->dentry);
    }

    if (hdl->inode) {
        get_inode(hdl->inode);
    }

    switch (hdl->type) {
        case TYPE_EPOLL:;
            struct shim_epoll_handle* epoll = &hdl->info.epoll;
            if (!create_lock(&epoll->lock)) {
                return -ENOMEM;
            }
            CP_REBASE(epoll->waiters);
            /* `epoll->items` is rebased in epoll_items_list RS_FUNC. */
            break;
        default:
            break;
    }

    if (hdl->fs && hdl->fs->fs_ops && hdl->fs->fs_ops->checkin) {
        int ret = hdl->fs->fs_ops->checkin(hdl);
        if (ret < 0)
            return ret;
    }
}
END_RS_FUNC(handle)

BEGIN_CP_FUNC(fd_handle) {
    __UNUSED(size);
    assert(size == sizeof(struct shim_fd_handle));

    struct shim_fd_handle* fdhdl     = (struct shim_fd_handle*)obj;
    struct shim_fd_handle* new_fdhdl = NULL;

    size_t off = ADD_CP_OFFSET(sizeof(struct shim_fd_handle));
    new_fdhdl = (struct shim_fd_handle*)(base + off);
    *new_fdhdl = *fdhdl;
    DO_CP(handle, fdhdl->handle, &new_fdhdl->handle);
    ADD_CP_FUNC_ENTRY(off);

    if (objp)
        *objp = (void*)new_fdhdl;
}
END_CP_FUNC_NO_RS(fd_handle)

BEGIN_CP_FUNC(handle_map) {
    __UNUSED(size);
    assert(size >= sizeof(struct shim_handle_map));

    struct shim_handle_map* handle_map     = (struct shim_handle_map*)obj;
    struct shim_handle_map* new_handle_map = NULL;
    struct shim_fd_handle** ptr_array;

    lock(&handle_map->lock);

    int fd_size = handle_map->fd_top != FD_NULL ? handle_map->fd_top + 1 : 0;

    size = sizeof(struct shim_handle_map) + (sizeof(struct shim_fd_handle*) * fd_size);

    size_t off = GET_FROM_CP_MAP(obj);

    if (!off) {
        off            = ADD_CP_OFFSET(size);
        new_handle_map = (struct shim_handle_map*)(base + off);

        *new_handle_map = *handle_map;

        ptr_array = (void*)new_handle_map + sizeof(struct shim_handle_map);

        new_handle_map->fd_size = fd_size;
        new_handle_map->map     = fd_size ? ptr_array : NULL;

        REF_SET(new_handle_map->ref_count, 0);
        clear_lock(&new_handle_map->lock);

        for (int i = 0; i < fd_size; i++) {
            if (HANDLE_ALLOCATED(handle_map->map[i]))
                DO_CP(fd_handle, handle_map->map[i], &ptr_array[i]);
            else
                ptr_array[i] = NULL;
        }

        ADD_CP_FUNC_ENTRY(off);
    } else {
        new_handle_map = (struct shim_handle_map*)(base + off);
    }

    unlock(&handle_map->lock);

    if (objp)
        *objp = (void*)new_handle_map;
}
END_CP_FUNC(handle_map)

BEGIN_RS_FUNC(handle_map) {
    struct shim_handle_map* handle_map = (void*)(base + GET_CP_FUNC_ENTRY());
    __UNUSED(offset);

    CP_REBASE(handle_map->map);
    assert(handle_map->map);

    DEBUG_RS("size=%d,top=%d", handle_map->fd_size, handle_map->fd_top);

    if (!create_lock(&handle_map->lock)) {
        return -ENOMEM;
    }
    lock(&handle_map->lock);

    if (handle_map->fd_top != FD_NULL)
        for (uint32_t i = 0; i <= handle_map->fd_top; i++) {
            CP_REBASE(handle_map->map[i]);
            if (HANDLE_ALLOCATED(handle_map->map[i])) {
                CP_REBASE(handle_map->map[i]->handle);
                struct shim_handle* hdl = handle_map->map[i]->handle;
                assert(hdl);
                get_handle(hdl);
                DEBUG_RS("[%d]%s", i, hdl->uri ?: hdl->fs_type);
            }
        }

    unlock(&handle_map->lock);
}
END_RS_FUNC(handle_map)
