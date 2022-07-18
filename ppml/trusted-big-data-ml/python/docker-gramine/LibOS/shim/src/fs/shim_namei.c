/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2017, University of North Carolina at Chapel Hill and Fortanix, Inc. */

/*
 * This file contains code for parsing a FS path and looking up in the directory cache.
 */

#include <asm/fcntl.h>
#include <linux/fcntl.h>
#include <stdbool.h>

#include "perm.h"
#include "shim_fs.h"
#include "shim_handle.h"
#include "shim_lock.h"
#include "shim_process.h"
#include "stat.h"

int check_permissions(struct shim_dentry* dent, mode_t mask) {
    assert(locked(&g_dcache_lock));

    if (!dent->inode)
        return -ENOENT;

    /* If we only check if the file exists, at this point we know that */
    if (mask == F_OK)
        return 0;

    /* Check the "user" part of mode against mask */
    if (((dent->inode->perm >> 6) & mask) == mask)
        return 0;

    return -EACCES;
}

/* This function works like `lookup_dcache`, but if the dentry is not in cache, it creates a new,
 * negative one. */
static struct shim_dentry* lookup_dcache_or_create(struct shim_dentry* parent, const char* name,
                                                   size_t name_len) {
    assert(locked(&g_dcache_lock));
    assert(parent);

    struct shim_dentry* dent = lookup_dcache(parent, name, name_len);
    if (!dent)
        dent = get_new_dentry(parent->mount, parent, name, name_len);
    return dent;
}

/* Performs lookup operation in the underlying filesystem. Treats -ENOENT from lookup operation as
 * success (but leaves the dentry negative). */
static int lookup_dentry(struct shim_dentry* dent) {
    assert(locked(&g_dcache_lock));

    if (dent->inode)
        return 0;

    assert(dent->mount);
    assert(dent->mount->fs->d_ops);
    assert(dent->mount->fs->d_ops->lookup);
    int ret = dent->mount->fs->d_ops->lookup(dent);
    if (ret < 0) {
        assert(!dent->inode);
        /* Treat -ENOENT as successful lookup (but leave the dentry negative) */
        return ret == -ENOENT ? 0 : ret;
    }
    assert(dent->inode);
    return 0;
}

static int do_path_lookupat(struct shim_dentry* start, const char* path, int flags,
                            struct shim_dentry** found, unsigned int link_depth);

/* Helper function that follows a symbolic link, performing a nested call to `do_path_lookupat`  */
static int path_lookupat_follow(struct shim_dentry* link, int flags, struct shim_dentry** found,
                                unsigned int link_depth) {
    int ret;
    char* target = NULL;

    assert(locked(&g_dcache_lock));

    assert(link->inode);
    struct shim_fs* fs = link->inode->fs;
    assert(fs->d_ops);
    assert(fs->d_ops->follow_link);
    ret = fs->d_ops->follow_link(link, &target);
    if (ret < 0)
        goto out;

    struct shim_dentry* up = dentry_up(link);
    if (!up)
        up = g_dentry_root;
    ret = do_path_lookupat(up, target, flags, found, link_depth);

out:
    free(target);
    return ret;
}

/*!
 * \brief Traverse mountpoints and look up a dentry.
 *
 * \param[in,out] dent  The dentry.
 *
 * \returns 0 on success, negative error code otherwise.
 *
 * If `*dent` is a mountpoint, this function converts it to the underlying filesystem root
 * (iterating if necessary), then performs a lookup on the last dentry. As in `lookup_dentry`, if
 * the file is not found, the function will still return 0 (but `*dent` will be negative).
 *
 * The caller should hold a reference to `*dent`. On success, the reference will be converted: the
 * function will decrease the reference count for the original dentry, and increase it for the new
 * one.
 *
 * The caller should hold `g_dcache_lock`.
 */
static int traverse_mount_and_lookup(struct shim_dentry** dent) {
    assert(locked(&g_dcache_lock));

    struct shim_dentry* cur_dent = *dent;
    while (cur_dent->attached_mount) {
        cur_dent = cur_dent->attached_mount->root;
    }

    int ret = lookup_dentry(cur_dent);
    if (ret < 0)
        return ret;

    if (cur_dent != *dent) {
        get_dentry(cur_dent);
        put_dentry(*dent);
        *dent = cur_dent;
    }
    return 0;
}

/* State of the lookup algorithm */
struct lookup {
    /* Current dentry */
    struct shim_dentry* dent;

    /* Beginning of the next name to look up (can be empty if we're finished) */
    const char* name;

    /* Whether the whole path ends with slash */
    bool has_slash;

    /* Lookup options */
    int flags;

    /* Depth of followed symbolic links, to avoid too deep recursion */
    unsigned int link_depth;
};

/* Process a new dentry in the lookup: follow mounts and symbolic links, then check if the resulting
 * dentry is positive. */
static int lookup_enter_dentry(struct lookup* lookup) {
    int ret;
    bool is_final = (*lookup->name == '\0');
    bool has_slash = lookup->has_slash;

    if ((ret = traverse_mount_and_lookup(&lookup->dent)) < 0)
        return ret;

    if (lookup->dent->inode && lookup->dent->inode->type == S_IFLNK) {
        /* Traverse the symbolic link. This applies to all intermediate segments, final segments
         * ending with slash, and to all final segments if LOOKUP_FOLLOW is set. */
        if (!is_final || has_slash || (lookup->flags & LOOKUP_FOLLOW)) {
            if (lookup->link_depth >= MAX_LINK_DEPTH)
                return -ELOOP;

            /* If this is not the final segment (or it is the final segment, but ends with slash),
             * the nested lookup has different options: it always follows symlinks, needs to always
             * find the target, and the target has to be a directory. */
            int sub_flags = lookup->flags;
            if (!is_final || has_slash) {
                sub_flags |= LOOKUP_FOLLOW | LOOKUP_DIRECTORY;
                sub_flags &= ~LOOKUP_CREATE;
            }

            struct shim_dentry* target_dent;
            ret = path_lookupat_follow(lookup->dent, sub_flags, &target_dent,
                                       lookup->link_depth + 1);
            if (ret < 0)
                return ret;

            put_dentry(lookup->dent);
            lookup->dent = target_dent;
        }
    }

    if (!lookup->dent->inode) {
        /*
         * The file does not exist:
         *
         * - if LOOKUP_MAKE_SYNTHETIC is set, create a synthetic dentry,
         * - if LOOKUP_CREATE is set, allow it (but only as the last path segment),
         * - otherwise, fail with -ENOENT.
         */
        if (lookup->flags & LOOKUP_MAKE_SYNTHETIC) {
            ret = synthetic_setup_dentry(lookup->dent, S_IFDIR, PERM_r_xr_xr_x);
            if (ret < 0)
                return ret;
        } else if (is_final && (lookup->flags & LOOKUP_CREATE)) {
            /* proceed with a negative dentry */
        } else {
            return -ENOENT;
        }
    } else if (lookup->dent->inode->type != S_IFDIR) {
        /*
         * The file exists, but is not a directory. We expect a directory (and need to fail with
         * -ENOTDIR) in the following cases:
         *
         * - if this is an intermediate path segment,
         * - if this is the final path segment, but the path ends with a slash,
         * - if LOOKUP_DIRECTORY is set.
         */
        if (!is_final || has_slash || (lookup->flags & LOOKUP_DIRECTORY)) {
            return -ENOTDIR;
        }
    }

    return 0;
}

/* Advance the lookup: process the next name in the path, and search for corresponding dentry. */
static int lookup_advance(struct lookup* lookup) {
    const char* name = lookup->name;
    assert(*name != '/' && *name != '\0');

    const char* name_end = name;
    while (*name_end != '\0' && *name_end != '/')
        name_end++;
    size_t name_len = name_end - name;

    if (name_len > NAME_MAX)
        return -ENAMETOOLONG;

    struct shim_dentry* next_dent;
    if (name_len == 1 && name[0] == '.') {
        next_dent = lookup->dent;
        get_dentry(next_dent);
    } else if (name_len == 2 && name[0] == '.' && name[1] == '.') {
        next_dent = dentry_up(lookup->dent);
        if (!next_dent)
            next_dent = lookup->dent;
        get_dentry(next_dent);
    } else {
        next_dent = lookup_dcache_or_create(lookup->dent, name, name_len);
        if (!next_dent)
            return -ENOMEM;
    }

    put_dentry(lookup->dent);
    lookup->dent = next_dent;

    lookup->name = name_end;
    while (*lookup->name == '/')
        lookup->name++;
    return 0;
}

/*
 * This implementation is mostly iterative, but uses recursion to follow symlinks (which is why the
 * link depth is limited to MAX_LINK_DEPTH).
 */
static int do_path_lookupat(struct shim_dentry* start, const char* path, int flags,
                            struct shim_dentry** found, unsigned int link_depth) {
    assert(locked(&g_dcache_lock));

    struct shim_dentry* dent = NULL;
    int ret = 0;

    /* Empty path is invalid in POSIX */
    if (*path == '\0')
        return -ENOENT;

    if (*path == '/') {
        /* Absolute path, use process root even if `start` was provided (can happen for *at() system
         * calls) */
        lock(&g_process.fs_lock);
        dent = g_process.root;
        if (!dent) {
            /* Can happen during LibOS initialization */
            dent = g_dentry_root;
        }
        get_dentry(dent);
        unlock(&g_process.fs_lock);
    } else if (!start) {
        /* Relative part with no start dentry provided, use process current working directory */
        lock(&g_process.fs_lock);
        dent = g_process.cwd;
        if (!dent) {
            /* Can happen during LibOS initialization */
            start = g_dentry_root;
        }
        get_dentry(dent);
        unlock(&g_process.fs_lock);
    } else {
        /* Relative path with start dentry provided, use it */
        dent = start;
        get_dentry(dent);
    }

    size_t path_len = strlen(path);
    bool has_slash = (path_len > 0 && path[path_len - 1] == '/');

    const char* name = path;
    while (*name == '/')
        name++;

    struct lookup lookup = {
        .flags = flags,
        .name = name,
        .has_slash = has_slash,
        .dent = dent,
        .link_depth = link_depth,
    };

    /* Main part of the algorithm. Repeatedly call `lookup_enter_dentry`, then `lookup_advance`,
     * until we have finished the path. */

    ret = lookup_enter_dentry(&lookup);
    if (ret < 0)
        goto err;
    while (*lookup.name != '\0') {
        ret = lookup_advance(&lookup);
        if (ret < 0)
            goto err;
        ret = lookup_enter_dentry(&lookup);
        if (ret < 0)
            goto err;
    }

    *found = lookup.dent;
    return 0;

err:
    put_dentry(lookup.dent);
    *found = NULL;
    return ret;
}

int path_lookupat(struct shim_dentry* start, const char* path, int flags,
                   struct shim_dentry** found) {
    return do_path_lookupat(start, path, flags, found, /*link_depth=*/0);
}

static inline int open_flags_to_lookup_flags(int flags) {
    int retval = LOOKUP_FOLLOW;

    if (flags & O_NOFOLLOW)
        retval &= ~LOOKUP_FOLLOW;

    if (flags & O_CREAT)
        retval |= LOOKUP_CREATE;

    /*
     * See `man 2 open`:
     *
     * "When these two flags [O_CREAT and O_EXCL] are specified, symbolic links are not followed: if
     * pathname is a symbolic link, then open() fails regardless of where the symbolic link points."
     */
    if ((flags & (O_CREAT | O_EXCL)) == (O_CREAT | O_EXCL))
        retval &= ~LOOKUP_FOLLOW;

    if (flags & O_DIRECTORY)
        retval |= LOOKUP_DIRECTORY;

    return retval;
}

static void assoc_handle_with_dentry(struct shim_handle* hdl, struct shim_dentry* dent, int flags) {
    assert(locked(&g_dcache_lock));
    assert(dent->inode);

    hdl->dentry = dent;
    get_dentry(dent);

    hdl->inode = dent->inode;
    get_inode(dent->inode);

    hdl->fs = dent->inode->fs;
    hdl->flags = flags;
    hdl->acc_mode = ACC_MODE(flags & O_ACCMODE);
}

int dentry_open(struct shim_handle* hdl, struct shim_dentry* dent, int flags) {
    assert(locked(&g_dcache_lock));
    assert(dent->inode);
    assert(!hdl->dentry);

    int ret;
    struct shim_fs* fs = dent->inode->fs;

    if (!(fs->d_ops && fs->d_ops->open))
        return -EINVAL;

    ret = fs->d_ops->open(hdl, dent, flags);
    if (ret < 0)
        return ret;

    assoc_handle_with_dentry(hdl, dent, flags);

    /* NOTE: If we fail after this step, the handle will remain open and associated with dentry, but
     * the caller is going to delete it. */

    if (dent->inode->type == S_IFDIR) {
        /* Initialize directory handle */
        hdl->is_dir = true;

        hdl->dir_info.dents = NULL;
    }

    /* truncate regular writable file if O_TRUNC is given */
    if ((flags & O_TRUNC) && ((flags & O_RDWR) | (flags & O_WRONLY))
            && (dent->inode->type != S_IFDIR)
            && (dent->inode->type != S_IFLNK)) {

        if (!(fs->fs_ops && fs->fs_ops->truncate))
            return -EINVAL;

        ret = fs->fs_ops->truncate(hdl, 0);
        if (ret < 0)
            return ret;
    }

    return 0;
}

int open_namei(struct shim_handle* hdl, struct shim_dentry* start, const char* path, int flags,
               int mode, struct shim_dentry** found) {
    int lookup_flags = open_flags_to_lookup_flags(flags);
    mode_t acc_mode = ACC_MODE(flags & O_ACCMODE);
    int ret = 0;
    struct shim_dentry* dent = NULL;

    /* O_CREAT on a normal file triggers creat(), which needs a handle */
    if ((flags & O_CREAT) && !(flags & O_DIRECTORY))
        assert(hdl);

    if (hdl)
        assert(!hdl->dentry);

    lock(&g_dcache_lock);

    ret = path_lookupat(start, path, lookup_flags, &dent);
    if (ret < 0)
        goto err;

    if (dent->inode && dent->inode->type == S_IFDIR) {
        if (flags & O_WRONLY || flags & O_RDWR ||
                ((flags & O_CREAT) && !(flags & O_DIRECTORY) && !(flags & O_EXCL))) {
            ret = -EISDIR;
            goto err;
        }
    }

    if (dent->inode && dent->inode->type == S_IFLNK) {
        /*
         * Can happen if user specified O_NOFOLLOW, or O_TRUNC | O_EXCL. Posix requires us to fail
         * with -ELOOP when trying to open a symlink.
         *
         * (Linux allows opening a symlink with O_PATH, but Gramine does not support it yet).
         */
        ret = -ELOOP;
        goto err;
    }

    bool need_open = true;
    if (!dent->inode) {
        if (!(flags & O_CREAT)) {
            ret = -ENOENT;
            goto err;
        }

        /* Check the parent permission first */
        struct shim_dentry* dir = dentry_up(dent);
        if (dir) {
            ret = check_permissions(dir, MAY_WRITE | MAY_EXEC);
            if (ret < 0)
                goto err;
        }

        struct shim_fs* fs = dent->mount->fs;
        /* Create directory or file, depending on O_DIRECTORY. Return -EINVAL if the operation is
         * not supported for this filesystem. */
        if (flags & O_DIRECTORY) {
            if (!fs->d_ops->mkdir) {
                ret = -EINVAL;
                goto err;
            }
            ret = fs->d_ops->mkdir(dent, mode & ~S_IFMT);
            if (ret < 0)
                goto err;
        } else {
            if (!fs->d_ops->creat) {
                ret = -EINVAL;
                goto err;
            }
            ret = fs->d_ops->creat(hdl, dent, flags, mode & ~S_IFMT);
            if (ret < 0)
                goto err;
            assoc_handle_with_dentry(hdl, dent, flags);
            need_open = false;
        }
    } else {
        /* The file exists. This is not permitted if both O_CREAT and O_EXCL are set. */
        if ((flags & O_CREAT) && (flags & O_EXCL)) {
            ret = -EEXIST;
            goto err;
        }

        /* Check permissions. Note that we do it only if the file already exists: a newly created
         * file is allowed to have a mode that's incompatible with `acc_mode`. */
        ret = check_permissions(dent, acc_mode);
        if (ret < 0)
            goto err;
    }

    if (hdl && need_open) {
        ret = dentry_open(hdl, dent, flags);
        if (ret < 0)
            goto err;
    }

    if (found) {
        *found = dent;
    } else {
        put_dentry(dent);
    }
    unlock(&g_dcache_lock);
    return 0;

err:
    if (dent)
        put_dentry(dent);

    if (found)
        *found = NULL;

    unlock(&g_dcache_lock);
    return ret;
}

/* A list for `populate_directory` to hold file names from `readdir`. */
DEFINE_LIST(temp_dirent);
DEFINE_LISTP(temp_dirent);
struct temp_dirent {
    LIST_TYPE(temp_dirent) list;

    size_t name_len;
    char name[];
};

static int add_name(const char* name, void* arg) {
    LISTP_TYPE(temp_dirent)* ents = arg;

    size_t name_len = strlen(name);
    struct temp_dirent* ent = malloc(sizeof(*ent) + name_len + 1);
    if (!ent)
        return -ENOMEM;

    memcpy(&ent->name, name, name_len + 1);
    ent->name_len = name_len;
    LISTP_ADD(ent, ents, list);
    return 0;
}

/*
 * Ensure that a directory has a complete list of dentries, by calling `readdir` and then looking up
 * every name.
 *
 * While `readdir` is callback-based, we don't look up the names inside of callback, but first
 * finish `readdir`. Otherwise, the two filesystem operations (`readdir` and `lookup`) might
 * deadlock.
 */
static int populate_directory(struct shim_dentry* dent) {
    assert(locked(&g_dcache_lock));

    if (!dent->inode)
        return -ENOENT;

    struct shim_fs* fs = dent->inode->fs;
    if (!fs->d_ops || !fs->d_ops->readdir)
        return -EINVAL;

    LISTP_TYPE(temp_dirent) ents = LISTP_INIT;
    int ret = fs->d_ops->readdir(dent, &add_name, &ents);
    if (ret < 0)
        log_error("readdir error: %d", ret);

    struct shim_dentry* child;
    LISTP_FOR_EACH_ENTRY(child, &dent->children, siblings) {
        struct shim_inode* inode = child->inode;
        /* Check `inode->fs` so that we don't remove files added by Gramine (named pipes, sockets,
         * synthetic mountpoints) */
        if (inode && inode->fs == inode->mount->fs) {
            struct temp_dirent* ent;
            bool removed = true;
            LISTP_FOR_EACH_ENTRY(ent, &ents, list) {
                if (strcmp(ent->name, child->name) == 0) {
                    removed = false;
                    break;
                }
            }
            if (removed) {
                log_debug("File no longer present, detaching inode: %s", child->name);
                child->inode = NULL;
                put_inode(inode);
            }
        }
    }

    struct temp_dirent* ent;
    struct temp_dirent* tmp;

    LISTP_FOR_EACH_ENTRY(ent, &ents, list) {
        struct shim_dentry* child = lookup_dcache_or_create(dent, ent->name, ent->name_len);
        if (!child) {
            ret = -ENOMEM;
            goto out;
        }

        ret = traverse_mount_and_lookup(&child);
        put_dentry(child);
        if (ret < 0 && ret != -EACCES) {
            /* Fail on underlying lookup errors, except -EACCES (for which we will just ignore the
             * file). The lookup might fail with -EACCES for host symlinks pointing to inaccessible
             * target, since the "chroot" filesystem transparently follows symlinks instead of
             * reporting them to Gramine. */
            goto out;
        }
    }

    ret = 0;
out:
    LISTP_FOR_EACH_ENTRY_SAFE(ent, tmp, &ents, list) {
        LISTP_DEL(ent, &ents, list);
        free(ent);
    }
    return ret;
}

int populate_directory_handle(struct shim_handle* hdl) {
    struct shim_dir_handle* dirhdl = &hdl->dir_info;

    assert(locked(&hdl->lock));
    assert(locked(&g_dcache_lock));
    assert(hdl->dentry);

    int ret;

    if (dirhdl->dents)
        return 0;

    if ((ret = populate_directory(hdl->dentry)) < 0)
        goto err;

    size_t capacity = hdl->dentry->nchildren + 2; // +2 for ".", ".."

    dirhdl->dents = malloc(sizeof(struct shim_dentry) * capacity);
    if (!dirhdl->dents) {
        ret = -ENOMEM;
        goto err;
    }
    dirhdl->count = 0;

    struct shim_dentry* dot = hdl->dentry;
    get_dentry(dot);
    dirhdl->dents[dirhdl->count++] = dot;

    struct shim_dentry* dotdot = hdl->dentry->parent ?: hdl->dentry;
    get_dentry(dotdot);
    dirhdl->dents[dirhdl->count++] = dotdot;

    struct shim_dentry* tmp;
    struct shim_dentry* dent;
    LISTP_FOR_EACH_ENTRY_SAFE(dent, tmp, &hdl->dentry->children, siblings) {
        /* Traverse mount */
        struct shim_dentry* cur_dent = dent;
        while (cur_dent->attached_mount) {
            cur_dent = cur_dent->attached_mount->root;
        }

        if (cur_dent->inode) {
            get_dentry(cur_dent);
            assert(dirhdl->count < capacity);
            dirhdl->dents[dirhdl->count++] = cur_dent;
        }
        dentry_gc(dent);
    }

    return 0;

err:
    clear_directory_handle(hdl);
    return ret;
}

void clear_directory_handle(struct shim_handle* hdl) {
    struct shim_dir_handle* dirhdl = &hdl->dir_info;
    if (!dirhdl->dents)
        return;

    for (size_t i = 0; i < dirhdl->count; i++)
        put_dentry(dirhdl->dents[i]);
    free(dirhdl->dents);
    dirhdl->dents = NULL;
    dirhdl->count = 0;
}

int get_dirfd_dentry(int dirfd, struct shim_dentry** dir) {
    if (dirfd == AT_FDCWD) {
        lock(&g_process.fs_lock);
        *dir = g_process.cwd;
        get_dentry(*dir);
        unlock(&g_process.fs_lock);
        return 0;
    }

    if (dirfd < 0) {
        return -EBADF;
    }

    struct shim_handle* hdl = get_fd_handle(dirfd, NULL, NULL);
    if (!hdl) {
        return -EBADF;
    }

    if (!hdl->is_dir) {
        put_handle(hdl);
        return -ENOTDIR;
    }

    get_dentry(hdl->dentry);
    *dir = hdl->dentry;
    put_handle(hdl);
    return 0;
}
