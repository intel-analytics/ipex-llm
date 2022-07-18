/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * This file contains implementation of the "pseudo" filesystem.
 *
 * We store a pointer to a `pseudo_node` object in `inode->data`. Note that since `inode->data`
 * never changes, we do NOT take `inode->lock` when reading it.
 */

#include "shim_fs_pseudo.h"
#include "shim_lock.h"
#include "stat.h"

LISTP_TYPE(pseudo_node) g_pseudo_roots = LISTP_INIT;

/* Array of nodes by ID. Used for restoring a node from checkpoint (we send only node ID). We assume
 * that all Gramine processes within a single instance create exactly the same set of nodes, in the
 * same order, during initialization. */
static struct pseudo_node* g_pseudo_nodes[PSEUDO_MAX_NODES];
static unsigned int g_pseudo_node_count;

/* Find a root node with given name. */
static struct pseudo_node* pseudo_find_root(const char* name) {
    struct pseudo_node* node;
    LISTP_FOR_EACH_ENTRY(node, &g_pseudo_roots, siblings) {
        if (node->name && strcmp(name, node->name) == 0) {
            return node;
        }
    }

    log_debug("Cannot find pseudofs node: %s", name);
    return NULL;
}

/* Find a `pseudo_node` for given dentry. */
static struct pseudo_node* pseudo_find(struct shim_dentry* dent) {
    assert(locked(&g_dcache_lock));

    if (!dent->parent) {
        /* This is the filesystem root */
        return pseudo_find_root(dent->mount->uri);
    }

    assert(dent->parent->inode);
    struct pseudo_node* parent_node = dent->parent->inode->data;

    /* Look for a child node with matching name */
    assert(parent_node->type == PSEUDO_DIR);
    struct pseudo_node* node;
    LISTP_FOR_EACH_ENTRY(node, &parent_node->dir.children, siblings) {
        /* The node might have `name`, `name_exists`, or both. */
        bool match;
        if (node->name && node->name_exists) {
            match = !strcmp(dent->name, node->name) && node->name_exists(dent->parent, dent->name);
        } else if (node->name) {
            match = !strcmp(dent->name, node->name);
        } else if (node->name_exists) {
            match = node->name_exists(dent->parent, dent->name);
        } else {
            match = false;
        }

        if (match)
            return node;
    }
    return NULL;
}

static int pseudo_icheckpoint(struct shim_inode* inode, void** out_data, size_t* out_size) {
    unsigned int* id = malloc(sizeof(*id));
    if (!id)
        return -ENOMEM;

    struct pseudo_node* node = inode->data;
    *id = node->id;
    *out_data = (void*)id;
    *out_size = sizeof(*id);
    return 0;
}

static int pseudo_irestore(struct shim_inode* inode, void* data) {
    unsigned int* id = data;
    if (*id >= g_pseudo_node_count) {
        log_error("Invalid pseudo_node id: %u", *id);
        return -EINVAL;
    }

    struct pseudo_node* node = g_pseudo_nodes[*id];
    assert(node);
    inode->data = node;
    return 0;
}

static int pseudo_mount(struct shim_mount_params* params, void** mount_data) {
    if (!params->uri)
        return -EINVAL;

    __UNUSED(mount_data);
    return 0;
}

static int pseudo_open(struct shim_handle* hdl, struct shim_dentry* dent, int flags) {
    assert(locked(&g_dcache_lock));
    assert(dent->inode);

    struct pseudo_node* node = dent->inode->data;

    int ret;

    switch (node->type) {
        case PSEUDO_DIR:
            hdl->type = TYPE_PSEUDO;
            /* This is a directory handle, so it will be initialized by `dentry_open`. */
            break;

        case PSEUDO_LINK:
            return -EINVAL;

        case PSEUDO_STR: {
            char* str;
            size_t len;
            if (node->str.load) {
                ret = node->str.load(dent, &str, &len);
                if (ret < 0)
                    return ret;

                if (len > 0)
                    assert(str);
            } else {
                len = 0;
                str = NULL;
            }

            hdl->type = TYPE_STR;
            mem_file_init(&hdl->info.str.mem, str, len);
            hdl->pos = 0;
            break;
        }

        case PSEUDO_DEV: {
            hdl->type = TYPE_DEV;
            if (node->dev.dev_ops.open) {
                ret = node->dev.dev_ops.open(hdl, dent, flags);
                if (ret < 0)
                    return ret;
            }
            break;
        }
    }

    return 0;
}

static int pseudo_lookup(struct shim_dentry* dent) {
    assert(locked(&g_dcache_lock));
    assert(!dent->inode);

    struct pseudo_node* node = pseudo_find(dent);
    if (!node)
        return -ENOENT;

    mode_t type;
    switch (node->type) {
        case PSEUDO_DIR:
            type = S_IFDIR;
            break;
        case PSEUDO_LINK:
            type = S_IFLNK;
            break;
        case PSEUDO_STR:
            type = S_IFREG;
            break;
        case PSEUDO_DEV:
            type = S_IFCHR;
            break;
        default:
            BUG();
    }

    struct shim_inode* inode = get_new_inode(dent->mount, type, node->perm);
    if (!inode)
        return -ENOMEM;

    inode->data = node;

    dent->inode = inode;
    return 0;
}

static int count_nlink(const char* name, void* arg) {
    __UNUSED(name);
    size_t* nlink = arg;
    (*nlink)++;
    return 0;
}

static dev_t makedev(unsigned int major, unsigned int minor) {
    dev_t dev;
    dev  = (((dev_t)(major & 0x00000fffu)) <<  8);
    dev |= (((dev_t)(major & 0xfffff000u)) << 32);
    dev |= (((dev_t)(minor & 0x000000ffu)) <<  0);
    dev |= (((dev_t)(minor & 0xffffff00u)) << 12);
    return dev;
}

static int pseudo_istat(struct shim_dentry* dent, struct shim_inode* inode, struct stat* buf) {
    memset(buf, 0, sizeof(*buf));
    buf->st_dev = 1;
    buf->st_mode = inode->type | inode->perm;
    struct pseudo_node* node = inode->data;
    switch (node->type) {
        case PSEUDO_DIR: {
            /*
             * Count the actual number of children for `nlink`. This is not very efficient, but
             * libraries like hwloc check `nlink` in some places.
             *
             * Note that we might not be holding `g_dcache_lock` here, so we cannot call
             * `pseudo_readdir`.
             */
            size_t nlink = 2; // Initialize to 2 for `.` and parent
            struct pseudo_node* child_node;
            LISTP_FOR_EACH_ENTRY(child_node, &node->dir.children, siblings) {
                if (child_node->name) {
                    /* If `name_exists` callback is provided, check it. */
                    if (!child_node->name_exists || node->name_exists(dent, child_node->name))
                        nlink++;
                }
                if (child_node->list_names) {
                    int ret = child_node->list_names(dent, &count_nlink, &nlink);
                    if (ret < 0)
                        return ret;
                }
            }
            buf->st_nlink = nlink;
            break;
        }
        case PSEUDO_DEV:
            buf->st_rdev = makedev(node->dev.major, node->dev.minor);
            buf->st_nlink = 1;
            break;
        default:
            buf->st_nlink = 1;
            break;
    }
    return 0;
}

static int pseudo_stat(struct shim_dentry* dent, struct stat* buf) {
    assert(locked(&g_dcache_lock));
    assert(dent->inode);

    return pseudo_istat(dent, dent->inode, buf);
}

static int pseudo_hstat(struct shim_handle* handle, struct stat* buf) {
    return pseudo_istat(handle->dentry, handle->inode, buf);
}

static int pseudo_follow_link(struct shim_dentry* dent, char** out_target) {
    assert(locked(&g_dcache_lock));
    assert(dent->inode);

    char* target;

    struct pseudo_node* node = dent->inode->data;
    if (node->type != PSEUDO_LINK)
        return -EINVAL;

    if (node->link.follow_link) {
        int ret = node->link.follow_link(dent, &target);
        if (ret < 0)
            return ret;
        *out_target = target;
        return 0;
    }

    assert(node->link.target);
    target = strdup(node->link.target);
    if (!target)
        return -ENOMEM;

    *out_target = target;
    return 0;
}

static int pseudo_readdir(struct shim_dentry* dent, readdir_callback_t callback, void* arg) {
    assert(locked(&g_dcache_lock));
    assert(dent->inode);

    int ret;

    struct pseudo_node* parent_node = dent->inode->data;
    if (parent_node->type != PSEUDO_DIR)
        return -ENOTDIR;

    struct pseudo_node* node;
    LISTP_FOR_EACH_ENTRY(node, &parent_node->dir.children, siblings) {
        if (node->name) {
            /* If `name_exists` callback is provided, check it. */
            if (!node->name_exists || node->name_exists(dent, node->name)) {
                ret = callback(node->name, arg);
                if (ret < 0)
                    return ret;
            }
        }
        if (node->list_names) {
            ret = node->list_names(dent, callback, arg);
            if (ret < 0)
                return ret;
        }
    }
    return 0;
}

static ssize_t pseudo_read(struct shim_handle* hdl, void* buf, size_t size, file_off_t* pos) {
    struct pseudo_node* node = hdl->inode->data;
    switch (node->type) {
        case PSEUDO_STR: {
            assert(hdl->type == TYPE_STR);
            lock(&hdl->lock);
            ssize_t ret = mem_file_read(&hdl->info.str.mem, *pos, buf, size);
            if (ret > 0)
                *pos += ret;
            unlock(&hdl->lock);
            return ret;
        }

        case PSEUDO_DEV:
            if (!node->dev.dev_ops.read)
                return -EACCES;
            return node->dev.dev_ops.read(hdl, buf, size);

        default:
            return -ENOSYS;
    }
}

static ssize_t pseudo_write(struct shim_handle* hdl, const void* buf, size_t size,
                            file_off_t* pos) {
    struct pseudo_node* node = hdl->inode->data;
    switch (node->type) {
        case PSEUDO_STR: {
            assert(hdl->type == TYPE_STR);

            ssize_t ret;

            lock(&hdl->lock);

            struct shim_mem_file* mem = &hdl->info.str.mem;

            if (node->str.save) {
                /* If there's a `save` method, we want to invoke it, and the write should replace
                 * existing content. */

                ret = node->str.save(hdl->dentry, buf, size);
                if (ret < 0)
                    goto out;

                ret = mem_file_truncate(mem, 0);
                if (ret < 0)
                    goto out;
                *pos = 0;
            }

            ret = mem_file_write(mem, *pos, buf, size);
            if (ret < 0)
                goto out;
            *pos += ret;

        out:
            unlock(&hdl->lock);
            return ret;
        }

        case PSEUDO_DEV:
            if (!node->dev.dev_ops.write)
                return -EACCES;
            return node->dev.dev_ops.write(hdl, buf, size);

        default:
            return -ENOSYS;
    }
}

static file_off_t pseudo_seek(struct shim_handle* hdl, file_off_t offset, int whence) {
    file_off_t ret;

    struct pseudo_node* node = hdl->inode->data;
    switch (node->type) {
        case PSEUDO_STR: {
            lock(&hdl->lock);
            file_off_t pos = hdl->pos;
            ret = generic_seek(pos, hdl->info.str.mem.size, offset, whence, &pos);
            if (ret == 0) {
                hdl->pos = pos;
                ret = pos;
            }
            unlock(&hdl->lock);
            return ret;
        }

        case PSEUDO_DEV:
            if (!node->dev.dev_ops.seek)
                return -EACCES;
            return node->dev.dev_ops.seek(hdl, offset, whence);

        default:
            return -ENOSYS;
    }
}

static int pseudo_truncate(struct shim_handle* hdl, file_off_t size) {
    struct pseudo_node* node = hdl->inode->data;
    switch (node->type) {
        case PSEUDO_STR:
            assert(hdl->type == TYPE_STR);
            lock(&hdl->lock);
            int ret = mem_file_truncate(&hdl->info.str.mem, size);
            unlock(&hdl->lock);
            return ret;

        case PSEUDO_DEV:
            if (!node->dev.dev_ops.truncate)
                return -EACCES;
            return node->dev.dev_ops.truncate(hdl, size);

        default:
            return -ENOSYS;
    }
}

static int pseudo_flush(struct shim_handle* hdl) {
    struct pseudo_node* node = hdl->inode->data;
    switch (node->type) {
        case PSEUDO_DEV:
            if (!node->dev.dev_ops.flush)
                return -EINVAL;
            return node->dev.dev_ops.flush(hdl);

        default:
            return -ENOSYS;
    }
}

static int pseudo_close(struct shim_handle* hdl) {
    struct pseudo_node* node = hdl->inode->data;
    switch (node->type) {
        case PSEUDO_STR: {
            lock(&hdl->lock);
            mem_file_destroy(&hdl->info.str.mem);
            unlock(&hdl->lock);
            return 0;
        }

        case PSEUDO_DEV:
            if (!node->dev.dev_ops.close)
                return 0;
            return node->dev.dev_ops.close(hdl);

        default:
            return 0;
    }
}

static int pseudo_poll(struct shim_handle* hdl, int poll_type) {
    struct pseudo_node* node = hdl->inode->data;
    switch (node->type) {
        case PSEUDO_STR: {
            assert(hdl->type == TYPE_STR);
            lock(&hdl->pos_lock);
            lock(&hdl->lock);
            int ret = mem_file_poll(&hdl->info.str.mem, hdl->pos, poll_type);
            unlock(&hdl->lock);
            unlock(&hdl->pos_lock);
            return ret;
        }

        case PSEUDO_DEV: {
            int ret = 0;
            if ((poll_type & FS_POLL_RD) && node->dev.dev_ops.read)
                ret |= FS_POLL_RD;
            if ((poll_type & FS_POLL_WR) && node->dev.dev_ops.write)
                ret |= FS_POLL_WR;
            return ret;
        }

        default:
            return -ENOSYS;
    }
}

int pseudo_parse_ulong(const char* str, unsigned long max_value, unsigned long* out_value) {
    unsigned long value;
    const char* end;

    if (str_to_ulong(str, 10, &value, &end) < 0 || *end != '\0' || value > max_value)
        return -1;

    /* no leading zeroes */
    if (str[0] == '0' && str[1] != '\0')
        return -1;

    *out_value = value;
    return 0;
}


static struct pseudo_node* pseudo_add_ent(struct pseudo_node* parent_node, const char* name,
                                          enum pseudo_type type) {
    if (g_pseudo_node_count >= PSEUDO_MAX_NODES) {
        log_error("Pseudo node limit reached, increase PSEUDO_MAX_NODES");
        abort();
    }
    unsigned int id = g_pseudo_node_count++;

    struct pseudo_node* node = calloc(1, sizeof(*node));
    if (!node) {
        log_error("Out of memory when allocating pseudofs node");
        abort();
    }
    node->name = name;
    node->type = type;
    node->id = id;

    if (parent_node) {
        assert(parent_node->type == PSEUDO_DIR);
        node->parent = parent_node;
        LISTP_ADD(node, &parent_node->dir.children, siblings);
    } else {
        LISTP_ADD(node, &g_pseudo_roots, siblings);
    }
    g_pseudo_nodes[id] = node;
    return node;
}

struct pseudo_node* pseudo_add_root_dir(const char* name) {
    return pseudo_add_dir(/*parent_node=*/NULL, name);
}

struct pseudo_node* pseudo_add_dir(struct pseudo_node* parent_node, const char* name) {
    struct pseudo_node* node = pseudo_add_ent(parent_node, name, PSEUDO_DIR);
    node->perm = PSEUDO_PERM_DIR;
    return node;
}

struct pseudo_node* pseudo_add_link(struct pseudo_node* parent_node, const char* name,
                                    int (*follow_link)(struct shim_dentry*, char**)) {
    struct pseudo_node* node = pseudo_add_ent(parent_node, name, PSEUDO_LINK);
    node->link.follow_link = follow_link;
    node->perm = PSEUDO_PERM_LINK;
    return node;
}

struct pseudo_node* pseudo_add_str(struct pseudo_node* parent_node, const char* name,
                                   int (*load)(struct shim_dentry*, char**, size_t*)) {
    struct pseudo_node* node = pseudo_add_ent(parent_node, name, PSEUDO_STR);
    node->str.load = load;
    node->perm = PSEUDO_PERM_FILE_R;
    return node;
}

struct pseudo_node* pseudo_add_dev(struct pseudo_node* parent_node, const char* name) {
    struct pseudo_node* node = pseudo_add_ent(parent_node, name, PSEUDO_DEV);
    node->perm = PSEUDO_PERM_FILE_R;
    return node;
}

struct shim_fs_ops pseudo_fs_ops = {
    .mount    = &pseudo_mount,
    .hstat    = &pseudo_hstat,
    .read     = &pseudo_read,
    .write    = &pseudo_write,
    .seek     = &pseudo_seek,
    .truncate = &pseudo_truncate,
    .close    = &pseudo_close,
    .flush    = &pseudo_flush,
    .poll     = &pseudo_poll,
};

struct shim_d_ops pseudo_d_ops = {
    .open        = &pseudo_open,
    .lookup      = &pseudo_lookup,
    .readdir     = &pseudo_readdir,
    .stat        = &pseudo_stat,
    .follow_link = &pseudo_follow_link,
    .icheckpoint = &pseudo_icheckpoint,
    .irestore    = &pseudo_irestore,
};

struct shim_fs pseudo_builtin_fs = {
    .name   = "pseudo",
    .fs_ops = &pseudo_fs_ops,
    .d_ops  = &pseudo_d_ops,
};
