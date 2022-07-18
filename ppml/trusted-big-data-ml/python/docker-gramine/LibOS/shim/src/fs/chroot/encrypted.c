/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2022 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * This implements the `chroot_encrypted` filesystem, intended to replace PAL protected files.
 *
 * This filesystem keeps the following data:
 *
 * - The mount (`shim_mount`) holds a `shim_encrypted_files_key` object. This is the encryption key
 *   for files. Multiple mounts can use the same key. The list of keys is managed in
 *   `shim_fs_encrypted.c`.
 *
 * - Inodes (`shim_inode`, for regular files) hold a `shim_encrypted_file` object. This object lives
 *   as long as the inode, but is kept *open* only as long as there are `shim_handle` objects
 *   corresponding to it. We use `encrypted_file_{get,put}` operations to maintain that invariant.
 *
 *   An open `shim_encrypted_file` object keeps an open PAL handle and associated data
 *   (`pf_context_t`), so that operations (read, write, truncate...) can be performed on the file.
 *
 * - Handles (`shim_handle`) hold no extra data. File operations on a handle use the
 *   `shim_encrypted_file` object for its inode. As a result, multiple handles for a given file
 *   still correspond to one PAL handle.
 *
 * TODO:
 *
 * - truncate (the current `truncate` operation works only for extending the file, support for
 *   truncation needs to be added to the `protected_files` module)
 * - flush all files on process exit
 */

#define _POSIX_C_SOURCE 200809L /* for SSIZE_MAX */
#include <limits.h>

#include "perm.h"
#include "shim_fs.h"
#include "shim_fs_encrypted.h"
#include "shim_vma.h"
#include "stat.h"
#include "toml_utils.h"

/*
 * Always add read and write permissions to files created on host. PAL requires opening the file
 * even for operations such as `unlink` or `chmod`, and the underlying `shim_fs_encrypted` module
 * opens the file for reading and writing.
 */
#define HOST_PERM(perm) ((perm) | PERM_rw_______)

static int chroot_encrypted_mount(struct shim_mount_params* params, void** mount_data) {
    if (!params->uri || !strstartswith(params->uri, URI_PREFIX_FILE))
        return -EINVAL;

    const char* key_name = params->key_name ?: "default";

    struct shim_encrypted_files_key* key;
    int ret = get_or_create_encrypted_files_key(key_name, &key);
    if (ret < 0)
        return ret;

    *mount_data = key;
    return 0;
}

static ssize_t chroot_encrypted_checkpoint(void** checkpoint, void* mount_data) {
    struct shim_encrypted_files_key* key = mount_data;

    *checkpoint = strdup(key->name);
    if (!*checkpoint)
        return -ENOMEM;
    return strlen(key->name) + 1;
}

static int chroot_encrypted_migrate(void* checkpoint, void** mount_data) {
    const char* name = checkpoint;

    struct shim_encrypted_files_key* key;
    int ret = get_or_create_encrypted_files_key(name, &key);
    if (ret < 0)
        return ret;
    *mount_data = key;
    return 0;
}

static void chroot_encrypted_idrop(struct shim_inode* inode) {
    assert(locked(&inode->lock));

    if (inode->data) {
        struct shim_encrypted_file* enc = inode->data;
        encrypted_file_destroy(enc);
    }
}

static int chroot_encrypted_lookup(struct shim_dentry* dent) {
    assert(locked(&g_dcache_lock));

    char* uri = NULL;
    struct shim_inode* inode = NULL;

    /*
     * We don't know the file type yet, so we can't construct a PAL URI with the right prefix.
     * However, "file:" prefix is good enough here: `DkStreamAttributesQuery` will access the file
     * and report the right file type.
     *
     * See also the comment in `fs.c:chroot_lookup` (but note that this case is simpler, because we
     * don't allow "dev:" mounts).
     */
    int ret = chroot_dentry_uri(dent, S_IFREG, &uri);
    if (ret < 0)
        goto out;

    PAL_STREAM_ATTR pal_attr;
    ret = DkStreamAttributesQuery(uri, &pal_attr);
    if (ret < 0) {
        ret = pal_to_unix_errno(ret);
        goto out;
    }

    mode_t type;
    switch (pal_attr.handle_type) {
        case PAL_TYPE_FILE:
            type = S_IFREG;
            break;
        case PAL_TYPE_DIR:
            type = S_IFDIR;
            break;
        default:
            log_warning("trying to access '%s' which is not an encrypted file or directory",
                        uri);
            ret = -EACCES;
            goto out;
    }

    mode_t perm = pal_attr.share_flags;

    inode = get_new_inode(dent->mount, type, perm);
    if (!inode) {
        ret = -ENOMEM;
        goto out;
    }

    if (type == S_IFREG) {
        struct shim_encrypted_file* enc;
        file_off_t size;

        struct shim_encrypted_files_key* key = dent->mount->data;
        ret = encrypted_file_open(uri, key, &enc);
        if (ret < 0) {
            goto out;
        }

        ret = encrypted_file_get_size(enc, &size);
        encrypted_file_put(enc);

        if (ret < 0) {
            encrypted_file_destroy(enc);
            goto out;
        }
        inode->data = enc;
        inode->size = size;
    }
    dent->inode = inode;
    get_inode(inode);
    ret = 0;

out:
    if (inode)
        put_inode(inode);
    free(uri);
    return ret;
}

static int chroot_encrypted_open(struct shim_handle* hdl, struct shim_dentry* dent, int flags) {
    assert(locked(&g_dcache_lock));
    assert(dent->inode);
    __UNUSED(flags);

    int ret;

    if (dent->inode->type == S_IFREG) {
        struct shim_encrypted_file* enc = dent->inode->data;

        lock(&dent->inode->lock);
        ret = encrypted_file_get(enc);
        unlock(&dent->inode->lock);
        if (ret < 0)
            return ret;
    }

    hdl->inode = dent->inode;
    get_inode(dent->inode);
    hdl->type = TYPE_CHROOT_ENCRYPTED;
    hdl->pos = 0;
    return 0;
}

static int chroot_encrypted_creat(struct shim_handle* hdl, struct shim_dentry* dent, int flags,
                                  mode_t perm) {
    assert(locked(&g_dcache_lock));
    assert(!dent->inode);
    __UNUSED(flags);

    char* uri;
    int ret = chroot_dentry_uri(dent, S_IFREG, &uri);
    if (ret < 0)
        return ret;

    struct shim_inode* inode = get_new_inode(dent->mount, S_IFREG, HOST_PERM(perm));
    if (!inode) {
        ret = -ENOMEM;
        goto out;
    }

    struct shim_encrypted_files_key* key = dent->mount->data;
    struct shim_encrypted_file* enc;
    ret = encrypted_file_create(uri, HOST_PERM(perm), key, &enc);
    if (ret < 0)
        goto out;

    /* Keep `enc->use_count` at 1, since it will be associated with the handle */

    inode->data = enc;
    dent->inode = inode;
    get_inode(inode);

    hdl->inode = dent->inode;
    get_inode(inode);
    hdl->type = TYPE_CHROOT_ENCRYPTED;
    hdl->pos = 0;
    ret = 0;
out:
    free(uri);
    if (inode)
        put_inode(inode);
    return ret;
}

static int chroot_encrypted_mkdir(struct shim_dentry* dent, mode_t perm) {
    assert(locked(&g_dcache_lock));
    assert(!dent->inode);

    struct shim_inode* inode = get_new_inode(dent->mount, S_IFDIR, perm);
    if (!inode)
        return -ENOMEM;

    char* uri = NULL;

    int ret = chroot_dentry_uri(dent, S_IFDIR, &uri);
    if (ret < 0)
        goto out;

    /* This opens a "dir:..." URI */
    PAL_HANDLE palhdl;
    ret = DkStreamOpen(uri, PAL_ACCESS_RDONLY, HOST_PERM(perm), PAL_CREATE_ALWAYS,
                       PAL_OPTION_PASSTHROUGH, &palhdl);
    if (ret < 0) {
        ret = pal_to_unix_errno(ret);
        goto out;
    }
    DkObjectClose(palhdl);

    inode->type = S_IFDIR;
    inode->perm = perm;
    dent->inode = inode;
    get_inode(inode);
    ret = 0;

out:
    put_inode(inode);
    free(uri);
    return ret;
}

/* NOTE: this function is different from generic `chroot_unlink` only to add PAL_OPTION_PASSTHROUGH.
 * Once that option is removed, we can safely go back to using `chroot_unlink`. */
static int chroot_encrypted_unlink(struct shim_dentry* dent) {
    assert(locked(&g_dcache_lock));
    assert(dent->inode);

    char* uri;
    int ret = chroot_dentry_uri(dent, dent->inode->type, &uri);
    if (ret < 0)
        return ret;

    PAL_HANDLE palhdl;
    ret = DkStreamOpen(uri, PAL_ACCESS_RDONLY, /*share_flags=*/0, PAL_CREATE_NEVER,
                       PAL_OPTION_PASSTHROUGH, &palhdl);
    if (ret < 0) {
        ret = pal_to_unix_errno(ret);
        goto out;
    }

    ret = DkStreamDelete(palhdl, PAL_DELETE_ALL);
    DkObjectClose(palhdl);
    if (ret < 0) {
        ret = pal_to_unix_errno(ret);
        goto out;
    }
    ret = 0;
out:
    free(uri);
    return ret;
}

static int chroot_encrypted_rename(struct shim_dentry* old, struct shim_dentry* new) {
    assert(locked(&g_dcache_lock));
    assert(old->inode);
    assert(old->inode->type == S_IFREG);

    int ret;
    char* new_uri = NULL;

    ret = chroot_dentry_uri(new, old->inode->type, &new_uri);
    if (ret < 0)
        return ret;

    lock(&old->inode->lock);

    struct shim_encrypted_file* enc = old->inode->data;

    ret = encrypted_file_get(enc);
    if (ret < 0)
        goto out;

    ret = encrypted_file_rename(enc, new_uri);
    encrypted_file_put(enc);
out:
    unlock(&old->inode->lock);
    free(new_uri);
    return ret;
}

static int chroot_encrypted_chmod(struct shim_dentry* dent, mode_t perm) {
    assert(locked(&g_dcache_lock));
    assert(!dent->inode);

    char* uri = NULL;
    lock(&dent->inode->lock);

    int ret = chroot_dentry_uri(dent, dent->inode->type, &uri);
    if (ret < 0)
        goto out;

    PAL_HANDLE palhdl;
    ret = DkStreamOpen(uri, PAL_ACCESS_RDONLY, /*share_flags=*/0, PAL_CREATE_NEVER,
                       PAL_OPTION_PASSTHROUGH, &palhdl);
    if (ret < 0) {
        ret = pal_to_unix_errno(ret);
        goto out;
    }
    mode_t host_perm = HOST_PERM(perm);
    PAL_STREAM_ATTR attr = {.share_flags = host_perm};
    ret = DkStreamAttributesSetByHandle(palhdl, &attr);
    DkObjectClose(palhdl);
    if (ret < 0) {
        ret = pal_to_unix_errno(ret);
        goto out;
    }
    dent->inode->perm = perm;
    ret = 0;

out:
    unlock(&dent->inode->lock);
    free(uri);
    return ret;
}

static int chroot_encrypted_flush(struct shim_handle* hdl) {
    assert(hdl->type == TYPE_CHROOT_ENCRYPTED);
    if (hdl->inode->type != S_IFREG)
        return 0;

    struct shim_encrypted_file* enc = hdl->inode->data;

    /* If there are any MAP_SHARED mappings for the file, this will write data to `enc` */
    int ret = msync_handle(hdl);
    if (ret < 0)
        return ret;

    /* This will write changes from `enc` to host file */
    lock(&hdl->inode->lock);
    ret = encrypted_file_flush(enc);
    unlock(&hdl->inode->lock);
    return ret;
}

static void chroot_encrypted_hdrop(struct shim_handle* hdl) {
    assert(hdl->type == TYPE_CHROOT_ENCRYPTED);
    if (hdl->inode->type != S_IFREG)
        return;

    struct shim_encrypted_file* enc = hdl->inode->data;

    lock(&hdl->inode->lock);
    encrypted_file_put(enc);
    unlock(&hdl->inode->lock);
}

static ssize_t chroot_encrypted_read(struct shim_handle* hdl, void* buf, size_t count,
                                     file_off_t* pos) {
    assert(hdl->type == TYPE_CHROOT_ENCRYPTED);
    if (hdl->inode->type != S_IFREG) {
        assert(hdl->inode->type == S_IFDIR);
        return -EISDIR;
    }

    struct shim_encrypted_file* enc = hdl->inode->data;
    size_t actual_count;

    lock(&hdl->inode->lock);
    int ret = encrypted_file_read(enc, buf, count, *pos, &actual_count);
    unlock(&hdl->inode->lock);

    if (ret < 0)
        return ret;
    assert(actual_count <= count);
    *pos += actual_count;
    return actual_count;
}

static ssize_t chroot_encrypted_write(struct shim_handle* hdl, const void* buf, size_t count,
                                      file_off_t* pos) {
    assert(hdl->type == TYPE_CHROOT_ENCRYPTED);
    if (hdl->inode->type != S_IFREG) {
        assert(hdl->inode->type == S_IFDIR);
        return -EISDIR;
    }

    struct shim_encrypted_file* enc = hdl->inode->data;
    size_t actual_count;

    lock(&hdl->inode->lock);

    int ret = encrypted_file_write(enc, buf, count, *pos, &actual_count);
    if (ret < 0)
        goto out;

    assert(actual_count <= count);
    *pos += actual_count;
    if (hdl->inode->size < *pos)
        hdl->inode->size = *pos;

out:
    unlock(&hdl->inode->lock);
    return actual_count;
}

static int chroot_encrypted_truncate(struct shim_handle* hdl, file_off_t size) {
    assert(hdl->type == TYPE_CHROOT_ENCRYPTED);
    if (hdl->inode->type != S_IFREG) {
        assert(hdl->inode->type == S_IFDIR);
        return -EISDIR;
    }

    int ret;
    struct shim_encrypted_file* enc = hdl->inode->data;

    lock(&hdl->inode->lock);
    ret = encrypted_file_set_size(enc, size);
    if (ret == 0)
        hdl->inode->size = size;
    unlock(&hdl->inode->lock);

    return ret;
}

struct shim_fs_ops chroot_encrypted_fs_ops = {
    .mount      = &chroot_encrypted_mount,
    .flush      = &chroot_encrypted_flush,
    .read       = &chroot_encrypted_read,
    .write      = &chroot_encrypted_write,
    .seek       = &generic_inode_seek,
    .hstat      = &generic_inode_hstat,
    .truncate   = &chroot_encrypted_truncate,
    .poll       = &generic_inode_poll,
    .hdrop      = &chroot_encrypted_hdrop,
    .checkpoint = &chroot_encrypted_checkpoint,
    .migrate    = &chroot_encrypted_migrate,
    .mmap       = &generic_emulated_mmap,
    .msync      = &generic_emulated_msync,
};

struct shim_d_ops chroot_encrypted_d_ops = {
    .open          = &chroot_encrypted_open,
    .lookup        = &chroot_encrypted_lookup,
    .creat         = &chroot_encrypted_creat,
    .mkdir         = &chroot_encrypted_mkdir,
    .stat          = &generic_inode_stat,
    .readdir       = &chroot_readdir, /* same as in `chroot` filesystem */
    .unlink        = &chroot_encrypted_unlink,
    .rename        = &chroot_encrypted_rename,
    .chmod         = &chroot_encrypted_chmod,
    .idrop         = &chroot_encrypted_idrop,
};

struct shim_fs chroot_encrypted_builtin_fs = {
    .name   = "encrypted",
    .fs_ops = &chroot_encrypted_fs_ops,
    .d_ops  = &chroot_encrypted_d_ops,
};
