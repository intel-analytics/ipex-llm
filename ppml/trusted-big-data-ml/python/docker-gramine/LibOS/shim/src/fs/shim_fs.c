/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains code for creating filesystems in library OS.
 */

#include "api.h"
#include "list.h"
#include "pal.h"
#include "shim_checkpoint.h"
#include "shim_fs.h"
#include "shim_fs_encrypted.h"
#include "shim_fs_pseudo.h"
#include "shim_internal.h"
#include "shim_lock.h"
#include "shim_process.h"
#include "shim_utils.h"
#include "toml.h"
#include "toml_utils.h"

struct shim_fs* builtin_fs[] = {
    &chroot_builtin_fs,
    &chroot_encrypted_builtin_fs,
    &tmp_builtin_fs,
    &pipe_builtin_fs,
    &fifo_builtin_fs,
    &socket_builtin_fs,
    &epoll_builtin_fs,
    &eventfd_builtin_fs,
    &pseudo_builtin_fs,
    &synthetic_builtin_fs,
};

static struct shim_lock mount_mgr_lock;

#define SYSTEM_LOCK()   lock(&mount_mgr_lock)
#define SYSTEM_UNLOCK() unlock(&mount_mgr_lock)
#define SYSTEM_LOCKED() locked(&mount_mgr_lock)

#define MOUNT_MGR_ALLOC 64

#define OBJ_TYPE struct shim_mount
#include "memmgr.h"

static MEM_MGR mount_mgr = NULL;
DEFINE_LISTP(shim_mount);
/* Links to mount->list */
static LISTP_TYPE(shim_mount) mount_list;
static struct shim_lock mount_list_lock;

int init_fs(void) {
    mount_mgr = create_mem_mgr(init_align_up(MOUNT_MGR_ALLOC));
    if (!mount_mgr)
        return -ENOMEM;

    int ret;
    if (!create_lock(&mount_mgr_lock) || !create_lock(&mount_list_lock)) {
        ret = -ENOMEM;
        goto err;
    }

    if ((ret = init_encrypted_files()) < 0)
        goto err;

    if ((ret = init_procfs()) < 0)
        goto err;
    if ((ret = init_devfs()) < 0)
        goto err;
    if ((ret = init_sysfs()) < 0)
        goto err;

    return 0;

err:
    destroy_mem_mgr(mount_mgr);
    if (lock_created(&mount_mgr_lock))
        destroy_lock(&mount_mgr_lock);
    if (lock_created(&mount_list_lock))
        destroy_lock(&mount_list_lock);
    return ret;
}

static struct shim_mount* alloc_mount(void) {
    return get_mem_obj_from_mgr_enlarge(mount_mgr, size_align_up(MOUNT_MGR_ALLOC));
}

static void free_mount(struct shim_mount* mount) {
    free_mem_obj_to_mgr(mount_mgr, mount);
}

static bool mount_migrated = false;

static int mount_root(void) {
    int ret;
    char* fs_root_type     = NULL;
    char* fs_root_uri      = NULL;
    char* fs_root_key_name = NULL;

    assert(g_manifest_root);

    ret = toml_string_in(g_manifest_root, "fs.root.type", &fs_root_type);
    if (ret < 0) {
        log_error("Cannot parse 'fs.root.type'");
        ret = -EINVAL;
        goto out;
    }

    ret = toml_string_in(g_manifest_root, "fs.root.uri", &fs_root_uri);
    if (ret < 0) {
        log_error("Cannot parse 'fs.root.uri'");
        ret = -EINVAL;
        goto out;
    }

    ret = toml_string_in(g_manifest_root, "fs.root.key_name", &fs_root_key_name);
    if (ret < 0) {
        log_error("Cannot parse 'fs.root.key_name'");
        ret = -EINVAL;
        goto out;
    }

    struct shim_mount_params params = {
        .path = "/",
        .key_name = fs_root_key_name,
    };

    if (!fs_root_type && !fs_root_uri) {
        params.type = "chroot";
        params.uri = URI_PREFIX_FILE ".";
    } else if (!fs_root_type || !strcmp(fs_root_type, "chroot")) {
        if (!fs_root_uri) {
            log_error("No value provided for 'fs.root.uri'");
            ret = -EINVAL;
            goto out;
        }
        params.type = "chroot";
        params.uri = fs_root_uri;
    } else {
        params.type = fs_root_type;
        params.uri = fs_root_uri;
    }
    ret = mount_fs(&params);

out:
    free(fs_root_type);
    free(fs_root_uri);
    return ret;
}

static int mount_sys(void) {
    int ret;

    ret = mount_fs(&(struct shim_mount_params){
        .type = "pseudo",
        .path = "/proc",
        .uri = "proc",
    });
    if (ret < 0)
        return ret;

    ret = mount_fs(&(struct shim_mount_params){
        .type = "pseudo",
        .path = "/dev",
        .uri = "dev",
    });
    if (ret < 0)
        return ret;

    ret = mount_fs(&(struct shim_mount_params){
        .type = "chroot",
        .path = "/dev/tty",
        .uri = URI_PREFIX_DEV "tty",
    });
    if (ret < 0)
        return ret;

    ret = mount_fs(&(struct shim_mount_params){
        .type = "pseudo",
        .path = "/sys",
        .uri = "sys",
    });
    if (ret < 0)
        return ret;

    return 0;
}

static int mount_one_nonroot(toml_table_t* mount, const char* prefix) {
    assert(mount);

    int ret;

    char* mount_type     = NULL;
    char* mount_path     = NULL;
    char* mount_uri      = NULL;
    char* mount_key_name = NULL;

    ret = toml_string_in(mount, "type", &mount_type);
    if (ret < 0) {
        log_error("Cannot parse '%s.type'", prefix);
        ret = -EINVAL;
        goto out;
    }

    ret = toml_string_in(mount, "path", &mount_path);
    if (ret < 0) {
        log_error("Cannot parse '%s.path'", prefix);
        ret = -EINVAL;
        goto out;
    }

    ret = toml_string_in(mount, "uri", &mount_uri);
    if (ret < 0) {
        log_error("Cannot parse '%s.uri'", prefix);
        ret = -EINVAL;
        goto out;
    }

    ret = toml_string_in(mount, "key_name", &mount_key_name);
    if (ret < 0) {
        log_error("Cannot parse '%s.key_name'", prefix);
        ret = -EINVAL;
        goto out;
    }

    if (!mount_path) {
        log_error("No value provided for '%s.path'", prefix);
        ret = -EINVAL;
        goto out;
    }

    if (!strcmp(mount_path, "/")) {
        log_error("'%s.path' cannot be \"/\". The root mount (\"/\") can be customized "
                  "via the 'fs.root' manifest entry.", prefix);
        ret = -EINVAL;
        goto out;
    }

    if (mount_path[0] != '/') {
        /* FIXME: Relative paths are deprecated starting from Gramine v1.2, we can disallow them
         * completely two versions after it. */
        if (!strcmp(mount_path, ".") || !strcmp(mount_path, "..")) {
            log_error("Mount points '.' and '..' are not allowed, use absolute paths instead.");
            ret = -EINVAL;
            goto out;
        }
        log_error("Detected deprecated syntax: '%s.path' (\"%s\") is not absolute. "
                  "Consider converting it to absolute by adding \"/\" at the beginning.",
                  prefix, mount_path);
    }

    if (!mount_type || !strcmp(mount_type, "chroot")) {
        if (!mount_uri) {
            log_error("No value provided for '%s.uri'", prefix);
            ret = -EINVAL;
            goto out;
        }

        if (!strcmp(mount_uri, "file:/proc") ||
                !strcmp(mount_uri, "file:/sys") ||
                !strcmp(mount_uri, "file:/dev") ||
                !strncmp(mount_uri, "file:/proc/", strlen("file:/proc/")) ||
                !strncmp(mount_uri, "file:/sys/", strlen("file:/sys/")) ||
                !strncmp(mount_uri, "file:/dev/", strlen("file:/dev/"))) {
            log_error("Mounting %s may expose unsanitized, unsafe files to unsuspecting "
                      "application. Gramine will continue application execution, but this "
                      "configuration is not recommended for use in production!", mount_uri);
        }
    }

    struct shim_mount_params params = {
        .type = mount_type ?: "chroot",
        .path = mount_path,
        .uri = mount_uri,
        .key_name = mount_key_name,
    };
    ret = mount_fs(&params);

out:
    free(mount_type);
    free(mount_path);
    free(mount_uri);
    free(mount_key_name);
    return ret;
}

/*
 * Mount filesystems using the deprecated TOML-table syntax.
 *
 * FIXME: This is deprecated starting from Gramine v1.2 and can be removed two versions after it.
 */
static int mount_nonroot_from_toml_table(void) {
    int ret = 0;

    assert(g_manifest_root);
    toml_table_t* manifest_fs = toml_table_in(g_manifest_root, "fs");
    if (!manifest_fs)
        return 0;

    toml_table_t* manifest_fs_mounts = toml_table_in(manifest_fs, "mount");
    if (!manifest_fs_mounts)
        return 0;

    ssize_t mounts_cnt = toml_table_ntab(manifest_fs_mounts);
    if (mounts_cnt < 0)
        return -EINVAL;
    if (mounts_cnt == 0)
        return 0;

    log_error("Detected deprecated syntax: 'fs.mount'. Consider converting to the new array "
              "syntax: 'fs.mounts = [{ type = \"chroot\", uri = \"...\", path = \"...\" }]'.");

    /*
     * *** Warning: A _very_ ugly hack below ***
     *
     * In the TOML-table syntax, the entries are not ordered, but Gramine actually relies on the
     * specific mounting order (e.g. you can't mount /lib/asdf first and then /lib, but the other
     * way around works). The problem is, that TOML structure is just a dictionary, so the order of
     * keys is not preserved.
     *
     * To fix the issue, we use an ugly heuristic - we apply mounts sorted by the path length, which
     * in most cases should result in a proper mount order.
     *
     * We do this in O(n^2) because we don't have a sort function, but that shouldn't be an issue -
     * usually there are around 5 mountpoints with ~30 chars in paths, so it should still be quite
     * fast.
     *
     * Fortunately, the table syntax is deprecated, so we'll be able to remove this code in a future
     * Gramine release.
     *
     * Corresponding issue: https://github.com/gramineproject/gramine/issues/23.
     */
    const char** keys = malloc(mounts_cnt * sizeof(*keys));
    if (!keys)
        return -ENOMEM;

    size_t* lengths = malloc(mounts_cnt * sizeof(*lengths));
    if (!lengths) {
        ret = -ENOMEM;
        goto out;
    }

    size_t longest = 0;
    for (ssize_t i = 0; i < mounts_cnt; i++) {
        keys[i] = toml_key_in(manifest_fs_mounts, i);
        assert(keys[i]);

        toml_table_t* mount = toml_table_in(manifest_fs_mounts, keys[i]);
        assert(mount);
        char* mount_path;
        ret = toml_string_in(mount, "path", &mount_path);
        if (ret < 0 || !mount_path) {
            if (!ret)
                ret = -ENOENT;
            goto out;
        }
        lengths[i] = strlen(mount_path);
        longest = MAX(longest, lengths[i]);
        free(mount_path);
    }

    for (size_t i = 0; i <= longest; i++) {
        for (ssize_t j = 0; j < mounts_cnt; j++) {
            if (lengths[j] != i)
                continue;
            toml_table_t* mount = toml_table_in(manifest_fs_mounts, keys[j]);
            assert(mount);

            char* prefix = alloc_concat("fs.mount.", -1, keys[j], -1);
            if (!prefix) {
                ret = -ENOMEM;
                goto out;
            }
            ret = mount_one_nonroot(mount, prefix);
            free(prefix);
            if (ret < 0)
                goto out;
        }
    }
out:
    free(keys);
    free(lengths);
    return ret;
}

static int mount_nonroot_from_toml_array(void) {
    int ret;

    assert(g_manifest_root);
    toml_table_t* manifest_fs = toml_table_in(g_manifest_root, "fs");
    if (!manifest_fs)
        return 0;

    toml_array_t* manifest_fs_mounts = toml_array_in(manifest_fs, "mounts");
    if (!manifest_fs_mounts)
        return 0;

    ssize_t mounts_cnt = toml_array_nelem(manifest_fs_mounts);
    if (mounts_cnt < 0)
        return -EINVAL;

    for (size_t i = 0; i < (size_t)mounts_cnt; i++) {
        toml_table_t* mount = toml_table_at(manifest_fs_mounts, i);
        if (!mount) {
            log_error("Invalid mount in manifest at index %zd (not a TOML table)", i);
            return -EINVAL;
        }

        char prefix[static_strlen("fs.mounts[]") + 21];
        snprintf(prefix, sizeof(prefix), "fs.mounts[%zu]", i);

        ret = mount_one_nonroot(mount, prefix);
        if (ret < 0)
            return ret;
    }
    return 0;
}

/*
 * Find where a file with given URI was supposed to be mounted. For instance, if the URI is
 * `file:a/b/c/d`, and there is a mount `{ uri = "file:a/b", path = "/x/y" }`, then this function
 * will set `*out_file_path` to `/x/y/c/d`. Only "chroot" mounts are considered.
 *
 * The caller is supposed to free `*out_file_path`.
 *
 * This function is used for interpreting legacy `sgx.protected_files` syntax as mounts.
 */
static int find_host_file_mount_path(const char* uri, char** out_file_path) {
    if (!strstartswith(uri, URI_PREFIX_FILE))
        return -EINVAL;

    size_t uri_len = strlen(uri);

    bool found = false;
    char* file_path = NULL;

    /* Traverse the mount list in reverse: we want to find the latest mount that applies. */
    struct shim_mount* mount;
    lock(&mount_list_lock);
    LISTP_FOR_EACH_ENTRY_REVERSE(mount, &mount_list, list) {
        if (strcmp(mount->fs->name, "chroot") != 0 || !strstartswith(mount->uri, URI_PREFIX_FILE))
            continue;

        const char* mount_uri = mount->uri;
        size_t mount_uri_len = strlen(mount_uri);

        /* Check if `mount_uri` is equal to `uri`, or is an ancestor of `uri` */
        if (mount_uri_len <= uri_len && !memcmp(mount_uri, uri, mount_uri_len) &&
                (!uri[mount_uri_len] || uri[mount_uri_len] == '/')) {
            /* `rest` is either empty, or begins with '/' */
            const char* rest = uri + mount_uri_len;
            found = true;
            file_path = alloc_concat(mount->path, -1, rest, -1);
            break;
        }

        /* Special case: this is the mount of the current directory, and `uri` is not absolute */
        if (!strcmp(mount_uri, "file:.") && uri[static_strlen(URI_PREFIX_FILE)] != '/') {
            found = true;
            file_path = strdup(uri + static_strlen(URI_PREFIX_FILE));
            break;
        }
    }
    unlock(&mount_list_lock);

    if (!found)
        return -ENOENT;

    if (!file_path)
        return -ENOMEM;

    *out_file_path = file_path;
    return 0;
}

/*
 * Parse an array of SGX protected files (`sgx.{array_name}`) and interpret every entry as a `type =
 * "encrypted"` mount.
 *
 * NOTE: This covers only the simplest cases. It will not work correctly if a given file was mounted
 * multiple times, or mounted under a path shadowed by another mount.
 */
static int mount_protected_files(const char* array_name, const char* key_name) {
    assert(g_manifest_root);
    toml_table_t* manifest_sgx = toml_table_in(g_manifest_root, "sgx");
    if (!manifest_sgx)
        return 0;

    toml_array_t* manifest_sgx_protected_files = toml_array_in(manifest_sgx, array_name);
    if (!manifest_sgx_protected_files)
        return 0;

    ssize_t protected_files_cnt = toml_array_nelem(manifest_sgx_protected_files);
    if (protected_files_cnt < 0)
        return -EINVAL;

    if (protected_files_cnt == 0)
        return 0;

    log_error("Detected deprecated syntax: 'sgx.%s'. Consider converting it to a list "
              "of mounts with 'type = \"encrypted\"' and 'key_name = \"%s\"'.",
              array_name, key_name);

    int ret;
    char* uri = NULL;
    char* mount_path = NULL;
    for (ssize_t i = 0; i < protected_files_cnt; i++) {
        toml_raw_t uri_raw = toml_raw_at(manifest_sgx_protected_files, i);
        if (!uri_raw) {
            log_error("Cannot parse 'sgx.%s[%zd]'", array_name, i);
            ret = -EINVAL;
            goto out;
        }

        ret = toml_rtos(uri_raw, &uri);
        if (ret < 0) {
            log_error("Cannot parse 'sgx.%s[%zd]' (not a string)", array_name, i);
            ret = -EINVAL;
            goto out;
        }

        ret = find_host_file_mount_path(uri, &mount_path);
        if (ret < 0) {
            log_error("Cannot determine mount path for encrypted file '%s'. "
                      "Consider converting 'sgx.%s' to a list of mounts, as mentioned above.",
                      uri, array_name);
            goto out;
        }

        struct shim_mount_params params = {
            .type = "encrypted",
            .path = mount_path,
            .uri = uri,
            .key_name = key_name,
        };
        ret = mount_fs(&params);
        if (ret < 0) {
            ret = -EINVAL;
            goto out;
        }

        free(uri);
        uri = NULL;
        free(mount_path);
        mount_path = NULL;
    }
    ret = 0;
out:
    free(uri);
    free(mount_path);
    return ret;
}

int init_mount_root(void) {
    if (mount_migrated)
        return 0;

    int ret;

    ret = mount_root();
    if (ret < 0)
        return ret;

    ret = mount_sys();
    if (ret < 0)
        return ret;

    return 0;
}

int init_mount(void) {
    if (mount_migrated)
        return 0;

    int ret;

    ret = mount_nonroot_from_toml_table();
    if (ret < 0)
        return ret;

    ret = mount_nonroot_from_toml_array();
    if (ret < 0)
        return ret;

    /*
     * If we're under SGX PAL, handle old protected files syntax.
     *
     * TODO: this is deprecated in v1.2, remove two versions later.
     */
    if (!strcmp(g_pal_public_state->host_type, "Linux-SGX")) {
        ret = mount_protected_files("protected_files", "default");
        if (ret < 0)
            return ret;
        ret = mount_protected_files("protected_mrenclave_files", "_sgx_mrenclave");
        if (ret < 0)
            return ret;
        ret = mount_protected_files("protected_mrsigner_files", "_sgx_mrsigner");
        if (ret < 0)
            return ret;
    }

    assert(g_manifest_root);

    char* fs_start_dir = NULL;
    ret = toml_string_in(g_manifest_root, "fs.start_dir", &fs_start_dir);
    if (ret < 0) {
        log_error("Can't parse 'fs.start_dir'");
        return ret;
    }

    if (fs_start_dir) {
        struct shim_dentry* dent = NULL;

        lock(&g_dcache_lock);
        ret = path_lookupat(/*start=*/NULL, fs_start_dir, LOOKUP_FOLLOW | LOOKUP_DIRECTORY, &dent);
        unlock(&g_dcache_lock);

        free(fs_start_dir);
        if (ret < 0) {
            log_error("Invalid 'fs.start_dir' in manifest.");
            return ret;
        }
        lock(&g_process.fs_lock);
        put_dentry(g_process.cwd);
        g_process.cwd = dent;
        unlock(&g_process.fs_lock);
    }
    /* Otherwise `cwd` is already initialized. */

    return 0;
}

struct shim_fs* find_fs(const char* name) {
    for (size_t i = 0; i < ARRAY_SIZE(builtin_fs); i++) {
        struct shim_fs* fs = builtin_fs[i];
        if (!strncmp(fs->name, name, sizeof(fs->name)))
            return fs;
    }

    return NULL;
}

static int mount_fs_at_dentry(struct shim_mount_params* params, struct shim_dentry* mount_point) {
    assert(locked(&g_dcache_lock));
    assert(!mount_point->attached_mount);

    int ret;
    struct shim_fs* fs = find_fs(params->type);
    if (!fs || !fs->fs_ops || !fs->fs_ops->mount)
        return -ENODEV;

    if (!fs->d_ops || !fs->d_ops->lookup)
        return -ENODEV;

    void* mount_data = NULL;

    /* Call filesystem-specific mount operation */
    if ((ret = fs->fs_ops->mount(params, &mount_data)) < 0)
        return ret;

    /* Allocate and set up `shim_mount` object */

    struct shim_mount* mount = alloc_mount();
    if (!mount) {
        ret = -ENOMEM;
        goto err;
    }
    memset(mount, 0, sizeof(*mount));

    mount->path = strdup(params->path);
    if (!mount->path) {
        ret = -ENOMEM;
        goto err;
    }
    if (params->uri) {
        mount->uri = strdup(params->uri);
        if (!mount->uri) {
            ret = -ENOMEM;
            goto err;
        }
    } else {
        mount->uri = NULL;
    }
    mount->fs = fs;
    mount->data = mount_data;

    /* Attach mount to mountpoint, and the other way around */

    mount->mount_point = mount_point;
    get_dentry(mount_point);
    mount_point->attached_mount = mount;
    get_mount(mount);

    /* Initialize root dentry of the new filesystem */

    mount->root = get_new_dentry(mount, /*parent=*/NULL, mount_point->name, mount_point->name_len);
    if (!mount->root) {
        ret = -ENOMEM;
        goto err;
    }

    /*
     * Trigger filesystem lookup for the root dentry, so that it's already positive. If there is a
     * problem looking up the root, we want the mount operation to fail.
     *
     * We skip the lookup for the `encrypted` filesystem, because the key for encrypted files might
     * not be set yet.
     */
    if (strcmp(params->type, "encrypted") != 0) {
        struct shim_dentry* root;
        if ((ret = path_lookupat(g_dentry_root, params->path, LOOKUP_NO_FOLLOW, &root))) {
            log_warning("error looking up mount root %s: %d", params->path, ret);
            goto err;
        }
        assert(root == mount->root);
        put_dentry(root);
    }

    /* Add `mount` to the global list */

    lock(&mount_list_lock);
    LISTP_ADD_TAIL(mount, &mount_list, list);
    get_mount(mount);
    unlock(&mount_list_lock);

    return 0;

err:
    if (mount_point->attached_mount)
        mount_point->attached_mount = NULL;

    if (mount) {
        if (mount->mount_point)
            put_dentry(mount_point);

        if (mount->root)
            put_dentry(mount->root);

        free_mount(mount);
    }

    if (fs->fs_ops->unmount) {
        int ret_unmount = fs->fs_ops->unmount(mount_data);
        if (ret_unmount < 0) {
            log_warning("error unmounting %s: %d", params->path, ret_unmount);
        }
    }

    return ret;
}

int mount_fs(struct shim_mount_params* params) {
    int ret;
    struct shim_dentry* mount_point = NULL;

    log_debug("mounting \"%s\" (%s) under %s", params->uri, params->type, params->path);

    lock(&g_dcache_lock);

    if (!g_dentry_root->attached_mount && !strcmp(params->path, "/")) {
        /* `g_dentry_root` does not belong to any mounted filesystem, so lookup will fail. Use it
         * directly. */
        mount_point = g_dentry_root;
        get_dentry(g_dentry_root);
    } else {
        int lookup_flags = LOOKUP_NO_FOLLOW | LOOKUP_MAKE_SYNTHETIC;
        ret = path_lookupat(g_dentry_root, params->path, lookup_flags, &mount_point);
        if (ret < 0) {
            log_error("error looking up mountpoint %s: %d", params->path, ret);
            goto out;
        }
    }

    if ((ret = mount_fs_at_dentry(params, mount_point)) < 0) {
        log_error("error mounting \"%s\" (%s) under %s: %d", params->uri, params->type,
                  params->path, ret);
        goto out;
    }

    ret = 0;
out:
    if (mount_point)
        put_dentry(mount_point);
    unlock(&g_dcache_lock);

    return ret;
}

/*
 * XXX: These two functions are useless - `mount` is not freed even if refcount reaches 0.
 * Unfortunately Gramine is not keeping track of this refcount correctly, so we cannot free
 * the object. Fixing this would require revising whole filesystem implementation - but this code
 * is, uhm, not the best achievement of humankind and probably requires a complete rewrite.
 */
void get_mount(struct shim_mount* mount) {
    __UNUSED(mount);
    // REF_INC(mount->ref_count);
}

void put_mount(struct shim_mount* mount) {
    __UNUSED(mount);
    // REF_DEC(mount->ref_count);
}

int walk_mounts(int (*walk)(struct shim_mount* mount, void* arg), void* arg) {
    struct shim_mount* mount;
    struct shim_mount* n;
    int ret = 0;
    int nsrched = 0;

    lock(&mount_list_lock);

    LISTP_FOR_EACH_ENTRY_SAFE(mount, n, &mount_list, list) {
        if ((ret = (*walk)(mount, arg)) < 0)
            break;

        if (ret > 0)
            nsrched++;
    }

    unlock(&mount_list_lock);
    return ret < 0 ? ret : (nsrched ? 0 : -ESRCH);
}

struct shim_mount* find_mount_from_uri(const char* uri) {
    struct shim_mount* mount;
    struct shim_mount* found = NULL;
    size_t longest_path = 0;

    lock(&mount_list_lock);
    LISTP_FOR_EACH_ENTRY(mount, &mount_list, list) {
        if (!mount->uri)
            continue;

        if (strcmp(mount->uri, uri) == 0) {
            size_t path_len = strlen(mount->path);
            if (path_len > longest_path) {
                longest_path = path_len;
                found = mount;
            }
        }
    }

    if (found)
        get_mount(found);

    unlock(&mount_list_lock);
    return found;
}

/*
 * Note that checkpointing the `shim_fs` structure copies it, instead of using a pointer to
 * corresponding global object on the remote side. This does not waste too much memory (because each
 * global object is only copied once), but it means that `shim_fs` objects cannot be compared by
 * pointer.
 */
BEGIN_CP_FUNC(fs) {
    __UNUSED(size);
    assert(size == sizeof(struct shim_fs));

    struct shim_fs* fs = (struct shim_fs*)obj;
    struct shim_fs* new_fs = NULL;

    size_t off = GET_FROM_CP_MAP(obj);

    if (!off) {
        off = ADD_CP_OFFSET(sizeof(struct shim_fs));
        ADD_TO_CP_MAP(obj, off);

        new_fs = (struct shim_fs*)(base + off);

        memcpy(new_fs->name, fs->name, sizeof(new_fs->name));
        new_fs->fs_ops = NULL;
        new_fs->d_ops = NULL;

        ADD_CP_FUNC_ENTRY(off);
    } else {
        new_fs = (struct shim_fs*)(base + off);
    }

    if (objp)
        *objp = (void*)new_fs;
}
END_CP_FUNC(fs)

BEGIN_RS_FUNC(fs) {
    __UNUSED(offset);
    __UNUSED(rebase);
    struct shim_fs* fs = (void*)(base + GET_CP_FUNC_ENTRY());

    struct shim_fs* builtin_fs = find_fs(fs->name);
    if (!builtin_fs)
        return -EINVAL;

    fs->fs_ops = builtin_fs->fs_ops;
    fs->d_ops = builtin_fs->d_ops;
}
END_RS_FUNC(fs)

BEGIN_CP_FUNC(mount) {
    __UNUSED(size);
    assert(size == sizeof(struct shim_mount));

    struct shim_mount* mount     = (struct shim_mount*)obj;
    struct shim_mount* new_mount = NULL;

    size_t off = GET_FROM_CP_MAP(obj);

    if (!off) {
        off = ADD_CP_OFFSET(sizeof(struct shim_mount));
        ADD_TO_CP_MAP(obj, off);

        mount->cpdata = NULL;
        if (mount->fs->fs_ops && mount->fs->fs_ops->checkpoint) {
            void* cpdata = NULL;
            int bytes = mount->fs->fs_ops->checkpoint(&cpdata, mount->data);
            if (bytes > 0) {
                mount->cpdata = cpdata;
                mount->cpsize = bytes;
            }
        }

        new_mount  = (struct shim_mount*)(base + off);
        *new_mount = *mount;

        DO_CP(fs, mount->fs, &new_mount->fs);

        if (mount->cpdata) {
            size_t cp_off = ADD_CP_OFFSET(mount->cpsize);
            memcpy((char*)base + cp_off, mount->cpdata, mount->cpsize);
            new_mount->cpdata = (char*)base + cp_off;
        }

        new_mount->data        = NULL;
        new_mount->mount_point = NULL;
        new_mount->root        = NULL;
        INIT_LIST_HEAD(new_mount, list);
        REF_SET(new_mount->ref_count, 0);

        DO_CP_MEMBER(str, mount, new_mount, path);

        if (mount->uri)
            DO_CP_MEMBER(str, mount, new_mount, uri);

        if (mount->mount_point)
            DO_CP_MEMBER(dentry, mount, new_mount, mount_point);

        if (mount->root)
            DO_CP_MEMBER(dentry, mount, new_mount, root);

        ADD_CP_FUNC_ENTRY(off);
    } else {
        new_mount = (struct shim_mount*)(base + off);
    }

    if (objp)
        *objp = (void*)new_mount;
}
END_CP_FUNC(mount)

BEGIN_RS_FUNC(mount) {
    __UNUSED(offset);
    struct shim_mount* mount = (void*)(base + GET_CP_FUNC_ENTRY());

    CP_REBASE(mount->cpdata);
    CP_REBASE(mount->list);
    CP_REBASE(mount->mount_point);
    CP_REBASE(mount->root);
    CP_REBASE(mount->path);
    CP_REBASE(mount->uri);

    if (mount->mount_point) {
        get_dentry(mount->mount_point);
    }

    if (mount->root) {
        get_dentry(mount->root);
    }

    CP_REBASE(mount->fs);
    if (mount->fs->fs_ops && mount->fs->fs_ops->migrate && mount->cpdata) {
        void* mount_data = NULL;
        if (mount->fs->fs_ops->migrate(mount->cpdata, &mount_data) == 0)
            mount->data = mount_data;
        mount->cpdata = NULL;
    }

    LISTP_ADD_TAIL(mount, &mount_list, list);

    if (mount->path) {
        DEBUG_RS("type=%s,uri=%s,path=%s", mount->type, mount->uri, mount->path);
    } else {
        DEBUG_RS("type=%s,uri=%s", mount->type, mount->uri);
    }
}
END_RS_FUNC(mount)

BEGIN_CP_FUNC(all_mounts) {
    __UNUSED(obj);
    __UNUSED(size);
    __UNUSED(objp);
    struct shim_mount* mount;
    lock(&mount_list_lock);
    LISTP_FOR_EACH_ENTRY(mount, &mount_list, list) {
        DO_CP(mount, mount, NULL);
    }
    unlock(&mount_list_lock);

    /* add an empty entry to mark as migrated */
    ADD_CP_FUNC_ENTRY(0UL);
}
END_CP_FUNC(all_mounts)

BEGIN_RS_FUNC(all_mounts) {
    __UNUSED(entry);
    __UNUSED(base);
    __UNUSED(offset);
    __UNUSED(rebase);
    /* to prevent file system from being mount again */
    mount_migrated = true;
}
END_RS_FUNC(all_mounts)
