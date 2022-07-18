/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*!
 * \file
 *
 * This file contains the implementation of `/dev` pseudo-filesystem.
 */

#include "pal.h"
#include "shim_fs_pseudo.h"

static ssize_t dev_null_read(struct shim_handle* hdl, void* buf, size_t count) {
    __UNUSED(hdl);
    __UNUSED(buf);
    __UNUSED(count);
    return 0;
}

static ssize_t dev_null_write(struct shim_handle* hdl, const void* buf, size_t count) {
    __UNUSED(hdl);
    __UNUSED(buf);
    __UNUSED(count);
    return count;
}

static int64_t dev_null_seek(struct shim_handle* hdl, int64_t offset, int whence) {
    __UNUSED(hdl);
    __UNUSED(offset);
    __UNUSED(whence);
    return 0;
}

/* TODO: ftruncate() on /dev/null should fail, but open() with O_TRUNC should succeed */
static int dev_null_truncate(struct shim_handle* hdl, uint64_t size) {
    __UNUSED(hdl);
    __UNUSED(size);
    return 0;
}

static ssize_t dev_zero_read(struct shim_handle* hdl, void* buf, size_t count) {
    __UNUSED(hdl);
    memset(buf, 0, count);
    return count;
}

static ssize_t dev_random_read(struct shim_handle* hdl, void* buf, size_t count) {
    __UNUSED(hdl);
    int ret = DkRandomBitsRead(buf, count);

    if (ret < 0)
        return pal_to_unix_errno(ret);
    return count;
}

int init_devfs(void) {
    struct pseudo_node* root = pseudo_add_root_dir("dev");

    /* Device minor numbers for pseudo-devices:
     * https://elixir.bootlin.com/linux/v5.9/source/drivers/char/mem.c#L950 */

    struct pseudo_node* null = pseudo_add_dev(root, "null");
    null->perm = PSEUDO_PERM_FILE_RW;
    null->dev.major = 1;
    null->dev.minor = 3;
    null->dev.dev_ops.read = &dev_null_read;
    null->dev.dev_ops.write = &dev_null_write;
    null->dev.dev_ops.seek = &dev_null_seek;
    null->dev.dev_ops.truncate = &dev_null_truncate;

    struct pseudo_node* zero = pseudo_add_dev(root, "zero");
    zero->perm = PSEUDO_PERM_FILE_RW;
    zero->dev.major = 1;
    zero->dev.minor = 5;
    zero->dev.dev_ops.read = &dev_zero_read;
    zero->dev.dev_ops.write = &dev_null_write;
    zero->dev.dev_ops.seek = &dev_null_seek;
    zero->dev.dev_ops.truncate = &dev_null_truncate;

    struct pseudo_node* random = pseudo_add_dev(root, "random");
    random->perm = PSEUDO_PERM_FILE_RW;
    random->dev.major = 1;
    random->dev.minor = 8;
    random->dev.dev_ops.read = &dev_random_read;
    /* writes in /dev/random add entropy in normal Linux, but not implemented in Gramine */
    random->dev.dev_ops.write = &dev_null_write;
    random->dev.dev_ops.seek = &dev_null_seek;

    struct pseudo_node* urandom = pseudo_add_dev(root, "urandom");
    urandom->perm = PSEUDO_PERM_FILE_RW;
    urandom->dev.major = 1;
    urandom->dev.minor = 9;
    /* /dev/urandom is implemented the same as /dev/random, so it has the same operations */
    urandom->dev.dev_ops = random->dev.dev_ops;

    struct pseudo_node* stdin = pseudo_add_link(root, "stdin", NULL);
    stdin->link.target = "/proc/self/fd/0";
    struct pseudo_node* stdout = pseudo_add_link(root, "stdout", NULL);
    stdout->link.target = "/proc/self/fd/1";
    struct pseudo_node* stderr = pseudo_add_link(root, "stderr", NULL);
    stderr->link.target = "/proc/self/fd/2";

    int ret = init_attestation(root);
    if (ret < 0)
        return ret;

    return 0;
}
