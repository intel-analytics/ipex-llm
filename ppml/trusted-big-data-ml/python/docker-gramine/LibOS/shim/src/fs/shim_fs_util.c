/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

#include "cpu.h"
#include "shim_flags_conv.h"
#include "shim_fs.h"
#include "shim_lock.h"
#include "stat.h"

int generic_seek(file_off_t pos, file_off_t size, file_off_t offset, int origin,
                 file_off_t* out_pos) {
    assert(pos >= 0);
    assert(size >= 0);

    switch (origin) {
        case SEEK_SET:
            pos = offset;
            break;

        case SEEK_CUR:
            if (__builtin_add_overflow(pos, offset, &pos))
                return -EOVERFLOW;
            break;

        case SEEK_END:
            if (__builtin_add_overflow(size, offset, &pos))
                return -EOVERFLOW;
            break;

        default:
            return -EINVAL;
    }

    if (pos < 0)
        return -EINVAL;

    *out_pos = pos;
    return 0;
}

int generic_readdir(struct shim_dentry* dent, readdir_callback_t callback, void* arg) {
    assert(locked(&g_dcache_lock));
    assert(dent->inode);
    assert(dent->inode->type == S_IFDIR);

    struct shim_dentry* child;
    LISTP_FOR_EACH_ENTRY(child, &dent->children, siblings) {
        if (child->inode) {
            int ret = callback(child->name, arg);
            if (ret < 0)
                return ret;
        }
    }
    return 0;
}

static int generic_istat(struct shim_inode* inode, struct stat* buf) {
    memset(buf, 0, sizeof(*buf));

    lock(&inode->lock);
    buf->st_mode = inode->type | inode->perm;
    buf->st_size = inode->size;
    /* Some programs (e.g. some tests from LTP) require this value. We've picked some random,
     * pretty looking constant - exact value should not affect anything (perhaps except
     * performance). */
    buf->st_blksize = 0x1000;
    /*
     * Pretend `nlink` is 2 for directories (to account for "." and ".."), 1 for other files.
     *
     * Applications are unlikely to depend on exact value of `nlink`, and for us, it's inconvenient
     * to keep track of the exact value (we would have to list the directory, and also take into
     * account synthetic files created by Graphene, such as named pipes and sockets).
     */
    buf->st_nlink = (inode->type == S_IFDIR ? 2 : 1);

    if (inode->mount->uri)
        buf->st_dev = hash_str(inode->mount->uri);

    unlock(&inode->lock);
    return 0;
}

int generic_inode_stat(struct shim_dentry* dent, struct stat* buf) {
    assert(locked(&g_dcache_lock));
    assert(dent->inode);

    return generic_istat(dent->inode, buf);
}

int generic_inode_hstat(struct shim_handle* hdl, struct stat* buf) {
    assert(hdl->inode);

    return generic_istat(hdl->inode, buf);
}

file_off_t generic_inode_seek(struct shim_handle* hdl, file_off_t offset, int origin) {
    file_off_t ret;

    lock(&hdl->pos_lock);
    lock(&hdl->inode->lock);
    file_off_t pos = hdl->pos;
    file_off_t size = hdl->inode->size;

    ret = generic_seek(pos, size, offset, origin, &pos);
    if (ret == 0) {
        hdl->pos = pos;
        ret = pos;
    }
    unlock(&hdl->inode->lock);
    unlock(&hdl->pos_lock);
    return ret;
}

int generic_inode_poll(struct shim_handle* hdl, int poll_type) {
    int ret;

    lock(&hdl->pos_lock);
    lock(&hdl->inode->lock);

    if (hdl->inode->type == S_IFREG) {
        ret = 0;
        if (poll_type & FS_POLL_WR)
            ret |= FS_POLL_WR;
        /* TODO: The `hdl->pos < hdl->inode->size` condition is wrong, the `poll` syscall treats
         * end-of-file as readable. Check if removing this condition doesn't break anything
         * in our `poll` implementation. */
        if ((poll_type & FS_POLL_RD) && hdl->pos < hdl->inode->size)
            ret |= FS_POLL_RD;
    } else {
        ret = -EAGAIN;
    }

    unlock(&hdl->inode->lock);
    unlock(&hdl->pos_lock);
    return ret;
}

int generic_emulated_mmap(struct shim_handle* hdl, void* addr, size_t size, int prot, int flags,
                          uint64_t offset) {
    assert(addr);

    int ret;

    pal_prot_flags_t pal_prot = LINUX_PROT_TO_PAL(prot, flags);
    pal_prot_flags_t pal_prot_writable = pal_prot | PAL_PROT_WRITE;

    void* actual_addr = addr;
    ret = DkVirtualMemoryAlloc(&actual_addr, size, /*alloc_type=*/0, pal_prot_writable);
    if (ret < 0)
        return pal_to_unix_errno(ret);

    assert(actual_addr == addr);

    size_t read_size = size;
    char* read_addr = addr;
    file_off_t pos = offset;
    while (read_size > 0) {
        ssize_t count = hdl->fs->fs_ops->read(hdl, read_addr, read_size, &pos);
        if (count < 0) {
            if (count == -EINTR)
                continue;
            ret = count;
            goto err;
        }

        if (count == 0)
            break;

        assert((size_t)count <= read_size);
        read_size -= count;
        read_addr += count;
    }

    if (pal_prot != pal_prot_writable) {
        ret = DkVirtualMemoryProtect(addr, size, pal_prot);
        if (ret < 0) {
            ret = pal_to_unix_errno(ret);
            goto err;
        }
    }

    return 0;

err:;
    int free_ret = DkVirtualMemoryFree(addr, size);
    if (free_ret < 0) {
        log_debug("%s: DkVirtualMemoryFree failed on cleanup: %d", __func__, free_ret);
        BUG();
    }
    return ret;
}

int generic_emulated_msync(struct shim_handle* hdl, void* addr, size_t size, int prot, int flags,
                           uint64_t offset) {
    assert(!(flags & MAP_PRIVATE));

    lock(&hdl->inode->lock);
    file_off_t file_size = hdl->inode->size;
    unlock(&hdl->inode->lock);

    pal_prot_flags_t pal_prot = LINUX_PROT_TO_PAL(prot, flags);
    pal_prot_flags_t pal_prot_readable = pal_prot | PAL_PROT_READ;

    int ret;
    if (pal_prot != pal_prot_readable) {
        ret = DkVirtualMemoryProtect(addr, size, pal_prot_readable);
        if (ret < 0)
            return pal_to_unix_errno(ret);
    }

    size_t write_size = offset > (uint64_t)file_size ? 0 : MIN(size, (uint64_t)file_size - offset);
    char* write_addr = addr;
    file_off_t pos = offset;
    while (write_size > 0) {
        ssize_t count = hdl->fs->fs_ops->write(hdl, write_addr, write_size, &pos);
        if (count < 0) {
            if (count == -EINTR)
                continue;
            ret = count;
            goto out;
        }

        if (count == 0) {
            log_debug("%s: Failed to write back the whole mapping", __func__);
            ret = -EIO;
            goto out;
        }

        assert((size_t)count <= write_size);
        write_size -= count;
        write_addr += count;
    }

    ret = 0;

out:
    if (pal_prot != pal_prot_readable) {
        int protect_ret = DkVirtualMemoryProtect(addr, size, pal_prot);
        if (protect_ret < 0) {
            log_debug("%s: DkVirtualMemoryProtect failed on cleanup: %d", __func__, protect_ret);
            BUG();
        }
    }
    return ret;
}
