/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains operands to handle streams with URIs that start with "file:" or "dir:".
 */

#include "api.h"
#include "pal.h"
#include "pal_error.h"
#include "pal_flags_conv.h"
#include "pal_internal.h"
#include "pal_linux.h"
#include "pal_linux_error.h"
#include "stat.h"

/* 'open' operation for file streams */
static int file_open(PAL_HANDLE* handle, const char* type, const char* uri, enum pal_access access,
                     pal_share_flags_t share, enum pal_create_mode create,
                     pal_stream_options_t options) {
    if (strcmp(type, URI_TYPE_FILE))
        return -PAL_ERROR_INVAL;

    assert(WITHIN_MASK(share,   PAL_SHARE_MASK));
    assert(WITHIN_MASK(options, PAL_OPTION_MASK));

    /* try to do the real open */
    // FIXME: No idea why someone hardcoded O_CLOEXEC here. We should drop it and carefully
    // investigate if this causes any descriptor leaks.
    int ret = DO_SYSCALL(open, uri, PAL_ACCESS_TO_LINUX_OPEN(access)  |
                                    PAL_CREATE_TO_LINUX_OPEN(create)  |
                                    PAL_OPTION_TO_LINUX_OPEN(options) |
                                    O_CLOEXEC,
                         share);

    if (ret < 0)
        return unix_to_pal_error(ret);

    /* if try_create_path succeeded, prepare for the file handle */
    size_t uri_size = strlen(uri) + 1;
    PAL_HANDLE hdl = calloc(1, HANDLE_SIZE(file) + uri_size);
    if (!hdl) {
        DO_SYSCALL(close, ret);
        return -PAL_ERROR_NOMEM;
    }

    init_handle_hdr(hdl, PAL_TYPE_FILE);
    hdl->flags |= PAL_HANDLE_FD_READABLE | PAL_HANDLE_FD_WRITABLE;
    hdl->file.fd = ret;
    hdl->file.map_start = NULL;
    char* path = (void*)hdl + HANDLE_SIZE(file);
    ret = get_norm_path(uri, path, &uri_size);
    if (ret < 0) {
        DO_SYSCALL(close, hdl->file.fd);
        free(hdl);
        return ret;
    }

    hdl->file.realpath = path;

    struct stat st;
    ret = DO_SYSCALL(fstat, hdl->file.fd, &st);
    if (ret < 0) {
        DO_SYSCALL(close, hdl->file.fd);
        free(hdl);
        return unix_to_pal_error(ret);
    }

    hdl->file.seekable = !S_ISFIFO(st.st_mode);

    *handle = hdl;
    return 0;
}

/* 'read' operation for file streams. */
static int64_t file_read(PAL_HANDLE handle, uint64_t offset, uint64_t count, void* buffer) {
    int fd = handle->file.fd;
    int64_t ret;

    if (handle->file.seekable) {
        ret = DO_SYSCALL(pread64, fd, buffer, count, offset);
    } else {
        ret = DO_SYSCALL(read, fd, buffer, count);
    }

    if (ret < 0)
        return unix_to_pal_error(ret);

    return ret;
}

/* 'write' operation for file streams. */
static int64_t file_write(PAL_HANDLE handle, uint64_t offset, uint64_t count, const void* buffer) {
    int fd = handle->file.fd;
    int64_t ret;

    if (handle->file.seekable) {
        ret = DO_SYSCALL(pwrite64, fd, buffer, count, offset);
    } else {
        ret = DO_SYSCALL(write, fd, buffer, count);
    }

    if (ret < 0)
        return unix_to_pal_error(ret);

    return ret;
}

/* 'close' operation for file streams */
static int file_close(PAL_HANDLE handle) {
    int fd = handle->file.fd;

    int ret = DO_SYSCALL(close, fd);

    /* initial realpath is part of handle object and will be freed with it */
    if (handle->file.realpath && handle->file.realpath != (void*)handle + HANDLE_SIZE(file)) {
        free((void*)handle->file.realpath);
    }

    return ret < 0 ? unix_to_pal_error(ret) : 0;
}

/* 'delete' operation for file streams */
static int file_delete(PAL_HANDLE handle, enum pal_delete_mode delete_mode) {
    if (delete_mode != PAL_DELETE_ALL)
        return -PAL_ERROR_INVAL;

    DO_SYSCALL(unlink, handle->file.realpath);
    return 0;
}

/* 'map' operation for file stream. */
static int file_map(PAL_HANDLE handle, void** addr, pal_prot_flags_t prot, uint64_t offset,
                    uint64_t size) {
    int fd = handle->file.fd;
    void* mem = *addr;
    /*
     * work around for fork emulation
     * the first exec image to be loaded has to be at same address
     * as parent.
     */
    if (mem == NULL && handle->file.map_start != NULL) {
        mem = handle->file.map_start;
        /* this address is used. don't over-map it later */
        handle->file.map_start = NULL;
    }
    int flags = PAL_MEM_FLAGS_TO_LINUX(0, prot) | (mem ? MAP_FIXED : 0);
    int linux_prot = PAL_PROT_TO_LINUX(prot);

    /* The memory will always be allocated with flag MAP_PRIVATE. */
    // TODO: except it will not since `assert(flags & MAP_PRIVATE)` fails on LTP
    mem = (void*)DO_SYSCALL(mmap, mem, size, linux_prot, flags, fd, offset);

    if (IS_PTR_ERR(mem))
        return unix_to_pal_error(PTR_TO_ERR(mem));

    *addr = mem;
    return 0;
}

/* 'setlength' operation for file stream. */
static int64_t file_setlength(PAL_HANDLE handle, uint64_t length) {
    int ret = DO_SYSCALL(ftruncate, handle->file.fd, length);

    if (ret < 0)
        return (ret == -EINVAL || ret == -EBADF) ? -PAL_ERROR_BADHANDLE
                                                 : -PAL_ERROR_DENIED;

    return (int64_t)length;
}

/* 'flush' operation for file stream. */
static int file_flush(PAL_HANDLE handle) {
    int ret = DO_SYSCALL(fsync, handle->file.fd);

    if (ret < 0)
        return (ret == -EINVAL || ret == -EBADF) ? -PAL_ERROR_BADHANDLE
                                                 : -PAL_ERROR_DENIED;

    return 0;
}

static inline int file_stat_type(struct stat* stat) {
    if (S_ISREG(stat->st_mode))
        return PAL_TYPE_FILE;
    if (S_ISDIR(stat->st_mode))
        return PAL_TYPE_DIR;
    if (S_ISCHR(stat->st_mode))
        return PAL_TYPE_DEV;
    if (S_ISFIFO(stat->st_mode))
        return PAL_TYPE_PIPE;
    if (S_ISSOCK(stat->st_mode))
        return PAL_TYPE_DEV;

    return 0;
}

/* copy attr content from POSIX stat struct to PAL_STREAM_ATTR */
static inline void file_attrcopy(PAL_STREAM_ATTR* attr, struct stat* stat) {
    attr->handle_type  = file_stat_type(stat);
    attr->nonblocking  = false;
    attr->share_flags  = stat->st_mode & PAL_SHARE_MASK;
    attr->pending_size = stat->st_size;
}

/* 'attrquery' operation for file streams */
static int file_attrquery(const char* type, const char* uri, PAL_STREAM_ATTR* attr) {
    if (strcmp(type, URI_TYPE_FILE) && strcmp(type, URI_TYPE_DIR))
        return -PAL_ERROR_INVAL;

    struct stat stat_buf;
    /* try to do the real open */
    int ret = DO_SYSCALL(stat, uri, &stat_buf);

    /* if it failed, return the right error code */
    if (ret < 0)
        return unix_to_pal_error(ret);

    file_attrcopy(attr, &stat_buf);
    return 0;
}

/* 'attrquerybyhdl' operation for file streams */
static int file_attrquerybyhdl(PAL_HANDLE handle, PAL_STREAM_ATTR* attr) {
    int fd = handle->file.fd;
    struct stat stat_buf;

    int ret = DO_SYSCALL(fstat, fd, &stat_buf);

    if (ret < 0)
        return unix_to_pal_error(ret);

    file_attrcopy(attr, &stat_buf);
    return 0;
}

static int file_attrsetbyhdl(PAL_HANDLE handle, PAL_STREAM_ATTR* attr) {
    int ret;
    int fd = handle->file.fd;

    ret = DO_SYSCALL(fchmod, fd, attr->share_flags);
    if (ret < 0)
        return unix_to_pal_error(ret);

    return 0;
}

static int file_rename(PAL_HANDLE handle, const char* type, const char* uri) {
    if (strcmp(type, URI_TYPE_FILE))
        return -PAL_ERROR_INVAL;

    char* tmp = strdup(uri);
    if (!tmp)
        return -PAL_ERROR_NOMEM;

    int ret = DO_SYSCALL(rename, handle->file.realpath, uri);
    if (ret < 0) {
        free(tmp);
        return unix_to_pal_error(ret);
    }

    /* initial realpath is part of handle object and will be freed with it */
    if (handle->file.realpath && handle->file.realpath != (void*)handle + HANDLE_SIZE(file)) {
        free((void*)handle->file.realpath);
    }

    handle->file.realpath = tmp;
    return 0;
}

static int file_getname(PAL_HANDLE handle, char* buffer, size_t count) {
    if (!handle->file.realpath)
        return 0;

    size_t len = strlen(handle->file.realpath);
    char* tmp = strcpy_static(buffer, URI_PREFIX_FILE, count);

    if (!tmp || buffer + count < tmp + len + 1)
        return -PAL_ERROR_TOOLONG;

    memcpy(tmp, handle->file.realpath, len + 1);
    return tmp + len - buffer;
}

static const char* file_getrealpath(PAL_HANDLE handle) {
    return handle->file.realpath;
}

struct handle_ops g_file_ops = {
    .getname        = &file_getname,
    .getrealpath    = &file_getrealpath,
    .open           = &file_open,
    .read           = &file_read,
    .write          = &file_write,
    .close          = &file_close,
    .delete         = &file_delete,
    .map            = &file_map,
    .setlength      = &file_setlength,
    .flush          = &file_flush,
    .attrquery      = &file_attrquery,
    .attrquerybyhdl = &file_attrquerybyhdl,
    .attrsetbyhdl   = &file_attrsetbyhdl,
    .rename         = &file_rename,
};

/* 'open' operation for directory stream. Directory stream does not have a
   specific type prefix, its URI looks the same file streams, plus it
   ended with slashes. dir_open will be called by file_open. */
static int dir_open(PAL_HANDLE* handle, const char* type, const char* uri, enum pal_access access,
                    pal_share_flags_t share, enum pal_create_mode create,
                    pal_stream_options_t options) {
    __UNUSED(access);
    assert(create != PAL_CREATE_IGNORED);
    if (strcmp(type, URI_TYPE_DIR))
        return -PAL_ERROR_INVAL;

    if (create == PAL_CREATE_TRY || create == PAL_CREATE_ALWAYS) {
        int ret = DO_SYSCALL(mkdir, uri, share);

        if (ret < 0) {
            if (ret == -EEXIST && create == PAL_CREATE_ALWAYS)
                return -PAL_ERROR_STREAMEXIST;
            if (ret != -EEXIST)
                return unix_to_pal_error(ret);
            assert(ret == -EEXIST && create == PAL_CREATE_TRY);
        }
    }

    int fd = DO_SYSCALL(open, uri, O_DIRECTORY | PAL_OPTION_TO_LINUX_OPEN(options) | O_CLOEXEC, 0);
    if (fd < 0)
        return unix_to_pal_error(fd);

    size_t len = strlen(uri);
    PAL_HANDLE hdl = calloc(1, HANDLE_SIZE(dir) + len + 1);
    if (!hdl) {
        DO_SYSCALL(close, fd);
        return -PAL_ERROR_NOMEM;
    }

    init_handle_hdr(hdl, PAL_TYPE_DIR);

    hdl->flags |= PAL_HANDLE_FD_READABLE;
    hdl->dir.fd = fd;

    char* path = (void*)hdl + HANDLE_SIZE(dir);
    memcpy(path, uri, len + 1);

    hdl->dir.realpath    = path;
    hdl->dir.buf         = NULL;
    hdl->dir.ptr         = NULL;
    hdl->dir.end         = NULL;
    hdl->dir.endofstream = false;

    *handle = hdl;
    return 0;
}

static inline bool is_dot_or_dotdot(const char* name) {
    return (name[0] == '.' && !name[1]) || (name[0] == '.' && name[1] == '.' && !name[2]);
}

/* 'read' operation for directory stream. Directory stream will not
   need a 'write' operation. */
static int64_t dir_read(PAL_HANDLE handle, uint64_t offset, size_t count, void* _buf) {
    size_t bytes_written = 0;
    char* buf = (char*)_buf;

    if (offset) {
        return -PAL_ERROR_INVAL;
    }

    if (handle->dir.endofstream) {
        return 0;
    }

    while (1) {
        while ((char*)handle->dir.ptr < (char*)handle->dir.end) {
            struct linux_dirent64* dirent = (struct linux_dirent64*)handle->dir.ptr;

            if (is_dot_or_dotdot(dirent->d_name)) {
                goto skip;
            }

            bool is_dir = dirent->d_type == DT_DIR;
            size_t len = strlen(dirent->d_name);

            if (len + 1 + (is_dir ? 1 : 0) > count) {
                goto out;
            }

            memcpy(buf, dirent->d_name, len);
            if (is_dir) {
                buf[len++] = '/';
            }
            buf[len++] = '\0';

            buf += len;
            bytes_written += len;
            count -= len;
        skip:
            handle->dir.ptr = (char*)handle->dir.ptr + dirent->d_reclen;
        }

        if (!count) {
            /* No space left, returning */
            goto out;
        }

        if (!handle->dir.buf) {
            handle->dir.buf = malloc(DIRBUF_SIZE);
            if (!handle->dir.buf) {
                return -PAL_ERROR_NOMEM;
            }
        }

        int size = DO_SYSCALL(getdents64, handle->dir.fd, handle->dir.buf, DIRBUF_SIZE);
        if (size < 0) {
            /* If something was written just return that and pretend
             * no error was seen - it will be caught next time. */
            if (bytes_written) {
                return bytes_written;
            }
            return unix_to_pal_error(size);
        }

        if (!size) {
            handle->dir.endofstream = true;
            goto out;
        }

        handle->dir.ptr = handle->dir.buf;
        handle->dir.end = (char*)handle->dir.buf + size;
    }

out:
    return (int64_t)bytes_written;
}

/* 'close' operation of directory streams */
static int dir_close(PAL_HANDLE handle) {
    int fd = handle->dir.fd;

    int ret = DO_SYSCALL(close, fd);

    if (handle->dir.buf) {
        free(handle->dir.buf);
        handle->dir.buf = handle->dir.ptr = handle->dir.end = NULL;
    }

    /* initial realpath is part of handle object and will be freed with it */
    if (handle->dir.realpath && handle->dir.realpath != (void*)handle + HANDLE_SIZE(dir)) {
        free((void*)handle->dir.realpath);
    }

    if (ret < 0)
        return -PAL_ERROR_BADHANDLE;

    return 0;
}

/* 'delete' operation of directory streams */
static int dir_delete(PAL_HANDLE handle, enum pal_delete_mode delete_mode) {
    if (delete_mode != PAL_DELETE_ALL)
        return -PAL_ERROR_INVAL;

    int ret = dir_close(handle);

    if (ret < 0)
        return ret;

    ret = DO_SYSCALL(rmdir, handle->dir.realpath);

    return (ret < 0 && ret != -ENOENT) ? -PAL_ERROR_DENIED : 0;
}

static int dir_rename(PAL_HANDLE handle, const char* type, const char* uri) {
    if (strcmp(type, URI_TYPE_DIR))
        return -PAL_ERROR_INVAL;

    char* tmp = strdup(uri);
    if (!tmp)
        return -PAL_ERROR_NOMEM;

    int ret = DO_SYSCALL(rename, handle->dir.realpath, uri);
    if (ret < 0) {
        free(tmp);
        return unix_to_pal_error(ret);
    }

    /* initial realpath is part of handle object and will be freed with it */
    if (handle->dir.realpath && handle->dir.realpath != (void*)handle + HANDLE_SIZE(dir)) {
        free((void*)handle->dir.realpath);
    }

    handle->dir.realpath = tmp;
    return 0;
}

static int dir_getname(PAL_HANDLE handle, char* buffer, size_t count) {
    if (!handle->dir.realpath)
        return 0;

    size_t len = strlen(handle->dir.realpath);
    char* tmp = strcpy_static(buffer, URI_PREFIX_DIR, count);

    if (!tmp || buffer + count < tmp + len + 1)
        return -PAL_ERROR_TOOLONG;

    memcpy(tmp, handle->dir.realpath, len + 1);
    return tmp + len - buffer;
}

static const char* dir_getrealpath(PAL_HANDLE handle) {
    return handle->dir.realpath;
}

struct handle_ops g_dir_ops = {
    .getname        = &dir_getname,
    .getrealpath    = &dir_getrealpath,
    .open           = &dir_open,
    .read           = &dir_read,
    .close          = &dir_close,
    .delete         = &dir_delete,
    .attrquery      = &file_attrquery,
    .attrquerybyhdl = &file_attrquerybyhdl,
    .attrsetbyhdl   = &file_attrsetbyhdl,
    .rename         = &dir_rename,
};
