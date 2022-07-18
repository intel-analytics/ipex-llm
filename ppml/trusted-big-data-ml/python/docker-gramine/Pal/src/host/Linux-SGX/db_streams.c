/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains APIs to open, read, write and get attribute of streams.
 */

#include <asm/fcntl.h>
#include <asm/socket.h>
#include <asm/stat.h>
#include <linux/in.h>
#include <linux/in6.h>
#include <linux/types.h>
#include <stdalign.h>
#include <stdbool.h>

#include "api.h"
#include "crypto.h"
#include "enclave_pages.h"
#include "pal.h"
#include "pal_error.h"
#include "pal_internal.h"
#include "pal_linux.h"
#include "pal_linux_error.h"
#include "perm.h"

#define DUMMYPAYLOAD     "dummypayload"
#define DUMMYPAYLOADSIZE (sizeof(DUMMYPAYLOAD))

static int g_log_fd = PAL_LOG_DEFAULT_FD;

struct hdl_header {
    bool has_fd;       /* true if PAL handle has a corresponding host file descriptor */
    size_t  data_size; /* total size of serialized PAL handle */
};

static size_t addr_size(const struct sockaddr* addr) {
    switch (addr->sa_family) {
        case AF_INET:
            return sizeof(struct sockaddr_in);
        case AF_INET6:
            return sizeof(struct sockaddr_in6);
        default:
            return 0;
    }
}

/* _DkStreamUnmap for internal use. Unmap stream at certain memory address.
   The memory is unmapped as a whole.*/
int _DkStreamUnmap(void* addr, uint64_t size) {
    return free_enclave_pages(addr, size);
}

static ssize_t handle_serialize(PAL_HANDLE handle, void** data) {
    int ret;
    const void* d1;
    const void* d2;
    size_t dsz1 = 0;
    size_t dsz2 = 0;
    bool free_d1 = false;

    /* find fields to serialize (depends on the handle type) and assign them to d1/d2; note that
     * no handle type has more than two such fields, and some have none at all */
    switch (PAL_GET_TYPE(handle)) {
        case PAL_TYPE_FILE:
            d1   = handle->file.realpath;
            dsz1 = strlen(handle->file.realpath) + 1;
            break;
        case PAL_TYPE_PIPE:
        case PAL_TYPE_PIPECLI:
            /* session key is part of handle but need to serialize SSL context */
            if (handle->pipe.ssl_ctx) {
                free_d1 = true;
                ret = _DkStreamSecureSave(handle->pipe.ssl_ctx, (const uint8_t**)&d1, &dsz1);
                if (ret < 0)
                    return -PAL_ERROR_DENIED;
            }
            break;
        case PAL_TYPE_PIPESRV:
            break;
        case PAL_TYPE_DEV:
            /* devices have no fields to serialize */
            break;
        case PAL_TYPE_DIR:
            if (handle->dir.realpath) {
                d1   = handle->dir.realpath;
                dsz1 = strlen(handle->dir.realpath) + 1;
            }
            break;
        case PAL_TYPE_TCP:
        case PAL_TYPE_TCPSRV:
        case PAL_TYPE_UDP:
        case PAL_TYPE_UDPSRV:
            if (handle->sock.bind) {
                d1   = (const void*)handle->sock.bind;
                dsz1 = addr_size(handle->sock.bind);
            }
            if (handle->sock.conn) {
                d2   = (const void*)handle->sock.conn;
                dsz2 = addr_size(handle->sock.conn);
            }
            break;
        case PAL_TYPE_PROCESS:
            /* session key is part of handle but need to serialize SSL context */
            if (handle->process.ssl_ctx) {
                free_d1 = true;
                ret = _DkStreamSecureSave(handle->process.ssl_ctx, (const uint8_t**)&d1, &dsz1);
                if (ret < 0)
                    return -PAL_ERROR_DENIED;
            }
            break;
        case PAL_TYPE_EVENTFD:
            break;
        default:
            return -PAL_ERROR_INVAL;
    }

    size_t hdlsz = handle_size(handle);
    void* buffer = malloc(hdlsz + dsz1 + dsz2);
    if (!buffer) {
        ret = -PAL_ERROR_NOMEM;
        goto out;
    }

    /* copy into buffer all handle fields and then serialized fields */
    memcpy(buffer, handle, hdlsz);
    if (dsz1)
        memcpy(buffer + hdlsz, d1, dsz1);
    if (dsz2)
        memcpy(buffer + hdlsz + dsz1, d2, dsz2);

    ret = 0;
out:
    if (free_d1)
        free((void*)d1);
    if (ret < 0)
        return ret;

    *data = buffer;
    return hdlsz + dsz1 + dsz2;
}

static int handle_deserialize(PAL_HANDLE* handle, const void* data, size_t size, int host_fd) {
    int ret;

    PAL_HANDLE hdl = malloc(size);
    if (!hdl)
        return -PAL_ERROR_NOMEM;

    memcpy(hdl, data, size);
    size_t hdlsz = handle_size(hdl);

    /* update handle fields to point to correct contents (located right after handle itself) */
    switch (PAL_GET_TYPE(hdl)) {
        case PAL_TYPE_FILE:
            hdl->file.realpath = hdl->file.realpath ? (const char*)hdl + hdlsz : NULL;
            hdl->file.chunk_hashes = NULL;
            break;
        case PAL_TYPE_PIPE:
        case PAL_TYPE_PIPECLI:
            /* session key is part of handle but need to deserialize SSL context */
            hdl->pipe.fd = host_fd; /* correct host FD must be passed to SSL context */
            ret = _DkStreamSecureInit(hdl, hdl->pipe.is_server, &hdl->pipe.session_key,
                                      (LIB_SSL_CONTEXT**)&hdl->pipe.ssl_ctx,
                                      (const uint8_t*)hdl + hdlsz, size - hdlsz);
            if (ret < 0) {
                free(hdl);
                return -PAL_ERROR_DENIED;
            }
            break;
        case PAL_TYPE_PIPESRV:
            break;
        case PAL_TYPE_DEV:
            break;
        case PAL_TYPE_DIR:
            hdl->dir.realpath = hdl->dir.realpath ? (const char*)hdl + hdlsz : NULL;
            break;
        case PAL_TYPE_TCP:
        case PAL_TYPE_TCPSRV:
        case PAL_TYPE_UDP:
        case PAL_TYPE_UDPSRV: {
            size_t s1 = hdl->sock.bind ? addr_size((struct sockaddr*)((uint8_t*)hdl + hdlsz)) : 0;
            size_t s2 = hdl->sock.conn ? addr_size((struct sockaddr*)((uint8_t*)hdl + hdlsz + s1)) : 0;
            if (s1)
                hdl->sock.bind = (struct sockaddr*)((uint8_t*)hdl + hdlsz);
            if (s2)
                hdl->sock.conn = (struct sockaddr*)((uint8_t*)hdl + hdlsz + s2);
            break;
        }
        case PAL_TYPE_PROCESS:
            /* session key is part of handle but need to deserialize SSL context */
            hdl->process.stream = host_fd; /* correct host FD must be passed to SSL context */
            ret = _DkStreamSecureInit(hdl, hdl->process.is_server, &hdl->process.session_key,
                                      (LIB_SSL_CONTEXT**)&hdl->process.ssl_ctx,
                                      (const uint8_t*)hdl + hdlsz, size - hdlsz);
            if (ret < 0) {
                free(hdl);
                return -PAL_ERROR_DENIED;
            }
            break;
        case PAL_TYPE_EVENTFD:
            break;
        default:
            free(hdl);
            return -PAL_ERROR_BADHANDLE;
    }

    *handle = hdl;
    return 0;
}

int _DkSendHandle(PAL_HANDLE target_process, PAL_HANDLE cargo) {
    if (HANDLE_HDR(target_process)->type != PAL_TYPE_PROCESS)
        return -PAL_ERROR_BADHANDLE;

    /* serialize cargo handle into a blob hdl_data */
    void* hdl_data = NULL;
    ssize_t hdl_data_size = handle_serialize(cargo, &hdl_data);
    if (hdl_data_size < 0)
        return hdl_data_size;

    ssize_t ret;
    struct hdl_header hdl_hdr = {
        .has_fd = cargo->flags & (PAL_HANDLE_FD_READABLE | PAL_HANDLE_FD_WRITABLE),
        .data_size = hdl_data_size
    };
    int fd = target_process->process.stream;

    /* first send hdl_hdr so the recipient knows how many FDs were transferred + how large is cargo */
    ret = ocall_send(fd, &hdl_hdr, sizeof(struct hdl_header), NULL, 0, NULL, 0);
    if (ret < 0) {
        free(hdl_data);
        return unix_to_pal_error(ret);
    }

    /* construct ancillary data with FD-to-transfer in a control message */
    alignas(struct cmsghdr) char control_buf[CMSG_SPACE(sizeof(int))] = { 0 };

    struct cmsghdr* control_hdr = (struct cmsghdr*)control_buf;
    control_hdr->cmsg_level     = SOL_SOCKET;
    control_hdr->cmsg_type      = SCM_RIGHTS;
    if (hdl_hdr.has_fd) {
        /* XXX: change to `SAME_TYPE` once `PAL_HANDLE` uses `int` to store fds */
        static_assert(sizeof(cargo->generic.fd) == sizeof(int), "required");
        control_hdr->cmsg_len = CMSG_LEN(sizeof(int));
        memcpy(CMSG_DATA(control_hdr), &cargo->generic.fd, sizeof(int));
    } else {
        control_hdr->cmsg_len = CMSG_LEN(0);
    }

    /* next send FD-to-transfer as ancillary data */
    ret = ocall_send(fd, DUMMYPAYLOAD, DUMMYPAYLOADSIZE, NULL, 0, control_hdr,
                     control_hdr->cmsg_len);
    if (ret < 0) {
        free(hdl_data);
        return unix_to_pal_error(ret);
    }

    /* finally send the serialized cargo as payload (possibly encrypted) */
    if (target_process->process.ssl_ctx) {
        ret = _DkStreamSecureWrite(target_process->process.ssl_ctx, (uint8_t*)hdl_data,
                                   hdl_hdr.data_size,
                                   /*is_blocking=*/!target_process->process.nonblocking);
    } else {
        ret = ocall_write(fd, hdl_data, hdl_hdr.data_size);
        ret = ret < 0 ? unix_to_pal_error(ret) : ret;
    }

    free(hdl_data);
    return ret < 0 ? ret : 0;
}

int _DkReceiveHandle(PAL_HANDLE source_process, PAL_HANDLE* out_cargo) {
    if (HANDLE_HDR(source_process)->type != PAL_TYPE_PROCESS)
        return -PAL_ERROR_BADHANDLE;

    ssize_t ret;
    struct hdl_header hdl_hdr;
    int fd = source_process->process.stream;

    /* first receive hdl_hdr so that we know how many FDs were transferred + how large is cargo */
    ret = ocall_recv(fd, &hdl_hdr, sizeof(hdl_hdr), NULL, NULL, NULL, NULL);
    if (ret < 0)
        return unix_to_pal_error(ret);

    if ((size_t)ret != sizeof(hdl_hdr)) {
        /* This check is to shield from a Iago attack. We know that ocall_send() in _DkSendHandle()
         * transfers the message atomically, and that our ocall_recv() receives it atomically. So
         * the only valid values for ret must be zero or the size of the header. */
        if (!ret)
            return -PAL_ERROR_TRYAGAIN;
        return -PAL_ERROR_DENIED;
    }

    alignas(struct cmsghdr) char control_buf[CMSG_SPACE(sizeof(int))] = { 0 };
    size_t control_buf_size = sizeof(control_buf);

    /* next receive FDs-to-transfer as ancillary data */
    char dummypayload[DUMMYPAYLOADSIZE];
    ret = ocall_recv(fd, dummypayload, DUMMYPAYLOADSIZE, NULL, NULL, control_buf,
                     &control_buf_size);
    if (ret < 0)
        return unix_to_pal_error(ret);
    if (control_buf_size < sizeof(struct cmsghdr)) {
        return -PAL_ERROR_DENIED;
    }

    /* finally receive the serialized cargo as payload (possibly encrypted) */
    char hdl_data[hdl_hdr.data_size];

    if (source_process->process.ssl_ctx) {
        ret = _DkStreamSecureRead(source_process->process.ssl_ctx,
                                  (uint8_t*)hdl_data, hdl_hdr.data_size,
                                  /*is_blocking=*/!source_process->process.nonblocking);
    } else {
        ret = ocall_read(fd, hdl_data, hdl_hdr.data_size);
        ret = ret < 0 ? unix_to_pal_error(ret) : ret;
    }
    if (ret < 0)
        return ret;

    struct cmsghdr* control_hdr = (struct cmsghdr*)control_buf;
    if (control_hdr->cmsg_type != SCM_RIGHTS)
        return -PAL_ERROR_DENIED;
    if (hdl_hdr.has_fd && control_hdr->cmsg_len != CMSG_LEN(sizeof(int))) {
        return -PAL_ERROR_DENIED;
    }

    int host_fd = -1;
    if (hdl_hdr.has_fd) {
        memcpy(&host_fd, CMSG_DATA(control_hdr), sizeof(int));
    }

    /* deserialize cargo handle from a blob hdl_data */
    PAL_HANDLE handle = NULL;
    ret = handle_deserialize(&handle, hdl_data, hdl_hdr.data_size, host_fd);
    if (ret < 0)
        return ret;

    /* restore cargo handle's FD from the received FD-to-transfer */
    if (hdl_hdr.has_fd) {
        handle->generic.fd = host_fd;
    } else {
        handle->flags &= ~(PAL_HANDLE_FD_READABLE | PAL_HANDLE_FD_WRITABLE);
    }

    *out_cargo = handle;
    return 0;
}

int _DkInitDebugStream(const char* path) {
    int ret;

    if (g_log_fd != PAL_LOG_DEFAULT_FD) {
        ret = ocall_close(g_log_fd);
        g_log_fd = PAL_LOG_DEFAULT_FD;
        if (ret < 0)
            return unix_to_pal_error(ret);
    }

    ret = ocall_open(path, O_WRONLY | O_APPEND | O_CREAT, PERM_rw_______);
    if (ret < 0)
        return unix_to_pal_error(ret);
    g_log_fd = ret;
    return 0;
}

int _DkDebugLog(const void* buf, size_t size) {
    if (g_log_fd < 0)
        return -PAL_ERROR_BADHANDLE;

    // TODO: add retrying on EINTR
    ssize_t ret = ocall_write(g_log_fd, buf, size);
    if (ret < 0 || (size_t)ret != size) {
        return ret < 0 ? unix_to_pal_error(ret) : -PAL_ERROR_INTERRUPTED;
    }
    return 0;
}
