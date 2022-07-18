/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains operands to handle streams with URIs that start with "pipe:" or "pipe.srv:".
 */

#include <asm/fcntl.h>
#include <asm/ioctls.h>
#include <asm/poll.h>
#include <linux/time.h>
#include <linux/types.h>
#include <linux/un.h>
#include <sys/socket.h>

#include "api.h"
#include "linux_utils.h"
#include "pal.h"
#include "pal_error.h"
#include "pal_flags_conv.h"
#include "pal_internal.h"
#include "pal_linux.h"

/*!
 * \brief Create a listening abstract UNIX socket as preparation for connecting two ends of a pipe.
 *
 * An abstract UNIX socket with name "/gramine/<instance_id>/<pipename>" is opened for listening. A
 * corresponding PAL handle with type `pipesrv` is created. This PAL handle typically serves only as
 * an intermediate step to connect two ends of the pipe (`pipecli` and `pipe`). As soon as the other
 * end of the pipe connects to this listening socket, a new accepted socket and the corresponding
 * PAL handle are created, and this `pipesrv` handle can be closed.
 *
 * \param[out] handle  PAL handle of type `pipesrv` with abstract UNIX socket opened for listening.
 * \param[in]  name    String uniquely identifying the pipe.
 * \param[in]  options May contain PAL_OPTION_NONBLOCK.
 * \return             0 on success, negative PAL error code otherwise.
 */
static int pipe_listen(PAL_HANDLE* handle, const char* name, pal_stream_options_t options) {
    int ret;

    struct sockaddr_un addr;
    ret = get_gramine_unix_socket_addr(g_pal_common_state.instance_id, name, &addr);
    if (ret < 0)
        return -PAL_ERROR_DENIED;

    int nonblock = options & PAL_OPTION_NONBLOCK ? SOCK_NONBLOCK : 0;

    int fd = DO_SYSCALL(socket, AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | nonblock, 0);
    if (fd < 0)
        return unix_to_pal_error(fd);

    ret = DO_SYSCALL(bind, fd, &addr, sizeof(addr.sun_path) - 1);
    if (ret < 0) {
        DO_SYSCALL(close, fd);
        return unix_to_pal_error(ret);
    }

    ret = DO_SYSCALL(listen, fd, 1);
    if (ret < 0) {
        DO_SYSCALL(close, fd);
        return unix_to_pal_error(ret);
    }

    PAL_HANDLE hdl = calloc(1, HANDLE_SIZE(pipe));
    if (!hdl) {
        DO_SYSCALL(close, fd);
        return -PAL_ERROR_NOMEM;
    }

    init_handle_hdr(hdl, PAL_TYPE_PIPESRV);
    hdl->flags |= PAL_HANDLE_FD_READABLE;  /* cannot write to a listening socket */
    hdl->pipe.fd            = fd;
    hdl->pipe.nonblocking   = !!(options & PAL_OPTION_NONBLOCK);

    /* padding with zeros is for uniformity with other PALs (in particular, Linux-SGX) */
    memset(&hdl->pipe.name.str, 0, sizeof(hdl->pipe.name.str));
    memcpy(&hdl->pipe.name.str, name, strlen(name) + 1);

    *handle = hdl;
    return 0;
}

/*!
 * \brief Accept the other end of the pipe and create PAL handle for our end of the pipe.
 *
 * Caller creates a `pipesrv` PAL handle with the underlying abstract UNIX socket opened for
 * listening, and then calls this function to wait for the other end of the pipe to connect.
 * When the connection request arrives, a new `pipecli` PAL handle is created with the
 * corresponding underlying socket and is returned in `client`. This `pipecli` PAL handle denotes
 * our end of the pipe. Typically, `pipesrv` handle is not needed after this and can be closed.
 *
 * \param[in]  handle  PAL handle of type `pipesrv` with abstract UNIX socket opened for listening.
 * \param[out] client  PAL handle of type `pipecli` connected to the other end of the pipe (`pipe`).
 * \param      options flags to set on \p client handle.
 * \return             0 on success, negative PAL error code otherwise.
 */
static int pipe_waitforclient(PAL_HANDLE handle, PAL_HANDLE* client, pal_stream_options_t options) {
    if (HANDLE_HDR(handle)->type != PAL_TYPE_PIPESRV)
        return -PAL_ERROR_NOTSERVER;

    if (handle->pipe.fd == PAL_IDX_POISON)
        return -PAL_ERROR_DENIED;

    static_assert(O_CLOEXEC == SOCK_CLOEXEC && O_NONBLOCK == SOCK_NONBLOCK, "assumed below");
    int flags = PAL_OPTION_TO_LINUX_OPEN(options) | SOCK_CLOEXEC;
    int newfd = DO_SYSCALL(accept4, handle->pipe.fd, NULL, NULL, flags);
    if (newfd < 0)
        return unix_to_pal_error(newfd);

    PAL_HANDLE clnt = calloc(1, HANDLE_SIZE(pipe));
    if (!clnt) {
        DO_SYSCALL(close, newfd);
        return -PAL_ERROR_NOMEM;
    }

    init_handle_hdr(clnt, PAL_TYPE_PIPECLI);
    clnt->flags |= PAL_HANDLE_FD_READABLE | PAL_HANDLE_FD_WRITABLE;
    clnt->pipe.fd            = newfd;
    clnt->pipe.name          = handle->pipe.name;
    clnt->pipe.nonblocking   = !!(flags & SOCK_NONBLOCK);

    *client = clnt;
    return 0;
}

/*!
 * \brief Connect to the other end of the pipe and create PAL handle for our end of the pipe.
 *
 * This function connects to the other end of the pipe, represented as an abstract UNIX socket
 * "/gramine/<instance_id>/<pipename>" opened for listening. When the connection succeeds, a new
 * `pipe` PAL handle is created with the corresponding underlying socket and is returned in
 * `handle`. The other end of the pipe is typically of type `pipecli`.
 *
 * \param[out] handle  PAL handle of type `pipe` with abstract UNIX socket connected to another end.
 * \param[in]  name    String uniquely identifying the pipe.
 * \param[in]  options May contain PAL_OPTION_NONBLOCK.
 * \return             0 on success, negative PAL error code otherwise.
 */
static int pipe_connect(PAL_HANDLE* handle, const char* name, pal_stream_options_t options) {
    int ret;

    struct sockaddr_un addr;
    ret = get_gramine_unix_socket_addr(g_pal_common_state.instance_id, name, &addr);
    if (ret < 0)
        return -PAL_ERROR_DENIED;

    int nonblock = options & PAL_OPTION_NONBLOCK ? SOCK_NONBLOCK : 0;

    int fd = DO_SYSCALL(socket, AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | nonblock, 0);
    if (fd < 0)
        return -PAL_ERROR_DENIED;

    ret = DO_SYSCALL(connect, fd, &addr, sizeof(addr.sun_path) - 1);
    if (ret < 0) {
        DO_SYSCALL(close, fd);
        return unix_to_pal_error(ret);
    }

    PAL_HANDLE hdl = calloc(1, HANDLE_SIZE(pipe));
    if (!hdl) {
        DO_SYSCALL(close, fd);
        return -PAL_ERROR_NOMEM;
    }

    init_handle_hdr(hdl, PAL_TYPE_PIPE);
    hdl->flags |= PAL_HANDLE_FD_READABLE | PAL_HANDLE_FD_WRITABLE;
    hdl->pipe.fd            = fd;
    hdl->pipe.nonblocking   = !!(options & PAL_OPTION_NONBLOCK);

    /* padding with zeros is for uniformity with other PALs (in particular, Linux-SGX) */
    memset(&hdl->pipe.name.str, 0, sizeof(hdl->pipe.name.str));
    memcpy(&hdl->pipe.name.str, name, strlen(name) + 1);

    *handle = hdl;
    return 0;
}

/*!
 * \brief Create PAL handle of type `pipesrv` or `pipe` depending on `type` and `uri`.
 *
 * Depending on the combination of `type` and `uri`, the following PAL handles are created:
 *
 * - `type` is URI_TYPE_PIPE_SRV: create `pipesrv` handle (intermediate listening socket) with
 *                                the name created by `get_gramine_unix_socket_addr`. Caller is
 *                                expected to call pipe_waitforclient() afterwards.
 *
 * - `type` is URI_TYPE_PIPE: create `pipe` handle (connecting socket) with the name created by
 *                            `get_gramine_unix_socket_addr`.
 *
 * \param[out] handle  Created PAL handle of type `pipesrv` or `pipe`.
 * \param[in]  type    Can be URI_TYPE_PIPE or URI_TYPE_PIPE_SRV.
 * \param[in]  uri     Content is either NUL (for anonymous pipe) or a string with pipe name.
 * \param[in]  access  Not used.
 * \param[in]  share   Not used.
 * \param[in]  create  Not used.
 * \param[in]  options May contain PAL_OPTION_NONBLOCK.
 * \return             0 on success, negative PAL error code otherwise.
 */
static int pipe_open(PAL_HANDLE* handle, const char* type, const char* uri, enum pal_access access,
                     pal_share_flags_t share, enum pal_create_mode create,
                     pal_stream_options_t options) {
    __UNUSED(access);
    __UNUSED(create);
    assert(create == PAL_CREATE_IGNORED);

    if (!WITHIN_MASK(share, PAL_SHARE_MASK) || !WITHIN_MASK(options, PAL_OPTION_MASK))
        return -PAL_ERROR_INVAL;

    if (strlen(uri) + 1 > PIPE_NAME_MAX)
        return -PAL_ERROR_INVAL;

    if (!strcmp(type, URI_TYPE_PIPE_SRV))
        return pipe_listen(handle, uri, options);

    if (!strcmp(type, URI_TYPE_PIPE))
        return pipe_connect(handle, uri, options);

    return -PAL_ERROR_INVAL;
}

/*!
 * \brief Read from pipe.
 *
 * \param[in]  handle  PAL handle of type `pipecli` or `pipe`.
 * \param[in]  offset  Not used.
 * \param[in]  len     Size of user-supplied buffer.
 * \param[out] buffer  User-supplied buffer to read data to.
 * \return             Number of bytes read on success, negative PAL error code otherwise.
 */
static int64_t pipe_read(PAL_HANDLE handle, uint64_t offset, uint64_t len, void* buffer) {
    if (offset)
        return -PAL_ERROR_INVAL;

    if (HANDLE_HDR(handle)->type != PAL_TYPE_PIPECLI && HANDLE_HDR(handle)->type != PAL_TYPE_PIPE)
        return -PAL_ERROR_NOTCONNECTION;

    ssize_t bytes = DO_SYSCALL(read, handle->pipe.fd, buffer, len);
    if (bytes < 0)
        return unix_to_pal_error(bytes);

    return bytes;
}

/*!
 * \brief Write to pipe.
 *
 * \param[in] handle  PAL handle of type `pipecli` or `pipe`.
 * \param[in] offset  Not used.
 * \param[in] len     Size of user-supplied buffer.
 * \param[in] buffer  User-supplied buffer to write data from.
 * \return            Number of bytes written on success, negative PAL error code otherwise.
 */
static int64_t pipe_write(PAL_HANDLE handle, uint64_t offset, size_t len, const void* buffer) {
    if (offset)
        return -PAL_ERROR_INVAL;

    if (HANDLE_HDR(handle)->type != PAL_TYPE_PIPECLI && HANDLE_HDR(handle)->type != PAL_TYPE_PIPE)
        return -PAL_ERROR_NOTCONNECTION;

    ssize_t bytes = DO_SYSCALL(write, handle->pipe.fd, buffer, len);
    if (bytes < 0)
        return unix_to_pal_error(bytes);

    return bytes;
}

/*!
 * \brief Close pipe.
 *
 * \param[in] handle  PAL handle of type `pipesrv`, `pipecli`, or `pipe`.
 * \return            0 on success, negative PAL error code otherwise.
 */
static int pipe_close(PAL_HANDLE handle) {
    if (handle->pipe.fd != PAL_IDX_POISON) {
        DO_SYSCALL(close, handle->pipe.fd);
        handle->pipe.fd = PAL_IDX_POISON;
    }
    return 0;
}

/*!
 * \brief Shut down pipe.
 *
 * \param[in] handle       PAL handle of type `pipesrv`, `pipecli`, or `pipe`.
 * \param[in] delete_mode  See #pal_delete_mode.
 * \return                 0 on success, negative PAL error code otherwise.
 */
static int pipe_delete(PAL_HANDLE handle, enum pal_delete_mode delete_mode) {
    int shutdown;
    switch (delete_mode) {
        case PAL_DELETE_ALL:
            shutdown = SHUT_RDWR;
            break;
        case PAL_DELETE_READ:
            shutdown = SHUT_RD;
            break;
        case PAL_DELETE_WRITE:
            shutdown = SHUT_WR;
            break;
        default:
            return -PAL_ERROR_INVAL;
    }

    if (handle->pipe.fd != PAL_IDX_POISON) {
        DO_SYSCALL(shutdown, handle->pipe.fd, shutdown);
    }

    return 0;
}

/*!
 * \brief Retrieve attributes of PAL handle.
 *
 * \param[in]  handle  PAL handle of type `pipesrv`, `pipecli`, or `pipe`.
 * \param[out] attr    User-supplied buffer to store handle's current attributes.
 * \return             0 on success, negative PAL error code otherwise.
 */
static int pipe_attrquerybyhdl(PAL_HANDLE handle, PAL_STREAM_ATTR* attr) {
    int ret;

    if (handle->pipe.fd == PAL_IDX_POISON)
        return -PAL_ERROR_BADHANDLE;

    attr->handle_type  = HANDLE_HDR(handle)->type;
    attr->nonblocking  = handle->pipe.nonblocking;

    /* get number of bytes available for reading (doesn't make sense for "listening" pipes) */
    attr->pending_size = 0;
    if (HANDLE_HDR(handle)->type != PAL_TYPE_PIPESRV) {
        int val;
        ret = DO_SYSCALL(ioctl, handle->pipe.fd, FIONREAD, &val);
        if (ret < 0)
            return unix_to_pal_error(ret);

        attr->pending_size = val;
    }

    return 0;
}

/*!
 * \brief Set attributes of PAL handle.
 *
 * Currently only `nonblocking` attribute can be set.
 *
 * \param[in] handle  PAL handle of type `pipesrv`, `pipecli`, or `pipe`.
 * \param[in] attr    User-supplied buffer with new handle's attributes.
 * \return            0 on success, negative PAL error code otherwise.
 */
static int pipe_attrsetbyhdl(PAL_HANDLE handle, PAL_STREAM_ATTR* attr) {
    if (handle->pipe.fd == PAL_IDX_POISON)
        return -PAL_ERROR_BADHANDLE;

    bool* nonblocking = &handle->pipe.nonblocking;

    if (attr->nonblocking != *nonblocking) {
        int ret = DO_SYSCALL(fcntl, handle->pipe.fd, F_SETFL, attr->nonblocking ? O_NONBLOCK : 0);
        if (ret < 0)
            return unix_to_pal_error(ret);

        *nonblocking = attr->nonblocking;
    }

    return 0;
}

/*!
 * \brief Retrieve full URI of PAL handle.
 *
 * Full URI is composed of the type and pipe name: "<type>:<pipename>".
 *
 * \param[in]  handle  PAL handle of type `pipesrv`, `pipecli`, or `pipe`.
 * \param[out] buffer  User-supplied buffer to write URI to.
 * \param[in]  count   Size of the user-supplied buffer.
 * \return             Number of bytes written on success, negative PAL error code otherwise.
 */
static int pipe_getname(PAL_HANDLE handle, char* buffer, size_t count) {
    size_t old_count = count;
    int ret;

    const char* prefix = NULL;
    size_t prefix_len  = 0;

    switch (PAL_GET_TYPE(handle)) {
        case PAL_TYPE_PIPESRV:
        case PAL_TYPE_PIPECLI:
            prefix_len = static_strlen(URI_TYPE_PIPE_SRV);
            prefix     = URI_TYPE_PIPE_SRV;
            break;
        case PAL_TYPE_PIPE:
            prefix_len = static_strlen(URI_TYPE_PIPE);
            prefix     = URI_TYPE_PIPE;
            break;
        default:
            return -PAL_ERROR_INVAL;
    }

    if (prefix_len >= count)
        return -PAL_ERROR_OVERFLOW;

    memcpy(buffer, prefix, prefix_len);
    buffer[prefix_len] = ':';
    buffer += prefix_len + 1;
    count -= prefix_len + 1;

    ret = snprintf(buffer, count, "%s\n", handle->pipe.name.str);
    if (buffer[ret - 1] != '\n') {
        memset(buffer, 0, count);
        return -PAL_ERROR_OVERFLOW;
    }

    buffer[ret - 1] = 0;
    buffer += ret - 1;
    count -= ret - 1;

    return old_count - count;
}

struct handle_ops g_pipe_ops = {
    .getname        = &pipe_getname,
    .open           = &pipe_open,
    .waitforclient  = &pipe_waitforclient,
    .read           = &pipe_read,
    .write          = &pipe_write,
    .close          = &pipe_close,
    .delete         = &pipe_delete,
    .attrquerybyhdl = &pipe_attrquerybyhdl,
    .attrsetbyhdl   = &pipe_attrsetbyhdl,
};
