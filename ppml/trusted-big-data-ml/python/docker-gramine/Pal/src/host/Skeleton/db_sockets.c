/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains operands for streams with URIs that start with "tcp:", "tcp.srv:", "udp:",
 * "udp.srv:".
 */

#include "api.h"
#include "pal.h"
#include "pal_error.h"
#include "pal_internal.h"

/* 96 bytes is the minimal size of buffer to store a IPv4/IPv6 address */
#define PAL_SOCKADDR_SIZE 96

/* listen on a tcp socket */
static int tcp_listen(PAL_HANDLE* handle, char* uri, pal_stream_options_t options) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

/* accept a tcp connection */
static int tcp_accept(PAL_HANDLE handle, PAL_HANDLE* client, pal_stream_options_t options) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

/* connect on a tcp socket */
static int tcp_connect(PAL_HANDLE* handle, char* uri, pal_stream_options_t options) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

/* 'open' operation of tcp stream */
static int tcp_open(PAL_HANDLE* handle, const char* type, const char* uri, enum pal_access access,
                    pal_share_flags_t share, enum pal_create_mode create,
                    pal_stream_options_t options) {
    size_t uri_size = strlen(uri) + 1;

    if (uri_size > PAL_SOCKADDR_SIZE)
        return -PAL_ERROR_TOOLONG;

    char uri_buf[PAL_SOCKADDR_SIZE];
    memcpy(uri_buf, uri, uri_size);

    if (!strcmp(type, URI_TYPE_TCP_SRV))
        return tcp_listen(handle, uri_buf, create);

    if (!strcmp(type, URI_TYPE_TCP))
        return tcp_connect(handle, uri_buf, create);

    return -PAL_ERROR_NOTSUPPORT;
}

/* 'read' operation of tcp stream */
static int64_t tcp_read(PAL_HANDLE handle, uint64_t offset, size_t len, void* buf) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

/* write' operation of tcp stream */
static int64_t tcp_write(PAL_HANDLE handle, uint64_t offset, size_t len, const void* buf) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

/* used by 'open' operation of tcp stream for bound socket */
static int udp_bind(PAL_HANDLE* handle, char* uri, pal_stream_options_t options) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

/* used by 'open' operation of tcp stream for connected socket */
static int udp_connect(PAL_HANDLE* handle, char* uri) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

static int udp_open(PAL_HANDLE* hdl, const char* type, const char* uri, enum pal_access access,
                    pal_share_flags_t share, enum pal_create_mode create,
                    pal_stream_options_t options) {
    char buf[PAL_SOCKADDR_SIZE];
    size_t len = strlen(uri);

    if (len >= PAL_SOCKADDR_SIZE)
        return -PAL_ERROR_TOOLONG;

    memcpy(buf, uri, len + 1);

    if (!strcmp(type, URI_TYPE_UDP_SRV))
        return udp_bind(hdl, buf, create);

    if (!strcmp(type, URI_TYPE_UDP))
        return udp_connect(hdl, buf);

    return -PAL_ERROR_NOTSUPPORT;
}

static int64_t udp_receive(PAL_HANDLE handle, uint64_t offset, size_t len, void* buf) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

static int64_t udp_receivebyaddr(PAL_HANDLE handle, uint64_t offset, size_t len, void* buf,
                                 char* addr, size_t addrlen) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

static int64_t udp_send(PAL_HANDLE handle, uint64_t offset, size_t len, const void* buf) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

static int64_t udp_sendbyaddr(PAL_HANDLE handle, uint64_t offset, size_t len, const void* buf,
                              const char* addr, size_t addrlen) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

static int socket_delete(PAL_HANDLE handle, enum pal_delete_mode delete_mode) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

static int socket_close(PAL_HANDLE handle) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

static int socket_attrquerybyhdl(PAL_HANDLE handle, PAL_STREAM_ATTR* attr) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

static int socket_getname(PAL_HANDLE handle, char* buffer, size_t count) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

struct handle_ops g_tcp_ops = {
    .getname        = &socket_getname,
    .open           = &tcp_open,
    .waitforclient  = &tcp_accept,
    .read           = &tcp_read,
    .write          = &tcp_write,
    .delete         = &socket_delete,
    .close          = &socket_close,
    .attrquerybyhdl = &socket_attrquerybyhdl,
};

struct handle_ops g_udp_ops = {
    .getname        = &socket_getname,
    .open           = &udp_open,
    .read           = &udp_receive,
    .write          = &udp_send,
    .delete         = &socket_delete,
    .close          = &socket_close,
    .attrquerybyhdl = &socket_attrquerybyhdl,
};

struct handle_ops g_udpsrv_ops = {
    .getname        = &socket_getname,
    .open           = &udp_open,
    .readbyaddr     = &udp_receivebyaddr,
    .writebyaddr    = &udp_sendbyaddr,
    .delete         = &socket_delete,
    .close          = &socket_close,
    .attrquerybyhdl = &socket_attrquerybyhdl,
};
