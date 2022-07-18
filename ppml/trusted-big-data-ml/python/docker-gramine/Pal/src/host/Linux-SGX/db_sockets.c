/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains operands for streams with URIs that start with "tcp:", "tcp.srv:", "udp:",
 * "udp.srv:".
 */

#include <asm-generic/socket.h>
#include <asm/fcntl.h>
#include <linux/in.h>
#include <linux/in6.h>
#include <linux/poll.h>
#include <linux/types.h>

#include "api.h"
#include "pal.h"
#include "pal_error.h"
#include "pal_flags_conv.h"
#include "pal_internal.h"
#include "pal_linux.h"
#include "pal_linux_error.h"

#ifndef SOL_TCP
#define SOL_TCP 6
#endif

#ifndef TCP_NODELAY
#define TCP_NODELAY 1
#endif

#ifndef TCP_CORK
#define TCP_CORK 3
#endif

/* 96 bytes is the minimal size of buffer to store a IPv4/IPv6
   address */
#define PAL_SOCKADDR_SIZE 96

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

/* parsing the string of uri, and fill in the socket address structure.
   the latest pointer of uri, length of socket address are returned. */
static int inet_parse_uri(char** uri, struct sockaddr* addr, size_t* addrlen) {
    char* tmp      = *uri;
    char* end;
    char* addr_str = NULL;
    char* port_str;
    int af;
    void* addr_buf;
    int addr_len;
    __be16* port_buf;
    size_t slen;

    assert(addrlen);

    if (tmp[0] == '[') {
        /* for IPv6, the address will be in the form of
           "[xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx]:port". */
        struct sockaddr_in6* addr_in6 = (struct sockaddr_in6*)addr;

        slen = sizeof(*addr_in6);
        if (*addrlen < slen)
            goto inval;

        memset(addr, 0, slen);

        end = strchr(tmp + 1, ']');
        if (!end || *(end + 1) != ':')
            goto inval;

        addr_str = tmp + 1;
        addr_len = end - tmp - 1;
        port_str = end + 2;
        for (end = port_str; *end >= '0' && *end <= '9'; end++)
            ;
        addr_in6->sin6_family = af = AF_INET6;
        addr_buf                   = &addr_in6->sin6_addr.s6_addr;
        port_buf                   = &addr_in6->sin6_port;
    } else {
        /* for IP, the address will be in the form of "x.x.x.x:port". */
        struct sockaddr_in* addr_in = (struct sockaddr_in*)addr;

        slen = sizeof(*addr_in);
        if (*addrlen < slen)
            goto inval;

        memset(addr, 0, slen);

        end = strchr(tmp, ':');
        if (!end)
            goto inval;

        addr_str = tmp;
        addr_len = end - tmp;
        port_str = end + 1;
        for (end = port_str; *end >= '0' && *end <= '9'; end++)
            ;
        addr_in->sin_family = af = AF_INET;
        addr_buf                 = &addr_in->sin_addr.s_addr;
        port_buf                 = &addr_in->sin_port;
    }

    if (af == AF_INET) {
        if (!inet_pton4(addr_str, addr_len, addr_buf))
            goto inval;
    } else {
        if (!inet_pton6(addr_str, addr_len, addr_buf))
            goto inval;
    }

    *port_buf = __htons(atoi(port_str));
    *uri      = *end ? end + 1 : NULL;

    *addrlen = slen;

    return 0;

inval:
    return -PAL_ERROR_INVAL;
}

/* create the string of uri from the given socket address */
static int inet_create_uri(char* buf, size_t buf_size, struct sockaddr* addr, size_t addrlen,
                           size_t* output_len) {
    int len = 0; /* snprintf returns `int` */

    if (addr->sa_family == AF_INET) {
        if (addrlen < sizeof(struct sockaddr_in))
            return -PAL_ERROR_INVAL;

        struct sockaddr_in* addr_in = (struct sockaddr_in*)addr;
        char* addr = (char*)&addr_in->sin_addr.s_addr;

        /* for IP, the address will be in the form of "x.x.x.x:port". */
        len = snprintf(buf, buf_size, "%u.%u.%u.%u:%u", (unsigned char)addr[0],
                       (unsigned char)addr[1], (unsigned char)addr[2], (unsigned char)addr[3],
                       __ntohs(addr_in->sin_port));
    } else if (addr->sa_family == AF_INET6) {
        if (addrlen < sizeof(struct sockaddr_in6))
            return -PAL_ERROR_INVAL;

        struct sockaddr_in6* addr_in6 = (struct sockaddr_in6*)addr;
        unsigned short* addr          = (unsigned short*)&addr_in6->sin6_addr.s6_addr;

        /* for IPv6, the address will be in the form of
           "[xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx]:port". */
        len = snprintf(buf, buf_size, "[%04x:%04x:%04x:%04x:%04x:%04x:%04x:%04x]:%u",
                       __ntohs(addr[0]), __ntohs(addr[1]), __ntohs(addr[2]), __ntohs(addr[3]),
                       __ntohs(addr[4]), __ntohs(addr[5]), __ntohs(addr[6]), __ntohs(addr[7]),
                       __ntohs(addr_in6->sin6_port));
    } else {
        return -PAL_ERROR_INVAL;
    }

    if (len < 0)
        return len;
    if ((size_t)len >= buf_size)
        return -PAL_ERROR_TOOLONG;

    if (output_len)
        *output_len = (size_t)len;
    return 0;
}

/*
 * Parse a URI for a socket stream. The URI might have both binding address and connecting
 * address, or connecting address only. The form of URI will be either
 * "bind-addr:bind-port:connect-addr:connect-port" or "addr:port".
 */
static int socket_parse_uri(char* uri, struct sockaddr** bind_addr, size_t* bind_addrlen,
                            struct sockaddr** dest_addr, size_t* dest_addrlen) {
    int ret;

    if (!bind_addr && !dest_addr)
        return 0;

    if (!uri || !(*uri)) {
        if (bind_addr)
            *bind_addr = NULL;
        if (bind_addrlen)
            *bind_addrlen = 0;
        if (dest_addr)
            *dest_addr = NULL;
        if (dest_addrlen)
            *dest_addrlen = 0;
        return 0;
    }

    /* at least parse uri once */
    if ((ret = inet_parse_uri(&uri, bind_addr ? *bind_addr : *dest_addr,
                              bind_addr ? bind_addrlen : dest_addrlen)) < 0)
        return ret;

    if (!(bind_addr && dest_addr))
        return 0;

    /* if you reach here, it can only be connection address */
    if (!uri || (ret = inet_parse_uri(&uri, *dest_addr, dest_addrlen)) < 0) {
        *dest_addr    = *bind_addr;
        *dest_addrlen = *bind_addrlen;
        *bind_addr    = NULL;
        *bind_addrlen = 0;
    }

    return 0;
}

/* fill in the PAL handle based on the file descriptors and address given. */
static inline PAL_HANDLE socket_create_handle(int type, int fd, pal_stream_options_t options,
                                              struct sockaddr* bind_addr, size_t bind_addrlen,
                                              struct sockaddr* dest_addr, size_t dest_addrlen,
                                              struct sockopt* sock_options) {
    PAL_HANDLE hdl = calloc(1, HANDLE_SIZE(sock) + (bind_addr ? bind_addrlen : 0)
                               + (dest_addr ? dest_addrlen : 0));
    if (!hdl)
        return NULL;

    init_handle_hdr(hdl, type);
    hdl->flags |= PAL_HANDLE_FD_READABLE;
    if (type != PAL_TYPE_TCPSRV) {
        hdl->flags |= PAL_HANDLE_FD_WRITABLE;
    }
    hdl->sock.fd = fd;
    uint8_t* addr = (uint8_t*)hdl + HANDLE_SIZE(sock);
    if (bind_addr) {
        hdl->sock.bind = (struct sockaddr*)addr;
        memcpy(addr, bind_addr, bind_addrlen);
        addr += bind_addrlen;
    } else {
        hdl->sock.bind = NULL;
    }
    if (dest_addr) {
        hdl->sock.conn = (struct sockaddr*)addr;
        memcpy(addr, dest_addr, dest_addrlen);
        addr += dest_addrlen;
    } else {
        hdl->sock.conn = NULL;
    }

    hdl->sock.nonblocking = !!(options & PAL_OPTION_NONBLOCK);

    hdl->sock.linger            = sock_options->linger;
    hdl->sock.receivebuf        = sock_options->receivebuf;
    hdl->sock.sendbuf           = sock_options->sendbuf;
    hdl->sock.receivetimeout_us = sock_options->receivetimeout_us;
    hdl->sock.sendtimeout_us    = sock_options->sendtimeout_us;
    hdl->sock.tcp_cork          = sock_options->tcp_cork;
    hdl->sock.tcp_keepalive     = sock_options->tcp_keepalive;
    hdl->sock.tcp_nodelay       = sock_options->tcp_nodelay;
    return hdl;
}

static inline int sock_type(int type, pal_stream_options_t options) {
    if (options & PAL_OPTION_NONBLOCK)
        type |= SOCK_NONBLOCK;
    return type;
}

/* listen on a tcp socket */
static int tcp_listen(PAL_HANDLE* handle, char* uri, pal_stream_options_t options) {
    struct sockaddr_storage buffer;
    struct sockaddr* bind_addr = (struct sockaddr*)&buffer;
    size_t bind_addrlen = sizeof(buffer);
    int ret;

    if ((ret = socket_parse_uri(uri, &bind_addr, &bind_addrlen, NULL, NULL)) < 0)
        return ret;
    if (!bind_addr)
        return -PAL_ERROR_INVAL;

    int ipv6_v6only = options & PAL_OPTION_DUALSTACK ? 0 : 1;
    ret = ocall_listen(bind_addr->sa_family, sock_type(SOCK_STREAM, options), 0, ipv6_v6only,
                       bind_addr, &bind_addrlen);
    if (ret < 0)
        return unix_to_pal_error(ret);

    struct sockopt sock_options = {
        .reuseaddr = 1, /* sockets are always set as reusable in Gramine */
    };

    *handle = socket_create_handle(PAL_TYPE_TCPSRV, ret, options, bind_addr, bind_addrlen, NULL, 0,
                                   &sock_options);
    if (!(*handle)) {
        ocall_close(ret);
        return -PAL_ERROR_NOMEM;
    }

    return 0;
}

/* accept a tcp connection */
static int tcp_accept(PAL_HANDLE handle, PAL_HANDLE* client, pal_stream_options_t options) {
    if (HANDLE_HDR(handle)->type != PAL_TYPE_TCPSRV || !handle->sock.bind || handle->sock.conn)
        return -PAL_ERROR_NOTSERVER;

    if (handle->sock.fd == PAL_IDX_POISON)
        return -PAL_ERROR_BADHANDLE;

    struct sockaddr_storage bind_addr;
    size_t bind_addrlen = sizeof(bind_addr);
    struct sockaddr_storage dest_addr;
    size_t dest_addrlen = sizeof(dest_addr);
    int ret = 0;
    static_assert(O_CLOEXEC == SOCK_CLOEXEC && O_NONBLOCK == SOCK_NONBLOCK, "assumed below");
    int flags = PAL_OPTION_TO_LINUX_OPEN(options);

    ret = ocall_accept(handle->sock.fd, (struct sockaddr*)&dest_addr, &dest_addrlen, (struct sockaddr*)&bind_addr, &bind_addrlen, flags);
    if (ret < 0)
        return unix_to_pal_error(ret);

    struct sockopt sock_options = {
        .reuseaddr = 1, /* sockets are always set as reusable in Gramine */
    };

    *client = socket_create_handle(PAL_TYPE_TCP, ret, options, (struct sockaddr*)&bind_addr, bind_addrlen,
                                   (struct sockaddr*)&dest_addr, dest_addrlen, &sock_options);

    if (!(*client)) {
        ocall_close(ret);
        return -PAL_ERROR_NOMEM;
    }

    return 0;
}

/* connect on a tcp socket */
static int tcp_connect(PAL_HANDLE* handle, char* uri, pal_stream_options_t options) {
    struct sockaddr_storage buffer[2];
    struct sockaddr* bind_addr = (struct sockaddr*)&buffer[0];
    size_t bind_addrlen = sizeof(buffer[0]);
    struct sockaddr* dest_addr = (struct sockaddr*)&buffer[1];
    size_t dest_addrlen = sizeof(buffer[1]);
    int ret;

    /* accepting two kind of different uri:
       dest-ip:dest-port or bind-ip:bind-port:dest-ip:dest-port */
    if ((ret = socket_parse_uri(uri, &bind_addr, &bind_addrlen, &dest_addr, &dest_addrlen)) < 0)
        return ret;

    if (!dest_addr)
        return -PAL_ERROR_INVAL;

    if (bind_addr && bind_addr->sa_family != dest_addr->sa_family)
        return -PAL_ERROR_INVAL;

    ret = ocall_connect(dest_addr->sa_family, sock_type(SOCK_STREAM, options), 0, /*ipv6_v6only=*/0,
                        dest_addr, dest_addrlen, bind_addr, &bind_addrlen);
    if (ret < 0)
        return unix_to_pal_error(ret);

    struct sockopt sock_options = {
        .reuseaddr = 1, /* sockets are always set as reusable in Gramine */
    };

    *handle = socket_create_handle(PAL_TYPE_TCP, ret, options, bind_addr, bind_addrlen, dest_addr,
                                   dest_addrlen, &sock_options);

    if (!(*handle)) {
        ocall_close(ret);
        return -PAL_ERROR_NOMEM;
    }

    return 0;
}

/* 'open' operation of tcp stream */
static int tcp_open(PAL_HANDLE* handle, const char* type, const char* uri, enum pal_access access,
                    pal_share_flags_t share, enum pal_create_mode create,
                    pal_stream_options_t options) {
    __UNUSED(access);
    __UNUSED(share);
    __UNUSED(create);
    assert(create == PAL_CREATE_IGNORED);

    assert(WITHIN_MASK(share,   PAL_SHARE_MASK));
    assert(WITHIN_MASK(options, PAL_OPTION_MASK));

    size_t uri_size = strlen(uri) + 1;

    if (uri_size > PAL_SOCKADDR_SIZE)
        return -PAL_ERROR_TOOLONG;

    char uri_buf[PAL_SOCKADDR_SIZE];
    memcpy(uri_buf, uri, uri_size);

    if (!strcmp(type, URI_TYPE_TCP_SRV))
        return tcp_listen(handle, uri_buf, options);

    if (!strcmp(type, URI_TYPE_TCP))
        return tcp_connect(handle, uri_buf, options);

    return -PAL_ERROR_NOTSUPPORT;
}

/* 'read' operation of tcp stream */
static int64_t tcp_read(PAL_HANDLE handle, uint64_t offset, uint64_t len, void* buf) {
    if (offset)
        return -PAL_ERROR_INVAL;

    if (HANDLE_HDR(handle)->type != PAL_TYPE_TCP || !handle->sock.conn)
        return -PAL_ERROR_NOTCONNECTION;

    if (handle->sock.fd == PAL_IDX_POISON)
        return 0;

    ssize_t bytes = ocall_recv(handle->sock.fd, buf, len, NULL, NULL, NULL, NULL);

    if (bytes < 0)
        return unix_to_pal_error(bytes);

    return bytes;
}

/* write' operation of tcp stream */
static int64_t tcp_write(PAL_HANDLE handle, uint64_t offset, uint64_t len, const void* buf) {
    if (offset)
        return -PAL_ERROR_INVAL;

    if (HANDLE_HDR(handle)->type != PAL_TYPE_TCP || !handle->sock.conn)
        return -PAL_ERROR_NOTCONNECTION;

    if (handle->sock.fd == PAL_IDX_POISON)
        return -PAL_ERROR_CONNFAILED;

    ssize_t bytes = ocall_send(handle->sock.fd, buf, len, NULL, 0, NULL, 0);
    if (bytes < 0)
        return unix_to_pal_error(bytes);

    return bytes;
}

/* used by 'open' operation of tcp stream for bound socket */
static int udp_bind(PAL_HANDLE* handle, char* uri, pal_stream_options_t options) {
    struct sockaddr_storage buffer;
    struct sockaddr* bind_addr = (struct sockaddr*)&buffer;
    size_t bind_addrlen = sizeof(buffer);
    int ret = 0;

    if ((ret = socket_parse_uri(uri, &bind_addr, &bind_addrlen, NULL, NULL)) < 0)
        return ret;

    if (!bind_addr)
        return -PAL_ERROR_INVAL;
    assert(bind_addrlen == addr_size(bind_addr));

    int ipv6_v6only = options & PAL_OPTION_DUALSTACK ? 0 : 1;
    ret = ocall_listen(bind_addr->sa_family, sock_type(SOCK_DGRAM, options), 0, ipv6_v6only,
                       bind_addr, &bind_addrlen);
    if (ret < 0)
        return unix_to_pal_error(ret);

    struct sockopt sock_options = {
        .reuseaddr = 1, /* sockets are always set as reusable in Gramine */
    };

    *handle = socket_create_handle(PAL_TYPE_UDPSRV, ret, options, bind_addr, bind_addrlen, NULL, 0,
                                   &sock_options);

    if (!(*handle)) {
        ocall_close(ret);
        return -PAL_ERROR_NOMEM;
    }

    return 0;
}

/* used by 'open' operation of tcp stream for connected socket */
static int udp_connect(PAL_HANDLE* handle, char* uri, pal_stream_options_t options) {
    struct sockaddr_storage buffer[2];
    struct sockaddr* bind_addr = (struct sockaddr*)&buffer[0];
    size_t bind_addrlen = sizeof(buffer[0]);
    struct sockaddr* dest_addr = (struct sockaddr*)&buffer[1];
    size_t dest_addrlen = sizeof(buffer[1]);
    int ret;

    if ((ret = socket_parse_uri(uri, &bind_addr, &bind_addrlen, &dest_addr, &dest_addrlen)) < 0)
        return ret;

    int ipv6_v6only = options & PAL_OPTION_DUALSTACK ? 0 : 1;
    ret = ocall_connect(dest_addr ? dest_addr->sa_family : AF_INET, sock_type(SOCK_DGRAM, options),
                        0, ipv6_v6only, dest_addr, dest_addrlen, bind_addr, &bind_addrlen);

    if (ret < 0)
        return unix_to_pal_error(ret);

    struct sockopt sock_options = {
        .reuseaddr = 1, /* sockets are always set as reusable in Gramine */
    };

    *handle = socket_create_handle(dest_addr ? PAL_TYPE_UDP : PAL_TYPE_UDPSRV, ret, options,
                                   bind_addr, bind_addrlen, dest_addr, dest_addrlen, &sock_options);

    if (!(*handle)) {
        ocall_close(ret);
        return -PAL_ERROR_NOMEM;
    }

    return 0;
}

static int udp_open(PAL_HANDLE* hdl, const char* type, const char* uri, enum pal_access access,
                    pal_share_flags_t share, enum pal_create_mode create,
                    pal_stream_options_t options) {
    __UNUSED(access);
    __UNUSED(share);
    __UNUSED(create);
    assert(create == PAL_CREATE_IGNORED);

    assert(WITHIN_MASK(share,   PAL_SHARE_MASK));
    assert(WITHIN_MASK(options, PAL_OPTION_MASK));

    char buf[PAL_SOCKADDR_SIZE];
    size_t len = strlen(uri);

    if (len >= PAL_SOCKADDR_SIZE)
        return -PAL_ERROR_TOOLONG;

    memcpy(buf, uri, len + 1);

    if (!strcmp(type, URI_TYPE_UDP_SRV))
        return udp_bind(hdl, buf, options);

    if (!strcmp(type, URI_TYPE_UDP))
        return udp_connect(hdl, buf, options);

    return -PAL_ERROR_NOTSUPPORT;
}

static int64_t udp_receive(PAL_HANDLE handle, uint64_t offset, uint64_t len, void* buf) {
    if (offset)
        return -PAL_ERROR_INVAL;

    if (HANDLE_HDR(handle)->type != PAL_TYPE_UDP)
        return -PAL_ERROR_NOTCONNECTION;

    if (handle->sock.fd == PAL_IDX_POISON)
        return -PAL_ERROR_BADHANDLE;

    ssize_t ret = ocall_recv(handle->sock.fd, buf, len, NULL, NULL, NULL, NULL);
    return ret < 0 ? unix_to_pal_error(ret) : ret;
}

static int64_t udp_receivebyaddr(PAL_HANDLE handle, uint64_t offset, uint64_t len, void* buf,
                                 char* addr, size_t addrlen) {
    if (offset)
        return -PAL_ERROR_INVAL;

    if (HANDLE_HDR(handle)->type != PAL_TYPE_UDPSRV)
        return -PAL_ERROR_NOTCONNECTION;

    if (handle->sock.fd == PAL_IDX_POISON)
        return -PAL_ERROR_BADHANDLE;

    struct sockaddr_storage conn_addr;
    size_t conn_addrlen = sizeof(conn_addr);

    ssize_t bytes = ocall_recv(handle->sock.fd, buf, len, (struct sockaddr*)&conn_addr,
                               &conn_addrlen, NULL, NULL);

    if (bytes < 0)
        return unix_to_pal_error(bytes);

    char* addr_uri = strcpy_static(addr, URI_PREFIX_UDP, addrlen);
    if (!addr_uri)
        return -PAL_ERROR_OVERFLOW;

    int ret = inet_create_uri(addr_uri, addr + addrlen - addr_uri, (struct sockaddr*)&conn_addr,
                              conn_addrlen, NULL);
    if (ret < 0)
        return ret;

    return bytes;
}

static int64_t udp_send(PAL_HANDLE handle, uint64_t offset, uint64_t len, const void* buf) {
    if (offset)
        return -PAL_ERROR_INVAL;

    if (HANDLE_HDR(handle)->type != PAL_TYPE_UDP)
        return -PAL_ERROR_NOTCONNECTION;

    if (handle->sock.fd == PAL_IDX_POISON)
        return -PAL_ERROR_BADHANDLE;

    ssize_t bytes = ocall_send(handle->sock.fd, buf, len, NULL, 0, NULL, 0);
    if (bytes < 0)
        return unix_to_pal_error(bytes);

    return bytes;
}

static int64_t udp_sendbyaddr(PAL_HANDLE handle, uint64_t offset, uint64_t len, const void* buf,
                              const char* addr, size_t addrlen) {
    if (offset)
        return -PAL_ERROR_INVAL;

    if (HANDLE_HDR(handle)->type != PAL_TYPE_UDPSRV)
        return -PAL_ERROR_NOTCONNECTION;

    if (handle->sock.fd == PAL_IDX_POISON)
        return -PAL_ERROR_BADHANDLE;

    if (!strstartswith(addr, URI_PREFIX_UDP))
        return -PAL_ERROR_INVAL;

    addr += static_strlen(URI_PREFIX_UDP);
    addrlen -= static_strlen(URI_PREFIX_UDP);

    char* addrbuf = __alloca(addrlen + 1);
    memcpy(addrbuf, addr, addrlen);
    addrbuf[addrlen] = '\0';

    struct sockaddr_storage conn_addr;
    size_t conn_addrlen = sizeof(conn_addr);

    int ret = inet_parse_uri(&addrbuf, (struct sockaddr*)&conn_addr, &conn_addrlen);
    if (ret < 0)
        return ret;

    ssize_t bytes = ocall_send(handle->sock.fd, buf, len, (struct sockaddr*)&conn_addr,
                               conn_addrlen, NULL, 0);
    if (bytes < 0)
        return unix_to_pal_error(bytes);

    return bytes;
}

static int socket_delete(PAL_HANDLE handle, enum pal_delete_mode delete_mode) {
    if (handle->sock.fd == PAL_IDX_POISON)
        return 0;

    if (HANDLE_HDR(handle)->type != PAL_TYPE_TCP && delete_mode != PAL_DELETE_ALL)
        return -PAL_ERROR_INVAL;

    if (HANDLE_HDR(handle)->type == PAL_TYPE_TCP || HANDLE_HDR(handle)->type == PAL_TYPE_TCPSRV) {
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

        ocall_shutdown(handle->sock.fd, shutdown);
    }

    return 0;
}

static int socket_close(PAL_HANDLE handle) {
    if (handle->sock.fd != PAL_IDX_POISON) {
        ocall_close(handle->sock.fd);
        handle->sock.fd = PAL_IDX_POISON;
    }

    if (handle->sock.bind)
        handle->sock.bind = NULL;

    if (handle->sock.conn)
        handle->sock.conn = NULL;

    return 0;
}

static int socket_attrquerybyhdl(PAL_HANDLE handle, PAL_STREAM_ATTR* attr) {
    int ret;

    if (handle->sock.fd == PAL_IDX_POISON)
        return -PAL_ERROR_BADHANDLE;

    attr->handle_type              = HANDLE_HDR(handle)->type;
    attr->nonblocking              = handle->sock.nonblocking;

    attr->socket.linger            = handle->sock.linger;
    attr->socket.receivebuf        = handle->sock.receivebuf;
    attr->socket.sendbuf           = handle->sock.sendbuf;
    attr->socket.receivetimeout_us = handle->sock.receivetimeout_us;
    attr->socket.sendtimeout_us    = handle->sock.sendtimeout_us;
    attr->socket.tcp_cork          = handle->sock.tcp_cork;
    attr->socket.tcp_keepalive     = handle->sock.tcp_keepalive;
    attr->socket.tcp_nodelay       = handle->sock.tcp_nodelay;

    /* get number of bytes available for reading (doesn't make sense for listening sockets) */
    attr->pending_size = 0;
    if (HANDLE_HDR(handle)->type != PAL_TYPE_TCPSRV) {
        ret = ocall_fionread(handle->sock.fd);
        if (ret < 0)
            return unix_to_pal_error(ret);

        attr->pending_size = ret;
    }

    return 0;
}

static int socket_attrsetbyhdl(PAL_HANDLE handle, PAL_STREAM_ATTR* attr) {
    if (handle->sock.fd == PAL_IDX_POISON)
        return -PAL_ERROR_BADHANDLE;

    int fd = handle->sock.fd, ret, val;
    if (attr->nonblocking != handle->sock.nonblocking) {
        ret = ocall_fsetnonblock(fd, attr->nonblocking);

        if (ret < 0)
            return unix_to_pal_error(ret);

        handle->sock.nonblocking = attr->nonblocking;
    }

    struct __kernel_linger {
        int l_onoff;
        int l_linger;
    };

    if (attr->socket.linger != handle->sock.linger) {
        struct __kernel_linger l;
        l.l_onoff  = attr->socket.linger ? 1 : 0;
        l.l_linger = attr->socket.linger;
        ret = ocall_setsockopt(fd, SOL_SOCKET, SO_LINGER, &l, sizeof(struct __kernel_linger));
        if (ret < 0)
            return unix_to_pal_error(ret);

        handle->sock.linger = attr->socket.linger;
    }

    if (attr->socket.receivebuf != handle->sock.receivebuf) {
        val = attr->socket.receivebuf;
        ret = ocall_setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &val, sizeof(int));
        if (ret < 0)
            return unix_to_pal_error(ret);

        handle->sock.receivebuf = attr->socket.receivebuf;
    }

    if (attr->socket.sendbuf != handle->sock.sendbuf) {
        val = attr->socket.sendbuf;
        ret = ocall_setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &val, sizeof(int));
        if (ret < 0)
            return unix_to_pal_error(ret);

        handle->sock.sendbuf = attr->socket.sendbuf;
    }

    if (attr->socket.receivetimeout_us != handle->sock.receivetimeout_us) {
        struct timeval tv;
        tv.tv_sec  = attr->socket.receivetimeout_us / TIME_US_IN_S;
        tv.tv_usec = attr->socket.receivetimeout_us % TIME_US_IN_S;
        ret = ocall_setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        if (ret < 0)
            return unix_to_pal_error(ret);

        handle->sock.receivetimeout_us = attr->socket.receivetimeout_us;
    }

    if (attr->socket.sendtimeout_us != handle->sock.sendtimeout_us) {
        struct timeval tv;
        tv.tv_sec  = attr->socket.sendtimeout_us / TIME_US_IN_S;
        tv.tv_usec = attr->socket.sendtimeout_us % TIME_US_IN_S;
        ret = ocall_setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        if (ret < 0)
            return unix_to_pal_error(ret);

        handle->sock.sendtimeout_us = attr->socket.sendtimeout_us;
    }

    if (HANDLE_TYPE(handle) == PAL_TYPE_TCP || HANDLE_TYPE(handle) == PAL_TYPE_TCPSRV) {
        if (attr->socket.tcp_cork != handle->sock.tcp_cork) {
            val = attr->socket.tcp_cork ? 1 : 0;
            ret = ocall_setsockopt(fd, SOL_TCP, TCP_CORK, &val, sizeof(int));
            if (ret < 0)
                return unix_to_pal_error(ret);

            handle->sock.tcp_cork = attr->socket.tcp_cork;
        }

        if (attr->socket.tcp_keepalive != handle->sock.tcp_keepalive) {
            val = attr->socket.tcp_keepalive ? 1 : 0;
            ret = ocall_setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &val, sizeof(int));
            if (ret < 0)
                return unix_to_pal_error(ret);

            handle->sock.tcp_keepalive = attr->socket.tcp_keepalive;
        }

        if (attr->socket.tcp_nodelay != handle->sock.tcp_nodelay) {
            val = attr->socket.tcp_nodelay ? 1 : 0;
            ret = ocall_setsockopt(fd, SOL_TCP, TCP_NODELAY, &val, sizeof(int));
            if (ret < 0)
                return unix_to_pal_error(ret);

            handle->sock.tcp_nodelay = attr->socket.tcp_nodelay;
        }
    }

    return 0;
}

static int socket_getname(PAL_HANDLE handle, char* buffer, size_t count) {
    size_t orig_count = count;
    int ret;

    const char* prefix = NULL;
    size_t prefix_len = 0;
    struct sockaddr* bind_addr = NULL;
    struct sockaddr* dest_addr = NULL;

    switch (PAL_GET_TYPE(handle)) {
        case PAL_TYPE_TCPSRV:
            prefix_len = static_strlen(URI_PREFIX_TCP_SRV);
            prefix = URI_PREFIX_TCP_SRV;
            bind_addr = handle->sock.bind;
            break;
        case PAL_TYPE_TCP:
            prefix_len = static_strlen(URI_PREFIX_TCP);
            prefix = URI_PREFIX_TCP;
            bind_addr = handle->sock.bind;
            dest_addr = handle->sock.conn;
            break;
        case PAL_TYPE_UDPSRV:
            prefix_len = static_strlen(URI_PREFIX_UDP_SRV);
            prefix = URI_PREFIX_UDP_SRV;
            bind_addr = handle->sock.bind;
            break;
        case PAL_TYPE_UDP:
            prefix_len = static_strlen(URI_PREFIX_UDP);
            prefix = URI_PREFIX_UDP;
            bind_addr = handle->sock.bind;
            dest_addr = handle->sock.conn;
            break;
        default:
            return -PAL_ERROR_INVAL;
    }

    if (count < prefix_len + 1) {
        return -PAL_ERROR_OVERFLOW;
    }

    memcpy(buffer, prefix, prefix_len + 1);
    buffer += prefix_len;
    count -= prefix_len;

    if (bind_addr) {
        size_t uri_size;
        ret = inet_create_uri(buffer, count, bind_addr, addr_size(bind_addr), &uri_size);
        if (ret < 0) {
            return ret;
        }

        buffer += uri_size;
        count -= uri_size;
    }

    if (dest_addr) {
        if (bind_addr) {
            if (count < 2) {
                return -PAL_ERROR_OVERFLOW;
            }

            *buffer++ = ':';
            *buffer = '\0';
            count--;
        }

        size_t uri_size;
        ret = inet_create_uri(buffer, count, dest_addr, addr_size(dest_addr), &uri_size);
        if (ret < 0) {
            return ret;
        }

        buffer += uri_size;
        count -= uri_size;
    }

    return orig_count - count;
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
    .attrsetbyhdl   = &socket_attrsetbyhdl,
};

struct handle_ops g_udp_ops = {
    .getname        = &socket_getname,
    .open           = &udp_open,
    .read           = &udp_receive,
    .write          = &udp_send,
    .delete         = &socket_delete,
    .close          = &socket_close,
    .attrquerybyhdl = &socket_attrquerybyhdl,
    .attrsetbyhdl   = &socket_attrsetbyhdl,
};

struct handle_ops g_udpsrv_ops = {
    .getname        = &socket_getname,
    .open           = &udp_open,
    .readbyaddr     = &udp_receivebyaddr,
    .writebyaddr    = &udp_sendbyaddr,
    .delete         = &socket_delete,
    .close          = &socket_close,
    .attrquerybyhdl = &socket_attrquerybyhdl,
    .attrsetbyhdl   = &socket_attrsetbyhdl,
};
