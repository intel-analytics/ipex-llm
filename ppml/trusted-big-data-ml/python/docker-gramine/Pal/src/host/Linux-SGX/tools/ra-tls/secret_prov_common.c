/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Labs */

/*!
 * \file
 *
 * This file contains common utilities for secret provisioning library.
 */

#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>

#include "mbedtls/ssl.h"

#include "secret_prov.h"

int secret_provision_write(struct ra_tls_ctx* ctx, const uint8_t* buf, size_t size) {
    int ret;

    if (!ctx || !ctx->ssl || size > INT_MAX)
        return -EINVAL;

    mbedtls_ssl_context* _ssl = (mbedtls_ssl_context*)ctx->ssl;

    size_t written = 0;
    while (written < size) {
        ret = mbedtls_ssl_write(_ssl, buf + written, size - written);
        if (ret == MBEDTLS_ERR_SSL_WANT_READ || ret == MBEDTLS_ERR_SSL_WANT_WRITE)
            continue;
        if (ret < 0) {
            /* use well-known error code for a typical case when remote party closes connection */
            return ret == MBEDTLS_ERR_SSL_PEER_CLOSE_NOTIFY ? -ECONNRESET : ret;
        }
        written += (size_t)ret;
    }
    assert(written == size);
    return (int)written;
}

int secret_provision_read(struct ra_tls_ctx* ctx, uint8_t* buf, size_t size) {
    int ret;

    if (!ctx || !ctx->ssl || size > INT_MAX)
        return -EINVAL;

    mbedtls_ssl_context* _ssl = (mbedtls_ssl_context*)ctx->ssl;

    size_t read = 0;
    while (read < size) {
        ret = mbedtls_ssl_read(_ssl, buf + read, size - read);
        if (!ret)
            return -ECONNRESET;
        if (ret == MBEDTLS_ERR_SSL_WANT_READ || ret == MBEDTLS_ERR_SSL_WANT_WRITE)
            continue;
        if (ret < 0) {
            /* use well-known error code for a typical case when remote party closes connection */
            return ret == MBEDTLS_ERR_SSL_PEER_CLOSE_NOTIFY ? -ECONNRESET : ret;
        }
        read += (size_t)ret;
    }

    assert(read == size);
    return (int)read;
}

int secret_provision_close(struct ra_tls_ctx* ctx) {
    if (!ctx || !ctx->ssl)
        return 0;

    mbedtls_ssl_context* _ssl = (mbedtls_ssl_context*)ctx->ssl;

    int ret = -1;
    while (ret < 0) {
        ret = mbedtls_ssl_close_notify(_ssl);
        if (ret == MBEDTLS_ERR_SSL_WANT_READ || ret == MBEDTLS_ERR_SSL_WANT_WRITE) {
            continue;
        }
        if (ret < 0) {
            /* use well-known error code for a typical case when remote party closes connection */
            return ret == MBEDTLS_ERR_SSL_PEER_CLOSE_NOTIFY ? -ECONNRESET : ret;
        }
    }
    return 0;
}
