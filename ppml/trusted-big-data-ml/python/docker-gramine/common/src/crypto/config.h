/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2017 Fortanix, Inc.
 * Copyright (C) 2021 Intel Corp.
 */

/* This mbedTLS config is for v2.26.0 and assumes Intel x86-64 CPU with AESNI and SSE2 support */

#ifndef MBEDTLS_CONFIG_H
#define MBEDTLS_CONFIG_H

#define MBEDTLS_AESNI_C
#define MBEDTLS_AES_C
#define MBEDTLS_BASE64_C
#define MBEDTLS_BIGNUM_C
#define MBEDTLS_CIPHER_C
#define MBEDTLS_CMAC_C
#define MBEDTLS_CTR_DRBG_C
#define MBEDTLS_DHM_C
#define MBEDTLS_ENTROPY_C
#define MBEDTLS_ENTROPY_HARDWARE_ALT
#define MBEDTLS_ERROR_C
#define MBEDTLS_GCM_C
#define MBEDTLS_GENPRIME
#define MBEDTLS_HAVE_ASM
#define MBEDTLS_HAVE_SSE2
#define MBEDTLS_HAVE_X86_64
#define MBEDTLS_HKDF_C
#define MBEDTLS_KEY_EXCHANGE_PSK_ENABLED
#define MBEDTLS_MD_C
#define MBEDTLS_NET_C
#define MBEDTLS_NO_PLATFORM_ENTROPY
#define MBEDTLS_NO_UDBL_DIVISION
#define MBEDTLS_OID_C
#define MBEDTLS_PKCS1
#define MBEDTLS_PKCS1_V15
#define MBEDTLS_PLATFORM_C
#define MBEDTLS_RSA_C
#define MBEDTLS_SHA256_C
#define MBEDTLS_SSL_CIPHERSUITES MBEDTLS_TLS_PSK_WITH_AES_128_GCM_SHA256
#define MBEDTLS_SSL_CLI_C
#define MBEDTLS_SSL_CONTEXT_SERIALIZATION
#define MBEDTLS_SSL_PROTO_TLS1_2
#define MBEDTLS_SSL_SRV_C
#define MBEDTLS_SSL_TLS_C

#define MBEDTLS_PLATFORM_NO_STD_FUNCTIONS

#include <limits.h>
#include <stddef.h>

void* calloc(size_t nmem, size_t size);
void free(void*);

#define MBEDTLS_PLATFORM_STD_CALLOC   calloc
#define MBEDTLS_PLATFORM_STD_FREE     free
#define MBEDTLS_PLATFORM_STD_SNPRINTF snprintf

#endif
