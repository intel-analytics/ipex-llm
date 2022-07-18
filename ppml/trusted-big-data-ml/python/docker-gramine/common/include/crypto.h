/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * Cryptographic primitive abstractions. This layer provides a way to
 * change the crypto library without changing the rest of Gramine code
 * by providing a small crypto library adaptor implementing these methods.
 */

#ifndef CRYPTO_H
#define CRYPTO_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "api.h"

#define SHA256_DIGEST_LEN 32
#define DH_SIZE           384 /* DH_SIZE is tied to the choice of parameters in mbedtls_adapter.c */

#ifdef CRYPTO_USE_MBEDTLS
#define CRYPTO_PROVIDER_SPECIFIED

#include "mbedtls/cmac.h"
#include "mbedtls/ctr_drbg.h"
#include "mbedtls/dhm.h"
#include "mbedtls/entropy.h"
#include "mbedtls/rsa.h"
#include "mbedtls/sha256.h"
#include "mbedtls/ssl.h"

typedef struct AES LIB_AES_CONTEXT;

typedef mbedtls_sha256_context LIB_SHA256_CONTEXT;

typedef mbedtls_dhm_context LIB_DH_CONTEXT;
typedef struct {
    mbedtls_cipher_type_t cipher;
    mbedtls_cipher_context_t ctx;
} LIB_AESCMAC_CONTEXT;

typedef struct {
    mbedtls_entropy_context entropy;
    mbedtls_ctr_drbg_context ctr_drbg;
    mbedtls_ssl_config conf;
    mbedtls_ssl_context ssl;
    int ciphersuites[2]; /* [0] is actual ciphersuite, [1] must be 0 to indicate end of array */
    ssize_t (*pal_recv_cb)(int fd, void* buf, size_t buf_size);
    ssize_t (*pal_send_cb)(int fd, const void* buf, size_t buf_size);
    int stream_fd;
} LIB_SSL_CONTEXT;

#endif /* CRYPTO_USE_MBEDTLS */

#ifndef CRYPTO_PROVIDER_SPECIFIED
#error "Unknown crypto provider. Set CRYPTO_PROVIDER in Makefile"
#endif

/* SHA256 */
int lib_SHA256Init(LIB_SHA256_CONTEXT* context);
int lib_SHA256Update(LIB_SHA256_CONTEXT* context, const uint8_t* data, size_t data_size);
int lib_SHA256Final(LIB_SHA256_CONTEXT* context, uint8_t* output);

/* HKDF-SHA256 for key derivation */
int lib_HKDF_SHA256(const uint8_t* input_key, size_t input_key_size, const uint8_t* salt,
                    size_t salt_size, const uint8_t* info, size_t info_size, uint8_t* output_key,
                    size_t output_key_size);

/* Diffie-Hellman Key Exchange */
int lib_DhInit(LIB_DH_CONTEXT* context);
int lib_DhCreatePublic(LIB_DH_CONTEXT* context, uint8_t* public, size_t public_size);
int lib_DhCalcSecret(LIB_DH_CONTEXT* context, uint8_t* peer, size_t peer_size, uint8_t* secret,
                     size_t* secret_size);
void lib_DhFinal(LIB_DH_CONTEXT* context);

/* AES-CMAC */
int lib_AESCMAC(const uint8_t* key, size_t key_size, const uint8_t* input, size_t input_size,
                uint8_t* mac, size_t mac_size);
/* GCM encrypt, iv is assumed to be 12 bytes (and is changed by this call).
 * input_size doesn't have to be a multiple of 16.
 * Additional authenticated data (aad) may be NULL if absent.
 * Output size is the same as input_size. */
int lib_AESGCMEncrypt(const uint8_t* key, size_t key_size, const uint8_t* iv, const uint8_t* input,
                      size_t input_size, const uint8_t* aad, size_t aad_size, uint8_t* output,
                      uint8_t* tag, size_t tag_size);
/* GCM decrypt, iv is assumed to be 12 bytes (and is changed by this call).
 * input_len doesn't have to be a multiple of 16.
 * Additional authenticated data (aad) may be NULL if absent.
 * Output len is the same as input_len. */
int lib_AESGCMDecrypt(const uint8_t* key, size_t key_size, const uint8_t* iv, const uint8_t* input,
                      size_t input_size, const uint8_t* aad, size_t aad_size, uint8_t* output,
                      const uint8_t* tag, size_t tag_size);

/* note: 'lib_AESCMAC' is the combination of 'lib_AESCMACInit',
 * 'lib_AESCMACUpdate', and 'lib_AESCMACFinish'. */
int lib_AESCMACInit(LIB_AESCMAC_CONTEXT* context, const uint8_t* key, size_t key_size);
int lib_AESCMACUpdate(LIB_AESCMAC_CONTEXT* context, const uint8_t* input, size_t input_size);
int lib_AESCMACFinish(LIB_AESCMAC_CONTEXT* context, uint8_t* mac, size_t mac_size);

/* SSL/TLS */
int lib_SSLInit(LIB_SSL_CONTEXT* ssl_ctx, int stream_fd, bool is_server, const uint8_t* psk,
                size_t psk_size, ssize_t (*pal_recv_cb)(int fd, void* buf, size_t buf_size),
                ssize_t (*pal_send_cb)(int fd, const void* buf, size_t buf_size),
                const uint8_t* buf_load_ssl_ctx, size_t buf_size);
int lib_SSLFree(LIB_SSL_CONTEXT* ssl_ctx);
int lib_SSLHandshake(LIB_SSL_CONTEXT* ssl_ctx);
int lib_SSLRead(LIB_SSL_CONTEXT* ssl_ctx, uint8_t* buf, size_t buf_size);
int lib_SSLWrite(LIB_SSL_CONTEXT* ssl_ctx, const uint8_t* buf, size_t buf_size);
int lib_SSLSave(LIB_SSL_CONTEXT* ssl_ctx, uint8_t* buf, size_t buf_size, size_t* out_size);

#endif /* CRYPTO_H */
