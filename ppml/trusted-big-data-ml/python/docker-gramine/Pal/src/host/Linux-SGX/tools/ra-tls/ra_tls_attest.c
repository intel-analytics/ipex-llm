/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Labs */

/*!
 * \file
 *
 * This file contains the implementation of server-side attestation for TLS libraries. It contains
 * functions to create a self-signed RA-TLS certificate with an SGX quote embedded in it. It works
 * with both EPID-based (quote v2) and ECDSA-based (quote v3 or DCAP) SGX quotes (in fact, it is
 * agnostic to the format of the SGX quote).
 *
 * This file is part of the RA-TLS attestation library which is typically linked into server
 * applications. This library is *not* thread-safe.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <mbedtls/ctr_drbg.h>
#include <mbedtls/entropy.h>
#include <mbedtls/pk.h>
#include <mbedtls/rsa.h>
#include <mbedtls/sha256.h>
#include <mbedtls/x509_crt.h>

#include "ra_tls.h"
#include "sgx_arch.h"
#include "sgx_attest.h"

#define CERT_SUBJECT_NAME_VALUES  "CN=RATLS,O=GramineDevelopers,C=US"
#define CERT_TIMESTAMP_NOT_BEFORE_DEFAULT "20010101000000"
#define CERT_TIMESTAMP_NOT_AFTER_DEFAULT  "20301231235959"

static ssize_t rw_file(const char* path, uint8_t* buf, size_t len, bool do_write) {
    ssize_t bytes = 0;
    ssize_t ret = 0;

    int fd = open(path, do_write ? O_WRONLY : O_RDONLY);
    if (fd < 0)
        return fd;

    while ((ssize_t)len > bytes) {
        if (do_write)
            ret = write(fd, buf + bytes, len - bytes);
        else
            ret = read(fd, buf + bytes, len - bytes);

        if (ret > 0) {
            bytes += ret;
        } else if (ret == 0) {
            /* end of file */
            break;
        } else {
            if (ret < 0 && (errno == EAGAIN || errno == EINTR))
                continue;
            break;
        }
    }

    close(fd);
    return ret < 0 ? ret : bytes;
}

/*! given public key \p pk, generate an RA-TLS certificate \p writecrt with \p quote embedded */
static int generate_x509(mbedtls_pk_context* pk, const uint8_t* quote, size_t quote_size,
                         mbedtls_x509write_cert* writecrt) {
    int ret;
    char* cert_timestamp_not_before = NULL;
    char* cert_timestamp_not_after  = NULL;

    mbedtls_mpi serial;
    mbedtls_mpi_init(&serial);

    mbedtls_x509write_crt_init(writecrt);
    mbedtls_x509write_crt_set_md_alg(writecrt, MBEDTLS_MD_SHA256);

    /* generated certificate is self-signed, so declares itself both a subject and an issuer */
    mbedtls_x509write_crt_set_subject_key(writecrt, pk);
    mbedtls_x509write_crt_set_issuer_key(writecrt, pk);

    /* set (dummy) subject names for both subject and issuer */
    ret = mbedtls_x509write_crt_set_subject_name(writecrt, CERT_SUBJECT_NAME_VALUES);
    if (ret < 0)
        goto out;

    ret = mbedtls_x509write_crt_set_issuer_name(writecrt, CERT_SUBJECT_NAME_VALUES);
    if (ret < 0)
        goto out;

    /* set a serial number (dummy "1") for the generated certificate */
    ret = mbedtls_mpi_read_string(&serial, 10, "1");
    if (ret < 0)
        goto out;

    ret = mbedtls_x509write_crt_set_serial(writecrt, &serial);
    if (ret < 0)
        goto out;

    cert_timestamp_not_before = strdup(getenv(RA_TLS_CERT_TIMESTAMP_NOT_BEFORE) ? :
                                       CERT_TIMESTAMP_NOT_BEFORE_DEFAULT);
    if (!cert_timestamp_not_before) {
        ret = MBEDTLS_ERR_X509_ALLOC_FAILED;
        goto out;
    }

    cert_timestamp_not_after = strdup(getenv(RA_TLS_CERT_TIMESTAMP_NOT_AFTER) ? :
                                      CERT_TIMESTAMP_NOT_AFTER_DEFAULT);
    if (!cert_timestamp_not_after) {
        ret = MBEDTLS_ERR_X509_ALLOC_FAILED;
        goto out;
    }

    ret = mbedtls_x509write_crt_set_validity(writecrt, cert_timestamp_not_before,
                                             cert_timestamp_not_after);
    if (ret < 0)
        goto out;

    ret = mbedtls_x509write_crt_set_basic_constraints(writecrt, /*is_ca=*/0, /*max_pathlen=*/-1);
    if (ret < 0)
        goto out;

    ret = mbedtls_x509write_crt_set_subject_key_identifier(writecrt);
    if (ret < 0)
        goto out;

    ret = mbedtls_x509write_crt_set_authority_key_identifier(writecrt);
    if (ret < 0)
        goto out;

    /* finally, embed the quote into the generated certificate (as X.509 extension) */
    ret = mbedtls_x509write_crt_set_extension(writecrt, (const char*)quote_oid, quote_oid_len,
                                              /*critical=*/0, quote, quote_size);
    if (ret < 0)
        goto out;

    ret = 0;
out:
    free(cert_timestamp_not_before);
    free(cert_timestamp_not_after);
    mbedtls_mpi_free(&serial);
    return ret;
}

/*! calculate sha256 over public key \p pk and copy it into \p sha */
static int sha256_over_pk(mbedtls_pk_context* pk, uint8_t* sha) {
    uint8_t pk_der[PUB_KEY_SIZE_MAX] = {0};

    /* below function writes data at the end of the buffer */
    int pk_der_size_byte = mbedtls_pk_write_pubkey_der(pk, pk_der, PUB_KEY_SIZE_MAX);
    if (pk_der_size_byte != RSA_PUB_3072_KEY_DER_LEN)
        return MBEDTLS_ERR_PK_INVALID_PUBKEY;

    /* move the data to the beginning of the buffer, to avoid pointer arithmetic later */
    memmove(pk_der, pk_der + PUB_KEY_SIZE_MAX - pk_der_size_byte, pk_der_size_byte);

    return mbedtls_sha256_ret(pk_der, pk_der_size_byte, sha, /*is224=*/0);
}

/*! given public key \p pk, generate an RA-TLS certificate \p writecrt */
static int create_x509(mbedtls_pk_context* pk, mbedtls_x509write_cert* writecrt) {
    sgx_report_data_t user_report_data = {0};
    int ret = sha256_over_pk(pk, user_report_data.d);
    if (ret < 0)
        return ret;

    ssize_t written = rw_file("/dev/attestation/user_report_data", user_report_data.d,
                              sizeof(user_report_data.d), /*do_write=*/true);
    if (written != sizeof(user_report_data))
        return MBEDTLS_ERR_X509_FILE_IO_ERROR;

    uint8_t* quote = malloc(SGX_QUOTE_MAX_SIZE);
    if (!quote)
        return MBEDTLS_ERR_X509_ALLOC_FAILED;

    ssize_t quote_size = rw_file("/dev/attestation/quote", quote, SGX_QUOTE_MAX_SIZE,
                                 /*do_write=*/false);
    if (quote_size < 0) {
        free(quote);
        return MBEDTLS_ERR_X509_FILE_IO_ERROR;
    }

    ret = generate_x509(pk, quote, quote_size, writecrt);

    free(quote);
    return ret;
}

static int create_key_and_crt(mbedtls_pk_context* key, mbedtls_x509_crt* crt, uint8_t** crt_der,
                              size_t* crt_der_size) {
    int ret;

    if (!key || (!crt && !(crt_der && crt_der_size))) {
        /* mbedTLS API (ra_tls_create_key_and_crt) and generic API (ra_tls_create_key_and_crt_der)
         * both use `key`, but the former uses `crt` and the latter uses `crt_der` */
        return MBEDTLS_ERR_X509_FATAL_ERROR;
    }

    mbedtls_ctr_drbg_context ctr_drbg;
    mbedtls_ctr_drbg_init(&ctr_drbg);

    mbedtls_entropy_context entropy;
    mbedtls_entropy_init(&entropy);

    mbedtls_x509write_cert writecrt;
    mbedtls_x509write_crt_init(&writecrt);

    uint8_t* crt_der_buf = NULL;
    uint8_t* output_buf = NULL;
    size_t output_buf_size = 16 * 1024; /* enough for any X.509 certificate */

    output_buf = malloc(output_buf_size);
    if (!output_buf) {
        ret = MBEDTLS_ERR_X509_ALLOC_FAILED;
        goto out;
    }

    ret = mbedtls_ctr_drbg_seed(&ctr_drbg, mbedtls_entropy_func, &entropy, /*custom=*/NULL,
                                /*customlen=*/0);
    if (ret < 0)
        goto out;

    ret = mbedtls_pk_setup(key, mbedtls_pk_info_from_type(MBEDTLS_PK_RSA));
    if (ret < 0)
        goto out;

    mbedtls_rsa_init((mbedtls_rsa_context*)key->pk_ctx, MBEDTLS_RSA_PKCS_V15, /*hash_id=*/0);

    ret = mbedtls_rsa_gen_key((mbedtls_rsa_context*)key->pk_ctx, mbedtls_ctr_drbg_random, &ctr_drbg,
                              RSA_PUB_3072_KEY_LEN, RSA_PUB_EXPONENT);
    if (ret < 0)
        goto out;

    ret = create_x509(key, &writecrt);
    if (ret < 0)
        goto out;

    int size = mbedtls_x509write_crt_der(&writecrt, output_buf, output_buf_size,
                                         mbedtls_ctr_drbg_random, &ctr_drbg);
    if (size < 0) {
        ret = size;
        goto out;
    }

    if (crt_der && crt_der_size) {
        crt_der_buf = malloc(size);
        if (!crt_der_buf) {
            ret = MBEDTLS_ERR_X509_ALLOC_FAILED;
            goto out;
        }

        /* note that mbedtls_x509write_crt_der() wrote data at the end of the output_buf */
        memcpy(crt_der_buf, output_buf + output_buf_size - size, size);
        *crt_der      = crt_der_buf;
        *crt_der_size = size;
    }

    if (crt) {
        ret = mbedtls_x509_crt_parse_der(crt, output_buf + output_buf_size - size, size);
        if (ret < 0)
            goto out;
    }

    ret = 0;
out:
    if (ret < 0) {
        free(crt_der_buf);
    }
    mbedtls_x509write_crt_free(&writecrt);
    mbedtls_entropy_free(&entropy);
    mbedtls_ctr_drbg_free(&ctr_drbg);
    free(output_buf);
    return ret;
}

int ra_tls_create_key_and_crt(mbedtls_pk_context* key, mbedtls_x509_crt* crt) {
    return create_key_and_crt(key, crt, NULL, NULL);
}

int ra_tls_create_key_and_crt_der(uint8_t** der_key, size_t* der_key_size, uint8_t** der_crt,
                                  size_t* der_crt_size) {
    int ret;

    if (!der_key || !der_key_size || !der_crt || !der_crt_size)
        return -EINVAL;

    mbedtls_pk_context key;
    mbedtls_pk_init(&key);

    uint8_t* der_key_buf   = NULL;
    uint8_t* output_buf    = NULL;
    size_t output_buf_size = 4096; /* enough for any public key in DER format */

    output_buf = malloc(output_buf_size);
    if (!output_buf) {
        ret = MBEDTLS_ERR_X509_ALLOC_FAILED;
        goto out;
    }

    ret = create_key_and_crt(&key, NULL, der_crt, der_crt_size);
    if (ret < 0) {
        goto out;
    }

    /* populate der_key; note that der_crt was already populated */
    int size = mbedtls_pk_write_key_der(&key, output_buf, output_buf_size);
    if (size < 0) {
        ret = size;
        goto out;
    }

    der_key_buf = malloc(size);
    if (!der_key_buf) {
        ret = MBEDTLS_ERR_X509_ALLOC_FAILED;
        goto out;
    }

    /* note that mbedtls_pk_write_key_der() wrote data at the end of the output_buf */
    memcpy(der_key_buf, output_buf + output_buf_size - size, size);
    *der_key      = der_key_buf;
    *der_key_size = size;

    ret = 0;
out:
    if (ret < 0) {
        free(der_key_buf);
    }
    mbedtls_pk_free(&key);
    free(output_buf);
    return ret;
}
