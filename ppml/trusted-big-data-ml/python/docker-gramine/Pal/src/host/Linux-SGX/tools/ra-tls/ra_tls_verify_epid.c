/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Labs */

/*!
 * \file
 *
 * This file contains the implementation of verification callbacks for TLS libraries. The callbacks
 * verify the correctness of a self-signed RA-TLS certificate with an SGX quote embedded in it. The
 * callbacks access Intel Attestation Service (IAS) for EPID-based attestation as part of the
 * verification process. A callback ra_tls_verify_callback() can be used directly in mbedTLS, and
 * a more generic version ra_tls_verify_callback_der() should be used for other TLS libraries.
 *
 * This file is part of the RA-TLS verification library which is typically linked into client
 * applications. This library is *not* thread-safe.
 */

#define _GNU_SOURCE
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mbedtls/x509_crt.h>

#include "attestation.h"
#include "ias.h"
#include "ra_tls.h"
#include "sgx_arch.h"
#include "sgx_attest.h"
#include "util.h"

extern verify_measurements_cb_t g_verify_measurements_cb;

/** Default base URL for IAS API endpoints. Remove "/dev" for production environment. */
#define IAS_URL_BASE "https://api.trustedservices.intel.com/sgx/dev"

/** Default URL for IAS "verify attestation evidence" API endpoint. */
#define IAS_URL_REPORT IAS_URL_BASE "/attestation/v4/report"

/** Default URL for IAS "Retrieve SigRL" API endpoint. EPID group id is added at the end. */
#define IAS_URL_SIGRL IAS_URL_BASE "/attestation/v4/sigrl"

static char* g_api_key    = NULL;
static char* g_report_url = NULL;
static char* g_sigrl_url  = NULL;

static int init_from_env(char** ptr, const char* env_name, const char* default_val) {
    assert(ptr == &g_api_key || ptr == &g_report_url || ptr == &g_sigrl_url);

    if (*ptr) {
        /* already initialized */
        return 0;
    }

    char* env_val = getenv(env_name);
    if (!env_val) {
        if (!default_val)
            return MBEDTLS_ERR_X509_BAD_INPUT_DATA;

        *ptr = strdup(default_val);
        if (!*ptr)
            return MBEDTLS_ERR_X509_ALLOC_FAILED;

        return 0;
    }

    size_t env_val_size = strlen(env_val) + 1;
    *ptr = malloc(env_val_size);
    if (!*ptr)
        return MBEDTLS_ERR_X509_ALLOC_FAILED;

    memcpy(*ptr, env_val, env_val_size);
    return 0;

}

static int generate_nonce(char* buf, size_t size) {
    if (size != IAS_REQUEST_NONCE_LEN + 1) {
        return MBEDTLS_ERR_X509_BAD_INPUT_DATA;
    }

    char random_data[IAS_REQUEST_NONCE_LEN / 2];

    FILE* f = fopen("/dev/urandom", "rb");
    if (!f)
        return MBEDTLS_ERR_X509_FILE_IO_ERROR;

    size_t nmemb = fread(&random_data, sizeof(random_data), 1, f);
    if (nmemb != 1) {
        fclose(f);
        return MBEDTLS_ERR_X509_FILE_IO_ERROR;
    }

    fclose(f);

    if (hexdump_mem_to_buffer(&random_data, sizeof(random_data), buf, size) < 0) {
        return MBEDTLS_ERR_X509_BAD_INPUT_DATA;
    }
    return 0;
}

static int getenv_ias_pub_key_pem(char** ias_pub_key_pem) {
    char* tmp = getenv(RA_TLS_IAS_PUB_KEY_PEM);
    if (!tmp) {
        /* return as NULL, and then a hard-coded public key of IAS is used */
        *ias_pub_key_pem = NULL;
        return 0;
    }

    tmp = strdup(tmp);
    if (!tmp)
        return MBEDTLS_ERR_X509_ALLOC_FAILED;

    *ias_pub_key_pem = tmp;
    return 0;
}

int ra_tls_verify_callback(void* data, mbedtls_x509_crt* crt, int depth, uint32_t* flags) {
    (void)data;

    int ret;
    struct ias_context_t* ias = NULL;
    char* ias_pub_key_pem     = NULL;

    char* report_data   = NULL;
    char* sig_data      = NULL;
    char* cert_data     = NULL;

    size_t report_data_size   = 0;
    size_t sig_data_size      = 0;
    size_t cert_data_size     = 0;

    uint8_t* quote_from_ias    = NULL;
    size_t quote_from_ias_size = 0;

    if (depth != 0) {
        /* the cert chain in RA-TLS consists of single self-signed cert, so we expect depth 0 */
        return MBEDTLS_ERR_X509_INVALID_FORMAT;
    }

    if (flags) {
        /* mbedTLS sets flags to signal that the cert is not to be trusted (e.g., it is not
         * correctly signed by a trusted CA; since RA-TLS uses self-signed certs, we don't care
         * what mbedTLS thinks and ignore internal cert verification logic of mbedTLS */
        *flags = 0;
    }

    ret = init_from_env(&g_api_key, RA_TLS_EPID_API_KEY, /*default_val=*/NULL);
    if (ret < 0)
        goto out;

    ret = init_from_env(&g_report_url, RA_TLS_IAS_REPORT_URL, IAS_URL_REPORT);
    if (ret < 0)
        goto out;

    ret = init_from_env(&g_sigrl_url, RA_TLS_IAS_SIGRL_URL, IAS_URL_SIGRL);
    if (ret < 0)
        goto out;

    /* extract SGX quote from "quote" OID extension from crt */
    sgx_quote_t* quote;
    size_t quote_size;
    ret = find_oid(crt->v3_ext.p, crt->v3_ext.len, quote_oid, quote_oid_len, (uint8_t**)&quote,
                   &quote_size);
    if (ret < 0)
        goto out;

    if (quote_size < sizeof(*quote)) {
        ret = MBEDTLS_ERR_X509_INVALID_EXTENSIONS;
        goto out;
    }

    /* compare public key's hash from cert against quote's report_data */
    ret = cmp_crt_pk_against_quote_report_data(crt, quote);
    if (ret < 0)
        goto out;

    /* initialize the IAS context, send the quote to the IAS and receive IAS attestation report */
    ias = ias_init(g_api_key, g_report_url, g_sigrl_url);
    if (!ias) {
        ret = MBEDTLS_ERR_X509_FATAL_ERROR;
        goto out;
    }

    char nonce[IAS_REQUEST_NONCE_LEN + 1];
    ret = generate_nonce(nonce, sizeof(nonce));
    if (ret < 0)
        goto out;

    ret = ias_verify_quote_raw(ias, quote, quote_size, nonce, &report_data, &report_data_size,
                               &sig_data, &sig_data_size, &cert_data, &cert_data_size);
    if (ret < 0) {
        ret = MBEDTLS_ERR_X509_FATAL_ERROR;
        goto out;
    }

    if (!report_data || !report_data_size || !sig_data || !sig_data_size) {
        /* received IAS attestation report doesn't contain report and/or signature */
        ret = MBEDTLS_ERR_X509_INVALID_FORMAT;
        goto out;
    }

    /* verify_ias_report_extract_quote() expects report_data and sig_data without the ending '\0' */
    assert(report_data[report_data_size - 1] == '\0');
    report_data_size--;
    assert(sig_data[sig_data_size - 1] == '\0');
    sig_data_size--;

    /* prepare user-supplied verification parameters "allow outdated TCB"/"allow debug enclave" */
    bool allow_outdated_tcb  = getenv_allow_outdated_tcb();
    bool allow_debug_enclave = getenv_allow_debug_enclave();

    ret = getenv_ias_pub_key_pem(&ias_pub_key_pem);
    if (ret < 0)
        goto out;

    /* below function verifies `isvEnclaveQuoteStatus` returned in the IAS attestation report; it
     * fails if the SGX quote is invalid or if the EPID group/private key/signature is revoked (see
     * https://www.intel.com/content/dam/develop/public/us/en/documents/sgx-attestation-api-spec.pdf
     * for details) */
    ret = verify_ias_report_extract_quote((uint8_t*)report_data, report_data_size,
                                          (uint8_t*)sig_data, sig_data_size,
                                          allow_outdated_tcb, nonce,
                                          ias_pub_key_pem, &quote_from_ias, &quote_from_ias_size);
    if (ret < 0) {
        ret = MBEDTLS_ERR_X509_CERT_VERIFY_FAILED;
        goto out;
    }

    /* verify that obtained SGX quote (extracted from IAS report) has reasonable size */
    if (quote_from_ias_size < sizeof(sgx_quote_body_t) ||
            quote_from_ias_size > SGX_QUOTE_MAX_SIZE) {
        ret = MBEDTLS_ERR_X509_CERT_VERIFY_FAILED;
        goto out;
    }

    sgx_quote_body_t* quote_body = (sgx_quote_body_t*)quote_from_ias;

    /* verify enclave attributes from the SGX quote body */
    ret = verify_quote_body_enclave_attributes(quote_body, allow_debug_enclave);
    if (ret < 0) {
        ret = MBEDTLS_ERR_X509_CERT_VERIFY_FAILED;
        goto out;
    }

    /* verify other relevant enclave information from the SGX quote */
    if (g_verify_measurements_cb) {
        /* use user-supplied callback to verify measurements */
        ret = g_verify_measurements_cb((const char*)&quote_body->report_body.mr_enclave,
                                       (const char*)&quote_body->report_body.mr_signer,
                                       (const char*)&quote_body->report_body.isv_prod_id,
                                       (const char*)&quote_body->report_body.isv_svn);
    } else {
        /* use default logic to verify measurements */
        ret = verify_quote_body_against_envvar_measurements(quote_body);
    }
    if (ret < 0) {
        ret = MBEDTLS_ERR_X509_CERT_VERIFY_FAILED;
        goto out;
    }

    ret = 0;
out:
    if (ias)
        ias_cleanup(ias);

    free(ias_pub_key_pem);
    free(quote_from_ias);
    free(report_data);
    free(sig_data);
    free(cert_data);

    return ret;
}
