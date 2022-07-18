/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Labs */

/*!
 * \file
 *
 * This file contains the implementation of verification callbacks for TLS libraries. The callbacks
 * verify the correctness of a self-signed RA-TLS certificate with an SGX quote embedded in it. The
 * callbacks call into the `libsgx_dcap_quoteverify` DCAP library for ECDSA-based verification. A
 * callback ra_tls_verify_callback() can be used directly in mbedTLS, and a more generic version
 * ra_tls_verify_callback_der() should be used for other TLS libraries.
 *
 * This file is part of the RA-TLS verification library which is typically linked into client
 * applications. This library is *not* thread-safe.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <mbedtls/pk.h>
#include <mbedtls/sha256.h>
#include <mbedtls/x509_crt.h>

#include "attestation.h"
#include "ra_tls.h"
#include "sgx_arch.h"
#include "sgx_attest.h"
#include "util.h"

extern verify_measurements_cb_t g_verify_measurements_cb;

/* we cannot include libsgx_dcap_verify headers because they conflict with Gramine SGX headers,
 * so we declare the used types and functions below */

/* QL stands for Quoting Library; QV stands for Quote Verification */
#define SGX_QL_QV_MK_ERROR(x) (0x0000A000 | (x))
typedef enum _sgx_ql_qv_result_t {
    /* quote verification passed and is at the latest TCB level */
    SGX_QL_QV_RESULT_OK = 0x0000,
    /* quote verification passed and the platform is patched to the latest TCB level but additional
     * configuration of the SGX platform may be needed */
    SGX_QL_QV_RESULT_CONFIG_NEEDED = SGX_QL_QV_MK_ERROR(0x0001),
    /* quote is good but TCB level of the platform is out of date; platform needs patching to be at
     * the latest TCB level */
    SGX_QL_QV_RESULT_OUT_OF_DATE = SGX_QL_QV_MK_ERROR(0x0002),
    /* quote is good but the TCB level of the platform is out of date and additional configuration
     * of the SGX platform at its current patching level may be needed; platform needs patching to
     * be at the latest TCB level */
    SGX_QL_QV_RESULT_OUT_OF_DATE_CONFIG_NEEDED = SGX_QL_QV_MK_ERROR(0x0003),
    /* signature over the application report is invalid */
    SGX_QL_QV_RESULT_INVALID_SIGNATURE = SGX_QL_QV_MK_ERROR(0x0004),
    /* attestation key or platform has been revoked */
    SGX_QL_QV_RESULT_REVOKED = SGX_QL_QV_MK_ERROR(0x0005),
    /* quote verification failed due to an error in one of the input */
    SGX_QL_QV_RESULT_UNSPECIFIED = SGX_QL_QV_MK_ERROR(0x0006),
    /* TCB level of the platform is up to date, but SGX SW hardening is needed */
    SGX_QL_QV_RESULT_SW_HARDENING_NEEDED = SGX_QL_QV_MK_ERROR(0x0007),
    /* TCB level of the platform is up to date, but additional configuration of the platform at its
     * current patching level may be needed; moreover, SGX SW hardening is also needed */
    SGX_QL_QV_RESULT_CONFIG_AND_SW_HARDENING_NEEDED = SGX_QL_QV_MK_ERROR(0x0008),
} sgx_ql_qv_result_t;

int sgx_qv_get_quote_supplemental_data_size(uint32_t* p_data_size);
int sgx_qv_verify_quote(const uint8_t* p_quote, uint32_t quote_size, void* p_quote_collateral,
                        const time_t expiration_check_date,
                        uint32_t* p_collateral_expiration_status,
                        sgx_ql_qv_result_t* p_quote_verification_result, void* p_qve_report_info,
                        uint32_t supplemental_data_size, uint8_t* p_supplemental_data);

int ra_tls_verify_callback(void* data, mbedtls_x509_crt* crt, int depth, uint32_t* flags) {
    (void)data;

    int ret;

    uint8_t* supplemental_data      = NULL;
    uint32_t supplemental_data_size = 0;

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

    /* prepare user-supplied verification parameters "allow outdated TCB"/"allow debug enclave" */
    bool allow_outdated_tcb  = getenv_allow_outdated_tcb();
    bool allow_debug_enclave = getenv_allow_debug_enclave();

    /* call into libsgx_dcap_quoteverify to verify ECDSA/based SGX quote */
    ret = sgx_qv_get_quote_supplemental_data_size(&supplemental_data_size);
    if (ret) {
        ret = MBEDTLS_ERR_X509_FATAL_ERROR;
        goto out;
    }

    supplemental_data = (uint8_t*)malloc(supplemental_data_size);
    if (!supplemental_data) {
        ret = MBEDTLS_ERR_X509_ALLOC_FAILED;
        goto out;
    }

    time_t current_time = time(NULL);
    if (current_time == ((time_t)-1)) {
        ret = MBEDTLS_ERR_X509_FATAL_ERROR;
        goto out;
    }

    uint32_t collateral_expiration_status  = 1;
    sgx_ql_qv_result_t verification_result = SGX_QL_QV_RESULT_UNSPECIFIED;

    ret = sgx_qv_verify_quote((uint8_t*)quote, (uint32_t)quote_size, /*p_quote_collateral=*/NULL,
                              current_time, &collateral_expiration_status, &verification_result,
                              /*p_qve_report_info=*/NULL, supplemental_data_size,
                              supplemental_data);
    if (ret) {
        ret = MBEDTLS_ERR_X509_CERT_VERIFY_FAILED;
        goto out;
    }

    switch (verification_result) {
        case SGX_QL_QV_RESULT_OK:
            ret = 0;
            break;
        case SGX_QL_QV_RESULT_CONFIG_NEEDED:
        case SGX_QL_QV_RESULT_OUT_OF_DATE:
        case SGX_QL_QV_RESULT_OUT_OF_DATE_CONFIG_NEEDED:
        case SGX_QL_QV_RESULT_SW_HARDENING_NEEDED:
        case SGX_QL_QV_RESULT_CONFIG_AND_SW_HARDENING_NEEDED:
            ret = allow_outdated_tcb ? 0 : MBEDTLS_ERR_X509_CERT_VERIFY_FAILED;
            break;
        case SGX_QL_QV_RESULT_INVALID_SIGNATURE:
        case SGX_QL_QV_RESULT_REVOKED:
        case SGX_QL_QV_RESULT_UNSPECIFIED:
        default:
            ret = MBEDTLS_ERR_X509_CERT_VERIFY_FAILED;
            break;
    }

    sgx_quote_body_t* quote_body = &quote->body;

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
    free(supplemental_data);
    return ret;
}
