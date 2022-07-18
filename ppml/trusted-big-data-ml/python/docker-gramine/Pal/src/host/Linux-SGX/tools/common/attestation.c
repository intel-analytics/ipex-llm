/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Invisible Things Lab
 *                    Rafal Wojdyla <omeg@invisiblethingslab.com>
 */

#define _GNU_SOURCE

#include "attestation.h"

#include <assert.h>
#include <stdalign.h>
#include <stdlib.h>
#include <string.h>

#include <mbedtls/base64.h>
#include <mbedtls/md.h>
#include <mbedtls/pk.h>

#ifdef HAVE_INTERNAL_CJSON
/* here we -I the cJSON's repo root, which directly contains the header */
#include <cJSON.h>
#else
#include <cjson/cJSON.h>
#endif

#include "sgx_arch.h"
#include "sgx_attest.h"
#include "util.h"

/*! This is the public RSA key of the IAS (PEM). It's used to verify IAS report signatures. */
const char* g_ias_public_key_pem =
    "-----BEGIN PUBLIC KEY-----\n"
    "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAqXot4OZuphR8nudFrAFi\n"
    "aGxxkgma/Es/BA+tbeCTUR106AL1ENcWA4FX3K+E9BBL0/7X5rj5nIgX/R/1ubhk\n"
    "KWw9gfqPG3KeAtIdcv/uTO1yXv50vqaPvE1CRChvzdS/ZEBqQ5oVvLTPZ3VEicQj\n"
    "lytKgN9cLnxbwtuvLUK7eyRPfJW/ksddOzP8VBBniolYnRCD2jrMRZ8nBM2ZWYwn\n"
    "XnwYeOAHV+W9tOhAImwRwKF/95yAsVwd21ryHMJBcGH70qLagZ7Ttyt++qO/6+KA\n"
    "XJuKwZqjRlEtSEz8gZQeFfVYgcwSfo96oSMAzVr7V0L6HSDLRnpb6xxmbPdqNol4\n"
    "tQIDAQAB\n"
    "-----END PUBLIC KEY-----\n";

// Copied from Gramine's api.h.
// TODO: Remove after Gramine's C utils get refactored into a separate module/header (we can't
// include it here, because these SGX tools should be independent of Gramine).
#define IS_ALIGNED_POW2(val, alignment)     (((val) & ((alignment) - 1)) == 0)
#define IS_ALIGNED_PTR_POW2(val, alignment) IS_ALIGNED_POW2((uintptr_t)(val), alignment)

// TODO: decode some known values (flags etc)
static void display_report_body(const sgx_report_body_t* body) {
    INFO(" cpu_svn          : ");
    HEXDUMP(body->cpu_svn);
    INFO(" misc_select      : ");
    HEXDUMP(body->misc_select);
    INFO(" reserved1        : ");
    HEXDUMP(body->reserved1);
    INFO(" isv_ext_prod_id  : ");
    HEXDUMP(body->isv_ext_prod_id);
    INFO(" attributes.flags : ");
    HEXDUMP(body->attributes.flags);
    INFO(" attributes.xfrm  : ");
    HEXDUMP(body->attributes.xfrm);
    INFO(" mr_enclave       : ");
    HEXDUMP(body->mr_enclave);
    INFO(" reserved2        : ");
    HEXDUMP(body->reserved2);
    INFO(" mr_signer        : ");
    HEXDUMP(body->mr_signer);
    INFO(" reserved3        : ");
    HEXDUMP(body->reserved3);
    INFO(" config_id        : ");
    HEXDUMP(body->config_id);
    INFO(" isv_prod_id      : ");
    HEXDUMP(body->isv_prod_id);
    INFO(" isv_svn          : ");
    HEXDUMP(body->isv_svn);
    INFO(" config_svn       : ");
    HEXDUMP(body->config_svn);
    INFO(" reserved4        : ");
    HEXDUMP(body->reserved4);
    INFO(" isv_family_id    : ");
    HEXDUMP(body->isv_family_id);
    INFO(" report_data      : ");
    HEXDUMP(body->report_data);
}

static void display_quote_body(const sgx_quote_body_t* quote_body) {
    INFO(" version          : ");
    HEXDUMP(quote_body->version);
    INFO(" sign_type        : ");
    HEXDUMP(quote_body->sign_type);
    INFO(" epid_group_id    : ");
    HEXDUMP(quote_body->epid_group_id);
    INFO(" qe_svn           : ");
    HEXDUMP(quote_body->qe_svn);
    INFO(" pce_svn          : ");
    HEXDUMP(quote_body->pce_svn);
    INFO(" xeid             : ");
    HEXDUMP(quote_body->xeid);
    INFO(" basename         : ");
    HEXDUMP(quote_body->basename);
}

void display_quote(const void* quote_data, size_t quote_size) {
    if (quote_size < sizeof(sgx_quote_body_t)) {
        ERROR("Quote size too small\n");
        return;
    }

    assert(IS_ALIGNED_PTR_POW2(quote_data, alignof(sgx_quote_t)));
    sgx_quote_t* quote = (sgx_quote_t*)quote_data;
    INFO("quote_body        :\n");
    display_quote_body(&quote->body);
    INFO("report_body       :\n");
    display_report_body(&quote->body.report_body);

    /* Quotes from IAS reports are missing signature fields. So display signature and signature_size
       fields only for DCAP-based quotes */
    if (quote_size >= sizeof(sgx_quote_body_t) + sizeof(quote->signature_size)) {
        INFO("signature_size    : %d (0x%x)\n", quote->signature_size, quote->signature_size);
    }

    if (quote_size >= sizeof(sgx_quote_t) + quote->signature_size) {
        INFO("signature         : ");
        hexdump_mem(&quote->signature, quote->signature_size);
        INFO("\n");
    }
}

int verify_ias_report_extract_quote(const uint8_t* ias_report, size_t ias_report_size,
                                    uint8_t* ias_sig_b64, size_t ias_sig_b64_size,
                                    bool allow_outdated_tcb, const char* nonce,
                                    const char* ias_pub_key_pem, uint8_t** out_quote,
                                    size_t* out_quote_size) {
    mbedtls_pk_context ias_pub_key;
    int ret = -1;
    uint8_t* ias_sig = NULL;
    uint8_t* report_quote = NULL;
    cJSON* json = NULL;

    // Load the IAS public key
    mbedtls_pk_init(&ias_pub_key);

    if (!ias_pub_key_pem)
        ias_pub_key_pem = g_ias_public_key_pem;

    ret = mbedtls_pk_parse_public_key(&ias_pub_key, (const unsigned char*)ias_pub_key_pem,
                                      strlen(ias_pub_key_pem) + 1);
    if (ret != 0) {
        ERROR("Failed to parse IAS public key: %d\n", ret);
        goto out;
    }

    DBG("IAS key: %s, %zu bits\n", mbedtls_pk_get_name(&ias_pub_key),
        mbedtls_pk_get_bitlen(&ias_pub_key));

    if (!mbedtls_pk_can_do(&ias_pub_key, MBEDTLS_PK_RSA)) {
        ret = -1;
        ERROR("IAS public key is not an RSA key\n");
        goto out;
    }

    size_t ias_sig_size = 0;

    // Drop trailing newlines
    if (ias_sig_b64_size == 0) {
        ret = -1;
        ERROR("Invalid signature size\n");
        goto out;
    }

    while (ias_sig_b64[ias_sig_b64_size - 1] == '\n' || ias_sig_b64[ias_sig_b64_size - 1] == '\r')
        ias_sig_b64[--ias_sig_b64_size] = '\0';

    ret = mbedtls_base64_decode(/*dest=*/NULL, /*dlen=*/0, &ias_sig_size, ias_sig_b64,
                                ias_sig_b64_size);
    if (ret != MBEDTLS_ERR_BASE64_BUFFER_TOO_SMALL) {
        ERROR("Failed to get size for base64 decoding of IAS signature\n");
        goto out;
    }

    ias_sig = malloc(ias_sig_size);
    if (!ias_sig) {
        ret = -1;
        ERROR("No memory\n");
        goto out;
    }

    ret = mbedtls_base64_decode(ias_sig, ias_sig_size, &ias_sig_size, ias_sig_b64,
                                ias_sig_b64_size);
    if (ret < 0) {
        ERROR("Failed to base64 decode IAS signature\n");
        goto out;
    }

    DBG("Decoded IAS signature size: %zu bytes\n", ias_sig_size);

    // Calculate report hash
    uint8_t report_hash[32];
    ret = mbedtls_md(mbedtls_md_info_from_type(MBEDTLS_MD_SHA256), (const unsigned char*)ias_report,
                     ias_report_size, report_hash);
    if (ret != 0) {
        ERROR("Failed to compute IAS report hash: %d\n", ret);
        goto out;
    }

    // Verify signature
    ret = mbedtls_pk_verify(&ias_pub_key, MBEDTLS_MD_SHA256, (const unsigned char*)report_hash,
                            sizeof(report_hash), ias_sig, ias_sig_size);
    if (ret != 0) {
        ERROR("Failed to verify IAS report signature: %d\n", ret);
        goto out;
    }

    INFO("IAS report: signature verified correctly\n");

    // Check quote status
    ret = -1;
    json = cJSON_Parse((const char*)ias_report);
    if (!json) {
        ERROR("Failed to parse IAS report\n");
        goto out;
    }

    cJSON* node = cJSON_GetObjectItem(json, "isvEnclaveQuoteStatus");
    if (!node) {
        ERROR("IAS report: failed to read quote status\n");
        goto out;
    }

    if (node->type != cJSON_String) {
        ERROR("IAS report: quote status is not a string\n");
        goto out;
    }

    if (strcmp("OK", node->valuestring) == 0) {
        ret = 0;
        INFO("IAS report: quote status OK\n");
    } else if (allow_outdated_tcb && (
               strcmp("GROUP_OUT_OF_DATE", node->valuestring) == 0
            || strcmp("CONFIGURATION_NEEDED", node->valuestring) == 0
            || strcmp("SW_HARDENING_NEEDED", node->valuestring) == 0
            || strcmp("CONFIGURATION_AND_SW_HARDENING_NEEDED", node->valuestring) == 0
            )) {
        ret = 0;
        INFO("IAS report: allowing quote status %s\n", node->valuestring);

        cJSON* url_node = cJSON_GetObjectItem(json, "advisoryURL");
        if (url_node && url_node->type == cJSON_String)
            INFO("            [ advisory URL: %s ]\n", url_node->valuestring);

        cJSON* ids_node = cJSON_GetObjectItem(json, "advisoryIDs");
        if (ids_node && ids_node->type == cJSON_Array) {
            char* ids_str = cJSON_Print(ids_node);
            if (!ids_str) {
                ERROR("IAS report: out-of-memory during reading advisoryIDs\n");
                ret = -1;
                goto out;
            }
            INFO("            [ advisory IDs: %s ]\n", ids_str);
            free(ids_str);
        }
    }

    if (ret != 0) {
        ERROR("IAS report: quote status is not OK (%s)\n", node->valuestring);
        goto out;
    }

    ret = -1;
    // Verify nonce if required
    if (nonce) {
        cJSON* node = cJSON_GetObjectItem(json, "nonce");
        if (!node) {
            ERROR("IAS report: failed to read nonce\n");
            goto out;
        }

        if (node->type != cJSON_String) {
            ERROR("IAS report: nonce is not a string\n");
            goto out;
        }

        if (strcmp(nonce, node->valuestring) != 0) {
            ERROR("IAS report: invalid nonce '%s', expected '%s'\n", node->valuestring, nonce);
            goto out;
        }

        DBG("IAS report: nonce OK\n");
    }

    // Extract quote from the report
    node = cJSON_GetObjectItem(json, "isvEnclaveQuoteBody");
    if (!node) {
        ERROR("IAS report: failed to get quote\n");
        goto out;
    }

    if (node->type != cJSON_String) {
        ERROR("IAS report: quote is not a string\n");
        goto out;
    }

    size_t quote_size = 0;
    ret = mbedtls_base64_decode(/*dest=*/NULL, /*dlen=*/0, &quote_size, (uint8_t*)node->valuestring,
                                strlen(node->valuestring));
    if (ret != MBEDTLS_ERR_BASE64_BUFFER_TOO_SMALL) {
        ERROR("IAS report: failed to get size for base64 decoding of report quote\n");
        goto out;
    }

    report_quote = malloc(quote_size);
    if (!report_quote) {
        ret = -1;
        ERROR("No memory\n");
        goto out;
    }

    ret = mbedtls_base64_decode(report_quote, quote_size, &quote_size, (uint8_t*)node->valuestring,
                                strlen(node->valuestring));
    if (ret < 0) {
        ERROR("IAS report: failed to decode report quote\n");
        goto out;
    }

    DBG("IAS report: quote decoded, size %zu bytes\n", quote_size);
    *out_quote      = report_quote;
    *out_quote_size = quote_size;
    ret = 0;
out:
    if (ret) {
        free(report_quote);
    }
    if (json)
        cJSON_Delete(json);
    mbedtls_pk_free(&ias_pub_key);
    free(ias_sig);
    return ret ? -1 : 0;
}

int verify_quote_body(const sgx_quote_body_t* quote_body, const char* mr_signer,
                      const char* mr_enclave, const char* isv_prod_id, const char* isv_svn,
                      const char* report_data, bool expected_as_str) {
    int ret = -1;

    sgx_quote_body_t* body = (sgx_quote_body_t*)quote_body;

    if (get_verbose())
        display_quote_body(body);

    sgx_report_body_t* report_body = &body->report_body;

    sgx_measurement_t expected_mr;
    if (mr_signer) {
        if (expected_as_str) {
            if (parse_hex(mr_signer, &expected_mr, sizeof(expected_mr), NULL) != 0)
                goto out;
        } else {
            memcpy(&expected_mr, mr_signer, sizeof(expected_mr));
        }

        if (memcmp(&report_body->mr_signer, &expected_mr, sizeof(expected_mr)) != 0) {
            ERROR("Quote: mr_signer doesn't match the expected value\n");
            if (get_verbose()) {
                ERROR("Quote mr_signer:\n");
                HEXDUMP(report_body->mr_signer);
                ERROR("Expected mr_signer:\n");
                HEXDUMP(expected_mr);
            }
            goto out;
        }

        DBG("Quote: mr_signer OK\n");
    }

    if (mr_enclave) {
        if (expected_as_str) {
            if (parse_hex(mr_enclave, &expected_mr, sizeof(expected_mr), NULL) != 0)
                goto out;
        } else {
            memcpy(&expected_mr, mr_enclave, sizeof(expected_mr));
        }

        if (memcmp(&report_body->mr_enclave, &expected_mr, sizeof(expected_mr)) != 0) {
            ERROR("Quote: mr_enclave doesn't match the expected value\n");
            if (get_verbose()) {
                ERROR("Quote mr_enclave:\n");
                HEXDUMP(report_body->mr_enclave);
                ERROR("Expected mr_enclave:\n");
                HEXDUMP(expected_mr);
            }
            goto out;
        }

        DBG("Quote: mr_enclave OK\n");
    }

    // Product ID must match, security version must be greater or equal
    if (isv_prod_id) {
        sgx_prod_id_t prod_id;

        if (expected_as_str) {
            prod_id = strtoul(isv_prod_id, NULL, 10);
        } else {
            memcpy(&prod_id, isv_prod_id, sizeof(prod_id));
        }

        if (report_body->isv_prod_id != prod_id) {
            ERROR("Quote: invalid isv_prod_id (%u, expected %u)\n", report_body->isv_prod_id,
                  prod_id);
            goto out;
        }

        DBG("Quote: isv_prod_id OK\n");
    }

    if (isv_svn) {
        sgx_isv_svn_t svn;

        if (expected_as_str) {
            svn = strtoul(isv_svn, NULL, 10);
        } else {
            memcpy(&svn, isv_svn, sizeof(svn));
        }

        if (report_body->isv_svn < svn) {
            ERROR("Quote: invalid isv_svn (%u < expected %u)\n", report_body->isv_svn, svn);
            goto out;
        }

        DBG("Quote: isv_svn OK\n");
    }

    if (report_data) {
        sgx_report_data_t rd;

        if (expected_as_str) {
            if (parse_hex(report_data, &rd, sizeof(rd), NULL) != 0)
                goto out;
        } else {
            memcpy(&rd, report_data, sizeof(rd));
        }

        if (memcmp(&report_body->report_data, &rd, sizeof(rd)) != 0) {
            ERROR("Quote: report_data doesn't match the expected value\n");
            if (get_verbose()) {
                ERROR("Quote report_data:\n");
                HEXDUMP(report_body->report_data);
                ERROR("Expected report_data:\n");
                HEXDUMP(rd);
            }
            goto out;
        }

        DBG("Quote: report_data OK\n");
    }

    ret = 0;
    // TODO: KSS support (isv_ext_prod_id, config_id, config_svn, isv_family_id)
out:
    return ret;
}

int verify_quote_body_enclave_attributes(sgx_quote_body_t* quote_body, bool allow_debug_enclave) {
    if (!allow_debug_enclave && (quote_body->report_body.attributes.flags & SGX_FLAGS_DEBUG)) {
        ERROR("Quote: DEBUG bit in enclave attributes is set\n");
        return -1;
    }

    /* sanity check: enclave must be initialized */
    if (!(quote_body->report_body.attributes.flags & SGX_FLAGS_INITIALIZED)) {
        ERROR("Quote: INIT bit in enclave attributes is not set\n");
        return -1;
    }

    /* sanity check: enclave must not have provision/EINIT token key */
    if ((quote_body->report_body.attributes.flags & SGX_FLAGS_PROVISION_KEY) ||
            (quote_body->report_body.attributes.flags & SGX_FLAGS_LICENSE_KEY)) {
        ERROR("Quote: PROVISION_KEY or LICENSE_KEY bit in enclave attributes is set\n");
        return -1;
    }

    /* currently only support 64-bit enclaves */
    if (!(quote_body->report_body.attributes.flags & SGX_FLAGS_MODE64BIT)) {
        ERROR("Quote: MODE64 bit in enclave attributes is not set\n");
        return -1;
    }

    DBG("Quote: enclave attributes OK\n");
    return 0;
}
