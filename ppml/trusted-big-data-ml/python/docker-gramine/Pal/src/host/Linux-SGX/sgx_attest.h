/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2017, Texas A&M University */

#ifndef SGX_ATTEST_H
#define SGX_ATTEST_H

#include <stdbool.h>
#include <stdint.h>

#include "sgx_arch.h"

#pragma pack(push, 1)

/* different attestation key algorithms */
typedef enum {
    SGX_QL_ALG_EPID       = 0, /* EPID 2.0 - Anonymous */
    SGX_QL_ALG_RESERVED_1 = 1, /* reserved */
    SGX_QL_ALG_ECDSA_P256 = 2, /* ECDSA-256-with-P-256 curve, non-anonymous */
    SGX_QL_ALG_ECDSA_P384 = 3, /* ECDSA-384-with-P-384 curve, non-anonymous */
    SGX_QL_ALG_CNT        = 4,
} sgx_ql_attestation_algorithm_id_t;

/* generic attestation key format */
typedef struct _att_key_id_t {
    uint8_t     att_key_id[256];
} sgx_att_key_id_t;

/* single DCAP attestation key, contains both QE identity and the attestation algorithm ID */
typedef struct _sgx_ql_att_key_id_t {
    uint16_t    id;                   /* structure ID */
    uint16_t    version;              /* structure version */
    uint16_t    mrsigner_length;      /* number of valid bytes in MRSIGNER */
    uint8_t     mrsigner[48];         /* SHA256 or SHA384 hash of public key that signed QE */
    uint32_t    prod_id;              /* legacy Product ID of QE */
    uint8_t     extended_prod_id[16]; /* extended Product ID of QE (all 0's for legacy) */
    uint8_t     config_id[64];        /* Config ID of QE */
    uint8_t     family_id[16];        /* Family ID of QE */
    uint32_t    algorithm_id;         /* Identity of the attestation key algorithm */
} sgx_ql_att_key_id_t;

typedef uint8_t sgx_epid_group_id_t[4];

typedef struct _sgx_basename_t {
    uint8_t name[32];
} sgx_basename_t;

typedef struct _sgx_quote_body_t {
    uint16_t version;
    uint16_t sign_type;
    sgx_epid_group_id_t epid_group_id;
    sgx_isv_svn_t qe_svn;
    sgx_isv_svn_t pce_svn;
    uint32_t xeid;
    sgx_basename_t basename;
    sgx_report_body_t report_body;
} sgx_quote_body_t;

typedef struct _sgx_quote_t {
    sgx_quote_body_t body;
    uint32_t signature_size;
    uint8_t signature[];
} sgx_quote_t;

typedef uint8_t sgx_spid_t[16];
typedef uint8_t sgx_quote_nonce_t[16];

enum {
    SGX_UNLINKABLE_SIGNATURE,
    SGX_LINKABLE_SIGNATURE
};

/* EPID SGX quotes are ~1K in size, DCAP SGX quotes ~4K, overapproximate to 8K */
#define SGX_QUOTE_MAX_SIZE 8192

/*!
 * \brief Obtain SGX Quote from the Quoting Enclave (communicate via AESM).
 *
 * First create enclave report (sgx_report_t) with target info of the Quoting Enclave, and
 * then call out of the enclave to request the corresponding Quote from the Quoting Enclave.
 * Communication is done via AESM service, in the form of protobuf request/response messages.
 *
 * \param[in]  spid         Software provider ID (SPID); if NULL then DCAP/ECDSA is used.
 * \param[in]  nonce        16B nonce to be included in the quote for freshness; ignored if
 *                          DCAP/ECDSA is used.
 * \param[in]  report_data  64B bytestring to be included in the report and the quote.
 * \param[in]  linkable     Quote type (linkable vs unlinkable); ignored if DCAP/ECDSA is used.
 * \param[out] quote        Quote returned by the Quoting Enclave (allocated via malloc() in this
 *                          function; the caller gets the ownership of the quote).
 * \param[out] quote_len    Length of the quote returned by the Quoting Enclave.
 * \return                  0 on success, negative PAL error code otherwise.
 */
int sgx_get_quote(const sgx_spid_t* spid, const sgx_quote_nonce_t* nonce,
                  const sgx_report_data_t* report_data, bool linkable, char** quote,
                  size_t* quote_len);

#pragma pack(pop)

#endif /* SGX_ATTEST_H */
