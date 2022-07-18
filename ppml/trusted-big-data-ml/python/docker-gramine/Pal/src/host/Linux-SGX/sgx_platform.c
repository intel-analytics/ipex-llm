/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2019, Texas A&M University.
 *               2020, Intel Labs.
 */

#include <asm/errno.h>
#include <linux/un.h>
#include <stdbool.h>

#include "aesm.pb-c.h"
#include "gsgx.h"
#include "linux_utils.h"
#include "sgx_attest.h"
#include "sgx_internal.h"
#include "sgx_log.h"

#define AESM_SOCKET_NAME_LEGACY "sgx_aesm_socket_base"
#define AESM_SOCKET_NAME_NEW    "/var/run/aesmd/aesm.socket"

/* hard-coded production attestation key of Intel reference QE (the only supported one) */
/* FIXME: should allow other attestation keys from non-Intel QEs */
static const sgx_ql_att_key_id_t g_default_ecdsa_p256_att_key_id = {
    .id               = 0,
    .version          = 0,
    .mrsigner_length  = 32,
    .mrsigner         = { 0x8c, 0x4f, 0x57, 0x75, 0xd7, 0x96, 0x50, 0x3e,
                          0x96, 0x13, 0x7f, 0x77, 0xc6, 0x8a, 0x82, 0x9a,
                          0x00, 0x56, 0xac, 0x8d, 0xed, 0x70, 0x14, 0x0b,
                          0x08, 0x1b, 0x09, 0x44, 0x90, 0xc5, 0x7b, 0xff,
                          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    .prod_id          = 1,
    .extended_prod_id = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    .config_id        = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    .family_id        = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    .algorithm_id     = SGX_QL_ALG_ECDSA_P256,
};

/*
 * Connect to the AESM service to interact with the architectural enclave. Must reconnect
 * for each request to the AESM service.
 */
static int connect_aesm_service(void) {
    int sock = DO_SYSCALL(socket, AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0)
        return sock;

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    (void)strcpy_static(addr.sun_path, "\0" AESM_SOCKET_NAME_LEGACY, sizeof(addr.sun_path));

    int ret = DO_SYSCALL(connect, sock, &addr, sizeof(addr));
    if (ret >= 0)
        return sock;
    if (ret != -ECONNREFUSED)
        goto err;

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    (void)strcpy_static(addr.sun_path, AESM_SOCKET_NAME_NEW, sizeof(addr.sun_path));

    ret = DO_SYSCALL(connect, sock, &addr, sizeof(addr));
    if (ret >= 0)
        return sock;

err:
    DO_SYSCALL(close, sock);
    log_error("Cannot connect to aesm_service (tried " AESM_SOCKET_NAME_LEGACY " and "
              AESM_SOCKET_NAME_NEW " UNIX sockets).\nPlease check its status! (`service aesmd "
              "status` on Ubuntu)");
    return ret;
}

/*
 * A wrapper for both creating a connection to the AESM service and submitting a request
 * to the service. Upon success, the function returns a response from the AESM service
 * back to the caller.
 */
static int request_aesm_service(Request* req, Response** res) {
    uint8_t* res_buf = NULL;
    int aesm_socket = connect_aesm_service();
    if (aesm_socket < 0)
        return aesm_socket;

    uint32_t req_len = (uint32_t)request__get_packed_size(req);
    uint8_t* req_buf = __alloca(req_len);
    request__pack(req, req_buf);

    int ret = write_all(aesm_socket, &req_len, sizeof(req_len));
    if (ret < 0)
        goto out;

    ret = write_all(aesm_socket, req_buf, req_len);
    if (ret < 0)
        goto out;

    uint32_t res_len;
    ret = read_all(aesm_socket, &res_len, sizeof(res_len));
    if (ret < 0)
        goto out;

    res_buf = malloc(res_len);
    if (!res_buf) {
        ret = -ENOMEM;
        goto out;
    }
    ret = read_all(aesm_socket, res_buf, res_len);
    if (ret < 0)
        goto out;

    *res = response__unpack(NULL, res_len, res_buf);
    ret = *res == NULL ? -EINVAL : 0;
out:
    free(res_buf);
    DO_SYSCALL(close, aesm_socket);
    if (ret < 0) {
        log_error("Cannot communicate with aesm_service (read/write returned error %d).\n"
                  "Please check its status! (`service aesmd status` on Ubuntu)", ret);
    }
    return ret;
}

int init_quoting_enclave_targetinfo(bool is_epid, sgx_target_info_t* qe_targetinfo) {
    int ret;

    Request req   = REQUEST__INIT;
    Response* res = NULL;

    if (is_epid) {
        Request__InitQuoteRequest initreq = REQUEST__INIT_QUOTE_REQUEST__INIT;
        req.initquotereq = &initreq;

        ret = request_aesm_service(&req, &res);
        if (ret < 0)
            return ret;

        ret = -EPERM;
        if (!res->initquoteres) {
            log_error("aesm_service returned wrong message");
            goto failed;
        }

        Response__InitQuoteResponse* r = res->initquoteres;
        if (r->errorcode != 0) {
            log_error("aesm_service returned error: %d", r->errorcode);
            goto failed;
        }

        if (r->targetinfo.len != sizeof(*qe_targetinfo)) {
            log_error("Quoting Enclave returned invalid target info");
            goto failed;
        }

        memcpy(qe_targetinfo, r->targetinfo.data, sizeof(*qe_targetinfo));
    } else {
        sgx_att_key_id_t default_att_key_id;
        memset(&default_att_key_id, 0, sizeof(default_att_key_id));
        memcpy(&default_att_key_id, &g_default_ecdsa_p256_att_key_id,
                sizeof(g_default_ecdsa_p256_att_key_id));

        Request__InitQuoteExRequest initexreq = REQUEST__INIT_QUOTE_EX_REQUEST__INIT;
        initexreq.has_att_key_id  = true;
        initexreq.att_key_id.data = (uint8_t*)&default_att_key_id;
        initexreq.att_key_id.len  = sizeof(default_att_key_id);
        initexreq.b_pub_key_id    = true;
        initexreq.has_buf_size    = true;
        initexreq.buf_size        = SGX_HASH_SIZE;
        req.initquoteexreq = &initexreq;

        ret = request_aesm_service(&req, &res);
        if (ret < 0)
            return ret;

        ret = -EPERM;
        if (!res->initquoteexres) {
            log_error("aesm_service returned wrong message");
            goto failed;
        }

        Response__InitQuoteExResponse* r = res->initquoteexres;
        if (r->errorcode != 0) {
            log_error("aesm_service returned error: %d", r->errorcode);
            goto failed;
        }

        if (r->target_info.len != sizeof(*qe_targetinfo)) {
            log_error("Quoting Enclave returned invalid target info");
            goto failed;
        }

        memcpy(qe_targetinfo, r->target_info.data, sizeof(*qe_targetinfo));
    }

    ret = 0;
failed:
    response__free_unpacked(res, NULL);
    return ret;
}

int retrieve_quote(const sgx_spid_t* spid, bool linkable, const sgx_report_t* report,
                   const sgx_quote_nonce_t* nonce, char** quote, size_t* quote_len) {
    int ret;

    Request req   = REQUEST__INIT;
    Response* res = NULL;

    sgx_quote_t* actual_quote = NULL;

    if (!spid) {
        /* No Software Provider ID (SPID) specified, it is DCAP attestation */
        __UNUSED(linkable);
        __UNUSED(nonce);

        sgx_att_key_id_t default_att_key_id;
        memset(&default_att_key_id, 0, sizeof(default_att_key_id));
        memcpy(&default_att_key_id, &g_default_ecdsa_p256_att_key_id,
                sizeof(g_default_ecdsa_p256_att_key_id));

        Request__GetQuoteExRequest getreq = REQUEST__GET_QUOTE_EX_REQUEST__INIT;
        getreq.report.data         = (uint8_t*)report;
        getreq.report.len          = SGX_REPORT_ACTUAL_SIZE;
        getreq.has_att_key_id      = true;
        getreq.att_key_id.data     = (uint8_t*)&default_att_key_id;
        getreq.att_key_id.len      = sizeof(default_att_key_id);
        getreq.has_qe_report_info  = false; /* used to detect early that QE was spoofed; ignore now */
        getreq.qe_report_info.data = NULL;
        getreq.qe_report_info.len  = 0;
        getreq.buf_size            = SGX_QUOTE_MAX_SIZE;
        req.getquoteexreq          = &getreq;

        ret = request_aesm_service(&req, &res);
        if (ret < 0)
            return ret;

        ret = -EPERM;
        if (!res->getquoteexres) {
            log_error("aesm_service returned wrong message");
            goto out;
        }

        Response__GetQuoteExResponse* r = res->getquoteexres;
        if (r->errorcode != 0) {
            log_error("aesm_service returned error: %d", r->errorcode);
            goto out;
        }

        if (!r->has_quote || r->quote.len < sizeof(sgx_quote_t)) {
            log_error("aesm_service returned invalid quote");
            goto out;
        }

        actual_quote = (sgx_quote_t*)r->quote.data;
    } else {
        /* SPID specified, it is EPID attestation */
        Request__GetQuoteRequest getreq = REQUEST__GET_QUOTE_REQUEST__INIT;
        getreq.report.data   = (uint8_t*)report;
        getreq.report.len    = SGX_REPORT_ACTUAL_SIZE;
        getreq.quote_type    = linkable ? SGX_LINKABLE_SIGNATURE : SGX_UNLINKABLE_SIGNATURE;
        getreq.spid.data     = (uint8_t*)spid;
        getreq.spid.len      = sizeof(*spid);
        getreq.has_nonce     = true;
        getreq.nonce.data    = (uint8_t*)nonce;
        getreq.nonce.len     = sizeof(*nonce);
        getreq.buf_size      = SGX_QUOTE_MAX_SIZE;
        getreq.has_qe_report = true;
        getreq.qe_report     = true;
        req.getquotereq      = &getreq;

        ret = request_aesm_service(&req, &res);
        if (ret < 0)
            return ret;

        ret = -EPERM;
        if (!res->getquoteres) {
            log_error("aesm_service returned wrong message");
            goto out;
        }

        Response__GetQuoteResponse* r = res->getquoteres;
        if (r->errorcode != 0) {
            log_error("aesm_service returned error: %d", r->errorcode);
            goto out;
        }

        if (!r->has_quote || r->quote.len < sizeof(sgx_quote_t)) {
            log_error("aesm_service returned invalid quote");
            goto out;
        }

        actual_quote = (sgx_quote_t*)r->quote.data;
    }

    /* Intel SGX SDK implementation of the Quoting Enclave always sets `quote.len` to user-provided
     * `getreq.buf_size` (see above) instead of the actual size. We calculate the actual size here
     * by peeking into the quote and determining the size of the signature. */
    size_t actual_quote_size = sizeof(sgx_quote_t) + actual_quote->signature_size;
    if (actual_quote_size > SGX_QUOTE_MAX_SIZE) {
        log_error("Size of the obtained SGX quote exceeds %d", SGX_QUOTE_MAX_SIZE);
        goto out;
    }

    char* mmapped = (char*)DO_SYSCALL(mmap, NULL, ALLOC_ALIGN_UP(actual_quote_size),
                                      PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (IS_PTR_ERR(mmapped)) {
        log_error("Failed to allocate memory for the quote");
        ret = -ENOMEM;
        goto out;
    }

    memcpy(mmapped, actual_quote, actual_quote_size);

    *quote = mmapped;
    *quote_len = actual_quote_size;

    ret = 0;
out:
    response__free_unpacked(res, NULL);
    return ret;
}
