/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

#ifndef PAL_LINUX_H
#define PAL_LINUX_H

#include <asm/mman.h>
#include <linux/mman.h>

#include "api.h"
#include "assert.h"

/*
 * XXX this ifdef is because there is no mbedtls linked in untrusted PAL; this should be fixed by
 * cleaning up this header which has become a rubbish bin for everything that didn't have a better
 * place
 */
#ifdef IN_ENCLAVE
#include "crypto.h"
#endif /* IN_ENCLAVE */

#include "enclave_ocalls.h"
#include "linux_types.h"
#include "log.h"
#include "pal.h"
#include "pal_internal.h"
#include "pal_linux_defs.h"
#include "sgx_api.h"
#include "sgx_arch.h"
#include "sgx_attest.h"
#include "sgx_syscall.h"
#include "sgx_tls.h"

/* Part of Linux-SGX PAL private state which is not shared with other PALs. */
extern struct pal_linuxsgx_state {
    /* enclave information */
    bool enclave_initialized;        /* thread creation ECALL is allowed only after this is set */
    sgx_target_info_t qe_targetinfo; /* received from untrusted host, use carefully */
    sgx_report_body_t enclave_info;  /* cached self-report result, trusted */

    /* remaining heap usable by application */
    void* heap_min;
    void* heap_max;
} g_pal_linuxsgx_state;


int init_child_process(int parent_stream_fd, PAL_HANDLE* out_parent, uint64_t* out_instance_id);

#ifdef IN_ENCLAVE

int init_enclave(void);
void init_untrusted_slab_mgr(void);

extern const size_t g_page_size;
extern size_t g_pal_internal_mem_size;

noreturn void pal_linux_main(char* uptr_libpal_uri, size_t libpal_uri_len, char* uptr_args,
                             size_t args_size, char* uptr_env, size_t env_size,
                             int parent_stream_fd, sgx_target_info_t* uptr_qe_targetinfo,
                             struct pal_topo_info* uptr_topo_info);
void pal_start_thread(void);

extern char __text_start, __text_end, __data_start, __data_end;
#define TEXT_START ((void*)(&__text_start))
#define TEXT_END   ((void*)(&__text_end))
#define DATA_START ((void*)(&__data_start))
#define DATA_END   ((void*)(&__data_end))

extern const uint32_t g_cpu_extension_sizes[];
extern const uint32_t g_cpu_extension_offsets[];

extern int g_xsave_enabled;
extern uint64_t g_xsave_features;
extern uint32_t g_xsave_size;
#define XSAVE_RESET_STATE_SIZE (512 + 64)  // 512 for legacy regs, 64 for xsave header
extern const uint32_t g_xsave_reset_state[];

void init_xsave_size(uint64_t xfrm);
void save_xregs(PAL_XREGS_STATE* xsave_area);
void restore_xregs(const PAL_XREGS_STATE* xsave_area);
noreturn void _restore_sgx_context(sgx_cpu_context_t* uc, PAL_XREGS_STATE* xsave_area);

void _DkExceptionHandler(unsigned int exit_info, sgx_cpu_context_t* uc,
                         PAL_XREGS_STATE* xregs_state);
/* `event_` is actually of `enum pal_event` type, but we call it from assembly, so we need to know
 * its underlying type. */
void _DkHandleExternalEvent(long event_, sgx_cpu_context_t* uc, PAL_XREGS_STATE* xregs_state);

bool is_tsc_usable(void);
uint64_t get_tsc_hz(void);
void init_tsc(void);

int init_enclave(void);
void init_untrusted_slab_mgr(void);

/* master key for all enclaves of one application, populated by the first enclave and inherited by
 * all other enclaves (children, their children, etc.); used as master key in pipes' encryption */
extern PAL_SESSION_KEY g_master_key;

/*
 * sgx_verify_report: verify a CPU-signed report from another local enclave
 * @report: the buffer storing the report to verify
 */
int sgx_verify_report(sgx_report_t* report);

/*!
 * \brief Obtain a CPU-signed report for local attestation.
 *
 * Caller must align all parameters to 512 bytes (cf. `__sgx_mem_aligned`).
 *
 * \param[in]  target_info  Information on the target enclave.
 * \param[in]  data         User-specified data to be included in the report.
 * \param[out] report       Output buffer to store the report.
 * \return                  0 on success, negative error code otherwise.
 */
int sgx_get_report(const sgx_target_info_t* target_info, const sgx_report_data_t* data,
                   sgx_report_t* report);

/*!
 * \brief Obtain an enclave/signer-bound key via EGETKEY(SGX_SEAL_KEY) for secret migration/sealing
 * of files.
 *
 * \param[in]  key_policy  Must be SGX_KEYPOLICY_MRENCLAVE or SGX_KEYPOLICY_MRSIGNER. Binds the
 *                         sealing key to MRENCLAVE (only the same enclave can unseal secrets) or
 *                         to MRSIGNER (all enclaves from the same signer can unseal secrets).
 * \param[out] seal_key    Output buffer to store the sealing key.
 * \return 0 on success, negative error code otherwise.
 */
int sgx_get_seal_key(uint16_t key_policy, sgx_key_128bit_t* seal_key);

/*!
 * \brief Verify the peer enclave during SGX local attestation.
 *
 * Verifies that the SGX information of the peer enclave is the same as ours (all Gramine enclaves
 * with the same configuration have the same SGX enclave info), and that the signer of the SGX
 * report is the owner of the newly established session key.
 *
 * \param  peer_enclave_info  SGX information of the peer enclave.
 * \param  expected_data      Expected SGX report data, contains SHA256(K_e || tag1); see
 *                            also top-level comment in db_process.c.
 * \return 0 on success, negative error code otherwise.
 */
bool is_peer_enclave_ok(sgx_report_body_t* peer_enclave_info,
                        sgx_report_data_t* expected_data);

/* perform Diffie-Hellman to establish a session key and also produce a hash over (K_e || tag1) for
 * parent enclave A and a hash over (K_e || tag2) for child enclave B; see also top-level comment in
 * db_process.c */
int _DkStreamKeyExchange(PAL_HANDLE stream, PAL_SESSION_KEY* out_key,
                         sgx_report_data_t* out_parent_report_data,
                         sgx_report_data_t* out_child_report_data);

/*!
 * \brief Request a local report on an RPC stream (typically called by parent enclave).
 *
 * \param  stream           Stream handle for sending and receiving messages.
 * \param  my_report_data   User-defined data to embed into outgoing SGX report.
 * \param  peer_report_data User-defined data expected in the incoming SGX report.
 * \return 0 on success, negative error code otherwise.
 */
int _DkStreamReportRequest(PAL_HANDLE stream, sgx_report_data_t* my_report_data,
                           sgx_report_data_t* peer_report_data);

/*!
 * \brief Respond with a local report on an RPC stream (typically called by child enclave).
 *
 * \param  stream           Stream handle for sending and receiving messages.
 * \param  my_report_data   User-defined data to embed into outgoing SGX report.
 * \param  peer_report_data User-defined data expected in the incoming SGX report.
 * \return 0 on success, negative error code otherwise.
 */
int _DkStreamReportRespond(PAL_HANDLE stream, sgx_report_data_t* my_report_data,
                           sgx_report_data_t* peer_report_data);

int _DkStreamSecureInit(PAL_HANDLE stream, bool is_server, PAL_SESSION_KEY* session_key,
                        LIB_SSL_CONTEXT** out_ssl_ctx, const uint8_t* buf_load_ssl_ctx,
                        size_t buf_size);
int _DkStreamSecureFree(LIB_SSL_CONTEXT* ssl_ctx);
int _DkStreamSecureRead(LIB_SSL_CONTEXT* ssl_ctx, uint8_t* buf, size_t len, bool is_blocking);
int _DkStreamSecureWrite(LIB_SSL_CONTEXT* ssl_ctx, const uint8_t* buf, size_t len,
                         bool is_blocking);
int _DkStreamSecureSave(LIB_SSL_CONTEXT* ssl_ctx, const uint8_t** obuf, size_t* olen);

#endif /* IN_ENCLAVE */

#endif /* PAL_LINUX_H */
