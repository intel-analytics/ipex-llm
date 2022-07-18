/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains definitions of functions, variables and data structures for internal uses.
 */

#ifndef SGX_INTERNAL_H
#define SGX_INTERNAL_H

#include <sys/syscall.h>

#include "api.h"
#include "pal_linux.h"
#include "sgx_syscall.h"
#include "toml.h"

extern const size_t g_page_size;
extern pid_t g_host_pid;

#undef IS_ALLOC_ALIGNED
#undef IS_ALLOC_ALIGNED_PTR
#undef ALLOC_ALIGN_UP
#undef ALLOC_ALIGN_UP_PTR
#undef ALLOC_ALIGN_DOWN
#undef ALLOC_ALIGN_DOWN_PTR

#define IS_ALLOC_ALIGNED(addr)     IS_ALIGNED_POW2(addr, g_page_size)
#define IS_ALLOC_ALIGNED_PTR(addr) IS_ALIGNED_PTR_POW2(addr, g_page_size)
#define ALLOC_ALIGN_UP(addr)       ALIGN_UP_POW2(addr, g_page_size)
#define ALLOC_ALIGN_UP_PTR(addr)   ALIGN_UP_PTR_POW2(addr, g_page_size)
#define ALLOC_ALIGN_DOWN(addr)     ALIGN_DOWN_POW2(addr, g_page_size)
#define ALLOC_ALIGN_DOWN_PTR(addr) ALIGN_DOWN_PTR_POW2(addr, g_page_size)

uint32_t htonl(uint32_t longval);
uint16_t htons(uint16_t shortval);
uint32_t ntohl(uint32_t longval);
uint16_t ntohs(uint16_t shortval);

struct pal_enclave {
    /* attributes */
    bool is_first_process; // Initial process in Gramine namespace is special.

    char* application_path;
    char* raw_manifest_data;
    unsigned long baseaddr;
    unsigned long size;
    unsigned long thread_num;
    unsigned long rpc_thread_num;
    unsigned long ssa_frame_size;
    bool nonpie_binary;
    bool remote_attestation_enabled;
    bool use_epid_attestation; /* Valid only if `remote_attestation_enabled` is true, selects
                                * EPID/DCAP attestation scheme. */
    char* libpal_uri; /* Path to the PAL binary */

#ifdef DEBUG
    /* profiling */
    bool profile_enable;
    int profile_mode;
    char profile_filename[64];
    bool profile_with_stack;
    int profile_frequency;
#endif
};

extern struct pal_enclave g_pal_enclave;

int open_sgx_driver(bool need_gsgx);
bool is_wrfsbase_supported(void);

int read_enclave_token(int token_file, sgx_arch_token_t* token);
int read_enclave_sigstruct(int sigfile, sgx_arch_enclave_css_t* sig);

int create_enclave(sgx_arch_secs_t* secs, sgx_arch_token_t* token);

enum sgx_page_type { SGX_PAGE_SECS, SGX_PAGE_TCS, SGX_PAGE_REG };
int add_pages_to_enclave(sgx_arch_secs_t* secs, void* addr, void* user_addr, unsigned long size,
                         enum sgx_page_type type, int prot, bool skip_eextend, const char* comment);

/*!
 * \brief Retrieve Quoting Enclave's sgx_target_info_t by talking to AESMD.
 *
 * \param[in]  is_epid        Use EPID attestation if true, DCAP/ECDSA if false.
 * \param[out] qe_targetinfo  Retrieved Quoting Enclave's target info.
 * \return                    0 on success, negative error code otherwise.
 */
int init_quoting_enclave_targetinfo(bool is_epid, sgx_target_info_t* qe_targetinfo);

/*!
 * \brief Obtain SGX Quote from the Quoting Enclave (communicate via AESM).
 *
 * \param[in]  spid       Software provider ID (SPID); if NULL then DCAP/ECDSA is used.
 * \param[in]  linkable   Quote type (linkable vs unlinkable); ignored if DCAP/ECDSA is used.
 * \param[in]  report     Enclave report to convert into a quote.
 * \param[in]  nonce      16B nonce to be included in the quote for freshness; ignored if
 *                        DCAP/ECDSA is used.
 * \param[out] quote      Quote returned by the Quoting Enclave (allocated via mmap() in this
 *                        function; the caller gets the ownership of the quote).
 * \param[out] quote_len  Length of the quote returned by the Quoting Enclave.
 * \return                0 on success, negative Linux error code otherwise.
 */
int retrieve_quote(const sgx_spid_t* spid, bool linkable, const sgx_report_t* report,
                   const sgx_quote_nonce_t* nonce, char** quote, size_t* quote_len);

int init_enclave(sgx_arch_secs_t* secs, sgx_arch_enclave_css_t* sigstruct, sgx_arch_token_t* token);

int sgx_ecall(long ecall_no, void* ms);
int sgx_raise(int event);

void async_exit_pointer(void);
void eresume_pointer(void);
void async_exit_pointer_end(void);

int get_tid_from_tcs(void* tcs);
int clone_thread(void);

void create_tcs_mapper(void* tcs_base, unsigned int thread_num);
int pal_thread_init(void* tcbptr);
void map_tcs(unsigned int tid);
void unmap_tcs(void);
int current_enclave_thread_cnt(void);
void thread_exit(int status);

int sgx_init_child_process(int parent_stream_fd, char** out_application_path, char** out_manifest);
int sgx_signal_setup(void);
int block_async_signals(bool block);

#ifdef DEBUG
/* SGX profiling (sgx_profile.c) */

/*
 * Default and maximum sampling frequency. We depend on Linux scheduler to interrupt us, so it's not
 * possible to achieve higher than 250.
 */
#define SGX_PROFILE_DEFAULT_FREQUENCY 50
#define SGX_PROFILE_MAX_FREQUENCY 250

enum {
    SGX_PROFILE_MODE_AEX = 1,
    SGX_PROFILE_MODE_OCALL_INNER = 2,
    SGX_PROFILE_MODE_OCALL_OUTER = 3,
};

/* Filenames for saved data */
#define SGX_PROFILE_FILENAME "sgx-perf.data"
#define SGX_PROFILE_FILENAME_WITH_PID "sgx-perf-%d.data"

/* Initialize based on g_pal_enclave settings */
int sgx_profile_init(void);

/* Finalize and close file */
void sgx_profile_finish(void);

/* Record a sample during AEX */
void sgx_profile_sample_aex(void* tcs);

/* Record a sample during OCALL (inner state) */
void sgx_profile_sample_ocall_inner(void* enclave_gpr);

/* Record a sample during OCALL (function to be executed) */
void sgx_profile_sample_ocall_outer(void* ocall_func);

/* Record a new mapped ELF */
void sgx_profile_report_elf(const char* filename, void* addr);
#endif

/* perf.data output (sgx_perf_data.h) */

#define PD_STACK_SIZE 8192

struct perf_data;

struct perf_data* pd_open(const char* file_name, bool with_stack);

/* Finalize and close; returns resulting file size */
ssize_t pd_close(struct perf_data* pd);

/* Write PERF_RECORD_COMM (report command name) */
int pd_event_command(struct perf_data* pd, const char* command, uint32_t pid, uint32_t tid);

/* Write PERF_RECORD_MMAP (report mmap of executable region) */
int pd_event_mmap(struct perf_data* pd, const char* filename, uint32_t pid, uint64_t addr,
                  uint64_t len, uint64_t pgoff);

/* Write PERF_RECORD_SAMPLE (simple version) */
int pd_event_sample_simple(struct perf_data* pd, uint64_t ip, uint32_t pid, uint32_t tid,
                           uint64_t period);

/* Write PERF_RECORD_SAMPLE (with stack sample, at most PD_STACK_SIZE bytes) */
int pd_event_sample_stack(struct perf_data* pd,  uint64_t ip, uint32_t pid, uint32_t tid,
                          uint64_t period, sgx_pal_gpr_t* gpr, void* stack, size_t stack_size);

#endif
