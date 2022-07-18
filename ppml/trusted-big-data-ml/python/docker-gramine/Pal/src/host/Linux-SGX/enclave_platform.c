/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2019, Texas A&M University */

#include "pal_linux.h"
#include "pal_linux_error.h"

int sgx_get_quote(const sgx_spid_t* spid, const sgx_quote_nonce_t* nonce,
                  const sgx_report_data_t* report_data, bool linkable, char** quote,
                  size_t* quote_len) {
    /* must align all arguments to sgx_report() so that EREPORT doesn't complain */
    __sgx_mem_aligned sgx_report_t report;
    __sgx_mem_aligned sgx_target_info_t targetinfo = g_pal_linuxsgx_state.qe_targetinfo;
    __sgx_mem_aligned sgx_report_data_t _report_data = *report_data;

    int ret = sgx_report(&targetinfo, &_report_data, &report);
    if (ret) {
        log_error("Failed to get enclave report");
        return -PAL_ERROR_DENIED;
    }

    ret = ocall_get_quote(spid, linkable, &report, nonce, quote, quote_len);
    if (ret < 0) {
        log_error("Failed to get quote");
        return unix_to_pal_error(ret);
    }
    return 0;
}
