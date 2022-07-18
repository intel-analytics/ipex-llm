/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Corporation */

/*
 * To load libsgx_dcap_quoteverify.so, it needs some symbols which are normally
 * provided by libsgx_urts.so (or libsgx_urts_sim.so). To use it in Gramine,
 * there are two workarounds to solve this: 1) load libsgx_urts.so or 2) create
 * dummy functions and always return failure. In this example we use 2).
 */
#define DUMMY_FUNCTION(f)                               \
    __attribute__((visibility("default"))) int f(void); \
    int f(void) {                                       \
        return /*SGX_ERROR_UNEXPECTED*/ 1;              \
    }

/* Provide these dummies since we are not using libsgx_urts.so. */
DUMMY_FUNCTION(sgx_create_enclave)
DUMMY_FUNCTION(sgx_destroy_enclave)
DUMMY_FUNCTION(sgx_ecall)
/* see https://github.com/intel/linux-sgx/blob/sgx_2.14/common/inc/sgx_tstdc.edl */
DUMMY_FUNCTION(sgx_oc_cpuidex)
DUMMY_FUNCTION(sgx_thread_set_untrusted_event_ocall)
DUMMY_FUNCTION(sgx_thread_setwait_untrusted_events_ocall)
DUMMY_FUNCTION(sgx_thread_set_multiple_untrusted_events_ocall)
DUMMY_FUNCTION(sgx_thread_wait_untrusted_event_ocall)
/* see https://github.com/intel/linux-sgx/blob/sgx_2.14/common/inc/sgx_pthread.edl */
DUMMY_FUNCTION(pthread_wait_timeout_ocall)
DUMMY_FUNCTION(pthread_create_ocall)
DUMMY_FUNCTION(pthread_wakeup_ocall)
