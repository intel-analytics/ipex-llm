#include "enclave_ecalls.h"

#include <stdalign.h>

#include "api.h"
#include "ecall_types.h"
#include "pal_linux.h"
#include "rpc_queue.h"
#include "sgx_arch.h"

#define SGX_CAST(type, item) ((type)(item))

extern void* g_enclave_base;
extern void* g_enclave_top;

static struct atomic_int g_enclave_start_called = ATOMIC_INIT(0);

/* returns 0 if rpc_queue is valid/not requested, otherwise -1 */
static int verify_and_init_rpc_queue(rpc_queue_t* untrusted_rpc_queue) {
    g_rpc_queue = NULL;

    if (!untrusted_rpc_queue) {
        /* user app didn't request RPC queue (i.e., the app didn't request exitless syscalls) */
        return 0;
    }

    if (!sgx_is_completely_outside_enclave(untrusted_rpc_queue, sizeof(*untrusted_rpc_queue))) {
        /* malicious RPC queue object, return error */
        return -1;
    }

    g_rpc_queue = untrusted_rpc_queue;
    return 0;
}

/*
 * Called from enclave_entry.S to execute ecalls.
 *
 * During normal operation handle_ecall will not return. The exception is that
 * it will return if invalid parameters are passed. In this case
 * enclave_entry.S will go into an endless loop since a clean return to urts is
 * not easy in all cases.
 *
 * Parameters:
 *
 *  ecall_index:
 *      Number of requested ecall. Untrusted.
 *
 *  ecall_args:
 *      Pointer to arguments for requested ecall. Untrusted.
 *
 *  exit_target:
 *      Address to return to after EEXIT. Untrusted.
 *
 *  enclave_base_addr:
 *      Base address of enclave. Calculated dynamically in enclave_entry.S.
 *      Trusted.
 */
void handle_ecall(long ecall_index, void* ecall_args, void* exit_target, void* enclave_base_addr) {
    if (ecall_index < 0 || ecall_index >= ECALL_NR)
        return;

    if (!g_enclave_top) {
        g_enclave_base = enclave_base_addr;
        g_enclave_top  = enclave_base_addr + GET_ENCLAVE_TLS(enclave_size);
    }

    /* disallow malicious URSP (that points into the enclave) */
    void* ursp = (void*)GET_ENCLAVE_TLS(gpr)->ursp;
    if (g_enclave_base <= ursp && ursp <= g_enclave_top)
        return;

    SET_ENCLAVE_TLS(exit_target,     exit_target);
    SET_ENCLAVE_TLS(ustack,          ursp);
    SET_ENCLAVE_TLS(ustack_top,      ursp);
    SET_ENCLAVE_TLS(clear_child_tid, NULL);
    SET_ENCLAVE_TLS(untrusted_area_cache.in_use, 0UL);

    int64_t t = 0;
    if (__atomic_compare_exchange_n(&g_enclave_start_called.counter, &t, 1, /*weak=*/false,
                                    __ATOMIC_SEQ_CST, __ATOMIC_RELAXED)) {
        // ENCLAVE_START not yet called, so only valid ecall is ENCLAVE_START.
        if (ecall_index != ECALL_ENCLAVE_START) {
            // To keep things simple, we treat an invalid ecall_index like an
            // unsuccessful call to ENCLAVE_START.
            return;
        }

        ms_ecall_enclave_start_t* ms = (ms_ecall_enclave_start_t*)ecall_args;

        if (!ms || !sgx_is_completely_outside_enclave(ms, sizeof(*ms))) {
            return;
        }

        if (verify_and_init_rpc_queue(READ_ONCE(ms->rpc_queue)))
            return;

        struct pal_topo_info* topo_info = READ_ONCE(ms->ms_topo_info);
        if (!topo_info || !sgx_is_completely_outside_enclave(topo_info, sizeof(*topo_info)))
            return;

        /* xsave size must be initialized early, from a trusted source (EREPORT result) */
        // TODO: This eats 1KB of a stack frame which lives for the whole lifespan of this enclave.
        //       We should move it somewhere else and deallocate right after use.
        __sgx_mem_aligned sgx_target_info_t target_info;
        alignas(128) char report_data[64] = {0};
        __sgx_mem_aligned sgx_report_t report;
        memset(&report, 0, sizeof(report));
        memset(&target_info, 0, sizeof(target_info));
        sgx_report(&target_info, &report_data, &report);
        init_xsave_size(report.body.attributes.xfrm);

        /* pal_linux_main is responsible for checking the passed arguments */
        pal_linux_main(READ_ONCE(ms->ms_libpal_uri), READ_ONCE(ms->ms_libpal_uri_len),
                       READ_ONCE(ms->ms_args), READ_ONCE(ms->ms_args_size), READ_ONCE(ms->ms_env),
                       READ_ONCE(ms->ms_env_size), READ_ONCE(ms->ms_parent_stream_fd),
                       READ_ONCE(ms->ms_qe_targetinfo), topo_info);
    } else {
        // ENCLAVE_START already called (maybe successfully, maybe not), so
        // only valid ecall is THREAD_START.
        if (ecall_index != ECALL_THREAD_START) {
            return;
        }

        // Only allow THREAD_START after successful enclave initialization.
        if (!g_pal_linuxsgx_state.enclave_initialized) {
            return;
        }

        pal_start_thread();
    }
    // pal_linux_main and pal_start_thread should never return.
}
