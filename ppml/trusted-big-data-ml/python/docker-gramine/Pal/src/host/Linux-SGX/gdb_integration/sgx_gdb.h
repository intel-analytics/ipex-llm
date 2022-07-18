#include <stdint.h>

#define MAX_DBG_THREADS 1024

/*
 * Pre-defined address at which Gramine-SGX writes the `enclave_dbginfo` object, and periodically
 * updates it. The object is being read by the GDB integration (`sgx_gdb.c`) using `ptrace`.
 *
 * The address should be in untrusted memory (outside of enclave), and should not overlap with the
 * ASan shadow memory area (see `asan.h`).
 */
#define DBGINFO_ADDR 0x10000000000ULL /* 1 TB */

/* This struct is read using PTRACE_PEEKDATA in 8B increments
 * therefore it is aligned as uint64_t. */
struct __attribute__((aligned(__alignof__(uint64_t)))) enclave_dbginfo {
    int pid;
    uint64_t base;
    uint64_t size;
    uint64_t ssa_frame_size;
    void* aep;
    void* eresume;
    int thread_tids[MAX_DBG_THREADS];
    void* tcs_addrs[MAX_DBG_THREADS];
    uint64_t thread_stepping[MAX_DBG_THREADS / 64]; /* bit vector */
};
