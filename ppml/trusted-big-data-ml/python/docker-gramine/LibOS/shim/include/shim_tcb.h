#ifndef _SHIM_TCB_H_
#define _SHIM_TCB_H_

#include "api.h"
#include "assert.h"
#include "atomic.h"
#include "gramine_entry_api.h"
#include "pal.h"
#include "shim_entry.h"
#include "shim_tcb-arch.h"

struct shim_context {
    PAL_CONTEXT* regs;
    long syscall_nr;
    uintptr_t tls; /* Used only in clone. */
};

typedef struct shim_tcb shim_tcb_t;
struct shim_tcb {
    shim_tcb_t*         self;

    /* Function pointer for patched code calling into Gramine. */
    void*               syscalldb;

    struct shim_thread* tp;
    void*               libos_stack_bottom;
    struct shim_context context;
    /* Scratch space to temporarily store a register. On some architectures (e.g. x86_64 inside
     * an SGX enclave) we lack a way to restore all (or at least some) registers atomically. */
    void*               syscall_scratch_pc;
    void*               vma_cache;
    char                log_prefix[32];
};

static_assert(
    offsetof(PAL_TCB, libos_tcb) + offsetof(shim_tcb_t, syscalldb) == GRAMINE_SYSCALL_OFFSET,
    "GRAMINE_SYSCALL_OFFSET must match");

static inline void __shim_tcb_init(shim_tcb_t* shim_tcb) {
    shim_tcb->self = shim_tcb;
    shim_tcb->syscalldb = &syscalldb;
    shim_tcb->context.syscall_nr = -1;
    shim_tcb->vma_cache = NULL;
}

/* Call this function at the beginning of thread execution. */
static inline void shim_tcb_init(void) {
    PAL_TCB* tcb = pal_get_tcb();
    static_assert(sizeof(shim_tcb_t) <= sizeof(((PAL_TCB*)0)->libos_tcb),
                  "Not enough space for LibOS TCB inside Pal TCB");
    shim_tcb_t* shim_tcb = (shim_tcb_t*)tcb->libos_tcb;
    memset(shim_tcb, 0, sizeof(*shim_tcb));
    __shim_tcb_init(shim_tcb);
}

static inline shim_tcb_t* shim_get_tcb(void) {
    return SHIM_TCB_GET(self);
}

#endif /* _SHIM_H_ */
