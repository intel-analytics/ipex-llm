/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University
 * Copyright (C) 2020 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

/*
 * This file contains definition of x86_64-specific aspects of PAL.
 */

#ifndef PAL_H
// TODO: fix this
#error This header is usable only inside pal.h (due to a cyclic dependency).
#endif

#ifndef PAL_ARCH_H
#define PAL_ARCH_H

#include <assert.h>
#include <stdint.h>

#include "api.h"
#include "cpu.h"
#include "pal.h"

typedef struct pal_tcb PAL_TCB;

#define PAL_LIBOS_TCB_SIZE 256


#define STACK_PROTECTOR_CANARY_DEFAULT  0xbadbadbadbadUL

typedef struct pal_tcb {
    struct pal_tcb* self;
    /* random per-thread canary used by GCC's stack protector; must be at %gs:[0x8]
     * because we specify `-mstack-protector-guard-reg=%gs -mstack-protector-guard-offset=8` */
    uint64_t stack_protector_canary;
    /* uint64_t for alignment */
    uint64_t libos_tcb[(PAL_LIBOS_TCB_SIZE + sizeof(uint64_t) - 1) / sizeof(uint64_t)];
    /* data private to PAL implementation follows this struct. */
} PAL_TCB;

__attribute_no_stack_protector
static inline void pal_tcb_arch_set_stack_canary(PAL_TCB* tcb, uint64_t canary) {
    tcb->stack_protector_canary = canary;
}

static_assert(offsetof(PAL_TCB, stack_protector_canary) == 0x8,
              "unexpected offset of stack_protector_canary in PAL_TCB struct");

#include "pal_host-arch.h"

static inline PAL_TCB* pal_get_tcb(void) {
    PAL_TCB* tcb;
    __asm__("movq %%gs:%c1, %0" : "=r"(tcb) : "i"(offsetof(struct pal_tcb, self)) : "memory");
    return tcb;
}

static inline unsigned long count_ulong_bits_set(unsigned long x) {
    unsigned long result;
    __asm__("popcnt %1, %0" : "=r"(result) : "r"(x) : "cc");
    return result;
}

/* holds /proc/cpuinfo data */
struct pal_cpu_info {
    const char* cpu_vendor;
    const char* cpu_brand;
    PAL_NUM cpu_family;
    PAL_NUM cpu_model;
    PAL_NUM cpu_stepping;
    double cpu_bogomips;
    const char* cpu_flags;
};

union pal_csgsfs {
    struct {
        uint16_t cs;
        uint16_t gs;
        uint16_t fs;
        uint16_t ss;
    };
    uint64_t csgsfs;
};

/* Adopt Linux style fp layout, _libc_fpstate of glibc:
 * Because self-contained definition is needed for Pal definition,
 * same layout is defined with PAL prefix.
 */
#define PAL_FP_XSTATE_MAGIC1      0x46505853U
#define PAL_FP_XSTATE_MAGIC2      0x46505845U
#define PAL_FP_XSTATE_MAGIC2_SIZE (sizeof(PAL_FP_XSTATE_MAGIC2))

enum PAL_XFEATURE {
    PAL_XFEATURE_FP,
    PAL_XFEATURE_SSE,
    PAL_XFEATURE_YMM,
    PAL_XFEATURE_BNDREGS,
    PAL_XFEATURE_BNDCSR,
    PAL_XFEATURE_OPMASK,
    PAL_XFEATURE_ZMM_Hi256,
    PAL_XFEATURE_Hi16_ZMM,
    PAL_XFEATURE_PT,
    PAL_XFEATURE_PKRU,

    PAL_XFEATURE_MAX,
};

#define PAL_XFEATURE_MASK_FP        (1UL << PAL_XFEATURE_FP)
#define PAL_XFEATURE_MASK_SSE       (1UL << PAL_XFEATURE_SSE)
#define PAL_XFEATURE_MASK_YMM       (1UL << PAL_XFEATURE_YMM)
#define PAL_XFEATURE_MASK_BNDREGS   (1UL << PAL_XFEATURE_BNDREGS)
#define PAL_XFEATURE_MASK_BNDCSR    (1UL << PAL_XFEATURE_BNDCSR)
#define PAL_XFEATURE_MASK_OPMASK    (1UL << PAL_XFEATURE_OPMASK)
#define PAL_XFEATURE_MASK_ZMM_Hi256 (1UL << PAL_XFEATURE_ZMM_Hi256)
#define PAL_XFEATURE_MASK_Hi16_ZMM  (1UL << PAL_XFEATURE_Hi16_ZMM)
#define PAL_XFEATURE_MASK_PT        (1UL << PAL_XFEATURE_PT)
#define PAL_XFEATURE_MASK_PKRU      (1UL << PAL_XFEATURE_PKRU)

#define PAL_XFEATURE_MASK_FPSSE     (PAL_XFEATURE_MASK_FP \
                                     | PAL_XFEATURE_MASK_SSE)
#define PAL_XFEATURE_MASK_AVX512    (PAL_XFEATURE_MASK_OPMASK \
                                     | PAL_XFEATURE_MASK_ZMM_Hi256 \
                                     | PAL_XFEATURE_MASK_Hi16_ZMM)

typedef struct {
    uint32_t magic1;        /*!< PAL_FP_XSTATE_MAGIC1 */
    uint32_t extended_size; /*!< g_xsave_size */
    uint64_t xfeatures;     /*!< XSAVE feature */
    uint32_t xstate_size;   /*!< g_xsave_size + PAL_FP_STATE_MAGIC2_SIZE */
    uint32_t padding[7];
} PAL_FPX_SW_BYTES;

typedef struct {
    uint32_t cwd;
    uint32_t swd;
    uint32_t twd;
    uint32_t fip;
    uint32_t fcs;
    uint32_t foo;
    uint32_t fos;
    uint32_t st_space[20];
    uint8_t ftop;
    uint8_t changed;
    uint8_t lookahead;
    uint8_t no_update;
    uint8_t rm;
    uint8_t alimit;
    void* info; /* struct math_emu_info */
    uint32_t entry_eip;
} PAL_SWREGS_STATE;

typedef struct {
    uint16_t significand[4];
    uint16_t exponent;
    uint16_t padding[3];
} PAL_FPXREG;

typedef struct {
    uint32_t element[4];
} PAL_XMMREG;

typedef struct {
    /* 64-bit FXSAVE format.  */
    uint16_t cwd;
    uint16_t swd;
    uint16_t ftw;
    uint16_t fop;
    uint64_t rip;
    uint64_t rdp;
    uint32_t mxcsr;
    uint32_t mxcr_mask;
    PAL_FPXREG st[8];
    PAL_XMMREG xmm[16];
    union {
        uint32_t padding[24];
        struct {
            uint32_t padding2[12];
            PAL_FPX_SW_BYTES sw_reserved;
        };
    };
} PAL_FPSTATE;

typedef struct {
    uint64_t xfeatures;
    uint64_t xcomp_bv;
    uint64_t reserved[6];
} __attribute__((packed)) PAL_XSTATE_HEADER;

#define PAL_XSTATE_ALIGN 64

typedef struct {
    PAL_FPSTATE fpstate;
    PAL_XSTATE_HEADER header;
} __attribute__((packed, aligned(PAL_XSTATE_ALIGN))) PAL_XREGS_STATE;

/* Define PAL_CONTEXT outside the typedef for Doxygen */
struct PAL_CONTEXT {
    uint64_t r8;
    uint64_t r9;
    uint64_t r10;
    uint64_t r11;
    uint64_t r12;
    uint64_t r13;
    uint64_t r14;
    uint64_t r15;
    uint64_t rdi;
    uint64_t rsi;
    uint64_t rbp;
    uint64_t rbx;
    uint64_t rdx;
    uint64_t rax;
    uint64_t rcx;
    uint64_t rsp;
    uint64_t rip;
    uint64_t efl;
    uint64_t csgsfsss;
    uint64_t err;
    uint64_t trapno;
    uint64_t oldmask;
    uint64_t cr2;

    PAL_XREGS_STATE* fpregs;

    uint32_t mxcsr; /* MXCSR control/status register (for SSE/AVX/...) */
    uint16_t fpcw;  /* FPU Control Word (for x87) */
    uint8_t is_fpregs_used; /* Equal to 0 iff `fpregs` is not populated. */
    uint8_t _pad;
};
typedef struct PAL_CONTEXT PAL_CONTEXT;

typedef int64_t arch_syscall_arg_t;

#define ALL_SYSCALL_ARGS(context) \
    (context)->rdi, \
    (context)->rsi, \
    (context)->rdx, \
    (context)->r10, \
    (context)->r8, \
    (context)->r9


static inline void pal_context_set_ip(PAL_CONTEXT* context, PAL_NUM insnptr) {
    context->rip = insnptr;
}
static inline PAL_NUM pal_context_get_ip(PAL_CONTEXT* context) {
    return context->rip;
}

static inline void pal_context_set_sp(PAL_CONTEXT* context, PAL_NUM sp) {
    context->rsp = sp;
}
static inline PAL_NUM pal_context_get_sp(PAL_CONTEXT* context) {
    return context->rsp;
}

static inline void pal_context_set_retval(PAL_CONTEXT* context, uint64_t val) {
    context->rax = val;
}
static inline uint64_t pal_context_get_retval(PAL_CONTEXT* context) {
    return context->rax;
}

static inline uint64_t pal_context_get_syscall(PAL_CONTEXT* context) {
    return context->rax;
}

/* Copies `PAL_CONTEXT` without extended FPU/SSE state (but keeping control words). */
static inline void pal_context_copy(PAL_CONTEXT* dst, PAL_CONTEXT* src) {
    *dst = *src;
    dst->is_fpregs_used = 0;
    dst->fpregs = NULL;
    if (src->is_fpregs_used) {
        dst->mxcsr = src->fpregs->fpstate.mxcsr;
        dst->fpcw = src->fpregs->fpstate.cwd;
    }
}

#endif /* PAL_ARCH_H */
