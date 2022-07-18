#ifndef _SHIM_TCB_ARCH_H_
#define _SHIM_TCB_ARCH_H_

#include <stdint.h>

#include "pal.h"

/* adopt Linux x86-64 structs for FP layout: self-contained definition is needed for LibOS, so
 * define the exact same layout with `shim_` prefix; taken from
 * https://elixir.bootlin.com/linux/v5.9/source/arch/x86/include/uapi/asm/sigcontext.h */
#define SHIM_FP_XSTATE_MAGIC1      0x46505853U
#define SHIM_FP_XSTATE_MAGIC2      0x46505845U
#define SHIM_FP_XSTATE_MAGIC2_SIZE (sizeof(SHIM_FP_XSTATE_MAGIC2))

#define SHIM_XSTATE_ALIGN 64

enum SHIM_XFEATURE {
    SHIM_XFEATURE_FP,
    SHIM_XFEATURE_SSE,
    SHIM_XFEATURE_YMM,
    SHIM_XFEATURE_BNDREGS,
    SHIM_XFEATURE_BNDCSR,
    SHIM_XFEATURE_OPMASK,
    SHIM_XFEATURE_ZMM_Hi256,
    SHIM_XFEATURE_Hi16_ZMM,
    SHIM_XFEATURE_PT,
    SHIM_XFEATURE_PKRU,
};

#define SHIM_XFEATURE_MASK_FP        (1UL << SHIM_XFEATURE_FP)
#define SHIM_XFEATURE_MASK_SSE       (1UL << SHIM_XFEATURE_SSE)
#define SHIM_XFEATURE_MASK_YMM       (1UL << SHIM_XFEATURE_YMM)
#define SHIM_XFEATURE_MASK_BNDREGS   (1UL << SHIM_XFEATURE_BNDREGS)
#define SHIM_XFEATURE_MASK_BNDCSR    (1UL << SHIM_XFEATURE_BNDCSR)
#define SHIM_XFEATURE_MASK_OPMASK    (1UL << SHIM_XFEATURE_OPMASK)
#define SHIM_XFEATURE_MASK_ZMM_Hi256 (1UL << SHIM_XFEATURE_ZMM_Hi256)
#define SHIM_XFEATURE_MASK_Hi16_ZMM  (1UL << SHIM_XFEATURE_Hi16_ZMM)
#define SHIM_XFEATURE_MASK_PT        (1UL << SHIM_XFEATURE_PT)
#define SHIM_XFEATURE_MASK_PKRU      (1UL << SHIM_XFEATURE_PKRU)

#define SHIM_XFEATURE_MASK_FPSSE     (SHIM_XFEATURE_MASK_FP | SHIM_XFEATURE_MASK_SSE)
#define SHIM_XFEATURE_MASK_AVX512    (SHIM_XFEATURE_MASK_OPMASK | SHIM_XFEATURE_MASK_ZMM_Hi256 \
                                      | SHIM_XFEATURE_MASK_Hi16_ZMM)

/* Bytes 464..511 in the 512B layout of the FXSAVE/FXRSTOR frame are reserved for SW usage. On
 * CPUs supporting XSAVE/XRSTOR, these bytes are used to extend the fpstate pointer in the
 * sigcontext, which includes the extended state information along with fpstate information. */
struct shim_fpx_sw_bytes {
    uint32_t magic1;        /*!< SHIM_FP_XSTATE_MAGIC1 (it is an xstate context) */
    uint32_t extended_size; /*!< g_shim_xsave_size + SHIM_FP_STATE_MAGIC2_SIZE */
    uint64_t xfeatures;     /*!< XSAVE features (feature bit mask, including FP/SSE/extended) */
    uint32_t xstate_size;   /*!< g_shim_xsave_size (XSAVE area size as reported by CPUID) */
    uint32_t padding[7];    /*!< for future use */
};

/* 64-bit FPU frame (FXSAVE format, 512B total size) */
struct shim_fpstate {
    uint16_t cwd;
    uint16_t swd;
    uint16_t twd;
    uint16_t fop;
    uint64_t rip;
    uint64_t rdp;
    uint32_t mxcsr;
    uint32_t mxcr_mask;
    uint32_t st_space[32];  /*  8x  FP registers, 16 bytes each */
    uint32_t xmm_space[64]; /* 16x XMM registers, 16 bytes each */
    uint32_t reserved2[12];
    union {
        uint32_t reserved3[12];
        struct shim_fpx_sw_bytes sw_reserved; /* potential extended state is encoded here */
    };
};

struct shim_xstate_header {
    uint64_t xfeatures;
    uint64_t reserved1[2];
    uint64_t reserved2[5];
};

struct shim_xstate {
    struct shim_fpstate fpstate;          /* 512B legacy FXSAVE/FXRSTOR frame (FP and XMM regs) */
    struct shim_xstate_header xstate_hdr; /* 64B header for newer XSAVE/XRSTOR frame */
    /* rest is filled with extended regs (YMM, ZMM, ...) by HW + 4B of MAGIC2 value by SW */
} __attribute__((aligned(SHIM_XSTATE_ALIGN)));

#define SHIM_TCB_GET(member)                                            \
    ({                                                                  \
        shim_tcb_t* tcb;                                                \
        __typeof__(tcb->member) ret;                                    \
        static_assert(sizeof(ret) == 8 ||                               \
                      sizeof(ret) == 4 ||                               \
                      sizeof(ret) == 2 ||                               \
                      sizeof(ret) == 1,                                 \
                      "SHIM_TCB_GET can be used only for "              \
                      "8, 4, 2, or 1-byte(s) members");                 \
        __asm__("mov %%gs:%c1, %0\n"                                    \
                : "=r"(ret)                                             \
                : "i" (offsetof(PAL_TCB, libos_tcb) +                   \
                       offsetof(shim_tcb_t, member))                    \
                : "memory");                                            \
        ret;                                                            \
    })

#define SHIM_TCB_SET(member, value)                                     \
    do {                                                                \
        shim_tcb_t* tcb;                                                \
        static_assert(sizeof(tcb->member) == 8 ||                       \
                      sizeof(tcb->member) == 4 ||                       \
                      sizeof(tcb->member) == 2 ||                       \
                      sizeof(tcb->member) == 1,                         \
                      "SHIM_TCB_SET can be used only for "              \
                      "8, 4, 2, or 1-byte(s) members");                 \
        switch (sizeof(tcb->member)) {                                  \
        case 8:                                                         \
            __asm__("movq %0, %%gs:%c1\n"                               \
                    :: "ir"(value),                                     \
                     "i"(offsetof(PAL_TCB, libos_tcb) +                 \
                         offsetof(shim_tcb_t, member))                  \
                    : "memory");                                        \
            break;                                                      \
        case 4:                                                         \
            __asm__("movl %0, %%gs:%c1\n"                               \
                    :: "ir"(value),                                     \
                     "i"(offsetof(PAL_TCB, libos_tcb) +                 \
                         offsetof(shim_tcb_t, member))                  \
                    : "memory");                                        \
            break;                                                      \
        case 2:                                                         \
            __asm__("movw %0, %%gs:%c1\n"                               \
                    :: "ir"(value),                                     \
                     "i"(offsetof(PAL_TCB, libos_tcb) +                 \
                         offsetof(shim_tcb_t, member))                  \
                    : "memory");                                        \
            break;                                                      \
        case 1:                                                         \
            __asm__("movb %0, %%gs:%c1\n"                               \
                    :: "ir"(value),                                     \
                     "i"(offsetof(PAL_TCB, libos_tcb) +                 \
                         offsetof(shim_tcb_t, member))                  \
                    : "memory");                                        \
            break;                                                      \
        }                                                               \
    } while (0)

static inline void set_tls(uintptr_t tls) {
    DkSegmentBaseSet(PAL_SEGMENT_FS, tls);
}

static inline void set_default_tls(void) {
    set_tls(0);
}

static inline uintptr_t get_tls(void) {
    uintptr_t addr = 0;
    (void)DkSegmentBaseGet(PAL_SEGMENT_FS, &addr);
    return addr;
}

#endif /* _SHIM_TCB_ARCH_H_ */
