/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains APIs for miscellaneous use.
 */

#include <asm/fcntl.h>
#include <stdint.h>

#include "api.h"
#include "cpu.h"
#include "hex.h"
#include "pal.h"
#include "pal_error.h"
#include "pal_internal.h"
#include "pal_linux.h"
#include "seqlock.h"
#include "sgx_api.h"
#include "sgx_attest.h"
#include "spinlock.h"
#include "toml_utils.h"
#include "topo_info.h"

/* The timeout of 50ms was found to be a safe TSC drift correction periodicity based on results
 * from multiple systems. Any higher or lower could pose risks of negative time drift or
 * performance hit respectively.
 */
#define TSC_REFINE_INIT_TIMEOUT_USECS 50000

#define CPU_VENDOR_LEAF     0x0
#define AMX_TILE_INFO_LEAF  0x1D
#define AMX_TMUL_INFO_LEAF  0x1E

uint64_t g_tsc_hz = 0; /* TSC frequency for fast and accurate time ("invariant TSC" HW feature) */
static uint64_t g_start_tsc = 0;
static uint64_t g_start_usec = 0;
static seqlock_t g_tsc_lock = INIT_SEQLOCK_UNLOCKED;

/**
 * Initialize the data structures used for date/time emulation using TSC
 */
void init_tsc(void) {
    if (is_tsc_usable()) {
        g_tsc_hz = get_tsc_hz();
    }
}

/* TODO: result comes from the untrusted host, introduce some schielding */
int _DkSystemTimeQuery(uint64_t* out_usec) {
    int ret;

    if (!g_tsc_hz) {
        /* RDTSC is not allowed or no Invariant TSC feature -- fallback to the slow ocall */
        return ocall_gettime(out_usec);
    }

    uint32_t seq;
    uint64_t start_tsc;
    uint64_t start_usec;
    do {
        seq = read_seqbegin(&g_tsc_lock);
        start_tsc  = g_start_tsc;
        start_usec = g_start_usec;
    } while (read_seqretry(&g_tsc_lock, seq));

    uint64_t usec = 0;
    if (start_tsc > 0 && start_usec > 0) {
        /* baseline TSC/usec pair was initialized, can calculate time via RDTSC (but should be
         * careful with integer overflow during calculations) */
        uint64_t diff_tsc = get_tsc() - start_tsc;
        if (diff_tsc < UINT64_MAX / 1000000) {
            uint64_t diff_usec = diff_tsc * 1000000 / g_tsc_hz;
            if (diff_usec < TSC_REFINE_INIT_TIMEOUT_USECS) {
                /* less than TSC_REFINE_INIT_TIMEOUT_USECS passed from the previous update of
                 * TSC/usec pair (time drift is contained), use the RDTSC-calculated time */
                usec = start_usec + diff_usec;
                if (usec < start_usec)
                    return -PAL_ERROR_OVERFLOW;
            }
        }
    }

    if (usec) {
        *out_usec = usec;
        return 0;
    }

    /* if we are here, either the baseline TSC/usec pair was not yet initialized or too much time
     * passed since the previous TSC/usec update, so let's refresh them to contain the time drift */
    uint64_t tsc_cyc1 = get_tsc();
    ret = ocall_gettime(&usec);
    if (ret < 0)
        return -PAL_ERROR_DENIED;
    uint64_t tsc_cyc2 = get_tsc();

    /* we need to match the OCALL-obtained timestamp (`usec`) with the RDTSC-obtained number of
     * cycles (`tsc_cyc`); since OCALL is a time-consuming operation, we estimate `tsc_cyc` as a
     * mid-point between the RDTSC values obtained right-before and right-after the OCALL. */
    uint64_t tsc_cyc = tsc_cyc1 + (tsc_cyc2 - tsc_cyc1) / 2;
    if (tsc_cyc < tsc_cyc1)
        return -PAL_ERROR_OVERFLOW;

    /* refresh the baseline data if no other thread updated g_start_tsc */
    write_seqbegin(&g_tsc_lock);
    if (g_start_tsc < tsc_cyc) {
        g_start_tsc  = tsc_cyc;
        g_start_usec = usec;
    }
    write_seqend(&g_tsc_lock);

    *out_usec = usec;
    return 0;
}

#define CPUID_CACHE_SIZE 64 /* cache only 64 distinct CPUID entries; sufficient for most apps */
static struct pal_cpuid {
    unsigned int leaf, subleaf;
    unsigned int values[4];
} g_pal_cpuid_cache[CPUID_CACHE_SIZE];

static int g_pal_cpuid_cache_top = 0;
static spinlock_t g_cpuid_cache_lock = INIT_SPINLOCK_UNLOCKED;

static int get_cpuid_from_cache(unsigned int leaf, unsigned int subleaf, unsigned int values[4]) {
    int ret = -PAL_ERROR_DENIED;

    spinlock_lock(&g_cpuid_cache_lock);
    for (int i = 0; i < g_pal_cpuid_cache_top; i++) {
        if (g_pal_cpuid_cache[i].leaf == leaf && g_pal_cpuid_cache[i].subleaf == subleaf) {
            values[0] = g_pal_cpuid_cache[i].values[0];
            values[1] = g_pal_cpuid_cache[i].values[1];
            values[2] = g_pal_cpuid_cache[i].values[2];
            values[3] = g_pal_cpuid_cache[i].values[3];
            ret = 0;
            break;
        }
    }
    spinlock_unlock(&g_cpuid_cache_lock);
    return ret;
}

static void add_cpuid_to_cache(unsigned int leaf, unsigned int subleaf, unsigned int values[4]) {
    spinlock_lock(&g_cpuid_cache_lock);

    struct pal_cpuid* chosen = NULL;
    if (g_pal_cpuid_cache_top < CPUID_CACHE_SIZE) {
        for (int i = 0; i < g_pal_cpuid_cache_top; i++) {
            if (g_pal_cpuid_cache[i].leaf == leaf && g_pal_cpuid_cache[i].subleaf == subleaf) {
                /* this CPUID entry is already present in the cache, no need to add */
                break;
            }
        }
        chosen = &g_pal_cpuid_cache[g_pal_cpuid_cache_top++];
    }

    if (chosen) {
        chosen->leaf      = leaf;
        chosen->subleaf   = subleaf;
        chosen->values[0] = values[0];
        chosen->values[1] = values[1];
        chosen->values[2] = values[2];
        chosen->values[3] = values[3];
    }

    spinlock_unlock(&g_cpuid_cache_lock);
}

static inline uint32_t extension_enabled(uint32_t xfrm, uint32_t bit_idx) {
    uint32_t feature_bit = 1U << bit_idx;
    return xfrm & feature_bit;
}

/*!
 * \brief Sanitize untrusted CPUID inputs.
 *
 * \param[in] leaf       CPUID leaf
 * \param[in] subleaf    CPUID subleaf
 * \param[in,out] values untrusted result to sanitize
 *
 * The basic idea is that there are only a handful of extensions and we know the size needed to
 * store each extension's state. Use this to sanitize host's untrusted cpuid output. We also know
 * through xfrm what extensions are enabled inside the enclave.
 */
static void sanitize_cpuid(uint32_t leaf, uint32_t subleaf, uint32_t values[4]) {
    uint64_t xfrm = g_pal_linuxsgx_state.enclave_info.attributes.xfrm;

    if (leaf == CPU_VENDOR_LEAF) {
        /* hardcode the only possible values for SGX PAL */
        values[CPUID_WORD_EBX] = 0x756e6547; /* 'Genu' */
        values[CPUID_WORD_EDX] = 0x49656e69; /* 'ineI' */
        values[CPUID_WORD_ECX] = 0x6c65746e; /* 'ntel' */
    } else if (leaf == EXTENDED_STATE_LEAF) {
        switch (subleaf) {
            case X87:
                /* From the SDM: "EDX:EAX is a bitmap of all the user state components that can be
                 * managed using the XSAVE feature set. A bit can be set in XCR0 if and only if the
                 * corresponding bit is set in this bitmap. Every processor that supports the XSAVE
                 * feature set will set EAX[0] (x87 state) and EAX[1] (SSE state)."
                 *
                 * On EENTER/ERESUME, the system installs xfrm into XCR0. Hence, we return xfrm here
                 * in EAX.
                 */
                values[CPUID_WORD_EAX] = xfrm;

                /* From the SDM: "EBX enumerates the size (in bytes) required by the XSAVE
                 * instruction for an XSAVE area containing all the user state components
                 * corresponding to bits currently set in XCR0."
                 */
                uint32_t xsave_size = 0;
                /* Start from AVX since x87 and SSE are always captured using XSAVE. Also, x87 and
                 * SSE state size is implicitly included in the extension's offset, e.g., AVX's
                 * offset is 576 which includes x87 and SSE state as well as the XSAVE header. */
                for (int i = AVX; i < LAST_CPU_EXTENSION; i++) {
                    if (extension_enabled(xfrm, i)) {
                        xsave_size = g_cpu_extension_offsets[i] + g_cpu_extension_sizes[i];
                    }
                }
                values[CPUID_WORD_EBX] = xsave_size;

                /* From the SDM: "ECX enumerates the size (in bytes) required by the XSAVE
                 * instruction for an XSAVE area containing all the user state components supported
                 * by this processor."
                 *
                 * We are assuming here that inside the enclave, ECX and EBX for leaf 0xD and
                 * subleaf 0x1 should always be identical, while outside they can potentially be
                 * different. Also, outside of SGX EBX can change at runtime, while ECX is a static
                 * property.
                 */
                values[CPUID_WORD_ECX] = values[CPUID_WORD_EBX];
                values[CPUID_WORD_EDX] = 0;

                break;
            case SSE: {
                const uint32_t xsave_legacy_size = 512;
                const uint32_t xsave_header = 64;
                uint32_t save_size_bytes = xsave_legacy_size + xsave_header;

                /* Start with AVX, since x87 and SSE state is already included when initializing
                 * `save_size_bytes`. */
                for (int i = AVX; i < LAST_CPU_EXTENSION; i++) {
                    if (extension_enabled(xfrm, i)) {
                        save_size_bytes += g_cpu_extension_sizes[i];
                    }
                }
                /* EBX reports the actual size occupied by those extensions irrespective of their
                 * offsets within the xsave area.
                 */
                values[CPUID_WORD_EBX] = save_size_bytes;

                break;
            }
            case AVX:
            case MPX_BNDREGS:
            case MPX_BNDCSR:
            case AVX512_OPMASK:
            case AVX512_ZMM256:
            case AVX512_ZMM512:
            case PKRU:
            case AMX_TILECFG:
            case AMX_TILEDATA:
                /*
                 * Sanitize ECX:
                 *   - bit 0 is always clear because all features are user state (in XCR0)
                 *   - bit 1 is always set because all features are located on 64B boundary
                 *   - bit 2 is set only for AMX_TILEDATA (support for XFD faulting)
                 *   - bits 3-31 are reserved and are zeros
                 */
                values[CPUID_WORD_ECX] = 0x2;
                if (subleaf == AMX_TILEDATA)
                    values[CPUID_WORD_ECX] |= 0x4;

                if (values[CPUID_WORD_EDX] != 0) {
                    log_error("Non-null EDX value in Processor Extended State Enum CPUID leaf");
                    _DkProcessExit(1);
                }

                if (extension_enabled(xfrm, subleaf)) {
                    if (values[CPUID_WORD_EAX] != g_cpu_extension_sizes[subleaf] ||
                            values[CPUID_WORD_EBX] != g_cpu_extension_offsets[subleaf]) {
                        log_error("Unexpected values in Processor Extended State Enum CPUID leaf");
                        _DkProcessExit(1);
                    }
                } else {
                    /* SGX enclave doesn't use this CPU extension, pretend it doesn't exist by
                     * forcing EAX ("size in bytes of the save area for an extended state feature")
                     * and EBX ("offset in bytes of this extended state component's save area from
                     * the beginning of the XSAVE/XRSTOR area") to zero */
                    values[CPUID_WORD_EAX] = 0;
                    values[CPUID_WORD_EBX] = 0;
                }
                break;
        }
    } else if (leaf == AMX_TILE_INFO_LEAF) {
        if (subleaf == 0x0) {
            /* EAX = 1DH, ECX = 0: special subleaf, returns EAX=max_palette, EBX=ECX=EDX=0 */
            if (!IS_IN_RANGE_INCL(values[CPUID_WORD_EAX], 1, 16) || values[CPUID_WORD_EBX] != 0
                    || values[CPUID_WORD_ECX] != 0 || values[CPUID_WORD_EDX] != 0) {
                log_error("Unexpected values in Tile Information CPUID Leaf (subleaf=0x0)");
                _DkProcessExit(1);
            }
        } else {
            /* EAX = 1DH, ECX > 0: subleaf for each supported palette, returns palette limits */
            uint32_t total_tile_bytes = values[CPUID_WORD_EAX] & 0xFFFF;
            uint32_t bytes_per_tile = values[CPUID_WORD_EAX] >> 16;
            uint32_t bytes_per_row = values[CPUID_WORD_EBX] & 0xFFFF;
            uint32_t max_names = values[CPUID_WORD_EBX] >> 16; /* (# of tile regs) */
            uint32_t max_rows = values[CPUID_WORD_ECX] & 0xFFFF;
            if (!IS_IN_RANGE_INCL(total_tile_bytes, 1, 0xFFFF)
                    || !IS_IN_RANGE_INCL(bytes_per_tile, 1, 0xFFFF)
                    || !IS_IN_RANGE_INCL(bytes_per_row, 1, 0xFFFF)
                    || !IS_IN_RANGE_INCL(max_names, 1, 256)
                    || !IS_IN_RANGE_INCL(max_rows, 1, 256)
                    || (values[CPUID_WORD_ECX] >> 16) != 0 || values[CPUID_WORD_EDX] != 0) {
                log_error("Unexpected values in Tile Information CPUID Leaf (subleaf=%#x)",
                          subleaf);
                _DkProcessExit(1);
            }
        }
    } else if (leaf == AMX_TMUL_INFO_LEAF) {
        /* EAX = 1EH, ECX = 0: returns TMUL hardware unit limits */
        uint32_t tmul_maxk = values[CPUID_WORD_EBX] & 0xFF; /* (rows or columns) */
        uint32_t tmul_maxn = (values[CPUID_WORD_EBX] >> 8) & 0xFFFF;
        if (!IS_IN_RANGE_INCL(tmul_maxk, 1, 0xFF)
                || !IS_IN_RANGE_INCL(tmul_maxn, 1, 0xFFFF)
                || (values[CPUID_WORD_EBX] >> 24) != 0
                || values[CPUID_WORD_EAX] != 0
                || values[CPUID_WORD_ECX] != 0
                || values[CPUID_WORD_EDX] != 0) {
            log_error("Unexpected values in TMUL Information CPUID Leaf");
            _DkProcessExit(1);
        }
    }
}

struct cpuid_leaf {
    unsigned int leaf;
    bool zero_subleaf; /* if subleaf is not used by this leaf, then CPUID instruction expects it to
                          be explicitly zeroed out (see _DkCpuIdRetrieve() implementation below) */
    bool cache;        /* if leaf + subleaf pair is constant across all cores and sockets, then we
                          can add the returned CPUID values of this pair to the local cache (see
                          _DkCpuIdRetrieve() implementation below) */
};

/* NOTE: some CPUID leaves/subleaves may theoretically return different values when accessed from
 *       different sockets in a multisocket system and thus should not be declared with
 *       `.cache = true` below, but we don't know of any such systems and currently ignore this */
static const struct cpuid_leaf cpuid_known_leaves[] = {
    /* basic CPUID leaf functions start here */
    {.leaf = 0x00, .zero_subleaf = true,  .cache = true},  /* Highest Func Param and Manufacturer */
    {.leaf = 0x01, .zero_subleaf = true,  .cache = false}, /* Processor Info and Feature Bits */
    {.leaf = 0x02, .zero_subleaf = true,  .cache = true},  /* Cache and TLB Descriptor */
    {.leaf = 0x03, .zero_subleaf = true,  .cache = true},  /* Processor Serial Number */
    {.leaf = 0x04, .zero_subleaf = false, .cache = false}, /* Deterministic Cache Parameters */
    {.leaf = 0x05, .zero_subleaf = true,  .cache = true},  /* MONITOR/MWAIT */
    {.leaf = 0x06, .zero_subleaf = true,  .cache = true},  /* Thermal and Power Management */
    {.leaf = 0x07, .zero_subleaf = false, .cache = true},  /* Structured Extended Feature Flags */
    /* NOTE: 0x08 leaf is reserved, see code below */
    {.leaf = 0x09, .zero_subleaf = true,  .cache = true},  /* Direct Cache Access Information */
    {.leaf = 0x0A, .zero_subleaf = true,  .cache = true},  /* Architectural Performance Monitoring */
    {.leaf = 0x0B, .zero_subleaf = false, .cache = false}, /* Extended Topology Enumeration */
    /* NOTE: 0x0C leaf is reserved, see code below */
    {.leaf = 0x0D, .zero_subleaf = false, .cache = true},  /* Processor Extended State Enumeration */
    /* NOTE: 0x0E leaf is reserved, see code below */
    {.leaf = 0x0F, .zero_subleaf = false, .cache = true},  /* Intel RDT Monitoring */
    {.leaf = 0x10, .zero_subleaf = false, .cache = true},  /* RDT/L2/L3 Cache Allocation Tech */
    /* NOTE: 0x11 leaf is reserved, see code below */
    {.leaf = 0x12, .zero_subleaf = false, .cache = true},  /* Intel SGX Capability */
    /* NOTE: 0x13 leaf is reserved, see code below */
    {.leaf = 0x14, .zero_subleaf = false, .cache = true},  /* Intel Processor Trace Enumeration */
    {.leaf = 0x15, .zero_subleaf = true,  .cache = true},  /* Time Stamp Counter/Core Clock */
    {.leaf = 0x16, .zero_subleaf = true,  .cache = true},  /* Processor Frequency Information */
    {.leaf = 0x17, .zero_subleaf = false, .cache = true},  /* System-On-Chip Vendor Attribute */
    {.leaf = 0x18, .zero_subleaf = false, .cache = true},  /* Deterministic Address Translation */
    {.leaf = 0x19, .zero_subleaf = true,  .cache = true},  /* Key Locker */
    {.leaf = 0x1A, .zero_subleaf = true,  .cache = false}, /* Hybrid Information Enumeration */
    {.leaf = 0x1B, .zero_subleaf = false, .cache = false}, /* PCONFIG Information */
    /* NOTE: 0x1C leaf is not recognized, see code below */
    {.leaf = 0x1D, .zero_subleaf = false, .cache = true},  /* Tile Information Main Leaf (AMX) */
    {.leaf = 0x1E, .zero_subleaf = false, .cache = true},  /* TMUL Information Main Leaf (AMX) */
    {.leaf = 0x1F, .zero_subleaf = false, .cache = false}, /* Intel V2 Ext Topology Enumeration */
    /* basic CPUID leaf functions end here */

    /* invalid CPUID leaf functions (no existing or future CPU will return any meaningful
     * information in these leaves) occupy 40000000 - 4FFFFFFFH -- they are treated the same as
     * unrecognized leaves, see code below */

    /* extended CPUID leaf functions start here */
    {.leaf = 0x80000000, .zero_subleaf = true, .cache = true}, /* Get Highest Extended Function */
    {.leaf = 0x80000001, .zero_subleaf = true, .cache = true}, /* Extended Processor Info */
    {.leaf = 0x80000002, .zero_subleaf = true, .cache = true}, /* Processor Brand String 1 */
    {.leaf = 0x80000003, .zero_subleaf = true, .cache = true}, /* Processor Brand String 2 */
    {.leaf = 0x80000004, .zero_subleaf = true, .cache = true}, /* Processor Brand String 3 */
    {.leaf = 0x80000005, .zero_subleaf = true, .cache = true}, /* L1 Cache and TLB Identifiers */
    {.leaf = 0x80000006, .zero_subleaf = true, .cache = true}, /* Extended L2 Cache Features */
    {.leaf = 0x80000007, .zero_subleaf = true, .cache = true}, /* Advanced Power Management */
    {.leaf = 0x80000008, .zero_subleaf = true, .cache = true}, /* Virtual/Physical Address Sizes */
    /* extended CPUID leaf functions end here */
};

int _DkCpuIdRetrieve(uint32_t leaf, uint32_t subleaf, uint32_t values[4]) {
    uint64_t xfrm = g_pal_linuxsgx_state.enclave_info.attributes.xfrm;

    /* A few basic leaves are considered reserved and always return zeros; see corresponding EAX
     * cases in the "Operation" section of CPUID description in Intel SDM, Vol. 2A, Chapter 3.2.
     *
     * NOTE: Leaves 0x11 and 0x13 are not marked as reserved in Intel SDM but the actual CPUs return
     *       all-zeros on them (as if these leaves are reserved). It is unclear why this discrepancy
     *       exists, but we decided to emulate how actual CPUs behave. */
    if (leaf == 0x08 || leaf == 0x0C || leaf == 0x0E || leaf == 0x11 || leaf == 0x13) {
        values[0] = 0;
        values[1] = 0;
        values[2] = 0;
        values[3] = 0;
        return 0;
    }

    const struct cpuid_leaf* known_leaf = NULL;
    for (size_t i = 0; i < ARRAY_SIZE(cpuid_known_leaves); i++) {
        if (leaf == cpuid_known_leaves[i].leaf) {
            known_leaf = &cpuid_known_leaves[i];
            break;
        }
    }

    if ((!extension_enabled(xfrm, AMX_TILECFG) || !extension_enabled(xfrm, AMX_TILEDATA)) &&
            (leaf == AMX_TILE_INFO_LEAF || leaf == AMX_TMUL_INFO_LEAF)) {
        /* the Intel AMX feature is disabled, so we pretend that the CPU doesn't support it at all
         * (by marking the TILE_INFO and TMUL_INFO AMX-related leaves as unrecognized) */
        known_leaf = NULL;
    }

    if (!known_leaf) {
        /* leaf is not recognized (EAX value is outside of recongized range for CPUID), return info
         * for highest basic information leaf (see cpuid_known_leaves table); also if the highest
         * basic information leaf data depend on the ECX input value (subleaf), ECX is honored; see
         * the DEFAULT case in the "Operation" section of CPUID description in Intel SDM, Vol. 2A,
         * Chapter 3.2 */
        leaf = 0x1F;
        for (size_t i = 0; i < ARRAY_SIZE(cpuid_known_leaves); i++) {
            if (leaf == cpuid_known_leaves[i].leaf) {
                known_leaf = &cpuid_known_leaves[i];
                break;
            }
        }
    }

    /* FIXME: these leaves may have more subleaves in the future, we need a better way of
     *        restricting subleaves (e.g., decide based on CPUID leaf 0x01) */
    if ((leaf == 0x07 && subleaf != 0 && subleaf != 1) ||
        (leaf == 0x0F && subleaf != 0 && subleaf != 1) ||
        (leaf == 0x10 && subleaf != 0 && subleaf != 1 && subleaf != 2 && subleaf != 3) ||
        (leaf == 0x14 && subleaf != 0 && subleaf != 1)) {
        /* leaf-specific checks: some leaves have only specific subleaves */
        goto fail;
    }

    if (known_leaf->zero_subleaf)
        subleaf = 0;

    if (known_leaf->cache && !get_cpuid_from_cache(leaf, subleaf, values))
        return 0;

    if (ocall_cpuid(leaf, subleaf, values) < 0)
        return -PAL_ERROR_DENIED;

    sanitize_cpuid(leaf, subleaf, values);

    if (known_leaf->cache)
        add_cpuid_to_cache(leaf, subleaf, values);

    return 0;
fail:
    log_error("Unrecognized leaf/subleaf in CPUID (EAX=0x%x, ECX=0x%x). Exiting...", leaf, subleaf);
    _DkProcessExit(1);
}

int _DkAttestationReport(const void* user_report_data, PAL_NUM* user_report_data_size,
                         void* target_info, PAL_NUM* target_info_size, void* report,
                         PAL_NUM* report_size) {
    __sgx_mem_aligned sgx_report_data_t stack_report_data = {0};
    __sgx_mem_aligned sgx_target_info_t stack_target_info = {0};
    __sgx_mem_aligned sgx_report_t stack_report = {0};

    if (!user_report_data_size || !target_info_size || !report_size)
        return -PAL_ERROR_INVAL;

    if (*user_report_data_size != sizeof(stack_report_data) ||
        *target_info_size != sizeof(stack_target_info) || *report_size != sizeof(stack_report)) {
        /* inform the caller of SGX sizes for user_report_data, target_info, and report */
        goto out;
    }

    if (!user_report_data || !target_info) {
        /* cannot produce report without user_report_data or target_info */
        goto out;
    }

    bool populate_target_info = false;
    if (!memcmp(target_info, &stack_target_info, sizeof(stack_target_info))) {
        /* caller supplied all-zero target_info, wants to get this enclave's target info */
        populate_target_info = true;
    }

    memcpy(&stack_report_data, user_report_data, sizeof(stack_report_data));
    memcpy(&stack_target_info, target_info, sizeof(stack_target_info));

    int ret = sgx_report(&stack_target_info, &stack_report_data, &stack_report);
    if (ret < 0) {
        /* caller already provided reasonable sizes, so just error out without updating them */
        return -PAL_ERROR_INVAL;
    }

    if (populate_target_info) {
        sgx_target_info_t* ti = (sgx_target_info_t*)target_info;
        memcpy(&ti->attributes, &stack_report.body.attributes, sizeof(ti->attributes));
        memcpy(&ti->config_id, &stack_report.body.config_id, sizeof(ti->config_id));
        memcpy(&ti->config_svn, &stack_report.body.config_svn, sizeof(ti->config_svn));
        memcpy(&ti->misc_select, &stack_report.body.misc_select, sizeof(ti->misc_select));
        memcpy(&ti->mr_enclave, &stack_report.body.mr_enclave, sizeof(ti->mr_enclave));
    }

    if (report) {
        /* report may be NULL if caller only wants to know the size of target_info and/or report */
        memcpy(report, &stack_report, sizeof(stack_report));
    }

out:
    *user_report_data_size = sizeof(stack_report_data);
    *target_info_size      = sizeof(stack_target_info);
    *report_size           = sizeof(stack_report);
    return 0;
}

int _DkAttestationQuote(const void* user_report_data, PAL_NUM user_report_data_size,
                        void* quote, PAL_NUM* quote_size) {
    if (user_report_data_size != sizeof(sgx_report_data_t))
        return -PAL_ERROR_INVAL;

    int ret;
    bool is_epid;
    sgx_spid_t spid = {0};
    bool linkable;

    /* read sgx.ra_client_spid from manifest (must be hex string) */
    char* ra_client_spid_str = NULL;
    ret = toml_string_in(g_pal_public_state.manifest_root, "sgx.ra_client_spid",
                         &ra_client_spid_str);
    if (ret < 0) {
        log_error("Cannot parse 'sgx.ra_client_spid'");
        return -PAL_ERROR_INVAL;
    }

    if (!ra_client_spid_str || strlen(ra_client_spid_str) == 0) {
        /* No Software Provider ID (SPID) specified in the manifest, it is DCAP attestation --
         * for DCAP, spid and linkable arguments are ignored (we unset them for sanity) */
        is_epid = false;
        linkable = false;
    } else {
        /* SPID specified in the manifest, it is EPID attestation -- read spid and linkable */
        is_epid = true;

        if (strlen(ra_client_spid_str) != sizeof(sgx_spid_t) * 2) {
            log_error("Malformed 'sgx.ra_client_spid' value in the manifest: %s",
                      ra_client_spid_str);
            free(ra_client_spid_str);
            return -PAL_ERROR_INVAL;
        }

        for (size_t i = 0; i < strlen(ra_client_spid_str); i++) {
            int8_t val = hex2dec(ra_client_spid_str[i]);
            if (val < 0) {
                log_error("Malformed 'sgx.ra_client_spid' value in the manifest: %s",
                          ra_client_spid_str);
                free(ra_client_spid_str);
                return -PAL_ERROR_INVAL;
            }
            spid[i / 2] = spid[i / 2] * 16 + (uint8_t)val;
        }

        /* read sgx.ra_client_linkable from manifest */
        ret = toml_bool_in(g_pal_public_state.manifest_root, "sgx.ra_client_linkable",
                           /*defaultval=*/false, &linkable);
        if (ret < 0) {
            log_error("Cannot parse 'sgx.ra_client_linkable' (the value must be `true` or "
                      "`false`)");
            free(ra_client_spid_str);
            return -PAL_ERROR_INVAL;
        }
    }

    free(ra_client_spid_str);

    sgx_quote_nonce_t nonce;
    ret = _DkRandomBitsRead(&nonce, sizeof(nonce));
    if (ret < 0)
        return ret;

    char* pal_quote       = NULL;
    size_t pal_quote_size = 0;

    ret = sgx_get_quote(is_epid ? &spid : NULL, &nonce, user_report_data, linkable, &pal_quote,
                        &pal_quote_size);
    if (ret < 0)
        return ret;

    if (*quote_size < pal_quote_size) {
        *quote_size = pal_quote_size;
        free(pal_quote);
        return -PAL_ERROR_NOMEM;
    }

    if (quote) {
        /* quote may be NULL if caller only wants to know the size of the quote */
        assert(pal_quote);
        memcpy(quote, pal_quote, pal_quote_size);
    }

    *quote_size = pal_quote_size;
    free(pal_quote);
    return 0;
}

int _DkGetSpecialKey(const char* name, void* key, size_t* key_size) {
    sgx_key_128bit_t sgx_key;

    if (*key_size < sizeof(sgx_key))
        return -PAL_ERROR_INVAL;

    int ret;
    if (!strcmp(name, "_sgx_mrenclave")) {
        ret = sgx_get_seal_key(SGX_KEYPOLICY_MRENCLAVE, &sgx_key);
    } else if (!strcmp(name, "_sgx_mrsigner")) {
        ret = sgx_get_seal_key(SGX_KEYPOLICY_MRSIGNER, &sgx_key);
    } else {
        return -PAL_ERROR_NOTIMPLEMENTED;
    }
    if (ret < 0)
        return ret;

    memcpy(key, &sgx_key, sizeof(sgx_key));
    *key_size = sizeof(sgx_key);
    return 0;
}

/* Rest is moved from old `db_main-x86_64.c`. */

#define CPUID_LEAF_INVARIANT_TSC 0x80000007
#define CPUID_LEAF_TSC_FREQ 0x15
#define CPUID_LEAF_PROC_FREQ 0x16

ssize_t read_file_buffer(const char* filename, char* buf, size_t buf_size) {
    int fd;

    fd = ocall_open(filename, O_RDONLY, 0);
    if (fd < 0)
        return fd;

    ssize_t n = ocall_read(fd, buf, buf_size);
    ocall_close(fd);

    return n;
}

bool is_tsc_usable(void) {
    uint32_t words[CPUID_WORD_NUM];
    _DkCpuIdRetrieve(CPUID_LEAF_INVARIANT_TSC, 0, words);
    return words[CPUID_WORD_EDX] & 1 << 8;
}

/* return TSC frequency or 0 if invariant TSC is not supported */
uint64_t get_tsc_hz(void) {
    uint32_t words[CPUID_WORD_NUM];

    _DkCpuIdRetrieve(CPUID_LEAF_TSC_FREQ, 0, words);
    if (!words[CPUID_WORD_EAX] || !words[CPUID_WORD_EBX]) {
        /* TSC/core crystal clock ratio is not enumerated, can't use RDTSC for accurate time */
        return 0;
    }

    if (words[CPUID_WORD_ECX] > 0) {
        /* calculate TSC frequency as core crystal clock frequency (EAX) * EBX / EAX; cast to 64-bit
         * first to prevent integer overflow */
        uint64_t ecx_hz = words[CPUID_WORD_ECX];
        return ecx_hz * words[CPUID_WORD_EBX] / words[CPUID_WORD_EAX];
    }

    /* some Intel CPUs do not report nominal frequency of crystal clock, let's calculate it
     * based on Processor Frequency Information Leaf (CPUID 16H); this leaf always exists if
     * TSC Frequency Leaf exists; logic is taken from Linux 5.11's arch/x86/kernel/tsc.c */
    _DkCpuIdRetrieve(CPUID_LEAF_PROC_FREQ, 0, words);
    if (!words[CPUID_WORD_EAX]) {
        /* processor base frequency (in MHz) is not enumerated, can't calculate frequency */
        return 0;
    }

    /* processor base frequency is in MHz but we need to return TSC frequency in Hz; cast to 64-bit
     * first to prevent integer overflow */
    uint64_t base_frequency_mhz = words[CPUID_WORD_EAX];
    return base_frequency_mhz * 1000000;
}

int _DkRandomBitsRead(void* buffer, size_t size) {
    uint32_t rand;
    for (size_t i = 0; i < size; i += sizeof(rand)) {
        rand = rdrand();
        memcpy(buffer + i, &rand, MIN(sizeof(rand), size - i));
    }
    return 0;
}

int _DkSegmentBaseGet(enum pal_segment_reg reg, uintptr_t* addr) {
    switch (reg) {
        case PAL_SEGMENT_FS:
            *addr = GET_ENCLAVE_TLS(fsbase);
            return 0;
        case PAL_SEGMENT_GS:
            /* GS is internally used, deny any access to it */
            return -PAL_ERROR_DENIED;
        default:
            return -PAL_ERROR_INVAL;
    }
}

int _DkSegmentBaseSet(enum pal_segment_reg reg, uintptr_t addr) {
    switch (reg) {
        case PAL_SEGMENT_FS:
            SET_ENCLAVE_TLS(fsbase, addr);
            wrfsbase((uint64_t)addr);
            return 0;
        case PAL_SEGMENT_GS:
            /* GS is internally used, deny any access to it */
            return -PAL_ERROR_DENIED;
        default:
            return -PAL_ERROR_INVAL;
    }
}
