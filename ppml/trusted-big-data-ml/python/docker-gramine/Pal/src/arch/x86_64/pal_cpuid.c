/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2022 Intel Corporation
 *                    Michał Kowalczyk <mkow@invisiblethingslab.com>
 */

#include "pal_internal.h"

#define BPI  32
#define POWER2(power) \
    (1ULL << (power))
#define RIGHTMASK(width) \
    (((unsigned long)(width) >= BPI) ? ~0ULL : POWER2(width) - 1ULL)

#define BIT_EXTRACT_LE(value, start, after) \
    (((unsigned long)(value) & RIGHTMASK(after)) >> start)

#define FOUR_CHARS_VALUE(s, w)      \
    (s)[0] = (w) & 0xff;            \
    (s)[1] = ((w) >>  8) & 0xff;    \
    (s)[2] = ((w) >> 16) & 0xff;    \
    (s)[3] = ((w) >> 24) & 0xff;

static const char* const g_cpu_flags[] = {
    "fpu",    // "x87 FPU on chip"
    "vme",    // "virtual-8086 mode enhancement"
    "de",     // "debugging extensions"
    "pse",    // "page size extensions"
    "tsc",    // "time stamp counter"
    "msr",    // "RDMSR and WRMSR support"
    "pae",    // "physical address extensions"
    "mce",    // "machine check exception"
    "cx8",    // "CMPXCHG8B inst."
    "apic",   // "APIC on chip"
    NULL,
    "sep",    // "SYSENTER and SYSEXIT"
    "mtrr",   // "memory type range registers"
    "pge",    // "PTE global bit"
    "mca",    // "machine check architecture"
    "cmov",   // "conditional move/compare instruction"
    "pat",    // "page attribute table"
    "pse36",  // "page size extension"
    "pn",     // "processor serial number"
    "clflush",    // "CLFLUSH instruction"
    NULL,
    "dts",    // "debug store"
    "acpi",   // "Onboard thermal control"
    "mmx",    // "MMX Technology"
    "fxsr",   // "FXSAVE/FXRSTOR"
    "sse",    // "SSE extensions"
    "sse2",   // "SSE2 extensions"
    "ss",     // "self snoop"
    "ht",     // "hyper-threading / multi-core supported"
    "tm",     // "therm. monitor"
    "ia64",   // "IA64"
    "pbe",    // "pending break event"
};

int _DkGetCPUInfo(struct pal_cpu_info* ci) {
    unsigned int words[CPUID_WORD_NUM];
    int rv = 0;
    char* flags = NULL;
    char* brand = NULL;
    char* vendor_id = NULL;

    const size_t VENDOR_ID_SIZE = 13;
    vendor_id = malloc(VENDOR_ID_SIZE);
    if (!vendor_id)
        return -PAL_ERROR_NOMEM;

    _DkCpuIdRetrieve(0, 0, words);
    FOUR_CHARS_VALUE(&vendor_id[0], words[CPUID_WORD_EBX]);
    FOUR_CHARS_VALUE(&vendor_id[4], words[CPUID_WORD_EDX]);
    FOUR_CHARS_VALUE(&vendor_id[8], words[CPUID_WORD_ECX]);
    vendor_id[VENDOR_ID_SIZE - 1] = '\0';
    ci->cpu_vendor = vendor_id;

    const size_t BRAND_SIZE = 49;
    brand = malloc(BRAND_SIZE);
    if (!brand) {
        rv = -PAL_ERROR_NOMEM;
        goto out_err;
    }
    _DkCpuIdRetrieve(0x80000002, 0, words);
    memcpy(&brand[ 0], words, sizeof(unsigned int) * CPUID_WORD_NUM);
    _DkCpuIdRetrieve(0x80000003, 0, words);
    memcpy(&brand[16], words, sizeof(unsigned int) * CPUID_WORD_NUM);
    _DkCpuIdRetrieve(0x80000004, 0, words);
    memcpy(&brand[32], words, sizeof(unsigned int) * CPUID_WORD_NUM);
    brand[BRAND_SIZE - 1] = '\0';
    ci->cpu_brand = brand;

    _DkCpuIdRetrieve(1, 0, words);
    ci->cpu_family   = BIT_EXTRACT_LE(words[CPUID_WORD_EAX], 8, 12);
    ci->cpu_model    = BIT_EXTRACT_LE(words[CPUID_WORD_EAX], 4, 8);
    ci->cpu_stepping = BIT_EXTRACT_LE(words[CPUID_WORD_EAX], 0, 4);

    if (!memcmp(vendor_id, "GenuineIntel", 12) || !memcmp(vendor_id, "AuthenticAMD", 12)) {
        ci->cpu_family += BIT_EXTRACT_LE(words[CPUID_WORD_EAX], 20, 28);
        ci->cpu_model  += BIT_EXTRACT_LE(words[CPUID_WORD_EAX], 16, 20) << 4;
    }

    size_t flen = 0;
    size_t fmax = 80;
    flags = malloc(fmax);
    if (!flags) {
        rv = -PAL_ERROR_NOMEM;
        goto out_err;
    }

    for (int i = 0; i < 32; i++) {
        if (!g_cpu_flags[i])
            continue;

        if (BIT_EXTRACT_LE(words[CPUID_WORD_EDX], i, i + 1)) {
            size_t len = strlen(g_cpu_flags[i]);
            if (flen + len + 1 > fmax) {
                char* new_flags = malloc(fmax * 2);
                if (!new_flags) {
                    rv = -PAL_ERROR_NOMEM;
                    goto out_err;
                }
                memcpy(new_flags, flags, flen);
                free(flags);
                fmax *= 2;
                flags = new_flags;
            }
            memcpy(flags + flen, g_cpu_flags[i], len);
            flen += len;
            flags[flen++] = ' ';
        }
    }

    flags[flen ? flen - 1 : 0] = 0;
    ci->cpu_flags = flags;

    ci->cpu_bogomips = _DkGetBogomips();
    if (ci->cpu_bogomips == 0.0) {
        log_warning("bogomips could not be retrieved, passing 0.0 to the application");
    }

    return 0;

out_err:
    free(flags);
    free(brand);
    free(vendor_id);
    return rv;
}
