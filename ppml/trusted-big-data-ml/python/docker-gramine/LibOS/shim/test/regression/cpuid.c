/* Sanity checks on values returned by CPUID. */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct regs {
    uint32_t eax;
    uint32_t ebx;
    uint32_t ecx;
    uint32_t edx;
};

static void clear_regs(struct regs* r) {
    r->eax = 0x0;
    r->ebx = 0x0;
    r->ecx = 0x0;
    r->edx = 0x0;
}

static void set_dummy_regs(struct regs* r) {
    r->eax = 0xdead;
    r->ebx = 0xbeef;
    r->ecx = 0xdeaf;
    r->edx = 0xbabe;
}

static bool are_dummy_regs(struct regs* r) {
    return r->eax == 0xdead && r->ebx == 0xbeef && r->ecx == 0xdeaf && r->edx == 0xbabe;
}

static void cpuid(uint32_t leaf, uint32_t subleaf, struct regs* r) {
    __asm__ volatile("cpuid"
                     : "=a"(r->eax), "=b"(r->ebx), "=c"(r->ecx), "=d"(r->edx)
                     : "0"(leaf), "2"(subleaf));
}

static void test_cpuid_leaf_0xd(void) {
    struct regs r = {0, };

    const uint32_t leaf = 0xd;
    const uint32_t extension_unavailable = 0;

    // Sub-leaf IDs for the various extensions.
    enum cpu_extension {
        X87, SSE, AVX, MPX_BNDREGS, MPX_BNDCSR, AVX512_OPMASK, AVX512_ZMM256, AVX512_ZMM512,
        PKRU = 9,
        AMX_TILECFG = 17, AMX_TILEDATA,
    };
    const uint32_t extension_sizes_bytes[] = {
        [AVX] = 256,
        [MPX_BNDREGS] = 64, [MPX_BNDCSR] = 64,
        [AVX512_OPMASK] = 64, [AVX512_ZMM256] = 512, [AVX512_ZMM512] = 1024,
        [PKRU] = 8,
        [AMX_TILECFG] = 64, [AMX_TILEDATA] = 8192,
    };

    cpuid(leaf, AVX, &r);
    if (!(r.eax == extension_unavailable || r.eax == extension_sizes_bytes[AVX]))
        abort();
    clear_regs(&r);

    cpuid(leaf, MPX_BNDREGS, &r);
    if (!(r.eax == extension_unavailable || r.eax == extension_sizes_bytes[MPX_BNDREGS]))
        abort();
    clear_regs(&r);

    cpuid(leaf, MPX_BNDCSR, &r);
    if (!(r.eax == extension_unavailable || r.eax == extension_sizes_bytes[MPX_BNDCSR]))
        abort();
    clear_regs(&r);

    cpuid(leaf, AVX512_OPMASK, &r);
    if (!(r.eax == extension_unavailable || r.eax == extension_sizes_bytes[AVX512_OPMASK]))
        abort();
    clear_regs(&r);

    cpuid(leaf, AVX512_ZMM256, &r);
    if (!(r.eax == extension_unavailable || r.eax == extension_sizes_bytes[AVX512_ZMM256]))
        abort();
    clear_regs(&r);

    cpuid(leaf, AVX512_ZMM512, &r);
    if (!(r.eax == extension_unavailable || r.eax == extension_sizes_bytes[AVX512_ZMM512]))
        abort();
    clear_regs(&r);

    cpuid(leaf, PKRU, &r);
    if (!(r.eax == extension_unavailable || r.eax == extension_sizes_bytes[PKRU]))
        abort();
    clear_regs(&r);

    cpuid(leaf, AMX_TILECFG, &r);
    if (!(r.eax == extension_unavailable || r.eax == extension_sizes_bytes[AMX_TILECFG]))
        abort();
    clear_regs(&r);

    cpuid(leaf, AMX_TILEDATA, &r);
    if (!(r.eax == extension_unavailable || r.eax == extension_sizes_bytes[AMX_TILEDATA]))
        abort();
}

static void test_cpuid_leaf_reserved(void) {
    /* Gramine returns all zeros for reserved CPUID leaves */
    struct regs r;
    set_dummy_regs(&r);

    cpuid(0x8, 0x0, &r); /* subleaf value doesn't matter */
    if (r.eax || r.ebx || r.ecx || r.edx)
        abort();
    set_dummy_regs(&r);

    cpuid(0xE, 0x42, &r); /* subleaf value doesn't matter */
    if (r.eax || r.ebx || r.ecx || r.edx)
        abort();
}

static void test_cpuid_leaf_not_recognized(void) {
    /* in case of unrecognized leaves, Gramine returns info for highest basic information leaf */
    struct regs r;
    set_dummy_regs(&r);

    cpuid(0x1b, 0x0, &r);
    /* return values may be anything (including all-zeros), so just check that it's not dummy */
    if (are_dummy_regs(&r))
        abort();
    set_dummy_regs(&r);

    /* range 0x40000000 - 0x4FFFFFFF is called "invalid" in Intel SDM, but in reality these leaves
     * are treated same as unrecognized leaves */
    cpuid(0x40000000, 0x0, &r);
    /* return values may be anything (including all-zeros), so just check that it's not dummy */
    if (are_dummy_regs(&r))
        abort();
    set_dummy_regs(&r);

    cpuid(0x4FFFFFFF, 0x0, &r);
    /* return values may be anything (including all-zeros), so just check that it's not dummy */
    if (are_dummy_regs(&r))
        abort();
}

int main(int argc, char** argv, char** envp) {
    test_cpuid_leaf_0xd();
    test_cpuid_leaf_reserved();
    test_cpuid_leaf_not_recognized();
    printf("CPUID test passed.\n");
    return 0;
}
