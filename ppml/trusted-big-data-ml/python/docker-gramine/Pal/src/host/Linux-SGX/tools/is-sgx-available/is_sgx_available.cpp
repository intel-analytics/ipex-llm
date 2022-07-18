/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Invisible Things Lab
 *                    Michal Kowalczyk <mkow@invisiblethingslab.com>
 */

#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sys/stat.h>

enum ExitCodes {
    SUCCESS = 0,
    NO_CPU_SUPPORT = 1,
    NO_BIOS_SUPPORT = 2,
    PSW_NOT_INSTALLED = 3,
    AESMD_NOT_INSTALLED = 4
};

bool file_exists(const char* path) {
    struct stat statbuf;
    return stat(path, &statbuf) == 0;
}

void cpuid(uint32_t leaf, uint32_t subleaf, uint32_t* eax, uint32_t* ebx,
           uint32_t* ecx, uint32_t* edx) {
    unsigned long eax_, ebx_, ecx_, edx_;
    __asm__ ("cpuid\n"
        : "=a" (eax_), "=b" (ebx_), "=c" (ecx_), "=d" (edx_)
        : "0" (leaf), "2" (subleaf));
    if (eax) *eax = (eax_ & 0xFFFFFFFF);
    if (ebx) *ebx = (ebx_ & 0xFFFFFFFF);
    if (ecx) *ecx = (ecx_ & 0xFFFFFFFF);
    if (edx) *edx = (edx_ & 0xFFFFFFFF);
}

static bool is_cpuid_supported() {
    // Checks whether (R/E)FLAGS.ID is writable (bit 21).
    uint64_t write_diff;
    __asm__ (
        "pushf\n"

        "pushf\n"
        "xorq $(1<<21), (%%rsp)\n"
        "popf\n"
        "pushf\n"
        "popq %0\n"
        "xorq (%%rsp), %0\n"

        "popf\n"
        : "=r" (write_diff)
    );
    return write_diff;
}

// 2**exp with saturation for unsigned types.
template<class T> T saturating_exp2(T exp) {
    // Protect against UB.
    if (exp >= sizeof(T)*8)
        return ~(T)0;
    // No overflow.
    return (T)1 << exp;
}

class SgxCpuChecker {
    bool cpuid_supported_ = false;
    bool is_intel_cpu_ = false;
    bool sgx_supported_ = false;
    bool sgx1_supported_ = false;
    bool sgx2_supported_ = false;
    bool flc_supported_ = false;
    bool sgx_virt_supported_ = false;
    bool sgx_mem_concurrency_supported_ = false;
    bool cet_supported_ = false;
    bool kss_supported_ = false;
    uint64_t maximum_enclave_size_x86_ = false;
    uint64_t maximum_enclave_size_x64_ = false;
    uint64_t epc_region_size_ = 0;

public:
    explicit SgxCpuChecker() {
        uint32_t cpuid_max_leaf_value;
        uint32_t cpuid_0_ebx;
        uint32_t cpuid_0_ecx;
        uint32_t cpuid_0_edx;

        uint32_t cpuid_7_0_eax;
        uint32_t cpuid_7_0_ebx;
        uint32_t cpuid_7_0_ecx;
        uint32_t cpuid_7_0_edx;
        // Used only if sgx_supported().
        uint32_t cpuid_12_0_eax;
        uint32_t cpuid_12_0_ebx;
        uint32_t cpuid_12_0_ecx;
        uint32_t cpuid_12_0_edx;
        uint32_t cpuid_12_1_eax;
        uint32_t cpuid_12_1_ebx;
        uint32_t cpuid_12_1_ecx;
        uint32_t cpuid_12_1_edx;

        cpuid_supported_ = is_cpuid_supported();
        if (!cpuid_supported_)
            return;
        cpuid(0, 0, &cpuid_max_leaf_value, &cpuid_0_ebx, &cpuid_0_ecx, &cpuid_0_edx);
        is_intel_cpu_ = cpuid_0_ebx == __builtin_bswap32('Genu')
                     && cpuid_0_edx == __builtin_bswap32('ineI')
                     && cpuid_0_ecx == __builtin_bswap32('ntel');
        if (!is_intel_cpu_ || cpuid_max_leaf_value < 7)
            return;
        // See chapter 36.7 in Intel SDM vol.3
        cpuid(7, 0, &cpuid_7_0_eax, &cpuid_7_0_ebx, &cpuid_7_0_ecx, &cpuid_7_0_edx);
        sgx_supported_ = cpuid_7_0_ebx & (1 << 2);
        if (!sgx_supported_ || cpuid_max_leaf_value < 0x12)
            return;
        flc_supported_ = cpuid_7_0_ecx & (1 << 30);
        cpuid(0x12, 0, &cpuid_12_0_eax, &cpuid_12_0_ebx, &cpuid_12_0_ecx, &cpuid_12_0_edx);
        cpuid(0x12, 1, &cpuid_12_1_eax, &cpuid_12_1_ebx, &cpuid_12_1_ecx, &cpuid_12_1_edx);
        sgx1_supported_ = cpuid_12_0_eax & (1 << 0);
        sgx2_supported_ = cpuid_12_0_eax & (1 << 1);
        sgx_virt_supported_ = cpuid_12_0_eax & (1 << 5);
        sgx_mem_concurrency_supported_ = cpuid_12_0_eax & (1 << 6);
        cet_supported_ = cpuid_12_1_eax & (1 << 6);
        kss_supported_ = cpuid_12_1_eax & (1 << 7);
        maximum_enclave_size_x86_ = saturating_exp2<uint64_t>(cpuid_12_0_edx & 0xFF);
        maximum_enclave_size_x64_ = saturating_exp2<uint64_t>((cpuid_12_0_edx >> 8) & 0xFF);
        // Check if there's any EPC region allocated by BIOS
        for (uint32_t subleaf = 2; subleaf >= 2; subleaf++) {
            uint32_t eax, ebx, ecx, edx;
            cpuid(0x12, subleaf, &eax, &ebx, &ecx, &edx);
            auto type = eax & 0xF;
            if (!type)
                break;
            if (type == 1) {
                // EAX and EBX report the physical address of the base of the EPC region,
                // but we only care about the EPC size
                if (ecx & 0xFFFFF000 || edx & 0xFFFFF) {
                    epc_region_size_ += ecx & 0xFFFFF000;
                    epc_region_size_ += (uint64_t)(edx & 0xFFFFF) << 32;
                }
            }
        }
    }

    bool cpuid_supported() const { return cpuid_supported_; }
    bool is_intel_cpu() const { return is_intel_cpu_; }
    bool sgx_supported() const { return sgx_supported_; }
    bool sgx1_supported() const { return sgx1_supported_; }
    // SGX2 enclave dynamic memory management (EDMM) support (EAUG, EACCEPT, EMODPR, ...)
    bool sgx2_supported() const { return sgx2_supported_; }
    // Flexible Launch Control support (IA32_SGXPUBKEYHASH{0..3} MSRs)
    bool flc_supported() const { return flc_supported_; }
    // Extensions for virtualizers (EINCVIRTCHILD, EDECVIRTCHILD, ESETCONTEXT).
    bool sgx_virt_supported() const { return sgx_virt_supported_; }
    // Extensions for concurrent memory management (ETRACKC, ERDINFO, ELDBC, ELDUC).
    bool sgx_mem_concurrency_supported() const { return sgx_mem_concurrency_supported_; }
    // CET enclave attributes support (See Table 37-5 in the SDM)
    bool cet_supported() const { return cet_supported_; }
    // Key separation and sharing (KSS) support (CONFIGID, CONFIGSVN, ISVEXTPRODID, ISVFAMILYID report fields)
    bool kss_supported() const { return kss_supported_; }
    uint64_t maximum_enclave_size_x86() const { return maximum_enclave_size_x86_; }
    uint64_t maximum_enclave_size_x64() const { return maximum_enclave_size_x64_; }
    uint64_t epc_region_size() const { return epc_region_size_; }
};

bool sgx_driver_loaded() {
    return file_exists("/dev/isgx") // LKM version
        || file_exists("/dev/sgx")  // old in-kernel patchset (<= 5.10) or DCAP drivers
        || file_exists("/dev/sgx_enclave"); // upstreamed drivers (>= 5.11)
}

bool psw_installed() {
    bool aesmd_installed = file_exists("/etc/aesmd.conf");
    return sgx_driver_loaded() && aesmd_installed;
}

bool aesmd_installed() {
    return file_exists("/var/run/aesmd/aesm.socket");
}

void print_detailed_info(const SgxCpuChecker& cpu_checker) {
    if (!cpu_checker.cpuid_supported()) {
        puts("`cpuid` instruction not available");
        return;
    }
    if (!cpu_checker.is_intel_cpu()) {
        puts("Not an Intel CPU");
        return;
    }
    auto bool2str = [](bool b){ return b ? "true" : "false"; };
    auto sgx_supported = cpu_checker.sgx_supported();
    printf("SGX supported by CPU: %s\n", bool2str(sgx_supported));
    if (!sgx_supported)
        return;
    printf("SGX1 (ECREATE, EENTER, ...): %s\n", bool2str(cpu_checker.sgx1_supported()));
    printf("SGX2 (EAUG, EACCEPT, EMODPR, ...): %s\n", bool2str(cpu_checker.sgx2_supported()));
    printf("Flexible Launch Control (IA32_SGXPUBKEYHASH{0..3} MSRs): %s\n",
           bool2str(cpu_checker.flc_supported()));
    printf("SGX extensions for virtualizers (EINCVIRTCHILD, EDECVIRTCHILD, ESETCONTEXT): %s\n",
           bool2str(cpu_checker.sgx_virt_supported()));
    printf("Extensions for concurrent memory management (ETRACKC, ELDBC, ELDUC, ERDINFO): %s\n",
           bool2str(cpu_checker.sgx_mem_concurrency_supported()));
    printf("CET enclave attributes support (See Table 37-5 in the SDM): %s\n",
           bool2str(cpu_checker.cet_supported()));
    printf("Key separation and sharing (KSS) support (CONFIGID, CONFIGSVN, ISVEXTPRODID, "
           "ISVFAMILYID report fields): %s\n", bool2str(cpu_checker.kss_supported()));
    printf("Max enclave size (32-bit): 0x%" PRIx64 "\n", cpu_checker.maximum_enclave_size_x86());
    printf("Max enclave size (64-bit): 0x%" PRIx64 "\n", cpu_checker.maximum_enclave_size_x64());
    printf("EPC size: 0x%" PRIx64 "\n", cpu_checker.epc_region_size());
    printf("SGX driver loaded: %s\n", bool2str(sgx_driver_loaded()));
    printf("AESMD installed: %s\n", bool2str(aesmd_installed()));
    printf("SGX PSW/libsgx installed: %s\n", bool2str(psw_installed()));
}

int main(int argc, char* argv[]) {
    bool quiet = (argc >= 2) && !strcmp(argv[1], "--quiet");

    SgxCpuChecker cpu_checker;
    if (!quiet)
        print_detailed_info(cpu_checker);

    if (!cpu_checker.cpuid_supported() || !cpu_checker.is_intel_cpu()
        || !cpu_checker.sgx_supported() || (!cpu_checker.sgx1_supported()
                                            && !cpu_checker.sgx2_supported()))
        return ExitCodes::NO_CPU_SUPPORT;
    // We currently fail also when only one from 32/64 bit modes is available.
    if (cpu_checker.maximum_enclave_size_x86() == 0
        || cpu_checker.maximum_enclave_size_x64() == 0
        || cpu_checker.epc_region_size() == 0)
        return ExitCodes::NO_BIOS_SUPPORT;
    if (!psw_installed())
        return ExitCodes::PSW_NOT_INSTALLED;
    if (!aesmd_installed())
        return ExitCodes::AESMD_NOT_INSTALLED;
    return ExitCodes::SUCCESS;
}
