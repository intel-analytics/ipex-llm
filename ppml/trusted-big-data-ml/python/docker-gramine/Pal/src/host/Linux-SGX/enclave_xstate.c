/*
 * this function is shamelessly stolen from linux-sgx
 * https://github.com/intel/linux-sgx/blob/9ddec08fb98c1636ed3b1a77bbc4fa3520344ede/sdk/trts/trts_xsave.cpp
 * It has BSD lisence.
 */

/*
 * Copyright (C) 2011-2019 Intel Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of Intel Corporation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "api.h"
#include "pal.h"
#include "pal_linux.h"

int g_xsave_enabled = 0;
uint64_t g_xsave_features = 0;
uint32_t g_xsave_size = 0;

const uint32_t g_cpu_extension_sizes[] = {
    [AVX] = 256,
    [MPX_BNDREGS] = 64, [MPX_BNDCSR] = 64,
    [AVX512_OPMASK] = 64, [AVX512_ZMM256] = 512, [AVX512_ZMM512] = 1024,
    [PKRU] = 8,
    [AMX_TILECFG] = 64, [AMX_TILEDATA] = 8192,
};

/* Note that AVX offset is 576 bytes and MPX_BNDREGS starts at 960. The AVX state size is 256,
 * leaving 128 bytes unaccounted for (a gap between AVX and MPX_BNDREGS). Similarly, there is a gap
 * between PKRU and AMX_TILECFG. */
const uint32_t g_cpu_extension_offsets[] = {
    [AVX] = 576,
    [MPX_BNDREGS] = 960, [MPX_BNDCSR] = 1024,
    [AVX512_OPMASK] = 1088, [AVX512_ZMM256] = 1152, [AVX512_ZMM512] = 1664,
    [PKRU] = 2688,
    [AMX_TILECFG] = 2752, [AMX_TILEDATA] = 2816,
};

/* FXRSTOR only cares about the first 512 bytes, while XRSTOR in compacted mode will ignore
 * the first 512 bytes. */
const uint32_t g_xsave_reset_state[XSAVE_RESET_STATE_SIZE / sizeof(uint32_t)] __attribute__((
    aligned(PAL_XSTATE_ALIGN))) = {
    0x037F, 0, 0, 0, 0, 0, 0x1F80,     0xFFFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,      0, 0, 0, 0, 0, 0,          0,      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,      0, 0, 0, 0, 0, 0,          0,      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,      0, 0, 0, 0, 0, 0,          0,      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,      0, 0, 0, 0, 0, 0,          0,      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,      0, 0, 0, 0, 0, 0x80000000, 0,      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // XCOMP_BV[63] = 1, compaction mode
};

void init_xsave_size(uint64_t xfrm) {
    const struct {
        uint64_t bits;
        uint32_t size;
    } xsave_size_table[] = {
        /* `size` is calculated as the offset of the feature in XSAVE area + size of each
         * sub-feature. Note that g_xsave_size should be in ascending order. */
        {SGX_XFRM_LEGACY, XSAVE_RESET_STATE_SIZE},
        {SGX_XFRM_AVX,    g_cpu_extension_offsets[AVX] + g_cpu_extension_sizes[AVX]},
        {SGX_XFRM_MPX,    g_cpu_extension_offsets[MPX_BNDREGS] +
                              g_cpu_extension_sizes[MPX_BNDREGS] +
                              g_cpu_extension_sizes[MPX_BNDCSR]},
        {SGX_XFRM_AVX512, g_cpu_extension_offsets[AVX512_OPMASK] +
                              g_cpu_extension_sizes[AVX512_OPMASK] +
                              g_cpu_extension_sizes[AVX512_ZMM256] +
                              g_cpu_extension_sizes[AVX512_ZMM512]},
        {SGX_XFRM_PKRU,   g_cpu_extension_offsets[PKRU] + g_cpu_extension_sizes[PKRU]},
        {SGX_XFRM_AMX,    g_cpu_extension_offsets[AMX_TILECFG] +
                              g_cpu_extension_sizes[AMX_TILECFG] +
                              g_cpu_extension_sizes[AMX_TILEDATA]},
    };

    /* fxsave/fxrstore as fallback */
    g_xsave_enabled = 0;
    g_xsave_features = PAL_XFEATURE_MASK_FPSSE;
    g_xsave_size = 512 + 64;
    if (!xfrm || (xfrm & SGX_XFRM_RESERVED)) {
        log_debug("xsave is disabled, xfrm 0x%lx", xfrm);
        return;
    }

    g_xsave_enabled = (xfrm == SGX_XFRM_LEGACY) ? 0 : 1;
    for (size_t i = 0; i < ARRAY_SIZE(xsave_size_table); i++) {
        if ((xfrm & xsave_size_table[i].bits) == xsave_size_table[i].bits) {
            g_xsave_features = xfrm;
            g_xsave_size = xsave_size_table[i].size;
        }
    }
    log_debug("xsave is enabled with g_xsave_size: %u", g_xsave_size);
}
