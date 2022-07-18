/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University
 * Copyright (C) 2020 Intel Corporation
 *                    Michał Kowalczyk <mkow@invisiblethingslab.com>
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

#include <asm/prctl.h>

#include "pal.h"
#include "shim_internal.h"
#include "shim_table.h"
#include "shim_tcb.h"

/* Linux v5.16 supports Intel AMX. To enable this feature, Linux added several XSTATE-related
 * arch_prctl() commands. To support Gramine on older Linux kernels, we explicitly define these
 * commands. See
 * https://elixir.bootlin.com/linux/v5.16/source/arch/x86/include/uapi/asm/prctl.h */
#ifndef ARCH_GET_XCOMP_SUPP
#define ARCH_GET_XCOMP_SUPP 0x1021
#endif
#ifndef ARCH_GET_XCOMP_PERM
#define ARCH_GET_XCOMP_PERM 0x1022
#endif
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif

long shim_do_arch_prctl(int code, unsigned long addr) {
    unsigned int values[CPUID_WORD_NUM];
    int ret;

    switch (code) {
        case ARCH_SET_FS:
            set_tls(addr);
            return 0;

        case ARCH_GET_FS:
            return pal_to_unix_errno(DkSegmentBaseGet(PAL_SEGMENT_FS, (unsigned long*)addr));

        /* Emulate ARCH_GET_XCOMP_SUPP, ARCH_GET_XCOMP_PERM, ARCH_REQ_XCOMP_PERM by querying CPUID,
         * it's safe because the PAL already requested AMX permission at startup. Note that
         * supported and currently enabled sets are always the same in Gramine (because PAL always
         * enables all it can at startup). */
        case ARCH_GET_XCOMP_SUPP:
        case ARCH_GET_XCOMP_PERM:
            ret = DkCpuIdRetrieve(EXTENDED_STATE_LEAF, EXTENDED_STATE_SUBLEAF_FEATURES, values);
            if (ret < 0) {
                return pal_to_unix_errno(ret);
            }

            if (!is_user_memory_writable((uint64_t*)addr, sizeof(uint64_t))) {
                return -EFAULT;
            }

            *(uint64_t*)addr = values[CPUID_WORD_EAX] | ((uint64_t)values[CPUID_WORD_EDX] << 32);
            return 0;

        case ARCH_REQ_XCOMP_PERM:
            /* The request must be the highest state component number related to that facility,
             * current Linux kernel supports only AMX_TILEDATA (bit 18) */
            if (addr != AMX_TILEDATA) {
                log_warning("ARCH_REQ_XCOMP_PERM on unsupported feature %lu requested", addr);
                return -EOPNOTSUPP;
            }

            ret = DkCpuIdRetrieve(EXTENDED_STATE_LEAF, EXTENDED_STATE_SUBLEAF_FEATURES, values);
            if (ret < 0) {
                return pal_to_unix_errno(ret);
            }

            if (!(values[CPUID_WORD_EAX] & (1 << AMX_TILEDATA))) {
                log_warning("AMX is not supported on this CPU (XSAVE bits are %#x)",
                            values[CPUID_WORD_EAX]);
                return -EINVAL;
            }

            /* PAL already requested AMX permission at startup, here just a no-op */
            return 0;

        default:
            log_warning("Not supported flag (0x%x) passed to arch_prctl", code);
            return -ENOSYS;
    }
}
