/* SPDX-License-Identifier: LGPL-3.0-or-later */

/*
 * This file contains Linux-SGX-specific functions related to the PAL.
 */

#ifndef __LINUX_SGX_X86_64_PAL_HOST_ARCH_H__
#define __LINUX_SGX_X86_64_PAL_HOST_ARCH_H__

/* Linux v5.16 supports Intel AMX. To enable this feature, Linux added several XSTATE-related
 * arch_prctl() commands. To support Gramine on older Linux kernels, we explicitly define
 * the required ARCH_REQ_XCOMP_PERM command. See
 * https://elixir.bootlin.com/linux/v5.16/source/arch/x86/include/uapi/asm/prctl.h */
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif

#endif /* __LINUX_SGX_X86_64_PAL_HOST_ARCH_H__ */
