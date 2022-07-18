/* Copyright (C) 2002 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, write to the Free
   Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
   02111-1307 USA.  */

#ifndef _LINUX_X86_64_BITS_SIGCONTEXT_H
#define _LINUX_X86_64_BITS_SIGCONTEXT_H 1

#include <bits/wordsize.h>
#include <stdint.h>

struct _fpreg {
    unsigned short significand[4];
    unsigned short exponent;
};

struct _fpxreg {
    unsigned short significand[4];
    unsigned short exponent;
    unsigned short padding[3];
};

struct _xmmreg {
    uint32_t element[4];
};

#if __WORDSIZE == 32

struct _fpstate {
    /* Regular FPU environment.  */
    uint32_t cw;
    uint32_t sw;
    uint32_t tag;
    uint32_t ipoff;
    uint32_t cssel;
    uint32_t dataoff;
    uint32_t datasel;
    struct _fpreg _st[8];
    unsigned short status;
    unsigned short magic;

    /* FXSR FPU environment.  */
    uint32_t _fxsr_env[6];
    uint32_t mxcsr;
    uint32_t reserved;
    struct _fpxreg _fxsr_st[8];
    struct _xmmreg _xmm[8];
    uint32_t padding[56];
};

#ifndef sigcontext_struct
/* Kernel headers before 2.1.1 define a struct sigcontext_struct, but
   we need sigcontext.  Some packages have come to rely on
   sigcontext_struct being defined on 32-bit x86, so define this for
   their benefit.  */
#define sigcontext_struct sigcontext
#endif

struct sigcontext {
    unsigned short gs, __gsh;
    unsigned short fs, __fsh;
    unsigned short es, __esh;
    unsigned short ds, __dsh;
    unsigned long edi;
    unsigned long esi;
    unsigned long ebp;
    unsigned long esp;
    unsigned long ebx;
    unsigned long edx;
    unsigned long ecx;
    unsigned long eax;
    unsigned long trapno;
    unsigned long err;
    unsigned long eip;
    unsigned short cs, __csh;
    unsigned long eflags;
    unsigned long esp_at_signal;
    unsigned short ss, __ssh;
    struct _fpstate* fpstate;
    unsigned long oldmask;
    unsigned long cr2;
};

#else /* __WORDSIZE == 64 */

struct _fpstate {
    /* FPU environment matching the 64-bit FXSAVE layout.  */
    uint16_t cwd;
    uint16_t swd;
    uint16_t ftw;
    uint16_t fop;
    uint64_t rip;
    uint64_t rdp;
    uint32_t mxcsr;
    uint32_t mxcr_mask;
    struct _fpxreg _st[8];
    struct _xmmreg _xmm[16];
    uint32_t padding[24];
};

struct sigcontext {
    unsigned long r8;
    unsigned long r9;
    unsigned long r10;
    unsigned long r11;
    unsigned long r12;
    unsigned long r13;
    unsigned long r14;
    unsigned long r15;
    unsigned long rdi;
    unsigned long rsi;
    unsigned long rbp;
    unsigned long rbx;
    unsigned long rdx;
    unsigned long rax;
    unsigned long rcx;
    unsigned long rsp;
    unsigned long rip;
    unsigned long eflags;
    unsigned short cs;
    unsigned short gs;
    unsigned short fs;
    unsigned short __pad0;
    unsigned long err;
    unsigned long trapno;
    unsigned long oldmask;
    unsigned long cr2;
    struct _fpstate* fpstate;
    unsigned long __reserved1[8];
};

#endif /* __WORDSIZE == 64 */

#endif /* _LINUX_X86_64_BITS_SIGCONTEXT_H */
