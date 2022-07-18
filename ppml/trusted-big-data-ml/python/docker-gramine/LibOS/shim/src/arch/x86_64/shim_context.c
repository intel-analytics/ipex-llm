/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

/*
 * This file contains code for x86_64-specific CPU context manipulation.
 */

#include <stdalign.h>
#include <stdnoreturn.h>

#include "pal.h"
#include "shim_context.h"
#include "shim_entry.h"
#include "shim_internal.h"
#include "shim_thread.h"
#include "ucontext.h"

#define XSTATE_RESET_SIZE (sizeof(struct shim_fpstate))

/* By default fall back to old-style FXSAVE. */
static bool     g_shim_xsave_enabled  = false;
static uint64_t g_shim_xsave_features = SHIM_XFEATURE_MASK_FPSSE;
static uint32_t g_shim_xsave_size     = XSTATE_RESET_SIZE;

static const uint32_t g_shim_xstate_reset_state[XSTATE_RESET_SIZE / sizeof(uint32_t)]
__attribute__((aligned(SHIM_XSTATE_ALIGN))) = {
    0x037F, 0, 0, 0, 0, 0, 0x1F80, 0xFFFF,
};

#define CPUID_FEATURE_XSAVE   (1UL << 26)
#define CPUID_FEATURE_OSXSAVE (1UL << 27)

#define CPUID_LEAF_PROCINFO 0x00000001
#define CPUID_LEAF_XSAVE 0x0000000d

uint64_t shim_xstate_size(void) {
    return g_shim_xsave_size + (g_shim_xsave_enabled ? SHIM_FP_XSTATE_MAGIC2_SIZE : 0);
}

void shim_xstate_init(void) {
    unsigned int value[4];
    if (DkCpuIdRetrieve(CPUID_LEAF_PROCINFO, 0, value) < 0)
        goto out;

    if (!(value[CPUID_WORD_ECX] & CPUID_FEATURE_XSAVE) ||
        !(value[CPUID_WORD_ECX] & CPUID_FEATURE_OSXSAVE))
        goto out;

    if (DkCpuIdRetrieve(CPUID_LEAF_XSAVE, 0, value) < 0)
        goto out;

    uint32_t xsavesize = value[CPUID_WORD_ECX];
    uint64_t xfeatures = value[CPUID_WORD_EAX] | ((uint64_t)value[CPUID_WORD_EDX] << 32);
    if (!xsavesize || !xfeatures) {
        /* could not read xfeatures; fall back to old-style FXSAVE */
        goto out;
    }

    if (xfeatures & ~SHIM_XFEATURE_MASK_FPSSE) {
        /* support more than just FP and SSE, can use XSAVE (it was introduced with AVX) */
        g_shim_xsave_enabled = true;
    }

    g_shim_xsave_features  = xfeatures;
    g_shim_xsave_size      = xsavesize;

out:
    log_debug("LibOS xsave_enabled %d, xsave_size 0x%x(%u), xsave_features 0x%lx",
              g_shim_xsave_enabled, g_shim_xsave_size, g_shim_xsave_size, g_shim_xsave_features);
}

#if 0
/* Currently not used. */
void shim_xstate_save(void* xstate_extended) {
    assert(IS_ALIGNED_PTR(xstate_extended, SHIM_XSTATE_ALIGN));

    struct shim_xstate* xstate = (struct shim_xstate*)xstate_extended;
    char* bytes_after_xstate   = (char*)xstate_extended + g_shim_xsave_size;

    if (g_shim_xsave_enabled) {
        memset(&xstate->xstate_hdr, 0, sizeof(xstate->xstate_hdr));
        __builtin_ia32_xsave64(xstate, /*mask=*/-1LL);
    } else {
        __builtin_ia32_fxsave64(xstate);
    }

    /* Emulate software format for bytes 464..511 in the 512-byte layout of the FXSAVE/FXRSTOR
     * frame that Linux uses for x86-64:
     *   https://elixir.bootlin.com/linux/v5.9/source/arch/x86/kernel/fpu/signal.c#L86
     *   https://elixir.bootlin.com/linux/v5.9/source/arch/x86/kernel/fpu/signal.c#L517
     *
     * This format is assumed by Glibc; this will also be useful if we implement checks in LibOS
     * similar to Linux's check_for_xstate(). Note that we don't care about CPUs older than
     * FXSAVE-enabled (so-called "legacy frames"), therefore we always use MAGIC1 and MAGIC2. */
    struct shim_fpx_sw_bytes* fpx_sw = &xstate->fpstate.sw_reserved;
    fpx_sw->magic1        = SHIM_FP_XSTATE_MAGIC1;
    fpx_sw->extended_size = g_shim_xsave_size + SHIM_FP_XSTATE_MAGIC2_SIZE;
    fpx_sw->xfeatures     = g_shim_xsave_features;
    fpx_sw->xstate_size   = g_shim_xsave_size;
    memset(&fpx_sw->padding, 0, sizeof(fpx_sw->padding));

    /* the last 32-bit word of the extended FXSAVE/XSAVE area (at the xstate + extended_size
     * - FP_XSTATE_MAGIC2_SIZE address) is set to FP_XSTATE_MAGIC2 so that app/Gramine can sanity
     * check FXSAVE/XSAVE size calculations */
    *((__typeof__(SHIM_FP_XSTATE_MAGIC2)*)bytes_after_xstate) = SHIM_FP_XSTATE_MAGIC2;
}
#endif

__attribute__((used)) static int is_xstate_extended(const struct shim_xstate* xstate) {
    assert(IS_ALIGNED_PTR(xstate, SHIM_XSTATE_ALIGN));

    if (!g_shim_xsave_enabled) {
        return 0;
    }

    const struct shim_fpx_sw_bytes* fpx_sw = &xstate->fpstate.sw_reserved;
    if (fpx_sw->magic1 != SHIM_FP_XSTATE_MAGIC1) {
        return 0;
    }
    if (fpx_sw->extended_size > shim_xstate_size()) {
        return 0;
    }
    if (fpx_sw->xfeatures & ~g_shim_xsave_features) {
        return 0;
    }
    if (fpx_sw->xstate_size < sizeof(struct shim_xstate)) {
        return 0;
    }
    if (fpx_sw->xstate_size > g_shim_xsave_size) {
        return 0;
    }
    if (fpx_sw->xstate_size > fpx_sw->extended_size) {
        return 0;
    }
    const void* bytes_after_xstate = (const char*)xstate + fpx_sw->xstate_size;
    if (*(const uint32_t*)bytes_after_xstate != SHIM_FP_XSTATE_MAGIC2) {
        return 0;
    }
    return 1;
}

/* Written in asm because we need to make sure it does not touch FPU/SSE after restoring their
 * state. Older gcc versions do not support `naked` attribute on x86, hence: */
__asm__(
".pushsection .text\n"
".global shim_xstate_restore\n"
".type shim_xstate_restore, @function\n"
"shim_xstate_restore:\n"
    "push %rdi\n"
    "call is_xstate_extended\n"
    "test %eax, %eax\n"
    "pop %rdi\n"
    "je .Lnot_xstate\n"

    "mov $-1, %eax\n"
    "mov %eax, %edx\n"
    "xrstor64 (%rdi)\n"
    "ret\n"

    ".Lnot_xstate:\n"
    "fxrstor64 (%rdi)\n"
    "ret\n"
".popsection\n"
);

/* Copies FPU state. Returns whether the copied state was xsave-made. */
static bool shim_xstate_copy(struct shim_xstate* dst, const struct shim_xstate* src) {
    if (src == NULL) {
        src = (const struct shim_xstate*)g_shim_xstate_reset_state;
    }

    size_t copy_size = sizeof(struct shim_fpstate);
    int src_is_xstate = is_xstate_extended(src);
    if (src_is_xstate) {
        copy_size = src->fpstate.sw_reserved.xstate_size + SHIM_FP_XSTATE_MAGIC2_SIZE;
    }

    memcpy(dst, src, copy_size);

    if (!src_is_xstate) {
        memset(&dst->fpstate.sw_reserved, 0, sizeof(dst->fpstate.sw_reserved));
    }

    return src_is_xstate;
}

noreturn void restore_child_context_after_clone(struct shim_context* context) {
    assert(context->regs);

    /* Set 0 as child return value. */
    context->regs->rax = 0;

    context->syscall_nr = -1;

    set_tls(context->tls);

    PAL_CONTEXT* regs = context->regs;
    context->regs = NULL;

    return_from_syscall(regs);
}

struct sigframe {
    ucontext_t uc;
    siginfo_t siginfo;
};

void prepare_sigframe(PAL_CONTEXT* context, siginfo_t* siginfo, void* handler, void* restorer,
                      bool should_use_altstack, __sigset_t* old_mask) {
    struct shim_thread* current = get_cur_thread();

    uint64_t stack = get_stack_for_sighandler(context->rsp, should_use_altstack);

    struct shim_xstate* xstate = NULL;
    stack = ALIGN_DOWN(stack - shim_xstate_size(), alignof(*xstate));
    xstate = (struct shim_xstate*)stack;

    struct sigframe* sigframe = NULL;
    stack = ALIGN_DOWN(stack - sizeof(*sigframe), alignof(*sigframe));
    /* x64 SysV ABI requires that stack is aligned to 8 mod 0x10 after function call, so we have to
     * mimic that in signal handler. `sigframe` will be aligned to 0x10 and we will push a return
     * value (restorer address) on top of that later on. */
    stack = ALIGN_DOWN(stack, 0x10);
    /* Make sure `stack` is now aligned to both `alignof(*sigframe)` and 0x10. */
    static_assert(alignof(*sigframe) % 0x10 == 0 || 0x10 % alignof(*sigframe) == 0,
                  "Incorrect sigframe alignment");

    sigframe = (struct sigframe*)stack;
    /* This could probably be omited as we set all fields explicitly below. */
    memset(sigframe, 0, sizeof(*sigframe));

    sigframe->siginfo = *siginfo;

    /* Gramine does not change SS (stack segment register) and assumes that it is constant so
     * these flags are not strictly needed, but we do store SS in ucontext so let's just set them. */
    sigframe->uc.uc_flags = UC_SIGCONTEXT_SS | UC_STRICT_RESTORE_SS;
    sigframe->uc.uc_link = NULL;
    /* TODO: add support for SA_AUTODISARM
     * Tracked in https://github.com/gramineproject/gramine/issues/84. */
    sigframe->uc.uc_stack = current->signal_altstack;

    pal_context_to_ucontext(&sigframe->uc, context);

    /* XXX: Currently we assume that `struct shim_xstate`, `PAL_XREGS_STATE` and `struct _fpstate`
     * (just the header) are the very same structure. This mess needs to be fixed. */
    static_assert(sizeof(struct shim_xstate) == sizeof(PAL_XREGS_STATE),
                  "SSE state structs differ");
    static_assert(sizeof(struct shim_fpstate) == sizeof(struct _fpstate),
                  "SSE state structs differ");
    if (shim_xstate_copy(xstate, (struct shim_xstate*)sigframe->uc.uc_mcontext.fpstate)) {
        /* This is a xsave-made xstate - it has the extended state info. */
        sigframe->uc.uc_flags |= UC_FP_XSTATE;
    }
    sigframe->uc.uc_mcontext.fpstate = (struct _fpstate*)xstate;

    sigframe->uc.uc_sigmask = *old_mask;

    /* We always set all 3 arguments, even if it is not `SA_SIGINFO` handler. We can and it is
     * easier this way. */
    context->rdi = (long)siginfo->si_signo;
    context->rsi = (uint64_t)&sigframe->siginfo;
    context->rdx = (uint64_t)&sigframe->uc;

    stack -= 8;
    *(uint64_t*)stack = (uint64_t)restorer;

    context->rip = (uint64_t)handler;
    context->rsp = stack;
    /* x64 SysV ABI mandates that DF flag is cleared and states that rest of flags is *not*
     * preserved across function calls, hence we just set flags to a default value (IF). */
    context->efl = 0x202;
    /* If handler was defined as variadic/without prototype it would expect the number of vector
     * register arguments in `rax`. */
    context->rax = 0;

    log_debug("Created sigframe for sig: %d at %p (handler: %p, restorer: %p)",
              siginfo->si_signo, sigframe, handler, restorer);
}

void restart_syscall(PAL_CONTEXT* context, uint64_t syscall_nr) {
    context->rax = syscall_nr;
    context->rip = (uint64_t)&syscalldb;
}

void restore_sigreturn_context(PAL_CONTEXT* context, __sigset_t* new_mask) {
    struct sigframe* sigframe = (struct sigframe*)context->rsp;
    *new_mask = sigframe->uc.uc_sigmask;

    struct shim_xstate* syscall_fpregs_buf = (struct shim_xstate*)context->fpregs;
    assert(syscall_fpregs_buf);

    ucontext_to_pal_context(context, &sigframe->uc);

    shim_xstate_copy(syscall_fpregs_buf, (struct shim_xstate*)sigframe->uc.uc_mcontext.fpstate);
    context->fpregs = (PAL_XREGS_STATE*)syscall_fpregs_buf;
    context->is_fpregs_used = 1;
}

bool maybe_emulate_syscall(PAL_CONTEXT* context) {
    uint8_t* rip = (uint8_t*)context->rip;
    if (rip[0] == 0x0f && rip[1] == 0x05) {
        /* This is syscall instruction, let's emulate it. */
        context->rcx = (uint64_t)rip + 2;
        context->rip = (uint64_t)&syscalldb;
        return true;
    }
    return false;
}
