#include <asm/errno.h>

#include "ecall_types.h"
#include "generated-offsets-build.h"
#include "gsgx.h"
#include "ocall_types.h"
#include "pal.h"
#include "pal_linux_defs.h"
#include "sgx_arch.h"
#include "sgx_tls.h"

const char* generated_offsets_name = "PAL_SGX";

const struct generated_offset generated_offsets[] = {
    /* defines from sgx_arch.h */
    DEFINE(SGX_FLAGS_DEBUG, SGX_FLAGS_DEBUG),
    DEFINE(SGX_FLAGS_MODE64BIT, SGX_FLAGS_MODE64BIT),
    DEFINE(SGX_XFRM_LEGACY, SGX_XFRM_LEGACY),
    DEFINE(SGX_XFRM_AVX, SGX_XFRM_AVX),
    DEFINE(SGX_XFRM_MPX, SGX_XFRM_MPX),
    DEFINE(SGX_XFRM_AVX512, SGX_XFRM_AVX512),
    DEFINE(SGX_XFRM_PKRU, SGX_XFRM_PKRU),
    DEFINE(SGX_XFRM_AMX, SGX_XFRM_AMX),
    DEFINE(SGX_MISCSELECT_EXINFO, SGX_MISCSELECT_EXINFO),
    DEFINE(SE_KEY_SIZE, SE_KEY_SIZE),

    DEFINE(SGX_FLAGS_MASK_CONST, SGX_FLAGS_MASK_CONST),
    DEFINE(SGX_XFRM_MASK_CONST, SGX_XFRM_MASK_CONST),
    DEFINE(SGX_MISCSELECT_MASK_CONST, SGX_MISCSELECT_MASK_CONST),

    /* defines from pal-arch.h */
    DEFINE(STACK_PROTECTOR_CANARY_DEFAULT, STACK_PROTECTOR_CANARY_DEFAULT),

    /* sgx_measurement_t */
    DEFINE(SGX_HASH_SIZE, sizeof(sgx_measurement_t)),

    /* sgx_pal_gpr_t */
    OFFSET_T(SGX_GPR_RAX, sgx_pal_gpr_t, rax),
    OFFSET_T(SGX_GPR_RCX, sgx_pal_gpr_t, rcx),
    OFFSET_T(SGX_GPR_RDX, sgx_pal_gpr_t, rdx),
    OFFSET_T(SGX_GPR_RBX, sgx_pal_gpr_t, rbx),
    OFFSET_T(SGX_GPR_RSP, sgx_pal_gpr_t, rsp),
    OFFSET_T(SGX_GPR_RBP, sgx_pal_gpr_t, rbp),
    OFFSET_T(SGX_GPR_RSI, sgx_pal_gpr_t, rsi),
    OFFSET_T(SGX_GPR_RDI, sgx_pal_gpr_t, rdi),
    OFFSET_T(SGX_GPR_R8, sgx_pal_gpr_t, r8),
    OFFSET_T(SGX_GPR_R9, sgx_pal_gpr_t, r9),
    OFFSET_T(SGX_GPR_R10, sgx_pal_gpr_t, r10),
    OFFSET_T(SGX_GPR_R11, sgx_pal_gpr_t, r11),
    OFFSET_T(SGX_GPR_R12, sgx_pal_gpr_t, r12),
    OFFSET_T(SGX_GPR_R13, sgx_pal_gpr_t, r13),
    OFFSET_T(SGX_GPR_R14, sgx_pal_gpr_t, r14),
    OFFSET_T(SGX_GPR_R15, sgx_pal_gpr_t, r15),
    OFFSET_T(SGX_GPR_RFLAGS, sgx_pal_gpr_t, rflags),
    OFFSET_T(SGX_GPR_RIP, sgx_pal_gpr_t, rip),
    OFFSET_T(SGX_GPR_EXITINFO, sgx_pal_gpr_t, exitinfo),
    DEFINE(SGX_GPR_SIZE, sizeof(sgx_pal_gpr_t)),

    /* sgx_cpu_context_t */
    OFFSET_T(SGX_CPU_CONTEXT_RAX, sgx_cpu_context_t, rax),
    OFFSET_T(SGX_CPU_CONTEXT_RCX, sgx_cpu_context_t, rcx),
    OFFSET_T(SGX_CPU_CONTEXT_RDX, sgx_cpu_context_t, rdx),
    OFFSET_T(SGX_CPU_CONTEXT_RBX, sgx_cpu_context_t, rbx),
    OFFSET_T(SGX_CPU_CONTEXT_RSP, sgx_cpu_context_t, rsp),
    OFFSET_T(SGX_CPU_CONTEXT_RBP, sgx_cpu_context_t, rbp),
    OFFSET_T(SGX_CPU_CONTEXT_RSI, sgx_cpu_context_t, rsi),
    OFFSET_T(SGX_CPU_CONTEXT_RDI, sgx_cpu_context_t, rdi),
    OFFSET_T(SGX_CPU_CONTEXT_R8, sgx_cpu_context_t, r8),
    OFFSET_T(SGX_CPU_CONTEXT_R9, sgx_cpu_context_t, r9),
    OFFSET_T(SGX_CPU_CONTEXT_R10, sgx_cpu_context_t, r10),
    OFFSET_T(SGX_CPU_CONTEXT_R11, sgx_cpu_context_t, r11),
    OFFSET_T(SGX_CPU_CONTEXT_R12, sgx_cpu_context_t, r12),
    OFFSET_T(SGX_CPU_CONTEXT_R13, sgx_cpu_context_t, r13),
    OFFSET_T(SGX_CPU_CONTEXT_R14, sgx_cpu_context_t, r14),
    OFFSET_T(SGX_CPU_CONTEXT_R15, sgx_cpu_context_t, r15),
    OFFSET_T(SGX_CPU_CONTEXT_RFLAGS, sgx_cpu_context_t, rflags),
    OFFSET_T(SGX_CPU_CONTEXT_RIP, sgx_cpu_context_t, rip),
    DEFINE(SGX_CPU_CONTEXT_SIZE, sizeof(sgx_cpu_context_t)),
    DEFINE(SGX_CPU_CONTEXT_XSTATE_ALIGN_SUB, sizeof(sgx_cpu_context_t) % PAL_XSTATE_ALIGN),

    /* struct enclave_tls */
    OFFSET(SGX_COMMON_SELF, enclave_tls, common.self),
    OFFSET(SGX_COMMON_STACK_PROTECTOR_CANARY, enclave_tls, common.stack_protector_canary),
    OFFSET(SGX_ENCLAVE_SIZE, enclave_tls, enclave_size),
    OFFSET(SGX_TCS_OFFSET, enclave_tls, tcs_offset),
    OFFSET(SGX_INITIAL_STACK_ADDR, enclave_tls, initial_stack_addr),
    OFFSET(SGX_TMP_RIP, enclave_tls, tmp_rip),
    OFFSET(SGX_ECALL_RETURN_ADDR, enclave_tls, ecall_return_addr),
    OFFSET(SGX_SIG_STACK_LOW, enclave_tls, sig_stack_low),
    OFFSET(SGX_SIG_STACK_HIGH, enclave_tls, sig_stack_high),
    OFFSET(SGX_SSA, enclave_tls, ssa),
    OFFSET(SGX_GPR, enclave_tls, gpr),
    OFFSET(SGX_EXIT_TARGET, enclave_tls, exit_target),
    OFFSET(SGX_FSBASE, enclave_tls, fsbase),
    OFFSET(SGX_PRE_OCALL_STACK, enclave_tls, pre_ocall_stack),
    OFFSET(SGX_USTACK_TOP, enclave_tls, ustack_top),
    OFFSET(SGX_USTACK, enclave_tls, ustack),
    OFFSET(SGX_THREAD, enclave_tls, thread),
    OFFSET(SGX_OCALL_EXIT_CALLED, enclave_tls, ocall_exit_called),
    OFFSET(SGX_THREAD_STARTED, enclave_tls, thread_started),
    OFFSET(SGX_READY_FOR_EXCEPTIONS, enclave_tls, ready_for_exceptions),
    OFFSET(SGX_MANIFEST_SIZE, enclave_tls, manifest_size),
    OFFSET(SGX_HEAP_MIN, enclave_tls, heap_min),
    OFFSET(SGX_HEAP_MAX, enclave_tls, heap_max),
    OFFSET(SGX_CLEAR_CHILD_TID, enclave_tls, clear_child_tid),

    /* struct pal_tcb_urts aka PAL_TCB_URTS */
    OFFSET(PAL_TCB_URTS_TCS, pal_tcb_urts, tcs),
    OFFSET(PAL_TCB_URTS_IN_AEX_PROF, pal_tcb_urts, is_in_aex_profiling),
    OFFSET(PAL_TCB_URTS_EENTER_CNT, pal_tcb_urts, eenter_cnt),
    OFFSET(PAL_TCB_URTS_EEXIT_CNT, pal_tcb_urts, eexit_cnt),
    OFFSET(PAL_TCB_URTS_AEX_CNT, pal_tcb_urts, aex_cnt),
    OFFSET(PAL_TCB_URTS_LAST_ASYNC_EVENT, pal_tcb_urts, last_async_event),

    /* sgx_arch_tcs_t */
    OFFSET_T(TCS_FLAGS, sgx_arch_tcs_t, flags),
    OFFSET_T(TCS_OSSA, sgx_arch_tcs_t, ossa),
    OFFSET_T(TCS_CSSA, sgx_arch_tcs_t, cssa),
    OFFSET_T(TCS_NSSA, sgx_arch_tcs_t, nssa),
    OFFSET_T(TCS_OENTRY, sgx_arch_tcs_t, oentry),
    OFFSET_T(TCS_OFS_BASE, sgx_arch_tcs_t, ofs_base),
    OFFSET_T(TCS_OGS_BASE, sgx_arch_tcs_t, ogs_base),
    OFFSET_T(TCS_OFS_LIMIT, sgx_arch_tcs_t, ofs_limit),
    OFFSET_T(TCS_OGS_LIMIT, sgx_arch_tcs_t, ogs_limit),
    DEFINE(TCS_SIZE, sizeof(sgx_arch_tcs_t)),

    /* sgx_attributes_t */
    OFFSET_T(SGX_ATTRIBUTES_XFRM, sgx_attributes_t, xfrm),

    /* sgx_arch_enclave_css_t */
    OFFSET_T(SGX_ARCH_ENCLAVE_CSS_HEADER, sgx_arch_enclave_css_t, header.header),
    OFFSET_T(SGX_ARCH_ENCLAVE_CSS_TYPE, sgx_arch_enclave_css_t, header.type),
    OFFSET_T(SGX_ARCH_ENCLAVE_CSS_MODULE_VENDOR, sgx_arch_enclave_css_t, header.module_vendor),
    OFFSET_T(SGX_ARCH_ENCLAVE_CSS_DATE, sgx_arch_enclave_css_t, header.date),
    OFFSET_T(SGX_ARCH_ENCLAVE_CSS_HEADER2, sgx_arch_enclave_css_t, header.header2),
    OFFSET_T(SGX_ARCH_ENCLAVE_CSS_HW_VERSION, sgx_arch_enclave_css_t, header.hw_version),
    OFFSET_T(SGX_ARCH_ENCLAVE_CSS_MODULUS, sgx_arch_enclave_css_t, key.modulus),
    OFFSET_T(SGX_ARCH_ENCLAVE_CSS_EXPONENT, sgx_arch_enclave_css_t, key.exponent),
    OFFSET_T(SGX_ARCH_ENCLAVE_CSS_SIGNATURE, sgx_arch_enclave_css_t, key.signature),
    OFFSET_T(SGX_ARCH_ENCLAVE_CSS_MISC_SELECT, sgx_arch_enclave_css_t, body.misc_select),
    OFFSET_T(SGX_ARCH_ENCLAVE_CSS_MISC_MASK, sgx_arch_enclave_css_t, body.misc_mask),
    OFFSET_T(SGX_ARCH_ENCLAVE_CSS_ISV_FAMILY_ID, sgx_arch_enclave_css_t, body.isv_family_id),
    OFFSET_T(SGX_ARCH_ENCLAVE_CSS_ATTRIBUTES, sgx_arch_enclave_css_t, body.attributes),
    OFFSET_T(SGX_ARCH_ENCLAVE_CSS_ATTRIBUTE_MASK, sgx_arch_enclave_css_t, body.attribute_mask),
    OFFSET_T(SGX_ARCH_ENCLAVE_CSS_ENCLAVE_HASH, sgx_arch_enclave_css_t, body.enclave_hash),
    OFFSET_T(SGX_ARCH_ENCLAVE_CSS_ISVEXT_PROD_ID, sgx_arch_enclave_css_t, body.isvext_prod_id),
    OFFSET_T(SGX_ARCH_ENCLAVE_CSS_ISV_PROD_ID, sgx_arch_enclave_css_t, body.isv_prod_id),
    OFFSET_T(SGX_ARCH_ENCLAVE_CSS_ISV_SVN, sgx_arch_enclave_css_t, body.isv_svn),
    OFFSET_T(SGX_ARCH_ENCLAVE_CSS_Q1, sgx_arch_enclave_css_t, buffer.q1),
    OFFSET_T(SGX_ARCH_ENCLAVE_CSS_Q2, sgx_arch_enclave_css_t, buffer.q2),
    DEFINE(SGX_ARCH_ENCLAVE_CSS_SIZE, sizeof(sgx_arch_enclave_css_t)),

    /* pal_linux_def.h */
    DEFINE(SSA_FRAME_NUM, SSA_FRAME_NUM),
    DEFINE(SSA_FRAME_SIZE, SSA_FRAME_SIZE),
    DEFINE(ENCLAVE_STACK_SIZE, ENCLAVE_STACK_SIZE),
    DEFINE(ENCLAVE_SIG_STACK_SIZE, ENCLAVE_SIG_STACK_SIZE),
    DEFINE(DEFAULT_ENCLAVE_BASE, DEFAULT_ENCLAVE_BASE),
    DEFINE(MMAP_MIN_ADDR, MMAP_MIN_ADDR),

    /* pal_linux.h */
    DEFINE(PAGESIZE, PRESET_PAGESIZE),

    /* pal.h */
    DEFINE(PAL_EVENT_NO_EVENT, PAL_EVENT_NO_EVENT),
    DEFINE(PAL_EVENT_NUM_BOUND, PAL_EVENT_NUM_BOUND),

    /* errno */
    DEFINE(EINTR, EINTR),

    /* Ecall numbers */
    DEFINE(ECALL_ENCLAVE_START, ECALL_ENCLAVE_START),
    DEFINE(ECALL_THREAD_START, ECALL_THREAD_START),
    DEFINE(ECALL_THREAD_RESET, ECALL_THREAD_RESET),

    /* Ocall Index */
    DEFINE(OCALL_EXIT, OCALL_EXIT),

    /* fp regs */
    OFFSET_T(XSAVE_HEADER_OFFSET, PAL_XREGS_STATE, header),
    DEFINE(PAL_XSTATE_ALIGN, PAL_XSTATE_ALIGN),
    DEFINE(PAL_FP_XSTATE_MAGIC2_SIZE, PAL_FP_XSTATE_MAGIC2_SIZE),

    /* SGX_DCAP */
#ifdef SGX_DCAP
    DEFINE(SGX_DCAP, SGX_DCAP),
#endif

    OFFSET_END,
};
