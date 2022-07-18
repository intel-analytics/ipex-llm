#include "generated-offsets-build.h"
#include "shim_internal.h"
#include "shim_tcb.h"

const char* generated_offsets_name = "LIBOS";

const struct generated_offset generated_offsets[] = {
    OFFSET_T(SHIM_TCB_OFF, PAL_TCB, libos_tcb),
    OFFSET_T(SHIM_TCB_LIBOS_STACK_OFF, shim_tcb_t, libos_stack_bottom),
    OFFSET_T(SHIM_TCB_SCRATCH_PC_OFF, shim_tcb_t, syscall_scratch_pc),

    OFFSET_T(PAL_CONTEXT_FPREGS_OFF, struct PAL_CONTEXT, fpregs),
    OFFSET_T(PAL_CONTEXT_MXCSR_OFF, struct PAL_CONTEXT, mxcsr),
    OFFSET_T(PAL_CONTEXT_FPCW_OFF, struct PAL_CONTEXT, fpcw),
    OFFSET_T(PAL_CONTEXT_FPREGS_USED_OFF, struct PAL_CONTEXT, is_fpregs_used),

    DEFINE(SHIM_XSTATE_ALIGN, SHIM_XSTATE_ALIGN),
    DEFINE(RED_ZONE_SIZE, RED_ZONE_SIZE),

    OFFSET_END,
};
