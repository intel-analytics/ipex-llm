/* Our PAL loader invokes this entrypoint with all registers unspecified, except %rsp.
 * For detailed description of the stack layout see Pal/src/db_rtld.c:start_execution().
 */

    .text
    .globl _start
    .type _start,@function
_start:
    .cfi_startproc

    # Clear the frame pointer. The ABI suggests this be done, to mark
    # the outermost frame obviously.
    xorq %rbp, %rbp

    # Arguments for PAL-executable main:
    movq 0(%rsp), %rdi         # argc
    leaq 8(%rsp), %rsi         # argv
    leaq 8(%rsi,%rdi,8), %rdx  # envp, after all args (including argv[argc] = NULL)

    # Required by System V AMD64 ABI.
    andq  $~0xF, %rsp

    callq *main@GOTPCREL(%rip)

    mov %rax, %rdi
    callq *DkProcessExit@GOTPCREL(%rip)

    .cfi_endproc
