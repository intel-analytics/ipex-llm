#ifndef SGX_SYSCALL_H_
#define SGX_SYSCALL_H_

#include "syscall.h"

long do_syscall_intr(long nr, ...);
void do_syscall_intr_after_check1(void);
void do_syscall_intr_after_check2(void);
void do_syscall_intr_eintr(void);

#define DO_SYSCALL_INTERRUPTIBLE(name, args...) do_syscall_intr(__NR_##name, ##args)

#endif // SGX_SYSCALL_H_
