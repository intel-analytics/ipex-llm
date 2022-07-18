#ifndef SYSCALL_H_
#define SYSCALL_H_

#include <asm/unistd.h>
#include <stdint.h>
#include <stdnoreturn.h>

long do_syscall(long nr, ...);
long clone(int (*f)(void*), void* stack, int flags, void* arg, void* parent_tid, void* tls,
           void* child_tid, void (*exit_func)(int));
long vfork(void) __attribute__((returns_twice));
void gramine_raw_syscalls_code_begin(void);
void gramine_raw_syscalls_code_end(void);

noreturn void _DkThreadExit_asm_stub(uint32_t* thread_stack_spinlock, int* clear_child_tid);

#define _extend(x) ((unsigned long)(x))

#define DO_SYSCALL_0(name) do_syscall(name)
#define DO_SYSCALL_1(name, a1) do_syscall(name, _extend(a1))
#define DO_SYSCALL_2(name, a1, a2) do_syscall(name, _extend(a1), _extend(a2))
#define DO_SYSCALL_3(name, a1, a2, a3) do_syscall(name, _extend(a1), _extend(a2), _extend(a3))
#define DO_SYSCALL_4(name, a1, a2, a3, a4) \
    do_syscall(name, _extend(a1), _extend(a2), _extend(a3), _extend(a4))
#define DO_SYSCALL_5(name, a1, a2, a3, a4, a5) \
    do_syscall(name, _extend(a1), _extend(a2), _extend(a3), _extend(a4), _extend(a5))
#define DO_SYSCALL_6(name, a1, a2, a3, a4, a5, a6) \
    do_syscall(name, _extend(a1), _extend(a2), _extend(a3), _extend(a4), _extend(a5), _extend(a6))

#define PICK_SYSCALL_MACRO(name, _1, _2, _3, _4, _5, _6, NARGS, ...) DO_SYSCALL_##NARGS

#define DO_SYSCALL(name, args...) \
    PICK_SYSCALL_MACRO(name, ##args, 6, 5, 4, 3, 2, 1, 0)(__NR_##name, ##args)

#define IS_PTR_ERR(val) ((unsigned long)(val) >= -4095UL)
#define PTR_TO_ERR(val) ((long)(val))
#define IS_UNIX_ERR(val) ((val) >= -133 /* EHWPOISON */ && (val) <= -1 /* EPERM */)

#endif // SYSCALL_H_
