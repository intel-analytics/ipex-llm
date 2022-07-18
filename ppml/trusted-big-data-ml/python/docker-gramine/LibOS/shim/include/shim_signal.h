#ifndef _SHIM_SIGNAL_H_
#define _SHIM_SIGNAL_H_

#include <stdbool.h>
#include <stdint.h>

#include "shim_defs.h"
#include "shim_types.h"

#define __WCOREDUMP_BIT 0x80

void sigaction_make_defaults(struct __kernel_sigaction* sig_action);
void thread_sigaction_reset_on_execve(void);

#define BITS_PER_WORD (8 * sizeof(unsigned long))
/* The standard def of this macro is dumb */
#undef _SIGSET_NWORDS
#define _SIGSET_NWORDS (SIGS_CNT / BITS_PER_WORD)

/* Return a mask that includes the bit for SIG only.  */
#define __sigmask(sig) \
    (((unsigned long int)1) << (((sig) - 1) % (8 * sizeof(unsigned long int))))

/* Return the word index for SIG.  */
#define __sigword(sig) (((sig) - 1) / (8 * sizeof(unsigned long int)))

/* Clear all signals from SET.  */
#define __sigemptyset(set)           \
    (__extension__({                 \
        int __cnt = _SIGSET_NWORDS;  \
        __sigset_t* __set = (set);   \
        while (--__cnt >= 0)         \
            __set->__val[__cnt] = 0; \
        0;                           \
    }))

/* Set all signals in SET.  */
#define __sigfillset(set)               \
    (__extension__({                    \
        int __cnt = _SIGSET_NWORDS;     \
        __sigset_t* __set = (set);      \
        while (--__cnt >= 0)            \
            __set->__val[__cnt] = ~0UL; \
        0;                              \
    }))

#define __sigisemptyset(set)                             \
    (__extension__({                                     \
        int __cnt = _SIGSET_NWORDS;                      \
        const __sigset_t* __set = (set);                 \
        unsigned long int __ret = __set->__val[--__cnt]; \
        while (!__ret && --__cnt >= 0)                   \
            __ret = __set->__val[__cnt];                 \
        __ret == 0;                                      \
    }))

#define __sigandset(dest, left, right)                                             \
    (__extension__({                                                               \
        int __cnt = _SIGSET_NWORDS;                                                \
        __sigset_t* __dest = (dest);                                               \
        const __sigset_t* __left  = (left);                                        \
        const __sigset_t* __right = (right);                                       \
        while (--__cnt >= 0)                                                       \
            __dest->__val[__cnt] = (__left->__val[__cnt] & __right->__val[__cnt]); \
        0;                                                                         \
    }))

#define __sigorset(dest, left, right)                                              \
    (__extension__({                                                               \
        int __cnt = _SIGSET_NWORDS;                                                \
        __sigset_t* __dest = (dest);                                               \
        const __sigset_t* __left  = (left);                                        \
        const __sigset_t* __right = (right);                                       \
        while (--__cnt >= 0)                                                       \
            __dest->__val[__cnt] = (__left->__val[__cnt] | __right->__val[__cnt]); \
        0;                                                                         \
    }))

#define __signotset(dest, left, right)                                              \
    (__extension__({                                                                \
        int __cnt = _SIGSET_NWORDS;                                                 \
        __sigset_t* __dest = (dest);                                                \
        const __sigset_t* __left  = (left);                                         \
        const __sigset_t* __right = (right);                                        \
        while (--__cnt >= 0)                                                        \
            __dest->__val[__cnt] = (__left->__val[__cnt] & ~__right->__val[__cnt]); \
        0;                                                                          \
    }))

#define __SIGSETFN(NAME, BODY, CONST)                            \
    static inline int NAME(CONST __sigset_t* __set, int __sig) { \
        unsigned long int __mask = __sigmask(__sig);             \
        unsigned long int __word = __sigword(__sig);             \
        return BODY;                                             \
    }

__SIGSETFN(shim_sigismember, (__set->__val[__word] & __mask) ? 1 : 0, __const)
__SIGSETFN(shim_sigaddset, ((__set->__val[__word] |= __mask), 0), )
__SIGSETFN(shim_sigdelset, ((__set->__val[__word] &= ~__mask), 0), )

#define __sigismember shim_sigismember
#define __sigaddset   shim_sigaddset
#define __sigdelset   shim_sigdelset

void clear_illegal_signals(__sigset_t* set);

struct shim_signal {
    siginfo_t siginfo;
};

void get_all_pending_signals(__sigset_t* set);
bool have_pending_signals(void);

/*!
 * \brief Return stack pointer to use in a signal handler.
 *
 * \param sp            Current stack pointer value.
 * \param use_altstack  True if alternative stack should be used.
 *
 * Returns value to be used as a new stack pointer in the signal handler, depending on the current
 * thread's settings.
 */
uintptr_t get_stack_for_sighandler(uintptr_t sp, bool use_altstack);
/*!
 * \brief Check whether address is on alternative stack.
 *
 * \param sp         Stack pointer value.
 * \param alt_stack  Pointer to the alternative stack.
 *
 * \returns `true` if \p sp is on \p alt_stack.
 */
bool is_on_altstack(uintptr_t sp, stack_t* alt_stack);

struct shim_thread;

int init_signal_handling(void);

int append_signal(struct shim_thread* thread, siginfo_t* info);

/*!
 * \brief Pop any of the pending signals allowed in \p mask.
 *
 * \param[in]  mask    Mask of blocked signals to compare against. If NULL, then the current thread
 *                     signal mask is used.
 * \param[out] signal  Pointer to signal, filled with signal info.
 *
 * Checks whether there is a pending unblocked signal among normal per-thread/per-process signals
 * and host-injected signals. Returns the first such signal in \p signal, or sets
 * `signal.siginfo.si_signo = 0` if no signal is found.
 */
void pop_unblocked_signal(__sigset_t* mask, struct shim_signal* signal);

void get_sig_mask(struct shim_thread* thread, __sigset_t* mask);
void set_sig_mask(struct shim_thread* thread, const __sigset_t* new_set);

/*!
 * \brief Set signal mask requested by user.
 *
 * \param mask_ptr  Pointer to the user signal mask (may be NULL).
 * \param setsize   Size of `__sigset_t`.
 *
 * \returns `0` on success, negative error code on failure.
 *
 * This function saves the current signal mask (in `struct shim_thread`) and sets it to \p mask_ptr.
 * If \p mask_ptr is NULL, this does nothing.
 */
int set_user_sigmask(const __sigset_t* mask_ptr, size_t setsize);

int kill_current_proc(siginfo_t* info);
int do_kill_thread(IDTYPE sender, IDTYPE tgid, IDTYPE tid, int sig);
int do_kill_proc(IDTYPE sender, IDTYPE pid, int sig);
int do_kill_pgroup(IDTYPE sender, IDTYPE pgid, int sig);

void fill_siginfo_code_and_status(siginfo_t* info, int signal, int exit_code);

int do_nanosleep(uint64_t timeout_us, struct __kernel_timespec* rem);

#endif /* _SHIM_SIGNAL_H_ */
