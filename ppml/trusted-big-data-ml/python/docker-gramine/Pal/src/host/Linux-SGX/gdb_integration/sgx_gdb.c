#define _GNU_SOURCE
#include "sgx_gdb.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/syscall.h>
#include <sys/user.h>
#include <unistd.h>
#include <wait.h>

#include "assert.h"
#include "../sgx_arch.h"

/* Used by GDB with PTRACE_GETREGSET. */
#define NT_X86_XSTATE 0x202

//#define DEBUG_GDB_PTRACE 1

#if DEBUG_GDB_PTRACE == 1
#define DEBUG_LOG(fmt, ...)                  \
    do {                                     \
        fprintf(stderr, fmt, ##__VA_ARGS__); \
    } while (0)
#else
#define DEBUG_LOG(fmt, ...) \
    do {                    \
    } while (0)
#endif

static int g_memdevs_cnt = 0;
static struct {
    struct enclave_dbginfo ei;
    pid_t pid;
    int memdev;
} g_memdevs[32];

#if DEBUG_GDB_PTRACE == 1
static char* str_ptrace_request(enum __ptrace_request request) {
    static char buf[50];
    int prev_errno;

    switch (request) {
        case PTRACE_TRACEME:
            return "TRACEME";
        case PTRACE_PEEKTEXT:
            return "PEEKTEXT";
        case PTRACE_PEEKDATA:
            return "PEEKDATA";
        case PTRACE_POKETEXT:
            return "POKETEXT";
        case PTRACE_POKEDATA:
            return "POKEDATA";
        case PTRACE_PEEKUSER:
            return "PEEKUSER";
        case PTRACE_POKEUSER:
            return "POKEUSER";
        case PTRACE_GETREGS:
            return "GETREGS";
        case PTRACE_SETREGS:
            return "SETREGS";
        case PTRACE_GETREGSET:
            return "GETREGSET";
        case PTRACE_SETREGSET:
            return "SETREGSET";
        case PTRACE_SINGLESTEP:
            return "SINGLESTEP";
        case PTRACE_CONT:
            return "CONT";
        case PTRACE_KILL:
            return "KILL";
        case PTRACE_ATTACH:
            return "ATTACH";
        case PTRACE_DETACH:
            return "DETACH";
        default: /* fallthrough */;
    }

    prev_errno = errno; /* snprintf can contaminate errno */
    snprintf(buf, sizeof(buf), "0x%x", request);
    errno = prev_errno;
    return buf;
}
#endif

static void fill_regs(struct user_regs_struct* regs, const sgx_pal_gpr_t* gpr) {
    regs->orig_rax = gpr->rax;
    regs->rax      = gpr->rax;
    regs->rcx      = gpr->rcx;
    regs->rdx      = gpr->rdx;
    regs->rbx      = gpr->rbx;
    regs->rsp      = gpr->rsp;
    regs->rbp      = gpr->rbp;
    regs->rsi      = gpr->rsi;
    regs->rdi      = gpr->rdi;
    regs->r8       = gpr->r8;
    regs->r9       = gpr->r9;
    regs->r10      = gpr->r10;
    regs->r11      = gpr->r11;
    regs->r12      = gpr->r12;
    regs->r13      = gpr->r13;
    regs->r14      = gpr->r14;
    regs->r15      = gpr->r15;
    regs->rip      = gpr->rip;
    regs->eflags   = gpr->rflags;
    regs->fs_base  = gpr->fsbase;
    regs->gs_base  = gpr->gsbase;
    /* dummy values for non-SGX-saved regs */
    regs->cs = 0;
    regs->ss = 0;
    regs->ds = 0;
    regs->es = 0;
    regs->fs = 0;
    regs->gs = 0;
}

static void fill_gpr(sgx_pal_gpr_t* gpr, const struct user_regs_struct* regs) {
    gpr->rax    = regs->rax;
    gpr->rcx    = regs->rcx;
    gpr->rdx    = regs->rdx;
    gpr->rbx    = regs->rbx;
    gpr->rsp    = regs->rsp;
    gpr->rbp    = regs->rbp;
    gpr->rsi    = regs->rsi;
    gpr->rdi    = regs->rdi;
    gpr->r8     = regs->r8;
    gpr->r9     = regs->r9;
    gpr->r10    = regs->r10;
    gpr->r11    = regs->r11;
    gpr->r12    = regs->r12;
    gpr->r13    = regs->r13;
    gpr->r14    = regs->r14;
    gpr->r15    = regs->r15;
    gpr->rip    = regs->rip;
    gpr->rflags = regs->eflags;
    gpr->fsbase = regs->fs_base;
    gpr->gsbase = regs->gs_base;
}

/* This function emulates Glibc ptrace() by issuing ptrace syscall and
 * setting errno on error. It is used to access non-enclave memory. */
static long int host_ptrace(enum __ptrace_request request, pid_t tid, void* addr, void* data) {
    long int res, ret, is_dbginfo_addr;
    int ptrace_errno;

    /* If request is PTRACE_PEEKTEXT, PTRACE_PEEKDATA, or PTRACE_PEEKUSER
     * then ptrace syscall stores result at address specified by data;
     * our wrapper however conforms to Glibc and returns the result as
     * return value (with data being ignored). See ptrace(2) NOTES. */
    if (request == PTRACE_PEEKTEXT || request == PTRACE_PEEKDATA || request == PTRACE_PEEKUSER) {
        data = &res;
    }

    errno = 0;
    ret = syscall((long int)SYS_ptrace, (long int)request, (long int)tid, (long int)addr,
                  (long int)data);
    ptrace_errno = errno;

    /* check on dbginfo address to filter ei peeks for less noisy debug */
    is_dbginfo_addr = (addr >= (void*)DBGINFO_ADDR &&
                       addr < (void*)(DBGINFO_ADDR + sizeof(struct enclave_dbginfo)));

    if (!is_dbginfo_addr)
        DEBUG_LOG("[GDB %d] Executed host_ptrace(%s, %d, %p, %p) = %ld\n", getpid(),
              str_ptrace_request(request), tid, addr, data, ret);

    if (ret < 0 && ptrace_errno != 0) {
        errno = ptrace_errno; /* DEBUG/getpid could contaminate errno */
        return -1;
    }

    ret = 0;
    if (request == PTRACE_PEEKTEXT || request == PTRACE_PEEKDATA || request == PTRACE_PEEKUSER) {
        ret = res;
    }

    return ret;
}

/* Update GDB's copy of ei.thread_tids with current enclave's ei.thread_tids */
static int update_thread_tids(struct enclave_dbginfo* ei) {
    long int res;
    void* src = (void*)DBGINFO_ADDR + offsetof(struct enclave_dbginfo, thread_tids);
    void* dst = (void*)ei + offsetof(struct enclave_dbginfo, thread_tids);

    static_assert((sizeof(ei->thread_tids) % sizeof(long)) == 0,
                  "Unsupported ei->thread_tids size");

    for (size_t off = 0; off < sizeof(ei->thread_tids); off += sizeof(long)) {
        errno = 0;
        res = host_ptrace(PTRACE_PEEKDATA, ei->pid, src + off, NULL);
        if (res < 0 && errno != 0) {
            /* benign failure: enclave is not yet initialized */
            return -1;
        }
        *(long int*)(dst + off) = res;
    }

    return 0;
}

static void* get_ssa_addr(int memdev, pid_t tid, struct enclave_dbginfo* ei) {
    void* tcs_addr = NULL;
    struct {
        uint64_t ossa;
        uint32_t cssa, nssa;
    } tcs_part;
    ssize_t ret;

    for (int i = 0; i < MAX_DBG_THREADS; i++)
        if (ei->thread_tids[i] == tid) {
            tcs_addr = ei->tcs_addrs[i];
            break;
        }

    if (!tcs_addr) {
        DEBUG_LOG("Cannot find enclave thread %d to peek/poke its SSA\n", tid);
        return NULL;
    }

    ret = pread(memdev, &tcs_part, sizeof(tcs_part),
                (off_t)tcs_addr + offsetof(sgx_arch_tcs_t, ossa));
    if (ret < (ssize_t)sizeof(tcs_part)) {
        DEBUG_LOG("Cannot read TCS data (%p) of enclave thread %d\n", tcs_addr, tid);
        return NULL;
    }

    DEBUG_LOG("[enclave thread %d] TCS at 0x%lx: TCS.ossa = 0x%lx TCS.cssa = %d\n", tid, (long)tcs_addr,
          tcs_part.ossa, tcs_part.cssa);
    assert(tcs_part.cssa > 0);
    /* CSSA points to the next empty slot, so to read the current frame, we look at CSSA - 1. */
    return (void*)ei->base + tcs_part.ossa + ei->ssa_frame_size * (tcs_part.cssa - 1);
}

static void* get_gpr_addr(int memdev, pid_t tid, struct enclave_dbginfo* ei) {
    void* ssa = get_ssa_addr(memdev, tid, ei);
    if (!ssa)
        return NULL;

    return ssa + ei->ssa_frame_size - sizeof(sgx_pal_gpr_t);
}

static int peek_gpr(int memdev, pid_t tid, struct enclave_dbginfo* ei, sgx_pal_gpr_t* gpr) {
    ssize_t ret;

    void* gpr_addr = get_gpr_addr(memdev, tid, ei);
    if (!gpr_addr)
        return -1;

    ret = pread(memdev, gpr, sizeof(*gpr), (off_t)gpr_addr);
    if (ret < (ssize_t)sizeof(sgx_pal_gpr_t)) {
        DEBUG_LOG("Cannot read GPR data (%p) of enclave thread %d\n", gpr_addr, tid);
        return -1;
    }

    DEBUG_LOG("[enclave thread %d] Peek GPR: RIP 0x%08lx RBP 0x%08lx\n", tid, gpr->rip, gpr->rbp);
    return 0;
}

static int poke_gpr(int memdev, pid_t tid, struct enclave_dbginfo* ei, const sgx_pal_gpr_t* gpr) {
    ssize_t ret;

    void* gpr_addr = get_gpr_addr(memdev, tid, ei);
    if (!gpr_addr)
        return -1;

    ret = pwrite(memdev, gpr, sizeof(sgx_pal_gpr_t), (off_t)gpr_addr);
    if (ret < (ssize_t)sizeof(sgx_pal_gpr_t)) {
        DEBUG_LOG("Cannot write GPR data (%p) of enclave thread %d\n", (void*)gpr_addr, tid);
        return -1;
    }

    DEBUG_LOG("[enclave thread %d] Poke GPR: RIP 0x%08lx RBP 0x%08lx\n", tid, gpr->rip, gpr->rbp);
    return 0;
}

static int peek_xsave(int memdev, pid_t tid, struct enclave_dbginfo* ei, struct iovec* iov) {
    ssize_t ret;

    void* ssa_addr = get_ssa_addr(memdev, tid, ei);
    if (!ssa_addr)
        return -1;

    /* The SSA.XSAVE field has an offset of 0, so we use ssa_addr directly. */
    ret = pread(memdev, iov->iov_base, iov->iov_len, (off_t)ssa_addr);
    if (ret < (ssize_t)iov->iov_len) {
        DEBUG_LOG("Cannot read XSAVE data (%p) of enclave thread %d\n", ssa_addr, tid);
        return -1;
    }

    DEBUG_LOG("[enclave thread %d] Peek XSAVE: read %zu bytes\n", tid, iov->iov_len);
    return 0;
}

static int poke_xsave(int memdev, pid_t tid, struct enclave_dbginfo* ei, struct iovec* iov) {
    ssize_t ret;

    void* ssa_addr = get_ssa_addr(memdev, tid, ei);
    if (!ssa_addr)
        return -1;

    /* The SSA.XSAVE field has an offset of 0, so we use ssa_addr directly. */
    ret = pwrite(memdev, iov->iov_base, iov->iov_len, (off_t)ssa_addr);
    if (ret < (ssize_t)iov->iov_len) {
        DEBUG_LOG("Cannot write XSAVE data (%p) of enclave thread %d\n", (void*)ssa_addr, tid);
        return -1;
    }

    DEBUG_LOG("[enclave thread %d] Poke XSAVE: wrote %zu bytes\n", tid, iov->iov_len);
    return 0;
}

static int peek_user(int memdev, pid_t tid, struct enclave_dbginfo* ei, struct user* ud) {
    int ret;
    sgx_pal_gpr_t gpr;

    ret = peek_gpr(memdev, tid, ei, &gpr);
    if (ret < 0)
        return ret;

    memset(ud, 0, sizeof(*ud));
    fill_regs(&ud->regs, &gpr);
    return 0;
}

static int poke_user(int memdev, pid_t tid, struct enclave_dbginfo* ei, const struct user* ud) {
    int ret;
    sgx_pal_gpr_t gpr;

    ret = peek_gpr(memdev, tid, ei, &gpr);
    if (ret < 0)
        return ret;

    fill_gpr(&gpr, &ud->regs);
    return poke_gpr(memdev, tid, ei, &gpr);
}

static int peek_regs(int memdev, pid_t tid, struct enclave_dbginfo* ei,
                     struct user_regs_struct* regdata) {
    int ret;
    sgx_pal_gpr_t gpr;

    ret = peek_gpr(memdev, tid, ei, &gpr);
    if (ret < 0)
        return ret;

    fill_regs(regdata, &gpr);
    return 0;
}

static int poke_regs(int memdev, pid_t tid, struct enclave_dbginfo* ei,
                     const struct user_regs_struct* regdata) {
    int ret;
    sgx_pal_gpr_t gpr;

    ret = peek_gpr(memdev, tid, ei, &gpr);
    if (ret < 0)
        return ret;

    fill_gpr(&gpr, regdata);
    return poke_gpr(memdev, tid, ei, &gpr);
}

/* Find corresponding memdevice of thread tid (open and populate on first
 * access). Return 0 on success, -1 on benign failure (enclave in not yet
 * initialized), -2 on other, severe failures.
 *  */
static int open_memdevice(pid_t tid, int* memdev, struct enclave_dbginfo** ei) {
    struct enclave_dbginfo eib = {.pid = -1};
    char memdev_path[40];
    uint64_t flags;
    int ret;
    int fd;

    /* Check if corresponding memdevice of this thread was already opened;
     * this works when tid = pid (single-threaded apps) but does not work
     * for other threads of multi-threaded apps, this case covered below */
    for (int i = 0; i < g_memdevs_cnt; i++) {
        if (g_memdevs[i].pid == tid) {
            *memdev = g_memdevs[i].memdev;
            *ei = &g_memdevs[i].ei;
            return update_thread_tids(*ei);
        }
    }

    static_assert(sizeof(eib) % sizeof(long) == 0, "Unsupported eib size");

    for (size_t off = 0; off < sizeof(eib); off += sizeof(long)) {
        errno = 0;
        long val = host_ptrace(PTRACE_PEEKDATA, tid, (void*)DBGINFO_ADDR + off, NULL);
        if (val < 0 && errno != 0) {
            /* benign failure: enclave is not yet initialized */
            return -1;
        }

        memcpy((void*)&eib + off, &val, sizeof(val));
    }

    /* Check again if corresponding memdevice was already opened but now
     * using actual pid of app (eib.pid), case for multi-threaded apps */
    for (int i = 0; i < g_memdevs_cnt; i++) {
        if (g_memdevs[i].pid == eib.pid) {
            *memdev = g_memdevs[i].memdev;
            *ei = &g_memdevs[i].ei;
            return update_thread_tids(*ei);
        }
    }

    DEBUG_LOG("Retrieved debug information (enclave reports PID %d)\n", eib.pid);

    if (g_memdevs_cnt == sizeof(g_memdevs) / sizeof(g_memdevs[0])) {
        DEBUG_LOG("Too many debugged processes (max = %d)\n", g_memdevs_cnt);
        return -2;
    }

    snprintf(memdev_path, sizeof(memdev_path), "/proc/%d/mem", tid);
    fd = open(memdev_path, O_RDWR);
    if (fd < 0) {
        DEBUG_LOG("Cannot open %s\n", memdev_path);
        return -2;
    }

    /* setting debug bit in TCS flags */
    for (int i = 0; i < MAX_DBG_THREADS; i++) {
        if (!eib.tcs_addrs[i])
            continue;

        void* flags_addr = eib.tcs_addrs[i] + offsetof(sgx_arch_tcs_t, flags);

        ssize_t bytes_read = pread(fd, &flags, sizeof(flags), (off_t)flags_addr);
        if (bytes_read < 0 || (size_t)bytes_read < sizeof(flags)) {
            DEBUG_LOG("Cannot read TCS flags (address = %p)\n", flags_addr);
            ret = -2;
            goto out_err;
        }

        if (flags & TCS_FLAGS_DBGOPTIN)
            continue;

        flags |= TCS_FLAGS_DBGOPTIN;
        DEBUG_LOG("Setting TCS debug flag at %p (%lx)\n", flags_addr, flags);

        ssize_t bytes_written = pwrite(fd, &flags, sizeof(flags), (off_t)flags_addr);
        if (bytes_written < 0 || (size_t)bytes_written < sizeof(flags)) {
            DEBUG_LOG("Cannot write TCS flags (address = %p)\n", flags_addr);
            ret = -2;
            goto out_err;
        }
    }

    g_memdevs[g_memdevs_cnt].pid    = eib.pid;
    g_memdevs[g_memdevs_cnt].memdev = fd;
    g_memdevs[g_memdevs_cnt].ei     = eib;
    memset(g_memdevs[g_memdevs_cnt].ei.thread_stepping, 0,
           sizeof(g_memdevs[g_memdevs_cnt].ei.thread_stepping));

    *memdev = fd;
    *ei = &g_memdevs[g_memdevs_cnt].ei;
    g_memdevs_cnt++;

    return 0;
out_err:
    close(fd);
    return ret;
}

static int is_in_enclave(pid_t tid, struct enclave_dbginfo* ei) {
    struct user_regs_struct regs;

    int ret = host_ptrace(PTRACE_GETREGS, tid, NULL, &regs);
    if (ret < 0) {
        DEBUG_LOG("Failed getting registers: TID %d\n", tid);
        return -1;
    }

    DEBUG_LOG("%d: User RIP 0x%08llx%s\n", tid, regs.rip,
          ((void*)regs.rip == ei->aep) ? " (in enclave)" : "");

    return ((void*)regs.rip == ei->aep) ? 1 : 0;
}

long int ptrace(enum __ptrace_request request, ...) {
    long int ret = 0, res;
    va_list ap;
    pid_t tid;
    void* addr;
    void* data;
    int memdev;
    bool in_enclave;
    struct enclave_dbginfo* ei;
    struct user userdata;

    va_start(ap, request);
    tid  = va_arg(ap, pid_t);
    addr = va_arg(ap, void*);
    data = va_arg(ap, void*);
    va_end(ap);

    DEBUG_LOG("[GDB %d] Intercepted ptrace(%s, %d, %p, %p)\n", getpid(), str_ptrace_request(request),
          tid, addr, data);

    ret = open_memdevice(tid, &memdev, &ei);
    if (ret < 0) {
        if (ret == -1) {
            /* benign failure: enclave is not yet initialized */
            return host_ptrace(request, tid, addr, data);
        }
        errno = EFAULT;
        return -1;
    }

    ret = is_in_enclave(tid, ei);
    if (ret < 0) {
        errno = EFAULT;
        return -1;
    }

    in_enclave = (ret != 0);

    switch (request) {
        case PTRACE_PEEKTEXT:
        case PTRACE_PEEKDATA: {
            if ((addr + sizeof(long)) <= (void*)ei->base || addr >= (void*)(ei->base + ei->size)) {
                /* peek into strictly non-enclave memory */
                return host_ptrace(PTRACE_PEEKDATA, tid, addr, NULL);
            }

            ret = pread(memdev, &res, sizeof(long), (unsigned long)addr);
            if (ret < 0) {
                /* In some cases, GDB wants to read td_thrinfo_t object from
                 * in-LibOS Glibc. If host OS's Glibc and in-LibOS Glibc
                 * versions do not match, GDB's supplied addr is incorrect
                 * and leads to EIO failure of pread(). Circumvent this
                 * issue by returning a dummy 0. NOTE: this doesn't lead to
                 * noticeable debugging issues, at least on Ubuntu 16.04. */
                if (errno == EIO) {
                    errno = 0;
                    return 0;
                }
                errno = EFAULT;
                return -1;
            }
            return res;
        }

        case PTRACE_POKETEXT:
        case PTRACE_POKEDATA: {
            if ((addr + sizeof(long)) <= (void*)ei->base || addr >= (void*)(ei->base + ei->size)) {
                /* poke strictly non-enclave memory */
                return host_ptrace(PTRACE_POKEDATA, tid, addr, data);
            }

            ret = pwrite(memdev, &data, sizeof(long), (off_t)addr);
            if (ret < 0) {
                errno = EFAULT;
                return -1;
            }
            return 0;
        }

        case PTRACE_PEEKUSER: {
            if ((size_t)addr + sizeof(long) > sizeof(struct user)) {
                errno = EINVAL;
                return -1;
            }

            if (!in_enclave)
                return host_ptrace(PTRACE_PEEKUSER, tid, addr, data);

            if ((size_t)addr + sizeof(long) > sizeof(struct user_regs_struct))
                return host_ptrace(PTRACE_PEEKUSER, tid, addr, NULL);

            ret = peek_user(memdev, tid, ei, &userdata);
            if (ret < 0) {
                errno = EFAULT;
                return -1;
            }

            return *(long*)((void*)&userdata + (off_t)addr);
        }

        case PTRACE_POKEUSER: {
            if ((size_t)addr + sizeof(long) > sizeof(struct user)) {
                errno = EINVAL;
                return -1;
            }

            if (!in_enclave || (size_t)addr + sizeof(long) > sizeof(struct user_regs_struct))
                return host_ptrace(PTRACE_POKEUSER, tid, addr, data);

            ret = peek_user(memdev, tid, ei, &userdata);
            if (ret < 0) {
                errno = EFAULT;
                return -1;
            }

            *(long*)((void*)&userdata + (off_t)addr) = (long)data;

            ret = poke_user(memdev, tid, ei, &userdata);
            if (ret < 0) {
                errno = EFAULT;
                return -1;
            }

            return 0;
        }

        case PTRACE_GETREGS: {
            if (!in_enclave)
                return host_ptrace(PTRACE_GETREGS, tid, addr, data);

            ret = peek_regs(memdev, tid, ei, (struct user_regs_struct*)data);
            if (ret < 0) {
                errno = EFAULT;
                return -1;
            }

            return 0;
        }

        case PTRACE_SETREGS: {
            if (!in_enclave)
                return host_ptrace(PTRACE_SETREGS, tid, addr, data);

            ret = poke_regs(memdev, tid, ei, (struct user_regs_struct*)data);
            if (ret < 0) {
                errno = EFAULT;
                return -1;
            }

            return 0;
        }

        case PTRACE_GETREGSET: {
            if (!in_enclave)
                return host_ptrace(PTRACE_GETREGSET, tid, addr, data);

            if ((size_t)addr != NT_X86_XSTATE)
                return host_ptrace(PTRACE_GETREGSET, tid, addr, data);

            ret = peek_xsave(memdev, tid, ei, (struct iovec*)data);
            if (ret < 0) {
                errno = EFAULT;
                return -1;
            }

            return 0;
        }

        case PTRACE_SETREGSET: {
            if (!in_enclave)
                return host_ptrace(PTRACE_SETREGSET, tid, addr, data);

            if ((size_t)addr != NT_X86_XSTATE)
                return host_ptrace(PTRACE_SETREGSET, tid, addr, data);

            ret = poke_xsave(memdev, tid, ei, (struct iovec*)data);
            if (ret < 0) {
                errno = EFAULT;
                return -1;
            }

            return 0;
        }

        case PTRACE_SINGLESTEP: {
            if (!in_enclave)
                return host_ptrace(PTRACE_SINGLESTEP, tid, addr, data);

            struct user_regs_struct regs;
            ret = host_ptrace(PTRACE_GETREGS, tid, NULL, &regs);
            if (ret < 0) {
                errno = EFAULT;
                return -1;
            }
            regs.rip = (unsigned long long)ei->eresume;
            ret = host_ptrace(PTRACE_SETREGS, tid, NULL, &regs);
            if (ret < 0) {
                DEBUG_LOG("Cannot set RIP to ERESUME to continue single-step in enclave thread %d\n",
                      tid);
                errno = EFAULT;
                return -1;
            }

            for (int i = 0; i < MAX_DBG_THREADS; i++) {
                if (ei->thread_tids[i] == tid) {
                    ei->thread_stepping[i / 64] |= 1ULL << (i % 64);
                    return host_ptrace(PTRACE_SINGLESTEP, tid, addr, data);
                }
            }

            DEBUG_LOG("Cannot find enclave thread %d to single-step\n", tid);
            errno = EFAULT;
            return -1;
        }

        default:
            return host_ptrace(request, tid, addr, data);
    }

    /* should not reach here */
    return 0;
}

pid_t waitpid(pid_t tid, int* status, int options) {
    ssize_t ret;
    int memdev;
    pid_t wait_res;
    struct enclave_dbginfo* ei;
    sgx_pal_gpr_t gpr;
    uint8_t code;
    int wait_errno;

    DEBUG_LOG("[GDB %d] Intercepted waitpid(%d)\n", getpid(), tid);

    errno = 0;
    wait_res = syscall((long int)SYS_wait4, (long int)tid, (long int)status, (long int)options,
                       (long int)NULL);
    wait_errno = errno;

    DEBUG_LOG("[GDB %d] Executed host waitpid(%d, <status>, %d) = %d\n", getpid(), tid, options,
          wait_res);

    if (wait_res < 0) {
        errno = wait_errno;
        return -1;
    }

    if (!status) {
        return wait_res;
    }

    if (WIFSTOPPED(*status) && WSTOPSIG(*status) == SIGTRAP) {
        tid = wait_res;

        ret = open_memdevice(tid, &memdev, &ei);
        if (ret < 0) {
            if (ret == -1) {
                errno = 0; /* benign failure: enclave is not yet initialized */
                return wait_res;
            }
            errno = ECHILD;
            return -1;
        }

        /* for singlestepping case, unset enclave thread's stepping bit */
        for (int i = 0; i < MAX_DBG_THREADS; i++) {
            if (ei->thread_tids[i] == tid) {
                if (ei->thread_stepping[i / 64] & (1ULL << (i % 64))) {
                    ei->thread_stepping[i / 64] &= ~(1ULL << (i % 64));
                    return wait_res;
                }
                break;
            }
        }

        ret = is_in_enclave(tid, ei);
        if (ret < 0) {
            errno = ECHILD;
            return -1;
        }

        if (!ret)
            return wait_res;

        /* target thread is inside the enclave */
        ret = peek_gpr(memdev, tid, ei, &gpr);
        if (ret < 0) {
            errno = ECHILD;
            return -1;
        }

        ret = pread(memdev, &code, sizeof(code), (off_t)gpr.rip);
        if (ret < (ssize_t)sizeof(code)) {
            DEBUG_LOG("Cannot read RIP of enclave thread %d\n", tid);
            errno = ECHILD;
            return -1;
        }

        if (code != 0xcc)
            return wait_res;

        DEBUG_LOG("RIP 0x%lx points to a breakpoint\n", gpr.rip);

        /* This is a quirk of SGX hardware implementation. GDB expects that
         * RIP points to one byte *after* INT3 (which GDB put in place of
         * original instruction to induce breakpoint trap). But under SGX,
         * breakpoint is trapped such that RIP points *to* INT3. Thus, we
         * need to adjust RIP according to GDB's expectation.*/
        gpr.rip++;
        ret = poke_gpr(memdev, tid, ei, &gpr);
        if (ret < 0) {
            errno = ECHILD;
            return -1;
        }
    }

    return wait_res;
}
