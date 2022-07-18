/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University
 */
#include "sgx_enclave.h"

#include <asm/errno.h>
#include <asm/ioctls.h>
#include <asm/mman.h>
#include <asm/socket.h>
#include <limits.h>
#include <linux/futex.h>
#include <linux/in.h>
#include <linux/in6.h>
#include <linux/signal.h>

#include "cpu.h"
#include "debug_map.h"
#include "ecall_types.h"
#include "linux_utils.h"
#include "ocall_types.h"
#include "rpc_queue.h"
#include "sgx_internal.h"
#include "sgx_process.h"
#include "sgx_tls.h"
#include "sigset.h"

#define DEFAULT_BACKLOG 2048

#define ODEBUG(code, ms) \
    do {                 \
    } while (0)

extern bool g_vtune_profile_enabled;

static long sgx_ocall_exit(void* pms) {
    ms_ocall_exit_t* ms = (ms_ocall_exit_t*)pms;
    ODEBUG(OCALL_EXIT, NULL);

    if (ms->ms_exitcode != (int)((uint8_t)ms->ms_exitcode)) {
        log_debug("Saturation error in exit code %d, getting rounded down to %u",
                  ms->ms_exitcode, (uint8_t)ms->ms_exitcode);
        ms->ms_exitcode = 255;
    }

    /* exit the whole process if exit_group() */
    if (ms->ms_is_exitgroup) {
        update_and_print_stats(/*process_wide=*/true);
#ifdef DEBUG
        sgx_profile_finish();
#endif

#ifdef SGX_VTUNE_PROFILE
        if (g_vtune_profile_enabled) {
            extern void __itt_fini_ittlib(void);
            __itt_fini_ittlib();
        }
#endif
        DO_SYSCALL(exit_group, (int)ms->ms_exitcode);
        die_or_inf_loop();
    }

    /* otherwise call SGX-related thread reset and exit this thread */
    block_async_signals(true);
    ecall_thread_reset();

    unmap_tcs();

    if (!current_enclave_thread_cnt()) {
        /* no enclave threads left, kill the whole process */
        update_and_print_stats(/*process_wide=*/true);
#ifdef DEBUG
        sgx_profile_finish();
#endif
#ifdef SGX_VTUNE_PROFILE
        if (g_vtune_profile_enabled) {
            extern void __itt_fini_ittlib(void);
            __itt_fini_ittlib();
        }
#endif
        DO_SYSCALL(exit_group, (int)ms->ms_exitcode);
        die_or_inf_loop();
    }

    thread_exit((int)ms->ms_exitcode);
    return 0;
}

static long sgx_ocall_mmap_untrusted(void* pms) {
    ms_ocall_mmap_untrusted_t* ms = (ms_ocall_mmap_untrusted_t*)pms;
    void* addr;
    ODEBUG(OCALL_MMAP_UNTRUSTED, ms);

    addr = (void*)DO_SYSCALL(mmap, ms->ms_addr, ms->ms_size, ms->ms_prot, ms->ms_flags, ms->ms_fd,
                             ms->ms_offset);
    if (IS_PTR_ERR(addr))
        return PTR_TO_ERR(addr);

    ms->ms_addr = addr;
    return 0;
}

static long sgx_ocall_munmap_untrusted(void* pms) {
    ms_ocall_munmap_untrusted_t* ms = (ms_ocall_munmap_untrusted_t*)pms;
    ODEBUG(OCALL_MUNMAP_UNTRUSTED, ms);
    DO_SYSCALL(munmap, ms->ms_addr, ms->ms_size);
    return 0;
}

static long sgx_ocall_cpuid(void* pms) {
    ms_ocall_cpuid_t* ms = (ms_ocall_cpuid_t*)pms;
    ODEBUG(OCALL_CPUID, ms);
    __asm__ volatile("cpuid"
                     : "=a"(ms->ms_values[0]),
                       "=b"(ms->ms_values[1]),
                       "=c"(ms->ms_values[2]),
                       "=d"(ms->ms_values[3])
                     : "a"(ms->ms_leaf), "c"(ms->ms_subleaf)
                     : "memory");
    return 0;
}

static long sgx_ocall_open(void* pms) {
    ms_ocall_open_t* ms = (ms_ocall_open_t*)pms;
    long ret;
    ODEBUG(OCALL_OPEN, ms);
    // FIXME: No idea why someone hardcoded O_CLOEXEC here. We should drop it and carefully
    // investigate if this cause any descriptor leaks.
    ret = DO_SYSCALL_INTERRUPTIBLE(open, ms->ms_pathname, ms->ms_flags | O_CLOEXEC, ms->ms_mode);
    return ret;
}

static long sgx_ocall_close(void* pms) {
    ms_ocall_close_t* ms = (ms_ocall_close_t*)pms;
    ODEBUG(OCALL_CLOSE, ms);
    /* Callers cannot retry close on `-EINTR`, so we do not call `DO_SYSCALL_INTERRUPTIBLE`. */
    return DO_SYSCALL(close, ms->ms_fd);
}

static long sgx_ocall_read(void* pms) {
    ms_ocall_read_t* ms = (ms_ocall_read_t*)pms;
    long ret;
    ODEBUG(OCALL_READ, ms);
    ret = DO_SYSCALL_INTERRUPTIBLE(read, ms->ms_fd, ms->ms_buf, ms->ms_count);
    return ret;
}

static long sgx_ocall_write(void* pms) {
    ms_ocall_write_t* ms = (ms_ocall_write_t*)pms;
    long ret;
    ODEBUG(OCALL_WRITE, ms);
    ret = DO_SYSCALL_INTERRUPTIBLE(write, ms->ms_fd, ms->ms_buf, ms->ms_count);
    return ret;
}

static long sgx_ocall_pread(void* pms) {
    ms_ocall_pread_t* ms = (ms_ocall_pread_t*)pms;
    long ret;
    ODEBUG(OCALL_PREAD, ms);
    ret = DO_SYSCALL_INTERRUPTIBLE(pread64, ms->ms_fd, ms->ms_buf, ms->ms_count, ms->ms_offset);
    return ret;
}

static long sgx_ocall_pwrite(void* pms) {
    ms_ocall_pwrite_t* ms = (ms_ocall_pwrite_t*)pms;
    long ret;
    ODEBUG(OCALL_PWRITE, ms);
    ret = DO_SYSCALL_INTERRUPTIBLE(pwrite64, ms->ms_fd, ms->ms_buf, ms->ms_count, ms->ms_offset);
    return ret;
}

static long sgx_ocall_fstat(void* pms) {
    ms_ocall_fstat_t* ms = (ms_ocall_fstat_t*)pms;
    long ret;
    ODEBUG(OCALL_FSTAT, ms);
    ret = DO_SYSCALL_INTERRUPTIBLE(fstat, ms->ms_fd, &ms->ms_stat);
    return ret;
}

static long sgx_ocall_fionread(void* pms) {
    ms_ocall_fionread_t* ms = (ms_ocall_fionread_t*)pms;
    long ret;
    int val;
    ODEBUG(OCALL_FIONREAD, ms);
    ret = DO_SYSCALL_INTERRUPTIBLE(ioctl, ms->ms_fd, FIONREAD, &val);
    return ret < 0 ? ret : val;
}

static long sgx_ocall_fsetnonblock(void* pms) {
    ms_ocall_fsetnonblock_t* ms = (ms_ocall_fsetnonblock_t*)pms;
    long ret;
    int flags;
    ODEBUG(OCALL_FSETNONBLOCK, ms);

    ret = DO_SYSCALL(fcntl, ms->ms_fd, F_GETFL);
    if (ret < 0)
        return ret;

    flags = ret;
    if (ms->ms_nonblocking) {
        if (!(flags & O_NONBLOCK))
            ret = DO_SYSCALL(fcntl, ms->ms_fd, F_SETFL, flags | O_NONBLOCK);
    } else {
        if (flags & O_NONBLOCK)
            ret = DO_SYSCALL(fcntl, ms->ms_fd, F_SETFL, flags & ~O_NONBLOCK);
    }

    return ret;
}

static long sgx_ocall_fchmod(void* pms) {
    ms_ocall_fchmod_t* ms = (ms_ocall_fchmod_t*)pms;
    long ret;
    ODEBUG(OCALL_FCHMOD, ms);
    ret = DO_SYSCALL(fchmod, ms->ms_fd, ms->ms_mode);
    return ret;
}

static long sgx_ocall_fsync(void* pms) {
    ms_ocall_fsync_t* ms = (ms_ocall_fsync_t*)pms;
    ODEBUG(OCALL_FSYNC, ms);
    return DO_SYSCALL_INTERRUPTIBLE(fsync, ms->ms_fd);
}

static long sgx_ocall_ftruncate(void* pms) {
    ms_ocall_ftruncate_t* ms = (ms_ocall_ftruncate_t*)pms;
    long ret;
    ODEBUG(OCALL_FTRUNCATE, ms);
    ret = DO_SYSCALL(ftruncate, ms->ms_fd, ms->ms_length);
    return ret;
}

static long sgx_ocall_mkdir(void* pms) {
    ms_ocall_mkdir_t* ms = (ms_ocall_mkdir_t*)pms;
    long ret;
    ODEBUG(OCALL_MKDIR, ms);
    ret = DO_SYSCALL(mkdir, ms->ms_pathname, ms->ms_mode);
    return ret;
}

static long sgx_ocall_getdents(void* pms) {
    ms_ocall_getdents_t* ms = (ms_ocall_getdents_t*)pms;
    long ret;
    ODEBUG(OCALL_GETDENTS, ms);
    unsigned int count = ms->ms_size <= UINT_MAX ? ms->ms_size : UINT_MAX;
    ret = DO_SYSCALL_INTERRUPTIBLE(getdents64, ms->ms_fd, ms->ms_dirp, count);
    return ret;
}

static long sgx_ocall_resume_thread(void* pms) {
    ODEBUG(OCALL_RESUME_THREAD, pms);
    int tid = get_tid_from_tcs(pms);
    if (tid < 0)
        return tid;

    long ret = DO_SYSCALL(tgkill, g_host_pid, tid, SIGCONT);
    return ret;
}

static long sgx_ocall_sched_setaffinity(void* pms) {
    ms_ocall_sched_setaffinity_t* ms = (ms_ocall_sched_setaffinity_t*)pms;
    ODEBUG(OCALL_SCHED_SETAFFINITY, ms);
    int tid = get_tid_from_tcs(ms->ms_tcs);
    if (tid < 0)
        return tid;

    long ret = DO_SYSCALL(sched_setaffinity, tid, ms->ms_cpumask_size, ms->ms_cpu_mask);
    return ret;
}

static long sgx_ocall_sched_getaffinity(void* pms) {
    ms_ocall_sched_getaffinity_t* ms = (ms_ocall_sched_getaffinity_t*)pms;
    ODEBUG(OCALL_SCHED_GETAFFINITY, ms);
    int tid = get_tid_from_tcs(ms->ms_tcs);
    if (tid < 0)
        return tid;

    long ret = DO_SYSCALL(sched_getaffinity, tid, ms->ms_cpumask_size, ms->ms_cpu_mask);
    return ret;
}

static long sgx_ocall_clone_thread(void* pms) {
    __UNUSED(pms);
    ODEBUG(OCALL_CLONE_THREAD, pms);
    return clone_thread();
}

static long sgx_ocall_create_process(void* pms) {
    ms_ocall_create_process_t* ms = (ms_ocall_create_process_t*)pms;
    ODEBUG(OCALL_CREATE_PROCESS, ms);

    return sgx_create_process(ms->ms_nargs, ms->ms_args, g_pal_enclave.raw_manifest_data,
                              &ms->ms_stream_fd);
}

static long sgx_ocall_futex(void* pms) {
    ms_ocall_futex_t* ms = (ms_ocall_futex_t*)pms;
    long ret;
    ODEBUG(OCALL_FUTEX, ms);

    struct timespec timeout = { 0 };
    bool have_timeout = ms->ms_timeout_us != (uint64_t)-1;
    if (have_timeout) {
        time_get_now_plus_ns(&timeout, ms->ms_timeout_us * TIME_NS_IN_US);
    }

    /* `FUTEX_WAIT` treats timeout parameter as a relative value. We want to have an absolute one
     * (since we need to get start time anyway, to calculate remaining time later on), hence we use
     * `FUTEX_WAIT_BITSET` with `FUTEX_BITSET_MATCH_ANY`. */
    uint32_t val3 = 0;
    int priv_flag = ms->ms_op & FUTEX_PRIVATE_FLAG;
    int op = ms->ms_op & ~FUTEX_PRIVATE_FLAG;
    if (op == FUTEX_WAKE) {
        op = FUTEX_WAKE_BITSET;
        val3 = FUTEX_BITSET_MATCH_ANY;
    } else if (op == FUTEX_WAIT) {
        op = FUTEX_WAIT_BITSET;
        val3 = FUTEX_BITSET_MATCH_ANY;
    } else {
        /* Other operations are not supported atm. */
        return -EINVAL;
    }

    ret = DO_SYSCALL_INTERRUPTIBLE(futex, ms->ms_futex, op | priv_flag, ms->ms_val,
                                   have_timeout ? &timeout : NULL, NULL, val3);

    if (have_timeout) {
        int64_t diff = time_ns_diff_from_now(&timeout);
        if (diff < 0) {
            /* We might have slept a bit too long. */
            diff = 0;
        }
        ms->ms_timeout_us = (uint64_t)diff / TIME_NS_IN_US;
    }
    return ret;
}

static long sgx_ocall_listen(void* pms) {
    ms_ocall_listen_t* ms = (ms_ocall_listen_t*)pms;
    long ret;
    int fd;
    ODEBUG(OCALL_LISTEN, ms);

    if (ms->ms_addrlen > INT_MAX) {
        ret = -EINVAL;
        goto err;
    }

    ret = DO_SYSCALL(socket, ms->ms_domain, ms->ms_type | SOCK_CLOEXEC, ms->ms_protocol);
    if (ret < 0)
        goto err;

    fd = ret;

    /* must set the socket to be reuseable */
    int reuseaddr = 1;
    ret = DO_SYSCALL(setsockopt, fd, SOL_SOCKET, SO_REUSEADDR, &reuseaddr, sizeof(reuseaddr));
    if (ret < 0)
        goto err_fd;

    if (ms->ms_domain == AF_INET6) {
        /* IPV6_V6ONLY socket option can only be set before first bind */
        ret = DO_SYSCALL(setsockopt, fd, IPPROTO_IPV6, IPV6_V6ONLY, &ms->ms_ipv6_v6only,
                         sizeof(ms->ms_ipv6_v6only));
        if (ret < 0)
            goto err_fd;
    }

    ret = DO_SYSCALL_INTERRUPTIBLE(bind, fd, ms->ms_addr, (int)ms->ms_addrlen);
    if (ret < 0)
        goto err_fd;

    if (ms->ms_addr) {
        int addrlen = ms->ms_addrlen;
        ret = DO_SYSCALL(getsockname, fd, ms->ms_addr, &addrlen);
        if (ret < 0)
            goto err_fd;
        ms->ms_addrlen = addrlen;
    }

    if (ms->ms_type & SOCK_STREAM) {
        ret = DO_SYSCALL_INTERRUPTIBLE(listen, fd, DEFAULT_BACKLOG);
        if (ret < 0)
            goto err_fd;
    }

    return fd;

err_fd:
    DO_SYSCALL(close, fd);
err:
    return ret;
}

static long sgx_ocall_accept(void* pms) {
    ms_ocall_accept_t* ms = (ms_ocall_accept_t*)pms;
    long ret;
    ODEBUG(OCALL_ACCEPT, ms);

    if (ms->ms_addrlen > INT_MAX) {
        ret = -EINVAL;
        goto err;
    }
    int addrlen = ms->ms_addrlen;
    int options = ms->options | SOCK_CLOEXEC;
    assert(WITHIN_MASK(options, SOCK_CLOEXEC | SOCK_NONBLOCK));

    ret = DO_SYSCALL_INTERRUPTIBLE(accept4, ms->ms_sockfd, ms->ms_addr, &addrlen, options);
    if (ret < 0)
        goto err;

    int fd = ret;
    ms->ms_addrlen = addrlen;

    if (ms->ms_bind_addr && ms->ms_bind_addrlen) {
        int addrlen = ms->ms_bind_addrlen;
        ret = DO_SYSCALL(getsockname, fd, ms->ms_bind_addr, &addrlen);
        if (ret < 0)
            goto err_fd;
        ms->ms_bind_addrlen = addrlen;
    }

    return fd;

err_fd:
    DO_SYSCALL(close, fd);

err:
    return ret;
}

static long sgx_ocall_connect(void* pms) {
    ms_ocall_connect_t* ms = (ms_ocall_connect_t*)pms;
    long ret;
    int fd;
    ODEBUG(OCALL_CONNECT, ms);

    if (ms->ms_addrlen > INT_MAX || ms->ms_bind_addrlen > INT_MAX) {
        ret = -EINVAL;
        goto err;
    }

    ret = DO_SYSCALL(socket, ms->ms_domain, ms->ms_type | SOCK_CLOEXEC, ms->ms_protocol);
    if (ret < 0)
        goto err;

    fd = ret;

    if (ms->ms_bind_addr && ms->ms_bind_addr->sa_family) {
        if (ms->ms_domain == AF_INET6) {
            /* IPV6_V6ONLY socket option can only be set before first bind */
            ret = DO_SYSCALL(setsockopt, fd, IPPROTO_IPV6, IPV6_V6ONLY, &ms->ms_ipv6_v6only,
                             sizeof(ms->ms_ipv6_v6only));
            if (ret < 0)
                goto err_fd;
        }

        ret = DO_SYSCALL_INTERRUPTIBLE(bind, fd, ms->ms_bind_addr, ms->ms_bind_addrlen);
        if (ret < 0)
            goto err_fd;
    }

    if (ms->ms_addr) {
        ret = DO_SYSCALL_INTERRUPTIBLE(connect, fd, ms->ms_addr, ms->ms_addrlen);

        if (ret == -EINPROGRESS) {
            do {
                struct pollfd pfd = {
                    .fd      = fd,
                    .events  = POLLOUT,
                    .revents = 0,
                };
                ret = DO_SYSCALL_INTERRUPTIBLE(ppoll, &pfd, 1, NULL, NULL);
            } while (ret == -EWOULDBLOCK);
        }

        if (ret < 0)
            goto err_fd;
    }

    if (ms->ms_bind_addr && !ms->ms_bind_addr->sa_family) {
        int addrlen = ms->ms_bind_addrlen;
        ret = DO_SYSCALL(getsockname, fd, ms->ms_bind_addr, &addrlen);
        if (ret < 0)
            goto err_fd;
        ms->ms_bind_addrlen = addrlen;
    }

    return fd;

err_fd:
    DO_SYSCALL(close, fd);
err:
    return ret;
}

static long sgx_ocall_recv(void* pms) {
    ms_ocall_recv_t* ms = (ms_ocall_recv_t*)pms;
    long ret;
    ODEBUG(OCALL_RECV, ms);
    struct sockaddr* addr = ms->ms_addr;

    if (ms->ms_addr && ms->ms_addrlen > INT_MAX) {
        return -EINVAL;
    }
    int addrlen = ms->ms_addr ? ms->ms_addrlen : 0;

    struct msghdr hdr;
    struct iovec iov[1];

    iov[0].iov_base    = ms->ms_buf;
    iov[0].iov_len     = ms->ms_count;
    hdr.msg_name       = addr;
    hdr.msg_namelen    = addrlen;
    hdr.msg_iov        = iov;
    hdr.msg_iovlen     = 1;
    hdr.msg_control    = ms->ms_control;
    hdr.msg_controllen = ms->ms_controllen;
    hdr.msg_flags      = 0;

    ret = DO_SYSCALL_INTERRUPTIBLE(recvmsg, ms->ms_sockfd, &hdr, 0);

    if (ret >= 0 && hdr.msg_name) {
        /* note that ms->ms_addr is filled by recvmsg() itself */
        ms->ms_addrlen = hdr.msg_namelen;
    }

    if (ret >= 0 && hdr.msg_control) {
        /* note that ms->ms_control is filled by recvmsg() itself */
        ms->ms_controllen = hdr.msg_controllen;
    }

    return ret;
}

static long sgx_ocall_send(void* pms) {
    ms_ocall_send_t* ms = (ms_ocall_send_t*)pms;
    long ret;
    ODEBUG(OCALL_SEND, ms);
    const struct sockaddr* addr = ms->ms_addr;

    if (ms->ms_addr && ms->ms_addrlen > INT_MAX) {
        return -EINVAL;
    }
    int addrlen = ms->ms_addr ? ms->ms_addrlen : 0;

    struct msghdr hdr;
    struct iovec iov[1];

    iov[0].iov_base    = (void*)ms->ms_buf;
    iov[0].iov_len     = ms->ms_count;
    hdr.msg_name       = (void*)addr;
    hdr.msg_namelen    = addrlen;
    hdr.msg_iov        = iov;
    hdr.msg_iovlen     = 1;
    hdr.msg_control    = ms->ms_control;
    hdr.msg_controllen = ms->ms_controllen;
    hdr.msg_flags      = 0;

    ret = DO_SYSCALL_INTERRUPTIBLE(sendmsg, ms->ms_sockfd, &hdr, MSG_NOSIGNAL);
    return ret;
}

static long sgx_ocall_setsockopt(void* pms) {
    ms_ocall_setsockopt_t* ms = (ms_ocall_setsockopt_t*)pms;
    long ret;
    ODEBUG(OCALL_SETSOCKOPT, ms);
    if (ms->ms_optlen > INT_MAX) {
        return -EINVAL;
    }
    ret = DO_SYSCALL(setsockopt, ms->ms_sockfd, ms->ms_level, ms->ms_optname, ms->ms_optval,
                     (int)ms->ms_optlen);
    return ret;
}

static long sgx_ocall_shutdown(void* pms) {
    ms_ocall_shutdown_t* ms = (ms_ocall_shutdown_t*)pms;
    ODEBUG(OCALL_SHUTDOWN, ms);
    DO_SYSCALL_INTERRUPTIBLE(shutdown, ms->ms_sockfd, ms->ms_how);
    return 0;
}

static long sgx_ocall_gettime(void* pms) {
    ms_ocall_gettime_t* ms = (ms_ocall_gettime_t*)pms;
    ODEBUG(OCALL_GETTIME, ms);
    struct timeval tv;
    DO_SYSCALL(gettimeofday, &tv, NULL);
    ms->ms_microsec = tv.tv_sec * (uint64_t)1000000 + tv.tv_usec;
    return 0;
}

static long sgx_ocall_sched_yield(void* pms) {
    __UNUSED(pms);
    ODEBUG(OCALL_SCHED_YIELD, pms);
    DO_SYSCALL_INTERRUPTIBLE(sched_yield);
    return 0;
}

static long sgx_ocall_poll(void* pms) {
    ms_ocall_poll_t* ms = (ms_ocall_poll_t*)pms;
    long ret;
    ODEBUG(OCALL_POLL, ms);

    struct timespec* timeout = NULL;
    struct timespec end_time = { 0 };
    bool have_timeout = ms->ms_timeout_us != (uint64_t)-1;
    if (have_timeout) {
        uint64_t timeout_ns = ms->ms_timeout_us * TIME_NS_IN_US;
        timeout = __alloca(sizeof(*timeout));
        timeout->tv_sec = timeout_ns / TIME_NS_IN_S;
        timeout->tv_nsec = timeout_ns % TIME_NS_IN_S;
        time_get_now_plus_ns(&end_time, timeout_ns);
    }

    ret = DO_SYSCALL_INTERRUPTIBLE(ppoll, ms->ms_fds, ms->ms_nfds, timeout, NULL);

    if (have_timeout) {
        int64_t diff = time_ns_diff_from_now(&end_time);
        if (diff < 0) {
            /* We might have slept a bit too long. */
            diff = 0;
        }
        ms->ms_timeout_us = (uint64_t)diff / TIME_NS_IN_US;
    }

    return ret;
}

static long sgx_ocall_rename(void* pms) {
    ms_ocall_rename_t* ms = (ms_ocall_rename_t*)pms;
    long ret;
    ODEBUG(OCALL_RENAME, ms);
    ret = DO_SYSCALL(rename, ms->ms_oldpath, ms->ms_newpath);
    return ret;
}

static long sgx_ocall_delete(void* pms) {
    ms_ocall_delete_t* ms = (ms_ocall_delete_t*)pms;
    long ret;
    ODEBUG(OCALL_DELETE, ms);

    ret = DO_SYSCALL(unlink, ms->ms_pathname);

    if (ret == -EISDIR)
        ret = DO_SYSCALL(rmdir, ms->ms_pathname);

    return ret;
}

static long sgx_ocall_eventfd(void* pms) {
    ms_ocall_eventfd_t* ms = (ms_ocall_eventfd_t*)pms;
    long ret;
    ODEBUG(OCALL_EVENTFD, ms);

    ret = DO_SYSCALL(eventfd2, 0, ms->ms_flags);

    return ret;
}


static long sgx_ocall_ioctl(void* pms) {
    ms_ocall_ioctl_t* ms = (ms_ocall_ioctl_t*)pms;
    ODEBUG(OCALL_IOCTL, ms);
    long ret;
    if(ms->ms_fd)
        ret = DO_SYSCALL(ioctl, ms->ms_fd, ms->ms_cmd, ms->ms_arg);
    else {
        ret = DO_SYSCALL(socket, AF_UNIX, SOCK_STREAM, 0);
        if (ret < 0)
            goto err;
        long fd = ret;
        ret = DO_SYSCALL(ioctl, fd, ms->ms_cmd, ms->ms_arg);
        DO_SYSCALL(close, fd);
    }
err:
    return ret;
}

static long sgx_ocall_debug_map_add(void* pms) {
    ms_ocall_debug_map_add_t* ms = (ms_ocall_debug_map_add_t*)pms;

#ifdef DEBUG
    int ret = debug_map_add(ms->ms_name, ms->ms_addr);
    if (ret < 0)
        log_error("debug_map_add(%s, %p): %d", ms->ms_name, ms->ms_addr, ret);

    sgx_profile_report_elf(ms->ms_name, ms->ms_addr);
#else
    __UNUSED(ms);
#endif
    return 0;
}

static long sgx_ocall_debug_map_remove(void* pms) {
    ms_ocall_debug_map_remove_t* ms = (ms_ocall_debug_map_remove_t*)pms;

#ifdef DEBUG
    int ret = debug_map_remove(ms->ms_addr);
    if (ret < 0)
        log_error("debug_map_remove(%p): %d", ms->ms_addr, ret);
#else
    __UNUSED(ms);
#endif
    return 0;
}

static long sgx_ocall_debug_describe_location(void* pms) {
    ms_ocall_debug_describe_location_t* ms = (ms_ocall_debug_describe_location_t*)pms;

#ifdef DEBUG
    return debug_describe_location(ms->ms_addr, ms->ms_buf, ms->ms_buf_size);
#else
    __UNUSED(ms);
    return -ENOSYS;
#endif
}

static long sgx_ocall_get_quote(void* pms) {
    ms_ocall_get_quote_t* ms = (ms_ocall_get_quote_t*)pms;
    ODEBUG(OCALL_GET_QUOTE, ms);
    return retrieve_quote(ms->ms_is_epid ? &ms->ms_spid : NULL, ms->ms_linkable, &ms->ms_report,
                          &ms->ms_nonce, &ms->ms_quote, &ms->ms_quote_len);
}

sgx_ocall_fn_t ocall_table[OCALL_NR] = {
    [OCALL_EXIT]                     = sgx_ocall_exit,
    [OCALL_MMAP_UNTRUSTED]           = sgx_ocall_mmap_untrusted,
    [OCALL_MUNMAP_UNTRUSTED]         = sgx_ocall_munmap_untrusted,
    [OCALL_CPUID]                    = sgx_ocall_cpuid,
    [OCALL_OPEN]                     = sgx_ocall_open,
    [OCALL_CLOSE]                    = sgx_ocall_close,
    [OCALL_READ]                     = sgx_ocall_read,
    [OCALL_WRITE]                    = sgx_ocall_write,
    [OCALL_PREAD]                    = sgx_ocall_pread,
    [OCALL_PWRITE]                   = sgx_ocall_pwrite,
    [OCALL_FSTAT]                    = sgx_ocall_fstat,
    [OCALL_FIONREAD]                 = sgx_ocall_fionread,
    [OCALL_FSETNONBLOCK]             = sgx_ocall_fsetnonblock,
    [OCALL_FCHMOD]                   = sgx_ocall_fchmod,
    [OCALL_FSYNC]                    = sgx_ocall_fsync,
    [OCALL_FTRUNCATE]                = sgx_ocall_ftruncate,
    [OCALL_MKDIR]                    = sgx_ocall_mkdir,
    [OCALL_GETDENTS]                 = sgx_ocall_getdents,
    [OCALL_RESUME_THREAD]            = sgx_ocall_resume_thread,
    [OCALL_SCHED_SETAFFINITY]        = sgx_ocall_sched_setaffinity,
    [OCALL_SCHED_GETAFFINITY]        = sgx_ocall_sched_getaffinity,
    [OCALL_CLONE_THREAD]             = sgx_ocall_clone_thread,
    [OCALL_CREATE_PROCESS]           = sgx_ocall_create_process,
    [OCALL_FUTEX]                    = sgx_ocall_futex,
    [OCALL_LISTEN]                   = sgx_ocall_listen,
    [OCALL_ACCEPT]                   = sgx_ocall_accept,
    [OCALL_CONNECT]                  = sgx_ocall_connect,
    [OCALL_RECV]                     = sgx_ocall_recv,
    [OCALL_SEND]                     = sgx_ocall_send,
    [OCALL_SETSOCKOPT]               = sgx_ocall_setsockopt,
    [OCALL_SHUTDOWN]                 = sgx_ocall_shutdown,
    [OCALL_GETTIME]                  = sgx_ocall_gettime,
    [OCALL_SCHED_YIELD]              = sgx_ocall_sched_yield,
    [OCALL_POLL]                     = sgx_ocall_poll,
    [OCALL_RENAME]                   = sgx_ocall_rename,
    [OCALL_DELETE]                   = sgx_ocall_delete,
    [OCALL_DEBUG_MAP_ADD]            = sgx_ocall_debug_map_add,
    [OCALL_DEBUG_MAP_REMOVE]         = sgx_ocall_debug_map_remove,
    [OCALL_DEBUG_DESCRIBE_LOCATION]  = sgx_ocall_debug_describe_location,
    [OCALL_EVENTFD]                  = sgx_ocall_eventfd,
    [OCALL_IOCTL]                    = sgx_ocall_ioctl,
    [OCALL_GET_QUOTE]                = sgx_ocall_get_quote,
};

#define EDEBUG(code, ms) \
    do {                 \
    } while (0)

rpc_queue_t* g_rpc_queue = NULL; /* pointer to untrusted queue */

static int rpc_thread_loop(void* arg) {
    __UNUSED(arg);
    long mytid = DO_SYSCALL(gettid);

    /* block all signals except SIGUSR2 for RPC thread */
    __sigset_t mask;
    __sigfillset(&mask);
    __sigdelset(&mask, SIGUSR2);
    DO_SYSCALL(rt_sigprocmask, SIG_SETMASK, &mask, NULL, sizeof(mask));

    spinlock_lock(&g_rpc_queue->lock);
    g_rpc_queue->rpc_threads[g_rpc_queue->rpc_threads_cnt] = mytid;
    g_rpc_queue->rpc_threads_cnt++;
    spinlock_unlock(&g_rpc_queue->lock);

    static const uint64_t SPIN_ATTEMPTS_MAX = 10000;     /* rather arbitrary */
    static const uint64_t SLEEP_TIME_MAX    = 100000000; /* nanoseconds (0.1 seconds) */
    static const uint64_t SLEEP_TIME_STEP   = 1000000;   /* 100 steps before capped */

    /* no races possible since vars are thread-local and RPC threads don't receive signals */
    uint64_t spin_attempts = 0;
    uint64_t sleep_time    = 0;

    while (1) {
        rpc_request_t* req = rpc_dequeue(g_rpc_queue);
        if (!req) {
            if (spin_attempts == SPIN_ATTEMPTS_MAX) {
                if (sleep_time < SLEEP_TIME_MAX)
                    sleep_time += SLEEP_TIME_STEP;

                struct timespec tv = {.tv_sec = 0, .tv_nsec = sleep_time};
                (void)DO_SYSCALL(nanosleep, &tv, /*rem=*/NULL);
            } else {
                spin_attempts++;
                CPU_RELAX();
            }
            continue;
        }

        /* new request came, reset spin/sleep heuristics */
        spin_attempts = 0;
        sleep_time    = 0;

        /* call actual function and notify awaiting enclave thread when done */
        sgx_ocall_fn_t f = ocall_table[req->ocall_index];
        req->result = f(req->buffer);

        /* this code is based on Mutex 2 from Futexes are Tricky */
        int old_lock_state = __atomic_fetch_sub(&req->lock.lock, 1, __ATOMIC_ACQ_REL);
        if (old_lock_state == SPINLOCK_LOCKED_WITH_WAITERS) {
            /* must unlock and wake waiters */
            spinlock_unlock(&req->lock);
            int ret = DO_SYSCALL(futex, &req->lock.lock, FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);
            if (ret == -1)
                log_error("RPC thread failed to wake up enclave thread");
        }
    }

    /* NOTREACHED */
    return 0;
}

static int start_rpc(size_t threads_cnt) {
    g_rpc_queue = (rpc_queue_t*)DO_SYSCALL(mmap, NULL,
                                           ALIGN_UP(sizeof(rpc_queue_t), PRESET_PAGESIZE),
                                           PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE,
                                           -1, 0);
    if (IS_PTR_ERR(g_rpc_queue))
        return -ENOMEM;

    /* initialize g_rpc_queue just for sanity, it will be overwritten by in-enclave code */
    rpc_queue_init(g_rpc_queue);

    for (size_t i = 0; i < threads_cnt; i++) {
        void* stack = (void*)DO_SYSCALL(mmap, NULL, RPC_STACK_SIZE, PROT_READ | PROT_WRITE,
                                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (IS_PTR_ERR(stack))
            return -ENOMEM;

        void* child_stack_top = stack + RPC_STACK_SIZE;
        child_stack_top = ALIGN_DOWN_PTR(child_stack_top, 16);

        int dummy_parent_tid_field = 0;
        int ret = clone(rpc_thread_loop, child_stack_top,
                        CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SYSVSEM |
                        CLONE_THREAD | CLONE_SIGHAND | CLONE_PTRACE | CLONE_PARENT_SETTID,
                        /*arg=*/NULL, &dummy_parent_tid_field, /*tls=*/NULL, /*child_tid=*/NULL,
                        thread_exit);

        if (ret < 0) {
            DO_SYSCALL(munmap, stack, RPC_STACK_SIZE);
            return -ENOMEM;
        }
    }

    /* wait until all RPC threads are initialized in rpc_thread_loop */
    while (1) {
        spinlock_lock(&g_rpc_queue->lock);
        size_t n = g_rpc_queue->rpc_threads_cnt;
        spinlock_unlock(&g_rpc_queue->lock);
        if (n == g_pal_enclave.rpc_thread_num)
            break;
        DO_SYSCALL(sched_yield);
    }

    return 0;
}

int ecall_enclave_start(char* libpal_uri, char* args, size_t args_size, char* env,
                        size_t env_size, int parent_stream_fd, sgx_target_info_t* qe_targetinfo,
                        struct pal_topo_info* topo_info) {
    g_rpc_queue = NULL;

    if (g_pal_enclave.rpc_thread_num > 0) {
        int ret = start_rpc(g_pal_enclave.rpc_thread_num);
        if (ret < 0) {
            /* failed to create RPC threads */
            return ret;
        }
        /* after this point, g_rpc_queue != NULL */
    }

    ms_ecall_enclave_start_t ms;
    ms.ms_libpal_uri       = libpal_uri;
    ms.ms_libpal_uri_len   = strlen(ms.ms_libpal_uri);
    ms.ms_args             = args;
    ms.ms_args_size        = args_size;
    ms.ms_env              = env;
    ms.ms_env_size         = env_size;
    ms.ms_parent_stream_fd = parent_stream_fd;
    ms.ms_qe_targetinfo    = qe_targetinfo;
    ms.ms_topo_info        = topo_info;
    ms.rpc_queue           = g_rpc_queue;
    EDEBUG(ECALL_ENCLAVE_START, &ms);
    return sgx_ecall(ECALL_ENCLAVE_START, &ms);
}

int ecall_thread_start(void) {
    EDEBUG(ECALL_THREAD_START, NULL);
    return sgx_ecall(ECALL_THREAD_START, NULL);
}

int ecall_thread_reset(void) {
    EDEBUG(ECALL_THREAD_RESET, NULL);
    return sgx_ecall(ECALL_THREAD_RESET, NULL);
}
