/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University
 * Copyright (C) 2020 Intel Corporation
 *                    Michał Kowalczyk <mkow@invisiblethingslab.com>
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

#ifndef _SHIM_TABLE_H_
#define _SHIM_TABLE_H_

#if defined(__i386__) || defined(__x86_64__)
#include <asm/ldt.h>
#endif

#include "shim_types.h"

typedef void (*shim_fp)(void);

extern shim_fp shim_table[];

/* syscall implementation */
long shim_do_read(int fd, void* buf, size_t count);
long shim_do_write(int fd, const void* buf, size_t count);
long shim_do_open(const char* file, int flags, mode_t mode);
long shim_do_close(int fd);
long shim_do_stat(const char* file, struct stat* statbuf);
long shim_do_fstat(int fd, struct stat* statbuf);
long shim_do_lstat(const char* file, struct stat* stat);
long shim_do_statfs(const char* path, struct statfs* buf);
long shim_do_fstatfs(int fd, struct statfs* buf);
long shim_do_poll(struct pollfd* fds, unsigned int nfds, int timeout);
long shim_do_lseek(int fd, off_t offset, int origin);
void* shim_do_mmap(void* addr, size_t length, int prot, int flags, int fd, unsigned long offset);
long shim_do_mprotect(void* addr, size_t len, int prot);
long shim_do_munmap(void* addr, size_t len);
void* shim_do_brk(void* brk);
long shim_do_rt_sigaction(int signum, const struct __kernel_sigaction* act,
                          struct __kernel_sigaction* oldact, size_t sigsetsize);
long shim_do_rt_sigprocmask(int how, const __sigset_t* set, __sigset_t* oldset, size_t sigsetsize);
long shim_do_rt_sigreturn(void);
long shim_do_ioctl(unsigned int fd, unsigned int cmd, unsigned long arg);
long shim_do_pread64(int fd, char* buf, size_t count, loff_t pos);
long shim_do_pwrite64(int fd, char* buf, size_t count, loff_t pos);
long shim_do_readv(unsigned long fd, const struct iovec* vec, unsigned long vlen);
long shim_do_writev(unsigned long fd, const struct iovec* vec, unsigned long vlen);
long shim_do_access(const char* file, mode_t mode);
long shim_do_pipe(int* fildes);
long shim_do_select(int nfds, fd_set* readfds, fd_set* writefds, fd_set* errorfds,
                    struct __kernel_timeval* timeout);
long shim_do_sched_yield(void);
long shim_do_msync(unsigned long start, size_t len, int flags);
long shim_do_mincore(void* start, size_t len, unsigned char* vec);
long shim_do_madvise(unsigned long start, size_t len_in, int behavior);
long shim_do_dup(unsigned int fd);
long shim_do_dup2(unsigned int oldfd, unsigned int newfd);
long shim_do_pause(void);
long shim_do_nanosleep(struct __kernel_timespec* req, struct __kernel_timespec* rem);
long shim_do_getitimer(int which, struct __kernel_itimerval* value);
long shim_do_alarm(unsigned int seconds);
long shim_do_setitimer(int which, struct __kernel_itimerval* value,
                       struct __kernel_itimerval* ovalue);
long shim_do_getpid(void);
long shim_do_sendfile(int out_fd, int in_fd, off_t* offset, size_t count);
long shim_do_socket(int family, int type, int protocol);
long shim_do_connect(int sockfd, struct sockaddr* addr, int addrlen);
long shim_do_accept(int fd, struct sockaddr* addr, int* addrlen);
long shim_do_sendto(int fd, const void* buf, size_t len, int flags,
                    const struct sockaddr* dest_addr, int addrlen);
long shim_do_recvfrom(int fd, void* buf, size_t len, int flags, struct sockaddr* addr,
                      int* addrlen);
long shim_do_bind(int sockfd, struct sockaddr* addr, int addrlen);
long shim_do_listen(int sockfd, int backlog);
long shim_do_sendmsg(int fd, struct msghdr* msg, int flags);
long shim_do_recvmsg(int fd, struct msghdr* msg, int flags);
long shim_do_shutdown(int sockfd, int how);
long shim_do_getsockname(int sockfd, struct sockaddr* addr, int* addrlen);
long shim_do_getpeername(int sockfd, struct sockaddr* addr, int* addrlen);
long shim_do_socketpair(int domain, int type, int protocol, int* sv);
long shim_do_setsockopt(int fd, int level, int optname, char* optval, int optlen);
long shim_do_getsockopt(int fd, int level, int optname, char* optval, int* optlen);
long shim_do_clone(unsigned long flags, unsigned long user_stack_addr, int* parent_tidptr,
                   int* child_tidptr, unsigned long tls);
long shim_do_fork(void);
long shim_do_vfork(void);
long shim_do_execve(const char* file, const char** argv, const char** envp);
long shim_do_exit(int error_code);
long shim_do_waitid(int which, pid_t id, siginfo_t* infop, int options, struct __kernel_rusage* ru);
long shim_do_wait4(pid_t pid, int* stat_addr, int options, struct __kernel_rusage* ru);
long shim_do_kill(pid_t pid, int sig);
long shim_do_uname(struct new_utsname* buf);
long shim_do_fcntl(int fd, int cmd, unsigned long arg);
long shim_do_fsync(int fd);
long shim_do_fdatasync(int fd);
long shim_do_truncate(const char* path, loff_t length);
long shim_do_ftruncate(int fd, loff_t length);
long shim_do_fallocate(int fd, int mode, loff_t offset, loff_t len);
long shim_do_getdents(int fd, struct linux_dirent* buf, unsigned int count);
long shim_do_getcwd(char* buf, size_t size);
long shim_do_chdir(const char* filename);
long shim_do_fchdir(int fd);
long shim_do_rename(const char* oldname, const char* newname);
long shim_do_mkdir(const char* pathname, int mode);
long shim_do_rmdir(const char* pathname);
long shim_do_creat(const char* path, mode_t mode);
long shim_do_unlink(const char* file);
long shim_do_readlink(const char* file, char* buf, int bufsize);
long shim_do_chmod(const char* filename, mode_t mode);
long shim_do_fchmod(int fd, mode_t mode);
long shim_do_chown(const char* filename, uid_t user, gid_t group);
long shim_do_fchown(int fd, uid_t user, gid_t group);
long shim_do_umask(mode_t mask);
long shim_do_gettimeofday(struct __kernel_timeval* tv, struct __kernel_timezone* tz);
long shim_do_getrlimit(int resource, struct __kernel_rlimit* rlim);
long shim_do_getuid(void);
long shim_do_getgid(void);
long shim_do_setuid(uid_t uid);
long shim_do_setgid(gid_t gid);
long shim_do_setgroups(int gidsetsize, gid_t* grouplist);
long shim_do_getgroups(int gidsetsize, gid_t* grouplist);
long shim_do_geteuid(void);
long shim_do_getegid(void);
long shim_do_getppid(void);
long shim_do_setpgid(pid_t pid, pid_t pgid);
long shim_do_getpgrp(void);
long shim_do_setsid(void);
long shim_do_getpgid(pid_t pid);
long shim_do_getsid(pid_t pid);
long shim_do_rt_sigpending(__sigset_t* set, size_t sigsetsize);
long shim_do_rt_sigtimedwait(const __sigset_t* unblocked_ptr, siginfo_t* info,
                             struct __kernel_timespec* timeout, size_t setsize);
long shim_do_sigaltstack(const stack_t* ss, stack_t* oss);
long shim_do_setpriority(int which, int who, int niceval);
long shim_do_getpriority(int which, int who);
long shim_do_sched_setparam(pid_t pid, struct __kernel_sched_param* param);
long shim_do_sched_getparam(pid_t pid, struct __kernel_sched_param* param);
long shim_do_sched_setscheduler(pid_t pid, int policy, struct __kernel_sched_param* param);
long shim_do_sched_getscheduler(pid_t pid);
long shim_do_sched_get_priority_max(int policy);
long shim_do_sched_get_priority_min(int policy);
long shim_do_sched_rr_get_interval(pid_t pid, struct timespec* interval);
long shim_do_mlock(unsigned long start, size_t len);
long shim_do_munlock(unsigned long start, size_t len);
long shim_do_mlockall(int flags);
long shim_do_munlockall(void);
long shim_do_rt_sigsuspend(const __sigset_t* mask, size_t setsize);
long shim_do_arch_prctl(int code, unsigned long addr);
long shim_do_setrlimit(int resource, struct __kernel_rlimit* rlim);
long shim_do_chroot(const char* filename);
long shim_do_sethostname(char* name, int len);
long shim_do_setdomainname(char* name, int len);
long shim_do_gettid(void);
long shim_do_tkill(int pid, int sig);
long shim_do_time(time_t* tloc);
long shim_do_futex(int* uaddr, int op, int val, void* utime, int* uaddr2, int val3);
long shim_do_sched_setaffinity(pid_t pid, unsigned int cpumask_size, unsigned long* user_mask_ptr);
long shim_do_sched_getaffinity(pid_t pid, unsigned int cpumask_size, unsigned long* user_mask_ptr);
long shim_do_set_tid_address(int* tidptr);
long shim_do_epoll_create(int size);
long shim_do_getdents64(int fd, struct linux_dirent64* buf, size_t count);
long shim_do_epoll_wait(int epfd, struct epoll_event* events, int maxevents, int timeout_ms);
long shim_do_epoll_ctl(int epfd, int op, int fd, struct epoll_event* event);
long shim_do_clock_gettime(clockid_t which_clock, struct timespec* tp);
long shim_do_clock_getres(clockid_t which_clock, struct timespec* tp);
long shim_do_clock_nanosleep(clockid_t clock_id, int flags, struct __kernel_timespec* req,
                             struct __kernel_timespec* rem);
long shim_do_exit_group(int error_code);
long shim_do_tgkill(int tgid, int pid, int sig);
long shim_do_mbind(void* start, unsigned long len, int mode, unsigned long* nmask,
                   unsigned long maxnode, int flags);
long shim_do_openat(int dfd, const char* filename, int flags, int mode);
long shim_do_mkdirat(int dfd, const char* pathname, int mode);
long shim_do_newfstatat(int dirfd, const char* pathname, struct stat* statbuf, int flags);
long shim_do_unlinkat(int dfd, const char* pathname, int flag);
long shim_do_readlinkat(int dirfd, const char* file, char* buf, int bufsize);
long shim_do_renameat(int olddfd, const char* pathname, int newdfd, const char* newname);
long shim_do_fchmodat(int dfd, const char* filename, mode_t mode);
long shim_do_fchownat(int dfd, const char* filename, uid_t user, gid_t group, int flags);
long shim_do_faccessat(int dfd, const char* filename, mode_t mode);
long shim_do_pselect6(int nfds, fd_set* readfds, fd_set* writefds, fd_set* exceptfds,
                      const struct __kernel_timespec* tsp, void* sigmask_argpack);
long shim_do_ppoll(struct pollfd* fds, unsigned int nfds, struct timespec* tsp,
                   const __sigset_t* sigmask_ptr, size_t sigsetsize);
long shim_do_set_robust_list(struct robust_list_head* head, size_t len);
long shim_do_get_robust_list(pid_t pid, struct robust_list_head** head, size_t* len);
long shim_do_epoll_pwait(int epfd, struct epoll_event* events, int maxevents, int timeout_ms,
                         const __sigset_t* sigmask, size_t sigsetsize);
long shim_do_accept4(int sockfd, struct sockaddr* addr, int* addrlen, int flags);
long shim_do_dup3(unsigned int oldfd, unsigned int newfd, int flags);
long shim_do_epoll_create1(int flags);
long shim_do_pipe2(int* fildes, int flags);
long shim_do_mknod(const char* pathname, mode_t mode, dev_t dev);
long shim_do_mknodat(int dirfd, const char* pathname, mode_t mode, dev_t dev);
long shim_do_recvmmsg(int sockfd, struct mmsghdr* msg, unsigned int vlen, int flags,
                      struct __kernel_timespec* timeout);
long shim_do_prlimit64(pid_t pid, int resource, const struct __kernel_rlimit64* new_rlim,
                       struct __kernel_rlimit64* old_rlim);
long shim_do_sendmmsg(int sockfd, struct mmsghdr* msg, unsigned int vlen, int flags);
long shim_do_eventfd2(unsigned int count, int flags);
long shim_do_eventfd(unsigned int count);
long shim_do_getcpu(unsigned* cpu, unsigned* node, struct getcpu_cache* unused);
long shim_do_getrandom(char* buf, size_t count, unsigned int flags);
long shim_do_mlock2(unsigned long start, size_t len, int flags);
long shim_do_sysinfo(struct sysinfo* info);

#define GRND_NONBLOCK 0x0001
#define GRND_RANDOM   0x0002
#define GRND_INSECURE 0x0004

#ifndef MADV_FREE
#define MADV_FREE 8
#endif
#ifdef __x86_64__
#ifndef MADV_WIPEONFORK
#define MADV_WIPEONFORK 18
#endif
#ifndef MADV_KEEPONFORK
#define MADV_KEEPONFORK 19
#endif
#else /* __x86_64__ */
#error "Unsupported platform"
#endif

#endif /* _SHIM_TABLE_H_ */
