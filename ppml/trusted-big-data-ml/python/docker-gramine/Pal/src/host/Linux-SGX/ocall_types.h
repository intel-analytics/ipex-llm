/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Corporation */

#ifndef OCALL_TYPES_H
#define OCALL_TYPES_H

/*
 * These structures are used in trusted -> untrusted world calls (OCALLS).
 */

#include <stdbool.h>
#include <stddef.h>

#include "linux_types.h"
#include "pal.h"
#include "sgx_arch.h"
#include "sgx_attest.h"

/*
 * These structures must be packed, otherwise will leak data in their padding when copied to
 * untrusted world.
 */
#pragma pack(push, 1)

typedef long (*sgx_ocall_fn_t)(void*);

enum {
    OCALL_EXIT = 0,
    OCALL_MMAP_UNTRUSTED,
    OCALL_MUNMAP_UNTRUSTED,
    OCALL_CPUID,
    OCALL_OPEN,
    OCALL_CLOSE,
    OCALL_READ,
    OCALL_WRITE,
    OCALL_PREAD,
    OCALL_PWRITE,
    OCALL_FSTAT,
    OCALL_FIONREAD,
    OCALL_FSETNONBLOCK,
    OCALL_FCHMOD,
    OCALL_FSYNC,
    OCALL_FTRUNCATE,
    OCALL_MKDIR,
    OCALL_GETDENTS,
    OCALL_RESUME_THREAD,
    OCALL_SCHED_SETAFFINITY,
    OCALL_SCHED_GETAFFINITY,
    OCALL_CLONE_THREAD,
    OCALL_CREATE_PROCESS,
    OCALL_FUTEX,
    OCALL_LISTEN,
    OCALL_ACCEPT,
    OCALL_CONNECT,
    OCALL_RECV,
    OCALL_SEND,
    OCALL_SETSOCKOPT,
    OCALL_SHUTDOWN,
    OCALL_GETTIME,
    OCALL_SCHED_YIELD,
    OCALL_POLL,
    OCALL_RENAME,
    OCALL_DELETE,
    OCALL_DEBUG_MAP_ADD,
    OCALL_DEBUG_MAP_REMOVE,
    OCALL_DEBUG_DESCRIBE_LOCATION,
    OCALL_EVENTFD,
    OCALL_IOCTL,
    OCALL_GET_QUOTE,
    OCALL_NR,
};

typedef struct {
    int ms_exitcode;
    int ms_is_exitgroup;
} ms_ocall_exit_t;

typedef struct {
    void* ms_addr;
    size_t ms_size;
    int ms_prot;
    int ms_flags;
    int ms_fd;
    off_t ms_offset;
} ms_ocall_mmap_untrusted_t;

typedef struct {
    const void* ms_addr;
    size_t ms_size;
} ms_ocall_munmap_untrusted_t;

typedef struct {
    unsigned int ms_leaf;
    unsigned int ms_subleaf;
    unsigned int ms_values[4];
} ms_ocall_cpuid_t;

typedef struct {
    const char* ms_pathname;
    int ms_flags;
    unsigned short ms_mode;
} ms_ocall_open_t;

typedef struct {
    int ms_fd;
} ms_ocall_close_t;

typedef struct {
    int ms_fd;
    void* ms_buf;
    unsigned int ms_count;
} ms_ocall_read_t;

typedef struct {
    int ms_fd;
    const void* ms_buf;
    unsigned int ms_count;
} ms_ocall_write_t;

typedef struct {
    int ms_fd;
    void* ms_buf;
    size_t ms_count;
    off_t ms_offset;
} ms_ocall_pread_t;

typedef struct {
    int ms_fd;
    const void* ms_buf;
    size_t ms_count;
    off_t ms_offset;
} ms_ocall_pwrite_t;

typedef struct {
    int ms_fd;
    struct stat ms_stat;
} ms_ocall_fstat_t;

typedef struct {
    int ms_fd;
} ms_ocall_fionread_t;

typedef struct {
    int ms_fd;
    int ms_nonblocking;
} ms_ocall_fsetnonblock_t;

typedef struct {
    int ms_fd;
    unsigned short ms_mode;
} ms_ocall_fchmod_t;

typedef struct {
    int ms_fd;
} ms_ocall_fsync_t;

typedef struct {
    int ms_fd;
    uint64_t ms_length;
} ms_ocall_ftruncate_t;

typedef struct {
    const char* ms_pathname;
    unsigned short ms_mode;
} ms_ocall_mkdir_t;

typedef struct {
    int ms_fd;
    struct linux_dirent64* ms_dirp;
    size_t ms_size;
} ms_ocall_getdents_t;

typedef struct {
    int ms_stream_fd;
    size_t ms_nargs;
    const char* ms_args[];
} ms_ocall_create_process_t;

typedef struct {
    void* ms_tcs;
    size_t ms_cpumask_size;
    void* ms_cpu_mask;
} ms_ocall_sched_setaffinity_t;

typedef struct {
    void* ms_tcs;
    size_t ms_cpumask_size;
    void* ms_cpu_mask;
} ms_ocall_sched_getaffinity_t;

typedef struct {
    uint32_t* ms_futex;
    int ms_op, ms_val;
    uint64_t ms_timeout_us;
} ms_ocall_futex_t;

typedef struct {
    int ms_domain;
    int ms_type;
    int ms_protocol;
    int ms_ipv6_v6only;
    const struct sockaddr* ms_addr;
    size_t ms_addrlen;
} ms_ocall_listen_t;

typedef struct {
    int ms_sockfd;
    int options;
    struct sockaddr* ms_addr;
    size_t ms_addrlen;
    struct sockaddr* ms_bind_addr;
    size_t ms_bind_addrlen;
} ms_ocall_accept_t;

typedef struct {
    int ms_domain;
    int ms_type;
    int ms_protocol;
    int ms_ipv6_v6only;
    const struct sockaddr* ms_addr;
    size_t ms_addrlen;
    struct sockaddr* ms_bind_addr;
    size_t ms_bind_addrlen;
} ms_ocall_connect_t;

typedef struct {
    PAL_IDX ms_sockfd;
    void* ms_buf;
    size_t ms_count;
    struct sockaddr* ms_addr;
    size_t ms_addrlen;
    void* ms_control;
    size_t ms_controllen;
} ms_ocall_recv_t;

typedef struct {
    PAL_IDX ms_sockfd;
    const void* ms_buf;
    size_t ms_count;
    const struct sockaddr* ms_addr;
    size_t ms_addrlen;
    void* ms_control;
    size_t ms_controllen;
} ms_ocall_send_t;

typedef struct {
    int ms_sockfd;
    int ms_level;
    int ms_optname;
    const void* ms_optval;
    size_t ms_optlen;
} ms_ocall_setsockopt_t;

typedef struct {
    int ms_sockfd;
    int ms_how;
} ms_ocall_shutdown_t;

typedef struct {
    uint64_t ms_microsec;
} ms_ocall_gettime_t;

typedef struct {
    struct pollfd* ms_fds;
    size_t ms_nfds;
    uint64_t ms_timeout_us;
} ms_ocall_poll_t;

typedef struct {
    const char* ms_oldpath;
    const char* ms_newpath;
} ms_ocall_rename_t;

typedef struct {
    const char* ms_pathname;
} ms_ocall_delete_t;

typedef struct {
    const char* ms_name;
    void* ms_addr;
} ms_ocall_debug_map_add_t;

typedef struct {
    void* ms_addr;
} ms_ocall_debug_map_remove_t;

typedef struct {
    uintptr_t ms_addr;
    char* ms_buf;
    size_t ms_buf_size;
} ms_ocall_debug_describe_location_t;

typedef struct {
    int          ms_flags;
} ms_ocall_eventfd_t;

typedef struct {
    int           ms_fd;
    unsigned int  ms_cmd;
    unsigned long ms_arg;
} ms_ocall_ioctl_t;

typedef struct {
    bool              ms_is_epid;
    sgx_spid_t        ms_spid;
    bool              ms_linkable;
    sgx_report_t      ms_report;
    sgx_quote_nonce_t ms_nonce;
    char*             ms_quote;
    size_t            ms_quote_len;
} ms_ocall_get_quote_t;

#pragma pack(pop)

#endif /* OCALL_TYPES_H */
