/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This source file contains functions to create a child process and terminate the running process.
 * Child does not inherit any objects or memory from its parent process. A parent process may not
 * modify the execution of its children. It can wait for a child to exit using its handle. Also,
 * parent and child may communicate through I/O streams provided by the parent to the child at
 * creation.
 */

#include <asm/errno.h>
#include <asm/fcntl.h>

#include "linux_utils.h"
#include "sgx_internal.h"
#include "sgx_process.h"

extern char* g_pal_loader_path;
extern char* g_libpal_path;

struct proc_args {
    size_t       application_path_size; // application path will follow this struct on the pipe.
    size_t       manifest_size; // manifest will follow application path on the pipe.
};

static int vfork_exec(const char** argv) {
    int ret = vfork();
    if (ret)
        return ret;

    extern char** environ;
    DO_SYSCALL(execve, g_pal_loader_path, argv, environ);
    DO_SYSCALL(exit_group, 1);
    die_or_inf_loop();
}

int sgx_create_process(size_t nargs, const char** args, const char* manifest, int* out_stream_fd) {
    int ret, rete;
    int fds[2] = {-1, -1};

    int socktype = SOCK_STREAM;
    ret = DO_SYSCALL(socketpair, AF_UNIX, socktype, 0, fds);
    if (ret < 0)
        goto out;

    ret = DO_SYSCALL(fcntl, fds[1], F_SETFD, FD_CLOEXEC);
    if (ret < 0) {
        goto out;
    }

    const char** argv = __alloca(sizeof(const char*) * (nargs + 5));
    argv[0] = g_pal_loader_path;
    argv[1] = g_libpal_path;
    argv[2] = "child";
    char parent_fd_str[16];
    snprintf(parent_fd_str, sizeof(parent_fd_str), "%u", fds[0]);
    argv[3] = parent_fd_str;
    memcpy(argv + 4, args, sizeof(const char*) * nargs);
    argv[nargs + 4] = NULL;

    /* child's signal handler may mess with parent's memory during vfork(), so block signals */
    ret = block_async_signals(true);
    if (ret < 0) {
        goto out;
    }

    ret = vfork_exec(argv);
    if (ret < 0)
        goto out;

    /* parent continues here */

    /* children unblock async signals by sgx_signal_setup() */
    ret = block_async_signals(false);
    if (ret < 0) {
        goto out;
    }

    /* TODO: add error checking. */
    DO_SYSCALL(close, fds[0]); /* child stream */

    struct proc_args proc_args;
    proc_args.application_path_size = strlen(g_pal_enclave.application_path);
    proc_args.manifest_size         = strlen(manifest);

    ret = write_all(fds[1], &proc_args, sizeof(struct proc_args));
    if (ret < 0) {
        goto out;
    }

    ret = write_all(fds[1], g_pal_enclave.application_path, proc_args.application_path_size);
    if (ret < 0) {
        goto out;
    }

    ret = write_all(fds[1], manifest, proc_args.manifest_size);
    if (ret < 0) {
        goto out;
    }

    ret = read_all(fds[1], &rete, sizeof(rete));
    if (ret < 0) {
        goto out;
    }

    if (rete < 0) {
        ret = rete;
        goto out;
    }

    *out_stream_fd = fds[1];

    ret = 0;
out:
    if (ret < 0) {
        if (fds[0] >= 0)
            DO_SYSCALL(close, fds[0]);
        if (fds[1] >= 0)
            DO_SYSCALL(close, fds[1]);
    }

    return ret;
}

int sgx_init_child_process(int parent_stream_fd, char** out_application_path, char** out_manifest) {
    int ret;
    struct proc_args proc_args;
    char* manifest = NULL;
    char* application_path = NULL;

    ret = read_all(parent_stream_fd, &proc_args, sizeof(struct proc_args));
    if (ret < 0) {
        goto out;
    }

    application_path = malloc(proc_args.application_path_size + 1);
    if (!application_path) {
        ret = -ENOMEM;
        goto out;
    }

    manifest = malloc(proc_args.manifest_size + 1);
    if (!manifest) {
        ret = -ENOMEM;
        goto out;
    }

    ret = read_all(parent_stream_fd, application_path, proc_args.application_path_size);
    if (ret < 0) {
        goto out;
    }
    application_path[proc_args.application_path_size] = '\0';

    ret = read_all(parent_stream_fd, manifest, proc_args.manifest_size);
    if (ret < 0) {
        goto out;
    }
    manifest[proc_args.manifest_size] = '\0';

    int child_status = 0;
    ret = write_all(parent_stream_fd, &child_status, sizeof(child_status));
    if (ret < 0) {
        goto out;
    }

    *out_application_path = application_path;
    *out_manifest = manifest;
    ret = 0;
out:
    if (ret < 0) {
        free(application_path);
        free(manifest);
    }

    return ret;
}
