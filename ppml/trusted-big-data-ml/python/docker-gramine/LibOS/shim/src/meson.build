if skeleton
    libpal = libpal_skeleton_dep
elif direct
    libpal = libpal_direct_dep
elif sgx
    libpal = libpal_sgx_dep
else
    error('need to link LibOS against a PAL library, but no PAL version is enabled')
endif

libos_sources = files(
    'bookkeep/shim_handle.c',
    'bookkeep/shim_pid.c',
    'bookkeep/shim_process.c',
    'bookkeep/shim_signal.c',
    'bookkeep/shim_thread.c',
    'bookkeep/shim_vma.c',
    'fs/chroot/encrypted.c',
    'fs/chroot/fs.c',
    'fs/dev/attestation.c',
    'fs/dev/fs.c',
    'fs/eventfd/fs.c',
    'fs/pipe/fs.c',
    'fs/proc/fs.c',
    'fs/proc/info.c',
    'fs/proc/ipc-thread.c',
    'fs/proc/thread.c',
    'fs/shim_dcache.c',
    'fs/shim_fs.c',
    'fs/shim_fs_encrypted.c',
    'fs/shim_fs_hash.c',
    'fs/shim_fs_lock.c',
    'fs/shim_fs_mem.c',
    'fs/shim_fs_pseudo.c',
    'fs/shim_fs_synthetic.c',
    'fs/shim_fs_util.c',
    'fs/shim_namei.c',
    'fs/socket/fs.c',
    'fs/sys/cache_info.c',
    'fs/sys/cpu_info.c',
    'fs/sys/fs.c',
    'fs/sys/node_info.c',
    'fs/tmpfs/fs.c',
    'ipc/shim_ipc.c',
    'ipc/shim_ipc_child.c',
    'ipc/shim_ipc_fs_lock.c',
    'ipc/shim_ipc_pid.c',
    'ipc/shim_ipc_process_info.c',
    'ipc/shim_ipc_signal.c',
    'ipc/shim_ipc_sync.c',
    'ipc/shim_ipc_vmid.c',
    'ipc/shim_ipc_worker.c',
    'shim_async.c',
    'shim_call.c',
    'shim_checkpoint.c',
    'shim_debug.c',
    'shim_init.c',
    'shim_malloc.c',
    'shim_object.c',
    'shim_parser.c',
    'shim_pollable_event.c',
    'shim_rtld.c',
    'shim_syscalls.c',
    'shim_utils.c',
    'sync/shim_sync_client.c',
    'sync/shim_sync_server.c',
    'sys/shim_access.c',
    'sys/shim_alarm.c',
    'sys/shim_brk.c',
    'sys/shim_clone.c',
    'sys/shim_dup.c',
    'sys/shim_epoll.c',
    'sys/shim_eventfd.c',
    'sys/shim_exec.c',
    'sys/shim_exit.c',
    'sys/shim_fcntl.c',
    'sys/shim_file.c',
    'sys/shim_fork.c',
    'sys/shim_futex.c',
    'sys/shim_getcwd.c',
    'sys/shim_getpid.c',
    'sys/shim_getrandom.c',
    'sys/shim_getrlimit.c',
    'sys/shim_getuid.c',
    'sys/shim_ioctl.c',
    'sys/shim_mlock.c',
    'sys/shim_mmap.c',
    'sys/shim_open.c',
    'sys/shim_pipe.c',
    'sys/shim_poll.c',
    'sys/shim_sched.c',
    'sys/shim_sigaction.c',
    'sys/shim_sleep.c',
    'sys/shim_socket.c',
    'sys/shim_stat.c',
    'sys/shim_time.c',
    'sys/shim_uname.c',
    'sys/shim_wait.c',
    'sys/shim_wrappers.c',
    'utils/log.c',
)

# Arch-specific meson.build must define the following Meson variables:
#   - `libos_sources_arch` - a list of arch-specific sources,
#   - `shim_lds` - a full path to LibOS linker script.
subdir('arch')
libos_sources += libos_sources_arch

shim_map = join_paths(meson.current_source_dir(), 'shim.map')

subdir('vdso')

vdso_data_o = custom_target('vdso-data.o',
    input: 'vdso/vdso-data.S',
    output: 'vdso-data.o',
    command: [
        cc.cmd_array(),
        cflags_libos,
        '-Wa,--noexecstack',
        '-DVDSO_SO_FULL_PATH="@0@"'.format(vdso_so.full_path()),
        '-c',
        '@INPUT@',
        '-o', '@OUTPUT@',
    ],
    depends: vdso_so,
)

gramine_hash_src = vcs_tag(
    command: ['git', 'rev-parse', '--verify', 'HEAD' ],
    input: 'gramine_hash.c.in',
    output: 'gramine_hash.c',
    replace_string: '@GRAMINE_COMMIT_HASH@',
)

libos = shared_library('sysdb',
    libos_sources,
    vdso_data_o,
    libos_asm_offsets_h,
    gramine_hash_src,

    c_args: [
        cflags_libos,
        cflags_sanitizers,

        # use TLS-based stack protector of GCC (we rely on the fact that LibOS reuses the same TCB
        # as its underlying PAL which must have a canary in its PAL_TCB at offset 0x8, so no
        # additional enabling is required in the LibOS code)
        cflags_custom_stack_protector,
    ],
    link_args: [
        '-nostdlib',

        # PAL expects that LibOS binary has old-style hash (DT_HASH) table instead of the default
        # GNU hash (DT_GNU_HASH) table; PAL symbol resolution uses DT_HASH to figure out the number
        # of dynamic symbols in the SYMTAB symbol table
        '-Wl,--hash-style=sysv',

        # PAL expects that LibOS binary is linked with RELRO (Relocation Read-Only)
        '-Wl,-zrelro',
        '-Wl,-znow',

        '-Wl,-zcombreloc',
        '-Wl,--no-undefined',
        '-Wl,--version-script=@0@'.format(shim_map),
        '-Wl,-T@0@'.format(shim_lds),
        '-Wl,--eh-frame-hdr',
    ],
    link_depends: [
        shim_map,
        shim_lds,
    ],
    dependencies: [
        common_dep,
        cryptoadapter_dep,
        protected_files_dep,
        libpal,
    ],

    include_directories: includes_libos,
    install: true,
    install_dir: pkglibdir,
)

if debug
    custom_target('libsysdb.so.map',
        input: libos,
        command: gen_symbol_map_cmd,
        capture: true,
        output: 'libsysdb.so.map',

        install: true,
        install_dir: pkglibdir,
    )
endif
