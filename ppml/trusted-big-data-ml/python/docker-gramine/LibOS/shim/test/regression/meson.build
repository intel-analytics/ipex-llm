tests = {
    'abort': {},
    'abort_multithread': {},
    'bootstrap': {},
    'bootstrap_pie': {
        'pie': true,
    },
    'bootstrap_static': {
        'static': true,
    },
    'debug': {
        'c_args': '-g3',
    },
    'devfs': {},
    'device_passthrough': {},
    'double_fork': {},
    'epoll_epollet': {},
    'epoll_test': {},
    'eventfd': {},
    'exec': {},
    'exec_fork': {},
    'exec_invalid_args': {},
    'exec_same': {},
    'exec_script': {},
    'exec_victim': {},
    'exit': {},
    'exit_group': {},
    'fcntl_lock': {},
    'fdleak': {},
    'file_check_policy': {},
    'file_size': {},
    'fopen_cornercases': {},
    'fork_and_exec': {},
    'fp_multithread': {
        'c_args': '-fno-builtin',  # see comment in the test's source
        'link_args': '-lm',
    },
    'fstat_cwd': {},
    'futex_bitset': {},
    'futex_requeue': {},
    'futex_timeout': {},
    'futex_wake_op': {},
    'getcwd': {},
    'getdents': {},
    'getdents_lseek': {},
    'getsockname': {},
    'getsockopt': {},
    'gettimeofday': {},
    'groups': {},
    'helloworld': {},
    'host_root_fs': {},
    'init_fail': {},
    'keys': {},
    'kill_all': {},
    'large_dir_read': {},
    'large_file': {},
    'large_mmap': {},
    'madvise': {},
    'mkfifo': {},
    'mmap_file': {},
    'mmap_file_backed': {},
    'mmap_file_emulated': {},
    'mprotect_file_fork': {},
    'mprotect_prot_growsdown': {},
    'multi_pthread': {},
    'openmp': {
        # NOTE: This will use `libgomp` in GCC and `libomp` in Clang.
        'c_args': '-fopenmp',
        'link_args': '-fopenmp',
    },
    'pipe': {},
    'pipe_nonblocking': {},
    'pipe_ocloexec': {},
    'poll': {},
    'poll_closed_fd': {},
    'poll_many_types': {},
    'ppoll': {},
    'proc_common': {},
    'proc_cpuinfo': {},
    'proc_path': {},
    'proc_stat': {},
    'pselect': {},
    'pthread_set_get_affinity': {},
    'readdir': {},
    'rename_unlink': {},
    'run_test': {
        'include_directories': include_directories(
            # for `shim_entry_api.h`
            '../../include',
            join_paths('../../include/arch', host_machine.cpu_family()),
        ),
    },
    'sched': {},
    'sched_set_get_affinity': {},
    'sealed_file': {},
    'sealed_file_mod': {
        'source': 'sealed_file.c',
        'c_args': '-DMODIFY_MRENCLAVE',  # see comment in the test's source
    },
    'select': {},
    'send_handle': {},
    'shared_object': {
        'pie': true,
    },
    'sigaction_per_process': {},
    'sigaltstack': {},
    'sighandler_reset': {},
    'sighandler_sigpipe': {},
    'signal_multithread': {},
    'sigprocmask_pending': {},
    'spinlock': {
        'include_directories': include_directories(
            # for `spinlock.h`
            '../../../../common/include',
            join_paths('../../../../common/include/arch', host_machine.cpu_family()),
        ),
    },
    'stat_invalid_args': {},
    'synthetic': {},
    'syscall': {},
    'syscall_restart': {},
    'sysfs_common': {},
    'tcp_ipv6_v6only': {},
    'tcp_msg_peek': {},
    'udp': {},
    'uid_gid': {},
    'unix': {},
    'vfork_and_exec': {},
}

if host_machine.cpu_family() == 'x86_64'
    tests += {
        'cpuid': {},
        'debug_regs-x86_64': {
            'c_args': '-g3',
        },
        'rdtsc': {},
        'sighandler_divbyzero': {},
    }
endif

tests_musl = tests

if host_machine.cpu_family() == 'x86_64'
    # We use musl-gcc wrapper, which does not support building c++, so this test is not enabled
    # with musl.
    tests += {
        'bootstrap_cpp': {
            'source': 'bootstrap_cpp.cpp',
        },
        'ioctl': {
            'source': 'ioctl.cpp',
        },
    }
endif

if sgx
    tests += {
        'attestation': {
            # for `sgx_arch.h`
            'include_directories': include_directories('../../../../Pal/src/host/Linux-SGX'),
            'source': [
                'attestation.c',
                '../../../../Pal/src/host/Linux-SGX/enclave_api.S',
            ],

            'dependencies': mbedtls_static_dep,
        },
    }
endif

install_dir = join_paths(pkglibdir, 'tests', 'libos', 'regression')

configurations = []
if enable_glibc
    common_lib = static_library('test_libos',
        'dump.c',
        'rw_file.c',
    )
    configurations += {
        'use_musl': false,
        'common_lib': common_lib,
        'tests': tests,
    }
endif
if enable_musl
    if meson.get_compiler('c').get_id() != 'gcc'
        error('Compiling tests is currently unsupported with Musl and compilers other than GCC. ' +
              'You need to either disable Musl (-Dmusl=disabled) or use GCC (CC=gcc).')
    endif

    musl_specs_path_cmd = run_command('sh', '-c',
        'grep -Po -e \'-specs "\K([^"]*)\' `which musl-gcc`')
    musl_specs_path = musl_specs_path_cmd.stdout().strip()
    if musl_specs_path == ''
        error('failed to find "musl-gcc", is it installed?')
    endif

    common_lib_musl = static_library('test_libos_musl',
        'dump.c',
        'rw_file.c',
        # XXX: this is a hack, to be removed when we fix building tests
        c_args: [ '-specs', musl_specs_path ],
        link_args: [ '-specs', musl_specs_path ],
    )
    configurations += {
        'use_musl': true,
        'common_lib': common_lib_musl,
        'tests': tests_musl,
    }

    musl_execs = []
endif

foreach conf: configurations
    foreach name, params : conf['tests']
        pie = params.get('pie', false)
        static = params.get('static', false)

        c_args = [
            # TODO: Many files define `main` with parameters, but don't use them
            '-Wno-unused-parameter',
        ]

        link_args = [
            '-pthread',
        ]

        if static
            link_args += '-static'
        endif

        # Note that we cannot use `pie: true/false`, because this doesn't properly disable PIE if the
        # compiler builds it by default. See issue: https://github.com/mesonbuild/meson/issues/4651
        if pie
            c_args += '-fPIE'
            link_args += '-pie'
        else
            c_args += '-fno-PIE'
            if not static
                # When building static binaries, Clang complains that the `-no-pie` flag is unused.
                link_args += '-no-pie'
            endif
        endif

        name_prefix = ''
        if conf['use_musl']
            # XXX: this is a hack, to be removed when we fix building tests
            c_args += ['-specs', musl_specs_path]
            link_args += ['-specs', musl_specs_path]
            # This prefix will be stripped when installing executables to a designated directory
            # below.
            name_prefix = 'musl_'
        endif

        exe = executable(name_prefix + name,
            params.get('source', '@0@.c'.format(name)),

            include_directories: params.get('include_directories', []),

            link_with: conf['common_lib'],

            c_args: [
                c_args,
                params.get('c_args', []),
            ],

            link_args: [
                link_args,
                params.get('link_args', []),
            ],

            dependencies: params.get('dependencies', []),

            install: true,
            install_dir: install_dir,
        )

        if conf['use_musl']
            musl_execs += exe
        endif
    endforeach
endforeach

if enable_musl
    meson.add_install_script(
        find_program('install_musl_tests.sh'),
        prefix / join_paths(install_dir, 'musl'),
        musl_execs,
    )
endif
