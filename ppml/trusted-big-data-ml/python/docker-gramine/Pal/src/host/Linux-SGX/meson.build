gsgx_h = configure_file(
    input: 'gsgx.h.in',
    output: 'gsgx.h',
    configuration: conf_sgx,
)

sgx_inc = [
    includes_pal_common,
    include_directories(
        '.',
        '../../../include/arch/@0@/Linux'.format(host_machine.cpu_family()),
        '../../../include/host/Linux-common',
    ),
]

cflags_pal_sgx = [
    cflags_pal_common,
    '-DHOST_TYPE=Linux-SGX',
    # Some of the code uses `alignof` on expressions, which is a GNU extension. Silence Clang - it
    # complains but does support it.
    '-Wno-gnu-alignof-expression',
]

pal_sgx_generated_offsets_print = executable('pal-sgx-generated-offsets-print',
    'generated-offsets.c',
    generated_offsets_print_src,
    include_directories: sgx_inc,
    c_args: cflags_pal_sgx,
)

pal_sgx_asm_offsets_h = custom_target('asm-offsets.h[pal-sgx]',
    command: [pal_sgx_generated_offsets_print, '--h'],
    capture: true,
    output: 'asm-offsets.h',
)

pal_sgx_offsets_py = custom_target('_offsets.py',
    command: [pal_sgx_generated_offsets_print, '--py'],
    capture: true,
    output: '_offsets.py',

    install: true,
    install_dir: python3_pkgdir,
)

pal_sgx_map = custom_target('pal.map',
    command: [gen_pal_map_prog, '@INPUT@', '@OUTPUT@'],
    input: ['../../pal-symbols', 'pal.map.template'],
    output: 'pal.map',
)

pal_sgx_lds = join_paths(meson.current_source_dir(), 'enclave.lds')

libpal_sgx = shared_library('pal',
    'db_devices.c',
    'db_eventfd.c',
    'db_events.c',
    'db_exception.c',
    'db_files.c',
    'db_main.c',
    'db_memory.c',
    'db_misc.c',
    'db_object.c',
    'db_pipes.c',
    'db_process.c',
    'db_rtld.c',
    'db_sockets.c',
    'db_streams.c',
    'db_threading.c',
    'enclave_api.S',
    'enclave_ecalls.c',
    'enclave_entry.S',
    'enclave_framework.c',
    'enclave_ocalls.c',
    'enclave_pages.c',
    'enclave_platform.c',
    'enclave_untrusted.c',
    'enclave_xstate.c',
    pal_sgx_asm_offsets_h,
    pal_common_sources,
    pal_linux_common_sources_enclave,
    gsgx_h,

    include_directories: sgx_inc,

    c_args: [
        cflags_pal_sgx,
        cflags_sanitizers,
        cflags_custom_stack_protector,
        '-DIN_ENCLAVE',
    ],

    link_args: [
        '-nostdlib',

        # For PAL symbol resolution, we use old-style hash (DT_HASH) table instead of the default
        # GNU hash (DT_GNU_HASH) table -- to figure out the number of dynamic symbols in the SYMTAB
        # symbol table
        '-Wl,--hash-style=sysv',

        # PAL is expected to be linked with RELRO (Relocation Read-Only)
        '-Wl,-zrelro',
        '-Wl,-znow',

        '-Wl,-zcombreloc',
        '-Wl,-zdefs',
        '-Wl,--version-script=@0@'.format(pal_sgx_map.full_path()),
        '-Wl,-T@0@'.format(pal_sgx_lds),
        '-Wl,--eh-frame-hdr',
    ],

    link_depends: [
        pal_sgx_map,
        pal_sgx_lds,
    ],

    dependencies: [
        common_dep,
        cryptoadapter_dep,
    ],

    install: true,
    install_dir: join_paths(pkglibdir, 'sgx'),
)

# URTS (untrusted runtime)
libpal_sgx_urts = executable('loader',
    'sgx_enclave.c',
    'sgx_entry.S',
    'sgx_exception.c',
    'sgx_framework.c',
    'sgx_log.c',
    'sgx_main.c',
    'sgx_perf_data.c',
    'sgx_platform.c',
    'sgx_process.c',
    'sgx_profile.c',
    'sgx_syscall.c',
    'sgx_thread.c',
    pal_linux_common_sources_urts,
    pal_sgx_asm_offsets_h,
    aesm_proto_ch,
    gsgx_h,

    include_directories: sgx_inc,
    c_args: [
        cflags_pal_sgx,
        cflags_sanitizers,

        # URTS uses stack protector with standard parameters (not the ones defined for Pal/LibOS)
        '-fstack-protector-strong',
    ],
    link_args: [
        '-Wl,-zrelro',
        '-Wl,-znow',
    ],
    pie: true,

    dependencies: [
        common_dep,
        protobuf_dep,
        vtune_dep,
        dl_dep,
    ],

    install: true,
    install_dir: join_paths(pkglibdir, 'sgx'),
)

libpal_sgx_dep = declare_dependency(
    link_with: libpal_sgx,
    include_directories: includes_pal_common,
)

if debug
    custom_target('libpal.so.map',
        input: libpal_sgx,
        command: gen_symbol_map_cmd,
        capture: true,
        output: 'libpal.so.map',

        install: true,
        install_dir: join_paths(pkglibdir, 'sgx'),
    )

    custom_target('loader.map',
        input: libpal_sgx_urts,
        command: gen_symbol_map_cmd,
        capture: true,
        output: 'loader.map',

        install: true,
        install_dir: join_paths(pkglibdir, 'sgx'),
    )

    shared_library('sgx_gdb',
        'gdb_integration/sgx_gdb.c',

        # just `sgx_gdb.so`, without `lib`
        name_prefix: '',

        c_args: [
            # optimize even in a debug build
            '-O3',
        ],

        install: true,
        install_dir: join_paths(pkglibdir, 'sgx/gdb_integration'),
    )

    install_data(
        'gdb_integration/gramine_sgx.gdb',
        'gdb_integration/gramine_sgx_gdb.py',
        install_dir: join_paths(pkglibdir, 'sgx/gdb_integration'),
    )
endif

subdir('tools')
