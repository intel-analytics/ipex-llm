if skeleton
    libpal = libpal_skeleton_dep
elif direct
    libpal = libpal_direct_dep
elif sgx
    libpal = libpal_sgx_dep
else
    error('need to link tests against a PAL library, but no PAL version is enabled')
endif

tests = {
    '..Bootstrap': {},
    'Bootstrap': {},
    'Bootstrap7': {},
    'Directory': {},
    'Event': {},
    'Exit': {},
    'File': {},
    'File2': {},
    'HelloWorld': {},
    'Hex': {},
    'Memory': {},
    'Misc': {},
    'Pie': {
        'pie': true,
    },
    'Pipe': {},
    'Process': {},
    'Process4': {},
    'SendHandle': {},
    'Socket': {},
    'Symbols': {},
    'Tcp': {},
    'Thread2': {},
    'Udp': {},
    'avl_tree_test': {},
    'normalize_path': {},
    'printf_test': {},
}

if host_machine.cpu_family() == 'x86_64'
    tests += {
        'Exception': {},
        'Exception2': {},
        'Segment': {},
        'Thread': {},
    }
endif

if sgx
   tests += {
       'AttestationReport': {
            'include_directories': include_directories(
                # for `sgx_arch.h`
                join_paths('../src/host/Linux-SGX'),
            ),
        },
   }
endif

install_dir = join_paths(pkglibdir, 'tests', 'pal')

test_c_args = [
    cflags_sanitizers,
    cflags_custom_stack_protector,
    '-fno-builtin',
    '-nostdlib',

    # TODO: Many files define `main` with parameters, but don't use them
    '-Wno-unused-parameter',
]

test_include_directories = [
    include_directories(
        # for `pal.h`
        '../include/pal',
        join_paths('../include/arch', host_machine.cpu_family()),
        join_paths('../include/arch', host_machine.cpu_family(), 'Linux'),
    ),
    common_inc,
]

common_lib = static_library('test_pal',
    'utils.c',
    c_args: [
        test_c_args,
        '-fPIC',
    ],
    include_directories: test_include_directories,
    dependencies: [
        common_dep,
    ],
)

# This file is needed to provide an entry point to the executable: GCC expects a symbol named
# `_start` that prepares the stack layout (argc, argv, env) and jumps into `main`.
user_start_src = files(join_paths('crt_init', 'arch', host_machine.cpu_family(), 'user_start.S'))

foreach name, params : tests
    # Note that we cannot use `pie: true/false`, because this doesn't properly disable PIE if the
    # compiler builds it by default. See issue: https://github.com/mesonbuild/meson/issues/4651
    pie = params.get('pie', false)

    executable(name,
        '@0@.c'.format(name),
        user_start_src,

        include_directories: [
            test_include_directories,
            params.get('include_directories', []),
        ],

        link_with: [
            common_lib,
        ],

        c_args: [
            test_c_args,
            pie ? '-fPIE' : '-fno-PIE',

            params.get('c_args', []),
        ],

        link_args: [
            pie ? '-pie' : '-no-pie',
            '-nostdlib',

            # PAL expects that PAL regression tests have old-style hash (DT_HASH) table instead of
            # the default GNU hash (DT_GNU_HASH) table; PAL symbol resolution uses DT_HASH to figure
            # out the number of dynamic symbols in the SYMTAB symbol table
            '-Wl,--hash-style=sysv',

            params.get('link_args', []),
        ],

        dependencies: libpal,

        install: true,
        install_dir: install_dir,
    )
endforeach
