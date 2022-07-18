project('glibc', 'c',
    version: '2.34')

fs = import('fs')

gcc_output_paths = [
    'x86_64-pc-linux-gnu/libgomp/.libs/libgomp.so.1',
    'x86_64-pc-linux-gnu/libgomp/.libs/libgomp.so.1.0.0',
]

if host_machine.cpu_family() != 'x86_64'
    error('The patched GCC (libgomp) build is available only x86_64')
endif

gcc_output = []
foreach path : gcc_output_paths
    gcc_output += fs.name(path)
endforeach

gcc = custom_target('gcc',
    command: [
        find_program('compile.sh'),
        host_machine.cpu_family(),
        '@CURRENT_SOURCE_DIR@',
        meson.current_build_dir(),
        '@PRIVATE_DIR@',
        get_option('prefix'),
        get_option('libdir'),
        gcc_output_paths,
    ],

    input: [
        'configure',
        'libgomp-replace-futex-instruction.patch',
    ],

    output: gcc_output,

    console: true,

    install: true,
    install_dir: join_paths(get_option('libdir'), 'gramine/runtime/glibc'),
)
