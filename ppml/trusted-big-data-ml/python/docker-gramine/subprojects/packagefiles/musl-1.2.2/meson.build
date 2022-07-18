project('musl', 'c', version: '1.2.2')

fs = import('fs')

musl_output_paths = [
    'lib/libc.so',
]

musl_output = []
foreach path : musl_output_paths
    musl_output += fs.name(path)
endforeach

is_debug = '0'
if get_option('buildtype') == 'debug' or get_option('buildtype') == 'debugoptimized'
    is_debug = '1'
endif

musl = custom_target('musl',
    command: [
        find_program('compile.sh'),
        host_machine.cpu_family(),
        '@CURRENT_SOURCE_DIR@',
        meson.current_build_dir(),
        '@PRIVATE_DIR@',
        get_option('prefix'),
        get_option('libdir'),
        is_debug,
        musl_output_paths,
    ],

    input: files([
        'configure',
        'musl-1.2.2.patch',
    ]),

    output: musl_output,

    console: true,

    install: true,
    install_dir: join_paths(get_option('libdir'), 'gramine/runtime/musl'),
)

meson.add_install_script(
    'sh', '-c',
    'ln -sf libc.so "${MESON_INSTALL_DESTDIR_PREFIX}/@0@"'.format(
        join_paths(get_option('libdir'), 'gramine/runtime/musl', 'ld-musl-x86_64.so.1')),
)
