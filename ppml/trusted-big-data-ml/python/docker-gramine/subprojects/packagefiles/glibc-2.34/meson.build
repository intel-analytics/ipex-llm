project('glibc', 'c',
    version: '2.34')

fs = import('fs')

glibc_output_paths = [
    'csu/crt1.o',
    'csu/crti.o',
    'csu/crtn.o',
    'dlfcn/libdl.so',
    'dlfcn/libdl.so.2',
    'libc.so',
    'libc.so.6',
    'login/libutil.so',
    'login/libutil.so.1',
    'math/libm.so',
    'math/libm.so.6',
    'nis/libnsl.so',
    'nis/libnsl.so.1',
    'nptl/libpthread.so',
    'nptl/libpthread.so.0',
    'nptl_db/libthread_db.so',
    'nptl_db/libthread_db.so.1',
    'nss/libnss_compat.so',
    'nss/libnss_compat.so.2',
    'nss/libnss_db.so',
    'nss/libnss_db.so.2',
    'nss/libnss_files.so',
    'nss/libnss_files.so.2',
    'resolv/libanl.so',
    'resolv/libanl.so.1',
    'resolv/libnss_dns.so',
    'resolv/libnss_dns.so.2',
    'resolv/libresolv.so',
    'resolv/libresolv.so.2',
    'rt/librt.so',
    'rt/librt.so.1',
]

if host_machine.cpu_family() == 'x86_64'
    glibc_output_paths += [
        'elf/ld-linux-x86-64.so.2',
        'elf/ld.so',
        'mathvec/libmvec.so',
        'mathvec/libmvec.so.1',
    ]
endif

glibc_output = []
foreach path : glibc_output_paths
    glibc_output += fs.name(path)
endforeach

extra_cflags = ''
if get_option('buildtype') == 'debug' or get_option('buildtype') == 'debugoptimized'
    extra_cflags += '-g'
endif

glibc = custom_target('glibc',
    command: [
        find_program('compile.sh'),
        host_machine.cpu_family(),
        '@CURRENT_SOURCE_DIR@',
        '@CURRENT_SOURCE_DIR@/glibc-2.34',
        meson.current_build_dir(),
        '@PRIVATE_DIR@',
        get_option('prefix'),
        get_option('libdir'),
        extra_cflags,
        glibc_output_paths,
    ],

    input: [
        'glibc-2.34/configure',
        'glibc-2.34.patch',
        'hp-timing-2.34.patch',
    ],

    output: glibc_output,

    console: true,

    install: true,
    install_dir: join_paths(get_option('libdir'), 'gramine/runtime/glibc'),
)
