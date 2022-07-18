sgx_util = shared_library('sgx_util',
    'attestation.c',
    'attestation.h',
    'ias.c',
    'ias.h',
    'pf_util.c',
    'pf_util.h',
    'util.c',
    'util.h',

    c_args: [
        '-DIN_TOOLS',
    ],

    include_directories: [
        sgx_inc,
        common_inc,
    ],
    dependencies: [
        cjson_dep,
        libcurl_dep,
        protected_files_dep,
        mbedtls_static_dep,
    ],

    install: true,
    install_rpath: join_paths(get_option('prefix'), get_option('libdir')),
)

sgx_util_dep = declare_dependency(
    link_with: sgx_util,
    include_directories: [
        include_directories('.'),
        sgx_inc, # this is mostly for sgx_arch.h
        protected_files_inc,
    ],
)

pkgconfig.generate(
    sgx_util,
    libraries: [
        '-Wl,-rpath,${libdir}',
        sgx_util,
    ],
)

meson.add_install_script('/bin/sh', '-c',
    'ln -sf ../../../libsgx_util.so ' +
    '"$MESON_INSTALL_DESTDIR_PREFIX"/@0@/gramine/runtime/glibc/'.format(
        get_option('libdir')))
