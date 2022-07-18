# Link mbedtls statically to Gramine-provided libraries, e.g., RA-TLS and Secret Provisioning
# libraries. This is because they are loaded dynamically to users' software and we don't want our
# patched mbedtls to collide with libraries the program already uses.

libra_tls_inc = include_directories('.')

ra_tls_args = [
    '-fvisibility=hidden',
]

libra_tls_attest = shared_library('ra_tls_attest',
    'ra_tls_attest.c',
    c_args: ra_tls_args,
    include_directories: sgx_inc,
    dependencies: [
        mbedtls_static_dep,
    ],
    install: true,
    install_rpath: join_paths(get_option('prefix'), get_option('libdir')),
)
meson.add_install_script('/bin/sh', '-c',
    'ln -sf ../../../libra_tls_attest.so ' +
    '"$MESON_INSTALL_DESTDIR_PREFIX"/@0@/gramine/runtime/glibc/'.format(
        get_option('libdir')))

libra_tls_verify_epid = shared_library('ra_tls_verify_epid',
    'ra_tls_verify_epid.c',
    'ra_tls_verify_common.c',
    'ra_tls.h',

    c_args: ra_tls_args,
    include_directories: sgx_inc,
    dependencies: [
        sgx_util_dep,
        mbedtls_static_dep,
    ],
    install: true,
    install_rpath: join_paths(get_option('prefix'), get_option('libdir')),
)
meson.add_install_script('/bin/sh', '-c',
    'ln -sf ../../../libra_tls_verify_epid.so ' +
    '"$MESON_INSTALL_DESTDIR_PREFIX"/@0@/gramine/runtime/glibc/'.format(
        get_option('libdir')))

libsecret_prov_attest = shared_library('secret_prov_attest',
    'secret_prov_attest.c',
    'secret_prov_common.c',
    'ra_tls_attest.c',
    'ra_tls.h',
    'secret_prov.h',

    c_args: ra_tls_args,
    include_directories: sgx_inc,
    dependencies: [
        mbedtls_static_dep,
        sgx_util_dep,
    ],
    install: true,
    install_rpath: join_paths(get_option('prefix'), get_option('libdir')),
)
meson.add_install_script('/bin/sh', '-c',
    'ln -sf ../../../libsecret_prov_attest.so ' +
    '"$MESON_INSTALL_DESTDIR_PREFIX"/@0@/gramine/runtime/glibc/'.format(
        get_option('libdir')))

libsecret_prov_verify_epid = shared_library('secret_prov_verify_epid',
    'ra_tls_verify_epid.c',
    'ra_tls_verify_common.c',
    'secret_prov_verify.c',
    'secret_prov_common.c',
    'ra_tls.h',
    'secret_prov.h',

    c_args: ra_tls_args,
    include_directories: sgx_inc,
    dependencies: [
        threads_dep,
        sgx_util_dep,
        mbedtls_static_dep,
    ],
    install: true,
    install_rpath: join_paths(get_option('prefix'), get_option('libdir')),
)
meson.add_install_script('/bin/sh', '-c',
    'ln -sf ../../../libsecret_prov_verify_epid.so ' +
    '"$MESON_INSTALL_DESTDIR_PREFIX"/@0@/gramine/runtime/glibc/'.format(
        get_option('libdir')))

if dcap
    libra_tls_verify_dcap = shared_library('ra_tls_verify_dcap',
        'ra_tls_verify_dcap.c',
        'ra_tls_verify_common.c',
        'ra_tls.h',

        c_args: ra_tls_args,
        include_directories: sgx_inc,
        dependencies: [
            sgx_dcap_quoteverify_dep,
            sgx_util_dep,
            mbedtls_static_dep,
        ],
        install: true,
        install_rpath: join_paths(get_option('prefix'), get_option('libdir')),
    )
    meson.add_install_script('/bin/sh', '-c',
        'ln -sf ../../../libra_tls_verify_dcap.so ' +
        '"$MESON_INSTALL_DESTDIR_PREFIX"/@0@/gramine/runtime/glibc/'.format(
            get_option('libdir')))

    libra_tls_verify_dcap_gramine = shared_library('ra_tls_verify_dcap_gramine',
        'ra_tls_verify_dcap.c',
        'ra_tls_verify_dcap_urts.c',
        'ra_tls_verify_common.c',
        'ra_tls.h',

        c_args: ra_tls_args,
        include_directories: sgx_inc,
        dependencies: [
            sgx_dcap_quoteverify_dep,
            sgx_util_dep,
            mbedtls_static_dep,
        ],
        install: true,
        install_rpath: join_paths(get_option('prefix'), get_option('libdir')),
    )
    meson.add_install_script('/bin/sh', '-c',
        'ln -sf ../../../libra_tls_verify_dcap_gramine.so ' +
        '"$MESON_INSTALL_DESTDIR_PREFIX"/@0@/gramine/runtime/glibc/'.format(
            get_option('libdir')))

    libsecret_prov_verify_dcap = shared_library('secret_prov_verify_dcap',
        'ra_tls_verify_dcap.c',
        'ra_tls_verify_common.c',
        'secret_prov_verify.c',
        'secret_prov_common.c',
        'ra_tls.h',
        'secret_prov.h',

        c_args: ra_tls_args,
        include_directories: sgx_inc,
        dependencies: [
            threads_dep,
            sgx_dcap_quoteverify_dep,
            sgx_util_dep,
            mbedtls_static_dep,
        ],
        install: true,
        install_rpath: join_paths(get_option('prefix'), get_option('libdir')),
    )
    meson.add_install_script('/bin/sh', '-c',
        'ln -sf ../../../libsecret_prov_verify_dcap.so ' +
        '"$MESON_INSTALL_DESTDIR_PREFIX"/@0@/gramine/runtime/glibc/'.format(
            get_option('libdir')))
endif
