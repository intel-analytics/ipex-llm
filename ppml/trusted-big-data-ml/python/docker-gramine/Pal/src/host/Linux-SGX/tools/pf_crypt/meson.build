executable('gramine-sgx-pf-crypt',
    'pf_crypt.c',

    dependencies: [
        sgx_util_dep,
    ],

    install: true,
    install_rpath: join_paths(get_option('prefix'), get_option('libdir')),
)

meson.add_install_script('/bin/sh', '-c',
    'ln -sf legacy-graphene-to-gramine "$MESON_INSTALL_DESTDIR_PREFIX"/@0@'.format(
        join_paths(get_option('bindir'), 'graphene-sgx-pf-crypt')))
