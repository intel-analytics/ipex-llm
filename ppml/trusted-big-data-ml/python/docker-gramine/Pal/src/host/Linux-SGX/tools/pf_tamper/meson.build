executable('gramine-sgx-pf-tamper',
    'pf_tamper.c',

    dependencies: [
        sgx_util_dep,
    ],

    include_directories: common_inc, # for api.h

    install: true,
    install_rpath: join_paths(get_option('prefix'), get_option('libdir')),
)

meson.add_install_script('/bin/sh', '-c',
    'ln -sf legacy-graphene-to-gramine "$MESON_INSTALL_DESTDIR_PREFIX"/@0@'.format(
        join_paths(get_option('bindir'), 'graphene-sgx-pf-tamper')))
