subdir('graminelibos')

install_data([
    'gramine-gen-depend',
    'gramine-manifest',
], install_dir: get_option('bindir'))

meson.add_install_script('/bin/sh', '-c',
    'ln -sf legacy-graphene-to-gramine "$MESON_INSTALL_DESTDIR_PREFIX"/@0@'.format(
        join_paths(get_option('bindir'), 'graphene-manifest')))

if enable_tests
    install_data([
        'gramine-test',
    ], install_dir: get_option('bindir'))
endif

if sgx
    install_data([
        'gramine-sgx-gen-private-key',
        'gramine-sgx-get-token',
        'gramine-sgx-sign',
    ], install_dir: get_option('bindir'))

    meson.add_install_script('/bin/sh', '-c',
        'ln -sf legacy-graphene-to-gramine "$MESON_INSTALL_DESTDIR_PREFIX"/@0@'.format(
            join_paths(get_option('bindir'), 'graphene-sgx-get-token')))
    meson.add_install_script('/bin/sh', '-c',
        'ln -sf legacy-graphene-to-gramine "$MESON_INSTALL_DESTDIR_PREFIX"/@0@'.format(
            join_paths(get_option('bindir'), 'graphene-sgx-sign')))
endif
