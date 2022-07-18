conf = configuration_data()
conf.set_quoted('IN_GIT', '')

if direct
    hostpalpath_direct = join_paths(prefix, pkglibdir, 'direct')
    conf_gramine_direct = configuration_data()
    conf_gramine_direct.merge_from(conf)
    conf_gramine_direct.set('SGX', 0)
    conf_gramine_direct.set_quoted('HOST_PAL_PATH', hostpalpath_direct)
    conf_gramine_direct.set_quoted('LIBPAL_PATH', join_paths(hostpalpath_direct, 'libpal.so'))
    conf_gramine_direct.set_quoted('PAL_CMD', join_paths(hostpalpath_direct, 'loader'))

    configure_file(
        input: 'gramine.in',
        output: 'gramine-direct',
        configuration: conf_gramine_direct,
        install_dir: get_option('bindir'),
    )

    meson.add_install_script('/bin/sh', '-c',
        'ln -sf legacy-graphene-to-gramine "$MESON_INSTALL_DESTDIR_PREFIX"/@0@'.format(
            join_paths(get_option('bindir'), 'graphene-direct')))
endif

if sgx
    hostpalpath_linux_sgx = join_paths(prefix, pkglibdir, 'sgx')
    conf_gramine_sgx = configuration_data()
    conf_gramine_sgx.merge_from(conf)
    conf_gramine_sgx.set('SGX', 1)
    conf_gramine_sgx.set_quoted('HOST_PAL_PATH', hostpalpath_linux_sgx)
    conf_gramine_sgx.set_quoted('LIBPAL_PATH', join_paths(hostpalpath_linux_sgx, 'libpal.so'))
    conf_gramine_sgx.set_quoted('PAL_CMD', join_paths(prefix, pkglibdir, 'sgx', 'loader'))

    configure_file(
        input: 'gramine.in',
        output: 'gramine-sgx',
        configuration: conf_gramine_sgx,
        install_dir: get_option('bindir'),
    )

    meson.add_install_script('/bin/sh', '-c',
        'ln -sf legacy-graphene-to-gramine "$MESON_INSTALL_DESTDIR_PREFIX"/@0@'.format(
            join_paths(get_option('bindir'), 'graphene-sgx')))
endif

executable('gramine-argv-serializer',
    'argv_serializer.c',
    install: true)

meson.add_install_script('/bin/sh', '-c',
    'ln -sf legacy-graphene-to-gramine "$MESON_INSTALL_DESTDIR_PREFIX"/@0@'.format(
        join_paths(get_option('bindir'), 'graphene-argv-serializer')))

install_data(
    'legacy-graphene-to-gramine',
    install_dir: get_option('bindir'))
