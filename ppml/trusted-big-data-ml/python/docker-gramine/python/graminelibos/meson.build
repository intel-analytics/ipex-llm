conf = configuration_data()
conf.set('VERSION', meson.project_version())
conf.set('PKGLIBDIR', join_paths(get_option('prefix'), pkglibdir))
conf.set('LIBDIR', join_paths(get_option('prefix'), get_option('libdir')))
conf.set('SYSLIBDIR', syslibdir)
conf.set('SGX_ENABLED', sgx ? '1' : '0')

init_py = configure_file(
    input: '__init__.py',
    output: '__init__.py',
    configuration: conf,
)

python_src = [
    init_py,
    'gen_jinja_env.py',
    'manifest.py',
]

if enable_tests
    python_src += [
        'ninja_syntax.py',
        'regression.py',
        'util_tests.py',
    ]
endif

if sgx
    python_src += [
        '_aesm_pb2.py',
        'sgx_get_token.py',
        'sgx_sign.py',
        'sigstruct.py',
    ]
endif

install_data(python_src, install_dir: python3_pkgdir)
