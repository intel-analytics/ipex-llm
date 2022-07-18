subdir('include')
subdir('src')

if enable_tests
    subdir('regression')
endif

if debug
    install_data(
        'gdb_integration/gramine.gdb',
        'gdb_integration/debug_map_gdb.py',
        'gdb_integration/language_gdb.py',
        'gdb_integration/pagination_gdb.py',
        install_dir: join_paths(pkglibdir, 'gdb_integration'),
    )
endif
