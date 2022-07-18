# Arch-specific meson.build must define the following Meson variables:
#   - `pal_skeleton_lds` - a full path to PAL linker script.
subdir('arch')

pal_skeleton_map = custom_target('pal.map[skeleton]',
    command: [gen_pal_map_prog, '@INPUT@', '@OUTPUT@'],
    input: ['../../pal-symbols', 'pal.map.template'],
    output: 'pal.map',
)

libpal_skeleton = shared_library('pal',
    'db_devices.c',
    'db_eventfd.c',
    'db_events.c',
    'db_exception.c',
    'db_files.c',
    'db_main.c',
    'db_memory.c',
    'db_misc.c',
    'db_object.c',
    'db_pipes.c',
    'db_process.c',
    'db_rtld.c',
    'db_sockets.c',
    'db_streams.c',
    'db_threading.c',
    pal_common_sources,

    include_directories: [
        includes_pal_common,
        include_directories(
            '../../../include/arch/@0@/Skeleton'.format(host_machine.cpu_family()),
        ),
    ],

    c_args: [
        cflags_pal_common,
        cflags_custom_stack_protector,
        cflags_sanitizers,
        '-DHOST_TYPE=Skeleton',

        # in Skeleton, every single parameter is unused
        '-Wno-unused-parameter',
    ],

    link_args: [
        '-nostdlib',

        # For PAL symbol resolution, we use old-style hash (DT_HASH) table instead of the
        # default GNU hash (DT_GNU_HASH) table -- to figure out the number of dynamic symbols in
        # the SYMTAB symbol table
        '-Wl,--hash-style=sysv',

        # PAL is expected to be linked with RELRO (Relocation Read-Only)
        '-Wl,-zrelro',
        '-Wl,-znow',

        '-Wl,-zcombreloc',
        '-Wl,-zdefs',
        '-Wl,--version-script=@0@'.format(pal_skeleton_map.full_path()),
        '-Wl,-T@0@'.format(pal_skeleton_lds),
        '-Wl,--eh-frame-hdr',
    ],
    link_depends: [
        pal_skeleton_map,
        pal_skeleton_lds,
    ],

    dependencies: [
        common_dep,
    ],

    install: true,
    install_dir: join_paths(pkglibdir, 'skeleton'),
)

libpal_skeleton_dep = declare_dependency(
    link_with: libpal_skeleton,
)

if debug
    custom_target('libpal.so.map',
        input: libpal_skeleton,
        command: gen_symbol_map_cmd,
        capture: true,
        output: 'libpal.so.map',

        install: true,
        install_dir: join_paths(pkglibdir, 'skeleton'),
    )
endif
