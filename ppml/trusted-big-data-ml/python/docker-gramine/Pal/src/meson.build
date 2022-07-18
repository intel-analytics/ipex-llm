cflags_pal_common = [
    '-DIN_PAL',
    '-fno-builtin',
]

pal_common_sources = files(
    'db_events.c',
    'db_exception.c',
    'db_main.c',
    'db_memory.c',
    'db_misc.c',
    'db_object.c',
    'db_process.c',
    'db_rtld.c',
    'db_streams.c',
    'db_threading.c',
    'pal_error.c',
    'printf.c',
    'slab.c',
)

# Arch-specific meson.build must define the following Meson variables:
#   - `pal_common_sources_arch` - a list of arch-specific sources.
subdir('arch')
pal_common_sources += pal_common_sources_arch

subdir('host')
