cflags_libos = [
    # TODO: check whether all standard functions implemented in Gramine match their APIs and remove
    # this
    '-fno-builtin',

    '-DIN_SHIM',
]

cflags_libos += cc.get_supported_arguments(
    # TODO: -Wnull-dereference option detection was broken in Makefile, and some problems slipped
    # through. In meson we enabled this option globally, so we need to debug before reenabling.
    '-Wno-null-dereference',

    # Some of the code uses alignof on expressions, which is a GNU extension.
    # Silence Clang - it complains but does support it.
    '-Wno-gnu-alignof-expression',

    # TODO: This is due to use of packed structs in IPC code, which triggers "taking address of
    # a packed member" warning in Clang and newer GCC. That code needs to be rewritten.
    '-Wno-address-of-packed-member',
)

if host_machine.cpu_family() == 'x86_64'
    cflags_libos += ['-mfxsr', '-mxsave']
endif

subdir('include')
subdir('src')

if enable_tests
    subdir('test')
endif
