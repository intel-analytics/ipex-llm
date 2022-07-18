pal_linux_common_sources_enclave = files(
    'bogomips.c',
    'gramine_unix_socket_addr.c',
)
pal_linux_common_sources_urts = files(
    'debug_map.c',
    'file_utils.c',
    'main_exec_path.c',
    'proc_maps.c',
    'timespec_utils.c',
    'topo_info.c',
)

# Arch-specific meson.build must define the following Meson variables:
#   - `pal_linux_common_sources_urts_arch` - a list of arch-specific sources.
subdir('arch')
pal_linux_common_sources_urts += pal_linux_common_sources_urts_arch

pal_linux_common_sources = [
    pal_linux_common_sources_enclave,
    pal_linux_common_sources_urts,
]
