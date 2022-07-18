includes_pal_common = [
    include_directories(
        '.',
        'pal',
        join_paths('arch', host_machine.cpu_family()),
    ),
    common_inc,
]

includes_pal_for_libos = [
    include_directories(
        '.',
        'pal',
        'elf',
        join_paths('arch', host_machine.cpu_family()),
        join_paths('arch', host_machine.cpu_family(), 'Linux'),
    ),
    # PAL headers use `api.h`, `cpu.h` etc.
    common_inc,
]
