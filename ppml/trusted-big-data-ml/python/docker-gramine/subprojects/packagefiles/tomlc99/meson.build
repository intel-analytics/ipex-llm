project('tomlc99', 'c')

patch_prog = find_program('patch')

tomlc99_src = []
foreach src : ['toml.c', 'toml.h']
    src_patch = '@0@.patch'.format(src)
    tomlc99_src += custom_target(src,
        command: [
            patch_prog,
            '-p1',
            '--quiet',
            '--directory=@CURRENT_SOURCE_DIR@',
            '--input=@0@'.format(src_patch),
            '--output=-',
        ],
        input: [src_patch, src],
        output: src,
        capture: true,
    )
endforeach

tomlc99_dep = declare_dependency(
    sources: tomlc99_src,
    include_directories: include_directories('.'),
)

