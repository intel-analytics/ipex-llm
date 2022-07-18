project('uthash', 'c')

patch_prog = find_program('patch')

subdir('src')

uthash_dep = declare_dependency(
    sources: uthash_h,
    include_directories: include_directories('src/'),
)
