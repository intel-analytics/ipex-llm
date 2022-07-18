project('cJSON', 'c')

cjson_dep = declare_dependency(
    sources: files('cJSON.c', 'cJSON.h'),
    include_directories: include_directories('.'),
    compile_args: ['-DHAVE_INTERNAL_CJSON'],
)
