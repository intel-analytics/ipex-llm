libos_sources_arch_list = {
    'shim_arch_prctl.c': {},
    'shim_elf_entry.nasm': { 'type': 'nasm' },
    'shim_context.c': {},
    'shim_table.c': {},
    'start.S': {},
    'syscallas.S': {},
}

libos_sources_arch = files()

foreach src, params : libos_sources_arch_list
    if params.get('type', '') == 'nasm'
        libos_sources_arch += nasm_gen.process(src)
    else
        libos_sources_arch += files(src)
    endif
endforeach

shim_lds = join_paths(meson.current_source_dir(), 'shim.lds')
