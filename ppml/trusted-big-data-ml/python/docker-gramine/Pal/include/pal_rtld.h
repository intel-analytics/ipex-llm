/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University
 * Copyright (C) 2021 Intel Labs
 */

#ifndef PAL_RTLD_H
#define PAL_RTLD_H

#include <endian.h>

#include "api.h"
#include "elf/elf.h"

/* Loaded shared object. */
struct link_map {
    /*
     * Difference between virtual addresses (p_vaddr) in ELF file and actual virtual addresses in
     * memory. Equal to 0x0 for ET_EXECs.
     *
     * Note that `l_base_diff` name is a departure from the ELF standard and Glibc code: the ELF
     * standard uses the term "Base Address" and Glibc uses `l_addr`. We find ELF/Glibc terms
     * misleading and therefore replace them with `l_base_diff`. For the context, below is the
     * exerpt from the ELF spec, Section "Program Header", Sub-section "Base Address":
     *
     *   The virtual addresses in the program headers might not represent the actual virtual
     *   addresses of the program's memory image. [...] The difference between the virtual address
     *   of any segment in memory and the corresponding virtual address in the file is thus a single
     *   constant value for any one executable or shared object in a given process. This difference
     *   is the *base address*. One use of the base address is to relocate the memory image of the
     *   program during dynamic linking.
     */
    elf_addr_t  l_base_diff;

    elf_addr_t  l_map_start;  /* Address shared object (its first LOAD segment) is loaded at. */
    const char* l_name;       /* Absolute file name object was found in. */
    elf_dyn_t*  l_ld;         /* Dynamic section of the shared object. */
    elf_addr_t  l_entry;      /* Entry point location (may be empty, e.g., for libs). */

    /* Chain of all shared objects loaded at startup. */
    struct link_map* l_next;
    struct link_map* l_prev;

    /* Relocation information, taken from DT_STRTAB, DT_SYMTAB and DT_HASH. */
    const char* string_table;
    elf_sym_t* symbol_table;
    uint32_t symbol_table_cnt;
};

/* for GDB debugging */
void _DkDebugMapAdd(const char* name, void* addr);
void _DkDebugMapRemove(void* addr);
int _DkDebugDescribeLocation(uintptr_t addr, char* buf, size_t buf_size);

/* loading ELF binaries */
int setup_pal_binary(void);
void set_pal_binary_name(const char* name);
int load_entrypoint(const char* uri);
int find_string_and_symbol_tables(elf_addr_t ehdr_addr, elf_addr_t base_addr,
                                  const char** out_string_table, elf_sym_t** out_symbol_table,
                                  uint32_t* out_symbol_table_cnt);

noreturn void start_execution(const char** arguments, const char** environs);

#endif /* PAL_RTLD_H */
