/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2022 Intel Corporation
 *                    Mariusz Zaborski <oshogbo@invisiblethingslab.com>
 */

#include <elf.h>
#include <stdbool.h>
#include <stddef.h>

/*
 * We have to add a function declaration to avoid warnings.
 * This function is used with NASM, so creating a header is pointless.
 */
int verify_auxiliary(Elf64_auxv_t* auxv);

/*
 * Set up in: LibOS/shim/src/shim_rtld.c: execute_elf_object()
 */
static struct {
    uint64_t type;
    bool exists;
} auxv_gramine_defaults[] = {
    { AT_PHDR, false },
    { AT_PHNUM, false },
    { AT_PAGESZ, false },
    { AT_ENTRY, false },
    { AT_BASE, false },
    { AT_RANDOM, false },
    { AT_PHENT, false },
    { AT_SYSINFO_EHDR, false },
};

int verify_auxiliary(Elf64_auxv_t* auxv) {
    size_t count = sizeof(auxv_gramine_defaults) / sizeof(auxv_gramine_defaults[0]);

    for (; auxv->a_type != AT_NULL; auxv++) {
        for (size_t i = 0; i < count; i++) {
            if (auxv_gramine_defaults[i].type == auxv->a_type) {
                /* Check for duplicates */
                if (auxv_gramine_defaults[i].exists) {
                    return 1;
                }
                auxv_gramine_defaults[i].exists = true;
            }
        }
    }

    for (size_t i = 0; i < count; i++) {
        if (!auxv_gramine_defaults[i].exists) {
            return 1;
        }
    }

    return 0;
}
